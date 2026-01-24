"""
Market Data Pipeline Core Foundation for Bearish Alpha Bot.

Provides multi-exchange data collection, storage, and health monitoring
for Phase 2.2 WebSocket integration foundation.
"""

import asyncio
import logging
import time
import pandas as pd
from collections import defaultdict
from collections import Counter
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from .ccxt_client import CcxtClient
from .indicators import add_indicators, rsi
from .data_validator import TIMEFRAME_SECONDS

logger = logging.getLogger(__name__)


class MarketDataPipeline:
    """
    Core market data pipeline for multi-exchange data collection and management.
    
    Features:
    - Multi-exchange data collection with fallback
    - Circular buffer memory management
    - Health monitoring and status tracking
    - Async-compatible design (sync methods for now)
    """
    
    # Buffer limits per timeframe to manage memory
    BUFFER_LIMITS = {
        '30m': 1000,
        '1h': 500,
        '4h': 200,
        '1d': 100
    }
    
    # Default exchange for WebSocket collector when not specified
    DEFAULT_EXCHANGE = 'bingx'
    
    # Extra candles buffer for indicator warmup to ensure sufficient historical data
    INDICATOR_WARMUP_BUFFER = 50

    # Safety margin (ms) used when determining if the last candle is closed
    SAFETY_MARGIN_MS = 2000

    # Extra fetch buffer to compensate for dropping the trailing forming candle
    FETCH_SAFETY_BUFFER = 5
    
    def __init__(self, exchanges: Dict[str, CcxtClient], config: Dict[str, Any] = None, websocket_manager: Optional[Any] = None):
        """
        Initialize MarketDataPipeline.
        
        Args:
            exchanges: Dictionary mapping exchange names to CcxtClient instances
            config: Optional configuration dict for pipeline settings
            websocket_manager: Optional WebSocketManager instance for data injection.
        """
        self.exchanges = exchanges
        self.config = config or {}
        self.websocket_manager = websocket_manager
        # Provide backfill handler to collector if available
        if self.websocket_manager and getattr(self.websocket_manager, "collector", None) and hasattr(self.websocket_manager.collector, "set_backfill_handler"):
            self.websocket_manager.collector.set_backfill_handler(self._backfill_handler)
        
        # Data storage: {exchange: {symbol: {timeframe: DataFrame}}}
        self.data_streams = defaultdict(lambda: defaultdict(dict))
        
        # Market metadata cache: {exchange: {symbol: market_metadata}}
        self._market_metadata_cache = {}
        
        # Dedicated thread pool for synchronous CCXT calls to avoid overhead
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix='ccxt_executor'
        )
        
        # Health monitoring
        self.start_time = datetime.now(timezone.utc)
        self.total_requests = 0
        self.failed_requests = 0
        self.last_update_time = {}
        
        # Pipeline state
        self.is_running = False
        self._trigger_diag_last_log: Dict[str, float] = {}
        # Optional hybrid mode state machine (disabled by default)
        self._hybrid_state: Dict[str, Dict[str, Any]] = {}

        # One-time hybrid configuration observability (log once)
        self._hybrid_cfg_logged: bool = False

        # Hybrid fallback metrics (counts by reason, per state_key)
        self._hybrid_fallback_counts: Dict[str, Counter] = {}
        self._hybrid_total_calls: Dict[str, int] = {}
        self._hybrid_metrics_last_log_ts: Dict[str, float] = {}
        self._hybrid_pivot_grace_last_log_ts: Dict[str, float] = {}
        self._hybrid_last_inject_ts_ms: Dict[str, int] = {}

        # Optional cleanup guardrail for long-running processes with dynamic universes.
        self._hybrid_last_seen_ts: Dict[str, float] = {}
        self._hybrid_cleanup_last_run_ts: float = 0.0

        # One-time (per source) state_key sample log for runtime verification.
        self._hybrid_sample_key_logged_sources: set[str] = set()
        
        logger.info(f"🔄 MarketDataPipeline initialized with {len(exchanges)} exchanges: {list(exchanges.keys())}")

        # Log effective hybrid settings once at startup for easy correlation with runtime behavior.
        try:
            ws_cfg = self.config.get('websocket', {}) if isinstance(self.config, dict) else {}
            forming_update_stale_ms = int(ws_cfg.get('forming_update_stale_ms', 15000))
            pivot_enabled = bool(ws_cfg.get('hybrid_pivot_grace_enabled', True))
            pivot_grace_ms_default = int(ws_cfg.get('hybrid_pivot_grace_ms', 90000))
            pivot_by_tf = ws_cfg.get('hybrid_pivot_grace_ms_by_tf')
            pivot_accept_prev_bucket = bool(ws_cfg.get('pivot_grace_accept_prev_bucket', False))
            metrics_interval_sec = float(ws_cfg.get('hybrid_fallback_metrics_interval_sec', 300))
            sm_enabled = bool(ws_cfg.get('hybrid_state_machine_enabled', False))
            sm_failures = int(ws_cfg.get('hybrid_failures_before_cooldown', 3))
            sm_cooldown_ms = int(ws_cfg.get('hybrid_cooldown_ms', 60000))
            logger.info(
                "[HYBRID-STARTUP] "
                f"forming_update_stale_ms={forming_update_stale_ms} "
                f"pivot_grace_enabled={pivot_enabled} pivot_grace_ms_default={pivot_grace_ms_default} "
                f"pivot_grace_ms_by_tf={'set' if isinstance(pivot_by_tf, dict) else 'none'} "
                f"pivot_grace_accept_prev_bucket={pivot_accept_prev_bucket} "
                f"hybrid_fallback_metrics_interval_sec={metrics_interval_sec} "
                f"hybrid_metrics_interval_s={metrics_interval_sec} "
                f"hybrid_state_machine_enabled={sm_enabled} "
                f"hybrid_failures_before_cooldown={sm_failures} hybrid_cooldown_ms={sm_cooldown_ms}"
            )
            self._hybrid_cfg_logged = True
        except Exception:
            # Never block startup on observability
            pass

    def _canonical_exchange_id(self, exchange: Optional[str]) -> str:
        """Return a stable canonical exchange id for state keys.

        Goal: avoid metrics/state splitting due to alias/case differences.
        Preference: underlying ccxt exchange client's .id (lowercased).
        """
        if not exchange:
            exchange = self.DEFAULT_EXCHANGE

        # Try direct match by config key.
        client = None
        if isinstance(self.exchanges, dict):
            client = self.exchanges.get(exchange)

        # Try case-insensitive match by key.
        if client is None and isinstance(self.exchanges, dict):
            ex_lower = str(exchange).lower()
            for k, v in self.exchanges.items():
                if str(k).lower() == ex_lower:
                    client = v
                    break

        candidate = None
        if client is not None:
            candidate = getattr(client, 'id', None) or getattr(client, 'name', None)
        if not candidate:
            candidate = exchange
        return str(candidate).lower()

    def _hybrid_state_key(self, exchange: str, symbol: str, timeframe: str) -> str:
        # Keep this stable and explicit to avoid confusion in logs/metrics.
        return f"{exchange}:{symbol}:{timeframe}"

    def _maybe_cleanup_hybrid_state(self, now_ts: float) -> None:
        """Evict per-state-key hybrid metrics state that hasn't been seen recently."""
        try:
            ws_cfg = self.config.get('websocket', {}) if isinstance(self.config, dict) else {}
            enabled = bool(ws_cfg.get('hybrid_state_cleanup_enabled', False))
            if not enabled:
                return

            ttl_s = float(ws_cfg.get('hybrid_state_cleanup_ttl_s', 86400))
            interval_s = float(ws_cfg.get('hybrid_state_cleanup_interval_s', 600))
            if ttl_s <= 0:
                return
            if interval_s <= 0:
                interval_s = 600

            if (now_ts - float(self._hybrid_cleanup_last_run_ts or 0.0)) < interval_s:
                return

            self._hybrid_cleanup_last_run_ts = now_ts

            cutoff = now_ts - ttl_s
            to_evict: List[str] = []
            for k, last_seen in list(self._hybrid_last_seen_ts.items()):
                try:
                    if float(last_seen or 0.0) < cutoff:
                        to_evict.append(k)
                except Exception:
                    continue

            if not to_evict:
                return

            for k in to_evict:
                self._hybrid_last_seen_ts.pop(k, None)
                self._hybrid_fallback_counts.pop(k, None)
                self._hybrid_total_calls.pop(k, None)
                self._hybrid_metrics_last_log_ts.pop(k, None)
                self._hybrid_pivot_grace_last_log_ts.pop(k, None)
                self._hybrid_last_inject_ts_ms.pop(k, None)
                self._hybrid_state.pop(k, None)

            logger.info(
                f"[HYBRID-CLEANUP] evicted={len(to_evict)} ttl_s={ttl_s:.0f} interval_s={interval_s:.0f}"
            )
        except Exception:
            return

    def _maybe_log_state_key_sample(
        self,
        *,
        source: str,
        raw_exchange: str,
        canonical_exchange_id: str,
        symbol: str,
        timeframe: str,
        state_key: str,
    ) -> None:
        try:
            if not logger.isEnabledFor(logging.DEBUG):
                return
            if source in self._hybrid_sample_key_logged_sources:
                return
            logger.debug(
                "[HYBRID-KEY-SAMPLE] "
                f"source={source} raw_exchange={raw_exchange} canonical_exchange_id={canonical_exchange_id} "
                f"symbol={symbol} tf={timeframe} state_key={state_key}"
            )
            self._hybrid_sample_key_logged_sources.add(source)
        except Exception:
            return

    def _record_hybrid_metrics(
        self,
        *,
        state_key: str,
        fallback_reason: Optional[str],
        timeframe: str,
        symbol: str,
        inject_ts_ms: Optional[int] = None,
    ) -> None:
        """Monotonic counters + periodic log for hybrid evaluation health."""
        try:
            now_ts = time.time()

            # Track last-seen and optionally run cleanup.
            self._hybrid_last_seen_ts[state_key] = now_ts
            self._maybe_cleanup_hybrid_state(now_ts)

            reason_key = "none" if fallback_reason is None else str(fallback_reason)

            bucket = self._hybrid_fallback_counts.get(state_key)
            if bucket is None:
                bucket = Counter()
                self._hybrid_fallback_counts[state_key] = bucket
            bucket[reason_key] += 1

            self._hybrid_total_calls[state_key] = int(self._hybrid_total_calls.get(state_key, 0) or 0) + 1

            if inject_ts_ms is not None:
                self._hybrid_last_inject_ts_ms[state_key] = int(inject_ts_ms)

            # Rate-limited explanation for pivot grace (normal state)
            if reason_key == "pivot_grace_prev_bucket":
                last_ts = float(self._hybrid_pivot_grace_last_log_ts.get(state_key, 0.0) or 0.0)
                if (now_ts - last_ts) >= 300.0:
                    logger.info(
                        f"[HYBRID-PIVOT-GRACE] symbol={symbol} tf={timeframe} "
                        "previous bucket still updating within grace window; using closed-only for safety"
                    )
                    self._hybrid_pivot_grace_last_log_ts[state_key] = now_ts

            # Periodic metrics summary
            ws_cfg = self.config.get('websocket', {}) if isinstance(self.config, dict) else {}
            interval_sec = float(ws_cfg.get('hybrid_fallback_metrics_interval_sec', 300))
            if interval_sec > 0:
                last_ts = float(self._hybrid_metrics_last_log_ts.get(state_key, 0.0) or 0.0)
                if (now_ts - last_ts) >= interval_sec:
                    uptime_s = max(0.0, (datetime.now(timezone.utc) - self.start_time).total_seconds())
                    last_inject_ts_ms = int(self._hybrid_last_inject_ts_ms.get(state_key, 0) or 0)
                    last_inject_age_s: Optional[float] = None
                    if last_inject_ts_ms > 0:
                        try:
                            last_inject_age_s = max(0.0, (int(time.time() * 1000) - last_inject_ts_ms) / 1000.0)
                        except Exception:
                            last_inject_age_s = None

                    logger.info(
                        f"[HYBRID-METRICS] {state_key} "
                        f"total_calls={self._hybrid_total_calls.get(state_key, 0)} "
                        f"uptime_s={uptime_s:.0f} "
                        f"last_inject_ts_ms={last_inject_ts_ms} "
                        f"last_inject_age_s={(f'{last_inject_age_s:.1f}' if last_inject_age_s is not None else 'none')} "
                        f"counts={dict(bucket)}"
                    )
                    self._hybrid_metrics_last_log_ts[state_key] = now_ts
        except Exception:
            return
    
    def _filter_closed_dataframe(self, df: pd.DataFrame, timeframe: str, context: str = "") -> pd.DataFrame:
        """
        Drop the last candle if it is still forming based on timeframe duration and safety margin.
        """
        if df is None or df.empty:
            return df

        interval_sec = TIMEFRAME_SECONDS.get(timeframe)
        if interval_sec is None:
            return df

        interval_ms = interval_sec * 1000
        now_ms = int(time.time() * 1000)
        last_open_ms = int(df.index[-1].timestamp() * 1000)

        if last_open_ms + interval_ms > (now_ms - self.SAFETY_MARGIN_MS):
            trimmed = df.iloc[:-1]
            logger.info(
                f"[CLOSED-ONLY]{context} Dropped trailing forming candle for {timeframe} "
                f"(last_open={last_open_ms}, now_ms={now_ms}, interval_ms={interval_ms})"
            )
            return trimmed

        return df

    async def fetch_missing_candles(self, symbol: str, timeframe: str, start_ts: int, end_ts: int, exchange: str = None) -> List[List[float]]:
        """
        Fetch missing candles from REST within the specified range [start_ts, end_ts].
        Returns list of OHLCV arrays with timestamp in ms.
        """
        if start_ts > end_ts:
            return []

        exchange = exchange or (next(iter(self.exchanges.keys())) if self.exchanges else None)
        if not exchange or exchange not in self.exchanges:
            logger.warning(f"[BACKFILL] No valid exchange available for backfill {symbol}")
            return []

        client = self.exchanges[exchange]
        interval_ms = TIMEFRAME_SECONDS.get(timeframe, 0) * 1000
        if interval_ms <= 0:
            logger.warning(f"[BACKFILL] Unknown timeframe {timeframe}, skipping.")
            return []

        # Calculate minimal limit to cover range, add small buffer
        bars_needed = int((end_ts - start_ts) // interval_ms) + 2
        try:
            ohlcv_df = await client.ohlcv(symbol, timeframe, limit=bars_needed + 2, add_indicators=False)
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(f"[BACKFILL] Empty response for {symbol} {timeframe}")
                return []

            # Filter by timestamp range (ms)
            ts_ms = (ohlcv_df.index.view("int64") // 1_000_000)
            mask = (ts_ms >= start_ts) & (ts_ms <= end_ts)
            filtered = ohlcv_df.loc[mask]
            if filtered.empty:
                return []

            candles = []
            for idx, row in filtered.iterrows():
                ts = int(idx.timestamp() * 1000)
                candles.append([ts, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"])])
            return candles
        except Exception as e:
            logger.warning(f"[BACKFILL] Fetch failed for {symbol} {timeframe}: {e}")
            return []

    async def _backfill_handler(self, exchange: str, symbol: str, timeframe: str, start_ts: int, end_ts: int, gap_bars: int) -> List[List[float]]:
        """
        Adapter passed to StreamDataCollector to fetch missing candles safely.
        """
        return await self.fetch_missing_candles(symbol, timeframe, start_ts, end_ts, exchange=exchange)

    async def get_market_metadata(self, symbol: str, exchange_id: str) -> Dict[str, Any]:
        """
        Get market metadata (precision, limits, etc.) for a given symbol on an exchange.
        
        This method is the proper way to access market information in the architecture.
        It handles caching and ensures data is loaded from the appropriate exchange.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT', 'ETH/USDT')
            exchange_id: Exchange identifier (e.g., 'bingx', 'kucoinfutures')
            
        Returns:
            Market metadata dictionary with precision, limits, etc.
            
        Raises:
            ValueError: If exchange is not available or symbol is invalid
        """
        # Check if exchange exists
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange '{exchange_id}' not available in MarketDataPipeline")
        
        # Create cache key
        cache_key = f"{exchange_id}:{symbol}"
        
        # Check cache first
        if cache_key in self._market_metadata_cache:
            logger.debug(f"[MARKET-META] Cache hit for {cache_key}")
            return self._market_metadata_cache[cache_key]
        
        # Cache miss - fetch from exchange
        logger.debug(f"[MARKET-META] Cache miss for {cache_key}, fetching from exchange")
        
        try:
            client = self.exchanges[exchange_id]
            
            # Ensure markets are loaded using dedicated executor
            loop = asyncio.get_running_loop()
            markets = await loop.run_in_executor(self._executor, client.load_markets)
            
            # Get market data for the symbol
            if symbol not in markets:
                # Try to normalize symbol variants
                symbol_variants = self._normalize_symbol_variants(symbol)
                found_symbol = None
                
                for variant in symbol_variants:
                    if variant in markets:
                        found_symbol = variant
                        logger.info(f"[MARKET-META] Symbol variant match: {symbol} -> {variant}")
                        break
                
                if not found_symbol:
                    raise ValueError(
                        f"Symbol '{symbol}' not found on exchange '{exchange_id}'. "
                        f"Tried variants: {symbol_variants}"
                    )
                
                symbol = found_symbol
            
            # Get market metadata
            market_metadata = markets[symbol]
            
            # Cache the result
            self._market_metadata_cache[cache_key] = market_metadata
            logger.info(f"[MARKET-META] Cached metadata for {cache_key}")
            
            return market_metadata
            
        except Exception as e:
            error_msg = f"Failed to get market metadata for {symbol} on {exchange_id}: {e}"
            logger.error(f"[MARKET-META] {error_msg}")
            raise ValueError(error_msg) from e
    
    def _normalize_symbol_variants(self, symbol: str) -> List[str]:
        """
        Generate potential symbol format variants.
        
        Examples:
        - 'BTC/USDT' -> ['BTC/USDT', 'BTC/USDT:USDT', 'BTC-USDT', 'BTCUSDT']
        - 'ETH/USDT:USDT' -> ['ETH/USDT:USDT', 'ETH/USDT', 'ETH-USDT', 'ETHUSDT']
        
        Args:
            symbol: Symbol to normalize
            
        Returns:
            List of symbol format variants
        """
        try:
            # Remove perpetual suffix: 'BTC/USDT:USDT' -> 'BTC/USDT'
            base_symbol = symbol.split(':')[0]
            
            # Split base and quote
            if '/' in base_symbol:
                parts = base_symbol.split('/')
            elif '-' in base_symbol:
                parts = base_symbol.split('-')
            else:
                return [symbol]  # Unrecognized format
            
            if len(parts) != 2:
                return [symbol]
                
            base, quote = parts[0], parts[1]
            
            # Generate different format variants (original first, then alternatives)
            variants = [
                symbol,                     # Original format
                f"{base}/{quote}",          # CCXT standard
                f"{base}/{quote}:{quote}",  # CCXT perpetual
                f"{base}-{quote}",          # BingX native format
                f"{base}{quote}",           # Compact format (BTCUSDT)
            ]
            
            # Return unique ordered list
            seen = set()
            ordered = []
            for v in variants:
                if v not in seen:
                    ordered.append(v)
                    seen.add(v)
            
            return ordered
            
        except Exception as e:
            logger.warning(f"Symbol normalization failed for {symbol}: {e}")
            return [symbol]
    
    async def _wait_for_websocket_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait for WebSocket manager's collector to be ready.
        
        This method prevents race conditions where MarketDataPipeline tries to
        inject data before the WebSocketManager's collector is fully initialized.
        
        Args:
            timeout: Maximum seconds to wait for collector (default: 10.0)
        
        Returns:
            True if collector is ready, False if timeout or no WebSocket manager
        """
        if not self.websocket_manager:
            logger.debug("[WS-READY] No WebSocket manager configured")
            return False
        
        start_time = asyncio.get_event_loop().time()
        check_interval = 0.1  # Check every 100ms
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Check timeout
            if elapsed >= timeout:
                logger.warning(f"[WS-READY] ⏱️ Timeout after {elapsed:.1f}s waiting for WebSocket collector")
                return False
            
            # Check if collector is ready
            if hasattr(self.websocket_manager, 'is_collector_ready'):
                if self.websocket_manager.is_collector_ready():
                    logger.info(f"[WS-READY] ✅ WebSocket collector ready after {elapsed:.2f}s")
                    return True
            elif hasattr(self.websocket_manager, 'collector') and self.websocket_manager.collector:
                logger.info(f"[WS-READY] ✅ WebSocket collector ready after {elapsed:.2f}s")
                return True
            
            # Wait before next check
            await asyncio.sleep(check_interval)
    
    async def prime_data_buffers_async(self, symbols: List[str], timeframes: List[str]):
        """
        Asynchronously fetches historical data for all symbols and timeframes to prime the data buffers.
        This is called at startup to prevent "Insufficient data" errors for indicators.
        """
        logger.info(f"[PRIME] Starting historical data priming for {len(symbols)} symbols and {len(timeframes)} timeframes.")

        # Optional operator kill-switch (kept explicit to avoid silent behavior changes).
        try:
            universe_cfg = self.config.get('universe', {}) if isinstance(self.config, dict) else {}
            prefetch_cfg = universe_cfg.get('prefetch', {}) if isinstance(universe_cfg, dict) else {}
            prefetch_enabled = bool(prefetch_cfg.get('enabled', True))
        except Exception:
            prefetch_enabled = True
            prefetch_cfg = {}

        if not prefetch_enabled:
            logger.warning("[PRIME] Prefetch disabled via universe.prefetch.enabled=false; skipping historical priming.")
            return
        
        # CRITICAL: Wait for WebSocket collector to be ready before priming
        if not await self._wait_for_websocket_ready(timeout=10.0):
            logger.warning("[PRIME] WebSocket collector not ready after 10s timeout - proceeding without WebSocket injection")
        
        tasks = []
        # We need enough data for indicators like EMA(200) and VWAP (target at least 2 days of 1m bars ~2880+)
        indicators_cfg = self.config.get('indicators', {}) if isinstance(self.config, dict) else {}
        try:
            ema_slow = int(indicators_cfg.get('ema_slow', 200))
        except Exception:
            ema_slow = 200

        limit = ema_slow + self.INDICATOR_WARMUP_BUFFER + self.FETCH_SAFETY_BUFFER

        # Single source of truth: universe.prefetch.startup_candle_count (schema enforces >= 2000).
        startup_floor = 3000
        try:
            if isinstance(prefetch_cfg, dict) and prefetch_cfg.get('startup_candle_count') is not None:
                startup_floor = int(prefetch_cfg.get('startup_candle_count'))
        except Exception:
            startup_floor = 3000
        if startup_floor < 2000:
            logger.warning("[PRIME] startup_candle_count=%s too low; clamping to 2000.", startup_floor)
            startup_floor = 2000

        required_limit = max(limit, startup_floor)
        logger.info("[PRIME] Priming candle target: required_limit=%s (ema_slow=%s, startup_floor=%s)", required_limit, ema_slow, startup_floor)

        for symbol in symbols:
            for timeframe in timeframes:
                # Assuming the first available exchange is the primary one for fetching.
                # A more complex logic could try multiple exchanges.
                exchange_name = next(iter(self.exchanges.keys()), None)
                if not exchange_name:
                    logger.error("[PRIME] No exchanges available to prime data.")
                    continue
                
                client = self.exchanges[exchange_name]
                tasks.append(self._fetch_and_store_async(client, exchange_name, symbol, timeframe, required_limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if isinstance(r, bool) and r)
        failure_count = len(results) - success_count
        
        logger.info(f"[PRIME] Historical data priming complete. Success: {success_count}, Failures: {failure_count}")
        if failure_count > 0:
            logger.warning("[PRIME] Some symbols/timeframes failed to load historical data. This may affect initial signal generation.")

    async def _fetch_and_store_async(self, client: CcxtClient, exchange_name: str, symbol: str, timeframe: str, limit: int) -> bool:
        """Helper to fetch and store data for a single symbol/timeframe asynchronously."""
        try:
            self.total_requests += 1
            
            # Paginated fetch to accumulate required_limit candles
            timeframe_ms = TIMEFRAME_SECONDS.get(timeframe, 60) * 1000
            chunk_size = 250
            all_chunks = []
            collected = 0
            # Start from (now - required_limit * interval)
            start_since = int(time.time() * 1000) - (limit * timeframe_ms)
            current_since = start_since

            while collected < limit:
                remaining = limit - collected
                chunk_limit = min(chunk_size, remaining)
                chunk_df = await client.ohlcv(symbol, timeframe, chunk_limit, add_indicators=False, since=current_since)
                if chunk_df is None or chunk_df.empty:
                    break
                all_chunks.append(chunk_df)
                collected += len(chunk_df)
                last_ts_ms = int(chunk_df.index[-1].timestamp() * 1000)
                next_since = last_ts_ms + timeframe_ms
                if next_since <= current_since:
                    break
                current_since = next_since

            if not all_chunks:
                logger.warning(f"[PRIME] Empty data for {symbol} {timeframe} from {exchange_name}")
                self.failed_requests += 1
                return False

            df = pd.concat(all_chunks).sort_index()
            df = df[~df.index.duplicated(keep='last')]
            if len(df) > limit:
                df = df.tail(limit)

            df = self._filter_closed_dataframe(df, timeframe, context="[PRIME]")
            if df is None or df.empty:
                logger.warning(f"[PRIME] No closed candles available after filtering for {symbol} {timeframe}")
                self.failed_requests += 1
                return False

            df.attrs["timeframe"] = timeframe
            df = add_indicators(df, self.config.get('indicators'))
        
            logger.info(f"✅ [PRIME] Loaded {len(df)} historical candles for {exchange_name} {symbol} {timeframe}")
            
            # --- VERİ ENJEKSİYON BLOĞU ---
            # Bu blok, verinin WebSocket deposuna aktarılmasını sağlar.
            if self.websocket_manager and hasattr(self.websocket_manager, 'collector') and self.websocket_manager.collector:
                try:
                    # CCXT sembol formatını ('BTC/USDT') WebSocket formatına ('BTC/USDT:USDT') çevir.
                    ws_symbol = f"{symbol}:{symbol.split('/')[-1]}" if ':' not in symbol and symbol.endswith('/USDT') else symbol
                    
                    # Collector'a DataFrame'i doğrudan gönder.
                    self.websocket_manager.collector.prime_buffer_with_dataframe(exchange_name, ws_symbol, timeframe, df)
                    logger.info(f"✅ [INJECT] Successfully injected {len(df)} candles into WebSocket buffer for {ws_symbol} {timeframe}")
                except Exception as e:
                    logger.error(f"❌ [INJECT] Failed to inject data into WebSocket buffer for {symbol} {timeframe}: {e}", exc_info=True)
                    # Enjeksiyon başarısız olursa bile prime işlemini başarısız sayma, sadece logla.
            else:
                logger.debug(f"[INJECT] No WebSocket manager or collector available - skipping data injection for {symbol} {timeframe}")
    
            return True
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"❌ [PRIME] Failed to fetch {symbol} {timeframe} on {exchange_name}: {e}", exc_info=True)
            return False

    def start_feeds(self, symbols: List[str], timeframes: List[str] = ['30m', '1h']) -> Dict[str, Any]:
        """
        Start data feeds for specified symbols and timeframes.
        
        Args:
            symbols: List of trading symbols to fetch (e.g., ['BTC/USDT:USDT', 'ETH/USDT:USDT'])
            timeframes: List of timeframes to fetch (default: ['30m', '1h'])
        
        Returns:
            Dict with summary of data collection results
        """
        logger.info(f"🔄 Starting data feeds for {len(symbols)} symbols across {len(timeframes)} timeframes")
        self.is_running = True
        
        results = {
            'symbols_processed': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'exchanges_used': set(),
            'errors': []
        }
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Try to fetch from best available exchange
                success = self._fetch_and_store(symbol, timeframe, results)
                
                if success:
                    results['symbols_processed'] += 1
                
                # Rate limiting between symbol fetches
                time.sleep(0.1)
        
        results['exchanges_used'] = list(results['exchanges_used'])
        
        logger.info(f"✅ Data feeds started: {results['successful_fetches']} successful, "
                   f"{results['failed_fetches']} failed")
        
        return results
    
    def _fetch_and_store(self, symbol: str, timeframe: str, results: Dict[str, Any]) -> bool:
        """
        Fetch data from exchanges and store with retry logic.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            results: Results dict to update
        
        Returns:
            True if fetch succeeded, False otherwise
        """
        # Get buffer limit for this timeframe
        limit = self.BUFFER_LIMITS.get(timeframe, 500)
        
        # Try each exchange with exponential backoff
        for exchange_name, client in self.exchanges.items():
            retry_count = 0
            max_retries = 3
            base_delay = 0.5
            
            while retry_count < max_retries:
                try:
                    self.total_requests += 1
                    
                    # Validate symbol first
                    validated_symbol = client.validate_and_get_symbol(symbol)
                    
                    # Fetch OHLCV data
                    if limit > 500 and hasattr(client, 'fetch_ohlcv_bulk'):
                        # Bu senkron bir fonksiyon, async değil.
                        ohlcv_data = client.fetch_ohlcv_bulk(validated_symbol, timeframe, limit)
                    else:
                        # Bu da senkron olmalı. Eğer async ise, burası çalışmaz.
                        # Ancak ccxt_client'taki ohlcv async, bu yüzden bu çağrı sorunlu olabilir.
                        # Şimdilik async olmadığını varsayıyoruz, ama burası potansiyel bir hata noktası.
                        # Şimdilik, ohlcv'nin de DataFrame döndürdüğünü varsayalım.
                        ohlcv_data = client.ohlcv(validated_symbol, timeframe, limit)
                    
                    # --- DEĞİŞİKLİK 3: Güvenli DataFrame kontrolü senkron fonksiyona da eklendi ---
                    if ohlcv_data is None or ohlcv_data.empty:
                        logger.warning(f"[SYNC] Empty data for {symbol} {timeframe} from {exchange_name}")
                        self.failed_requests += 1
                        break

                    df = self._filter_closed_dataframe(ohlcv_data, timeframe, context="[SYNC]")
                    if df is None or df.empty:
                        logger.warning(f"[SYNC] No closed candles available after filtering for {symbol} {timeframe}")
                        self.failed_requests += 1
                        break
                    
                    # Add indicators
                    df.attrs["timeframe"] = timeframe
                    df = add_indicators(df, self.config.get('indicators'))
                    
                    # Store data - DEFENSIVE checks before WebSocket injection
                    if not self.websocket_manager:
                        logger.debug(f"[INJECT-SYNC] No WebSocket manager - skipping data injection for {symbol} {timeframe}")
                    elif not hasattr(self.websocket_manager, 'collector') or not self.websocket_manager.collector:
                        logger.warning(f"⚠️ [INJECT-SYNC] WebSocket manager exists but collector not found. Skipping data injection for {symbol} {timeframe}")
                    else:
                        try:
                            ws_symbol = f"{symbol}:{symbol.split('/')[-1]}" if ':' not in symbol and symbol.endswith('/USDT') else symbol
                            self.websocket_manager.collector.prime_buffer_with_dataframe(exchange_name, ws_symbol, timeframe, df)
                            logger.debug(f"✅ [INJECT-SYNC] Injected {len(df)} candles for {ws_symbol} {timeframe}")
                        except Exception as e:
                            logger.error(f"❌ [INJECT-SYNC] Failed to inject data: {e}")
                            # Don't fail - continue without injection
                    
                    results['successful_fetches'] += 1
                    results['exchanges_used'].add(exchange_name)
                    
                    logger.info(f"✅ {exchange_name} {symbol} {timeframe}: {len(df)} candles")
                    return True
                    
                except Exception as e:
                    retry_count += 1
                    self.failed_requests += 1
                    
                    if retry_count < max_retries:
                        # Exponential backoff
                        delay = base_delay * (2 ** (retry_count - 1))
                        logger.warning(f"⚠️ Retry {retry_count}/{max_retries} for {symbol} {timeframe} "
                                     f"on {exchange_name} after {delay}s: {type(e).__name__}: {e}")
                        time.sleep(delay)
                    else:
                        error_msg = f"{exchange_name} {symbol} {timeframe}: {type(e).__name__}: {e}"
                        logger.error(f"❌ Failed after {max_retries} retries: {error_msg}")
                        results['errors'].append(error_msg)
                        results['failed_fetches'] += 1
                        break
        
        return False
    
    def _ohlcv_to_dataframe(self, ohlcv_data: List[List]) -> pd.DataFrame:
        """
        Convert OHLCV list data to pandas DataFrame.
        
        Args:
            ohlcv_data: List of OHLCV candles [[timestamp, open, high, low, close, volume], ...]
        
        Returns:
            DataFrame with timestamp index and OHLCV columns
        """
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(ohlcv_data, columns=cols)
        
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
        
        return df
    
    def _store_data(self, exchange: str, symbol: str, timeframe: str, df: pd.DataFrame):
        """
        DEPRECATED: This method is now a no-op. Data is stored centrally.
        It's kept for backward compatibility to prevent crashes if called.
        """
        # Bu metodun içi artık boş. Sadece eski koda uyumluluk için var.
        # logger.warning(f"DEPRECATED: _store_data for {exchange}:{symbol}:{timeframe} was called. This is a no-op.")
        pass
    
    # ------------------- DÜZELTİLMİŞ METOT -------------------
    async def get_candles(self, symbol: str, timeframe: str, exchange: str = None, limit: int = None, include_forming: bool = False) -> Optional[pd.DataFrame]:
        """
        Return closed-only OHLCV candles for a symbol/timeframe.
        This is an alias to get_latest_ohlcv to make intent explicit.
        """
        return await self.get_latest_ohlcv(symbol, timeframe, exchange, limit=limit, include_forming=include_forming)

    async def get_latest_ohlcv(self, symbol: str, timeframe: str, exchange: str = None, limit: int = None, include_forming: bool = False) -> Optional[pd.DataFrame]:
        """
        Get latest CLOSED OHLCV data with a robust WebSocket-first approach.
        (GÜNCELLENDİ: WebSocket verisini doğru işler ve REST fallback'i sadece gerektiğinde kullanır.)

        Priority:
        1. Try to get data from WebSocket collector (fastest, real-time).
        2. Fall back to REST API only if WebSocket data is insufficient or unavailable.
        3. Return None only if both sources fail.
        
        Technical indicators are added consistently before returning.
        """
        if not timeframe:
            logger.warning(
                "[MKT] get_latest_ohlcv called with empty timeframe; defaulting to 5m | sym=%s",
                symbol,
            )
            timeframe = "5m"

        df = None
        limit_override = limit
        merge_action = "none"
        fallback_reason: Optional[str] = None
        
        # STEP 1: Try WebSocket first (real-time data)
        if self.websocket_manager and self.websocket_manager.collector:
            try:
                # Determine which exchange to use
                ws_exchange = exchange if exchange else (next(iter(self.exchanges.keys())) if self.exchanges else self.DEFAULT_EXCHANGE)
                if isinstance(ws_exchange, str):
                    ws_exchange = ws_exchange.lower()
                canonical_exchange_id = self._canonical_exchange_id(ws_exchange)
                
                # Get required number of candles for indicators
                # Adding buffer to ensure sufficient data for indicator calculations
                limit_ws = limit_override or (
                    self.config.get('indicators', {}).get('ema_slow', 200)
                    + self.INDICATOR_WARMUP_BUFFER
                    + self.FETCH_SAFETY_BUFFER
                )
                
                # WebSocket collector'dan ham OHLCV listesini al
                ohlcv_list = self.websocket_manager.collector.get_latest_ohlcv(
                    exchange=ws_exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit_ws
                )

                # Gelen verinin doğru formatta olduğunu doğrula
                if ohlcv_list and isinstance(ohlcv_list, list) and len(ohlcv_list) > 0:
                    # Ham OHLCV listesini DataFrame'e çevir
                    closed_df = self._ohlcv_to_dataframe(ohlcv_list)
                    
                    if closed_df is not None and not closed_df.empty:
                        logger.debug(f"[WS] Retrieved {len(closed_df)} CLOSED candles for {symbol} {timeframe}")
                        state = None
                        if hasattr(self.websocket_manager.collector, "get_state"):
                            state = self.websocket_manager.collector.get_state(ws_exchange, symbol, timeframe)
                            logger.debug(
                                f"[WS-STATE] last_closed_ts={state.get('last_closed_ts')} "
                                f"forming_ts={state.get('forming_ts')} forming_last_update_ts={state.get('forming_last_update_ts')} "
                                f"gap_count={state.get('gap_count')}"
                            )
                        # İndikatörleri ekle ve hemen döndür. REST API'ye gitme.
                        closed_df.attrs["timeframe"] = timeframe
                        closed_df = add_indicators(closed_df, self.config.get('indicators'))
                        closed_df.attrs.setdefault("includes_forming", False)
                        closed_df.attrs["ohlcv_source"] = "ws"
                        closed_df.attrs["retrieved_at"] = datetime.now(timezone.utc).isoformat()
                        if state:
                            closed_df.attrs["last_closed_ts"] = state.get("last_closed_ts")
                            closed_df.attrs["forming_ts"] = state.get("forming_ts")
                            closed_df.attrs["forming_last_update_ts"] = state.get("forming_last_update_ts")
                            closed_df.attrs["gap_count"] = state.get("gap_count")

                            # Pre-compute pivot/bucket observability even if hybrid merge is skipped.
                            try:
                                interval_sec = TIMEFRAME_SECONDS.get(timeframe)
                                if interval_sec:
                                    interval_ms = int(interval_sec) * 1000
                                    now_ms = int(time.time() * 1000)
                                    expected_open = (now_ms // interval_ms) * interval_ms
                                    forming_ot = state.get("forming_ts")
                                    if forming_ot is not None:
                                        closed_df.attrs["forming_open_time"] = int(forming_ot)
                                        closed_df.attrs["expected_open"] = int(expected_open)
                                        closed_df.attrs["bucket_delta_ms"] = int(expected_open - int(forming_ot))
                                    ws_cfg_tmp = self.config.get('websocket', {}) if isinstance(self.config, dict) else {}
                                    pivot_grace_ms = int(ws_cfg_tmp.get('hybrid_pivot_grace_ms', 90000))
                                    by_tf = ws_cfg_tmp.get('hybrid_pivot_grace_ms_by_tf')
                                    if isinstance(by_tf, dict) and timeframe in by_tf:
                                        try:
                                            pivot_grace_ms = int(by_tf.get(timeframe))
                                        except Exception:
                                            pass
                                    closed_df.attrs["pivot_grace_ms"] = pivot_grace_ms
                            except Exception:
                                pass

                        # Optional hybrid state machine (disabled by default): can force closed-only
                        # for a cooldown window after repeated hybrid failures.
                        ws_cfg = self.config.get('websocket', {}) if isinstance(self.config, dict) else {}
                        sm_enabled = bool(ws_cfg.get('hybrid_state_machine_enabled', False))
                        sm_failures = int(ws_cfg.get('hybrid_failures_before_cooldown', 3))
                        sm_cooldown_ms = int(ws_cfg.get('hybrid_cooldown_ms', 60000))
                        state_key = self._hybrid_state_key(canonical_exchange_id, symbol, timeframe)
                        now_ms = int(time.time() * 1000)

                        if include_forming and sm_enabled:
                            slot = self._hybrid_state.get(state_key) or {"fail_count": 0, "cooldown_until_ms": 0}
                            cooldown_until = int(slot.get("cooldown_until_ms", 0) or 0)
                            if cooldown_until and now_ms < cooldown_until:
                                include_forming = False
                                merge_action = "none"
                                fallback_reason = "hybrid_cooldown"

                        if include_forming:
                            closed_df, merge_action, fallback_reason = self._merge_forming_candle(
                                closed_df,
                                ws_exchange,
                                symbol,
                                timeframe,
                                forming_last_update_ts=(state.get("forming_last_update_ts") if state else None),
                            )

                        if sm_enabled:
                            slot = self._hybrid_state.get(state_key) or {"fail_count": 0, "cooldown_until_ms": 0}
                            # Don't count "hybrid_cooldown" as a failure; it's the result of the state machine.
                            if fallback_reason and fallback_reason != "hybrid_cooldown":
                                slot["fail_count"] = int(slot.get("fail_count", 0) or 0) + 1
                                if sm_failures > 0 and slot["fail_count"] >= sm_failures and sm_cooldown_ms > 0:
                                    slot["cooldown_until_ms"] = now_ms + sm_cooldown_ms
                                    slot["fail_count"] = 0
                            elif not fallback_reason:
                                slot["fail_count"] = 0
                                slot["cooldown_until_ms"] = 0
                            self._hybrid_state[state_key] = slot
                        df = closed_df
                        # Unified hybrid attrs (always present for deterministic observability)
                        df.attrs["merge_action"] = merge_action or "none"
                        # IMPORTANT: store None (not the string 'none') when no fallback applies.
                        prev_reason = df.attrs.get("fallback_reason")
                        if isinstance(prev_reason, str) and prev_reason.strip().lower() in ("none", ""):
                            prev_reason = None
                        df.attrs["fallback_reason"] = (fallback_reason or None) if (fallback_reason or None) else prev_reason
                        df.attrs["includes_forming"] = bool(df.attrs.get("includes_forming", False))

                        # Hybrid metrics: increment every successful WS-returning call.
                        state_key = self._hybrid_state_key(canonical_exchange_id, symbol, timeframe)
                        self._maybe_log_state_key_sample(
                            source="ws",
                            raw_exchange=str(ws_exchange),
                            canonical_exchange_id=canonical_exchange_id,
                            symbol=symbol,
                            timeframe=timeframe,
                            state_key=state_key,
                        )
                        self._record_hybrid_metrics(
                            state_key=state_key,
                            fallback_reason=df.attrs.get("fallback_reason"),
                            timeframe=timeframe,
                            symbol=symbol,
                        )

                        return df
                else:
                    logger.debug(f"⚠️ WebSocket collector returned empty or invalid data for {symbol} {timeframe}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error getting data from WebSocket collector for {symbol} {timeframe}: {e}")
        else:
            logger.debug(f"ℹ️ WebSocketManager or its collector not available for {symbol} {timeframe}")
        
        # STEP 2: REST API Fallback (Sadece WebSocket başarısız olursa bu blok çalışır)
        logger.info(f"🔄 Falling back to REST API for {symbol} {timeframe}")
        
        try:
            # Kullanılacak exchange'i belirle
            if not exchange and self.exchanges:
                exchange = next(iter(self.exchanges.keys()))
            
            if not exchange or exchange not in self.exchanges:
                logger.error(f"❌ No valid exchange available for REST API fallback")
                return None
            
            client = self.exchanges[exchange]
            canonical_exchange_id = self._canonical_exchange_id(exchange)
            
            # Gerekli mum sayısını belirle
            limit_rest = limit_override or (
                self.config.get('indicators', {}).get('ema_slow', 200)
                + self.INDICATOR_WARMUP_BUFFER
                + self.FETCH_SAFETY_BUFFER
            )

            # REST API'yi çağır (zaten async)
            ohlcv_df = await client.ohlcv(symbol, timeframe, limit_rest, add_indicators=False)
            
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(f"[REST] REST API returned empty data for {symbol} {timeframe}")
                return None
            
            ohlcv_df = self._filter_closed_dataframe(ohlcv_df, timeframe, context="[REST]")
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(f"[REST] No closed candles available after filtering for {symbol} {timeframe}")
                return None
            
            # İndikatörleri ekle
            ohlcv_df.attrs["timeframe"] = timeframe
            df = add_indicators(ohlcv_df, self.config.get('indicators'))
            df.attrs["ohlcv_source"] = "rest"
            df.attrs["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            df.attrs["includes_forming"] = False
            df.attrs["merge_action"] = "rest_closed_only"
            df.attrs["fallback_reason"] = df.attrs.get("fallback_reason", None)
            logger.info(f"[REST] Retrieved {len(df)} candles from REST API for {symbol} {timeframe}")

            # Hybrid metrics: also count REST returns so totals are monotonic even if WS is unavailable.
            try:
                state_key = self._hybrid_state_key(canonical_exchange_id, symbol, timeframe)
                self._maybe_log_state_key_sample(
                    source="rest",
                    raw_exchange=str(exchange),
                    canonical_exchange_id=canonical_exchange_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    state_key=state_key,
                )
                self._record_hybrid_metrics(
                    state_key=state_key,
                    fallback_reason=df.attrs.get("fallback_reason"),
                    timeframe=timeframe,
                    symbol=symbol,
                )
            except Exception:
                pass
            return df
                
        except Exception as e:
            logger.error(f"❌ REST API fallback failed for {symbol} {timeframe}: {e}", exc_info=True)
            return None
    # ------------------- DÜZELTME SONU -------------------

    def _merge_forming_candle(
        self,
        closed_df: pd.DataFrame,
        exchange: str,
        symbol: str,
        timeframe: str,
        forming_last_update_ts: Optional[int] = None,
    ) -> tuple[pd.DataFrame, str, Optional[str]]:
        """Append/replace closed_df with forming candle while keeping volatility from closed bars.

        Hybrid policy notes:
        - Pivot/bucket policy is evaluated first (single source of truth).
        - Forming update-age staleness is evaluated only when bucket_delta==0.
        - Replace ordering is reachable: duplicate last-open is handled before step mismatch.
        """
        merge_action = "none"
        fallback_reason: Optional[str] = None

        # Used only for metrics observability (when available).
        state_key = None
        try:
            canonical_exchange_id = self._canonical_exchange_id(exchange)
            state_key = self._hybrid_state_key(canonical_exchange_id, symbol, timeframe)
        except Exception:
            state_key = None

        interval_ms = TIMEFRAME_SECONDS.get(timeframe)
        if interval_ms:
            interval_ms *= 1000
        if not interval_ms:
            closed_df.attrs["includes_forming"] = False
            closed_df.attrs["merge_action"] = "none"
            closed_df.attrs["fallback_reason"] = "unknown_timeframe"
            return closed_df, "none", "unknown_timeframe"

        forming = self.websocket_manager.collector.get_forming_ohlcv(exchange, symbol, timeframe)
        if not forming:
            closed_df.attrs["includes_forming"] = False
            closed_df.attrs["merge_action"] = "none"
            closed_df.attrs["fallback_reason"] = "no_forming"
            return closed_df, "none", "no_forming"

        try:
            forming_open_ms = int(forming[0])
        except Exception:
            logger.warning(f"[HYBRID-INJECT] Invalid forming payload for {symbol} {timeframe}: {forming}")
            closed_df.attrs["includes_forming"] = False
            closed_df.attrs["merge_action"] = "none"
            closed_df.attrs["fallback_reason"] = "invalid_forming_payload"
            return closed_df, "none", "invalid_forming_payload"

        last_open_ms = int(closed_df.index[-1].timestamp() * 1000)
        df_len_before = len(closed_df)

        # --- Unified hybrid observability (attrs) ---
        now_ms = int(time.time() * 1000)
        expected_open = (now_ms // interval_ms) * interval_ms
        bucket_delta_ms = int(expected_open - forming_open_ms)

        ws_config = self.config.get('websocket', {}) if isinstance(self.config, dict) else {}
        pivot_enabled = bool(ws_config.get('hybrid_pivot_grace_enabled', True))
        pivot_grace_ms = int(ws_config.get('hybrid_pivot_grace_ms', 90000))
        by_tf = ws_config.get('hybrid_pivot_grace_ms_by_tf')
        if isinstance(by_tf, dict) and timeframe in by_tf:
            try:
                pivot_grace_ms = int(by_tf.get(timeframe))
            except Exception:
                pass

        forming_update_age_ms: Optional[int] = None
        try:
            if forming_last_update_ts is not None:
                forming_update_age_ms = now_ms - int(forming_last_update_ts)
        except Exception:
            forming_update_age_ms = None

        closed_df.attrs["forming_open_time"] = forming_open_ms
        closed_df.attrs["forming_last_update_ts"] = forming_last_update_ts
        closed_df.attrs["forming_update_age_ms"] = forming_update_age_ms
        closed_df.attrs["expected_open"] = expected_open
        closed_df.attrs["bucket_delta_ms"] = bucket_delta_ms
        closed_df.attrs["pivot_grace_ms"] = pivot_grace_ms

        # --- Pivot-grace policy (single source of truth) ---
        # bucket_delta==0: healthy bucket (eligible to proceed)
        # bucket_delta==tf: 1 bucket behind -> grace window downgrade (closed-only) or pivot-stale
        # else: hard reject
        accepted_prev_bucket = False

        if bucket_delta_ms == interval_ms:
            within_grace = bool(pivot_enabled and now_ms <= (expected_open + max(pivot_grace_ms, 0)))
            accept_prev_bucket = bool(ws_config.get('pivot_grace_accept_prev_bucket', False))

            # Conservative default: within grace still downgrades to closed-only for determinism.
            # Opt-in: accept prev-bucket forming updates within grace if they are fresh.
            if within_grace and accept_prev_bucket:
                # Only accept if the update age is known and within staleness threshold.
                forming_update_stale_ms = int(ws_config.get('forming_update_stale_ms', 15000))
                age_ok = (
                    forming_update_age_ms is not None
                    and forming_update_stale_ms > 0
                    and int(forming_update_age_ms) <= int(forming_update_stale_ms)
                )
                if age_ok:
                    logger.info(
                        f"[HYBRID-INJECT] Pivot grace ACCEPT prev bucket | symbol={symbol} tf={timeframe} "
                        f"expected_open={expected_open} forming_ot={forming_open_ms} bucket_delta_ms={bucket_delta_ms} "
                        f"pivot_grace_ms={pivot_grace_ms} forming_update_age_ms={forming_update_age_ms}"
                    )
                    accepted_prev_bucket = True
                    # Proceed with merge logic below (treat as acceptable)
                else:
                    fallback_reason = "pivot_grace_prev_bucket"
                    logger.info(
                        f"[HYBRID-INJECT] Pivot grace downgrade | symbol={symbol} tf={timeframe} "
                        f"expected_open={expected_open} forming_ot={forming_open_ms} bucket_delta_ms={bucket_delta_ms} "
                        f"pivot_grace_ms={pivot_grace_ms} forming_update_age_ms={forming_update_age_ms} "
                        "explain=previous bucket still updating within grace window; using closed-only for safety"
                    )
                    closed_df.attrs["includes_forming"] = False
                    closed_df.attrs["merge_action"] = "none"
                    closed_df.attrs["fallback_reason"] = fallback_reason
                    if state_key:
                        self._hybrid_last_inject_ts_ms[state_key] = int(now_ms)
                    return closed_df, "none", fallback_reason
            else:
                if within_grace:
                    fallback_reason = "pivot_grace_prev_bucket"
                    logger.info(
                        f"[HYBRID-INJECT] Pivot grace downgrade | symbol={symbol} tf={timeframe} "
                        f"expected_open={expected_open} forming_ot={forming_open_ms} bucket_delta_ms={bucket_delta_ms} "
                        f"pivot_grace_ms={pivot_grace_ms} forming_update_age_ms={forming_update_age_ms} "
                        "explain=previous bucket still updating within grace window; using closed-only for safety"
                    )
                else:
                    fallback_reason = "pivot_stale_prev_bucket"
                    logger.warning(
                        f"[HYBRID-INJECT] Pivot stale (prev bucket) | symbol={symbol} tf={timeframe} "
                        f"expected_open={expected_open} forming_ot={forming_open_ms} bucket_delta_ms={bucket_delta_ms} "
                        f"pivot_grace_ms={pivot_grace_ms} forming_update_age_ms={forming_update_age_ms}"
                    )

                closed_df.attrs["includes_forming"] = False
                closed_df.attrs["merge_action"] = "none"
                closed_df.attrs["fallback_reason"] = fallback_reason
                if state_key:
                    self._hybrid_last_inject_ts_ms[state_key] = int(now_ms)
                return closed_df, "none", fallback_reason

        if bucket_delta_ms != 0 and not accepted_prev_bucket:
            # forming in the future or too old or misaligned
            if bucket_delta_ms < 0:
                fallback_reason = "forming_future_bucket"
            elif bucket_delta_ms > interval_ms:
                fallback_reason = "forming_too_old"
            else:
                fallback_reason = "bucket_misaligned"
            logger.warning(
                f"[HYBRID-INJECT] Fallback due to bucket delta | symbol={symbol} tf={timeframe} "
                f"expected_open={expected_open} forming_ot={forming_open_ms} bucket_delta_ms={bucket_delta_ms} "
                f"pivot_grace_ms={pivot_grace_ms} forming_update_age_ms={forming_update_age_ms} reason={fallback_reason}"
            )
            closed_df.attrs["includes_forming"] = False
            closed_df.attrs["merge_action"] = "none"
            closed_df.attrs["fallback_reason"] = fallback_reason
            if state_key:
                self._hybrid_last_inject_ts_ms[state_key] = int(now_ms)
            return closed_df, "none", fallback_reason

        # Centralized staleness: require that the forming candle has been updated recently (bucket_delta==0 only).
        forming_update_stale_ms = int(ws_config.get('forming_update_stale_ms', 15000))
        if forming_update_stale_ms > 0:
            last_update_ms: Optional[int]
            try:
                last_update_ms = int(forming_last_update_ts) if forming_last_update_ts is not None else None
            except Exception:
                last_update_ms = None

            if last_update_ms is None:
                fallback_reason = "forming_update_unknown"
                logger.warning(
                    f"[HYBRID-INJECT] Fallback due to unknown forming update timestamp | symbol={symbol} tf={timeframe} "
                    f"expected_open={expected_open} forming_ot={forming_open_ms}"
                )
                closed_df.attrs["includes_forming"] = False
                closed_df.attrs["merge_action"] = "none"
                closed_df.attrs["fallback_reason"] = fallback_reason
                if state_key:
                    self._hybrid_last_inject_ts_ms[state_key] = int(now_ms)
                return closed_df, "none", fallback_reason

            age_ms = now_ms - last_update_ms
            closed_df.attrs["forming_update_age_ms"] = age_ms
            if age_ms > forming_update_stale_ms:
                fallback_reason = "forming_update_stale"
                logger.warning(
                    f"[HYBRID-INJECT] Fallback due to stale forming updates | symbol={symbol} tf={timeframe} "
                    f"expected_open={expected_open} forming_ot={forming_open_ms} age_ms={age_ms} stale_ms={forming_update_stale_ms}"
                )
                closed_df.attrs["includes_forming"] = False
                closed_df.attrs["merge_action"] = "none"
                closed_df.attrs["fallback_reason"] = fallback_reason
                if state_key:
                    self._hybrid_last_inject_ts_ms[state_key] = int(now_ms)
                return closed_df, "none", fallback_reason

        # Replace/append ordering (reachable): duplicate handled before step mismatch.
        expected_next = last_open_ms + interval_ms
        forming_df = self._ohlcv_to_dataframe([forming])

        if forming_open_ms == last_open_ms:
            merge_action = "replaced_last"
            base_df = closed_df.iloc[:-1]
            logger.info(
                f"[HYBRID-INJECT] Replacing last row (dedupe) | symbol={symbol} tf={timeframe} ot={forming_open_ms}"
            )
        elif forming_open_ms == expected_next:
            merge_action = "appended"
            base_df = closed_df
        else:
            fallback_reason = "step_mismatch"
            logger.warning(
                f"[HYBRID-INJECT] Fallback due to step mismatch | symbol={symbol} tf={timeframe} "
                f"last_closed_ot={last_open_ms} forming_ot={forming_open_ms} expected_next={expected_next}"
            )
            closed_df.attrs["includes_forming"] = False
            closed_df.attrs["merge_action"] = "none"
            closed_df.attrs["fallback_reason"] = fallback_reason
            if state_key:
                self._hybrid_last_inject_ts_ms[state_key] = int(now_ms)
            return closed_df, "none", fallback_reason

        merged = pd.concat([base_df, forming_df])
        merged = merged[~merged.index.duplicated(keep="last")]

        if not merged.index.is_monotonic_increasing:
            fallback_reason = "non_monotonic"
            logger.warning(
                f"[HYBRID-INJECT] Fallback due to non-monotonic index | {symbol} {timeframe}"
            )
            closed_df.attrs["includes_forming"] = False
            closed_df.attrs["merge_action"] = "none"
            closed_df.attrs["fallback_reason"] = fallback_reason
            return closed_df, "none", fallback_reason

        # Recompute live RSI including forming candle; keep other indicators from closed bars.
        merged["rsi"] = rsi(merged["close"])
        indicator_cols = [
            c
            for c in closed_df.columns
            if c not in ["open", "high", "low", "close", "volume", "rsi"]
        ]
        for col in indicator_cols:
            merged[col] = merged[col].ffill()

        merged.attrs.update(closed_df.attrs)
        merged.attrs["includes_forming"] = True
        merged.attrs["forming_ts"] = forming_open_ms
        merged.attrs["merge_action"] = merge_action
        merged.attrs["forming_source"] = "ws"
        # IMPORTANT: store None (not the string 'none') when no fallback applies.
        merged.attrs["fallback_reason"] = None

        if state_key:
            self._hybrid_last_inject_ts_ms[state_key] = int(now_ms)

        logger.info(
            f"[HYBRID-INJECT] symbol={symbol} tf={timeframe} last_closed_ot={last_open_ms} "
            f"forming_ot={forming_open_ms} expected_open={expected_open} bucket_delta_ms={bucket_delta_ms} "
            f"forming_update_age_ms={forming_update_age_ms} pivot_grace_ms={pivot_grace_ms} "
            f"len_before={df_len_before} len_after={len(merged)} merge_action={merge_action} fallback_reason=none"
        )

        return merged, merge_action, fallback_reason

    def get_live_trigger_price(
        self,
        symbol: str,
        timeframe: str,
        source: str = "mid",
        exchange: Optional[str] = None,
        forming_close: Optional[float] = None,
    ) -> tuple[Optional[float], str, str]:
        """Return preferred trigger price with safe fallbacks (mark→mid→forming_close)."""
        if not self.websocket_manager or not getattr(self.websocket_manager, "collector", None):
            return forming_close, "forming_close", "no_ws"

        ws_exchange = exchange or (next(iter(self.exchanges.keys())) if self.exchanges else self.DEFAULT_EXCHANGE)
        if isinstance(ws_exchange, str):
            ws_exchange = ws_exchange.lower()
        if isinstance(ws_exchange, str):
            ws_exchange = ws_exchange.lower()
        ws_config = self.config.get('websocket', {}) if isinstance(self.config, dict) else {}
        ticker_stale_ms = int(ws_config.get('ticker_stale_ms', 5000))
        diag_interval = max(1, int(ws_config.get('trigger_diag_interval_sec', 60)))

        collector = getattr(self.websocket_manager, "collector", None)
        ticker_sample = collector.get_latest_ticker_sample(ws_exchange, symbol) if collector else None
        ticker = ticker_sample.get('data') if ticker_sample else None
        sample_ts = ticker_sample.get('timestamp') if ticker_sample else None

        now_dt = datetime.now(timezone.utc)
        ticker_age_ms: Optional[float] = None
        if sample_ts:
            try:
                ticker_age_ms = max(0.0, (now_dt - sample_ts).total_seconds() * 1000)
            except Exception:
                ticker_age_ms = None

        reason_tags: List[str] = []
        if ticker is None:
            reason_tags.append("ticker_none")
        elif ticker_age_ms is not None and ticker_age_ms > ticker_stale_ms:
            reason_tags.append("ticker_stale")

        fallback_chain: List[str] = []

        def _add_reason(tag: str) -> None:
            if tag not in fallback_chain:
                fallback_chain.append(tag)

        def _extract_mark(t: Dict[str, Any]) -> Optional[float]:
            keys = ["markPrice", "mark_price", "mark", "indexPrice", "lastPrice"]
            for k in keys:
                if t and k in t:
                    try:
                        val = float(t[k])
                        if val > 0:
                            return val
                    except Exception:
                        continue
            return None

        def _extract_bid_ask(t: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
            if not t:
                return None, None
            bid_keys = ["bestBid", "bid", "bidPrice", "best_bid", "B"]
            ask_keys = ["bestAsk", "ask", "askPrice", "best_ask", "A"]
            bid = ask = None
            for k in bid_keys:
                if k in t:
                    try:
                        bid = float(t[k])
                        break
                    except Exception:
                        continue
            for k in ask_keys:
                if k in t:
                    try:
                        ask = float(t[k])
                        break
                    except Exception:
                        continue
            if (bid is None or bid <= 0) and isinstance(t.get("bids"), list) and t["bids"]:
                try:
                    first_bid = t["bids"][0]
                    bid = float(first_bid[0] if isinstance(first_bid, (list, tuple)) else first_bid)
                except Exception:
                    bid = None
            if (ask is None or ask <= 0) and isinstance(t.get("asks"), list) and t["asks"]:
                try:
                    first_ask = t["asks"][0]
                    ask = float(first_ask[0] if isinstance(first_ask, (list, tuple)) else first_ask)
                except Exception:
                    ask = None
            if bid is not None and bid <= 0:
                bid = None
            if ask is not None and ask <= 0:
                ask = None
            return bid, ask

        def _compute_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
            if bid is not None and ask is not None:
                return (bid + ask) / 2.0
            return None

        def _extract_last(t: Dict[str, Any]) -> Optional[float]:
            keys = ["last", "close", "c", "price", "lastPrice"]
            for k in keys:
                if t and k in t:
                    try:
                        val = float(t[k])
                        if val > 0:
                            return val
                    except Exception:
                        continue
            return None

        mark_price = _extract_mark(ticker)
        bid_price, ask_price = _extract_bid_ask(ticker)
        mid_price = _compute_mid(bid_price, ask_price)
        last_price = _extract_last(ticker)

        resolved_source = source
        price = None

        if source == "mark":
            if mark_price is not None:
                price = mark_price
                resolved_source = "mark"
            else:
                _add_reason("mark_missing")
                if mid_price is not None:
                    price = mid_price
                    resolved_source = "mid"
                else:
                    _add_reason("mid_missing")
                    if last_price is not None:
                        price = last_price
                        resolved_source = "last"
                    else:
                        _add_reason("last_missing")
                        price = forming_close
                        resolved_source = "forming_close"
        elif source == "mid":
            if mid_price is not None:
                price = mid_price
                resolved_source = "mid"
            else:
                _add_reason("mid_missing")
                if mark_price is not None:
                    price = mark_price
                    resolved_source = "mark"
                elif last_price is not None:
                    _add_reason("mark_missing")
                    price = last_price
                    resolved_source = "last"
                else:
                    _add_reason("mark_missing")
                    _add_reason("last_missing")
                    price = forming_close
                    resolved_source = "forming_close"
        elif source == "last":
            if last_price is not None:
                price = last_price
                resolved_source = "last"
            elif mark_price is not None:
                _add_reason("last_missing")
                price = mark_price
                resolved_source = "mark"
            elif mid_price is not None:
                _add_reason("last_missing")
                _add_reason("mark_missing")
                price = mid_price
                resolved_source = "mid"
            else:
                _add_reason("last_missing")
                _add_reason("mark_missing")
                _add_reason("mid_missing")
                price = forming_close
                resolved_source = "forming_close"
        else:
            price = forming_close
            resolved_source = "forming_close"

        if fallback_chain:
            reason_tags.extend(fallback_chain)
        if resolved_source == "forming_close":
            reason_tags.append("fallback_forming_close")

        diag_key = f"{ws_exchange}:{symbol}"
        now_ts = time.time()
        last_diag = self._trigger_diag_last_log.get(diag_key, 0.0)
        if (now_ts - last_diag) >= diag_interval:
            reason_repr = f"[{','.join(reason_tags)}]" if reason_tags else "[none]"

            def _fmt(val: Optional[float]) -> Optional[str]:
                return None if val is None else f"{val:.2f}"

            logger.info(
                "[TRIGGER-DIAG] exchange=%s symbol=%s requested_source=%s resolved_source=%s mark=%s bid=%s ask=%s last=%s ticker_age_ms=%s reason_tags=%s",
                ws_exchange,
                symbol,
                source,
                resolved_source,
                _fmt(mark_price),
                _fmt(bid_price),
                _fmt(ask_price),
                _fmt(last_price),
                int(ticker_age_ms) if ticker_age_ms is not None else None,
                reason_repr,
            )
            self._trigger_diag_last_log[diag_key] = now_ts

        fallback_str = "->".join(fallback_chain) if fallback_chain else "none"
        return price, resolved_source, fallback_str


    async def get_spread_metrics(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        allow_rest_fallback: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Return bid/ask spread metrics using WS ticker sample (fast) with optional REST fallback."""
        if not symbol:
            return None

        ws_exchange = exchange or (next(iter(self.exchanges.keys())) if self.exchanges else self.DEFAULT_EXCHANGE)
        now_dt = datetime.now(timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)

        def _to_float(val: Any) -> Optional[float]:
            try:
                fval = float(val)
                if fval <= 0:
                    return None
                return fval
            except Exception:
                return None

        def _compute_metrics(bid: Optional[float], ask: Optional[float]) -> Dict[str, Any]:
            mid = None
            spread_abs = None
            spread_pct = None
            if bid is not None and ask is not None and ask >= bid:
                mid = (bid + ask) / 2.0
                spread_abs = ask - bid
                if mid and mid > 0:
                    spread_pct = spread_abs / mid
            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_abs": spread_abs,
                "spread_pct": spread_pct,
            }

        # 1) WebSocket-first (collector sample)
        collector = getattr(self.websocket_manager, "collector", None) if self.websocket_manager else None
        if collector:
            try:
                ticker_sample = collector.get_latest_ticker_sample(ws_exchange, symbol)
            except Exception:
                ticker_sample = None
            ticker = ticker_sample.get("data") if isinstance(ticker_sample, dict) else None
            sample_ts = ticker_sample.get("timestamp") if isinstance(ticker_sample, dict) else None

            if isinstance(ticker, dict):
                bid = _to_float(ticker.get("bid") or ticker.get("bestBid") or ticker.get("bidPrice"))
                ask = _to_float(ticker.get("ask") or ticker.get("bestAsk") or ticker.get("askPrice"))
                metrics = _compute_metrics(bid, ask)

                sample_ts_ms = None
                if isinstance(sample_ts, datetime):
                    try:
                        sample_ts_ms = int(sample_ts.timestamp() * 1000)
                    except Exception:
                        sample_ts_ms = None

                age_ms = None if sample_ts_ms is None else max(0, now_ms - sample_ts_ms)
                return {
                    "exchange": ws_exchange,
                    "symbol": symbol,
                    "ts_ms": sample_ts_ms or now_ms,
                    "age_ms": age_ms,
                    "source": "ws",
                    **metrics,
                }

        # 2) REST fallback (async to_thread to avoid blocking event loop)
        if allow_rest_fallback and self.exchanges and ws_exchange in self.exchanges:
            client = self.exchanges[ws_exchange]
            try:
                ticker = await asyncio.to_thread(client.fetch_ticker, symbol)
            except Exception as exc:
                logger.debug("[SPREAD] REST ticker fetch failed for %s on %s: %s", symbol, ws_exchange, exc)
                return None

            if isinstance(ticker, dict):
                bid = _to_float(ticker.get("bid") or ticker.get("bestBid") or ticker.get("bidPrice"))
                ask = _to_float(ticker.get("ask") or ticker.get("bestAsk") or ticker.get("askPrice"))
                metrics = _compute_metrics(bid, ask)

                ts_ms = ticker.get("timestamp")
                try:
                    ts_ms = int(ts_ms) if ts_ms is not None else now_ms
                except Exception:
                    ts_ms = now_ms
                age_ms = max(0, now_ms - ts_ms) if ts_ms else None

                return {
                    "exchange": ws_exchange,
                    "symbol": symbol,
                    "ts_ms": ts_ms,
                    "age_ms": age_ms,
                    "source": "rest",
                    **metrics,
                }

        return None


    async def get_latest_price(self, symbol: str, timeframe: str = '1m', exchange: str = None) -> Optional[float]:
        """
        Get latest price for a symbol with WebSocket-first approach and REST fallback.
        
        This is a centralized method that should be used by all components that need
        current price data. It automatically handles:
        1. WebSocket data retrieval (fastest, real-time)
        2. REST API fallback if WebSocket unavailable
        3. Multiple timeframe fallback strategy
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Preferred timeframe (default: '1m')
            exchange: Optional specific exchange name
        
        Returns:
            Latest close price as float, or None if all sources fail
        """
        # STEP 1: Try WebSocket first (real-time data)
        if self.websocket_manager:
            try:
                # Try to get data from preferred timeframe
                ws_data = self.websocket_manager.get_latest_data(symbol, timeframe, exchange)
                
                if ws_data and isinstance(ws_data, dict) and ws_data.get('ohlcv'):
                    ohlcv = ws_data['ohlcv']
                    if isinstance(ohlcv, list) and len(ohlcv) > 0:
                        latest_candle = ohlcv[-1]
                        if isinstance(latest_candle, list) and len(latest_candle) >= 5:
                            price = float(latest_candle[4])  # Close price
                            if price > 0:
                                logger.debug(f"✅ Price for {symbol} from WebSocket ({timeframe}): ${price:.2f}")
                                return price
                
                # Fallback to other timeframes if preferred one failed
                fallback_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h']
                if timeframe in fallback_timeframes:
                    fallback_timeframes.remove(timeframe)
                
                for tf in fallback_timeframes:
                    ws_data = self.websocket_manager.get_latest_data(symbol, tf, exchange)
                    if ws_data and isinstance(ws_data, dict) and ws_data.get('ohlcv'):
                        ohlcv = ws_data['ohlcv']
                        if isinstance(ohlcv, list) and len(ohlcv) > 0:
                            latest_candle = ohlcv[-1]
                            if isinstance(latest_candle, list) and len(latest_candle) >= 5:
                                price = float(latest_candle[4])
                                if price > 0:
                                    logger.debug(f"✅ Price for {symbol} from WebSocket ({tf} fallback): ${price:.2f}")
                                    return price
                
                logger.debug(f"⚠️ WebSocket data unavailable or invalid for {symbol}")
                
            except Exception as e:
                logger.debug(f"⚠️ Error getting price from WebSocket for {symbol}: {e}")
        
        # STEP 2: REST API Fallback
        logger.debug(f"🔄 Falling back to REST API for {symbol} price")
        
        try:
            # Determine which exchange to use
            if not exchange and self.exchanges:
                exchange = next(iter(self.exchanges.keys()))
            
            if not exchange or exchange not in self.exchanges:
                logger.error(f"❌ No valid exchange available for REST API price fetch")
                return None
            
            client = self.exchanges[exchange]
            
            # Fetch minimal data (just 1 candle) for efficiency
            try:
                ohlcv_data = await client.ohlcv(symbol, timeframe, limit=1, add_indicators=False)
            except Exception as api_error:
                logger.warning(f"⚠️ REST API call failed for {symbol}: {api_error}")
                return None
            
            # Extract price from response
            if ohlcv_data is not None:
                if isinstance(ohlcv_data, pd.DataFrame) and not ohlcv_data.empty:
                    filtered = self._filter_closed_dataframe(ohlcv_data, timeframe, context="[REST-PRICE]")
                    if filtered is None or filtered.empty:
                        logger.warning(f"[REST-PRICE] No closed candle available after filtering for {symbol} {timeframe}")
                        return None
                    price = float(filtered['close'].iloc[-1])
                    if price > 0:
                        logger.debug(f"[REST-PRICE] Price for {symbol}: ${price:.2f}")
                        return price
                elif isinstance(ohlcv_data, list) and len(ohlcv_data) > 0:
                    # Handle raw OHLCV list format
                    df_fallback = self._ohlcv_to_dataframe(ohlcv_data)
                    df_fallback = self._filter_closed_dataframe(df_fallback, timeframe, context="[REST-PRICE]")
                    if df_fallback is not None and not df_fallback.empty:
                        price = float(df_fallback['close'].iloc[-1])
                        if price > 0:
                            logger.debug(f"[REST-PRICE] Price for {symbol}: ${price:.2f}")
                            return price
            
            logger.warning(f"⚠️ REST API returned no valid price data for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"[REST-PRICE] REST API price fetch failed for {symbol}: {e}")
            return None

    async def get_latest_price_cache_only(
        self,
        symbol: str,
        timeframe: str = "1m",
        exchange: Optional[str] = None,
    ) -> Optional[float]:
        """
        Return the latest price using cache-only sources (WebSocket collector/ticker/forming candle).
        Never falls back to REST.
        """
        if not symbol:
            return None

        ws_exchange = exchange or (next(iter(self.exchanges.keys())) if self.exchanges else self.DEFAULT_EXCHANGE)
        if isinstance(ws_exchange, str):
            ws_exchange = ws_exchange.lower()

        forming_close = None
        try:
            forming_close = self.get_realtime_price(symbol, timeframe=timeframe, exchange=ws_exchange)
        except Exception:
            forming_close = None

        price = None
        try:
            price, _source, _fallback = self.get_live_trigger_price(
                symbol,
                timeframe,
                source="mid",
                exchange=ws_exchange,
                forming_close=forming_close,
            )
        except Exception:
            price = forming_close

        try:
            if price is not None:
                price = float(price)
        except Exception:
            price = None

        if price is not None and price > 0:
            return price

        # Fallback to latest closed candle from WS collector (still cache-only).
        if self.websocket_manager:
            try:
                ws_data = self.websocket_manager.get_latest_data(symbol, timeframe, exchange=ws_exchange)
            except Exception:
                ws_data = None
            ohlcv = ws_data.get("ohlcv") if isinstance(ws_data, dict) else None
            if isinstance(ohlcv, list) and ohlcv:
                latest = ohlcv[-1]
                if isinstance(latest, list) and len(latest) >= 5:
                    try:
                        price = float(latest[4])
                    except Exception:
                        price = None
                    if price is not None and price > 0:
                        return price

        return None
    
    def get_realtime_price(self, symbol: str, timeframe: str = '1m', exchange: str = None) -> Optional[float]:
        """
        Return the current forming candle price (real-time) without affecting indicator stability.
        """
        if self.websocket_manager and getattr(self.websocket_manager, "collector", None):
            ws_exchange = exchange if exchange else (next(iter(self.exchanges.keys())) if self.exchanges else self.DEFAULT_EXCHANGE)
            if isinstance(ws_exchange, str):
                ws_exchange = ws_exchange.lower()
            forming = self.websocket_manager.collector.get_forming_ohlcv(ws_exchange, symbol, timeframe)
            if forming and isinstance(forming, list) and len(forming) >= 5:
                price = float(forming[4])
                if price > 0:
                    return price
        return None
    def _get_best_data_source(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get data from the best available exchange source.
        
        Selects exchange with most recent data and most candles.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
        
        Returns:
            DataFrame from best exchange, or None if no data available
        """
        candidates = []
        
        for exchange_name in self.data_streams:
            if symbol in self.data_streams[exchange_name]:
                df = self.data_streams[exchange_name][symbol].get(timeframe)
                if df is not None and not df.empty:
                    key = f"{exchange_name}:{symbol}:{timeframe}"
                    last_update = self.last_update_time.get(key)
                    candidates.append({
                        'exchange': exchange_name,
                        'df': df,
                        'length': len(df),
                        'last_update': last_update
                    })
        
        if not candidates:
            return None
        
        # Sort by last update (most recent first), then by length (most candles first)
        candidates.sort(key=lambda x: (x['last_update'] or datetime.min.replace(tzinfo=timezone.utc), 
                                       x['length']), 
                       reverse=True)
        
        best = candidates[0]
        logger.debug(f"Best source for {symbol} {timeframe}: {best['exchange']} "
                    f"({best['length']} candles)")
        
        return best['df']
    
    def health_check(self) -> Dict[str, Any]:
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        error_rate = (self.failed_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        health_status = 'healthy'
        if error_rate > 20: health_status = 'degraded'
        if error_rate > 50: health_status = 'critical'
        
        return {
            'status': health_status,
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'error_rate': round(error_rate, 2),
            'active_streams': 0, # Deprecated
            'is_running': self.is_running
        }
        
    def get_pipeline_status(self) -> Dict[str, Any]:
        status = self.health_check()
        status['note'] = 'Data is now stored centrally in WebSocketManager. Status reflects priming jobs only.'
        # Yerel depoya dayalı hesaplamalar kaldırıldı.
        status['exchanges'] = {}
        status['memory_estimate_mb'] = 0
        status['data_freshness'] = {}
        return status
    
    def shutdown(self):
        """
        Shutdown the pipeline gracefully.
        """
        logger.info("🔄 Shutting down MarketDataPipeline...")
        self.is_running = False
        
        # Shutdown executor
        if hasattr(self, '_executor'):
            logger.debug("Shutting down thread pool executor...")
            self._executor.shutdown(wait=True, cancel_futures=False)
        
        # Log final stats
        final_stats = self.get_pipeline_status()
        logger.info(f"✅ Pipeline shutdown complete. Final stats:")
        logger.info(f"   - Total requests: {final_stats['total_requests']}")
        logger.info(f"   - Failed requests: {final_stats['failed_requests']}")
        logger.info(f"   - Error rate: {final_stats['error_rate']}%")
        logger.info(f"   - Active streams: {final_stats['active_streams']}")
        logger.info(f"   - Memory used: {final_stats['memory_estimate_mb']} MB")
    
    async def start_feeds_async(self, symbols: List[str], timeframes: List[str] = ['30m', '1h']) -> Dict[str, Any]:
        """
        Async version of start_feeds for asynchronous operation.
        
        Args:
            symbols: List of trading symbols to fetch
            timeframes: List of timeframes to fetch
        
        Returns:
            Dict with summary of data collection results
        """
        logger.info(f"🔄 Starting async data feeds for {len(symbols)} symbols across {len(timeframes)} timeframes")
        self.is_running = True
        
        results = {
            'symbols_processed': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'exchanges_used': set(),
            'errors': []
        }
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Try to fetch from best available exchange
                success = self._fetch_and_store(symbol, timeframe, results)
                
                if success:
                    results['symbols_processed'] += 1
                
                # Rate limiting between symbol fetches
                await asyncio.sleep(0.1)
        
        results['exchanges_used'] = list(results['exchanges_used'])
        
        logger.info(f"✅ Async data feeds started: {results['successful_fetches']} successful, "
                   f"{results['failed_fetches']} failed")
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the pipeline (alias for get_pipeline_status).
        
        Returns:
            Dict with health status and metrics
        """
        status = self.get_pipeline_status()
        
        # Simplify for health check
        return {
            'overall_status': status['status'],
            'uptime_seconds': status['uptime_seconds'],
            'active_feeds': status['active_streams'],
            'error_rate': status['error_rate'],
            'memory_mb': status['memory_estimate_mb']
        }
