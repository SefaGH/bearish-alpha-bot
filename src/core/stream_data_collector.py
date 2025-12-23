"""
Stream Data Collector for WebSocket Manager.
Maintains strict separation between closed candles and the current forming candle.
"""
import asyncio
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Awaitable, Callable
from datetime import datetime, timezone
from collections import deque

from .data_validator import TIMEFRAME_SECONDS

logger = logging.getLogger(__name__)


class StreamDataCollector:
    """Helper class to collect streaming data into buffers for analysis."""
    
    def __init__(self, buffer_size: int = 1000, config: Dict[str, Any] = None):
        """
        Initialize data collector.
        
        Args:
            buffer_size: Maximum number of items to keep in each buffer.
            config: Optional configuration dictionary to override buffer size.
        """
        resolved_buffer_size = buffer_size
        if config:
            resolved_buffer_size = config.get('websocket', {}).get('buffer_size', buffer_size)
        
        ws_config = config.get('websocket', {}) if config else {}
        self.throttle_interval_ms = ws_config.get('throttle_interval_ms', 250)

        self.buffer_size = int(resolved_buffer_size)

        # Closed (finalized) candles per exchange -> symbol_tf -> deque
        self._closed_data: Dict[str, Dict[str, deque]] = {}
        # Forming (live) candles per exchange -> symbol_tf -> dict
        self._forming_data: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Telemetry
        self._last_closed_ts: Dict[str, Dict[str, Optional[int]]] = {}
        self._forming_ts: Dict[str, Dict[str, Optional[int]]] = {}
        self._gap_count: Dict[str, Dict[str, int]] = {}
        self._last_backfill_ts: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
        self._out_of_order_drops: Dict[str, Dict[str, int]] = {}
        self._backfill_count: Dict[str, Dict[str, int]] = {}
        # Optional provider for WebSocket state per exchange
        self._ws_state_provider: Optional[Callable[[str], Dict[str, Any]]] = None

        # Optional async handler provided by MarketDataPipeline for backfill
        self._backfill_handler: Optional[Callable[[str, str, str, int, int, int], Awaitable[List[List[float]]]]] = None

        # Periodic state logger
        self._state_logger_interval = 60
        self._state_logger_thread: Optional[threading.Thread] = None
        self._state_logger_stop = threading.Event()

        # Legacy alias preserved for backward compatibility (now closed-only)
        self.ohlcv_data = self._closed_data

        self.ticker_data: Dict[str, Dict[str, deque]] = {}
        self._last_update_cache: Dict[str, float] = {}
        
        logger.info(f"StreamDataCollector initialized with buffer_size={self.buffer_size} and throttle_interval={self.throttle_interval_ms}ms")

    def set_ws_state_provider(self, provider: Callable[[str], Dict[str, Any]]):
        """Register provider to supply websocket connection state per exchange."""
        self._ws_state_provider = provider
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to consistent format with settlement currency.
        
        This ensures symbols are always in the format 'BASE/QUOTE:SETTLE' for futures.
        For example: 'BTC/USDT' -> 'BTC/USDT:USDT', 'BTC/USDT:USDT' -> 'BTC/USDT:USDT'
        
        Args:
            symbol: Trading symbol in any format
            
        Returns:
            Normalized symbol with settlement currency
        """
        if not symbol:
            return symbol
            
        # If already has settlement currency, return as-is
        if ':' in symbol:
            return symbol
            
        # Add USDT settlement for USDT pairs (futures/perpetuals)
        if symbol.endswith('/USDT'):
            return f"{symbol}:USDT"
            
        # For other pairs, return as-is
        return symbol
    
    def _get_buffer_key(self, symbol: str, timeframe: str) -> str:
        """
        Generate consistent buffer key for symbol and timeframe.
        
        This ensures both prime_buffer_with_dataframe and get_latest_ohlcv
        use the same key format to access the same data.
        
        Note: Automatically normalizes symbols to ensure consistent format with
        settlement currency (e.g., 'BTC/USDT' becomes 'BTC/USDT:USDT').
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT' or 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '1h')
        
        Returns:
            Buffer key in format 'normalized_symbol_timeframe'
        """
        normalized_symbol = self._normalize_symbol(symbol)
        return f"{normalized_symbol}_{timeframe}"

    def _get_interval_ms(self, timeframe: str) -> Optional[int]:
        """Return timeframe interval in milliseconds."""
        if timeframe not in TIMEFRAME_SECONDS:
            return None
        return TIMEFRAME_SECONDS[timeframe] * 1000

    def _ensure_structs(self, exchange: str, key: str):
        """Ensure internal dicts exist for exchange/key."""
        self._closed_data.setdefault(exchange, {})
        self._forming_data.setdefault(exchange, {})
        self._last_closed_ts.setdefault(exchange, {})
        self._forming_ts.setdefault(exchange, {})
        self._gap_count.setdefault(exchange, {})
        self._last_backfill_ts.setdefault(exchange, {})
        self._out_of_order_drops.setdefault(exchange, {})
        self._backfill_count.setdefault(exchange, {})
        if key not in self._closed_data[exchange]:
            self._closed_data[exchange][key] = deque(maxlen=self.buffer_size)
        if key not in self._gap_count[exchange]:
            self._gap_count[exchange][key] = 0
        if key not in self._forming_ts[exchange]:
            self._forming_ts[exchange][key] = None
        if key not in self._last_closed_ts[exchange]:
            self._last_closed_ts[exchange][key] = None
        if key not in self._last_backfill_ts[exchange]:
            self._last_backfill_ts[exchange][key] = 0.0
        if key not in self._out_of_order_drops[exchange]:
            self._out_of_order_drops[exchange][key] = 0
        if key not in self._backfill_count[exchange]:
            self._backfill_count[exchange][key] = 0

    def _set_forming(self, exchange: str, key: str, candle: List[float]):
        """Initialize or replace the forming candle."""
        self._ensure_structs(exchange, key)
        self._forming_data[exchange][key] = {
            "open_time": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        }
        self._forming_ts[exchange][key] = int(candle[0])

    def _commit_forming(self, exchange: str, symbol: str, timeframe: str, key: str, next_open_time: int, interval_ms: Optional[int]):
        """Commit current forming candle into closed buffer and perform gap checks/backfill."""
        forming = self._forming_data.get(exchange, {}).get(key)
        if not forming:
            return

        expected_next = None
        gap_bars = 0
        if interval_ms:
            expected_next = forming["open_time"] + interval_ms
            if next_open_time > expected_next:
                gap_bars = (next_open_time - expected_next) // interval_ms
                if gap_bars > 0:
                    self._gap_count[exchange][key] += gap_bars
                    logger.warning(
                        f"[WS] GAP_DETECTED exchange={exchange} key={key} missed={gap_bars} "
                        f"expected_next={expected_next} incoming_ot={next_open_time}"
                    )
                    self._maybe_backfill(exchange, symbol, timeframe, key, expected_next, next_open_time, interval_ms, gap_bars)

        committed = [
            forming["open_time"],
            forming["open"],
            forming["high"],
            forming["low"],
            forming["close"],
            forming["volume"],
        ]

        with self._lock:
            closed_buffer = self._closed_data.setdefault(exchange, {}).setdefault(key, deque(maxlen=self.buffer_size))
            # Dedupe/overwrite if same open_time already stored
            if closed_buffer and closed_buffer[-1][0] == forming["open_time"]:
                closed_buffer[-1] = committed
            else:
                closed_buffer.append(committed)

            self._last_closed_ts[exchange][key] = forming["open_time"]
            logger.debug(
                f"[WS] COMMIT exchange={exchange} key={key} closed_len={len(closed_buffer)} "
                f"last_closed_ts={self._last_closed_ts[exchange][key]} gap_count={self._gap_count[exchange][key]}"
            )
            if gap_bars > 0 and expected_next is not None:
                self._log_event(
                    {
                        "event": "gap_detected",
                        "ts": self._iso_now(),
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "key": key,
                        "expected_next": expected_next,
                        "incoming_ot": next_open_time,
                        "interval_ms": interval_ms,
                        "missed_bars": gap_bars,
                    }
                )

    def _process_incoming_candle(self, exchange: str, symbol: str, timeframe: str, key: str, interval_ms: Optional[int], candle: List[float]):
        """Process each incoming candle applying commit and gap detection rules."""
        if len(candle) < 6:
            logger.debug(f"[WS] Ignoring malformed candle for {exchange} {key}: {candle}")
            return

        open_time = int(candle[0])
        self._ensure_structs(exchange, key)
        current_forming = self._forming_data[exchange].get(key)

        # First candle for this stream
        if not current_forming:
            self._set_forming(exchange, key, candle)
            logger.debug(f"[WS] Set initial forming candle for {exchange} {key}: ot={open_time}")
            return

        current_open = current_forming["open_time"]

        # Out-of-order older than current forming
        if open_time < current_open:
            self._out_of_order_drops[exchange][key] += 1
            logger.warning(f"[WS] OUT_OF_ORDER for {exchange} {key}: incoming_ot={open_time} < forming_ot={current_open}")
            return

        # Same candle (forming update)
        if open_time == current_open:
            updated = {
                "open_time": current_open,
                "open": current_forming["open"],
                "high": max(current_forming["high"], float(candle[2])),
                "low": min(current_forming["low"], float(candle[3])),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            }
            self._forming_data[exchange][key] = updated
            self._forming_ts[exchange][key] = current_open
            return

        # Pivot to next candle -> commit previous forming
        self._commit_forming(exchange, symbol, timeframe, key, open_time, interval_ms)
        self._set_forming(exchange, key, candle)

    async def ohlcv_callback(self, exchange: str, symbol: str, timeframe: str, ohlcv: List):
        """
        Callback to collect OHLCV data and maintain strict closed vs forming separation.
        
        The incoming payload can be a single candle list or a list of candle lists.
        """
        buffer_key = self._get_buffer_key(symbol, timeframe)
        interval_ms = self._get_interval_ms(timeframe)

        # Normalize incoming payload
        candles: List[List[float]] = []
        if ohlcv and isinstance(ohlcv[0], (int, float)):
            candles = [ohlcv]  # single candle
        elif ohlcv and isinstance(ohlcv[0], list):
            candles = ohlcv  # already a list of candles

        if not candles:
            logger.debug(f"[WS] Empty/invalid kline payload for {exchange} {buffer_key}, skipping.")
            return

        for candle in candles:
            self._process_incoming_candle(exchange, symbol, timeframe, buffer_key, interval_ms, candle)

    async def ticker_callback(self, exchange: str, symbol: str, ticker: Dict):
        """Callback to collect ticker data."""
        if exchange not in self.ticker_data:
            self.ticker_data[exchange] = {}
        
        if symbol not in self.ticker_data[exchange]:
            self.ticker_data[exchange][symbol] = deque(maxlen=self.buffer_size)
        
        self.ticker_data[exchange][symbol].append({
            'timestamp': datetime.now(timezone.utc),
            'data': ticker
        })
        
        logger.debug(f"Collected ticker: {exchange} {symbol} (buffer: {len(self.ticker_data[exchange][symbol])})")
    
    def get_latest_ohlcv(self, exchange: str, symbol: str, timeframe: str, limit: Optional[int] = None) -> Optional[List[List]]:
        """
        Get latest CLOSED OHLCV data for a symbol as a list of lists.
        
        Args:
            exchange: Exchange name (e.g., 'bingx')
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '1h')
            limit: Maximum number of candles to return (returns last N candles)
        
        Returns:
            List of OHLCV candles in format [[timestamp, o, h, l, c, v], ...] or None
        """
        key = self._get_buffer_key(symbol, timeframe)
        
        logger.debug(f"[READ] Attempting to read CLOSED buffer: exchange={exchange}, key={key}, limit={limit}")
        
        if exchange not in self._closed_data:
            logger.debug(f"[READ] Exchange '{exchange}' not found in closed_data. Available exchanges: {list(self._closed_data.keys())}")
            return None
            
        if key not in self._closed_data[exchange]:
            logger.debug(f"[READ] Key '{key}' not found for exchange '{exchange}'. Available keys: {list(self._closed_data[exchange].keys())}")
            return None
            
        buffer = self._closed_data[exchange][key]
        if not buffer:
            logger.debug(f"[READ] Buffer exists but is empty for {exchange} {key}")
            return None
        
        all_candles = list(buffer)
        logger.debug(f"[READ] Found {len(all_candles)} CLOSED candles in buffer for {exchange} {key}")
        
        if limit is None:
            return all_candles
        else:
            return all_candles[-limit:]

    def get_forming_ohlcv(self, exchange: str, symbol: str, timeframe: str) -> Optional[List[float]]:
        """Return the current forming candle for a symbol/timeframe, if any."""
        key = self._get_buffer_key(symbol, timeframe)
        forming = self._forming_data.get(exchange, {}).get(key)
        if not forming:
            return None
        return [
            forming["open_time"],
            forming["open"],
            forming["high"],
            forming["low"],
            forming["close"],
            forming["volume"],
        ]

    def get_latest_ticker(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get the latest ticker data for a symbol."""
        if exchange in self.ticker_data and symbol in self.ticker_data[exchange]:
            buffer = self.ticker_data[exchange][symbol]
            return buffer[-1]['data'] if buffer else None
        return None
    
    def clear(self):
        """Clear all collected data."""
        self._closed_data.clear()
        self._forming_data.clear()
        self._last_closed_ts.clear()
        self._forming_ts.clear()
        self._gap_count.clear()
        self._last_backfill_ts.clear()
        self._out_of_order_drops.clear()
        self._backfill_count.clear()
        self.ticker_data.clear()
        logger.info("StreamDataCollector cleared")
    
    def prime_buffer_with_dataframe(self, exchange: str, symbol: str, timeframe: str, df):
        """
        Prime the buffer with historical data from a DataFrame.
        
        This method converts a pandas DataFrame to OHLCV list format and stores it
        in the CLOSED data buffer so that downstream consumers read only finalized bars.
        
        CRITICAL: This method uses the same key format as get_latest_ohlcv to ensure
        data written here can be read back correctly.
        
        Args:
            exchange: Exchange name (e.g., 'bingx')
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '1h')
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        """
        import pandas as pd

        key = self._get_buffer_key(symbol, timeframe)
        
        try:
            if df is None or df.empty:
                logger.warning(f"[PRIME] Empty DataFrame for {exchange} {symbol} {timeframe}, skipping.")
                return

            # Ensure exchange exists in data structure
            self._ensure_structs(exchange, key)

            # Ensure uniqueness by open_time and order
            df = df[~df.index.duplicated(keep='last')].sort_index()
            
            # Convert DataFrame to OHLCV list format
            ohlcv_list = []
            for timestamp, row in df.iterrows():
                timestamp_ms = int(pd.Timestamp(timestamp).timestamp() * 1000)
                ohlcv_list.append([
                    timestamp_ms,
                    float(row['open']), 
                    float(row['high']), 
                    float(row['low']), 
                    float(row['close']), 
                    float(row['volume'])
                ])
            
            # Create a new deque with the primed CLOSED data
            self._closed_data[exchange][key] = deque(ohlcv_list, maxlen=self.buffer_size)
            self._last_closed_ts[exchange][key] = ohlcv_list[-1][0]
            self._forming_ts[exchange][key] = None
            # Do not set gap_count here; it is accumulated during runtime
            
            logger.info(f"[PRIME] Primed CLOSED buffer with {len(ohlcv_list)} candles for {exchange} {key}. Buffer size: {len(self._closed_data[exchange][key])}")
            logger.debug(f"[PRIME] Buffer stored at: self._closed_data['{exchange}']['{key}']")

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"[PRIME] Failed to prime buffer for {exchange} {key}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"[PRIME] Unexpected error priming buffer for {exchange} {key}: {e}", exc_info=True)
            raise  # Re-raise unexpected errors

    def get_state(self, exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Return telemetry for a symbol/timeframe."""
        key = self._get_buffer_key(symbol, timeframe)
        return {
            "last_closed_ts": self._last_closed_ts.get(exchange, {}).get(key),
            "forming_ts": self._forming_ts.get(exchange, {}).get(key),
            "gap_count": self._gap_count.get(exchange, {}).get(key, 0),
            "closed_len": len(self._closed_data.get(exchange, {}).get(key, [])),
            "backfill_count": self._backfill_count.get(exchange, {}).get(key, 0),
            "last_backfill_ts": self._last_backfill_ts.get(exchange, {}).get(key),
            "cooldown_remaining_s": self._cooldown_remaining(exchange, key),
            "out_of_order_drops": self._out_of_order_drops.get(exchange, {}).get(key, 0),
        }

    def _can_backfill(self, exchange: str, key: str) -> bool:
        last_ts = self._last_backfill_ts.get(exchange, {}).get(key, 0.0)
        return (time.time() - last_ts) >= 60.0

    def _record_backfill(self, exchange: str, key: str):
        self._last_backfill_ts.setdefault(exchange, {})[key] = time.time()

    def _insert_closed_candles(self, exchange: str, key: str, candles: List[List[float]]):
        """Insert missing closed candles with dedupe/order under lock."""
        if not candles:
            return
        with self._lock:
            buffer = self._closed_data.setdefault(exchange, {}).setdefault(key, deque(maxlen=self.buffer_size))
            existing = {c[0] for c in buffer}
            to_add = [c for c in sorted(candles, key=lambda x: x[0]) if c[0] not in existing]
            for c in to_add:
                buffer.append(c)
                self._last_closed_ts[exchange][key] = c[0]
            if to_add:
                ex_backfills = self._backfill_count.setdefault(exchange, {})
                ex_backfills[key] = ex_backfills.get(key, 0) + len(to_add)

    def _maybe_backfill(self, exchange: str, symbol: str, timeframe: str, key: str, expected_next: int, incoming_ot: int, interval_ms: int, gap_bars: int):
        """Guarded auto-reconciliation for small gaps."""
        if gap_bars <= 0:
            return
        if gap_bars > 5:
            self._log_event(
                {
                    "event": "backfill_result",
                    "ts": self._iso_now(),
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "key": key,
                    "action": "skipped_too_large",
                    "reason": "missed_bars_gt_5",
                    "missed_bars": gap_bars,
                    "range_start": expected_next,
                    "range_end": incoming_ot - interval_ms,
                    "inserted": 0,
                    "expected": gap_bars,
                    "error": None,
                }
            )
            return
        if not self._can_backfill(exchange, key):
            self._log_event(
                {
                    "event": "backfill_result",
                    "ts": self._iso_now(),
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "key": key,
                    "action": "skipped_cooldown",
                    "reason": "cooldown",
                    "missed_bars": gap_bars,
                    "range_start": expected_next,
                    "range_end": incoming_ot - interval_ms,
                    "inserted": 0,
                    "expected": gap_bars,
                    "error": None,
                }
            )
            return
        if not self._backfill_handler:
            self._log_event(
                {
                    "event": "backfill_result",
                    "ts": self._iso_now(),
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "key": key,
                    "action": "skipped_no_handler",
                    "reason": "no_handler",
                    "missed_bars": gap_bars,
                    "range_start": expected_next,
                    "range_end": incoming_ot - interval_ms,
                    "inserted": 0,
                    "expected": gap_bars,
                    "error": None,
                }
            )
            return

        start_ts = expected_next
        end_ts = incoming_ot - interval_ms
        if end_ts < start_ts:
            return

        self._record_backfill(exchange, key)
        async def _run():
            try:
                candles = await self._backfill_handler(exchange, symbol, timeframe, start_ts, end_ts, gap_bars)
                inserted = 0
                if candles:
                    inserted = len(candles)
                    self._insert_closed_candles(exchange, key, candles)
                self._log_event(
                    {
                        "event": "backfill_result",
                        "ts": self._iso_now(),
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "key": key,
                        "action": "inserted",
                        "missed_bars": gap_bars,
                        "range_start": start_ts,
                        "range_end": end_ts,
                        "inserted": inserted,
                        "expected": gap_bars,
                        "reason": None,
                        "error": None,
                    }
                )
            except Exception as e:
                self._log_event(
                    {
                        "event": "backfill_result",
                        "ts": self._iso_now(),
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "key": key,
                        "action": "failed",
                        "missed_bars": gap_bars,
                        "range_start": start_ts,
                        "range_end": end_ts,
                        "inserted": 0,
                        "expected": gap_bars,
                        "reason": None,
                        "error": str(e),
                    }
                )

        asyncio.get_event_loop().create_task(_run())

    def _iso_now(self) -> str:
        return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    def _cooldown_remaining(self, exchange: str, key: str) -> int:
        last_ts = self._last_backfill_ts.get(exchange, {}).get(key, 0.0)
        remaining = int(max(0.0, 60.0 - (time.time() - last_ts)))
        return remaining

    def _log_event(self, payload: Dict[str, Any]):
        try:
            logger.info(json.dumps(payload, separators=(",", ":")))
        except Exception:
            logger.debug(f"[LOG-EVENT] Failed to serialize payload: {payload}")

    def emit_state_snapshots(self, connection_meta: Optional[Dict[str, Any]] = None):
        """
        Emit collector_state for all exchange/key pairs.
        connection_meta: optional dict keyed by exchange with info dict {connected, listen, subs, ws_messages}
        """
        conn_meta = connection_meta or {}
        now_iso = self._iso_now()
        with self._lock:
            for exchange, keys in self._closed_data.items():
                meta = conn_meta.get(exchange, {})
                if self._ws_state_provider:
                    try:
                        extra = self._ws_state_provider(exchange) or {}
                        meta = {**meta, **extra}
                    except Exception as e:
                        logger.debug(f"[STATE-LOGGER] ws_state_provider error for {exchange}: {e}")
                for key in keys.keys():
                    symbol, timeframe = key.rsplit("_", 1)
                    state = self.get_state(exchange, symbol, timeframe)
                    payload = {
                        "event": "collector_state",
                        "ts": now_iso,
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "key": key,
                        "connected": meta.get("connected"),
                        "listen": meta.get("listen"),
                        "subs": meta.get("subs"),
                        "ws_messages": meta.get("ws_messages"),
                        "closed_len": state["closed_len"],
                        "last_closed_ot": state["last_closed_ts"],
                        "forming_ot": state["forming_ts"],
                        "gap_count": state["gap_count"],
                        "out_of_order_drops": state["out_of_order_drops"],
                        "backfill_count": state["backfill_count"],
                        "last_backfill_ts": state["last_backfill_ts"],
                        "backfill_cooldown_remaining_s": state["cooldown_remaining_s"],
                        "data_source": "ws",
                    }
                    self._log_event(payload)

    def start_state_logger(self, interval: int = 60):
        """Start periodic collector_state emission."""
        self._state_logger_interval = interval
        if self._state_logger_thread and self._state_logger_thread.is_alive():
            return

        def _run():
            while not self._state_logger_stop.is_set():
                try:
                    self.emit_state_snapshots()
                except Exception as e:
                    logger.debug(f"[STATE-LOGGER] error emitting state: {e}")
                self._state_logger_stop.wait(self._state_logger_interval)

        self._state_logger_stop.clear()
        self._state_logger_thread = threading.Thread(target=_run, daemon=True)
        self._state_logger_thread.start()

    def stop_state_logger(self):
        """Stop periodic collector_state emission."""
        if self._state_logger_thread and self._state_logger_thread.is_alive():
            self._state_logger_stop.set()
            self._state_logger_thread.join(timeout=1.0)
    def set_backfill_handler(self, handler: Callable[[str, str, str, int, int, int], Awaitable[List[List[float]]]]):
        """Register async backfill handler: (exchange, symbol, timeframe, start_ts, end_ts, missed_bars) -> list of candles."""
        self._backfill_handler = handler
