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
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from .ccxt_client import CcxtClient
from .indicators import add_indicators

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
        
        # Data storage: {exchange: {symbol: {timeframe: DataFrame}}}
        self.data_streams = defaultdict(lambda: defaultdict(dict))
        
        # Health monitoring
        self.start_time = datetime.now(timezone.utc)
        self.total_requests = 0
        self.failed_requests = 0
        self.last_update_time = {}
        
        # Pipeline state
        self.is_running = False
        
        logger.info(f"🔄 MarketDataPipeline initialized with {len(exchanges)} exchanges: {list(exchanges.keys())}")
    
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
        
        # CRITICAL: Wait for WebSocket collector to be ready before priming
        if not await self._wait_for_websocket_ready(timeout=10.0):
            logger.warning("[PRIME] WebSocket collector not ready after 10s timeout - proceeding without WebSocket injection")
        
        tasks = []
        # We need enough data for indicators like EMA(200)
        limit = self.config.get('indicators', {}).get('ema_slow', 200) + 50  # 250 bars

        for symbol in symbols:
            for timeframe in timeframes:
                # Assuming the first available exchange is the primary one for fetching.
                # A more complex logic could try multiple exchanges.
                exchange_name = next(iter(self.exchanges.keys()), None)
                if not exchange_name:
                    logger.error("[PRIME] No exchanges available to prime data.")
                    continue
                
                client = self.exchanges[exchange_name]
                tasks.append(self._fetch_and_store_async(client, exchange_name, symbol, timeframe, limit))

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
            
            # Fetch OHLCV data
            ohlcv_data = await asyncio.to_thread(client.ohlcv, symbol, timeframe, limit)
            
            if not ohlcv_data:
                logger.warning(f"⚠️ [PRIME] Empty data for {symbol} {timeframe} from {exchange_name}")
                self.failed_requests += 1
                return False

            # Convert to DataFrame
            df = self._ohlcv_to_dataframe(ohlcv_data)
            
            # Add indicators
            df = add_indicators(df, self.config.get('indicators'))
        
            logger.info(f"✅ [PRIME] Loaded {len(df)} historical candles for {exchange_name} {symbol} {timeframe}")
            
            # YENİ BLOK: Çekilen veriyi WebSocketManager'ın önbelleğine aktar.
            # DEFENSIVE: Check both websocket_manager AND collector existence
            if not self.websocket_manager:
                logger.debug(f"[INJECT] No WebSocket manager - skipping data injection for {symbol} {timeframe}")
                return True
            
            if not hasattr(self.websocket_manager, 'collector') or not self.websocket_manager.collector:
                logger.warning(f"⚠️ [INJECT] WebSocket manager exists but collector not found. Skipping data injection for {symbol} {timeframe}")
                return True
            
            try:
                # CCXT sembol formatını ('BTC/USDT') WebSocket formatına ('BTC/USDT:USDT') çevir.
                ws_symbol = f"{symbol}:{symbol.split('/')[-1]}" if ':' not in symbol and symbol.endswith('/USDT') else symbol
                
                # Collector'a DataFrame'i doğrudan gönder.
                self.websocket_manager.collector.prime_buffer_with_dataframe(exchange_name, ws_symbol, timeframe, df)
                logger.info(f"✅ [INJECT] Injected {len(df)} candles into WebSocket buffer for {ws_symbol} {timeframe}")
            except Exception as e:
                logger.error(f"❌ [INJECT] Failed to inject data into WebSocket buffer for {symbol} {timeframe}: {e}")
                # Don't fail the entire prime operation if injection fails
                pass
    
            return True
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"❌ [PRIME] Failed to fetch {symbol} {timeframe} on {exchange_name}: {e}")
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
                        ohlcv_data = client.fetch_ohlcv_bulk(validated_symbol, timeframe, limit)
                    else:
                        ohlcv_data = client.ohlcv(validated_symbol, timeframe, limit)
                    
                    if not ohlcv_data:
                        logger.warning(f"⚠️ Empty data for {symbol} {timeframe} from {exchange_name}")
                        self.failed_requests += 1
                        break
                    
                    # Convert to DataFrame
                    df = self._ohlcv_to_dataframe(ohlcv_data)
                    
                    # Add indicators
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
    
    def get_latest_ohlcv(self, symbol: str, timeframe: str, exchange: str = None) -> Optional[pd.DataFrame]:
        """
        Get latest OHLCV data by reading from the central WebSocketManager buffer.
        This now acts as a proxy to the WebSocketManager, ensuring a single source of truth.
        """
        # DÜZELTİLDİ: Artık veriyi doğrudan merkezi depodan, yani WebSocketManager'dan okuyor.
        if not self.websocket_manager:
            logger.warning("Cannot get OHLCV data: WebSocketManager is not available.")
            return None
        
        try:
            # ProductionCoordinator'daki _ws_fetch_df_with_fallback metodundan ilham alındı.
            # WebSocketManager'da bu metodun olduğunu varsayıyoruz.
            # Olası metod adları: get_latest_dataframe veya get_latest_data
            df = None
            if hasattr(self.websocket_manager, 'get_latest_dataframe'):
                 df = self.websocket_manager.get_latest_dataframe(symbol, timeframe, exchange)
            elif hasattr(self.websocket_manager, 'get_latest_data'):
                 # get_latest_data bir dict döndürüyorsa, onu df'e çevirmemiz gerekir.
                 raw_data = self.websocket_manager.get_latest_data(symbol, timeframe, exchange)
                 if raw_data: # None değilse
                     if isinstance(raw_data, pd.DataFrame):
                         df = raw_data
                     elif 'ohlcv' in raw_data and raw_data['ohlcv']:
                         df = self._ohlcv_to_dataframe(raw_data['ohlcv'])
            else:
                logger.error("WebSocketManager has no standard method to get dataframe or data ('get_latest_dataframe' or 'get_latest_data').")
                return None
            
            # Add indicators to the DataFrame before returning
            if df is not None and not df.empty:
                df = add_indicators(df, self.config.get('indicators'))
            
            return df
        except Exception as e:
            logger.error(f"Error getting latest OHLCV from WebSocketManager for {symbol} {timeframe}: {e}")
            return None
            
        # Get from best available source
        return self._get_best_data_source(symbol, timeframe)
    
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
