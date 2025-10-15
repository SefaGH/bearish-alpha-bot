"""
Market Data Pipeline Core Foundation for Bearish Alpha Bot.

Provides multi-exchange data collection, storage, and health monitoring
for Phase 2.2 WebSocket integration foundation.
"""

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
    
    def __init__(self, exchanges: Dict[str, CcxtClient], config: Dict[str, Any] = None):
        """
        Initialize MarketDataPipeline.
        
        Args:
            exchanges: Dictionary mapping exchange names to CcxtClient instances
            config: Optional configuration dict for pipeline settings
        """
        self.exchanges = exchanges
        self.config = config or {}
        
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
                    
                    # Store data
                    self._store_data(exchange_name, symbol, timeframe, df)
                    
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
        Store data with circular buffer management.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe string
            df: DataFrame to store
        """
        # Apply buffer limit
        limit = self.BUFFER_LIMITS.get(timeframe, 500)
        if len(df) > limit:
            df = df.tail(limit)
        
        # Store data
        self.data_streams[exchange][symbol][timeframe] = df
        
        # Update last update time
        key = f"{exchange}:{symbol}:{timeframe}"
        self.last_update_time[key] = datetime.now(timezone.utc)
    
    def get_latest_ohlcv(self, symbol: str, timeframe: str, exchange: str = None) -> Optional[pd.DataFrame]:
        """
        Get latest OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            exchange: Optional specific exchange name, otherwise uses best source
        
        Returns:
            DataFrame with OHLCV data and indicators, or None if not available
        """
        if exchange:
            # Get from specific exchange
            if exchange in self.data_streams:
                if symbol in self.data_streams[exchange]:
                    return self.data_streams[exchange][symbol].get(timeframe)
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
        """
        Perform health check on the pipeline.
        
        Returns:
            Dict with health status and metrics
        """
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        error_rate = (self.failed_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        # Count data streams
        total_streams = 0
        for exchange in self.data_streams:
            for symbol in self.data_streams[exchange]:
                total_streams += len(self.data_streams[exchange][symbol])
        
        health_status = 'healthy'
        if error_rate > 20:
            health_status = 'degraded'
        if error_rate > 50:
            health_status = 'critical'
        
        return {
            'status': health_status,
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'error_rate': round(error_rate, 2),
            'active_streams': total_streams,
            'is_running': self.is_running
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get detailed pipeline status including exchange-level breakdown.
        
        Returns:
            Dict with comprehensive status information
        """
        status = self.health_check()
        
        # Exchange-level breakdown
        exchange_stats = {}
        for exchange_name in self.data_streams:
            symbols_count = len(self.data_streams[exchange_name])
            streams_count = sum(len(self.data_streams[exchange_name][symbol]) 
                              for symbol in self.data_streams[exchange_name])
            
            exchange_stats[exchange_name] = {
                'symbols': symbols_count,
                'streams': streams_count
            }
        
        # Calculate memory estimation
        total_rows = 0
        for exchange in self.data_streams:
            for symbol in self.data_streams[exchange]:
                for timeframe, df in self.data_streams[exchange][symbol].items():
                    if df is not None:
                        total_rows += len(df)
        
        # Rough memory estimation (assuming ~200 bytes per row with indicators)
        memory_mb = (total_rows * 200) / (1024 * 1024)
        
        # Data freshness
        freshness = {}
        now = datetime.now(timezone.utc)
        for key, last_update in self.last_update_time.items():
            age_seconds = (now - last_update).total_seconds()
            if age_seconds < 300:  # 5 minutes
                freshness[key] = 'fresh'
            elif age_seconds < 3600:  # 1 hour
                freshness[key] = 'stale'
            else:
                freshness[key] = 'expired'
        
        fresh_count = sum(1 for v in freshness.values() if v == 'fresh')
        stale_count = sum(1 for v in freshness.values() if v == 'stale')
        expired_count = sum(1 for v in freshness.values() if v == 'expired')
        
        status.update({
            'exchanges': exchange_stats,
            'memory_estimate_mb': round(memory_mb, 2),
            'data_freshness': {
                'fresh': fresh_count,
                'stale': stale_count,
                'expired': expired_count
            },
            'buffer_limits': self.BUFFER_LIMITS
        })
        
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
