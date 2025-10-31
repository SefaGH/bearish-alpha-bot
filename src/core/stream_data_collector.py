"""
Stream Data Collector for WebSocket Manager.
This helper class collects streaming data into buffers for analysis.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import deque

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
        self.ohlcv_data: Dict[str, Dict[str, 'deque']] = {}
        self.ticker_data: Dict[str, Dict[str, 'deque']] = {}
        self._last_update_cache: Dict[str, float] = {}
        
        logger.info(f"StreamDataCollector initialized with buffer_size={self.buffer_size} and throttle_interval={self.throttle_interval_ms}ms")
    
    def _get_buffer_key(self, symbol: str, timeframe: str) -> str:
        """
        Generate consistent buffer key for symbol and timeframe.
        
        This ensures both prime_buffer_with_dataframe and get_latest_ohlcv
        use the same key format to access the same data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '1h')
        
        Returns:
            Buffer key in format 'symbol_timeframe'
        """
        return f"{symbol}_{timeframe}"
    
    async def ohlcv_callback(self, exchange: str, symbol: str, timeframe: str, ohlcv: List):
        """Callback to collect OHLCV data with throttling/debouncing."""
        throttle_key = f"{exchange}_{symbol}_{timeframe}"
        now = time.time()
        
        last_update_time = self._last_update_cache.get(throttle_key, 0)
        buffer_key = self._get_buffer_key(symbol, timeframe)
        
        if (now - last_update_time) * 1000 < self.throttle_interval_ms:
            if exchange in self.ohlcv_data and buffer_key in self.ohlcv_data[exchange]:
                buffer = self.ohlcv_data[exchange][buffer_key]
                if buffer and ohlcv and isinstance(ohlcv[0], (int, float)):
                    buffer[-1] = ohlcv
                    return
                    
        self._last_update_cache[throttle_key] = now

        if exchange not in self.ohlcv_data:
            self.ohlcv_data[exchange] = {}
        
        if buffer_key not in self.ohlcv_data[exchange]:
            self.ohlcv_data[exchange][buffer_key] = deque(maxlen=self.buffer_size)
        
        if ohlcv and isinstance(ohlcv[0], (int, float)):
             self.ohlcv_data[exchange][buffer_key].append(ohlcv)
        elif ohlcv and isinstance(ohlcv[0], list):
             for candle in ohlcv:
                 self.ohlcv_data[exchange][buffer_key].append(candle)

        logger.debug(f"Collected OHLCV: {exchange} {buffer_key} (buffer: {len(self.ohlcv_data[exchange][buffer_key])})")
    
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
        Get latest OHLCV data for a symbol as a list of lists.
        
        Args:
            exchange: Exchange name (e.g., 'bingx')
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '1h')
            limit: Maximum number of candles to return (returns last N candles)
        
        Returns:
            List of OHLCV candles in format [[timestamp, o, h, l, c, v], ...] or None
        """
        key = self._get_buffer_key(symbol, timeframe)
        
        # Debug logging to help diagnose data access issues
        logger.debug(f"[READ] Attempting to read from buffer: exchange={exchange}, key={key}, limit={limit}")
        
        if exchange not in self.ohlcv_data:
            logger.debug(f"[READ] Exchange '{exchange}' not found in ohlcv_data. Available exchanges: {list(self.ohlcv_data.keys())}")
            return None
            
        if key not in self.ohlcv_data[exchange]:
            logger.debug(f"[READ] Key '{key}' not found for exchange '{exchange}'. Available keys: {list(self.ohlcv_data[exchange].keys())}")
            return None
            
        buffer = self.ohlcv_data[exchange][key]
        if not buffer:
            logger.debug(f"[READ] Buffer exists but is empty for {exchange} {key}")
            return None
        
        all_candles = list(buffer)
        logger.debug(f"[READ] ✓ Found {len(all_candles)} candles in buffer for {exchange} {key}")
        
        if limit is None:
            return all_candles
        else:
            return all_candles[-limit:]
    
    def get_latest_ticker(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get the latest ticker data for a symbol."""
        if exchange in self.ticker_data and symbol in self.ticker_data[exchange]:
            buffer = self.ticker_data[exchange][symbol]
            return buffer[-1]['data'] if buffer else None
        return None
    
    def clear(self):
        """Clear all collected data."""
        self.ohlcv_data.clear()
        self.ticker_data.clear()
        logger.info("StreamDataCollector cleared")
    
    def prime_buffer_with_dataframe(self, exchange: str, symbol: str, timeframe: str, df):
        """
        Prime the buffer with historical data from a DataFrame.
        
        This method converts a pandas DataFrame to OHLCV list format and stores it
        in the main data buffer (self.ohlcv_data) so that IndicatorValidator and
        other components can access the primed data.
        
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
            if exchange not in self.ohlcv_data:
                self.ohlcv_data[exchange] = {}
                logger.debug(f"[PRIME] Created new exchange entry for '{exchange}'")
            
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
            
            # Create a new deque with the primed data
            # This ensures atomic replacement of the buffer and avoids race conditions
            self.ohlcv_data[exchange][key] = deque(ohlcv_list, maxlen=self.buffer_size)
            
            logger.info(f"[PRIME] ✅ Primed buffer with {len(ohlcv_list)} candles for {exchange} {key}. Buffer size: {len(self.ohlcv_data[exchange][key])}")
            logger.debug(f"[PRIME] Buffer stored at: self.ohlcv_data['{exchange}']['{key}']")

        except (ValueError, TypeError, KeyError) as e:
            # Handle expected data conversion and access errors
            logger.error(f"[PRIME] ❌ Failed to prime buffer for {exchange} {key}: {e}", exc_info=True)
        except Exception as e:
            # Catch any other unexpected errors but log with full stack trace
            logger.error(f"[PRIME] ❌ Unexpected error priming buffer for {exchange} {key}: {e}", exc_info=True)
            raise  # Re-raise unexpected errors
