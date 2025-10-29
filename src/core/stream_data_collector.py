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
    
    async def ohlcv_callback(self, exchange: str, symbol: str, timeframe: str, ohlcv: List):
        """Callback to collect OHLCV data with throttling/debouncing."""
        key = f"{exchange}_{symbol}_{timeframe}"
        now = time.time()
        
        last_update_time = self._last_update_cache.get(key, 0)
        if (now - last_update_time) * 1000 < self.throttle_interval_ms:
            if exchange in self.ohlcv_data and f"{symbol}_{timeframe}" in self.ohlcv_data[exchange]:
                buffer = self.ohlcv_data[exchange][f"{symbol}_{timeframe}"]
                if buffer and ohlcv and isinstance(ohlcv[0], (int, float)):
                    buffer[-1] = ohlcv
                    return
                    
        self._last_update_cache[key] = now

        if exchange not in self.ohlcv_data:
            self.ohlcv_data[exchange] = {}
        
        buffer_key = f"{symbol}_{timeframe}"
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
        """Get latest OHLCV data for a symbol as a list of lists."""
        key = f"{symbol}_{timeframe}"
        if exchange in self.ohlcv_data and key in self.ohlcv_data[exchange]:
            buffer = self.ohlcv_data[exchange][key]
            if not buffer:
                return None
            
            all_candles = list(buffer)
            
            if limit is None:
                return all_candles
            else:
                return all_candles[-limit:]
        return None
    
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
        """Prime the buffer with historical data from a DataFrame."""
        import pandas as pd

        try:
            if df is None or df.empty:
                logger.warning(f"[PRIME] Empty DataFrame for {exchange} {symbol} {timeframe}, skipping.")
                return

            if exchange not in self.ohlcv_data:
                self.ohlcv_data[exchange] = {}
            
            key = f"{symbol}_{timeframe}"
            
            if key not in self.ohlcv_data[exchange]:
                self.ohlcv_data[exchange][key] = deque(maxlen=self.buffer_size)

            ohlcv_list = []
            for timestamp, row in df.iterrows():
                timestamp_ms = int(pd.Timestamp(timestamp).timestamp() * 1000)
                ohlcv_list.append([
                    timestamp_ms,
                    row['open'], row['high'], row['low'], row['close'], row['volume']
                ])
            
            self.ohlcv_data[exchange][key].clear()
            self.ohlcv_data[exchange][key].extend(ohlcv_list)
            
            logger.info(f"[PRIME] ✅ Primed buffer with {len(ohlcv_list)} candles for {exchange} {key}. Buffer size: {len(self.ohlcv_data[exchange][key])}")

        except Exception as e:
            logger.error(f"[PRIME] ❌ Failed to prime buffer for {exchange} {key}: {e}", exc_info=True)
