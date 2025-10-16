"""
Enhanced WebSocket client with parse_frame error handling and auto-recovery.
"""
import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)

class EnhancedWebSocketClient:
    def __init__(self, exchange_name: str, exchange_config: dict = None):
        self.exchange_name = exchange_name
        self.exchange_config = exchange_config or {}
        self.exchange = None
        self.is_connected = False
        self.error_count = 0
        self.parse_frame_errors = 0
        self.last_error_time = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Error tracking
        self.error_history = []
        self.max_error_history = 100
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize ccxt.pro exchange instance"""
        try:
            # Get exchange class
            exchange_class = getattr(ccxtpro, self.exchange_name)
            
            # Exchange config
            config = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # For perpetuals
                    'watchOHLCV': {
                        'limit': 500  # Default limit
                    }
                },
                **self.exchange_config
            }
            
            self.exchange = exchange_class(config)
            logger.info(f"✅ Initialized {self.exchange_name} WebSocket client")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name}: {e}")
            raise
    
    async def watch_ohlcv_safe(self, symbol: str, timeframe: str, limit: int = None):
        """
        Watch OHLCV with parse_frame error recovery.
        
        Returns:
            List of OHLCV candles or None if error
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Check if we need reconnection
                if self.parse_frame_errors > 5:
                    logger.warning(f"Too many parse_frame errors ({self.parse_frame_errors}), reconnecting...")
                    await self._reconnect()
                    self.parse_frame_errors = 0
                
                # Try to fetch data
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe, limit=limit)
                
                # Success - reset error counters
                if self.error_count > 0:
                    logger.info(f"✅ {self.exchange_name} recovered after {self.error_count} errors")
                    self.error_count = 0
                    self.parse_frame_errors = 0
                
                self.is_connected = True
                return ohlcv
                
            except AttributeError as e:
                if 'parse_frame' in str(e):
                    # parse_frame error - specific handling
                    self.parse_frame_errors += 1
                    self.error_count += 1
                    retry_count += 1
                    
                    self._log_error('parse_frame', str(e))
                    
                    if retry_count < max_retries:
                        wait_time = self.reconnect_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        logger.warning(f"⚠️ parse_frame error #{self.parse_frame_errors} for {symbol}, "
                                     f"waiting {wait_time}s before retry {retry_count}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        
                        # Force reconnection after 3 parse_frame errors
                        if self.parse_frame_errors >= 3:
                            await self._reconnect()
                    else:
                        logger.error(f"❌ Max retries reached for {symbol}, giving up")
                        self.is_connected = False
                        return None
                else:
                    # Other AttributeError
                    raise
                    
            except Exception as e:
                self.error_count += 1
                retry_count += 1
                
                error_type = type(e).__name__
                self._log_error(error_type, str(e))
                
                logger.error(f"WebSocket error for {symbol}: {error_type}: {e}")
                
                if retry_count < max_retries:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    self.is_connected = False
                    return None
        
        return None
    
    async def _reconnect(self):
        """Force reconnect WebSocket connection"""
        logger.info(f"🔄 Reconnecting {self.exchange_name} WebSocket...")
        
        try:
            # Close existing connection
            if self.exchange:
                await self.close()
            
            # Wait before reconnecting
            await asyncio.sleep(2)
            
            # Reinitialize exchange
            self._initialize_exchange()
            
            self.reconnect_attempts += 1
            self.is_connected = False  # Will be set to True on successful data fetch
            
            logger.info(f"✅ {self.exchange_name} WebSocket reconnection complete (attempt #{self.reconnect_attempts})")
            
        except Exception as e:
            logger.error(f"Failed to reconnect {self.exchange_name}: {e}")
            raise
    
    async def close(self):
        """Close WebSocket connection gracefully"""
        try:
            if self.exchange:
                await self.exchange.close()
                logger.info(f"Closed {self.exchange_name} WebSocket connection")
        except Exception as e:
            logger.warning(f"Error closing {self.exchange_name}: {e}")
    
    def _log_error(self, error_type: str, error_msg: str):
        """Log error to history for analysis"""
        error_entry = {
            'timestamp': datetime.now(),
            'type': error_type,
            'message': error_msg[:200]  # Limit message length
        }
        
        self.error_history.append(error_entry)
        
        # Keep only recent errors
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        self.last_error_time = datetime.now()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get WebSocket health status"""
        # Calculate error rate (last 5 minutes)
        recent_errors = [
            e for e in self.error_history 
            if e['timestamp'] > datetime.now() - timedelta(minutes=5)
        ]
        
        # Count parse_frame errors
        parse_frame_count = sum(
            1 for e in recent_errors 
            if e['type'] == 'parse_frame'
        )
        
        return {
            'exchange': self.exchange_name,
            'connected': self.is_connected,
            'total_errors': self.error_count,
            'parse_frame_errors': self.parse_frame_errors,
            'recent_errors_5min': len(recent_errors),
            'parse_frame_5min': parse_frame_count,
            'reconnect_attempts': self.reconnect_attempts,
            'last_error': self.last_error_time.isoformat() if self.last_error_time else None,
            'health': self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> str:
        """Calculate overall health score"""
        if not self.is_connected:
            return 'critical'
        
        if self.parse_frame_errors > 10:
            return 'critical'
        elif self.parse_frame_errors > 5:
            return 'degraded'
        elif self.error_count > 20:
            return 'warning'
        elif self.error_count > 0:
            return 'minor'
        else:
            return 'healthy'

    async def watch_ohlcv_batch(self, symbols: List[str], timeframe: str, limit: int = None):
        """
        Watch multiple symbols efficiently.
        
        Args:
            symbols: List of symbols to watch
            timeframe: Timeframe (e.g., '1m', '5m')
            limit: Number of candles
            
        Returns:
            Dict mapping symbol to OHLCV data
        """
        results = {}
        
        # Process in batches to avoid overwhelming
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            tasks = [
                self.watch_ohlcv_safe(symbol, timeframe, limit)
                for symbol in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching {symbol}: {result}")
                    results[symbol] = None
                else:
                    results[symbol] = result
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.5)
        
        return results
