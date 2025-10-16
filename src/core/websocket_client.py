"""
WebSocket Client for real-time market data streaming using CCXT Pro.
Enhanced with parse_frame error recovery and auto-reconnection.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    WebSocket client for real-time market data streaming with error recovery.
    
    Features:
    - Real-time OHLCV candle streaming
    - Real-time ticker updates
    - Automatic reconnection on connection loss
    - Parse_frame error recovery
    - Support for multiple exchanges (KuCoin, BingX, etc.)
    """
    
    def __init__(self, ex_name: str, creds: Optional[Dict[str, str]] = None):
        """
        Initialize WebSocket client.
        
        Args:
            ex_name: Exchange name (e.g., 'kucoinfutures', 'bingx')
            creds: Optional API credentials dict with 'apiKey', 'secret', 'password'
        
        Raises:
            AttributeError: If exchange name is invalid or not supported by CCXT Pro
        """
        if not hasattr(ccxtpro, ex_name):
            raise AttributeError(f"Unknown exchange for WebSocket: {ex_name}")
        
        ex_cls = getattr(ccxtpro, ex_name)
        params = {
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        }
        
        if creds:
            params.update(creds)
        
        self.ex = ex_cls(params)
        self.name = ex_name
        self._running = False
        self._tasks = []
        
        # Error tracking for parse_frame issues
        self.parse_frame_errors = {}  # Symbol-based error counter
        self.max_parse_frame_retries = 3
        self.reconnect_delay = 5  # seconds
        self.last_reconnect = None
        self.reconnect_count = 0
        
        logger.info(f"WebSocketClient initialized for {ex_name}")
    
    async def watch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                         callback: Optional[Callable] = None) -> List[List]:
        """
        Watch OHLCV data for a symbol in real-time with parse_frame error recovery.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '15m', '30m', '1h')
            callback: Optional callback function to process each new candle
        
        Returns:
            Latest OHLCV data or None if error
        
        Raises:
            Exception: If streaming fails after retries
        """
        retry_key = f"{symbol}:{timeframe}"
        
        # Initialize error counter if needed
        if retry_key not in self.parse_frame_errors:
            self.parse_frame_errors[retry_key] = 0
        
        try:
            logger.debug(f"Watching OHLCV for {symbol} {timeframe} on {self.name}")
            ohlcv = await self.ex.watch_ohlcv(symbol, timeframe)
            
            # Success - reset error counter
            if self.parse_frame_errors[retry_key] > 0:
                logger.info(f"✅ WebSocket recovered for {symbol} {timeframe}")
                self.parse_frame_errors[retry_key] = 0
            
            if callback:
                await callback(symbol, timeframe, ohlcv)
            
            logger.debug(f"Received OHLCV update for {symbol}: {len(ohlcv)} candles")
            return ohlcv
            
        except AttributeError as e:
            if 'parse_frame' in str(e):
                # Handle parse_frame error
                self.parse_frame_errors[retry_key] += 1
                
                logger.warning(f"⚠️ parse_frame error for {symbol} "
                             f"(attempt {self.parse_frame_errors[retry_key]}/{self.max_parse_frame_retries})")
                
                if self.parse_frame_errors[retry_key] >= self.max_parse_frame_retries:
                    logger.error(f"❌ Max parse_frame retries exceeded for {symbol}, attempting reconnect...")
                    await self._reconnect()
                    self.parse_frame_errors[retry_key] = 0  # Reset after reconnect
                    
                # Return None to signal error but continue
                return None
            else:
                # Other AttributeError
                logger.error(f"AttributeError (non-parse_frame) for {symbol}: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error watching OHLCV for {symbol} on {self.name}: {e}")
            raise
    
    async def watch_ticker(self, symbol: str, 
                          callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Watch ticker data for a symbol in real-time with error recovery.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
            callback: Optional callback function to process each ticker update
        
        Returns:
            Latest ticker data or None if error
        
        Raises:
            Exception: If streaming fails
        """
        try:
            logger.debug(f"Watching ticker for {symbol} on {self.name}")
            ticker = await self.ex.watch_ticker(symbol)
            
            if callback:
                await callback(symbol, ticker)
            
            logger.debug(f"Received ticker update for {symbol}: last={ticker.get('last')}")
            return ticker
            
        except AttributeError as e:
            if 'parse_frame' in str(e):
                logger.warning(f"⚠️ parse_frame error for ticker {symbol}")
                return None
            else:
                raise
                
        except Exception as e:
            logger.error(f"Error watching ticker for {symbol} on {self.name}: {e}")
            raise
    
    async def watch_ohlcv_loop(self, symbol: str, timeframe: str = '1m',
                               callback: Optional[Callable] = None,
                               max_iterations: Optional[int] = None):
        """
        Continuously watch OHLCV data in a loop with parse_frame recovery.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            callback: Callback function called for each update
            max_iterations: Maximum iterations (None for infinite)
        """
        self._running = True
        iteration = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        logger.info(f"Starting OHLCV watch loop for {symbol} {timeframe} on {self.name}")
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations}) for {symbol}")
                    break
                
                try:
                    ohlcv = await self.watch_ohlcv(symbol, timeframe, callback)
                    
                    if ohlcv is not None:
                        iteration += 1
                        consecutive_errors = 0
                    else:
                        # parse_frame error occurred but handled
                        consecutive_errors += 1
                        
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(f"Too many consecutive errors for {symbol}, stopping loop")
                            break
                        
                        # Wait before retry with exponential backoff
                        wait_time = min(60, 2 ** consecutive_errors)
                        logger.info(f"Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    logger.warning(f"Error in watch loop for {symbol}: {e}")
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Max consecutive errors reached for {symbol}")
                        break
                    
                    await asyncio.sleep(min(60, 2 ** consecutive_errors))
                    
        except asyncio.CancelledError:
            logger.info(f"OHLCV watch loop cancelled for {symbol}")
        finally:
            logger.info(f"OHLCV watch loop stopped for {symbol} (iterations: {iteration})")
    
    async def watch_ticker_loop(self, symbol: str,
                                callback: Optional[Callable] = None,
                                max_iterations: Optional[int] = None):
        """
        Continuously watch ticker data in a loop with error recovery.
        
        Args:
            symbol: Trading pair symbol
            callback: Callback function called for each update
            max_iterations: Maximum iterations (None for infinite)
        """
        self._running = True
        iteration = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        logger.info(f"Starting ticker watch loop for {symbol} on {self.name}")
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations}) for {symbol}")
                    break
                
                try:
                    ticker = await self.watch_ticker(symbol, callback)
                    
                    if ticker is not None:
                        iteration += 1
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(f"Too many consecutive errors for ticker {symbol}")
                            break
                        
                        await asyncio.sleep(min(60, 2 ** consecutive_errors))
                    
                except Exception as e:
                    logger.warning(f"Error in ticker watch loop for {symbol}: {e}")
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        break
                    
                    await asyncio.sleep(min(60, 2 ** consecutive_errors))
                    
        except asyncio.CancelledError:
            logger.info(f"Ticker watch loop cancelled for {symbol}")
        finally:
            logger.info(f"Ticker watch loop stopped for {symbol} (iterations: {iteration})")
    
    async def _reconnect(self):
        """
        Force reconnect WebSocket connection.
        """
        # Check if we recently reconnected to avoid reconnect storms
        if self.last_reconnect:
            time_since_reconnect = datetime.now() - self.last_reconnect
            if time_since_reconnect < timedelta(seconds=30):
                logger.info(f"Skipping reconnect (last attempt {time_since_reconnect.seconds}s ago)")
                return
        
        logger.info(f"🔄 Reconnecting WebSocket for {self.name}...")
        
        try:
            # Close existing connection
            await self.close()
            
            # Wait before reconnecting
            await asyncio.sleep(self.reconnect_delay)
            
            # Recreate exchange instance (ccxt.pro will handle reconnection)
            ex_cls = getattr(ccxtpro, self.name)
            params = self.ex.__dict__.get('options', {
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            
            self.ex = ex_cls(params)
            self.reconnect_count += 1
            self.last_reconnect = datetime.now()
            
            logger.info(f"✅ WebSocket reconnection complete for {self.name} (attempt #{self.reconnect_count})")
            
        except Exception as e:
            logger.error(f"Failed to reconnect WebSocket for {self.name}: {e}")
    
    async def close(self):
        """Close the WebSocket connection and cleanup resources."""
        self._running = False
        
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close the exchange connection
        try:
            await self.ex.close()
            logger.info(f"WebSocket connection closed for {self.name}")
        except Exception as e:
            logger.warning(f"Error closing WebSocket for {self.name}: {e}")
    
    def stop(self):
        """Stop all watch loops."""
        self._running = False
        logger.info(f"Stopping all watch loops for {self.name}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get WebSocket health status.
        
        Returns:
            Dictionary with health metrics
        """
        total_errors = sum(self.parse_frame_errors.values())
        max_errors = max(self.parse_frame_errors.values()) if self.parse_frame_errors else 0
        
        health = 'healthy'
        if total_errors > 50:
            health = 'critical'
        elif total_errors > 20:
            health = 'degraded'
        elif total_errors > 5:
            health = 'warning'
        
        return {
            'exchange': self.name,
            'status': health,
            'running': self._running,
            'total_parse_frame_errors': total_errors,
            'max_errors_per_stream': max_errors,
            'reconnect_count': self.reconnect_count,
            'last_reconnect': self.last_reconnect.isoformat() if self.last_reconnect else None,
            'error_streams': {k: v for k, v in self.parse_frame_errors.items() if v > 0}
        }
