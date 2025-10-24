"""
WebSocket Client for real-time market data streaming.
Enhanced with BingX direct WebSocket support when CCXT Pro is not available.
FIXED: Singleton listen task pattern to prevent concurrent recv errors.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta

# Try importing CCXT Pro first
try:
    import ccxt.pro as ccxtpro
    CCXT_PRO_AVAILABLE = True
except ImportError:
    CCXT_PRO_AVAILABLE = False
    import ccxt  # Regular CCXT for fallback

logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    WebSocket client for real-time market data streaming.
    Uses CCXT Pro when available, falls back to BingX Direct WebSocket or REST API.
    """
    
    def __init__(self, ex_name: str, creds: Optional[Dict[str, str]] = None):
        """
        Initialize WebSocket client.
        
        Args:
            ex_name: Exchange name (e.g., 'kucoinfutures', 'bingx')
            creds: Optional API credentials dict with 'apiKey', 'secret', 'password'
        """
        self.name = ex_name.lower()
        self._running = False
        self._tasks = []
        
        # Connection state tracking
        self._is_connected = False
        self._first_message_received = False
        self._last_message_time = None
        
        # ✅ NEW: Singleton listen task management
        self._listen_task: Optional[asyncio.Task] = None
        self._listen_lock = asyncio.Lock()  # Prevent concurrent listen task creation

        # Diagnostic / telemetry / error-tracking defaults
        # Ensure attributes used by get_health_status() and _log_error() exist
        self.error_history: List[Dict[str, Any]] = []        # recent error records
        self.max_error_history: int = 100                    # keep at most this many error records
        # parse_frame_errors may be used to count parse issues per stream
        self.parse_frame_errors: Dict[str, int] = {}
        self.max_parse_frame_retries: int = 3
        # reconnect settings & counters
        self.reconnect_delay: float = 5.0
        self.reconnect_count: int = 0
        self.last_reconnect: Optional[datetime] = None
        # flag used when we fell back to REST polling
        self.use_rest_fallback: bool = False
        
        # BingX-specific handling when CCXT Pro is not available
        if self.name == 'bingx' and not CCXT_PRO_AVAILABLE:
            logger.info("🎯 Using BingX Direct WebSocket (no CCXT Pro)")
            
            # Import and use BingX Direct WebSocket
            from .bingx_websocket import BingXWebSocket
            
            api_key = creds.get('apiKey') if creds else None
            api_secret = creds.get('secret') if creds else None
            
            self.ws_client = BingXWebSocket(
                api_key=api_key,
                api_secret=api_secret,
                futures=True
            )
            self.use_direct_ws = True

            # ✅ NEW: Singleton listen task management
            self._listen_task: Optional[asyncio.Task] = None
            self._listen_lock = asyncio.Lock()
            
        elif CCXT_PRO_AVAILABLE:
            # Use CCXT Pro for all exchanges
            if not hasattr(ccxtpro, self.name):
                raise AttributeError(f"Unknown exchange for WebSocket: {self.name}")
            
            ex_cls = getattr(ccxtpro, self.name)
            params = {
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            }
            
            if creds:
                params.update(creds)
            
            self.ex = ex_cls(params)
            self.use_direct_ws = False
            logger.info(f"Using CCXT Pro for {ex_name}")
            
        else:
            # Fallback to REST API
            logger.warning(f"No WebSocket available for {ex_name}, will use REST API polling")
            
            if not hasattr(ccxt, self.name):
                raise AttributeError(f"Unknown exchange: {self.name}")
            
            ex_cls = getattr(ccxt, self.name)
            params = {
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            }
            
            if creds:
                params.update(creds)
            
            self.ex = ex_cls(params)
            self.use_direct_ws = False
            self.use_rest_fallback = True
        
        # Error tracking
        self.parse_frame_errors = {}
        self.max_parse_frame_retries = 3
        self.reconnect_delay = 5
        
        logger.info(f"WebSocketClient initialized for {ex_name}")
    
    # ✅ NEW: Singleton listen task management method
    async def _ensure_connection_and_listener(self) -> bool:
        """
        Ensure WebSocket is connected and exactly ONE listen task is running.
        Thread-safe with lock protection.
        
        Returns:
            True if connection and listener are ready, False otherwise
        """
        async with self._listen_lock:
            try:
                # Check if we need to connect
                if not self._is_connected:
                    logger.info("Establishing WebSocket connection...")
                    connected = await self.ws_client.connect()
                    
                    if not connected:
                        logger.error("Failed to establish WebSocket connection")
                        return False
                    
                    self._is_connected = True
                    logger.info("✅ WebSocket connected")
                
                # Check if listen task needs to be started
                if self._listen_task is None or self._listen_task.done():
                    if self._listen_task and self._listen_task.done():
                        # Check if previous task had an exception
                        try:
                            exc = self._listen_task.exception()
                            if exc:
                                logger.warning(f"Previous listen task failed with: {exc}")
                        except (asyncio.CancelledError, asyncio.InvalidStateError):
                            pass
                    
                    # Create new listen task
                    logger.info("Starting WebSocket listener task...")
                    self._listen_task = asyncio.create_task(self.ws_client.listen())
                    
                    # Track in tasks list for cleanup
                    if self._listen_task not in self._tasks:
                        self._tasks.append(self._listen_task)
                    
                    logger.info("✅ WebSocket listener task started")
                else:
                    logger.debug("WebSocket listener already running")
                
                return True
                
            except Exception as e:
                logger.error(f"Error in connection/listener setup: {e}")
                self._is_connected = False
                return False
    
    async def watch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                         callback: Optional[Callable] = None) -> List[List]:
        """
        Watch OHLCV data for a symbol in real-time.
        """
        try:
            # Use BingX Direct WebSocket if available
            if hasattr(self, 'use_direct_ws') and self.use_direct_ws:
                # ✅ FIXED: Use singleton listen task pattern
                if not await self._ensure_connection_and_listener():
                    logger.error("Failed to establish connection/listener")
                    return []
                
                # Subscribe to kline (safe to call multiple times)
                await self.ws_client.subscribe_kline(symbol, timeframe)
                
                # Wait a bit for data
                await asyncio.sleep(1)
                
                # Get data
                ohlcv = self.ws_client.get_klines(symbol, timeframe)
                
                if ohlcv:
                    self._first_message_received = True
                    self._last_message_time = datetime.now()
                    
                    if callback:
                        await callback(symbol, timeframe, ohlcv)
                    
                    return ohlcv
                
                return []
                
            elif CCXT_PRO_AVAILABLE:
                # Original CCXT Pro code
                logger.debug(f"Watching OHLCV for {symbol} {timeframe} on {self.name}")
                ohlcv = await self.ex.watch_ohlcv(symbol, timeframe)
                
                if not self._first_message_received:
                    logger.info(f"✅ WebSocket connected and streaming for {self.name}")
                    self._first_message_received = True
                
                self._is_connected = True
                self._last_message_time = datetime.now()
                
                if callback:
                    await callback(symbol, timeframe, ohlcv)
                
                return ohlcv
                
            else:
                # REST API fallback
                logger.debug(f"Using REST API for {symbol} {timeframe}")
                ohlcv = self.ex.fetch_ohlcv(symbol, timeframe, limit=100)
                
                if callback:
                    await callback(symbol, timeframe, ohlcv)
                
                return ohlcv
                
        except Exception as e:
            logger.error(f"Error watching OHLCV for {symbol} on {self.name}: {e}")
            self._log_error('watch_ohlcv', str(e))
            return []
    
    async def watch_ticker(self, symbol: str, 
                          callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Watch ticker data for a symbol in real-time.
        """
        try:
            # Use BingX Direct WebSocket if available
            if hasattr(self, 'use_direct_ws') and self.use_direct_ws:
                # ✅ FIXED: Use singleton listen task pattern
                if not await self._ensure_connection_and_listener():
                    logger.error("Failed to establish connection/listener")
                    return {}
                
                # Subscribe to ticker (safe to call multiple times)
                await self.ws_client.subscribe_ticker(symbol)
                
                # Wait a bit for data
                await asyncio.sleep(1)
                
                # Get data
                ticker = self.ws_client.get_ticker(symbol)
                
                if ticker:
                    self._first_message_received = True
                    self._last_message_time = datetime.now()
                    
                    if callback:
                        await callback(symbol, ticker)
                    
                    return ticker
                
                return {}
                
            elif CCXT_PRO_AVAILABLE:
                # Original CCXT Pro code
                logger.debug(f"Watching ticker for {symbol} on {self.name}")
                ticker = await self.ex.watch_ticker(symbol)
                
                self._is_connected = True
                self._last_message_time = datetime.now()
                
                if callback:
                    await callback(symbol, ticker)
                
                return ticker
                
            else:
                # REST API fallback
                ticker = self.ex.fetch_ticker(symbol)
                
                if callback:
                    await callback(symbol, ticker)
                
                return ticker
                
        except Exception as e:
            logger.error(f"Error watching ticker for {symbol}: {e}")
            self._log_error('watch_ticker', str(e))
            return {}
    
    async def close(self):
        """Close the WebSocket connection and cleanup resources."""
        self._running = False
        self._is_connected = False
        
        # ✅ NEW: Cancel listen task specifically
        if self._listen_task and not self._listen_task.done():
            logger.info("Cancelling listen task...")
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close connection
        try:
            if hasattr(self, 'use_direct_ws') and self.use_direct_ws:
                await self.ws_client.disconnect()
            elif hasattr(self, 'ex') and hasattr(self.ex, 'close'):
                await self.ex.close()
                
            logger.info(f"WebSocket connection closed for {self.name}")
        except Exception as e:
            logger.warning(f"Error closing WebSocket for {self.name}: {e}")
    
    def stop(self):
        """Stop all watch loops."""
        self._running = False
        logger.info(f"Stopping all watch loops for {self.name}")
    
    def is_connected(self) -> bool:
        """
        Check if WebSocket is actually connected and streaming data.
        
        Returns:
            True if WebSocket is connected, running, and has received at least one message
        """
        return self._is_connected and self._running and self._first_message_received
    
    def _log_error(self, error_type: str, error_msg: str):
        """
        Log error to history for debugging.
        
        Args:
            error_type: Type of error (e.g., 'parse_frame', 'AttributeError')
            error_msg: Error message
        """
        error_entry = {
            'timestamp': datetime.now(),
            'type': error_type,
            'message': error_msg[:200]  # Limit message length
        }
        
        self.error_history.append(error_entry)
        
        # Keep only recent errors
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def _get_recent_error_count(self, seconds: int) -> int:
        """
        Get error count in last N seconds.
        
        Args:
            seconds: Time window in seconds
            
        Returns:
            Number of errors in the time window
        """
        if not self.error_history:
            return 0
        
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        return sum(1 for e in self.error_history if e['timestamp'] > cutoff_time)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get WebSocket health status with enhanced connection info.
        
        Returns:
            Dictionary with health metrics including connection state
        """
        total_errors = sum(self.parse_frame_errors.values())
        max_errors = max(self.parse_frame_errors.values()) if self.parse_frame_errors else 0
        
        # Enhanced health calculation
        health = 'healthy'
        if not self._is_connected:
            health = 'disconnected'
        elif not self._first_message_received:
            health = 'connecting'
        elif total_errors > 50:
            health = 'critical'
        elif total_errors > 20:
            health = 'degraded'
        elif total_errors > 5:
            health = 'warning'
        
        # ✅ NEW: Add listen task status
        listen_task_status = 'not_created'
        if self._listen_task:
            if self._listen_task.done():
                try:
                    exc = self._listen_task.exception()
                    listen_task_status = f'failed: {exc}' if exc else 'completed'
                except (asyncio.CancelledError, asyncio.InvalidStateError):
                    listen_task_status = 'cancelled'
            else:
                listen_task_status = 'running'
        
        return {
            'exchange': self.name,
            'status': health,
            'connected': self._is_connected,
            'streaming': self._first_message_received,
            'running': self._running,
            'last_message_time': self._last_message_time.isoformat() if self._last_message_time else None,
            'total_parse_frame_errors': total_errors,
            'max_errors_per_stream': max_errors,
            'reconnect_count': self.reconnect_count,
            'last_reconnect': self.last_reconnect.isoformat() if self.last_reconnect else None,
            'error_streams': {k: v for k, v in self.parse_frame_errors.items() if v > 0},
            'recent_errors_5min': self._get_recent_error_count(300),
            'listen_task_status': listen_task_status,  # ✅ NEW
            'total_tasks': len(self._tasks),  # ✅ NEW
            'active_tasks': sum(1 for t in self._tasks if not t.done())  # ✅ NEW
        }
