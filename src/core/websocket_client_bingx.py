"""
WebSocket Client wrapper for BingX using direct WebSocket
Compatible with existing WebSocketClient interface
FIXED: Singleton listen task pattern
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .bingx_websocket import BingXWebSocket

logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    WebSocket client wrapper for BingX.
    Provides CCXT-like interface using BingX direct WebSocket.
    """
    
    def __init__(self, ex_name: str, creds: Optional[Dict[str, str]] = None):
        """Initialize WebSocket client for BingX."""
        
        if ex_name.lower() != 'bingx':
            raise ValueError(f"This client only supports BingX, got: {ex_name}")
        
        self.name = 'bingx'
        self._running = False
        self._tasks = []
        
        # ✅ NEW: Singleton listen task management
        self._listen_task: Optional[asyncio.Task] = None
        self._listen_lock = asyncio.Lock()
        
        # Initialize BingX WebSocket
        api_key = creds.get('apiKey') if creds else None
        api_secret = creds.get('secret') if creds else None
        
        self.bingx_ws = BingXWebSocket(
            api_key=api_key,
            api_secret=api_secret,
            futures=True  # Use futures market
        )
        
        # Connection tracking
        self._is_connected = False
        self._first_message_received = False
        self._last_message_time = None
        
        # Diagnostic / telemetry / error-tracking defaults
        self.error_history: List[Dict[str, Any]] = []
        self.max_error_history: int = 100
        self.parse_frame_errors: Dict[str, int] = {}
        self.max_parse_frame_retries: int = 3
        self.reconnect_delay: float = 5.0
        self.reconnect_count: int = 0
        self.last_reconnect: Optional[datetime] = None
        self.use_rest_fallback: bool = False
        
        logger.info("BingX WebSocket client initialized")
    
    async def _ensure_connection_and_listener(self) -> bool:
        """
        Ensure WebSocket is connected and exactly ONE listen task is running.
        Thread-safe with lock protection.
        """
        async with self._listen_lock:
            try:
                # Check if we need to connect
                if not self._is_connected:
                    logger.info("Establishing BingX WebSocket connection...")
                    connected = await self.bingx_ws.connect()
                    
                    if not connected:
                        logger.error("Failed to establish BingX WebSocket connection")
                        return False
                    
                    self._is_connected = True
                    logger.info("✅ BingX WebSocket connected")
                
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
                    logger.info("Starting BingX WebSocket listener task...")
                    self._listen_task = asyncio.create_task(self.bingx_ws.listen())
                    
                    # Track in tasks list for cleanup
                    if self._listen_task not in self._tasks:
                        self._tasks.append(self._listen_task)
                    
                    logger.info("✅ BingX WebSocket listener task started")
                else:
                    logger.debug("BingX WebSocket listener already running")
                
                return True
                
            except Exception as e:
                logger.error(f"Error in BingX connection/listener setup: {e}")
                self._is_connected = False
                return False
    
    async def watch_ohlcv(self, symbol: str, timeframe: str = '1m',
                         callback: Optional[Callable] = None) -> List[List]:
        """
        Watch OHLCV data for a symbol.
        Compatible with CCXT Pro interface.
        """
        try:
            # ✅ Use singleton pattern
            if not await self._ensure_connection_and_listener():
                logger.error("Failed to establish connection/listener")
                return []
            
            # Subscribe to kline
            await self.bingx_ws.subscribe_kline(symbol, timeframe)
            
            # Wait for data
            await asyncio.sleep(1)
            
            # Get latest klines
            klines = self.bingx_ws.get_klines(symbol, timeframe)
            
            if klines:
                self._first_message_received = True
                self._last_message_time = datetime.now()
                
                if callback:
                    await callback(symbol, timeframe, klines)
                
                return klines
            
            return []
            
        except Exception as e:
            logger.error(f"Error watching OHLCV for {symbol}: {e}")
            return []
    
    async def watch_ticker(self, symbol: str,
                          callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Watch ticker data for a symbol.
        Compatible with CCXT Pro interface.
        """
        try:
            # ✅ Use singleton pattern
            if not await self._ensure_connection_and_listener():
                logger.error("Failed to establish connection/listener")
                return {}
            
            # Subscribe to ticker
            await self.bingx_ws.subscribe_ticker(symbol)
            
            # Wait for data
            await asyncio.sleep(1)
            
            # Get latest ticker
            ticker = self.bingx_ws.get_ticker(symbol)
            
            if ticker:
                self._first_message_received = True
                self._last_message_time = datetime.now()
                
                if callback:
                    await callback(symbol, ticker)
                
                return ticker
            
            return {}
            
        except Exception as e:
            logger.error(f"Error watching ticker for {symbol}: {e}")
            return {}
    
    async def watch_ohlcv_loop(self, symbol: str, timeframe: str = '1m',
                               callback: Optional[Callable] = None,
                               max_iterations: Optional[int] = None):
        """Continuously watch OHLCV data in a loop."""
        self._running = True
        iteration = 0
        
        # ✅ Use singleton pattern
        if not await self._ensure_connection_and_listener():
            logger.error("Failed to establish connection/listener")
            return
        
        # Subscribe to kline
        await self.bingx_ws.subscribe_kline(symbol, timeframe)
        
        # Register callback
        if callback:
            self.bingx_ws.on_kline(callback)
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    break
                
                iteration += 1
                await asyncio.sleep(1)
                
                # Check if listener is still running
                if self._listen_task and self._listen_task.done():
                    logger.warning("Listener task stopped, restarting...")
                    await self._ensure_connection_and_listener()
                
        finally:
            self._running = False
    
    async def watch_ticker_loop(self, symbol: str,
                                callback: Optional[Callable] = None,
                                max_iterations: Optional[int] = None):
        """Continuously watch ticker data in a loop."""
        self._running = True
        iteration = 0
        
        # ✅ Use singleton pattern
        if not await self._ensure_connection_and_listener():
            logger.error("Failed to establish connection/listener")
            return
        
        # Subscribe to ticker
        await self.bingx_ws.subscribe_ticker(symbol)
        
        # Register callback
        if callback:
            self.bingx_ws.on_ticker(callback)
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    break
                
                iteration += 1
                await asyncio.sleep(1)
                
                # Check if listener is still running
                if self._listen_task and self._listen_task.done():
                    logger.warning("Listener task stopped, restarting...")
                    await self._ensure_connection_and_listener()
                
        finally:
            self._running = False
    
    async def close(self):
        """Close WebSocket connection."""
        self._running = False
        self._is_connected = False
        
        # Cancel listen task first
        if self._listen_task and not self._listen_task.done():
            logger.info("Cancelling listen task...")
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        # Cancel other tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close BingX WebSocket
        await self.bingx_ws.disconnect()
        
        logger.info("BingX WebSocket client closed")
    
    def stop(self):
        """Stop all watch loops."""
        self._running = False
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._is_connected and self._running
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get WebSocket health status."""
        bingx_status = self.bingx_ws.get_status()
        
        # Add listen task status
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
            'exchange': 'bingx',
            'status': 'healthy' if bingx_status['connected'] else 'disconnected',
            'connected': bingx_status['connected'],
            'streaming': self._first_message_received,
            'running': self._running,
            'last_message_time': self._last_message_time.isoformat() if self._last_message_time else None,
            'message_count': bingx_status['message_count'],
            'subscriptions': bingx_status['subscriptions'],
            'reconnect_count': bingx_status['reconnect_attempts'],
            'listen_task_status': listen_task_status,
            'total_tasks': len(self._tasks),
            'active_tasks': sum(1 for t in self._tasks if not t.done())
        }
