"""
WebSocket Client wrapper for BingX using direct WebSocket
Compatible with existing WebSocketClient interface
FIXED: Singleton listen task pattern
"""

import asyncio
import logging
from .data_validator import validate_kline_timestamp
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime

from .bingx_websocket import BingXWebSocket

if TYPE_CHECKING:
    from .websocket_manager import StreamDataCollector

logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    WebSocket client wrapper for BingX.
    Provides CCXT-like interface using BingX direct WebSocket.
    """
    
    def __init__(self, ex_name: str, creds: Optional[Dict[str, str]] = None, 
                 collector: Optional['StreamDataCollector'] = None):
        """
        Initialize WebSocket client for BingX.
        
        Args:
            ex_name: Exchange name (must be 'bingx')
            creds: Optional credentials
            collector: Optional StreamDataCollector for bridging data
        """
        
        if ex_name.lower() != 'bingx':
            raise ValueError(f"This client only supports BingX, got: {ex_name}")
        
        self.name = 'bingx'
        self._running = False
        self._tasks = []
        self.collector = collector  # ✅ PATCH 3: Store collector reference
        
        self._tasks_started = False
        
        # Initialize BingX WebSocket with collector
        api_key = creds.get('apiKey') if creds else None
        api_secret = creds.get('secret') if creds else None
        
        # ✅ PATCH 3: Pass collector to BingXWebSocket
        self.bingx_ws = BingXWebSocket(
            api_key=api_key,
            api_secret=api_secret,
            futures=True,
            collector=collector
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
        Ensures the underlying connection is active and that the perpetual
        listener and ping tasks have been started exactly once.
        """
        # First, ensure we have a physical connection
        if not (self.bingx_ws.ws and self.bingx_ws.ws.open):
            logger.info("Underlying connection is down. Attempting to connect...")
            connected = await self.bingx_ws.connect()
            if not connected:
                logger.error("Failed to establish underlying BingX connection.")
                self._is_connected = False
                return False
            self._is_connected = True
            logger.info("✅ Underlying BingX connection established.")

        # Now, ensure the background tasks are started, but only once.
        if not self._tasks_started:
            logger.info("Starting underlying listener and ping tasks for the first time...")
            # The listen() method in bingx_websocket now handles starting the ping_loop as well.
            self.bingx_ws._listen_task = asyncio.create_task(self.bingx_ws.listen())
            self._tasks_started = True
            logger.info("✅ Underlying BingX tasks have been started.")
        
        return True
    
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
        """Delegates the close operation to the underlying BingXWebSocket instance."""
        self._running = False
        if self.bingx_ws:
            await self.bingx_ws.disconnect()
        logger.info("BingX WebSocket client wrapper closed successfully.")
    
    def stop(self):
        """Stop all watch loops."""
        self._running = False
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._is_connected and self._running
    
    def get_health_status(self) -> Dict[str, Any]:
        """Gets health status from the underlying BingXWebSocket instance."""
        if not self.bingx_ws:
            return {'status': 'uninitialized'}
        
        # Delegate directly to the underlying get_status method
        # and add wrapper-specific status
        status = self.bingx_ws.get_status()
        status['wrapper_running'] = self._running
        return status
