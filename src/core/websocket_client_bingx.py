"""
WebSocket Client wrapper for BingX using direct WebSocket
Compatible with existing WebSocketClient interface
FIXED: Singleton listen task pattern
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime

from .bingx_websocket import BingXWebSocket

if TYPE_CHECKING:
    from .stream_data_collector import StreamDataCollector

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
        self.collector = collector
        
        # === YARIŞ DURUMUNU ÖNLEMEK İÇİN KİLİT MEKANİZMASI ===
        self._connection_lock = asyncio.Lock()
        self._tasks_started = False
        
        # Initialize BingX WebSocket with collector
        api_key = creds.get('apiKey') if creds else None
        api_secret = creds.get('secret') if creds else None
        
        self.bingx_ws = BingXWebSocket(
            api_key=api_key,
            api_secret=api_secret,
            futures=True,
            collector=collector
        )
        
        # Connection tracking (tüm orijinal değişkenler korunuyor)
        self._is_connected = False
        self._first_message_received = False
        self._last_message_time = None
        
        # Diagnostic / telemetry / error-tracking defaults (tüm orijinal değişkenler korunuyor)
        self.error_history: List[Dict[str, Any]] = []
        self.max_error_history: int = 100
        self.parse_frame_errors: Dict[str, int] = {}
        self.max_parse_frame_retries: int = 3
        self.reconnect_delay: float = 5.0
        self.reconnect_count: int = 0
        self.last_reconnect: Optional[datetime] = None
        self.use_rest_fallback: bool = False
        
        logger.info("BingX WebSocket client initialized with Connection Lock")

    async def _ensure_connection_and_listener(self) -> bool:
        """
        Ensures the underlying WebSocket client's 'start' method is called once.
        Uses a lock to prevent race conditions.
        Waits for connection to establish before returning.
        """
        async with self._connection_lock:
            if not self._tasks_started:
                self.bingx_ws.start()  # Yeni, thread-safe başlatma metodu
                self._tasks_started = True
                logger.info("BingXWebSocket client 'start' method called.")
                
                # Wait for connection to establish (max 10 seconds)
                logger.info("Waiting for WebSocket connection to establish...")
                for i in range(20):  # 20 iterations * 0.5s = 10 seconds max
                    await asyncio.sleep(0.5)
                    if self.bingx_ws.is_connected():
                        logger.info(f"✅ WebSocket connected after {(i+1)*0.5:.1f}s")
                        return True
                
                # Connection not established within timeout
                logger.warning("⚠️ WebSocket connection not established within 10s, proceeding anyway (subscriptions will be queued)")
                return True  # Return True to allow subscriptions to be queued
            
            # Already started, just verify connection
            if not self.bingx_ws.is_connected():
                logger.debug("WebSocket not yet connected, subscriptions will be queued")
            
            return True
    
    async def watch_ohlcv_loop(self, symbol: str, timeframe: str = '1m',
                               callback: Optional[Callable] = None,
                               max_iterations: Optional[int] = None):
        """
        Subscribes to the OHLCV stream for a symbol.
        This is the correct, event-driven approach.
        """
        self._running = True
        
        if not await self._ensure_connection_and_listener():
            logger.error(f"Cannot start OHLCV loop for {symbol} due to connection failure.")
            return
        
        await self.bingx_ws.subscribe_kline(symbol, timeframe)
        
        if callback:
            self.bingx_ws.on_kline(callback)
            
        logger.info(f"Subscribed to OHLCV stream for {symbol} [{timeframe}]")
    
    async def watch_ticker_loop(self, symbol: str,
                                callback: Optional[Callable] = None,
                                max_iterations: Optional[int] = None):
        """Subscribes to the ticker stream for a symbol."""
        self._running = True
        
        if not await self._ensure_connection_and_listener():
            logger.error(f"Cannot start Ticker loop for {symbol} due to connection failure.")
            return

        await self.bingx_ws.subscribe_ticker(symbol)
        
        if callback:
            self.bingx_ws.on_ticker(callback)

        logger.info(f"Subscribed to Ticker stream for {symbol}")
    
    async def close(self):
        """Closes the connection and stops background tasks."""
        self._running = False
        
        if self.bingx_ws:
            await self.bingx_ws.stop() # Yeni asenkron stop metodu
            
        logger.info("BingX WebSocket client wrapper closed successfully.")
    
    def stop(self):
        """Stops the loops."""
        self._running = False
    
    def is_connected(self) -> bool:
        """Checks if the underlying WebSocket is connected."""
        return self.bingx_ws.is_connected() if self.bingx_ws else False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the underlying BingX WebSocket connection.
        This version is more robust and directly checks the underlying library's state.
        """
        if not self.bingx_ws:
            return {
                'connected': False,
                'listen_task_status': 'not_initialized',
                'subscriptions': 0,
                'message_count': 0
            }

        # Check connection status from BingXWebSocket
        # Use the _is_connected flag which is set in on_open/on_close callbacks
        is_connected = getattr(self.bingx_ws, '_is_connected', False)
        
        # Check if WebSocket thread is running
        ws_thread = getattr(self.bingx_ws, '_ws_thread', None)
        listen_status = "running" if ws_thread and ws_thread.is_alive() else "stopped"
        
        # Get subscription count
        subscriptions = len(getattr(self.bingx_ws, 'subscriptions', {}))
        
        # Get message count
        message_count = getattr(self.bingx_ws, 'message_count', 0)
        
        # Get last message time
        last_msg_time = getattr(self.bingx_ws, 'last_message_time', None)

        return {
            'connected': is_connected,
            'listen_task_status': listen_status,
            'subscriptions': subscriptions,
            'message_count': message_count,
            'last_message_time': last_msg_time
        }
