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
        """Closes the connection and cancels background tasks."""
        self._running = False
        
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        if self.bingx_ws:
            await self.bingx_ws.disconnect()
            
        logger.info("BingX WebSocket client wrapper closed successfully.")
    
    def stop(self):
        """Stops the loops."""
        self._running = False
    
    def is_connected(self) -> bool:
        """Checks if the underlying WebSocket is connected."""
        return self.bingx_ws.is_connected() if self.bingx_ws else False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Gets a reliable health status from the underlying BingXWebSocket instance."""
        if not self.bingx_ws:
            return {'status': 'uninitialized', 'connected': False}
        
        # Görevi doğrudan alt katmana devrediyoruz.
        # Alt katmanın get_status metodu artık güvenilir olduğu için
        # burada ek bir işlem yapmaya gerek yok.
        status = self.bingx_ws.get_status()
        status['wrapper_running'] = self._running
        return status
