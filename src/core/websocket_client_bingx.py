"""
WebSocket Client wrapper for BingX using direct WebSocket
Compatible with existing WebSocketClient interface
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
        
        logger.info("BingX WebSocket client initialized")
    
    async def watch_ohlcv(self, symbol: str, timeframe: str = '1m',
                         callback: Optional[Callable] = None) -> List[List]:
        """
        Watch OHLCV data for a symbol.
        
        Compatible with CCXT Pro interface.
        """
        try:
            # Subscribe if not already
            if not self._is_connected:
                await self.bingx_ws.connect()
                self._is_connected = True
            
            # Subscribe to kline
            await self.bingx_ws.subscribe_kline(symbol, timeframe)
            
            # Wait for data
            await asyncio.sleep(0.5)
            
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
            # Subscribe if not already
            if not self._is_connected:
                await self.bingx_ws.connect()
                self._is_connected = True
            
            # Subscribe to ticker
            await self.bingx_ws.subscribe_ticker(symbol)
            
            # Wait for data
            await asyncio.sleep(0.5)
            
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
        
        # Connect and subscribe once
        if not self._is_connected:
            await self.bingx_ws.connect()
            self._is_connected = True
        
        await self.bingx_ws.subscribe_kline(symbol, timeframe)
        
        # Register callback
        if callback:
            self.bingx_ws.on_kline(callback)
        
        # Start listening
        listen_task = asyncio.create_task(self.bingx_ws.listen())
        self._tasks.append(listen_task)
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    break
                
                iteration += 1
                await asyncio.sleep(1)
                
        finally:
            self._running = False
            listen_task.cancel()
    
    async def watch_ticker_loop(self, symbol: str,
                                callback: Optional[Callable] = None,
                                max_iterations: Optional[int] = None):
        """Continuously watch ticker data in a loop."""
        self._running = True
        iteration = 0
        
        # Connect and subscribe once
        if not self._is_connected:
            await self.bingx_ws.connect()
            self._is_connected = True
        
        await self.bingx_ws.subscribe_ticker(symbol)
        
        # Register callback
        if callback:
            self.bingx_ws.on_ticker(callback)
        
        # Start listening
        listen_task = asyncio.create_task(self.bingx_ws.listen())
        self._tasks.append(listen_task)
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    break
                
                iteration += 1
                await asyncio.sleep(1)
                
        finally:
            self._running = False
            listen_task.cancel()
    
    async def close(self):
        """Close WebSocket connection."""
        self._running = False
        self._is_connected = False
        
        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
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
        
        return {
            'exchange': 'bingx',
            'status': 'healthy' if bingx_status['connected'] else 'disconnected',
            'connected': bingx_status['connected'],
            'streaming': self._first_message_received,
            'running': self._running,
            'last_message_time': self._last_message_time.isoformat() if self._last_message_time else None,
            'message_count': bingx_status['message_count'],
            'subscriptions': bingx_status['subscriptions'],
            'reconnect_count': bingx_status['reconnect_attempts']
        }
