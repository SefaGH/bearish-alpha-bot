"""
WebSocket Client for real-time market data streaming using CCXT Pro.
Provides async streaming of OHLCV, ticker, and other market data.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    WebSocket client for real-time market data streaming.
    
    Features:
    - Real-time OHLCV candle streaming
    - Real-time ticker updates
    - Automatic reconnection on connection loss
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
        
        logger.info(f"WebSocketClient initialized for {ex_name}")
    
    async def watch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                         callback: Optional[Callable] = None) -> List[List]:
        """
        Watch OHLCV data for a symbol in real-time.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '15m', '30m', '1h')
            callback: Optional callback function to process each new candle
        
        Returns:
            Latest OHLCV data
        
        Raises:
            Exception: If streaming fails after retries
        """
        try:
            logger.debug(f"Watching OHLCV for {symbol} {timeframe} on {self.name}")
            ohlcv = await self.ex.watch_ohlcv(symbol, timeframe)
            
            if callback:
                await callback(symbol, timeframe, ohlcv)
            
            logger.debug(f"Received OHLCV update for {symbol}: {len(ohlcv)} candles")
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error watching OHLCV for {symbol} on {self.name}: {e}")
            raise
    
    async def watch_ticker(self, symbol: str, 
                          callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Watch ticker data for a symbol in real-time.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
            callback: Optional callback function to process each ticker update
        
        Returns:
            Latest ticker data
        
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
            
        except Exception as e:
            logger.error(f"Error watching ticker for {symbol} on {self.name}: {e}")
            raise
    
    async def watch_ohlcv_loop(self, symbol: str, timeframe: str = '1m',
                               callback: Optional[Callable] = None,
                               max_iterations: Optional[int] = None):
        """
        Continuously watch OHLCV data in a loop.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            callback: Callback function called for each update
            max_iterations: Maximum iterations (None for infinite)
        """
        self._running = True
        iteration = 0
        
        logger.info(f"Starting OHLCV watch loop for {symbol} {timeframe} on {self.name}")
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations}) for {symbol}")
                    break
                
                try:
                    ohlcv = await self.watch_ohlcv(symbol, timeframe, callback)
                    iteration += 1
                    
                except Exception as e:
                    logger.warning(f"Error in watch loop for {symbol}: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
                    
        except asyncio.CancelledError:
            logger.info(f"OHLCV watch loop cancelled for {symbol}")
        finally:
            logger.info(f"OHLCV watch loop stopped for {symbol} (iterations: {iteration})")
    
    async def watch_ticker_loop(self, symbol: str,
                                callback: Optional[Callable] = None,
                                max_iterations: Optional[int] = None):
        """
        Continuously watch ticker data in a loop.
        
        Args:
            symbol: Trading pair symbol
            callback: Callback function called for each update
            max_iterations: Maximum iterations (None for infinite)
        """
        self._running = True
        iteration = 0
        
        logger.info(f"Starting ticker watch loop for {symbol} on {self.name}")
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations}) for {symbol}")
                    break
                
                try:
                    ticker = await self.watch_ticker(symbol, callback)
                    iteration += 1
                    
                except Exception as e:
                    logger.warning(f"Error in ticker watch loop for {symbol}: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
                    
        except asyncio.CancelledError:
            logger.info(f"Ticker watch loop cancelled for {symbol}")
        finally:
            logger.info(f"Ticker watch loop stopped for {symbol} (iterations: {iteration})")
    
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
        await self.ex.close()
        logger.info(f"WebSocket connection closed for {self.name}")
    
    def stop(self):
        """Stop all watch loops."""
        self._running = False
        logger.info(f"Stopping all watch loops for {self.name}")
