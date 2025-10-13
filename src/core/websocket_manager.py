"""
WebSocket Manager for multi-exchange real-time data streaming.
Coordinates WebSocket connections across multiple exchanges.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
from .websocket_client import WebSocketClient

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Multi-Exchange WebSocket Manager for coordinated real-time streaming.
    
    Features:
    - Unified streaming interface across multiple exchanges
    - Stream multiplexing for multiple symbols
    - Automatic connection management
    - Integration with existing multi-exchange framework
    """
    
    def __init__(self, exchanges: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize WebSocketManager.
        
        Args:
            exchanges: Dict mapping exchange names to credentials
                      e.g., {'kucoinfutures': {'apiKey': '...', 'secret': '...'},
                             'bingx': {'apiKey': '...', 'secret': '...'}}
                      If None, creates unauthenticated clients for KuCoin and BingX
        """
        self.clients: Dict[str, WebSocketClient] = {}
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        if exchanges is None:
            # Default to unauthenticated clients
            exchanges = {
                'kucoinfutures': None,
                'bingx': None
            }
        
        for ex_name, creds in exchanges.items():
            try:
                self.clients[ex_name] = WebSocketClient(ex_name, creds)
                logger.info(f"WebSocket client initialized for {ex_name}")
            except Exception as e:
                logger.error(f"Failed to initialize WebSocket client for {ex_name}: {e}")
        
        logger.info(f"WebSocketManager initialized with {len(self.clients)} exchanges: {list(self.clients.keys())}")
    
    async def stream_ohlcv(self, 
                          symbols_per_exchange: Dict[str, List[str]],
                          timeframe: str = '1m',
                          callback: Optional[Callable] = None,
                          max_iterations: Optional[int] = None):
        """
        Stream OHLCV data from multiple exchanges simultaneously.
        
        Args:
            symbols_per_exchange: Dict mapping exchange names to lists of symbols
                                 e.g., {'kucoinfutures': ['BTC/USDT:USDT'],
                                        'bingx': ['VST/USDT:USDT', 'ETH/USDT:USDT']}
            timeframe: Timeframe for candles (e.g., '1m', '5m', '30m')
            callback: Optional callback function called for each update
                     Function signature: async def callback(exchange, symbol, timeframe, ohlcv)
            max_iterations: Maximum iterations per symbol (None for infinite)
        
        Returns:
            Dict of tasks for each symbol stream
        """
        self._running = True
        tasks = []
        
        for exchange_name, symbols in symbols_per_exchange.items():
            if exchange_name not in self.clients:
                logger.warning(f"Exchange '{exchange_name}' not initialized, skipping")
                continue
            
            client = self.clients[exchange_name]
            
            for symbol in symbols:
                # Create a callback wrapper that includes exchange info
                if callback:
                    async def wrapped_callback(sym, tf, ohlcv, ex=exchange_name):
                        await callback(ex, sym, tf, ohlcv)
                else:
                    wrapped_callback = None
                
                # Start watch loop for this symbol
                task = asyncio.create_task(
                    client.watch_ohlcv_loop(symbol, timeframe, wrapped_callback, max_iterations)
                )
                tasks.append(task)
                self._tasks.append(task)
                
                logger.info(f"Started OHLCV stream: {exchange_name} {symbol} {timeframe}")
        
        logger.info(f"Started {len(tasks)} OHLCV streams across {len(symbols_per_exchange)} exchanges")
        return tasks
    
    async def stream_tickers(self,
                            symbols_per_exchange: Dict[str, List[str]],
                            callback: Optional[Callable] = None,
                            max_iterations: Optional[int] = None):
        """
        Stream ticker data from multiple exchanges simultaneously.
        
        Args:
            symbols_per_exchange: Dict mapping exchange names to lists of symbols
            callback: Optional callback function called for each update
                     Function signature: async def callback(exchange, symbol, ticker)
            max_iterations: Maximum iterations per symbol (None for infinite)
        
        Returns:
            Dict of tasks for each symbol stream
        """
        self._running = True
        tasks = []
        
        for exchange_name, symbols in symbols_per_exchange.items():
            if exchange_name not in self.clients:
                logger.warning(f"Exchange '{exchange_name}' not initialized, skipping")
                continue
            
            client = self.clients[exchange_name]
            
            for symbol in symbols:
                # Create a callback wrapper that includes exchange info
                if callback:
                    async def wrapped_callback(sym, ticker, ex=exchange_name):
                        await callback(ex, sym, ticker)
                else:
                    wrapped_callback = None
                
                # Start watch loop for this symbol
                task = asyncio.create_task(
                    client.watch_ticker_loop(symbol, wrapped_callback, max_iterations)
                )
                tasks.append(task)
                self._tasks.append(task)
                
                logger.info(f"Started ticker stream: {exchange_name} {symbol}")
        
        logger.info(f"Started {len(tasks)} ticker streams across {len(symbols_per_exchange)} exchanges")
        return tasks
    
    async def run_streams(self, duration: Optional[float] = None):
        """
        Run all active streams.
        
        Args:
            duration: Optional duration in seconds to run streams
                     If None, runs indefinitely until stopped
        """
        if not self._tasks:
            logger.warning("No active streams to run")
            return
        
        logger.info(f"Running {len(self._tasks)} streams" + 
                   (f" for {duration}s" if duration else " indefinitely"))
        
        try:
            if duration:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=duration
                )
            else:
                await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            logger.info(f"Stream duration ({duration}s) completed")
        except Exception as e:
            logger.error(f"Error running streams: {e}")
        finally:
            logger.info("Stream execution completed")
    
    def stop(self):
        """Stop all active streams."""
        self._running = False
        
        for client in self.clients.values():
            client.stop()
        
        logger.info("All streams stopped")
    
    async def close(self):
        """Close all WebSocket connections and cleanup resources."""
        self.stop()
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close all clients
        for client in self.clients.values():
            await client.close()
        
        self._tasks.clear()
        logger.info("WebSocketManager closed")
    
    def get_stream_status(self) -> Dict[str, Any]:
        """
        Get status of all active streams.
        
        Returns:
            Dict with stream status information
        """
        active_tasks = [t for t in self._tasks if not t.done()]
        completed_tasks = [t for t in self._tasks if t.done()]
        
        return {
            'running': self._running,
            'total_streams': len(self._tasks),
            'active_streams': len(active_tasks),
            'completed_streams': len(completed_tasks),
            'exchanges': list(self.clients.keys())
        }


class StreamDataCollector:
    """
    Helper class to collect streaming data into buffers for analysis.
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize data collector.
        
        Args:
            buffer_size: Maximum number of items to keep in each buffer
        """
        self.buffer_size = buffer_size
        self.ohlcv_data: Dict[str, Dict[str, List]] = {}  # exchange -> symbol -> data
        self.ticker_data: Dict[str, Dict[str, List]] = {}  # exchange -> symbol -> data
        logger.info(f"StreamDataCollector initialized with buffer_size={buffer_size}")
    
    async def ohlcv_callback(self, exchange: str, symbol: str, timeframe: str, ohlcv: List):
        """Callback to collect OHLCV data."""
        if exchange not in self.ohlcv_data:
            self.ohlcv_data[exchange] = {}
        
        key = f"{symbol}_{timeframe}"
        if key not in self.ohlcv_data[exchange]:
            self.ohlcv_data[exchange][key] = []
        
        # Add timestamp and store
        self.ohlcv_data[exchange][key].append({
            'timestamp': datetime.now(timezone.utc),
            'data': ohlcv
        })
        
        # Trim buffer if needed
        if len(self.ohlcv_data[exchange][key]) > self.buffer_size:
            self.ohlcv_data[exchange][key] = self.ohlcv_data[exchange][key][-self.buffer_size:]
        
        logger.debug(f"Collected OHLCV: {exchange} {symbol} {timeframe} (buffer: {len(self.ohlcv_data[exchange][key])})")
    
    async def ticker_callback(self, exchange: str, symbol: str, ticker: Dict):
        """Callback to collect ticker data."""
        if exchange not in self.ticker_data:
            self.ticker_data[exchange] = {}
        
        if symbol not in self.ticker_data[exchange]:
            self.ticker_data[exchange][symbol] = []
        
        # Add timestamp and store
        self.ticker_data[exchange][symbol].append({
            'timestamp': datetime.now(timezone.utc),
            'data': ticker
        })
        
        # Trim buffer if needed
        if len(self.ticker_data[exchange][symbol]) > self.buffer_size:
            self.ticker_data[exchange][symbol] = self.ticker_data[exchange][symbol][-self.buffer_size:]
        
        logger.debug(f"Collected ticker: {exchange} {symbol} (buffer: {len(self.ticker_data[exchange][symbol])})")
    
    def get_latest_ohlcv(self, exchange: str, symbol: str, timeframe: str) -> Optional[List]:
        """Get the latest OHLCV data for a symbol."""
        key = f"{symbol}_{timeframe}"
        if exchange in self.ohlcv_data and key in self.ohlcv_data[exchange]:
            buffer = self.ohlcv_data[exchange][key]
            return buffer[-1]['data'] if buffer else None
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
