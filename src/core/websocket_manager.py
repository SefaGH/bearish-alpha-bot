"""
WebSocket Manager for multi-exchange real-time data streaming.
REFACTORED: This manager now delegates directly to dedicated exchange clients
like websocket_client_bingx.py, simplifying its responsibilities.

Author: SefaGH
Date: 2025-10-29
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
from .stream_data_collector import StreamDataCollector

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Multi-Exchange WebSocket Manager for coordinated real-time streaming.
    This version is simplified to use dedicated clients per exchange.
    """
    
    def __init__(self, exchanges: Optional[Dict[str, Any]] = None, config: Dict[str, Any] = None):
        """Initializes the WebSocketManager with dedicated clients."""
        self.clients: Dict[str, Any] = {}
        self.config = config or {}
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        # StreamDataCollector is integral to this manager.
        self._data_collector = StreamDataCollector(config=self.config)
        logger.info("StreamDataCollector initialized within WebSocketManager")
        
        exchanges = exchanges or {}

        for ex_name, ex_data in exchanges.items():
            try:
                ex_name_lower = ex_name.lower()
                
                # This manager now only supports exchanges with dedicated clients.
                if ex_name_lower == 'bingx':
                    from .websocket_client_bingx import WebSocketClient as BingxClient
                    
                    creds = None
                    from .ccxt_client import CcxtClient
                    if isinstance(ex_data, CcxtClient) and hasattr(ex_data.ex, 'apiKey') and ex_data.ex.apiKey:
                        creds = {'apiKey': ex_data.ex.apiKey, 'secret': ex_data.ex.secret}

                    # The collector is passed directly to the client.
                    self.clients[ex_name_lower] = BingxClient(ex_name_lower, creds, collector=self._data_collector)
                    logger.info(f"✅ Dedicated BingX WebSocket client initialized for '{ex_name_lower}'")
                else:
                    logger.warning(f"Exchange '{ex_name}' is not supported in this simplified manager.")
            
            except Exception as e:
                logger.error(f"Failed to initialize WebSocket client for {ex_name}: {e}", exc_info=True)

    @property
    def collector(self):
        """Public property to access the data collector."""
        return self._data_collector
    
    def is_collector_ready(self) -> bool:
        """Checks if the data collector is initialized and ready."""
        return self.collector is not None

    async def stream_ohlcv(self, 
                          symbols_per_exchange: Dict[str, List[str]],
                          timeframe: str = '1m',
                          callback: Optional[Callable] = None,
                          max_iterations: Optional[int] = None) -> List[asyncio.Task]:
        """
        Streams OHLCV data by delegating to the client's `watch_ohlcv_loop`.
        """
        self._running = True
        tasks = []
        
        for exchange_name, symbols in symbols_per_exchange.items():
            client = self.clients.get(exchange_name.lower())
            if not client:
                logger.warning(f"Exchange '{exchange_name}' not initialized, skipping.")
                continue
            
            for symbol in symbols:
                # Directly use the efficient `watch_ohlcv_loop`. No wrappers needed.
                task = asyncio.create_task(
                    client.watch_ohlcv_loop(symbol, timeframe, callback, max_iterations)
                )
                tasks.append(task)
                self._tasks.append(task)
                logger.info(f"Created OHLCV stream task: {exchange_name} {symbol} [{timeframe}]")
        
        logger.info(f"Created {len(tasks)} OHLCV stream tasks.")
        return tasks

    async def stream_tickers(self,
                            symbols_per_exchange: Dict[str, List[str]],
                            callback: Optional[Callable] = None,
                            max_iterations: Optional[int] = None) -> List[asyncio.Task]:
        """Streams ticker data by delegating to the client's `watch_ticker_loop`."""
        self._running = True
        tasks = []
        
        for exchange_name, symbols in symbols_per_exchange.items():
            client = self.clients.get(exchange_name.lower())
            if not client:
                logger.warning(f"Exchange '{exchange_name}' not initialized, skipping.")
                continue

            for symbol in symbols:
                task = asyncio.create_task(
                    client.watch_ticker_loop(symbol, callback, max_iterations)
                )
                tasks.append(task)
                self._tasks.append(task)
                logger.info(f"Created Ticker stream task: {exchange_name} {symbol}")
        
        logger.info(f"Created {len(tasks)} Ticker stream tasks.")
        return tasks
    
    async def close(self):
        """Closes all WebSocket connections and cleans up resources."""
        self._running = False
        
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        for client in self.clients.values():
            if hasattr(client, 'close'):
                await client.close()
        
        self._tasks.clear()
        logger.info("WebSocketManager closed successfully.")

    def get_latest_data(self, symbol: str, timeframe: str = '1m', exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Gets latest cached data for a symbol from the data collector."""
        if not self.collector:
            logger.warning("Data collector not available.")
            return None
        
        target_exchanges = [exchange.lower()] if exchange else list(self.clients.keys())
        
        for ex_name in target_exchanges:
            latest_ohlcv = self.collector.get_latest_ohlcv(ex_name, symbol, timeframe)
            if latest_ohlcv:
                return {
                    'exchange': ex_name,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'ohlcv': latest_ohlcv,
                    'timestamp': datetime.now(timezone.utc)
                }
        
        logger.debug(f"No cached data for {symbol} [{timeframe}] on exchanges: {target_exchanges}")
        return None

    def get_stream_status(self) -> Dict[str, Any]:
        """Gets the status of all managed streams."""
        active_tasks = [t for t in self._tasks if not t.done()]
        return {
            'running': self._running,
            'total_managed_tasks': len(self._tasks),
            'active_tasks': len(active_tasks),
            'exchanges': list(self.clients.keys()),
        }

    def get_connection_health(self) -> Dict[str, Any]:
        """Gets connection health from underlying clients."""
        health_reports = {}
        for name, client in self.clients.items():
            if hasattr(client, 'get_health_status'):
                health_reports[name] = client.get_health_status()
            else:
                health_reports[name] = {'status': 'unknown', 'reason': 'get_health_status not implemented'}
        return health_reports

    async def shutdown(self):
        """Alias for close() for graceful shutdown."""
        await self.close()
