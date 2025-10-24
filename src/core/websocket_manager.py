"""
WebSocket Manager for multi-exchange real-time data streaming.
Coordinates WebSocket connections across multiple exchanges.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union, Set
from datetime import datetime, timezone
from collections import defaultdict
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
    - Support for CcxtClient-based initialization for better Phase 1 integration
    """
    
    def __init__(self, exchanges: Optional[Union[Dict[str, Dict[str, str]], Dict[str, Any]]] = None, config: Dict[str, Any] = None):
        """
        Initialize WebSocketManager.
        
        Args:
            exchanges: Can be either:
                      1. Dict mapping exchange names to credentials (legacy):
                         {'kucoinfutures': {'apiKey': '...', 'secret': '...'},
                          'bingx': {'apiKey': '...', 'secret': '...'}}
                      2. Dict mapping exchange names to CcxtClient instances (Phase 1 integration):
                         {'kucoinfutures': CcxtClient(...), 'bingx': CcxtClient(...)}
                      If None, creates unauthenticated clients for KuCoin and BingX
            config: Optional configuration dict for WebSocket behavior
        """
        self.clients: Dict[str, WebSocketClient] = {}
        self.exchanges = exchanges or {}
        self.config = config or {}
        self.connections = {}
        self.callbacks = defaultdict(list)
        self.is_running = False
        self.reconnect_delays = {}
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        # Track active streams
        self._active_streams: Dict[str, Set[str]] = defaultdict(set)
        self._stream_limits: Dict[str, int] = {}
        
        # Health monitoring attributes
        self.start_time = time.time()
        self.last_message_time = {}
        self.message_count = 0
        self.reconnection_count = 0
        self.streams = {}  # Track active streams for health monitoring
        
        # ✅ FIX 2: Initialize data collector in __init__ (not in subscribe_to_symbols)
        self._data_collector = StreamDataCollector(buffer_size=100)
        logger.info("✅ StreamDataCollector initialized in __init__")
        
        # Detect if we're using CcxtClient instances or credentials
        self._use_ccxt_clients = False
        if exchanges:
            # Check if any value is a CcxtClient instance
            from .ccxt_client import CcxtClient
            first_value = next(iter(exchanges.values()), None)
            if isinstance(first_value, CcxtClient):
                self._use_ccxt_clients = True
                logger.info("Initializing with CcxtClient instances (Phase 1 integration)")
        
        if exchanges is None:
            # Default to unauthenticated clients
            exchanges = {
                'kucoinfutures': None,
                'bingx': None
            }
        
        # Initialize WebSocket clients based on input type
        for ex_name, ex_data in exchanges.items():
            try:
                ex_name_lower = ex_name.lower()
                
                # Determine which WebSocket client to use
                client_cls = WebSocketClient  # Default
                
                if ex_name_lower == 'bingx':
                    logger.info("🎯 Initializing BingX WebSocket client")
                    
                    # Try dedicated BingX client if available
                    try:
                        from .websocket_client_bingx import WebSocketClient as BingxClient
                        client_cls = BingxClient
                        logger.info("✅ Using dedicated websocket_client_bingx.py")
                    except ImportError:
                        # websocket_client.py will handle BingX automatically
                        logger.info("✅ Using websocket_client.py (auto-detects BingX mode)")
                
                # Extract credentials based on source type
                creds = None
                if self._use_ccxt_clients:
                    from .ccxt_client import CcxtClient
                    if isinstance(ex_data, CcxtClient):
                        # Get credentials from the CcxtClient's exchange object
                        if hasattr(ex_data.ex, 'apiKey') and ex_data.ex.apiKey:
                            creds = {
                                'apiKey': ex_data.ex.apiKey,
                                'secret': ex_data.ex.secret
                            }
                            if hasattr(ex_data.ex, 'password') and ex_data.ex.password:
                                creds['password'] = ex_data.ex.password
                else:
                    creds = ex_data
                
                # ✅ PATCH 3: Pass collector to WebSocket client
                self.clients[ex_name_lower] = client_cls(ex_name_lower, creds, collector=self._data_collector)
                logger.info(f"✅ WebSocket client initialized for {ex_name_lower} with collector")
                
            except Exception as e:
                logger.error(f"Failed to initialize WebSocket client for {ex_name}: {e}")
        
    def start_ohlcv_stream(self, exchange: str, symbol: str, timeframe: str) -> bool:
        """
        Start OHLCV stream for a specific symbol (PRODUCTION COORDINATOR COMPAT).
        
        Args:
            exchange: Exchange name (e.g., 'bingx')
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '5m')
            
        Returns:
            bool: True if stream started, False otherwise
        """
        try:
            # Check stream limit
            current_streams = len(self._active_streams[exchange])
            max_streams = self._stream_limits.get(exchange, 10)
            
            if current_streams >= max_streams:
                logger.warning(f"[{exchange}] Stream limit reached ({max_streams}), skipping {symbol}")
                return False
            
            # Track stream
            stream_key = f"{symbol}_{timeframe}"
            if stream_key in self._active_streams[exchange]:
                logger.debug(f"Stream already active: {exchange} {symbol} {timeframe}")
                return True
                
            self._active_streams[exchange].add(stream_key)
            logger.info(f"Started OHLCV stream: {exchange} {symbol} {timeframe}")
            
            # Note: Actual WebSocket connection would be handled by WebSocketClient
            # This is a tracking layer for production coordinator
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream {exchange} {symbol} {timeframe}: {e}")
            return False
    
    def _make_ohlcv_wrapper(self, client, symbol, timeframe, callback, max_iterations, iteration_delay=1.0):
        """
        Return a coroutine that repeatedly calls client's watch_ohlcv for compatibility
        when client does not provide watch_ohlcv_loop.
        """
        async def _wrapper():
            iterations = 0
            while self._running:
                if max_iterations is not None and iterations >= max_iterations:
                    break
                try:
                    ohlcv = await client.watch_ohlcv(symbol, timeframe, callback=None)
                    if ohlcv and callback:
                        try:
                            await callback(symbol, timeframe, ohlcv)
                        except Exception as cb_e:
                            logger.exception(f"Error in OHLCV wrapper callback for {symbol}: {cb_e}")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(f"OHLCV wrapper error for {symbol} on {getattr(client, 'name', 'unknown')}: {e}")
                    await asyncio.sleep(getattr(client, 'reconnect_delay', 1))
                iterations += 1
                await asyncio.sleep(iteration_delay)
            logger.info(f"OHLCV wrapper stopped for {symbol} on {getattr(client, 'name', 'unknown')}")
        return _wrapper
    
    def _make_ticker_wrapper(self, client, symbol, callback, max_iterations, iteration_delay=1.0):
        """
        Return a coroutine that repeatedly calls client's watch_ticker when ticker_loop is absent.
        """
        async def _wrapper():
            iterations = 0
            while self._running:
                if max_iterations is not None and iterations >= max_iterations:
                    break
                try:
                    ticker = await client.watch_ticker(symbol, callback=None)
                    if ticker and callback:
                        try:
                            await callback(symbol, ticker)
                        except Exception as cb_e:
                            logger.exception(f"Error in ticker wrapper callback for {symbol}: {cb_e}")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(f"Ticker wrapper error for {symbol} on {getattr(client, 'name', 'unknown')}: {e}")
                    await asyncio.sleep(getattr(client, 'reconnect_delay', 1))
                iterations += 1
                await asyncio.sleep(iteration_delay)
            logger.info(f"Ticker wrapper stopped for {symbol} on {getattr(client, 'name', 'unknown')}")
        return _wrapper
    
    def subscribe_to_symbols(self, symbols: List[str], timeframes: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Subscribe to multiple symbols across exchanges (PRODUCTION COORDINATOR COMPAT).
        
        Args:
            symbols: List of symbols to subscribe to
            timeframes: Optional list of timeframes (default: ['1m', '5m', '30m', '1h', '4h'])
            
        Returns:
            Dict mapping exchange names to list of successfully subscribed symbols
        """
        if not timeframes:
            timeframes = self.config.get('websocket', {}).get('stream_timeframes', ['1m', '5m'])
        
        subscribed = defaultdict(list)
        
        # Distribute symbols across available exchanges
        exchanges = list(self.clients.keys())
        if not exchanges:
            logger.error("No exchanges available for subscription")
            return dict(subscribed)
        
        for symbol in symbols:
            # Round-robin distribution
            for exchange in exchanges:
                success = True
                for timeframe in timeframes:
                    if not self.start_ohlcv_stream(exchange, symbol, timeframe):
                        success = False
                        break
                        
                if success:
                    subscribed[exchange].append(symbol)
                    break  # Symbol subscribed, move to next
        
        total_subscribed = sum(len(syms) for syms in subscribed.values())
        logger.info(f"Subscribed to {total_subscribed} symbols across {len(subscribed)} exchanges")
        # ✅ FIX 2: Data collector now initialized in __init__ (removed duplicate)
        
        return dict(subscribed)
    
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
                # Check stream limit
                if not self.start_ohlcv_stream(exchange_name, symbol, timeframe):
                    continue
                
                # Create a callback wrapper that includes exchange info AND data collector
                async def wrapped_callback(sym, tf, ohlcv, ex=exchange_name):
                    # Store in data collector
                    if hasattr(self, '_data_collector'):
                        await self._data_collector.ohlcv_callback(ex, sym, tf, ohlcv)
                    # Call user callback if provided
                    if callback:
                        await callback(ex, sym, tf, ohlcv)
                
                # Prefer watch_ohlcv_loop if available, otherwise use wrapper
                if hasattr(client, 'watch_ohlcv_loop'):
                    task = asyncio.create_task(
                        client.watch_ohlcv_loop(symbol, timeframe, wrapped_callback, max_iterations)
                    )
                else:
                    # Use wrapper for clients that don't have watch_ohlcv_loop
                    wrapper_coro = self._make_ohlcv_wrapper(client, symbol, timeframe, wrapped_callback, max_iterations)
                    task = asyncio.create_task(wrapper_coro())
                
                tasks.append(task)
                self._tasks.append(task)
                
                logger.info(f"Started OHLCV stream task: {exchange_name} {symbol} {timeframe}")
        
        logger.info(f"Started {len(tasks)} OHLCV stream tasks across {len(symbols_per_exchange)} exchanges")
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
                
                # Prefer watch_ticker_loop if available, otherwise use wrapper
                if hasattr(client, 'watch_ticker_loop'):
                    task = asyncio.create_task(
                        client.watch_ticker_loop(symbol, wrapped_callback, max_iterations)
                    )
                else:
                    # Use wrapper for clients that don't have watch_ticker_loop
                    wrapper_coro = self._make_ticker_wrapper(client, symbol, wrapped_callback, max_iterations)
                    task = asyncio.create_task(wrapper_coro())
                
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
        self._active_streams.clear()
        logger.info("WebSocketManager closed")
    
    def get_stream_status(self) -> Dict[str, Any]:
        """
        Get status of all active streams.
        
        Returns:
            Dict with stream status information
        """
        active_tasks = [t for t in self._tasks if not t.done()]
        completed_tasks = [t for t in self._tasks if t.done()]
        
        # Count total active streams
        total_streams = sum(len(streams) for streams in self._active_streams.values())
        
        return {
            'running': self._running,
            'total_streams': total_streams,
            'active_streams': len(active_tasks),
            'completed_streams': len(completed_tasks),
            'exchanges': list(self.clients.keys()),
            'stream_distribution': {
                exchange: len(streams) 
                for exchange, streams in self._active_streams.items()
            }
        }

    def get_latest_data(self, symbol: str, timeframe: str = '1m') -> Optional[Dict[str, Any]]:
        """
        Get latest cached data for a symbol (PRODUCTION COORDINATOR COMPAT).
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe (e.g., '1m', '5m')
            
        Returns:
            Dict with latest OHLCV data or None if not available
        """
        # ✅ FIX 2: Data collector now always initialized in __init__
        # No need for defensive check, but keep for backward compatibility
        if not hasattr(self, '_data_collector'):
            logger.warning("Data collector not initialized - this should not happen!")
            self._data_collector = StreamDataCollector(buffer_size=100)
            
        # Get from any available exchange
        for exchange in self._active_streams.keys():
            # Try to get cached data
            latest_ohlcv = self._data_collector.get_latest_ohlcv(exchange, symbol, timeframe)
            if latest_ohlcv:
                return {
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'ohlcv': latest_ohlcv,
                    'timestamp': datetime.now(timezone.utc)
                }
                
        # No cached data available
        logger.debug(f"No cached data for {symbol} {timeframe}")
        return None
    
    def get_connection_health(self):
        """Get WebSocket connection health."""
        uptime = time.time() - self.start_time
        if not self.streams:
            status = 'disconnected'
        elif self.message_count == 0:
            status = 'connecting'
        elif self.message_count < 10:
            status = 'warming_up'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'active_streams': len(self.streams),
            'symbols': list(self.streams.keys()),
            'total_messages': self.message_count,
            'uptime_seconds': uptime,
            'reconnection_count': self.reconnection_count
        }
    
    def is_data_fresh(self, symbol: str, timeframe: str, max_age_seconds: int = 60):
        """Check if data is fresh enough."""
        data = self.get_latest_data(symbol, timeframe)
        if not data or 'ohlcv' not in data or not data['ohlcv']:
            return False
        last_candle_time = data['ohlcv'][-1][0]
        current_time = time.time() * 1000
        age = (current_time - last_candle_time) / 1000
        return age < max_age_seconds
    
    def get_data_quality_score(self, symbol: str, timeframe: str):
        """Calculate data quality score (0-100)."""
        data = self.get_latest_data(symbol, timeframe)
        if not data or 'ohlcv' not in data:
            return 0.0
        
        score = 0.0
        # Freshness (40%)
        if self.is_data_fresh(symbol, timeframe, 60):
            score += 40
        elif self.is_data_fresh(symbol, timeframe, 120):
            score += 20
        # Completeness (30%)
        if len(data['ohlcv']) >= 100:
            score += 30
        elif len(data['ohlcv']) >= 50:
            score += 15
        # Update frequency (30%)
        if symbol in self.last_message_time:
            if time.time() - self.last_message_time[symbol] < 5:
                score += 30
            elif time.time() - self.last_message_time[symbol] < 15:
                score += 15
        return score
    
    async def subscribe_tickers(self, symbols: List[str], callback=None):
        """
        Subscribe to real-time ticker updates.
        
        This is a convenience method that distributes symbols across available exchanges
        and sets up ticker streaming with optional callback.
        
        Args:
            symbols: List of trading pair symbols (e.g., ['BTC/USDT:USDT', 'ETH/USDT:USDT'])
            callback: Optional callback function for ticker updates
                     Signature: async def callback(exchange, symbol, ticker)
        
        Returns:
            List of tasks for each ticker stream
        """
        if callback:
            self.callbacks['ticker'].append(callback)
        
        # Distribute symbols across available exchanges
        symbols_per_exchange = {}
        available_exchanges = list(self.clients.keys())
        
        for i, symbol in enumerate(symbols):
            # Round-robin distribution across exchanges
            exchange = available_exchanges[i % len(available_exchanges)]
            if exchange not in symbols_per_exchange:
                symbols_per_exchange[exchange] = []
            symbols_per_exchange[exchange].append(symbol)
        
        logger.info(f"Subscribing to tickers for {len(symbols)} symbols across {len(symbols_per_exchange)} exchanges")
        
        # Use existing stream_tickers method
        return await self.stream_tickers(
            symbols_per_exchange,
            callback=callback
        )
    
    async def subscribe_orderbook(self, symbol: str, depth: int = 20, callback=None):
        """
        Subscribe to L2 orderbook streams.
        
        Note: This is a placeholder for future orderbook support.
        Current implementation uses ticker data as proxy.
        
        Args:
            symbol: Trading pair symbol
            depth: Orderbook depth (default: 20)
            callback: Optional callback function for orderbook updates
        
        Returns:
            Task for the orderbook stream
        """
        logger.warning(f"Orderbook streaming not yet fully implemented, using ticker stream for {symbol}")
        
        if callback:
            self.callbacks['orderbook'].append(callback)
        
        # Use ticker streaming as a proxy until full orderbook support is added
        return await self.subscribe_tickers([symbol], callback)
    
    async def start_streams(self, subscriptions: Dict[str, List[str]]):
        """
        Start WebSocket streams for specified subscriptions.
        
        Args:
            subscriptions: Dict with subscription types and symbols
                          e.g., {'tickers': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
                                 'ohlcv': ['BTC/USDT:USDT']}
        
        Returns:
            Dict mapping subscription types to their tasks
        """
        self.is_running = True
        self._running = True
        stream_tasks = {}
        
        # Start ticker streams
        if 'tickers' in subscriptions:
            ticker_symbols = subscriptions['tickers']
            callback = self.callbacks['ticker'][0] if self.callbacks['ticker'] else None
            tasks = await self.subscribe_tickers(ticker_symbols, callback)
            stream_tasks['tickers'] = tasks
            logger.info(f"Started {len(tasks)} ticker streams")
        
        # Start OHLCV streams
        if 'ohlcv' in subscriptions:
            ohlcv_symbols = subscriptions['ohlcv']
            timeframe = subscriptions.get('timeframe', '1m')
            
            # Distribute symbols across exchanges
            symbols_per_exchange = {}
            available_exchanges = list(self.clients.keys())
            
            for i, symbol in enumerate(ohlcv_symbols):
                exchange = available_exchanges[i % len(available_exchanges)]
                if exchange not in symbols_per_exchange:
                    symbols_per_exchange[exchange] = []
                symbols_per_exchange[exchange].append(symbol)
            
            callback = self.callbacks['ohlcv'][0] if self.callbacks['ohlcv'] else None
            tasks = await self.stream_ohlcv(symbols_per_exchange, timeframe, callback)
            stream_tasks['ohlcv'] = tasks
            logger.info(f"Started {len(tasks)} OHLCV streams")
        
        # Start orderbook streams (if supported)
        if 'orderbook' in subscriptions:
            logger.warning("Orderbook streams not yet fully implemented")
        
        logger.info(f"Started {sum(len(tasks) for tasks in stream_tasks.values())} total streams")
        return stream_tasks
    
    def on_ticker_update(self, callback):
        """
        Register callback for ticker updates.
        
        Args:
            callback: Async function with signature: async def callback(exchange, symbol, ticker)
        
        Returns:
            self (for method chaining)
        """
        self.callbacks['ticker'].append(callback)
        logger.info("Registered ticker update callback")
        return self
    
    def on_orderbook_update(self, callback):
        """
        Register callback for orderbook updates.
        
        Args:
            callback: Async function with signature: async def callback(exchange, symbol, orderbook)
        
        Returns:
            self (for method chaining)
        """
        self.callbacks['orderbook'].append(callback)
        logger.info("Registered orderbook update callback")
        return self
    
    async def shutdown(self):
        """
        Graceful shutdown of all WebSocket connections.
        
        This is an alias for the existing close() method to match the API in problem statement.
        """
        logger.info("Initiating graceful shutdown of WebSocket connections")
        self.is_running = False
        await self.close()


class OptimizedWebSocketManager(WebSocketManager):
    """
    Optimized WebSocket Manager with production features.
    
    Additional features:
    - Fixed symbol support without market loading
    - Per-exchange stream limits
    - Automatic stream optimization
    """
    
    def __init__(self, exchanges: Optional[Dict[str, Any]] = None, 
                 config: Dict[str, Any] = None,
                 max_streams_per_exchange: int = 10):
        """
        Initialize optimized WebSocket manager.
        
        Args:
            exchanges: Exchange clients dictionary
            config: Configuration dictionary
            max_streams_per_exchange: Default max streams per exchange
        """
        super().__init__(exchanges, config)
        
        self.fixed_symbols: List[str] = []
        self.optimized = True
        self.default_max_streams = max_streams_per_exchange
        
        # Override stream limits with more conservative defaults
        for exchange in self.clients.keys():
            if exchange not in self._stream_limits:
                self._stream_limits[exchange] = self.default_max_streams
                
        logger.info(f"[WS-OPT] Optimized WebSocket Manager initialized")
        logger.info(f"[WS-OPT] Stream limits: {self._stream_limits}")
    
    def set_fixed_symbols(self, symbols: List[str]):
        """
        Set fixed symbols for optimized streaming.
        
        Args:
            symbols: List of symbols to stream
        """
        self.fixed_symbols = symbols
        logger.info(f"[WS-OPT] Configured with {len(symbols)} fixed symbols")
    
    def subscribe_optimized(self) -> Dict[str, int]:
        """
        Subscribe to fixed symbols with optimization.
        
        Returns:
            Dict mapping exchange names to number of subscribed streams
        """
        if not self.fixed_symbols:
            logger.warning("[WS-OPT] No fixed symbols configured")
            return {}
        
        result = {}
        timeframes = self.config.get('websocket', {}).get('stream_timeframes', ['1m', '5m', '30m', '1h', '4h'])
        
        for exchange in self.clients.keys():
            max_streams = self._stream_limits.get(exchange, self.default_max_streams)
            streams_per_symbol = len(timeframes)
            max_symbols = max_streams // streams_per_symbol
            
            symbols_to_subscribe = self.fixed_symbols[:max_symbols]
            
            for symbol in symbols_to_subscribe:
                for timeframe in timeframes:
                    self.start_ohlcv_stream(exchange, symbol, timeframe)
            
            result[exchange] = len(symbols_to_subscribe) * streams_per_symbol
            logger.info(f"[WS-OPT] {exchange}: Subscribed to {len(symbols_to_subscribe)} symbols "
                       f"({result[exchange]} total streams)")
        
        logger.info(f"[WS-OPT] Total streams: {sum(result.values())}")
        return result
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get optimization status and statistics.
        
        Returns:
            Dict with optimization metrics
        """
        total_possible = len(self.fixed_symbols) * len(self.clients) * 2  # 2 timeframes
        total_active = sum(len(streams) for streams in self._active_streams.values())
        
        return {
            'optimized': True,
            'fixed_symbols': len(self.fixed_symbols),
            'total_possible_streams': total_possible,
            'total_active_streams': total_active,
            'optimization_ratio': total_active / total_possible if total_possible > 0 else 0,
            'stream_limits': self._stream_limits,
            'active_distribution': {
                exchange: len(streams)
                for exchange, streams in self._active_streams.items()
            }
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
