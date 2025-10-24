"""
BingX Direct WebSocket Implementation
No CCXT Pro required - uses native BingX WebSocket API
"""

import json
import asyncio
import logging
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
import websockets
from collections import defaultdict

logger = logging.getLogger(__name__)


class BingXWebSocket:
    """
    Direct BingX WebSocket client for real-time market data.
    
    Features:
    - Real-time ticker updates
    - OHLCV/Kline streaming
    - Order book updates
    - Automatic reconnection
    - No CCXT Pro dependency
    """
    
    # BingX WebSocket endpoints
    WS_PUBLIC_SPOT = "wss://open-api-ws.bingx.com/market"
    WS_PUBLIC_SWAP = "wss://open-api-swap.bingx.com/swap-market"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 testnet: bool = False, futures: bool = True):
        """
        Initialize BingX WebSocket client.
        
        Args:
            api_key: Optional API key for authenticated endpoints
            api_secret: Optional API secret
            testnet: Use testnet endpoints
            futures: Use futures/swap market (True) or spot (False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.futures = futures
        
        # Select appropriate endpoint
        self.ws_url = self.WS_PUBLIC_SWAP if futures else self.WS_PUBLIC_SPOT
        if testnet:
            self.ws_url = self.ws_url.replace("bingx.com", "bingx.com")  # Testnet same URL
            
        # Connection management
        self.ws = None
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds
        
        # Data storage
        self.tickers = {}
        self.orderbooks = {}
        self.klines = defaultdict(dict)  # symbol -> timeframe -> data
        
        # Callbacks
        self.callbacks = {
            'ticker': [],
            'orderbook': [],
            'kline': []
        }
        
        # Subscription tracking
        self.subscriptions = set()
        
        # Statistics
        self.message_count = 0
        self.last_message_time = None
        self.connection_start_time = None
        
        logger.info(f"BingX WebSocket initialized ({'futures' if futures else 'spot'} market)")
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection.
        
        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to BingX WebSocket: {self.ws_url}")
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self._running = True
            self._reconnect_attempts = 0
            self.connection_start_time = datetime.now(timezone.utc)
            
            logger.info("✅ BingX WebSocket connected successfully")
            
            # Re-subscribe to previous subscriptions after reconnect
            if self.subscriptions:
                logger.info(f"Re-subscribing to {len(self.subscriptions)} channels...")
                for sub_msg in self.subscriptions:
                    await self.ws.send(sub_msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to BingX WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect WebSocket connection."""
        self._running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        logger.info("BingX WebSocket disconnected")
    
    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        Subscribe to ticker updates for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USDT' for BingX format)
            
        Returns:
            True if subscription successful
        """
        try:
            # Convert CCXT format (BTC/USDT:USDT) to BingX format (BTC-USDT)
            bingx_symbol = self._convert_symbol_to_bingx(symbol)
            
            sub_message = {
                "id": str(int(time.time() * 1000)),
                "reqType": "sub",
                "dataType": f"{bingx_symbol}@ticker"
            }
            
            sub_msg_str = json.dumps(sub_message)
            self.subscriptions.add(sub_msg_str)
            
            if self.ws:
                await self.ws.send(sub_msg_str)
                logger.info(f"Subscribed to ticker: {bingx_symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe to ticker {symbol}: {e}")
            return False
    
    async def subscribe_kline(self, symbol: str, interval: str = "1m") -> bool:
        """
        Subscribe to kline/candlestick updates.
        
        Args:
            symbol: Trading pair
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            
        Returns:
            True if subscription successful
        """
        try:
            bingx_symbol = self._convert_symbol_to_bingx(symbol)
            
            # Convert timeframe to BingX format
            bingx_interval = self._convert_timeframe(interval)
            
            sub_message = {
                "id": str(int(time.time() * 1000)),
                "reqType": "sub", 
                "dataType": f"{bingx_symbol}@kline_{bingx_interval}"
            }
            
            sub_msg_str = json.dumps(sub_message)
            self.subscriptions.add(sub_msg_str)
            
            if self.ws:
                await self.ws.send(sub_msg_str)
                logger.info(f"Subscribed to kline: {bingx_symbol} {bingx_interval}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe to kline {symbol} {interval}: {e}")
            return False
    
    async def subscribe_orderbook(self, symbol: str, depth: int = 20) -> bool:
        """
        Subscribe to order book updates.
        
        Args:
            symbol: Trading pair
            depth: Order book depth (5, 10, 20, 50, 100)
            
        Returns:
            True if subscription successful
        """
        try:
            bingx_symbol = self._convert_symbol_to_bingx(symbol)
            
            # BingX depth levels: depth5, depth10, depth20, depth50, depth100
            if depth not in [5, 10, 20, 50, 100]:
                depth = 20  # Default
            
            sub_message = {
                "id": str(int(time.time() * 1000)),
                "reqType": "sub",
                "dataType": f"{bingx_symbol}@depth{depth}"
            }
            
            sub_msg_str = json.dumps(sub_message)
            self.subscriptions.add(sub_msg_str)
            
            if self.ws:
                await self.ws.send(sub_msg_str)
                logger.info(f"Subscribed to orderbook: {bingx_symbol} depth{depth}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe to orderbook {symbol}: {e}")
            return False
    
    async def listen(self):
        """
        Main listening loop for WebSocket messages.
        
        Handles incoming messages and calls registered callbacks.
        """
        while self._running:
            try:
                if not self.ws:
                    logger.warning("WebSocket not connected, attempting to reconnect...")
                    if not await self._reconnect():
                        await asyncio.sleep(self._reconnect_delay)
                        continue
                
                # Receive message
                message = await self.ws.recv()
                await self._handle_message(message)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self.ws = None
                await self._reconnect()
                
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: str):
        """
        Process incoming WebSocket message.
        
        Args:
            message: Raw message string from WebSocket
        """
        try:
            data = json.loads(message)
            
            # Update statistics
            self.message_count += 1
            self.last_message_time = datetime.now(timezone.utc)
            
            # Check if it's a ping/pong message
            if data.get("ping"):
                await self._send_pong(data["ping"])
                return
            
            # Check message type
            if "code" in data and data["code"] != 0:
                logger.error(f"BingX error response: {data}")
                return
            
            data_type = data.get("dataType", "")
            
            if "@ticker" in data_type:
                await self._handle_ticker(data)
            elif "@kline" in data_type:
                await self._handle_kline(data)
            elif "@depth" in data_type:
                await self._handle_orderbook(data)
            else:
                logger.debug(f"Unknown message type: {data_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_ticker(self, data: Dict):
        """Process ticker update."""
        try:
            ticker_data = data.get("data", {})
            if not ticker_data:
                return
            
            # Extract symbol from dataType (e.g., "BTC-USDT@ticker")
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol:
                return
            
            # Convert BingX ticker format to standard format
            ticker = {
                'symbol': self._convert_symbol_from_bingx(symbol),
                'last': float(ticker_data.get('c', 0)),  # Last price
                'bid': float(ticker_data.get('b', 0)),   # Best bid
                'ask': float(ticker_data.get('a', 0)),   # Best ask
                'high': float(ticker_data.get('h', 0)),  # 24h high
                'low': float(ticker_data.get('l', 0)),   # 24h low
                'volume': float(ticker_data.get('v', 0)), # 24h volume
                'timestamp': data.get('ts', int(time.time() * 1000))
            }
            
            # Store ticker
            self.tickers[ticker['symbol']] = ticker
            
            # Call callbacks
            for callback in self.callbacks['ticker']:
                await callback(ticker['symbol'], ticker)
                
        except Exception as e:
            logger.error(f"Error handling ticker: {e}")
    
    async def _handle_kline(self, data: Dict):
        """Process kline/candlestick update."""
        try:
            kline_data = data.get("data", {})
            if not kline_data:
                return
            
            # Extract symbol and timeframe
            data_type = data.get("dataType", "")
            parts = data_type.split("@")
            if len(parts) != 2:
                return
            
            symbol = parts[0]
            timeframe_part = parts[1].replace("kline_", "")
            
            # Convert to standard format
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            ccxt_timeframe = self._convert_timeframe_from_bingx(timeframe_part)
            
            # Format: [timestamp, open, high, low, close, volume]
            kline = [
                kline_data.get('t', 0),           # timestamp
                float(kline_data.get('o', 0)),    # open
                float(kline_data.get('h', 0)),    # high
                float(kline_data.get('l', 0)),    # low
                float(kline_data.get('c', 0)),    # close
                float(kline_data.get('v', 0))     # volume
            ]
            
            # Store kline
            if ccxt_symbol not in self.klines:
                self.klines[ccxt_symbol] = {}
            if ccxt_timeframe not in self.klines[ccxt_symbol]:
                self.klines[ccxt_symbol][ccxt_timeframe] = []
            
            # Append and keep last 500 candles
            self.klines[ccxt_symbol][ccxt_timeframe].append(kline)
            if len(self.klines[ccxt_symbol][ccxt_timeframe]) > 500:
                self.klines[ccxt_symbol][ccxt_timeframe] = self.klines[ccxt_symbol][ccxt_timeframe][-500:]
            
            # Call callbacks
            for callback in self.callbacks['kline']:
                await callback(ccxt_symbol, ccxt_timeframe, [kline])
                
        except Exception as e:
            logger.error(f"Error handling kline: {e}")
    
    async def _handle_orderbook(self, data: Dict):
        """Process orderbook update."""
        try:
            ob_data = data.get("data", {})
            if not ob_data:
                return
            
            # Extract symbol
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol:
                return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            # Format orderbook
            orderbook = {
                'symbol': ccxt_symbol,
                'bids': [[float(p), float(q)] for p, q in ob_data.get('bids', [])],
                'asks': [[float(p), float(q)] for p, q in ob_data.get('asks', [])],
                'timestamp': data.get('ts', int(time.time() * 1000))
            }
            
            # Store orderbook
            self.orderbooks[ccxt_symbol] = orderbook
            
            # Call callbacks
            for callback in self.callbacks['orderbook']:
                await callback(ccxt_symbol, orderbook)
                
        except Exception as e:
            logger.error(f"Error handling orderbook: {e}")
    
    async def _send_pong(self, ping_id):
        """Send pong response to keep connection alive."""
        pong_message = {"pong": ping_id}
        if self.ws:
            await self.ws.send(json.dumps(pong_message))
    
    async def _reconnect(self) -> bool:
        """
        Attempt to reconnect to WebSocket.
        
        Returns:
            True if reconnection successful
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self._max_reconnect_attempts}) reached")
            return False
        
        self._reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")
        
        # Exponential backoff
        delay = min(60, self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)))
        await asyncio.sleep(delay)
        
        return await self.connect()
    
    def _convert_symbol_to_bingx(self, ccxt_symbol: str) -> str:
        """
        Convert CCXT symbol format to BingX format.
        
        Args:
            ccxt_symbol: CCXT format (e.g., 'BTC/USDT:USDT' or 'BTC/USDT')
            
        Returns:
            BingX format (e.g., 'BTC-USDT')
        """
        # Remove settlement currency for futures
        if ':' in ccxt_symbol:
            ccxt_symbol = ccxt_symbol.split(':')[0]
        
        # Replace / with -
        return ccxt_symbol.replace('/', '-')
    
    def _convert_symbol_from_bingx(self, bingx_symbol: str) -> str:
        """
        Convert BingX symbol format to CCXT format.
        
        Args:
            bingx_symbol: BingX format (e.g., 'BTC-USDT')
            
        Returns:
            CCXT format (e.g., 'BTC/USDT:USDT' for futures)
        """
        base_symbol = bingx_symbol.replace('-', '/')
        
        # Add settlement currency for futures
        if self.futures and 'USDT' in base_symbol:
            return f"{base_symbol}:USDT"
        
        return base_symbol
    
    def _convert_timeframe(self, ccxt_tf: str) -> str:
        """Convert CCXT timeframe to BingX format."""
        mapping = {
            '1m': '1m',
            '3m': '3m', 
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        return mapping.get(ccxt_tf, '1m')
    
    def _convert_timeframe_from_bingx(self, bingx_tf: str) -> str:
        """Convert BingX timeframe to CCXT format."""
        # Same format for most timeframes
        return bingx_tf
    
    # Public data access methods
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get latest ticker data for a symbol."""
        return self.tickers.get(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get latest orderbook for a symbol."""
        return self.orderbooks.get(symbol)
    
    def get_klines(self, symbol: str, timeframe: str) -> Optional[List]:
        """Get latest klines for a symbol and timeframe."""
        if symbol in self.klines and timeframe in self.klines[symbol]:
            return self.klines[symbol][timeframe]
        return None
    
    def on_ticker(self, callback: Callable):
        """Register callback for ticker updates."""
        self.callbacks['ticker'].append(callback)
    
    def on_kline(self, callback: Callable):
        """Register callback for kline updates."""
        self.callbacks['kline'].append(callback)
    
    def on_orderbook(self, callback: Callable):
        """Register callback for orderbook updates."""
        self.callbacks['orderbook'].append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get WebSocket connection status."""
        return {
            'connected': self.ws is not None,
            'running': self._running,
            'message_count': self.message_count,
            'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None,
            'connection_uptime': (
                (datetime.now(timezone.utc) - self.connection_start_time).total_seconds()
                if self.connection_start_time else 0
            ),
            'subscriptions': len(self.subscriptions),
            'tickers_tracked': len(self.tickers),
            'orderbooks_tracked': len(self.orderbooks),
            'reconnect_attempts': self._reconnect_attempts
        }
