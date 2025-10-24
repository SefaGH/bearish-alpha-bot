import gzip
import io
import json
import asyncio
import logging
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Callable, Union, Any
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
    - GZIP decompression
    - Ping/Pong handling
    """
    
    # BingX WebSocket endpoints
    WS_PUBLIC_SPOT = "wss://open-api-ws.bingx.com/market"
    WS_PUBLIC_SWAP = "wss://open-api-swap.bingx.com/swap-market"
    WS_VST_SWAP = "wss://vst-open-api-ws.bingx.com/swap-market"
    
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
        
        # Connection management
        self.ws = None
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds
        self._ping_interval = 30  # Send ping every 30 seconds
        self._last_ping_time = None
        
        # Data storage
        self.tickers = {}
        self.orderbooks = {}
        self.klines = defaultdict(dict)  # symbol -> timeframe -> data
        
        # Callbacks
        self.callbacks = {
            'ticker': [],
            'orderbook': [],
            'kline': [],
            'trade': []
        }
        
        # Subscription tracking
        self.subscriptions = {}  # id -> subscription info
        self.pending_subscriptions = {}  # id -> subscription message
        
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
                ping_interval=None,  # We handle ping/pong manually
                ping_timeout=None,
                close_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            
            self._running = True
            self._reconnect_attempts = 0
            self.connection_start_time = datetime.now(timezone.utc)
            self._last_ping_time = time.time()
            
            logger.info("✅ BingX WebSocket connected successfully")
            
            # Re-subscribe to previous subscriptions after reconnect
            if self.pending_subscriptions:
                logger.info(f"Re-subscribing to {len(self.pending_subscriptions)} channels...")
                for sub_id, sub_msg in self.pending_subscriptions.items():
                    await self.ws.send(json.dumps(sub_msg))
                    logger.debug(f"Re-sent subscription: {sub_msg}")
            
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
            symbol: Trading pair (e.g., 'BTC/USDT:USDT' for CCXT format)
            
        Returns:
            True if subscription successful
        """
        try:
            # Convert CCXT format to BingX format
            bingx_symbol = self._convert_symbol_to_bingx(symbol)
            
            sub_id = str(int(time.time() * 1000))
            sub_message = {
                "id": sub_id,
                "reqType": "sub",
                "dataType": f"{bingx_symbol}@ticker"
            }
            
            # Track subscription
            self.pending_subscriptions[sub_id] = sub_message
            
            if self.ws:
                await self.ws.send(json.dumps(sub_message))
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
            bingx_interval = self._convert_timeframe(interval)
            
            sub_id = str(int(time.time() * 1000))
            sub_message = {
                "id": sub_id,
                "reqType": "sub", 
                "dataType": f"{bingx_symbol}@kline_{bingx_interval}"
            }
            
            # Track subscription
            self.pending_subscriptions[sub_id] = sub_message
            
            if self.ws:
                await self.ws.send(json.dumps(sub_message))
                logger.info(f"Subscribed to kline: {bingx_symbol} {bingx_interval}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe to kline {symbol} {interval}: {e}")
            return False
    
    async def listen(self):
        """Listen to WebSocket messages with GZIP support"""
        while self._running:
            try:
                message = await self.ws.recv()
                
                # All BingX messages are GZIP compressed
                if isinstance(message, bytes):
                    try:
                        # Decompress using GzipFile (not gzip.decompress)
                        compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode='rb')
                        decompressed_data = compressed_data.read()
                        message_str = decompressed_data.decode('utf-8')
                        
                        # Handle Ping/Pong
                        if message_str == "Ping":
                            await self.ws.send("Pong")
                            self._last_ping_time = time.time()
                            logger.debug("Received Ping, sent Pong")
                            continue
                        
                        # Skip empty messages
                        if not message_str or message_str.strip() == "":
                            continue
                        
                        # Parse JSON
                        try:
                            data = json.loads(message_str)
                            await self._process_message(data)
                        except json.JSONDecodeError:
                            # Not JSON, might be a status message
                            logger.debug(f"Non-JSON message: {message_str[:100]}")
                            
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
                elif isinstance(message, str):
                    # Should not happen with BingX, but handle anyway
                    if message == "Ping":
                        await self.ws.send("Pong")
                        continue
                    
                    if message.strip():
                        try:
                            data = json.loads(message)
                            await self._process_message(data)
                        except json.JSONDecodeError:
                            logger.debug(f"Non-JSON string: {message[:100]}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Listen loop error: {e}")
                await asyncio.sleep(1)
        
        # Connection lost, try to reconnect
        if self._running:
            logger.info("Connection lost, attempting to reconnect...")
            await self._reconnect()
    
    async def _process_message(self, data: dict):
        """
        Process a parsed WebSocket message.
        
        Args:
            data: Parsed message dictionary
        """
        try:
            # Update statistics
            self.message_count += 1
            self.last_message_time = datetime.now(timezone.utc)
            
            # Check if it's a subscription confirmation
            if "id" in data and "code" in data:
                await self._handle_subscription_response(data)
                return
            
            # Check for error responses
            if "code" in data and data["code"] != 0:
                logger.error(f"BingX error response: {data}")
                return
            
            # Process by data type
            data_type = data.get("dataType", "")
            
            if "@ticker" in data_type:
                await self._handle_ticker(data)
            elif "@kline" in data_type:
                await self._handle_kline(data)
            elif "@depth" in data_type or "@incrDepth" in data_type:
                await self._handle_orderbook(data)
            elif "@trade" in data_type:
                await self._handle_trade(data)
            elif "@lastPrice" in data_type:
                await self._handle_last_price(data)
            elif "@markPrice" in data_type:
                await self._handle_mark_price(data)
            elif "@bookTicker" in data_type:
                await self._handle_book_ticker(data)
            else:
                logger.debug(f"Unknown message type: {data_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.debug(f"Message: {data}")
    
    async def _handle_subscription_response(self, data: dict):
        """Handle subscription confirmation."""
        sub_id = data.get("id")
        code = data.get("code")
        msg = data.get("msg", "")
        
        if code == 0:
            logger.info(f"✅ Subscription confirmed: {sub_id}")
            if sub_id in self.pending_subscriptions:
                self.subscriptions[sub_id] = self.pending_subscriptions[sub_id]
                del self.pending_subscriptions[sub_id]
        else:
            logger.error(f"❌ Subscription failed: {sub_id} - {msg}")
            if sub_id in self.pending_subscriptions:
                del self.pending_subscriptions[sub_id]
    
    async def _handle_ticker(self, data: dict):
        """Process ticker update."""
        try:
            ticker_data = data.get("data", {})
            if not ticker_data:
                return
            
            # Extract symbol from dataType
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
    
    async def _handle_kline(self, data: dict):
        """Process kline/candlestick update."""
        logger.info(f"[KLINE-DEBUG] Received kline data type: {type(data)}")
        logger.info(f"[KLINE-DEBUG] Data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
        logger.info(f"[KLINE-DEBUG] Data sample: {str(data)[:200]}")
        try:
            kline_data = data.get("data")
            if not kline_data:
                logger.debug("No kline data in message")
                return
            
            # Extract symbol and timeframe from dataType
            data_type = data.get("dataType", "")
            if not data_type:
                logger.warning("No dataType in kline message")
                return
                
            parts = data_type.split("@")
            if len(parts) != 2:
                logger.warning(f"Invalid dataType format: {data_type}")
                return
            
            symbol = parts[0]  # e.g., "BTC-USDT"
            timeframe_part = parts[1].replace("kline_", "")  # e.g., "1m"
            
            # Convert to CCXT format
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            ccxt_timeframe = self._convert_timeframe_from_bingx(timeframe_part)
            
            # BingX sends kline data as a dict with specific fields
            if isinstance(kline_data, dict):
                # Parse single kline
                kline = [
                    kline_data.get('t', int(time.time() * 1000)),  # timestamp (eğer yoksa şimdiki zaman)
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
                
                # Log successful kline update
                logger.debug(f"Kline updated for {ccxt_symbol} {ccxt_timeframe}: O={kline[1]}, H={kline[2]}, L={kline[3]}, C={kline[4]}, V={kline[5]}")
                
                # Call callbacks
                for callback in self.callbacks.get('kline', []):
                    await callback(ccxt_symbol, ccxt_timeframe, [kline])
                    
            elif isinstance(kline_data, list):
                # Eğer multiple kline gelirse (batch update)
                for k in kline_data:
                    if isinstance(k, dict):
                        kline = [
                            k.get('t', int(time.time() * 1000)),
                            float(k.get('o', 0)),
                            float(k.get('h', 0)),
                            float(k.get('l', 0)),
                            float(k.get('c', 0)),
                            float(k.get('v', 0))
                        ]
                        # Store each kline...
                        if ccxt_symbol not in self.klines:
                            self.klines[ccxt_symbol] = {}
                        if ccxt_timeframe not in self.klines[ccxt_symbol]:
                            self.klines[ccxt_symbol][ccxt_timeframe] = []
                        
                        self.klines[ccxt_symbol][ccxt_timeframe].append(kline)
                        
                # Trim to 500 candles
                if ccxt_symbol in self.klines and ccxt_timeframe in self.klines[ccxt_symbol]:
                    self.klines[ccxt_symbol][ccxt_timeframe] = self.klines[ccxt_symbol][ccxt_timeframe][-500:]
                    
                # Call callbacks with all new klines
                if ccxt_symbol in self.klines and ccxt_timeframe in self.klines[ccxt_symbol]:
                    for callback in self.callbacks.get('kline', []):
                        await callback(ccxt_symbol, ccxt_timeframe, self.klines[ccxt_symbol][ccxt_timeframe])
            else:
                logger.warning(f"Unexpected kline_data format: {type(kline_data)}")
                
        except Exception as e:
            logger.error(f"Error handling kline: {e}")
            logger.debug(f"Kline message: {data}")
    
    def _parse_kline_dict(self, k: dict) -> list:
        """Parse a kline dict to standard format [timestamp, o, h, l, c, v]"""
        return [
            k.get('t', 0),           # timestamp
            float(k.get('o', 0)),    # open
            float(k.get('h', 0)),    # high
            float(k.get('l', 0)),    # low
            float(k.get('c', 0)),    # close
            float(k.get('v', 0))     # volume
        ]
    
    def _store_kline(self, symbol: str, timeframe: str, kline: list):
        """Store kline data and trigger callbacks."""
        # Initialize storage
        if symbol not in self.klines:
            self.klines[symbol] = {}
        if timeframe not in self.klines[symbol]:
            self.klines[symbol][timeframe] = []
        
        # Store and limit size
        self.klines[symbol][timeframe].append(kline)
        if len(self.klines[symbol][timeframe]) > 500:
            self.klines[symbol][timeframe] = self.klines[symbol][timeframe][-500:]
        
        # Trigger callbacks
        for callback in self.callbacks.get('kline', []):
            asyncio.create_task(callback(symbol, timeframe, [kline]))
    
    async def _handle_orderbook(self, data: dict):
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
            
            # Check if it's incremental depth
            if "@incrDepth" in data_type:
                # Handle incremental updates
                action = ob_data.get("action", "")
                if action == "all":
                    # Full snapshot
                    self.orderbooks[ccxt_symbol] = {
                        'symbol': ccxt_symbol,
                        'bids': [[float(p), float(q)] for p, q in ob_data.get('bids', [])],
                        'asks': [[float(p), float(q)] for p, q in ob_data.get('asks', [])],
                        'timestamp': data.get('ts', int(time.time() * 1000)),
                        'lastUpdateId': ob_data.get('lastUpdateId', 0)
                    }
                elif action == "update":
                    # Incremental update
                    if ccxt_symbol in self.orderbooks:
                        self._apply_orderbook_update(ccxt_symbol, ob_data)
            else:
                # Regular depth snapshot
                self.orderbooks[ccxt_symbol] = {
                    'symbol': ccxt_symbol,
                    'bids': [[float(p), float(q)] for p, q in ob_data.get('bids', [])],
                    'asks': [[float(p), float(q)] for p, q in ob_data.get('asks', [])],
                    'timestamp': data.get('ts', int(time.time() * 1000))
                }
            
            # Call callbacks
            for callback in self.callbacks['orderbook']:
                await callback(ccxt_symbol, self.orderbooks[ccxt_symbol])
                
        except Exception as e:
            logger.error(f"Error handling orderbook: {e}")
    
    def _apply_orderbook_update(self, symbol: str, update_data: dict):
        """Apply incremental orderbook update."""
        if symbol not in self.orderbooks:
            return
        
        ob = self.orderbooks[symbol]
        
        # Update bids
        for price, qty in update_data.get('bids', []):
            price = float(price)
            qty = float(qty)
            
            if qty == 0:
                # Remove price level
                ob['bids'] = [bid for bid in ob['bids'] if bid[0] != price]
            else:
                # Update or add price level
                found = False
                for i, bid in enumerate(ob['bids']):
                    if bid[0] == price:
                        ob['bids'][i][1] = qty
                        found = True
                        break
                if not found:
                    ob['bids'].append([price, qty])
        
        # Update asks
        for price, qty in update_data.get('asks', []):
            price = float(price)
            qty = float(qty)
            
            if qty == 0:
                # Remove price level
                ob['asks'] = [ask for ask in ob['asks'] if ask[0] != price]
            else:
                # Update or add price level
                found = False
                for i, ask in enumerate(ob['asks']):
                    if ask[0] == price:
                        ob['asks'][i][1] = qty
                        found = True
                        break
                if not found:
                    ob['asks'].append([price, qty])
        
        # Sort orderbook
        ob['bids'].sort(key=lambda x: x[0], reverse=True)
        ob['asks'].sort(key=lambda x: x[0])
        
        # Update timestamp and lastUpdateId
        ob['timestamp'] = int(time.time() * 1000)
        ob['lastUpdateId'] = update_data.get('lastUpdateId', ob.get('lastUpdateId', 0))
    
    async def _handle_trade(self, data: dict):
        """Process trade update."""
        try:
            trade_data = data.get("data", {})
            if not trade_data:
                return
            
            # Extract symbol
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol:
                return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            # Format trade
            trade = {
                'symbol': ccxt_symbol,
                'price': float(trade_data.get('p', 0)),
                'quantity': float(trade_data.get('q', 0)),
                'side': 'buy' if trade_data.get('m', True) else 'sell',
                'timestamp': trade_data.get('t', int(time.time() * 1000))
            }
            
            # Call callbacks
            for callback in self.callbacks.get('trade', []):
                await callback(ccxt_symbol, trade)
                
        except Exception as e:
            logger.error(f"Error handling trade: {e}")
    
    async def _handle_last_price(self, data: dict):
        """Process last price update."""
        try:
            price_data = data.get("data", {})
            if not price_data:
                return
            
            # Extract symbol
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol:
                return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            # Update ticker with last price
            if ccxt_symbol not in self.tickers:
                self.tickers[ccxt_symbol] = {}
            
            self.tickers[ccxt_symbol]['last'] = float(price_data.get('p', 0))
            self.tickers[ccxt_symbol]['timestamp'] = data.get('ts', int(time.time() * 1000))
            
        except Exception as e:
            logger.error(f"Error handling last price: {e}")
    
    async def _handle_mark_price(self, data: dict):
        """Process mark price update."""
        try:
            price_data = data.get("data", {})
            if not price_data:
                return
            
            # Extract symbol
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol:
                return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            # Update ticker with mark price
            if ccxt_symbol not in self.tickers:
                self.tickers[ccxt_symbol] = {}
            
            self.tickers[ccxt_symbol]['mark'] = float(price_data.get('p', 0))
            self.tickers[ccxt_symbol]['timestamp'] = data.get('ts', int(time.time() * 1000))
            
        except Exception as e:
            logger.error(f"Error handling mark price: {e}")
    
    async def _handle_book_ticker(self, data: dict):
        """Process book ticker (best bid/ask) update."""
        try:
            book_data = data.get("data", {})
            if not book_data:
                return
            
            # Extract symbol
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol:
                return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            # Update ticker with best bid/ask
            if ccxt_symbol not in self.tickers:
                self.tickers[ccxt_symbol] = {}
            
            self.tickers[ccxt_symbol]['bid'] = float(book_data.get('b', 0))
            self.tickers[ccxt_symbol]['bidVolume'] = float(book_data.get('B', 0))
            self.tickers[ccxt_symbol]['ask'] = float(book_data.get('a', 0))
            self.tickers[ccxt_symbol]['askVolume'] = float(book_data.get('A', 0))
            self.tickers[ccxt_symbol]['timestamp'] = data.get('ts', int(time.time() * 1000))
            
        except Exception as e:
            logger.error(f"Error handling book ticker: {e}")
    
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
    
    def on_trade(self, callback: Callable):
        """Register callback for trade updates."""
        self.callbacks['trade'].append(callback)
    
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
            'pending_subscriptions': len(self.pending_subscriptions),
            'tickers_tracked': len(self.tickers),
            'orderbooks_tracked': len(self.orderbooks),
            'klines_tracked': sum(len(tf) for tf in self.klines.values()),
            'reconnect_attempts': self._reconnect_attempts,
            'last_ping': time.time() - self._last_ping_time if self._last_ping_time else None
        }
