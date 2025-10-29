import gzip
import io
import json
import asyncio
import logging
from .data_validator import validate_kline_timestamp
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Callable, Union, Any, TYPE_CHECKING
from datetime import datetime, timezone
import websockets
from collections import defaultdict

if TYPE_CHECKING:
    from .websocket_manager import StreamDataCollector

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
                 testnet: bool = False, futures: bool = True, 
                 collector: Optional['StreamDataCollector'] = None):
        """
        Initialize BingX WebSocket client.
        
        Args:
            api_key: Optional API key for authenticated endpoints
            api_secret: Optional API secret
            testnet: Use testnet endpoints
            futures: Use futures/swap market (True) or spot (False)
            collector: Optional StreamDataCollector for bridging data
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.futures = futures
        self.collector = collector  # ✅ PATCH 4: Store collector reference
        
        # Select appropriate endpoint
        self.ws_url = self.WS_PUBLIC_SWAP if futures else self.WS_PUBLIC_SPOT
        
        # --- Connection & Task Management ---
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._connection_lock = asyncio.Lock()
        self.auto_reconnect = True

        # --- Reconnection Logic ---
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5  # Initial delay

        # --- Data, Callbacks, and Subscriptions ---
        self.tickers = {}
        self.orderbooks = {}
        self.klines = defaultdict(lambda: defaultdict(list)) # Geliştirme: Veri yapısını daha tutarlı hale getir
        self.callbacks = defaultdict(list)
        # Geliştirme: Abonelikleri daha basit ve güvenilir bir yapıda sakla
        self.subscriptions = {}  # dataType -> sub_message
        
        # Statistics
        self.message_count = 0
        self.last_message_time = None
        self.connection_start_time = None
        
        logger.info(f"BingX WebSocket initialized ({'futures' if futures else 'spot'} market)")

    async def _ping_loop(self):
        """Proactively sends a 'Ping' to the server every 20 seconds to keep the connection alive."""
        while self._running:
            try:
                if self.ws and self.ws.open:
                    await self.ws.send("Ping")
                    logger.debug("Sent 'Ping' to BingX server.")
                await asyncio.sleep(20)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed during ping loop. Exiting.")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping loop: {e}")
                await asyncio.sleep(5)
        logger.info("Ping loop stopped.")
    
    async def connect(self) -> bool:
        """
        Establishes and maintains a single, persistent WebSocket connection.
        Starts listener and ping tasks only once upon successful connection.
        This method is thread-safe.
        """
        async with self._connection_lock:
            if self.ws and self.ws.open:
                return True
            
            try:
                logger.info(f"Connecting to BingX WebSocket: {self.ws_url}")
                self.ws = await websockets.connect(
                    self.ws_url,
                    ping_interval=None,  # We handle ping/pong manually
                    ping_timeout=None
                )
                
                self._running = True
                self.auto_reconnect = True
                self._reconnect_attempts = 0
                self.connection_start_time = datetime.now(timezone.utc)

                # Start background tasks only if they are not already running
                if self._listen_task is None or self._listen_task.done():
                    self._listen_task = asyncio.create_task(self.listen()) # Orijinal listen metodunu çağır
                
                if self._ping_task is None or self._ping_task.done():
                    self._ping_task = asyncio.create_task(self._ping_loop())

                logger.info("✅ BingX WebSocket connected successfully.")
                await self._resubscribe()
                return True

            except Exception as e:
                logger.error(f"Failed to connect to BingX WebSocket: {e}", exc_info=True)
                return False

    def disable_reconnect(self):
        """Permanently disables the auto-reconnect feature for graceful shutdown."""
        logger.info("Permanently disabling auto-reconnect for BingXWebSocket.")
        self.auto_reconnect = False
    
    async def disconnect(self):
        """Gracefully disconnects the WebSocket and stops all background tasks."""
        logger.info("Initiating graceful disconnect...")
        self.disable_reconnect()
        self._running = False

        tasks_to_cancel = [self._listen_task, self._ping_task]
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass # Görev iptali beklenen bir durumdur
        
        if self.ws and self.ws.open:
            await self.ws.close(code=1000, reason="Client shutdown")
        
        self.ws = None
        logger.info("BingX WebSocket disconnected successfully.")
    
    async def subscribe_ticker(self, symbol: str) -> bool:
        """Subscribes to a ticker stream."""
        try:
            bingx_symbol = self._convert_symbol_to_bingx(symbol)
            data_type = f"{bingx_symbol}@ticker"

            if data_type in self.subscriptions:
                return True

            sub_message = {"id": data_type, "reqType": "sub", "dataType": data_type}
            self.subscriptions[data_type] = sub_message

            if self.ws and self.ws.open:
                await self.ws.send(json.dumps(sub_message))
                logger.info(f"Subscribed to ticker: {bingx_symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to ticker {symbol}: {e}")
            return False

    async def subscribe_kline(self, symbol: str, interval: str = "1m") -> bool:
        """Subscribes to a kline/candlestick stream."""
        try:
            bingx_symbol = self._convert_symbol_to_bingx(symbol)
            bingx_interval = self._convert_timeframe(interval)
            data_type = f"{bingx_symbol}@kline_{bingx_interval}"
            
            if data_type in self.subscriptions:
                return True

            sub_message = {"id": data_type, "reqType": "sub", "dataType": data_type}
            self.subscriptions[data_type] = sub_message
            
            if self.ws and self.ws.open:
                await self.ws.send(json.dumps(sub_message))
                logger.info(f"Subscribed to kline: {bingx_symbol} {interval}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to kline {symbol} {interval}: {e}")
            return False

    async def _resubscribe(self):
        """Resubscribes to all tracked channels after a reconnection."""
        if not self.subscriptions:
            return
        
        logger.info(f"Resubscribing to {len(self.subscriptions)} channels...")
        # Create a copy to avoid issues if the dict changes during iteration
        for sub_id, sub_msg in list(self.subscriptions.items()):
            try:
                if self.ws and self.ws.open:
                    await self.ws.send(json.dumps(sub_msg))
                    logger.debug(f"Resent subscription for {sub_id}")
            except Exception as e:
                logger.error(f"Failed to resubscribe to {sub_id}: {e}")
    
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
        
        # Connection lost, try to reconnect if enabled and still running
        if self.auto_reconnect and self._running:
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
        """
        Process ticker update.
        
        BingX ticker response format:
        {
            "code": 0,
            "dataType": "BTC-USDT@ticker",
            "data": {
                "e": "24hTicker",
                "E": 1761327444754,  # Event time (ms)
                "s": "BTC-USDT",      # Symbol
                "c": "110267.8",      # Close/Last price
                "h": "112080.0",      # 24h high
                "l": "109283.7",      # 24h low
                "v": "15204.7267",    # 24h volume
                "q": "171854.69",     # 24h quote volume
                "o": "110703.8",      # Open price
                "A": "110267.8",      # Best ask price
                "a": "2.5786",        # Best ask quantity
                "B": "110267.6",      # Best bid price
                "b": "5.4305",        # Best bid quantity
                "p": "-436.0",        # Price change
                "P": "-0.39"          # Price change percent
            }
        }
        """
        try:
            ticker_data = data.get("data", {})
            if not ticker_data or not isinstance(ticker_data, dict):
                return
            
            # Extract symbol from dataType
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol:
                return
            
            # Convert BingX ticker format to standard format
            ticker = {
                'symbol': self._convert_symbol_from_bingx(symbol),
                'last': float(ticker_data.get('c', 0)),      # Last/Close price
                'bid': float(ticker_data.get('B', 0)),       # Best bid
                'ask': float(ticker_data.get('A', 0)),       # Best ask
                'bidVolume': float(ticker_data.get('b', 0)), # Best bid quantity
                'askVolume': float(ticker_data.get('a', 0)), # Best ask quantity
                'high': float(ticker_data.get('h', 0)),      # 24h high
                'low': float(ticker_data.get('l', 0)),       # 24h low
                'volume': float(ticker_data.get('v', 0)),    # 24h volume
                'quoteVolume': float(ticker_data.get('q', 0)), # 24h quote volume
                'open': float(ticker_data.get('o', 0)),      # Open price
                'change': float(ticker_data.get('p', 0)),    # Price change
                'percentage': float(ticker_data.get('P', 0)), # Price change %
                'timestamp': ticker_data.get('E', int(time.time() * 1000))
            }
            
            # Store ticker
            self.tickers[ticker['symbol']] = ticker
            
            # Call callbacks
            for callback in self.callbacks['ticker']:
                await callback(ticker['symbol'], ticker)
                
        except Exception as e:
            logger.error(f"Error handling ticker: {e}")
            logger.debug(f"Ticker data: {data}")
    
    async def _handle_kline(self, data: dict):
        """
        Process kline/candlestick update.
        
        BingX kline response format:
        {
            "code": 0,
            "dataType": "BTC-USDT@kline_1m",
            "s": "BTC-USDT",
            "data": [  # ALWAYS an array
                {
                    "c": "110267.6",  # close (string)
                    "o": "110298.6",  # open (string)
                    "h": "110298.6",  # high (string)
                    "l": "110265.1",  # low (string)
                    "v": "2.0741",    # volume (string)
                    "T": 1761327420000  # timestamp (number, ms)
                }
            ]
        }
        """
        try:
            # Get kline data - it's ALWAYS a list, not a dict
            kline_data = data.get("data", [])
            
            # Validate data
            if not kline_data or not isinstance(kline_data, list):
                logger.debug(f"Invalid or empty kline data: {type(kline_data)}")
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
            
            # Initialize storage if needed
            if ccxt_symbol not in self.klines:
                self.klines[ccxt_symbol] = {}
            if ccxt_timeframe not in self.klines[ccxt_symbol]:
                self.klines[ccxt_symbol][ccxt_timeframe] = []
            
            # Process all klines in the array (usually just 1)
            processed_klines = []
            for kline_obj in kline_data:
                if not isinstance(kline_obj, dict):
                    logger.warning(f"Kline element is not a dict: {type(kline_obj)}")
                    continue
                
                # Parse kline object to standard format [timestamp, o, h, l, c, v]
                # Note: 'T' is timestamp, 'c' is close, 'o' is open, etc.
                kline = [
                    kline_obj.get('T', int(time.time() * 1000)),  # timestamp in ms
                    float(kline_obj.get('o', 0)),    # open
                    float(kline_obj.get('h', 0)),    # high
                    float(kline_obj.get('l', 0)),    # low
                    float(kline_obj.get('c', 0)),    # close
                    float(kline_obj.get('v', 0))     # volume
                ]

                # 1. VERİ TUTARLILIĞINI KONTROL ET
                validate_kline_timestamp(timestamp=kline[0], timeframe=ccxt_timeframe, symbol=ccxt_symbol)
                
                # Store kline
                self.klines[ccxt_symbol][ccxt_timeframe].append(kline)
                processed_klines.append(kline)
                
                logger.debug(
                    f"Kline updated for {ccxt_symbol} {ccxt_timeframe}: "
                    f"T={kline[0]}, O={kline[1]:.2f}, H={kline[2]:.2f}, "
                    f"L={kline[3]:.2f}, C={kline[4]:.2f}, V={kline[5]:.4f}"
                )
            
            # Trim to last 500 candles
            if len(self.klines[ccxt_symbol][ccxt_timeframe]) > 500:
                self.klines[ccxt_symbol][ccxt_timeframe] = \
                    self.klines[ccxt_symbol][ccxt_timeframe][-500:]
            
            # ✅ PATCH 4: Bridge data to StreamDataCollector
            if processed_klines and self.collector:
                # Call collector's ohlcv_callback if it exists
                cb = getattr(self.collector, 'ohlcv_callback', None)
                if cb:
                    try:
                        # Collector expects: exchange, symbol, timeframe, ohlcv
                        if asyncio.iscoroutinefunction(cb):
                            await cb('bingx', ccxt_symbol, ccxt_timeframe, processed_klines)
                        else:
                            cb('bingx', ccxt_symbol, ccxt_timeframe, processed_klines)
                        logger.debug(f"✅ Bridged {len(processed_klines)} klines to collector: {ccxt_symbol} {ccxt_timeframe}")
                    except Exception as e:
                        logger.error(f"Failed to bridge kline to collector: {e}")
            
            # Call callbacks with the new klines
            if processed_klines:
                for callback in self.callbacks.get('kline', []):
                    await callback(ccxt_symbol, ccxt_timeframe, processed_klines)
                
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
        # --- DÜZENLEME: auto_reconnect'i kontrol et. Artık daha güvenilir olacak. ---
        if not getattr(self, 'auto_reconnect', True):
            logger.info("Auto-reconnect is disabled, will not attempt to reconnect.")
            return False
            
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
