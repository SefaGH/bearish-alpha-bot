import gzip
import io
import json
import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
from datetime import datetime, timezone
import websockets
from collections import defaultdict

from .data_validator import validate_kline_timestamp

if TYPE_CHECKING:
    from .stream_data_collector import StreamDataCollector

logger = logging.getLogger(__name__)


class BingXWebSocket:
    """
    Direct BingX WebSocket client for real-time market data.
    Refactored for improved health checking and robust ping/pong logic.
    """
    
    WS_PUBLIC_SPOT = "wss://open-api-ws.bingx.com/market"
    WS_PUBLIC_SWAP = "wss://open-api-swap.bingx.com/swap-market"
    WS_VST_SWAP = "wss://vst-open-api-ws.bingx.com/swap-market"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 testnet: bool = False, futures: bool = True, 
                 collector: Optional['StreamDataCollector'] = None):
        """
        Initialize BingX WebSocket client.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.futures = futures
        self.collector = collector
        
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
        self._reconnect_delay = 5

        # --- Data, Callbacks, and Subscriptions ---
        self.tickers = {}
        self.orderbooks = {}
        self.klines = defaultdict(dict)
        self.callbacks = defaultdict(list)
        self.subscriptions = {}
        self.pending_subscriptions = {}
        
        # --- Statistics & Health (DÜZELTİLDİ) ---
        self.message_count = 0
        self.last_message_time: Optional[datetime] = None
        self.connection_start_time: Optional[datetime] = None
        
        # Hata 1 Çözümü: Gerekli sağlık durumu özellikleri burada başlatılıyor.
        self._last_ping_time: Optional[float] = None
        self._last_pong_time: Optional[float] = None
        
        logger.info(f"BingX WebSocket initialized ({'futures' if futures else 'spot'} market)")

    async def _ping_loop(self):
        """Proactively sends a 'Ping' to the server every 20 seconds."""
        while self._running:
            try:
                if self.ws and self.ws.open:
                    await self.ws.send("Ping")
                    self._last_ping_time = time.time()
                    logger.debug("Sent Ping to server")
                await asyncio.sleep(20)
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                break
            except Exception as e:
                logger.error(f"Error in proactive ping loop: {e}")
                await asyncio.sleep(5)
    
    async def connect(self) -> bool:
        """Establishes a WebSocket connection."""
        if self.ws and self.ws.open:
            return True
        
        try:
            logger.info(f"Connecting to BingX WebSocket: {self.ws_url}")
            # websockets kütüphanesinin kendi ping mekanizmasını da kullanmak daha güvenlidir.
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=15
            )
            
            self._running = True
            self.auto_reconnect = True
            self._reconnect_attempts = 0
            self.connection_start_time = datetime.now(timezone.utc)
            logger.info("✅ BingX WebSocket connected successfully.")
            
            await self._resubscribe()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to BingX WebSocket: {e}", exc_info=True)
            self.ws = None
            return False

    async def listen(self):
        """Listen to WebSocket messages with GZIP support and manage ping/pong."""
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.create_task(self._ping_loop())

        while self._running:
            try:
                message = await self.ws.recv()
                
                if isinstance(message, bytes):
                    message_str = gzip.decompress(message).decode('utf-8')
                else:
                    message_str = message

                # Hata 2 Çözümü: Sunucudan gelen "Pong" mesajını doğru şekilde işle.
                if message_str == "Pong":
                    self._last_pong_time = time.time()
                    if self._last_ping_time:
                        latency = self._last_pong_time - self._last_ping_time
                        logger.debug(f"Received Pong from server (latency: {latency:.3f}s)")
                    else:
                        logger.debug("Received unsolicited Pong from server.")
                    continue
                
                # Sunucu da bize Ping gönderebilir, buna da hazırlıklı olalım.
                if message_str == "Ping":
                    await self.ws.send("Pong")
                    logger.debug("Received server-side Ping, sent Pong")
                    continue

                if message_str and message_str.strip():
                    data = json.loads(message_str)
                    await self._process_message(data)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed in listen loop: {e.code}")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Listen loop error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("BingX listen loop has stopped.")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket connection status."""
        latency = None
        if self._last_pong_time and self._last_ping_time:
            # Sadece pong, ping'den sonra geldiyse gecikmeyi hesapla
            if self._last_pong_time > self._last_ping_time:
                latency = self._last_pong_time - self._last_ping_time

        return {
            'connected': self.ws is not None and self.ws.open,
            'running': self._running,
            'message_count': self.message_count,
            'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None,
            'connection_uptime_seconds': (datetime.now(timezone.utc) - self.connection_start_time).total_seconds() if self.connection_start_time else 0,
            'subscriptions': len(self.subscriptions),
            'pending_subscriptions': len(self.pending_subscriptions),
            'reconnect_attempts': self._reconnect_attempts,
            'last_ping_sent_ago_seconds': time.time() - self._last_ping_time if self._last_ping_time else None,
            'last_pong_received_ago_seconds': time.time() - self._last_pong_time if self._last_pong_time else None,
            'latency_seconds': latency
        }
        
    # ================================================================= #
    # AŞAĞIDAKİ TÜM METOTLAR ORİJİNAL DOSYANIZDAKİ GİBİ DEĞİŞTİRİLMEDEN  #
    # KORUNUYOR. SADECE YUKARIDAKİ __init__, listen ve get_status       #
    # METOTLARI GÜNCELLENDİ.                                            #
    # ================================================================= #

    def disable_reconnect(self):
        """Permanently disables the auto-reconnect feature for graceful shutdown."""
        logger.info("Permanently disabling auto-reconnect for BingXWebSocket.")
        self.auto_reconnect = False
    
    async def disconnect(self):
        """Gracefully disconnects the WebSocket and stops all background tasks."""
        logger.info("Initiating graceful disconnect in bingx_websocket...")
        self.disable_reconnect()
        self._running = False

        tasks_to_cancel = [self._listen_task, self._ping_task]
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        if self.ws and self.ws.open:
            await self.ws.close(code=1000, reason="Client shutdown")
        
        self.ws = None
        self._listen_task = None
        self._ping_task = None
        logger.info("BingX direct WebSocket disconnected successfully.")
    
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
        for sub_id, sub_msg in list(self.subscriptions.items()):
            try:
                if self.ws and self.ws.open:
                    await self.ws.send(json.dumps(sub_msg))
            except Exception as e:
                logger.error(f"Failed to resubscribe to {sub_id}: {e}")
    
    async def _process_message(self, data: dict):
        """Process a parsed WebSocket message."""
        try:
            self.message_count += 1
            self.last_message_time = datetime.now(timezone.utc)
            
            if "id" in data and "code" in data:
                await self._handle_subscription_response(data)
                return
            
            if "code" in data and data["code"] != 0:
                logger.error(f"BingX error response: {data}")
                return
            
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
            if not ticker_data or not isinstance(ticker_data, dict):
                return
            
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol:
                return
            
            ticker = {
                'symbol': self._convert_symbol_from_bingx(symbol),
                'last': float(ticker_data.get('c', 0)),
                'bid': float(ticker_data.get('B', 0)),
                'ask': float(ticker_data.get('A', 0)),
                'bidVolume': float(ticker_data.get('b', 0)),
                'askVolume': float(ticker_data.get('a', 0)),
                'high': float(ticker_data.get('h', 0)),
                'low': float(ticker_data.get('l', 0)),
                'volume': float(ticker_data.get('v', 0)),
                'quoteVolume': float(ticker_data.get('q', 0)),
                'open': float(ticker_data.get('o', 0)),
                'change': float(ticker_data.get('p', 0)),
                'percentage': float(ticker_data.get('P', 0)),
                'timestamp': ticker_data.get('E', int(time.time() * 1000))
            }
            
            self.tickers[ticker['symbol']] = ticker
            
            for callback in self.callbacks['ticker']:
                await callback(ticker['symbol'], ticker)
                
        except Exception as e:
            logger.error(f"Error handling ticker: {e}")
            logger.debug(f"Ticker data: {data}")
    
    async def _handle_kline(self, data: dict):
        """Process kline/candlestick update."""
        try:
            kline_data = data.get("data", [])
            
            if not kline_data or not isinstance(kline_data, list):
                return
            
            data_type = data.get("dataType", "")
            if not data_type: return
                
            parts = data_type.split("@")
            if len(parts) != 2: return
            
            symbol = parts[0]
            timeframe_part = parts[1].replace("kline_", "")
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            ccxt_timeframe = self._convert_timeframe_from_bingx(timeframe_part)
            
            if ccxt_symbol not in self.klines: self.klines[ccxt_symbol] = {}
            if ccxt_timeframe not in self.klines[ccxt_symbol]: self.klines[ccxt_symbol][ccxt_timeframe] = []
            
            processed_klines = []
            for kline_obj in kline_data:
                if not isinstance(kline_obj, dict): continue
                
                kline = [
                    kline_obj.get('T', int(time.time() * 1000)),
                    float(kline_obj.get('o', 0)),
                    float(kline_obj.get('h', 0)),
                    float(kline_obj.get('l', 0)),
                    float(kline_obj.get('c', 0)),
                    float(kline_obj.get('v', 0))
                ]

                validate_kline_timestamp(timestamp=kline[0], timeframe=ccxt_timeframe, symbol=ccxt_symbol)
                
                self.klines[ccxt_symbol][ccxt_timeframe].append(kline)
                processed_klines.append(kline)
                
            if len(self.klines[ccxt_symbol][ccxt_timeframe]) > 500:
                self.klines[ccxt_symbol][ccxt_timeframe] = self.klines[ccxt_symbol][ccxt_timeframe][-500:]
            
            if processed_klines and self.collector:
                cb = getattr(self.collector, 'ohlcv_callback', None)
                if cb:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb('bingx', ccxt_symbol, ccxt_timeframe, processed_klines)
                        else:
                            cb('bingx', ccxt_symbol, ccxt_timeframe, processed_klines)
                    except Exception as e:
                        logger.error(f"Failed to bridge kline to collector: {e}")
            
            if processed_klines:
                for callback in self.callbacks.get('kline', []):
                    await callback(ccxt_symbol, ccxt_timeframe, processed_klines)
                
        except Exception as e:
            logger.error(f"Error handling kline: {e}")
            logger.debug(f"Kline message: {data}")
    
    def _parse_kline_dict(self, k: dict) -> list:
        """Parse a kline dict to standard format [timestamp, o, h, l, c, v]"""
        return [
            k.get('t', 0),
            float(k.get('o', 0)),
            float(k.get('h', 0)),
            float(k.get('l', 0)),
            float(k.get('c', 0)),
            float(k.get('v', 0))
        ]
    
    def _store_kline(self, symbol: str, timeframe: str, kline: list):
        """Store kline data and trigger callbacks."""
        if symbol not in self.klines: self.klines[symbol] = {}
        if timeframe not in self.klines[symbol]: self.klines[symbol][timeframe] = []
        
        self.klines[symbol][timeframe].append(kline)
        if len(self.klines[symbol][timeframe]) > 500:
            self.klines[symbol][timeframe] = self.klines[symbol][timeframe][-500:]
        
        for callback in self.callbacks.get('kline', []):
            asyncio.create_task(callback(symbol, timeframe, [kline]))
    
    async def _handle_orderbook(self, data: dict):
        """Process orderbook update."""
        try:
            ob_data = data.get("data", {})
            if not ob_data: return
            
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol: return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            if "@incrDepth" in data_type:
                action = ob_data.get("action", "")
                if action == "all":
                    self.orderbooks[ccxt_symbol] = {
                        'symbol': ccxt_symbol,
                        'bids': [[float(p), float(q)] for p, q in ob_data.get('bids', [])],
                        'asks': [[float(p), float(q)] for p, q in ob_data.get('asks', [])],
                        'timestamp': data.get('ts', int(time.time() * 1000)),
                        'lastUpdateId': ob_data.get('lastUpdateId', 0)
                    }
                elif action == "update":
                    if ccxt_symbol in self.orderbooks:
                        self._apply_orderbook_update(ccxt_symbol, ob_data)
            else:
                self.orderbooks[ccxt_symbol] = {
                    'symbol': ccxt_symbol,
                    'bids': [[float(p), float(q)] for p, q in ob_data.get('bids', [])],
                    'asks': [[float(p), float(q)] for p, q in ob_data.get('asks', [])],
                    'timestamp': data.get('ts', int(time.time() * 1000))
                }
            
            for callback in self.callbacks['orderbook']:
                await callback(ccxt_symbol, self.orderbooks[ccxt_symbol])
                
        except Exception as e:
            logger.error(f"Error handling orderbook: {e}")
    
    def _apply_orderbook_update(self, symbol: str, update_data: dict):
        """Apply incremental orderbook update."""
        if symbol not in self.orderbooks: return
        
        ob = self.orderbooks[symbol]
        
        for price, qty in update_data.get('bids', []):
            price, qty = float(price), float(qty)
            if qty == 0:
                ob['bids'] = [bid for bid in ob['bids'] if bid[0] != price]
            else:
                found = False
                for i, bid in enumerate(ob['bids']):
                    if bid[0] == price:
                        ob['bids'][i][1] = qty
                        found = True
                        break
                if not found: ob['bids'].append([price, qty])
        
        for price, qty in update_data.get('asks', []):
            price, qty = float(price), float(qty)
            if qty == 0:
                ob['asks'] = [ask for ask in ob['asks'] if ask[0] != price]
            else:
                found = False
                for i, ask in enumerate(ob['asks']):
                    if ask[0] == price:
                        ob['asks'][i][1] = qty
                        found = True
                        break
                if not found: ob['asks'].append([price, qty])
        
        ob['bids'].sort(key=lambda x: x[0], reverse=True)
        ob['asks'].sort(key=lambda x: x[0])
        
        ob['timestamp'] = int(time.time() * 1000)
        ob['lastUpdateId'] = update_data.get('lastUpdateId', ob.get('lastUpdateId', 0))
    
    async def _handle_trade(self, data: dict):
        """Process trade update."""
        try:
            trade_data = data.get("data", {})
            if not trade_data: return
            
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol: return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            trade = {
                'symbol': ccxt_symbol,
                'price': float(trade_data.get('p', 0)),
                'quantity': float(trade_data.get('q', 0)),
                'side': 'buy' if trade_data.get('m', True) else 'sell',
                'timestamp': trade_data.get('t', int(time.time() * 1000))
            }
            
            for callback in self.callbacks.get('trade', []):
                await callback(ccxt_symbol, trade)
                
        except Exception as e:
            logger.error(f"Error handling trade: {e}")
    
    async def _handle_last_price(self, data: dict):
        """Process last price update."""
        try:
            price_data = data.get("data", {})
            if not price_data: return
            
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol: return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            if ccxt_symbol not in self.tickers: self.tickers[ccxt_symbol] = {}
            
            self.tickers[ccxt_symbol]['last'] = float(price_data.get('p', 0))
            self.tickers[ccxt_symbol]['timestamp'] = data.get('ts', int(time.time() * 1000))
            
        except Exception as e:
            logger.error(f"Error handling last price: {e}")
    
    async def _handle_mark_price(self, data: dict):
        """Process mark price update."""
        try:
            price_data = data.get("data", {})
            if not price_data: return
            
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol: return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            if ccxt_symbol not in self.tickers: self.tickers[ccxt_symbol] = {}
            
            self.tickers[ccxt_symbol]['mark'] = float(price_data.get('p', 0))
            self.tickers[ccxt_symbol]['timestamp'] = data.get('ts', int(time.time() * 1000))
            
        except Exception as e:
            logger.error(f"Error handling mark price: {e}")
    
    async def _handle_book_ticker(self, data: dict):
        """Process book ticker (best bid/ask) update."""
        try:
            book_data = data.get("data", {})
            if not book_data: return
            
            data_type = data.get("dataType", "")
            symbol = data_type.split("@")[0] if "@" in data_type else None
            
            if not symbol: return
            
            ccxt_symbol = self._convert_symbol_from_bingx(symbol)
            
            if ccxt_symbol not in self.tickers: self.tickers[ccxt_symbol] = {}
            
            self.tickers[ccxt_symbol]['bid'] = float(book_data.get('b', 0))
            self.tickers[ccxt_symbol]['bidVolume'] = float(book_data.get('B', 0))
            self.tickers[ccxt_symbol]['ask'] = float(book_data.get('a', 0))
            self.tickers[ccxt_symbol]['askVolume'] = float(book_data.get('A', 0))
            self.tickers[ccxt_symbol]['timestamp'] = data.get('ts', int(time.time() * 1000))
            
        except Exception as e:
            logger.error(f"Error handling book ticker: {e}")
    
    async def _reconnect(self) -> bool:
        """Attempt to reconnect to WebSocket."""
        if not getattr(self, 'auto_reconnect', True):
            logger.info("Auto-reconnect is disabled, will not attempt to reconnect.")
            return False
            
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self._max_reconnect_attempts}) reached")
            return False
        
        self._reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")
        
        delay = min(60, self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)))
        await asyncio.sleep(delay)
        
        return await self.connect()
    
    def _convert_symbol_to_bingx(self, ccxt_symbol: str) -> str:
        """Convert CCXT symbol format to BingX format."""
        if ':' in ccxt_symbol:
            ccxt_symbol = ccxt_symbol.split(':')[0]
        return ccxt_symbol.replace('/', '-')
    
    def _convert_symbol_from_bingx(self, bingx_symbol: str) -> str:
        """Convert BingX symbol format to CCXT format."""
        base_symbol = bingx_symbol.replace('-', '/')
        if self.futures and 'USDT' in base_symbol:
            return f"{base_symbol}:USDT"
        return base_symbol
    
    def _convert_timeframe(self, ccxt_tf: str) -> str:
        """Convert CCXT timeframe to BingX format."""
        mapping = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
            '1d': '1d', '1w': '1w', '1M': '1M'
        }
        return mapping.get(ccxt_tf, '1m')
    
    def _convert_timeframe_from_bingx(self, bingx_tf: str) -> str:
        """Convert BingX timeframe to CCXT format."""
        return bingx_tf
    
    # Public data access methods
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        return self.tickers.get(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        return self.orderbooks.get(symbol)
    
    def get_klines(self, symbol: str, timeframe: str) -> Optional[List]:
        if symbol in self.klines and timeframe in self.klines[symbol]:
            return self.klines[symbol][timeframe]
        return None
    
    def on_ticker(self, callback: Callable):
        self.callbacks['ticker'].append(callback)
    
    def on_kline(self, callback: Callable):
        self.callbacks['kline'].append(callback)
    
    def on_orderbook(self, callback: Callable):
        self.callbacks['orderbook'].append(callback)
    
    def on_trade(self, callback: Callable):
        self.callbacks['trade'].append(callback)
