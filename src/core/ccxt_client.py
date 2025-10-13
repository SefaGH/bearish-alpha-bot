import ccxt
import time
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

EX_DEFAULTS = {
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
    "sandbox": False  # Force production mode for all exchanges
}


class CcxtClient:
    def __init__(self, ex_name: str, creds: dict | None = None):
        """
        Initialize CCXT exchange client.
        
        Args:
            ex_name: Exchange name (e.g., 'binance', 'bingx')
            creds: API credentials dict with 'apiKey', 'secret', optional 'password'
        
        Raises:
            AttributeError: If exchange name is invalid
            ccxt.AuthenticationError: If credentials are invalid
        """
        if not hasattr(ccxt, ex_name):
            raise AttributeError(f"Unknown exchange: {ex_name}")
        
        ex_cls = getattr(ccxt, ex_name)
        params = EX_DEFAULTS | (creds or {})
        
        # Force production mode for KuCoin exchanges
        if ex_name in ['kucoin', 'kucoinfutures']:
            params['sandbox'] = False
            logger.info(f"KuCoin {ex_name} initialized in PRODUCTION mode")
        
        self.ex = ex_cls(params)
        self.name = ex_name
        
        # Initialize caches for KuCoin Futures integration
        self._symbol_cache = {}
        self._last_symbol_update = 0
        self._server_time_offset = 0

    def ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> List[List]:
        """
        Fetch OHLCV data with retries.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h', '30m')
            limit: Number of candles to fetch
        
        Returns:
            List of OHLCV data [[timestamp, open, high, low, close, volume], ...]
        
        Raises:
            Exception: After 3 failed retry attempts
        """
        last_exc = None
        for attempt in range(3):
            try:
                logger.debug(f"Fetching OHLCV for {symbol} {timeframe} limit={limit} (attempt {attempt + 1}/3)")
                data = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                
                # Enhanced debug logging for KuCoin
                logger.info(f"Exchange: {self.name}, Sandbox: {self.ex.sandbox if hasattr(self.ex, 'sandbox') else 'N/A'}")
                logger.info(f"API Base URL: {getattr(self.ex, 'urls', {}).get('api', 'N/A')}")
                logger.debug(f"Fetched {len(data) if data else 0} candles for {symbol} on {self.name}")
                
                logger.info(f"Successfully fetched {len(data) if data else 0} candles for {symbol} {timeframe}")
                return data
            except Exception as e:
                last_exc = e
                logger.warning(f"OHLCV fetch attempt {attempt + 1}/3 failed for {symbol} {timeframe}: {type(e).__name__}: {e}")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(0.8)
        
        # All retries failed - log detailed error and raise
        error_msg = f"Failed to fetch OHLCV for {symbol} {timeframe} after 3 attempts"
        logger.error(f"{error_msg}. Last error: {type(last_exc).__name__}: {last_exc}")
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(error_msg)

    def ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker data for a symbol."""
        return self.ex.fetch_ticker(symbol)

    def tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all tickers. Returns empty dict on failure."""
        try:
            return self.ex.fetch_tickers()
        except Exception as e:
            # Don't raise - this is used for filtering, empty result is acceptable
            return {}

    def markets(self) -> Dict[str, Dict[str, Any]]:
        """Load market information."""
        try:
            logger.debug(f"Loading markets for {self.name}")
            markets = self.ex.load_markets()
            logger.info(f"Successfully loaded {len(markets)} markets for {self.name}")
            return markets
        except Exception as e:
            logger.error(f"Failed to load markets for {self.name}: {type(e).__name__}: {e}")
            raise

    def validate_and_get_symbol(self, requested_symbol="BTC/USDT"):
        """
        Validate symbol exists on exchange, try common variants if not.
        
        Args:
            requested_symbol: The symbol to validate (e.g., "BTC/USDT")
        
        Returns:
            str: Validated symbol that exists on the exchange
        
        Raises:
            RuntimeError: If no valid symbol variant is found or if market loading fails
        """
        try:
            logger.info(f"Validating symbol '{requested_symbol}' on {self.name}")
            markets = self.markets()
            symbols = set(markets.keys())
            
            # Try exact match first
            if requested_symbol in symbols:
                logger.info(f"✓ Exact match found: {requested_symbol}")
                return requested_symbol
                
            # Try common BTC variants if requested symbol starts with BTC
            if requested_symbol.upper().startswith("BTC"):
                # KuCoin Futures specific priority
                if self.name == 'kucoinfutures':
                    variants = [
                        "BTC/USDT:USDT",   # KuCoin Futures perpetual format (PRIORITY)
                        "XBTUSDM",         # Native KuCoin BTC perpetual
                        "BTCUSDM",         # Alternative native format
                        "BTC/USDT",        # Standard format
                        "BTCUSDT",         # Compact format
                        "BTC-USDT",        # Alternative format
                        "BTCUSD"           # USD-based fallback
                    ]
                else:
                    variants = [
                        "BTC/USDT",        # Standard spot/some futures
                        "BTC/USDT:USDT",   # Many perpetual futures
                        "BTCUSDT",         # Some exchanges (Binance style)
                        "BTC-USDT",        # Alternative format
                        "BTCUSD"           # USD-based (if USDT not available)
                    ]
                
                for variant in variants:
                    if variant in symbols:
                        msg = f"✅ Symbol fallback: {requested_symbol} → {variant}"
                        print(msg)
                        logger.info(msg)
                        return variant
            
            # If no variants work, show available BTC symbols for debugging
            btc_symbols = [s for s in symbols if 'BTC' in s.upper()][:10]
            error_msg = f"Symbol '{requested_symbol}' not found on {self.name}. Available BTC symbols: {btc_symbols}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except RuntimeError:
            raise  # Re-raise RuntimeError as-is
        except Exception as e:
            error_msg = f"Symbol validation failed for {self.name}: {type(e).__name__}: {e}"
            logger.error(error_msg)
            
            # Check if this is an authentication error using ccxt exception type
            if isinstance(e, ccxt.AuthenticationError):
                logger.error(f"⚠️ AUTHENTICATION ERROR: Please verify your {self.name.upper()} API credentials are correct")
                if self.name == 'kucoinfutures':
                    logger.error(f"   KuCoin Futures can use either KUCOIN_* or KUCOINFUTURES_* credentials")
                    logger.error(f"   Required: KUCOIN_KEY + KUCOIN_SECRET + KUCOIN_PASSWORD")
                    logger.error(f"   OR: KUCOINFUTURES_KEY + KUCOINFUTURES_SECRET + KUCOINFUTURES_PASSWORD")
                else:
                    logger.error(f"   Required: {self.name.upper()}_KEY, {self.name.upper()}_SECRET")
                    if self.name in ['kucoin', 'bitget', 'ascendex']:
                        logger.error(f"   Also required: {self.name.upper()}_PASSWORD")
            
            raise RuntimeError(error_msg) from e

    def create_order(self, symbol: str, side: str, type_: str, amount: float, 
                     price: float = None, params: dict = None) -> Dict[str, Any]:
        """
        Create an order.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            type_: Order type (e.g., 'market', 'limit')
            amount: Order size
            price: Price for limit orders
            params: Additional exchange-specific parameters
        
        Returns:
            Order response from exchange
        """
        return self.ex.create_order(symbol, type_, side, amount, price, params or {})

    def fetch_ohlcv_bulk(self, symbol: str, timeframe: str, target_limit: int) -> List[List]:
        """
        Ultimate KuCoin Futures bulk OHLCV with server sync + dynamic symbols.
        
        Args:
            symbol: ccxt symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe string (e.g., '30m', '1h')
            target_limit: Desired candles (up to 2000)
        
        Returns:
            Chronologically sorted OHLCV data
        """
        if target_limit <= 500:
            return self.ohlcv(symbol, timeframe, target_limit)
        
        # Get server-synchronized time
        server_time_ms = self._get_kucoin_server_time()
        interval_ms = self._get_timeframe_ms(timeframe)
        
        all_candles = []
        batches_needed = min(4, (target_limit + 499) // 500)
        
        logger.info(f"Bulk fetch: {target_limit} candles in {batches_needed} batches "
                   f"(server time: {server_time_ms})")
        
        for batch_idx in range(batches_needed):
            # Calculate time range using server time
            end_time = server_time_ms - (batch_idx * 500 * interval_ms)
            start_time = end_time - (500 * interval_ms)
            
            try:
                batch_data = self._fetch_with_ultimate_kucoin_format(
                    symbol, timeframe, start_time, end_time
                )
                
                if not batch_data:
                    logger.warning(f"Batch {batch_idx + 1} returned no data")
                    break
                    
                all_candles.extend(batch_data)
                logger.info(f"Batch {batch_idx + 1}/{batches_needed}: {len(batch_data)} candles "
                           f"({datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)})")
                
                if len(all_candles) >= target_limit:
                    break
                    
                time.sleep(0.7)  # Conservative rate limiting
                
            except Exception as e:
                logger.warning(f"Batch {batch_idx + 1} failed: {e}")
                if batch_idx == 0:
                    raise
                break
        
        # Sort chronologically and limit
        all_candles.sort(key=lambda x: x[0])
        result = all_candles[-target_limit:] if len(all_candles) > target_limit else all_candles
        
        logger.info(f"Ultimate bulk fetch complete: {len(result)} candles delivered")
        return result
    
    def _get_kucoin_server_time(self) -> int:
        """Get official KuCoin server timestamp with local fallback."""
        try:
            url = "https://api-futures.kucoin.com/api/v1/timestamp"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            server_data = response.json()
            if server_data.get('code') == '200000':
                server_time = int(server_data['data'])
                local_time = int(time.time() * 1000)
                self._server_time_offset = server_time - local_time
                
                logger.debug(f"Server time sync: {server_time} (offset: {self._server_time_offset}ms)")
                return server_time
            
        except Exception as e:
            logger.warning(f"Server time sync failed: {e}, using local time")
            
        # Fallback to local time with cached offset
        return int(time.time() * 1000) + self._server_time_offset
    
    def _get_dynamic_symbol_mapping(self) -> Dict[str, str]:
        """Get dynamic symbol mapping from KuCoin active contracts."""
        current_time = time.time()
        
        # Cache for 1 hour
        if (current_time - self._last_symbol_update) < 3600 and self._symbol_cache:
            return self._symbol_cache
            
        try:
            url = "https://api-futures.kucoin.com/api/v1/contracts/active"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            contracts_data = response.json()
            if contracts_data.get('code') == '200000':
                symbol_map = {}
                
                for contract in contracts_data['data']:
                    base = contract['baseCurrency']
                    native_symbol = contract['symbol']
                    
                    # Handle BTC → XBT mapping
                    if base == 'XBT':
                        ccxt_symbol = 'BTC/USDT:USDT'
                    else:
                        ccxt_symbol = f"{base}/USDT:USDT"
                    
                    symbol_map[ccxt_symbol] = native_symbol
                
                self._symbol_cache = symbol_map
                self._last_symbol_update = current_time
                
                logger.info(f"Dynamic symbol mapping updated: {len(symbol_map)} contracts")
                return symbol_map
                
        except Exception as e:
            logger.warning(f"Dynamic symbol fetch failed: {e}")
        
        # Fallback to essential mappings
        return {
            'BTC/USDT:USDT': 'XBTUSDM',
            'ETH/USDT:USDT': 'ETHUSDTM',
            'BNB/USDT:USDT': 'BNBUSDTM'
        }
    
    def _fetch_with_ultimate_kucoin_format(self, symbol: str, timeframe: str,
                                          start_time: int, end_time: int) -> List[List]:
        """Ultimate KuCoin fetch with dynamic symbols + server time."""
        if self.name not in ['kucoin', 'kucoinfutures']:
            return self.ex.fetch_ohlcv(symbol, timeframe, since=start_time, limit=500)
        
        # Get dynamic native symbol
        symbol_map = self._get_dynamic_symbol_mapping()
        native_symbol = symbol_map.get(symbol, symbol)
        
        # Native KuCoin parameters
        granularity = self._get_kucoin_granularity(timeframe)
        
        params = {
            'symbol': native_symbol,
            'granularity': granularity,
            'from': start_time,
            'to': end_time
        }
        
        logger.debug(f"Ultimate KuCoin API: {params}")
        
        return self.ex.fetch_ohlcv(
            symbol, timeframe,
            since=start_time,
            limit=500,
            params=params
        )
    
    def _get_kucoin_granularity(self, timeframe: str) -> int:
        """Convert timeframe to KuCoin granularity (minutes)."""
        granularity_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '8h': 480,
            '12h': 720, '1d': 1440, '1w': 10080
        }
        return granularity_map.get(timeframe, 30)
    
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """Convert timeframe to milliseconds."""
        timeframe_ms = {
            '1m': 60 * 1000, '5m': 5 * 60 * 1000, '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000, '1h': 60 * 60 * 1000, '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000, '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000, '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        }
        return timeframe_ms.get(timeframe, 30 * 60 * 1000)
