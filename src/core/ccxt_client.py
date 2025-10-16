import ccxt
import time
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from .bingx_authenticator import BingXAuthenticator

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
        
        # Lazy loading for market data
        self._markets_cache = None
        self._markets_cache_time = 0
        self._required_symbols_only = set()  # Sadece ihtiyaç duyulan semboller
        self._skip_market_load = False  # Skip market loading for fixed symbols mode
        
        # Add BingX authenticator
        if ex_name == 'bingx' and creds:
            self.bingx_auth = BingXAuthenticator(
                api_key=creds.get('apiKey', ''),
                secret_key=creds.get('secret', '')
            )
            logger.info("🔐 [CCXT-CLIENT] BingX authenticator added")
        else:
            self.bingx_auth = None

    def set_required_symbols(self, symbols: List[str]):
        """
        Set symbols that should be loaded. Enables selective market loading.
        When called, enables skip_market_load mode for maximum optimization.
        
        Args:
            symbols: List of symbols to load (e.g., ['BTC/USDT:USDT', 'ETH/USDT:USDT'])
        """
        self._required_symbols_only = set(symbols)
        self._skip_market_load = True
        logger.info(f"[{self.name}] Will only work with {len(symbols)} symbols (no market load)")

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

    def markets(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        """Load markets - or skip if we're in fixed symbols mode."""
        current_time = time.time()
    
        # Return cached markets if available and valid
        if not force_reload and self._markets_cache and (current_time - self._markets_cache_time) < 3600:
            return self._markets_cache
        
        # If skip_market_load is enabled (fixed symbols mode), return minimal fake markets
        if self._skip_market_load and not force_reload:
            logger.info(f"[{self.name}] Skipping market load (fixed symbols mode)")
            # Create minimal market structure for required symbols
            fake_markets = {}
            for symbol in self._required_symbols_only:
                # Extract quote currency from symbol (e.g., 'BTC/USDT:USDT' -> 'USDT')
                quote = 'USDT'  # default
                if '/' in symbol:
                    parts = symbol.split('/')
                    if len(parts) > 1:
                        # Get quote from second part, before any colon
                        quote = parts[1].split(':')[0]
                
                # Determine market type from symbol format
                is_swap = ':' in symbol  # Perpetual format like 'BTC/USDT:USDT'
                
                fake_markets[symbol] = {
                    'symbol': symbol,
                    'active': True,
                    'quote': quote,
                    'type': 'swap' if is_swap else 'spot',
                    'linear': True,
                    'swap': is_swap,
                    'spot': not is_swap
                }
            self._markets_cache = fake_markets
            self._markets_cache_time = current_time
            return fake_markets
    
        try:
            # Only load markets if we have required symbols
            if self._required_symbols_only:
                logger.info(f"Loading markets for {len(self._required_symbols_only)} required symbols only")
            
                # For BingX, we need to load all markets first (CCXT limitation)
                # But we'll filter the results
                all_markets = self.ex.load_markets()
            
                # Filter to only required symbols
                markets = {}
                for symbol in self._required_symbols_only:
                    if symbol in all_markets:
                        markets[symbol] = all_markets[symbol]
                    
                logger.info(f"Filtered to {len(markets)} markets from {len(all_markets)} total")
            
            else:
                # Load all markets (old behavior)
                markets = self.ex.load_markets()
                logger.info(f"Loaded all {len(markets)} markets for {self.name}")
        
            self._markets_cache = markets
            self._markets_cache_time = current_time
            return markets
        
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise

    def validate_and_get_symbol(self, requested_symbol="BTC/USDT"):
        """
        Validate symbol exists on exchange, try common variants if not.
        Enhanced with BingX-specific handling.
        
        Args:
            requested_symbol: The symbol to validate (e.g., "BTC/USDT")
        
        Returns:
            str: Validated symbol that exists on the exchange
        
        Raises:
            RuntimeError: If no valid symbol variant is found or if market loading fails
        """
        try:
            logger.info(f"Validating '{requested_symbol}' on {self.name}")
            
            # BingX için özel işlem
            if self.name == 'bingx':
                # Önce contract discovery yap
                symbol_map = self._get_bingx_contracts()
                
                # CCXT formatı BingX mapping'de var mı?
                if requested_symbol in symbol_map:
                    logger.info(f"✅ BingX symbol found via mapping: {requested_symbol}")
                    return requested_symbol
                    
                # Perpetual format dene
                if not requested_symbol.endswith(':USDT'):
                    perp_format = f"{requested_symbol}:USDT"
                    if perp_format in symbol_map:
                        logger.info(f"✅ BingX symbol with perpetual suffix: {perp_format}")
                        return perp_format
                        
                # Spot'tan perpetual'a dönüşüm
                if '/' in requested_symbol and not requested_symbol.endswith(':USDT'):
                    base = requested_symbol.split('/')[0]
                    perp_symbol = f"{base}/USDT:USDT"
                    if perp_symbol in symbol_map:
                        logger.info(f"✅ BingX converted to perpetual: {requested_symbol} → {perp_symbol}")
                        return perp_symbol
                        
                # Hata durumunda mevcut sembolleri göster
                available = list(symbol_map.keys())[:10]
                logger.error(f"❌ BingX symbol not found: {requested_symbol}")
                logger.error(f"   Available samples: {available}")
                raise RuntimeError(f"Symbol '{requested_symbol}' not found on BingX")
            
            # Diğer borsalar için mevcut logic
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
        Ultimate bulk OHLCV fetching with server sync + dynamic symbols.
        Supports both KuCoin and BingX exchanges.
        
        Args:
            symbol: ccxt symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe string (e.g., '30m', '1h')
            target_limit: Desired candles (up to 2000)
        
        Returns:
            Chronologically sorted OHLCV data
        """
        if target_limit <= 500:
            return self.ohlcv(symbol, timeframe, target_limit)
        
        # Get server-synchronized time based on exchange
        if self.name == 'bingx':
            server_time_ms = self._get_bingx_server_time()
        else:
            server_time_ms = self._get_kucoin_server_time()
        
        interval_ms = self._get_timeframe_ms(timeframe)
        
        all_candles = []
        batches_needed = min(4, (target_limit + 499) // 500)
        
        logger.info(f"Bulk fetch ({self.name}): {target_limit} candles in {batches_needed} batches "
                   f"(server time: {server_time_ms})")
        
        for batch_idx in range(batches_needed):
            # Calculate time range using server time
            end_time = server_time_ms - (batch_idx * 500 * interval_ms)
            start_time = end_time - (500 * interval_ms)
            
            try:
                if self.name == 'bingx':
                    batch_data = self._fetch_with_ultimate_bingx_format(
                        symbol, timeframe, start_time, end_time
                    )
                else:
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
    
    def _get_bingx_server_time(self) -> int:
        """Get official BingX server timestamp with local fallback."""
        try:
            url = "https://open-api.bingx.com/openApi/swap/v2/server/time"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            server_data = response.json()
            if server_data.get('code') == 0:
                # BingX returns data as dict with serverTime key
                data = server_data.get('data', {})
                server_time = int(data.get('serverTime', data)) if isinstance(data, dict) else int(data)
                local_time = int(time.time() * 1000)
                self._server_time_offset = server_time - local_time
                
                logger.debug(f"BingX server time sync: {server_time} (offset: {self._server_time_offset}ms)")
                return server_time
            
        except Exception as e:
            logger.warning(f"BingX server time sync failed: {e}, using local time")
            
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
    
    def _get_bingx_contracts(self) -> Dict[str, str]:
        """
        BingX contract discovery with caching.
        
        Returns:
            Dictionary mapping CCXT format symbols to BingX native format
            (e.g., 'BTC/USDT:USDT' -> 'BTC-USDT')
        """
        current_time = time.time()
        
        # 1 saatlik cache
        if (current_time - self._last_symbol_update) < 3600 and self._symbol_cache:
            return self._symbol_cache
            
        try:
            # BingX public endpoint - authentication gerekmez
            url = "https://open-api.bingx.com/openApi/swap/v2/quote/contracts"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == 0:
                symbol_map = {}
                
                for contract in data.get('data', []):
                    # BingX format: "BTC-USDT"
                    native_symbol = contract.get('symbol', '')
                    
                    # Base currency'yi çıkar
                    if '-USDT' in native_symbol:
                        base = native_symbol.replace('-USDT', '')
                        ccxt_symbol = f"{base}/USDT:USDT"
                        symbol_map[ccxt_symbol] = native_symbol
                        
                self._symbol_cache = symbol_map
                self._last_symbol_update = current_time
                
                logger.info(f"✅ BingX: {len(symbol_map)} perpetual contracts discovered")
                
                # Debug: İlk 5 mapping'i göster
                sample = list(symbol_map.items())[:5]
                for ccxt_sym, native_sym in sample:
                    logger.debug(f"  {ccxt_sym} → {native_sym}")
                    
                return symbol_map
                
        except Exception as e:
            logger.warning(f"BingX contract discovery failed: {e}, using fallback")
        
        # Fallback mapping
        return {
            'BTC/USDT:USDT': 'BTC-USDT',
            'ETH/USDT:USDT': 'ETH-USDT',
            'SOL/USDT:USDT': 'SOL-USDT',
            'BNB/USDT:USDT': 'BNB-USDT',
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
    
    def _fetch_with_ultimate_bingx_format(self, symbol: str, timeframe: str,
                                          start_time: int, end_time: int) -> List[List]:
        """Ultimate BingX fetch with dynamic symbols + server time."""
        if self.name != 'bingx':
            return self.ex.fetch_ohlcv(symbol, timeframe, since=start_time, limit=500)
        
        # Get dynamic native symbol
        symbol_map = self._get_bingx_contracts()
        native_symbol = symbol_map.get(symbol, symbol)
        
        # BingX interval format (e.g., "1m", "5m", "1h")
        interval = self._get_bingx_interval(timeframe)
        
        # BingX uses startTime/endTime in milliseconds
        params = {
            'symbol': native_symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 500
        }
        
        logger.debug(f"Ultimate BingX API: {params}")
        
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
    
    def _get_bingx_interval(self, timeframe: str) -> str:
        """Convert timeframe to BingX interval format (e.g., '1m', '1h')."""
        # BingX uses the same format as CCXT standard timeframes
        interval_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h',
            '12h': '12h', '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        return interval_map.get(timeframe, '30m')
    
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
    
    def _make_authenticated_bingx_request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Dict:
        """
        Make authenticated request to BingX API.
        
        Args:
            endpoint: API endpoint (e.g., '/openApi/swap/v2/user/balance')
            params: Optional request parameters
            method: HTTP method ('GET', 'POST', 'DELETE')
            
        Returns:
            API response as dictionary
            
        Raises:
            ValueError: If BingX authenticator is not configured
            requests.RequestException: If request fails
        """
        if not self.bingx_auth:
            raise ValueError("BingX authenticator not configured")
        
        auth_data = self.bingx_auth.prepare_authenticated_request(params)
        url = f"https://open-api.bingx.com{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=auth_data['params'], headers=auth_data['headers'])
            elif method == 'POST':
                response = requests.post(url, data=auth_data['params'], headers=auth_data['headers'])
            elif method == 'DELETE':
                response = requests.delete(url, params=auth_data['params'], headers=auth_data['headers'])
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"🔐 [BINGX-API] {method} {endpoint} successful")
            return result
            
        except requests.RequestException as e:
            logger.error(f"🔐 [BINGX-API] {method} {endpoint} failed: {e}")
            raise
    
    def get_bingx_balance(self) -> Dict:
        """
        Get BingX account balance with authentication.
        
        Returns:
            Balance information dictionary
        """
        logger.info("🔐 [BINGX-API] Fetching account balance")
        return self._make_authenticated_bingx_request('/openApi/swap/v2/user/balance')
    
    def get_bingx_positions(self, symbol: str = None) -> Dict:
        """
        Get BingX positions with authentication.
        
        Args:
            symbol: Optional symbol filter (CCXT format, e.g., 'BTC/USDT:USDT')
            
        Returns:
            Positions information dictionary
        """
        params = {}
        if symbol:
            params['symbol'] = self.bingx_auth.convert_symbol_to_bingx(symbol)
        
        logger.info(f"🔐 [BINGX-API] Fetching positions {symbol or 'all'}")
        return self._make_authenticated_bingx_request('/openApi/swap/v2/user/positions', params)
    
    def place_bingx_order(self, symbol: str, side: str, order_type: str, 
                         amount: float, price: float = None) -> Dict:
        """
        Place BingX order with authentication.
        
        Args:
            symbol: Trading pair (CCXT format, e.g., 'BTC/USDT:USDT')
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market' or 'limit')
            amount: Order amount/volume
            price: Optional price for limit orders
            
        Returns:
            Order response dictionary
        """
        bingx_symbol = self.bingx_auth.convert_symbol_to_bingx(symbol)
        
        params = {
            'symbol': bingx_symbol,
            'side': side.upper(),
            'positionSide': 'LONG',  # Default to LONG
            'type': order_type.upper(),
            'volume': str(amount)
        }
        
        if price and order_type.upper() == 'LIMIT':
            params['price'] = str(price)
        
        logger.info(f"🔐 [BINGX-API] Placing {side} order: {amount} {symbol} @ ${price}")
        return self._make_authenticated_bingx_request('/openApi/swap/v2/trade/order', params, 'POST')
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data dictionary
        """
        return self.ex.fetch_ticker(symbol)
