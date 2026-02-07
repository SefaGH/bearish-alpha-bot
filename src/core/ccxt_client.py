import ccxt
import time
import logging
import requests
import json
import re
from requests.adapters import HTTPAdapter
import asyncio
import inspect
import pandas as pd
import ssl
import certifi
import os
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .bingx_authenticator import BingXAuthenticator
from .execution_env import get_bingx_env, is_real_execution_enabled

logger = logging.getLogger(__name__)

try:
    import pandas_ta_classic as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    logger.warning("Kritik kütüphane 'pandas-ta-classic' bulunamadı! İndikatörler eklenemeyecek.")
    PANDAS_TA_AVAILABLE = False

DEFAULT_TIMEOUT = 10

EX_DEFAULTS = {
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
    "sandbox": False,  # Force production mode for all exchanges
    "aiohttp_trust_env": True,  # Trust environment SSL settings
    "httpsProxy": None,  # Disable proxy
    "verify": False  # Disable SSL verification (for GitHub Actions environment)
}

DEFAULT_TIMEOUT = 10

class CcxtClient:
    def __init__(self, ex_name: str, creds: dict | None = None):
        if not hasattr(ccxt, ex_name):
            raise AttributeError(f"Unknown exchange: {ex_name}")
        
        ex_cls = getattr(ccxt, ex_name)
        params = EX_DEFAULTS | (creds or {})

        # Env-driven timeout for CCXT (fails fast to allow retries upstream)
        try:
            timeout_ms = int(os.getenv("CCXT_TIMEOUT_MS", "5000"))
        except (TypeError, ValueError):
            timeout_ms = 5000
        params["timeout"] = timeout_ms
        
        if ex_name in ['kucoin', 'kucoinfutures']:
            params['sandbox'] = False
            logger.info(f"KuCoin {ex_name} initialized in PRODUCTION mode")
        
        # Configure SSL context to work around certificate issues in CI/CD environments
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.info("SSL warnings disabled for CI/CD environment")
        except:
            pass
        
        self.ex = ex_cls(params)
        self.exchange = self.ex
        self.name = ex_name

        self.bingx_env = None
        self._bingx_rest_base_url = None

        if self.name == "bingx":
            self.bingx_env = get_bingx_env()
            self._bingx_position_mode = None
            self._bingx_is_hedged = None
            self._bingx_position_mode_checked_at = None

            if self.bingx_env == "vst":
                try:
                    set_sandbox = getattr(self.ex, "set_sandbox_mode", None)
                    if not callable(set_sandbox):
                        raise RuntimeError("CCXT BingX exchange is missing set_sandbox_mode()")
                    set_sandbox(True)
                except Exception as exc:
                    raise RuntimeError(f"BINGX_ENV=vst requested but failed to enable CCXT sandbox mode: {exc}") from exc

            self._bingx_rest_base_url = (
                "https://open-api-vst.bingx.com" if self.bingx_env == "vst" else "https://open-api.bingx.com"
            )

            if self.bingx_env == "vst":
                swap_url = str(((getattr(self.ex, "urls", None) or {}).get("api") or {}).get("swap") or "")
                if "open-api-vst" not in swap_url:
                    raise RuntimeError(
                        "BINGX_ENV=vst requested but CCXT sandbox URL does not look like VST. "
                        f"exchange.urls.api.swap={swap_url!r}"
                    )
                if "open-api-vst" not in (self._bingx_rest_base_url or ""):
                    raise RuntimeError(
                        "BINGX_ENV=vst requested but REST base URL does not look like VST. "
                        f"rest_base_url={self._bingx_rest_base_url!r}"
                    )

            logger.info(
                "[BINGX-ENV] env=%s ccxt_sandbox=%s rest_base_url=%s",
                self.bingx_env,
                bool(getattr(self.ex, "sandbox", False) or self.bingx_env == "vst"),
                self._bingx_rest_base_url,
            )

            if self.bingx_env == "vst" and is_real_execution_enabled():
                logger.warning("[BINGX-ENV] VST enabled with real execution; hedge mode will be enforced at first order.")

        # Ticker resilience configuration
        try:
            self._ticker_cache_ttl_s = float(os.getenv("TICKER_CACHE_TTL_S", "1.0"))
        except (TypeError, ValueError):
            self._ticker_cache_ttl_s = 1.0
        try:
            self._ticker_attempts = int(os.getenv("TICKER_MAX_ATTEMPTS", "2"))
        except (TypeError, ValueError):
            self._ticker_attempts = 2
        try:
            self._ticker_base_delay_s = float(os.getenv("TICKER_RETRY_BASE_DELAY_S", "0.4"))
        except (TypeError, ValueError):
            self._ticker_base_delay_s = 0.4
        self._ticker_attempts = max(1, self._ticker_attempts)
        if self._ticker_cache_ttl_s < 0:
            self._ticker_cache_ttl_s = 0.0
        self._ticker_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        
        # Increase connection pool size to avoid urllib3 pool saturation warnings
        try:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self.ex.session = session
            logger.debug("Configured custom requests session with pool size 32")
        except Exception as pool_exc:
            logger.warning(f"Failed to extend requests pool: {pool_exc}")

        self._symbol_cache = {}
        self._last_symbol_update = 0
        self._server_time_offset = 0
        
        self._markets_cache = None
        self._markets_cache_time = 0
        self._required_symbols_only = set()
        self._skip_market_load = False
        self.symbols: List[str] = []
        
        if ex_name == 'bingx' and creds:
            self.bingx_auth = BingXAuthenticator(
                api_key=creds.get('apiKey', ''),
                secret_key=creds.get('secret', '')
            )
            logger.info("🔐 [CCXT-CLIENT] BingX authenticator added")
        else:
            self.bingx_auth = None

        # Convert symbol to BingX native format
    def _get_bingx_native_symbol(self, symbol: str) -> str:
        # Convert symbol to BingX native format
        if self.name != 'bingx':
            return symbol
        # "BTC/USDT:USDT" -> "BTC/USDT"
        base_symbol = symbol.split(':')[0]
        # "BTC/USDT" -> "BTC-USDT"
        return base_symbol.replace('/', '-')

    def set_required_symbols(self, symbols: List[str]):
        """
        Set required symbols and skip market loading.
        
        Args:
            symbols: List of symbols to load (e.g., ['BTC/USDT:USDT', 'ETH/USDT:USDT'])
        """
        self._required_symbols_only = set(symbols)
        self._skip_market_load = True
        self.symbols = list(symbols)
        logger.info(f"[{self.name}] Will only work with {len(symbols)} symbols (no market load)")

    async def ohlcv(self, symbol: str, timeframe: str, limit: int = 500, add_indicators: bool = False, since: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Asenkron olarak OHLCV verilerini çeker ve isteğe bağlı olarak teknik indikatörleri ekler.
        """
        loop = asyncio.get_running_loop()
        last_exc = None
        
        for attempt in range(3):
            try:
                effective_limit = limit
                if self.name == "bingx":
                    try:
                        effective_limit = int(effective_limit)
                        if effective_limit > 1440:
                            effective_limit = 1440
                    except Exception:
                        effective_limit = limit

                native_symbol = self._get_bingx_native_symbol(symbol)
                logger.debug(
                    f"Fetching OHLCV for {native_symbol} ({symbol}) {timeframe} limit={effective_limit} (attempt {attempt + 1}/3)"
                )
                
                # Senkron CCXT çağrısını asenkron hale getir
                data = await loop.run_in_executor(
                    None, 
                    lambda: self.ex.fetch_ohlcv(native_symbol, timeframe=timeframe, limit=effective_limit, since=since)
                )

                if not data:
                    logger.warning(f"No OHLCV data returned for {symbol} on attempt {attempt + 1}")
                    continue

                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                if add_indicators:

                    logger.info(f"Adding technical indicators to {symbol} [{timeframe}] data...")
                    custom_strategy = ta.Strategy(
                        name="ML_Features",
                        ta=[
                            {"kind": "rsi"},
                            {"kind": "macd"},
                            {"kind": "bbands", "length": 20, "std": 2},
                            {"kind": "ema", "length": 20},
                            {"kind": "ema", "length": 50},
                            {"kind": "atr", "length": 14},
                        ]
                    )
                    df.ta.strategy(custom_strategy)
                    df.columns = [col.lower().replace('_14_2_2', '') for col in df.columns] # Kolon isimlerini temizle
                    logger.info(f"✅ Indicators added. Columns: {list(df.columns)}")

                logger.info(f"Successfully fetched {len(df)} candles for {symbol} {timeframe}")
                return df

            except Exception as e:
                last_exc = e
                logger.warning(f"OHLCV fetch attempt {attempt + 1}/3 failed for {symbol} {timeframe}: {type(e).__name__}: {e}")
                if attempt < 2:
                    await asyncio.sleep(0.8)
        
        error_msg = f"Failed to fetch OHLCV for {symbol} {timeframe} after 3 attempts"
        logger.error(f"{error_msg}. Last error: {type(last_exc).__name__}: {last_exc}")
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(error_msg)

    def _is_transient_ccxt_error(self, e: Exception) -> bool:
        """Classify transient CCXT/network errors safely across versions."""
        try:
            from ccxt.base import errors as ccxt_errors
        except Exception:
            return False

        transient_names = [
            "RequestTimeout",
            "NetworkError",
            "ExchangeNotAvailable",
            "DDoSProtection",
            "BadGateway",
            "ServiceUnavailable",
        ]
        transient_types = tuple(
            err_type for err_type in (getattr(ccxt_errors, name, None) for name in transient_names) if err_type
        )

        if transient_types:
            return isinstance(e, transient_types)

        fallback_types = tuple(
            err_type for err_type in (
                getattr(ccxt, "RequestTimeout", None),
                getattr(ccxt, "NetworkError", None),
            ) if err_type
        )
        return isinstance(e, fallback_types) if fallback_types else False

    def _retry_ticker(self, native_symbol: str, symbol: str) -> Dict[str, Any]:
        """Retry wrapper around fetch_ticker with bounded backoff + jitter."""
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._ticker_attempts + 1):
            try:
                return self.ex.fetch_ticker(native_symbol)
            except Exception as e:
                if not self._is_transient_ccxt_error(e):
                    raise

                last_exc = e
                logger.warning(
                    f"[CCXT-TICKER-RETRY/{self.name}/{symbol}] "
                    f"attempt={attempt}/{self._ticker_attempts} {type(e).__name__}: {e}"
                )

                if attempt >= self._ticker_attempts:
                    raise

                delay = self._ticker_base_delay_s * (2 ** (attempt - 1)) + random.uniform(0.0, 0.2)
                time.sleep(delay)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Ticker retry failed without exception")

    def ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker data for a symbol."""
        now = time.monotonic()

        cached = self._ticker_cache.get(symbol)
        if cached:
            ts, data = cached
            if (now - ts) <= self._ticker_cache_ttl_s:
                return data

        # Convert symbol to BingX native format
        native_symbol = self._get_bingx_native_symbol(symbol)

        data = self._retry_ticker(native_symbol=native_symbol, symbol=symbol)
        self._ticker_cache[symbol] = (now, data)
        return data

    def tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all tickers. Returns empty dict on failure."""
        try:
            return self.ex.fetch_tickers()
        except Exception as e:
            # Don't raise - this is used for filtering, empty result is acceptable
            return {}

    def markets(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Load markets efficiently. If fixed symbols are set, generates a minimal
        market structure without a network call. Otherwise, loads from cache or network.
        """
        # 1. En verimli senaryo: Sabit semboller ayarlanmış ve yeniden yükleme zorunlu değil.
        # Bu, botun ana çalışma modudur ve AĞ ÇAĞRISINI TAMAMEN ATLAR.
        if self._skip_market_load and not force_reload:
            logger.info(f"[{self.name}] Generating minimal market structure for {len(self._required_symbols_only)} fixed symbols (no network call).")
            fake_markets = {}
            for symbol in self._required_symbols_only:
                # --- DÜZELTME: Sahte market oluştururken de doğru formatı kullan ---
                native_id = self._get_bingx_native_symbol(symbol)
                # CCXT'nin temel kontrolleri geçmesi için gereken minimum alanlar
                fake_markets[symbol] = {
                    'id': native_id, # örn: BTC-USDT
                    'symbol': symbol,
                    'base': symbol.split('/')[0],
                    'quote': 'USDT',
                    'active': True,
                    'type': 'swap',
                    'linear': True,
                    'swap': True,
                    'spot': False,
                    # BingX (CCXT) uses TICK_SIZE precision mode for swaps; these are tick sizes (not decimal places).
                    'precision': {'amount': 0.0001, 'price': 0.1}, # Makul varsayılanlar
                    'limits': {'amount': {'min': 0.0001, 'max': 100}}, # Makul varsayılanlar
                }
            return fake_markets

        # 2. Cache kontrolü (Eski davranış korunuyor)
        current_time = time.time()
        if not force_reload and self._markets_cache and (current_time - self._markets_cache_time) < 3600:
            return self._markets_cache

        # 3. Ağdan yükleme (Sadece gerekirse çalışacak fallback)
        logger.warning(f"[{self.name}] Performing full market load from network (this may be slow).")
        try:
            markets = self.ex.load_markets()
            logger.info(f"Loaded all {len(markets)} markets for {self.name}")
            self._markets_cache = markets
            self._markets_cache_time = current_time
            return markets
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise

    def load_markets(self, reload=False, params={}):
        """
        CCXT market loading wrapper with optimization for fixed symbols.
        This is now a SYNCHRONOUS method and robustly handles symbol parsing.
        """
        logger.info(f"[{self.exchange.id}] load_markets() wrapper called (reload={reload})")

        if self.symbols and not reload:
            logger.info(f"[{self.exchange.id}] Using pre-set minimal market structure for {len(self.symbols)} symbols.")
            try:
                if self.exchange.markets and not reload:
                    return self.exchange.markets

                minimal_markets = {}
                for symbol in self.symbols:
                    native_id = self._get_bingx_native_symbol(symbol)
                    
                    # --- NIHAI DÜZELTME: Sağlam sembol ayırma mantığı ---
                    parts = symbol.split('/')
                    if len(parts) < 2:
                        logger.warning(f"Skipping malformed symbol: {symbol}")
                        continue
                    base = parts[0]
                    quote_parts = parts[1].split(':')
                    quote = quote_parts[0]
                    # --- DÜZELTME SONU ---

                    market = self.exchange.safe_market_structure({
                        'id': native_id, 'symbol': symbol, 'base': base, 'quote': quote,
                        'baseId': base, 'quoteId': quote, 'active': True,
                        'type': 'swap', 'linear': True, 'swap': True,
                        # IMPORTANT: BingX swap in CCXT uses tick-size precision mode; precision fields must be tick sizes.
                        # If you set precision.amount=8 (decimal places) while precisionMode=TICK_SIZE, CCXT will truncate
                        # most valid sizes to '0' and raise InvalidOrder ("minimum amount precision of 8").
                        'precision': {'amount': 0.0001, 'price': 0.1},
                        'limits': {'amount': {'min': 0.0001}}, 'info': {},
                    })
                    minimal_markets[symbol] = market

                self.exchange.set_markets(minimal_markets)
                logger.info(f"[{self.exchange.id}] Injected minimal market structure for {len(self.symbols)} symbols.")
                return self.exchange.markets
            except Exception as e:
                logger.error(f"[{self.exchange.id}] Failed to create minimal market structure: {e}. Falling back to full load.", exc_info=True)
                return self.ex.load_markets(params=params)
        else:
            logger.info(f"[{self.exchange.id}] Performing full market load from network...")
            return self.ex.load_markets(params=params)

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
        """
        # --- DÜZELTME: Sembolü BingX formatına çevir ---
        native_symbol = self._get_bingx_native_symbol(symbol)
        return self.ex.create_order(native_symbol, type_, side, amount, price, params or {})

    def cancel_order(self, order_id: str, symbol: Optional[str] = None, params: dict = None) -> Dict[str, Any]:
        """Cancel an order by id (best-effort wrapper around ccxt.cancel_order)."""
        cancel_symbol = None
        if symbol:
            cancel_symbol = self._get_bingx_native_symbol(symbol) if self.name == "bingx" else symbol
        return self.ex.cancel_order(order_id, cancel_symbol, params or {})

    def create_trailing_percent_order(
        self,
        symbol: str,
        type_: str,
        side: str,
        amount: float,
        price: float = None,
        trailing_percent: float = None,
        trigger_price: float = None,
        params: dict = None,
    ) -> Dict[str, Any]:
        """
        Create a trailing stop order using CCXT's unified create_trailing_percent_order().

        Note: CCXT expects trailing_percent in percent units (e.g., 0.2 for 0.2%),
        and maps it to exchange-native `priceRate` where applicable.
        """
        native_symbol = self._get_bingx_native_symbol(symbol)
        create_fn = getattr(self.ex, "create_trailing_percent_order", None)
        if not callable(create_fn):
            raise RuntimeError(f"{self.name} CCXT adapter does not support create_trailing_percent_order()")
        return create_fn(native_symbol, type_, side, amount, price, trailing_percent, trigger_price, params or {})

    @staticmethod
    def _infer_bingx_hedge_mode(position_mode: Any) -> Optional[bool]:
        def parse_bool(value: Any) -> Optional[bool]:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {
                    "true",
                    "1",
                    "yes",
                    "on",
                    "hedge",
                    "hedged",
                    "dual",
                    "dual_side",
                    "dual-side",
                    "dual side",
                    "hedge_mode",
                    "hedge-mode",
                }:
                    return True
                if text in {
                    "false",
                    "0",
                    "no",
                    "off",
                    "oneway",
                    "one-way",
                    "one_way",
                    "single",
                    "single_side",
                    "single-side",
                    "single side",
                    "oneway_mode",
                    "oneway-mode",
                }:
                    return False
            return None

        def parse_mode(value: Any) -> Optional[bool]:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {
                    "hedge",
                    "hedged",
                    "dual",
                    "dual_side",
                    "dual-side",
                    "dual side",
                    "hedge_mode",
                    "hedge-mode",
                }:
                    return True
                if text in {
                    "oneway",
                    "one-way",
                    "one_way",
                    "single",
                    "single_side",
                    "single-side",
                    "single side",
                    "oneway_mode",
                    "oneway-mode",
                }:
                    return False
            return None

        if not isinstance(position_mode, dict):
            return None

        bool_keys = (
            "hedged",
            "isHedged",
            "is_hedged",
            "dualSidePosition",
            "dualSide",
            "dual_side",
            "dualSidePositionMode",
        )
        mode_keys = ("positionMode", "position_mode", "mode")

        def scan(mapping: Dict[str, Any]) -> Optional[bool]:
            for key in bool_keys:
                if key in mapping:
                    parsed = parse_bool(mapping.get(key))
                    if parsed is not None:
                        return parsed
            for key in mode_keys:
                if key in mapping:
                    parsed = parse_mode(mapping.get(key))
                    if parsed is not None:
                        return parsed
            return None

        parsed = scan(position_mode)
        if parsed is not None:
            return parsed

        info = position_mode.get("info")
        if isinstance(info, dict):
            parsed = scan(info)
            if parsed is not None:
                return parsed

        data = position_mode.get("data")
        if isinstance(data, dict):
            parsed = scan(data)
            if parsed is not None:
                return parsed

        return None

    def ensure_bingx_hedge_mode(self, symbol: str, require_hedged: bool = False) -> Optional[bool]:
        """
        Best-effort: fetch and cache BingX position mode (hedged vs one-way).

        Returns True (hedged) or False (one-way) when detected, otherwise None.
        If require_hedged=True and the account is not hedged, raises RuntimeError.
        """
        if self.name != "bingx":
            return None

        hedged = self._bingx_is_hedged
        if hedged is None:
            try:
                position_mode = self.ex.fetch_position_mode(symbol)
                hedged = self._infer_bingx_hedge_mode(position_mode)
                self._bingx_position_mode = position_mode
                if hedged is not None:
                    self._bingx_is_hedged = hedged
                self._bingx_position_mode_checked_at = datetime.utcnow().isoformat() + "Z"
            except Exception as exc:
                logger.warning("[BINGX-POSITION-MODE] fetch failed: %s", exc)
                if require_hedged:
                    raise RuntimeError(f"Unable to verify BingX hedge mode: {exc}") from exc
                return None

        logger.info(
            "[BINGX-POSITION-MODE] symbol=%s hedged=%s require_hedged=%s",
            symbol,
            hedged,
            require_hedged,
        )
        if require_hedged:
            if hedged is None:
                raise RuntimeError("Unable to verify BingX hedge mode.")
            if not hedged:
                raise RuntimeError("BingX hedge mode required but account is not hedged.")
        return hedged

    @staticmethod
    def _normalize_bingx_leverage_side(side: Optional[str]) -> Optional[str]:
        if side is None:
            return None
        value = str(side).strip().upper()
        if value in ("LONG", "SHORT", "BOTH"):
            return value
        if value == "BUY":
            return "LONG"
        if value == "SELL":
            return "SHORT"
        return None

    @staticmethod
    def _truncate_text(value: Optional[str], limit: int) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    @staticmethod
    def _scrub_sensitive_text(text: str) -> str:
        if not text:
            return text
        patterns = [
            (r"(?i)(apikey=)[^&\s]+", r"\1***"),
            (r"(?i)(api_key=)[^&\s]+", r"\1***"),
            (r"(?i)(signature=)[^&\s]+", r"\1***"),
            (r"(?i)(secret=)[^&\s]+", r"\1***"),
        ]
        scrubbed = text
        for pattern, repl in patterns:
            scrubbed = re.sub(pattern, repl, scrubbed)
        return scrubbed

    @classmethod
    def _redact_sensitive(cls, value: Any, depth: int = 0) -> Any:
        if depth > 3:
            return "<redacted>"
        if isinstance(value, dict):
            redacted: Dict[str, Any] = {}
            for key, val in value.items():
                key_str = str(key)
                key_lower = key_str.lower()
                if key_lower in ("url", "request", "request_url", "headers"):
                    redacted[key_str] = "<redacted>"
                    continue
                if any(token in key_lower for token in ("apikey", "api_key", "secret", "signature", "password")):
                    redacted[key_str] = "***"
                    continue
                redacted[key_str] = cls._redact_sensitive(val, depth + 1)
            return redacted
        if isinstance(value, list):
            return [cls._redact_sensitive(item, depth + 1) for item in value[:50]]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return cls._scrub_sensitive_text(value)
        return value

    @classmethod
    def _safe_serialize(cls, value: Any, limit: int) -> Optional[str]:
        if value is None:
            return None
        redacted = cls._redact_sensitive(value)
        try:
            text = json.dumps(redacted, ensure_ascii=True, separators=(",", ":"))
        except Exception:
            text = str(redacted)
        return cls._truncate_text(text, limit)

    def _extract_ccxt_error_details(
        self,
        exc: Exception,
        *,
        max_raw_len: int,
        max_msg_len: int,
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "exception_type": exc.__class__.__name__,
            "exception_category": "unknown",
            "code": None,
            "http_status": None,
            "message": None,
            "raw": None,
        }

        if isinstance(exc, ccxt.NetworkError):
            details["exception_category"] = "network"
        elif isinstance(exc, ccxt.ExchangeError):
            details["exception_category"] = "exchange"
        elif isinstance(exc, ccxt.BaseError):
            details["exception_category"] = "ccxt"

        for attr in ("code", "error_code", "ret_code", "retCode"):
            value = getattr(exc, attr, None)
            if value is not None:
                details["code"] = value
                break

        for attr in ("status", "status_code", "http_status", "statusCode"):
            value = getattr(exc, attr, None)
            if value is not None:
                details["http_status"] = value
                break

        message = getattr(exc, "message", None) or str(exc)
        message = self._scrub_sensitive_text(message)
        details["message"] = self._truncate_text(message, max_msg_len)

        raw = None
        for attr in ("body", "response", "data"):
            value = getattr(exc, attr, None)
            if value:
                raw = value
                break
        if raw is None:
            for arg in getattr(exc, "args", []) or []:
                if isinstance(arg, (dict, list)):
                    raw = arg
                    break
        details["raw"] = self._safe_serialize(raw, max_raw_len)
        return details

    def _get_bingx_hedge_mode(self, symbol: str) -> Optional[bool]:
        try:
            return self.ensure_bingx_hedge_mode(symbol, require_hedged=False)
        except Exception as exc:
            logger.warning("[BINGX-LEVERAGE] Hedge mode check failed: %s", exc)
            return None

    def _bingx_leverage_attempt_plan(self, symbol: str, side_hint: Optional[str], hedged: Optional[bool]) -> List[str]:
        normalized = self._normalize_bingx_leverage_side(side_hint)
        plan: List[str] = []

        def add(value: Optional[str]) -> None:
            if value and value not in plan:
                plan.append(value)

        if hedged is True:
            if normalized in ("LONG", "SHORT"):
                add(normalized)
                add("BOTH")
            elif normalized == "BOTH":
                add("BOTH")
                add("LONG")
                add("SHORT")
            else:
                add("LONG")
                add("SHORT")
        elif hedged is False:
            add("BOTH")
            if normalized in ("LONG", "SHORT"):
                add(normalized)
        else:
            if normalized in ("LONG", "SHORT"):
                add(normalized)
                add("BOTH")
            else:
                add("BOTH")
                add("LONG")
                add("SHORT")

        return plan

    @staticmethod
    def _bingx_leverage_fallback_from_error(details: Dict[str, Any]) -> List[str]:
        text_parts = [details.get("message") or "", details.get("raw") or ""]
        combined = " ".join(text_parts).lower()
        sides: List[str] = []
        if "one-way" in combined or "one way" in combined or "only be set to both" in combined:
            sides.append("BOTH")
        if "hedge" in combined or "dual" in combined or ("side" in combined and "long" in combined and "short" in combined):
            sides.extend(["LONG", "SHORT"])
        if "side" in combined and "both" in combined:
            if "BOTH" not in sides:
                sides.append("BOTH")
        return sides

    @staticmethod
    def _log_bingx_leverage_event(level: str, payload: Dict[str, Any]) -> None:
        try:
            serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        except Exception:
            serialized = str(payload)
        log_fn = logger.info if level == "info" else logger.warning
        log_fn("%s", serialized)

    def set_leverage(
        self,
        symbol: str,
        leverage: float,
        side: Optional[str] = None,
        *,
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Best-effort leverage setter (exchange-side) with BingX side handling.

        Notes:
        - Many exchanges require swap/futures market type (`options.defaultType=swap`) which we already set.
        - Some exchanges/symbols do not support API leverage changes; this must not crash the bot.
        """
        if leverage is None:
            return None

        try:
            leverage_int = int(float(leverage))
        except (TypeError, ValueError):
            logger.warning("[%s] set_leverage skipped: invalid leverage=%r", self.name, leverage)
            return None

        if leverage_int <= 0:
            leverage_int = 1

        setter = getattr(self.ex, "set_leverage", None) or getattr(self.exchange, "set_leverage", None)
        if not callable(setter):
            logger.warning("[%s] set_leverage not supported by CCXT adapter", self.name)
            return None

        candidates = [symbol]
        if self.name == "bingx":
            try:
                native_symbol = self._get_bingx_native_symbol(symbol)
                if native_symbol and native_symbol not in candidates:
                    candidates.append(native_symbol)
            except Exception:
                pass

        if self.name == "bingx":
            hedged = self._get_bingx_hedge_mode(symbol)
            sides = self._bingx_leverage_attempt_plan(symbol, side, hedged)
        else:
            hedged = None
            sides = [None]

        env_label = self.bingx_env if self.name == "bingx" else "n/a"
        results: List[Dict[str, Any]] = []
        attempted_sides: List[str] = []
        last_exc: Optional[Exception] = None
        last_details: Optional[Dict[str, Any]] = None
        max_attempts = 3
        allow_multi_success = self.name == "bingx" and side is None and hedged is True

        side_index = 0
        while side_index < len(sides) and len(attempted_sides) < max_attempts:
            side_value = sides[side_index]
            side_index += 1
            if side_value in attempted_sides:
                continue
            attempted_sides.append(side_value)

            params: Dict[str, Any] = {}
            if self.name == "bingx" and side_value:
                params["side"] = side_value

            last_exc = None
            side_success = False
            for sym in candidates:
                logger.info(
                    "[LEVERAGE] %s set_leverage request: symbol=%s leverage=%s side=%s env=%s",
                    self.name,
                    sym,
                    leverage_int,
                    side_value or "n/a",
                    env_label,
                )
                if self.name == "bingx":
                    self._log_bingx_leverage_event(
                        "info",
                        {
                            "event": "bingx_set_leverage_attempt",
                            "attempt_index": len(attempted_sides),
                            "side": side_value,
                            "symbol": sym,
                            "leverage": leverage_int,
                            "env": env_label,
                            "strict": strict,
                        },
                    )
                try:
                    if params:
                        resp = setter(leverage_int, sym, params)
                    else:
                        resp = setter(leverage_int, sym)
                    logger.info(
                        "[%s] leverage set: symbol=%s leverage=%s side=%s",
                        self.name,
                        sym,
                        leverage_int,
                        side_value or "n/a",
                    )
                    normalized = resp if isinstance(resp, dict) else {
                        "response": resp,
                        "symbol": sym,
                        "leverage": leverage_int,
                        "side": side_value,
                    }
                    results.append(normalized)
                    if self.name == "bingx":
                        self._log_bingx_leverage_event(
                            "info",
                            {
                                "event": "bingx_set_leverage_success",
                                "used_side": side_value,
                                "attempts_count": len(attempted_sides),
                                "symbol": sym,
                                "leverage": leverage_int,
                                "env": env_label,
                            },
                        )
                    last_exc = None
                    side_success = True
                    break
                except Exception as exc:
                    last_exc = exc
                    last_details = self._extract_ccxt_error_details(
                        exc,
                        max_raw_len=800,
                        max_msg_len=300,
                    )
                    if self.name == "bingx":
                        self._log_bingx_leverage_event(
                            "warning",
                            {
                                "event": "bingx_set_leverage_failed",
                                "symbol": sym,
                                "leverage": leverage_int,
                                "attempted_side": side_value,
                                "env": env_label,
                                "strict": strict,
                                "exception_type": last_details.get("exception_type"),
                                "exception_category": last_details.get("exception_category"),
                                "extracted_error_code": last_details.get("code"),
                                "extracted_http_status": last_details.get("http_status"),
                                "extracted_message": last_details.get("message"),
                                "extracted_raw": last_details.get("raw"),
                            },
                        )
                        for fallback_side in self._bingx_leverage_fallback_from_error(last_details):
                            if fallback_side not in sides and len(sides) < max_attempts:
                                sides.append(fallback_side)
                    else:
                        logger.warning(
                            "[%s] set_leverage failed: symbol=%s leverage=%s err=%s",
                            self.name,
                            sym,
                            leverage_int,
                            exc,
                        )
                    continue

            if side_success and not allow_multi_success:
                break

        if results:
            if len(results) == 1:
                return results[0]
            return {"results": results, "symbol": symbol, "leverage": leverage_int, "sides": attempted_sides}

        if self.name == "bingx":
            self._log_bingx_leverage_event(
                "warning",
                {
                    "event": "bingx_set_leverage_exhausted",
                    "symbol": symbol,
                    "leverage": leverage_int,
                    "attempted_sides": attempted_sides,
                    "env": env_label,
                    "strict": strict,
                    "last_error_code": (last_details or {}).get("code"),
                    "last_http_status": (last_details or {}).get("http_status"),
                    "last_message": (last_details or {}).get("message"),
                },
            )

        if strict and last_exc is not None:
            raise last_exc
        return None

    def fetch_ohlcv_bulk(self, symbol: str, timeframe: str, target_limit: int) -> List[List]:
        """
        Ultimate bulk OHLCV fetching with server sync + dynamic symbols.
        Supports both KuCoin and BingX exchanges.
        
        Args:
            symbol: ccxt symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe string (e.g., '30m', '1h')
            target_limit: Desired candles (500 per batch, unlimited total)
        
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
        
        total_batches = max(1, (target_limit + 499) // 500)
        all_candles = []
        
        logger.info(f"Bulk fetch ({self.name}): {target_limit} candles in up to {total_batches} batches "
               f"(server time: {server_time_ms})")

        for batch_idx in range(total_batches):
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
                logger.info(f"Batch {batch_idx + 1}/{total_batches}: {len(batch_data)} candles "
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
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
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
            base_url = self._bingx_rest_base_url or "https://open-api.bingx.com"
            url = f"{base_url}/openApi/swap/v2/server/time"
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
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
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
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
            base_url = self._bingx_rest_base_url or "https://open-api.bingx.com"
            url = f"{base_url}/openApi/swap/v2/quote/contracts"
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
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
        
        # --- DÜZELTME: Burada da çeviriciyi kullan ---
        native_symbol = self._get_bingx_native_symbol(symbol)
        
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
        base_url = self._bingx_rest_base_url or "https://open-api.bingx.com"
        url = f"{base_url}{endpoint}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if method == 'GET':
                    response = requests.get(url, params=auth_data['params'], headers=auth_data['headers'], timeout=DEFAULT_TIMEOUT)
                elif method == 'POST':
                    response = requests.post(url, data=auth_data['params'], headers=auth_data['headers'], timeout=DEFAULT_TIMEOUT)
                elif method == 'DELETE':
                    response = requests.delete(url, params=auth_data['params'], headers=auth_data['headers'], timeout=DEFAULT_TIMEOUT)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                if response.status_code == 429:
                    logger.warning(f"🔐 [BINGX-API] Rate limit hit (429). Retrying in 1s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(1)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                logger.debug(f"🔐 [BINGX-API] {method} {endpoint} successful")
                return result
                
            except requests.RequestException as e:
                # If it's a 429 that raised an exception (though we handle status_code above, raise_for_status might be called if we missed it)
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                     logger.warning(f"🔐 [BINGX-API] Rate limit hit (429). Retrying in 1s... (Attempt {attempt+1}/{max_retries})")
                     time.sleep(1)
                     continue

                if attempt == max_retries - 1:
                    logger.error(f"🔐 [BINGX-API] {method} {endpoint} failed: {e}")
                    raise
                logger.warning(f"🔐 [BINGX-API] Request failed: {e}. Retrying... (Attempt {attempt+1}/{max_retries})")
                time.sleep(0.5)
    
    def get_bingx_balance(self) -> Dict:
        """
        Get BingX account balance with authentication.
        
        Returns:
            Balance information dictionary
        """
        logger.info("🔐 [BINGX-API] Fetching account balance")
        return self._make_authenticated_bingx_request('/openApi/swap/v2/user/balance')

    @staticmethod
    def extract_bingx_usdt_available(balance_response: Any) -> Optional[float]:
        """
        Best-effort extraction of USDT available balance from BingX swap balance response.
        """
        if not isinstance(balance_response, dict):
            return None

        data = balance_response.get("data")
        candidates = ("availableBalance", "available", "free", "balance")

        def _as_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                asset = str(item.get("asset") or item.get("currency") or "").upper()
                if asset != "USDT":
                    continue
                for key in candidates:
                    if key in item:
                        parsed = _as_float(item.get(key))
                        if parsed is not None:
                            return parsed

        if isinstance(data, dict):
            asset = str(data.get("asset") or data.get("currency") or "").upper()
            if asset == "USDT":
                for key in candidates:
                    if key in data:
                        parsed = _as_float(data.get(key))
                        if parsed is not None:
                            return parsed

        return None
    
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
            # --- DÜZELTME: Burada da çeviriciyi kullan ---
            params['symbol'] = self._get_bingx_native_symbol(symbol)
        
        logger.info(f"🔐 [BINGX-API] Fetching positions {symbol or 'all'}")
        return self._make_authenticated_bingx_request('/openApi/swap/v2/user/positions', params)

    def cancel_all_bingx_open_orders(self, symbol: Optional[str] = None, order_type: Optional[str] = None) -> Dict:
        """
        Cancel all open swap orders on BingX.

        Uses BingX Swap v2 endpoint:
          DELETE /openApi/swap/v2/trade/allOpenOrders

        Args:
            symbol: Optional CCXT-format symbol (e.g., 'BTC/USDT:USDT'). If omitted, cancels all symbols.
            order_type: Optional BingX type filter (e.g., 'LIMIT', 'TRIGGER_MARKET').

        Returns:
            BingX API response dictionary
        """
        if self.name != "bingx":
            raise ValueError("cancel_all_bingx_open_orders is only supported for BingX client.")

        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = self._get_bingx_native_symbol(symbol)
        if order_type:
            params["type"] = str(order_type).upper()

        logger.info("🔐 [BINGX-API] Cancelling all open orders %s", symbol or "(all symbols)")
        return self._make_authenticated_bingx_request('/openApi/swap/v2/trade/allOpenOrders', params, 'DELETE')

    def close_all_bingx_positions(self, symbol: Optional[str] = None) -> Dict:
        """
        Close all swap positions on BingX via one-click close.

        Uses BingX Swap v2 endpoint:
          POST /openApi/swap/v2/trade/closeAllPositions

        Args:
            symbol: Optional CCXT-format symbol (e.g., 'BTC/USDT:USDT'). If omitted, closes all symbols.

        Returns:
            BingX API response dictionary
        """
        if self.name != "bingx":
            raise ValueError("close_all_bingx_positions is only supported for BingX client.")

        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = self._get_bingx_native_symbol(symbol)

        logger.warning("🔐 [BINGX-API] Close all positions %s", symbol or "(all symbols)")
        return self._make_authenticated_bingx_request('/openApi/swap/v2/trade/closeAllPositions', params, 'POST')
    
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
        # --- DÜZELTME: Burada da çeviriciyi kullan ---
        bingx_symbol = self._get_bingx_native_symbol(symbol)
        
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



    def timestamp(self) -> int:
        """Return current timestamp in milliseconds"""
        return int(time.time() * 1000)

    # --- NIHAI DÜZELTME: Güvenli async/sync çağrı mantığı eklendi ---
    async def check_api_health(self) -> Dict[str, Any]:
        """
        Performs a quick health check of the API connection and credentials.
        This version safely handles both synchronous and asynchronous ccxt methods.
        """
        if not hasattr(self, 'ex'):
            return {'status': 'UNHEALTHY', 'reason': 'Internal CCXT exchange object (self.ex) not found.'}

        try:
            fetch_balance_fn = getattr(self.ex, 'fetch_balance', None)
            if not callable(fetch_balance_fn):
                 return {'status': 'UNHEALTHY', 'reason': 'fetch_balance method not available on exchange object.'}

            # Fonksiyonun asenkron olup olmadığını kontrol et
            if inspect.iscoroutinefunction(fetch_balance_fn):
                await fetch_balance_fn()
            else:
                # Senkron ise, event loop'u bloklamamak için executor'da çalıştır
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, fetch_balance_fn)
            
            logger.info(f"✅ API Health Check for '{self.name}' passed (via fetch_balance).")
            return {'status': 'HEALTHY', 'reason': 'Authenticated connection confirmed.'}

        except ccxt.AuthenticationError as e:
            logger.warning(f"ℹ️ API Health Check for '{self.name}': No valid credentials provided. This is expected in public/dry-run mode.")
            return {'status': 'PUBLIC_ONLY', 'reason': f'AuthenticationError: {e}'}
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            logger.error(f"❌ API Health Check for '{self.name}' FAILED: Network or exchange issue. {e}")
            return {'status': 'UNHEALTHY', 'reason': f'NetworkError: {e}'}
        except Exception as e:
            logger.error(f"❌ API Health Check for '{self.name}' FAILED with an unexpected error: {e}", exc_info=True)
            return {'status': 'UNHEALTHY', 'reason': f'Unexpected error: {e}'}
    
    # --- NIHAI DÜZELTME: Güvenli async/sync çağrı mantığı eklendi ---
    async def close(self):
        """
        Safely close the exchange connection, handling both sync and async methods.
        """
        if self.ex and hasattr(self.ex, 'close'):
            try:
                close_fn = getattr(self.ex, 'close')
                if not callable(close_fn):
                    logger.debug(f"[{self.name}] 'close' attribute is not callable.")
                    return

                if inspect.iscoroutinefunction(close_fn):
                    await close_fn()
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, close_fn)
                
                logger.info(f"✅ [{self.name}] Exchange connection closed")
            except Exception as e:
                logger.error(f"⚠️ [{self.name}] Error closing exchange: {e}")
        else:
            logger.debug(f"[{self.name}] Exchange does not have close method or already closed")
