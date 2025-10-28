import ccxt
import time
import logging
import requests
import asyncio
import inspect
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from .bingx_authenticator import BingXAuthenticator

# HATA DÜZELTMESİ: pandas-ta kütüphanesini import et
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

logger = logging.getLogger(__name__)

EX_DEFAULTS = {
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
    "sandbox": False
}


class CcxtClient:
    def __init__(self, ex_name: str, creds: dict | None = None):
        if not hasattr(ccxt, ex_name):
            raise AttributeError(f"Unknown exchange: {ex_name}")
        
        ex_cls = getattr(ccxt, ex_name)
        params = EX_DEFAULTS | (creds or {})
        
        if ex_name in ['kucoin', 'kucoinfutures']:
            params['sandbox'] = False
            logger.info(f"KuCoin {ex_name} initialized in PRODUCTION mode")
        
        self.ex = ex_cls(params)
        self.exchange = self.ex
        self.name = ex_name
        
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

    def _get_bingx_native_symbol(self, symbol: str) -> str:
        if self.name != 'bingx':
            return symbol
        base_symbol = symbol.split(':')[0]
        return base_symbol.replace('/', '-')

    def set_required_symbols(self, symbols: List[str]):
        self._required_symbols_only = set(symbols)
        self._skip_market_load = True
        self.symbols = list(symbols)
        logger.info(f"[{self.name}] Will only work with {len(symbols)} symbols (no market load)")

    # --- ANA DÜZELTME: FONKSİYON ASENKRON HALE GETİRİLDİ VE INDIKATOR MANTIĞI EKLENDİ ---
    async def ohlcv(self, symbol: str, timeframe: str, limit: int = 500, add_indicators: bool = False) -> Optional[pd.DataFrame]:
        """
        Asenkron olarak OHLCV verilerini çeker ve isteğe bağlı olarak teknik indikatörleri ekler.
        """
        loop = asyncio.get_running_loop()
        last_exc = None
        
        for attempt in range(3):
            try:
                native_symbol = self._get_bingx_native_symbol(symbol)
                logger.debug(f"Fetching OHLCV for {native_symbol} ({symbol}) {timeframe} limit={limit} (attempt {attempt + 1}/3)")
                
                # Senkron CCXT çağrısını asenkron hale getir
                data = await loop.run_in_executor(
                    None, 
                    lambda: self.ex.fetch_ohlcv(native_symbol, timeframe=timeframe, limit=limit)
                )

                if not data:
                    logger.warning(f"No OHLCV data returned for {symbol} on attempt {attempt + 1}")
                    continue

                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                if add_indicators:
                    if not PANDAS_TA_AVAILABLE:
                        logger.error("'pandas-ta' kütüphanesi kurulu değil. İndikatörler eklenemiyor.")
                        return df # Ham veriyi döndür

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

    # ... (Mevcut dosyanızdaki diğer tüm fonksiyonlar (ticker, markets, create_order vb.) burada aynen kalmalıdır)
    # ... (Bu fonksiyonları buraya tekrar kopyalamıyorum, sadece ohlcv fonksiyonunu değiştirdim)
    # --- LÜTFEN DİKKAT: Sadece yukarıdaki __init__ ve ohlcv fonksiyonlarını ve en baştaki importları güncelleyin. ---
    # --- Dosyanızın geri kalanını silmeyin! ---

    # Mevcut dosyanızdaki diğer tüm fonksiyonlar...
    def ticker(self, symbol: str) -> Dict[str, Any]:
        native_symbol = self._get_bingx_native_symbol(symbol)
        return self.ex.fetch_ticker(native_symbol)

    def tickers(self) -> Dict[str, Dict[str, Any]]:
        try:
            return self.ex.fetch_tickers()
        except Exception:
            return {}

    def markets(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        if self._skip_market_load and not force_reload:
            logger.info(f"[{self.name}] Generating minimal market structure for {len(self._required_symbols_only)} fixed symbols (no network call).")
            fake_markets = {}
            for symbol in self._required_symbols_only:
                native_id = self._get_bingx_native_symbol(symbol)
                fake_markets[symbol] = {
                    'id': native_id,
                    'symbol': symbol,
                    'base': symbol.split('/')[0],
                    'quote': 'USDT',
                    'active': True,
                    'type': 'swap',
                    'linear': True,
                    'swap': True,
                    'spot': False,
                    'precision': {'amount': 0.001, 'price': 0.1},
                    'limits': {'amount': {'min': 0.001, 'max': 100}},
                }
            return fake_markets
        current_time = time.time()
        if not force_reload and self._markets_cache and (current_time - self._markets_cache_time) < 3600:
            return self._markets_cache
        logger.warning(f"[{self.name}] Performing full market load from network (this may be slow).")
        try:
            markets = self.ex.load_markets()
            self._markets_cache = markets
            self._markets_cache_time = current_time
            return markets
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise

    async def load_markets(self, reload=False, params={}):
        logger.info(f"[{self.exchange.id}] load_markets() wrapper called (reload={reload})")
        if self.symbols and not reload:
            logger.info(f"[{self.exchange.id}] Will only work with {len(self.symbols)} symbols (no market load)")
            try:
                minimal_markets = {}
                for symbol in self.symbols:
                    native_id = self._get_bingx_native_symbol(symbol)
                    market = self.exchange.safe_market_structure({
                        'id': native_id, 'symbol': symbol, 'base': symbol.split('/')[0],
                        'quote': symbol.split('/')[1].split(':')[0], 'baseId': symbol.split('/')[0],
                        'quoteId': symbol.split('/')[1].split(':')[0], 'active': True,
                        'type': 'swap', 'linear': True, 'inverse': False, 'spot': False,
                        'swap': True, 'future': False, 'option': False,
                        'precision': {'amount': 8, 'price': 8},
                        'limits': {'amount': {'min': 1e-8, 'max': None}, 'price': {'min': 1e-8, 'max': None}, 'cost': {'min': None, 'max': None}},
                        'info': {},
                    })
                    minimal_markets[symbol] = market
                self.exchange.set_markets(minimal_markets)
                logger.info(f"[{self.exchange.id}] Injected minimal market structure for {len(self.symbols)} symbols.")
                return self.exchange.markets
            except Exception as e:
                logger.error(f"[{self.exchange.id}] Failed to create minimal market structure: {e}. Falling back to full load.")
                return await self.ex.load_markets(params=params)
        else:
            if reload:
                logger.info(f"[{self.exchange.id}] Forcing reload of all markets...")
            else:
                logger.info(f"[{self.exchange.id}] Loading all available markets...")
            load_markets_fn = getattr(self.ex, 'load_markets')
            if inspect.iscoroutinefunction(load_markets_fn):
                return await load_markets_fn(params=params)
            else:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: load_markets_fn(params=params))

    # Diğer tüm fonksiyonlar... (validate_and_get_symbol, create_order, vb. olduğu gibi kalacak)
    # Bu fonksiyonları silmediğinizden emin olun.
    # ...
