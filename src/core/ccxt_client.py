import ccxt
import time
import logging
from typing import Dict, Any, List

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

    def fetch_ohlcv_bulk(self, symbol: str, timeframe: str, target_limit: int) -> List[List]:
        """
        Fetch OHLCV data with automatic pagination for large datasets.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h', '30m') 
            target_limit: Desired number of candles (can exceed exchange limits)
        
        Returns:
            List of OHLCV data with up to target_limit candles
        """
        if target_limit <= 200:
            # Single API call sufficient
            return self.ohlcv(symbol, timeframe, target_limit)
        
        # Multiple API calls needed
        all_candles = []
        batches_needed = min(5, (target_limit + 199) // 200)  # Max 5 batches (1000 candles)
        
        for batch in range(batches_needed):
            batch_limit = min(200, target_limit - len(all_candles))
            if batch_limit <= 0:
                break
                
            try:
                batch_data = self.ohlcv(symbol, timeframe, batch_limit)
                if not batch_data:
                    break
                    
                all_candles.extend(batch_data)
                logger.info(f"Batch {batch + 1}/{batches_needed}: fetched {len(batch_data)} candles, total: {len(all_candles)}")
                
                if len(all_candles) >= target_limit:
                    break
                    
                # Rate limiting between batches
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Batch {batch + 1} failed: {e}")
                if batch == 0:  # First batch failed
                    raise
                break  # Use partial data from successful batches
        
        result = all_candles[:target_limit] if all_candles else []
        logger.info(f"Bulk fetch complete: requested {target_limit}, got {len(result)} candles")
        return result

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
