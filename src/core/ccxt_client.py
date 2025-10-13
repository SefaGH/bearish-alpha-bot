import ccxt
import time
from typing import Dict, Any, List

EX_DEFAULTS = {
    "options": {"defaultType": "swap"},
    "enableRateLimit": True
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
                return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            except Exception as e:
                last_exc = e
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(0.8)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"fetch_ohlcv failed after retries for {symbol} {timeframe}")

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
        return self.ex.load_markets()

    def validate_and_get_symbol(self, requested_symbol="BTC/USDT"):
        """
        Validate symbol exists on exchange, try common variants if not.
        
        Args:
            requested_symbol: The symbol to validate (e.g., "BTC/USDT")
        
        Returns:
            str: Validated symbol that exists on the exchange
        
        Raises:
            SystemExit: If no valid symbol variant is found
        """
        try:
            markets = self.markets()
            symbols = set(markets.keys())
            
            # Try exact match first
            if requested_symbol in symbols:
                return requested_symbol
                
            # Try common BTC variants if requested symbol starts with BTC
            if requested_symbol.upper().startswith("BTC"):
                variants = [
                    "BTC/USDT",        # Standard spot/some futures
                    "BTC/USDT:USDT",   # Many perpetual futures
                    "BTCUSDT",         # Some exchanges (Binance style)
                    "BTC-USDT",        # Alternative format
                    "BTCUSD"           # USD-based (if USDT not available)
                ]
                
                for variant in variants:
                    if variant in symbols:
                        print(f"✅ Symbol fallback: {requested_symbol} → {variant}")
                        return variant
            
            # If no variants work, show available BTC symbols for debugging
            btc_symbols = [s for s in symbols if 'BTC' in s.upper()][:10]
            raise SystemExit(f"Symbol '{requested_symbol}' not found on {self.name}. Available BTC symbols: {btc_symbols}")
            
        except SystemExit:
            raise  # Re-raise SystemExit as-is
        except Exception as e:
            raise SystemExit(f"Symbol validation failed for {self.name}: {e}")

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
