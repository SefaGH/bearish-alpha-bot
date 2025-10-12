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
