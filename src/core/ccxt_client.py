import ccxt
import time

EX_DEFAULTS = {
    "options": {"defaultType": "swap"},
    "enableRateLimit": True
}


class CcxtClient:
    def __init__(self, ex_name: str, creds: dict | None = None):
        ex_cls = getattr(ccxt, ex_name)
        params = EX_DEFAULTS | (creds or {})
        self.ex = ex_cls(params)

    def validate_and_get_symbol(self, requested_symbol: str = "BTC/USDT"):
        """
        Validate symbol exists on exchange, try common variants if not.
        Returns: validated_symbol (str) that exists on the exchange
        Raises: SystemExit with helpful error message if no variant found
        """
        try:
            markets = self.markets()
            symbols = set(markets.keys())
            
            # Try exact match first
            if requested_symbol in symbols:
                return requested_symbol
                
            # Try common BTC variants if it's a BTC symbol
            if requested_symbol.upper().startswith("BTC"):
                variants = [
                    "BTC/USDT",
                    "BTC/USDT:USDT",
                    "BTCUSDT",
                    "BTC-USDT",
                    "BTCUSD"
                ]
                for variant in variants:
                    if variant in symbols:
                        print(f"✅ Symbol fallback: {requested_symbol} → {variant}")
                        return variant
            
            # If no variants work, show available BTC symbols for debugging
            btc_symbols = sorted([s for s in symbols if 'BTC' in s.upper()])[:10]
            error_msg = f"Symbol '{requested_symbol}' not found on exchange."
            if btc_symbols:
                error_msg += f" Available BTC symbols: {btc_symbols}"
            else:
                error_msg += " No BTC symbols found on this exchange."
            raise SystemExit(error_msg)
            
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"Symbol validation failed: {e}")

    def ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        last_exc = None
        for _ in range(3):
            try:
                return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            except Exception as e:
                last_exc = e
                time.sleep(0.8)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("fetch_ohlcv failed after retries with no captured exception")

    def ticker(self, symbol: str):
        return self.ex.fetch_ticker(symbol)

    def tickers(self):
        try:
            return self.ex.fetch_tickers()
        except Exception:
            return {}

    def markets(self):
        return self.ex.load_markets()

    def create_order(self, symbol, side, type_, amount, price=None, params=None):
        return self.ex.create_order(symbol, type_, side, amount, price, params or {})
