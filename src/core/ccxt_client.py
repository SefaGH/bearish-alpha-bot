import ccxt, time

EX_DEFAULTS = {
    'options': { 'defaultType': 'swap' },
    'enableRateLimit': True
}

class CcxtClient:
    def __init__(self, ex_name: str, creds: dict | None = None):
        ex_cls = getattr(ccxt, ex_name)
        params = EX_DEFAULTS | (creds or {})
        self.ex = ex_cls(params)

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
