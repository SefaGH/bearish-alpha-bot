from .notify import Telegram

class ExecEngine:
    def __init__(self, mode: str, client, fee_pct: float, slip_pct: float, tg: Telegram | None):
        self.mode = mode
        self.client = client
        self.fee = fee_pct
        self.slip = slip_pct
        self.tg = tg

    def market_order(self, symbol: str, side: str, qty: float):
        if self.mode == 'paper':
            return {'id':'paper', 'symbol':symbol, 'side':side, 'qty':qty}
        return self.client.create_order(symbol, side=side, type_='market', amount=qty)
