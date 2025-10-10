from dataclasses import dataclass

@dataclass
class RiskConfig:
    per_trade_risk_pct: float
    daily_loss_limit_pct: float
    cool_down_min: int

class RiskGuard:
    def __init__(self, equity_usd: float, cfg: RiskConfig):
        self.equity = equity_usd
        self.cfg = cfg
        self.daily_pl = 0.0
        self.loss_streak = 0

    def per_trade_risk_usd(self):
        return self.equity * self.cfg.per_trade_risk_pct

    def can_trade(self):
        return self.daily_pl > -self.equity * self.cfg.daily_loss_limit_pct and self.loss_streak < 3

    def register_fill(self, pnl_usd: float):
        self.daily_pl += pnl_usd
        self.loss_streak = 0 if pnl_usd > 0 else self.loss_streak + 1
