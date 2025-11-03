# src/strategies/oversold_bounce.py
import pandas as pd

from .base_strategy import BaseStrategy

class OversoldBounce(BaseStrategy):
    def __init__(self, cfg):
        super().__init__(strategy_name="oversold_bounce", config=cfg)

    def signal(self, df_30m: pd.DataFrame):
        # Ensure last valid row without chained assignment
        last = df_30m.dropna().iloc[-1]

        # Backward-compat threshold resolution
        rsi_max = self.strategy_config.get('rsi_max', self.strategy_config.get('rsi_min', 25))
        try:
            rsi_max = float(rsi_max)
        except Exception:
            rsi_max = 25.0

        rsi_val = float(last['rsi'])

        if rsi_val <= rsi_max:
            return {
                "side": "buy",
                "reason": f"RSI oversold {rsi_val:.1f}",
                "tp_pct": float(self.strategy_config.get("tp_pct", 0.015)),
                "sl_pct": (float(self.strategy_config["sl_pct"]) if "sl_pct" in self.strategy_config else None),
                "sl_atr_mult": float(self.strategy_config.get("sl_atr_mult", 1.0)),
            }
        return None
