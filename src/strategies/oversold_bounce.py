# src/strategies/oversold_bounce.py
import pandas as pd

class OversoldBounce:
    def __init__(self, cfg):
        self.cfg = cfg or {}

    def signal(self, df_30m: pd.DataFrame):
        # Ensure last valid row without chained assignment
        last = df_30m.dropna().iloc[-1]

        # Backward-compat threshold resolution
        rsi_max = self.cfg.get('rsi_max', self.cfg.get('rsi_min', 25))
        try:
            rsi_max = float(rsi_max)
        except Exception:
            rsi_max = 25.0

        rsi_val = float(last['rsi'])

        if rsi_val <= rsi_max:
            return {
                "side": "buy",
                "reason": f"RSI oversold {rsi_val:.1f}",
                "tp_pct": float(self.cfg.get("tp_pct", 0.015)),
                # prefer explicit sl_pct if provided; else risk module may use sl_atr_mult
                "sl_pct": (float(self.cfg["sl_pct"]) if "sl_pct" in self.cfg else None),
                "sl_atr_mult": float(self.cfg.get("sl_atr_mult", 1.0)),
            }
        return None
