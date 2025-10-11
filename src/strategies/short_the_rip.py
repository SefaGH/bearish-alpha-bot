# src/strategies/short_the_rip.py
import pandas as pd

class ShortTheRip:
    def __init__(self, cfg):
        self.cfg = cfg or {}

    def signal(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame):
        # Avoid chained assignment, take snapshots of last rows
        last30 = df_30m.dropna().iloc[-1]
        last1h = df_1h.dropna().iloc[-1]

        # thresholds
        rsi_min = self.cfg.get('rsi_min', 61)
        try:
            rsi_min = float(rsi_min)
        except Exception:
            rsi_min = 61.0

        rsi_val = float(last30['rsi'])

        # optional trend/EMA alignment, tolerant if columns missing
        ema_ok = True
        if all(col in last30.index for col in ('ema21','ema50','ema200')):
            ema_ok = float(last30['ema21']) < float(last30['ema50']) <= float(last30['ema200'])

        if rsi_val >= rsi_min and ema_ok:
            return {
                "side": "sell",
                "reason": f"RSI overbought {rsi_val:.1f} (rip)",
                "tp_pct": float(self.cfg.get("tp_pct", 0.012)),
                "sl_atr_mult": float(self.cfg.get("sl_atr_mult", 1.2)),
            }
        return None
