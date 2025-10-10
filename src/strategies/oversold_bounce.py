import pandas as pd

class OversoldBounce:
    def __init__(self, cfg):
        self.cfg = cfg

    def signal(self, df_30m: pd.DataFrame):
        last = df_30m.dropna().iloc[-1]
        if last['rsi'] <= self.cfg['rsi_max']:
            return {
                'side': 'buy',
                'reason': f"RSI oversold {last['rsi']:.1f}",
                'tp_pct': self.cfg['tp_pct'],
                'sl_pct': self.cfg['sl_pct']
            }
        return None
