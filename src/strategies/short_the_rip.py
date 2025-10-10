import pandas as pd

class ShortTheRip:
    def __init__(self, cfg):
        self.cfg = cfg

    def signal(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame):
        last30 = df_30m.dropna().iloc[-1]
        if last30['rsi'] < self.cfg['rsi_min']:
            return None
        last1h = df_1h.dropna().iloc[-1]
        near_mid = last30['close'] >= min(last1h['ema_fast'], last1h['ema_mid'])
        if near_mid:
            return {
                'side': 'sell',
                'reason': f"RSI {last30['rsi']:.1f} & touch 1h EMA band",
                'tp_pct': self.cfg['tp_pct'],
                'sl_atr_mult': self.cfg['sl_atr_mult']
            }
        return None
