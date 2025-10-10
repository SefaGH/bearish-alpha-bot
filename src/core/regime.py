import pandas as pd

def is_bearish_regime(df_4h: pd.DataFrame) -> bool:
    last = df_4h.dropna().iloc[-1]
    return bool((last['ema_mid'] < last['ema_slow']) and (last['close'] < last['ema_slow']))
