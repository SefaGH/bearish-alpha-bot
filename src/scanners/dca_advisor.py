import pandas as pd
import pandas_ta as ta

CORE = ["BTC/USDT","ETH/USDT"]

def dca_signal(df_daily: pd.DataFrame):
    sma200 = ta.sma(df_daily['close'], length=200)
    rsi = ta.rsi(df_daily['close'], length=14)
    last = df_daily.dropna().iloc[-1]
    last_sma = sma200.dropna().iloc[-1]
    last_rsi = rsi.dropna().iloc[-1]
    cond = (last['close'] < 0.9*last_sma) and (last_rsi < 35)
    return bool(cond), float(last_rsi), float(last['close']/last_sma - 1)
