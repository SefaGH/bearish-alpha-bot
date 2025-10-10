import pandas as pd
import pandas_ta as ta

DEFAULTS = dict(ema_fast=20, ema_mid=50, ema_slow=200, rsi=14, atr=14, adx=14)

def add_indicators(df: pd.DataFrame, cfg: dict):
    c = DEFAULTS | cfg
    df = df.copy()
    df['ema_fast'] = ta.ema(df['close'], length=c['ema_fast'])
    df['ema_mid']  = ta.ema(df['close'], length=c['ema_mid'])
    df['ema_slow'] = ta.ema(df['close'], length=c['ema_slow'])
    df['rsi']      = ta.rsi(df['close'], length=c['rsi'])
    df['atr']      = ta.atr(df['high'], df['low'], df['close'], length=c['atr'])
    adx            = ta.adx(df['high'], df['low'], df['close'], length=c['adx'])
    df['adx']      = adx.iloc[:,0] if not adx.empty else None
    return df
