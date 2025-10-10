import pandas as pd
import numpy as np

DEFAULTS = dict(ema_fast=20, ema_mid=50, ema_slow=200, rsi=14, atr=14, adx=14)

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    # Based on Welles Wilder's DMI/ADX with EMA smoothing
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=close.index).ewm(alpha=1/length, adjust=False).mean() / (atr + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=close.index).ewm(alpha=1/length, adjust=False).mean() / (atr + 1e-12))

    dx = 100 * ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12) )
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx

def add_indicators(df: pd.DataFrame, cfg: dict):
    c = DEFAULTS | cfg
    df = df.copy()
    df['ema_fast'] = _ema(df['close'], c['ema_fast'])
    df['ema_mid']  = _ema(df['close'], c['ema_mid'])
    df['ema_slow'] = _ema(df['close'], c['ema_slow'])
    df['rsi']      = _rsi(df['close'], c['rsi'])
    df['atr']      = _atr(df['high'], df['low'], df['close'], c['atr'])
    df['adx']      = _adx(df['high'], df['low'], df['close'], c['adx'])
    return df
