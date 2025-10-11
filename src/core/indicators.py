import pandas as pd
import numpy as np

# ---------------------------
# Helpers
# ---------------------------

def _as_len(x, default=14):
    """
    Config değeri sayı veya dict olabilir:
      - 14
      - {"length": 14}
    Hangisi gelirse gelsin int uzunluk döndür.
    """
    if isinstance(x, (int, float)) and x > 0:
        return int(x)
    if isinstance(x, dict):
        v = x.get('length', default)
        try:
            v = int(v)
            return v if v > 0 else default
        except Exception:
            return default
    return default

def _as_ma(x, default='ema'):
    """
    ATR için hareketli ortalama tipi:
      - "ema" (varsayılan)
      - "sma"
      - {"ma": "ema"}
    """
    if isinstance(x, str):
        return x.lower()
    if isinstance(x, dict):
        v = str(x.get('ma', default)).lower()
        return v
    return default

# ---------------------------
# Indicators
# ---------------------------

def _ema(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    # ewm'de alpha=1/length yaklaşımı ile EMA
    return series.ewm(alpha=1/length, adjust=False).mean()

def _rsi(close: pd.Series, length_or_cfg) -> pd.Series:
    length = _as_len(length_or_cfg, 14)
    length = max(int(length), 1)
    delta = close.diff()

    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1/length, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(method='bfill').fillna(50.0)

def _atr(df: pd.DataFrame, length_or_cfg) -> pd.Series:
    length = _as_len(length_or_cfg, 14)
    ma_type = _as_ma(length_or_cfg, 'ema')

    high = df['high']
    low  = df['low']
    close= df['close']

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    length = max(int(length), 1)
    if ma_type == 'sma':
        return tr.rolling(length, min_periods=1).mean()
    else:
        return tr.ewm(alpha=1/length, adjust=False).mean()

# ---------------------------
# Public API
# ---------------------------

def add_indicators(df: pd.DataFrame, cfg_indicators: dict) -> pd.DataFrame:
    """
    df: columns = ['ts','open','high','low','close','vol']
    cfg_indicators: {
      "rsi": {"length": 14},
      "ema": {"fast": 21, "mid": 50, "slow": 200},
      "atr": {"length": 14, "ma": "ema"|"sma"}
    }
    """
    df = df.copy()

    # RSI
    rsi_cfg = cfg_indicators.get('rsi', {'length': 14})
    df['rsi'] = _rsi(df['close'], rsi_cfg)

    # EMA'lar
    ema_cfg = cfg_indicators.get('ema', {'fast': 21, 'mid': 50, 'slow': 200})
    ema_fast_len = _as_len(ema_cfg.get('fast', 21), 21)
    ema_mid_len  = _as_len(ema_cfg.get('mid', 50), 50)
    ema_slow_len = _as_len(ema_cfg.get('slow', 200), 200)

    df['ema_fast'] = _ema(df['close'], ema_fast_len)
    df['ema_mid']  = _ema(df['close'], ema_mid_len)
    df['ema_slow'] = _ema(df['close'], ema_slow_len)

    # ATR
    atr_cfg = cfg_indicators.get('atr', {'length': 14, 'ma': 'ema'})
    df['atr'] = _atr(df, atr_cfg)

    return df
