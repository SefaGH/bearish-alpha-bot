# src/core/indicators.py (compat version)
# Adds EMA alias columns (ema_fast/ema_mid/ema_slow) so legacy strategies using 'ema_mid' won't crash.
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any

REQUIRED_COLS = ("open", "high", "low", "close")

def _require_ohlcv_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Indicators need columns {REQUIRED_COLS}, missing: {missing}")

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    alpha = 1.0 / max(int(period), 1)
    roll_up = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.replace([np.inf, -np.inf], np.nan).bfill().fillna(50.0)

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    _require_ohlcv_columns(df)
    tr = true_range(df["high"], df["low"], df["close"])
    alpha = 1.0 / max(int(period), 1)
    return tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

DEFAULTS = {
    "rsi_period": 14,
    "atr_period": 14,
    "ema_fast": 21,
    "ema_mid": 50,
    "ema_slow": 200,
}

def add_indicators(df: pd.DataFrame, cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    _require_ohlcv_columns(df)
    out = df.copy()

    c = dict(DEFAULTS)
    if isinstance(cfg, dict):
        for k in DEFAULTS.keys():
            if k in cfg and cfg[k] is not None:
                c[k] = cfg[k]

    out["rsi"] = rsi(out["close"], period=int(c["rsi_period"]))
    out["atr"] = atr(out, period=int(c["atr_period"]))

    # primary names
    out["ema21"]  = ema(out["close"], period=int(c["ema_fast"]))
    out["ema50"]  = ema(out["close"], period=int(c["ema_mid"]))
    out["ema200"] = ema(out["close"], period=int(c["ema_slow"]))

    # compatibility alias columns for legacy strategy code
    out["ema_fast"] = out["ema21"]
    out["ema_mid"]  = out["ema50"]
    out["ema_slow"] = out["ema200"]

    cols = ["rsi", "atr", "ema21", "ema50", "ema200", "ema_fast", "ema_mid", "ema_slow"]
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    return out
