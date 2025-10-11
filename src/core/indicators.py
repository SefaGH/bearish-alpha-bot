# src/core/indicators.py
# Drop-in indicator utilities with modern pandas usage and robust defaults.
# Includes:
# - Wilder RSI (no deprecation warnings, NaN/inf safe)
# - Wilder ATR
# - EMA(21/50/200) helpers
# - add_indicators(df, cfg) pipeline
#
# Safe against:
# - chained assignment warnings
# - deprecated fillna(method=...)
# - early NaN windows (via min_periods)
#
# Expected input df columns: ["timestamp", "open", "high", "low", "close", "volume"]
# Returns a NEW DataFrame (df.copy()) with extra columns: ["rsi", "atr", "ema21", "ema50", "ema200"]

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


# -------------------------
# Helpers & Validators
# -------------------------

REQUIRED_COLS = ("open", "high", "low", "close")

def _require_ohlcv_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Indicators need columns {REQUIRED_COLS}, missing: {missing}")


# -------------------------
# Wilder RSI
# -------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI with stable handling of NaNs and modern pandas API.
    - Uses ewm(..., adjust=False, min_periods=period) for Wilder smoothing.
    - Avoids deprecated fillna(method='bfill') by using .bfill().
    - Returns Series aligned to input index; early NaNs are backfilled, then defaulted to 50.0.
    """
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()

    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    alpha = 1.0 / max(int(period), 1)  # Wilder smoothing equivalent
    roll_up = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))

    # Clean residual inf/NaN and early window
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.bfill().fillna(50.0)


# -------------------------
# Wilder ATR
# -------------------------

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
    alpha = 1.0 / max(int(period), 1)  # Wilder smoothing
    out = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return out


# -------------------------
# EMA
# -------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# -------------------------
# Pipeline
# -------------------------

DEFAULTS = {
    "rsi_period": 14,
    "atr_period": 14,
    "ema_fast": 21,
    "ema_mid": 50,
    "ema_slow": 200,
}

def add_indicators(df: pd.DataFrame, cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Returns a NEW DataFrame with RSI, ATR and EMAs.
    cfg may override keys: rsi_period, atr_period, ema_fast, ema_mid, ema_slow
    """
    _require_ohlcv_columns(df)
    out = df.copy()

    # Resolve config with defaults
    c = dict(DEFAULTS)
    if isinstance(cfg, dict):
        for k in DEFAULTS.keys():
            if k in cfg and cfg[k] is not None:
                c[k] = cfg[k]

    # Compute indicators (avoid chained assignment)
    out["rsi"] = rsi(out["close"], period=int(c["rsi_period"]))
    out["atr"] = atr(out, period=int(c["atr_period"]))

    out["ema21"] = ema(out["close"], period=int(c["ema_fast"]))
    out["ema50"] = ema(out["close"], period=int(c["ema_mid"]))
    out["ema200"] = ema(out["close"], period=int(c["ema_slow"]))

    # Final clean for any residual infinities
    cols = ["rsi", "atr", "ema21", "ema50", "ema200"]
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)

    return out
