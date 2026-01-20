# src/core/indicators.py (compat version)
# Adds EMA alias columns (ema_fast/ema_mid/ema_slow) so legacy strategies using 'ema_mid' won't crash.
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .volatility_estimators import VolatilityEstimators

logger = logging.getLogger(__name__)

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


def _directional_movements(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Compute positive/negative directional movement components."""
    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)
    return plus_dm, minus_dm


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index for basic trend strength filtering."""
    _require_ohlcv_columns(df)
    period = max(int(period), 1)

    tr = true_range(df["high"], df["low"], df["close"])
    plus_dm, minus_dm = _directional_movements(df["high"], df["low"])

    tr_smoothed = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smoothed = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm_smoothed / tr_smoothed.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smoothed / tr_smoothed.replace(0, np.nan))

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = (dx * 100).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_series.replace([np.inf, -np.inf], np.nan)

def ema(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

DEFAULTS = {
    "rsi_period": 14,
    "atr_period": 14,
    "ema_fast": 21,
    "ema_mid": 50,
    "ema_slow": 200,
    "vwap_lookback": 1440,
    "vwap_band_multiplier": 2.0,
    "adx_period": 14,
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

    # VWAP (rolling) + bands
    try:
        lookback = max(int(c.get("vwap_lookback", DEFAULTS["vwap_lookback"])), 1)
    except Exception:
        lookback = DEFAULTS["vwap_lookback"]

    typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
    vp = typical_price * out["volume"]
    total_vp = vp.rolling(window=lookback, min_periods=lookback // 2).sum()
    total_vol = out["volume"].rolling(window=lookback, min_periods=lookback // 2).sum()
    vwap = total_vp / total_vol.replace(0, np.nan)
    out["vwap"] = vwap

    try:
        band_mult = float(c.get("vwap_band_multiplier", DEFAULTS["vwap_band_multiplier"]))
    except Exception:
        band_mult = DEFAULTS["vwap_band_multiplier"]

    vwap_std = out["close"].rolling(window=lookback, min_periods=lookback // 2).std()
    out["vwap_std"] = vwap_std
    out["vwap_upper"] = vwap + (vwap_std * band_mult)
    out["vwap_lower"] = vwap - (vwap_std * band_mult)

    # ADX trend strength
    out["adx"] = adx(out, period=int(c.get("adx_period", DEFAULTS["adx_period"])))

    cols = ["rsi", "atr", "ema21", "ema50", "ema200", "ema_fast", "ema_mid", "ema_slow"]
    cols += ["vwap", "vwap_std", "vwap_upper", "vwap_lower", "adx"]

    cfg_dict = cfg if isinstance(cfg, dict) else {}
    adv = cfg_dict.get("advanced_volatility")
    if adv is None:
        ind_block = cfg_dict.get("indicators")
        adv = ind_block.get("advanced_volatility") if isinstance(ind_block, dict) else {}
    if isinstance(adv, dict) and bool(adv.get("enabled", False)):
        tf = out.attrs.get("timeframe")
        enabled_tfs = adv.get("enabled_timeframes", [])
        if isinstance(enabled_tfs, str):
            enabled_tfs = [x.strip() for x in enabled_tfs.split(",") if x.strip()]
        allow_without_tf = bool(adv.get("allow_without_timeframe", False))
        if not enabled_tfs:
            logger.debug("[ADV-VOL] enabled_timeframes empty; skipping advanced volatility compute")
        elif tf is None and not allow_without_tf:
            logger.debug("[ADV-VOL] attrs.timeframe missing and allow_without_timeframe=false; skipping")
        elif tf is not None and tf not in enabled_tfs:
            logger.debug("[ADV-VOL] timeframe=%s not in enabled_timeframes=%s; skipping", tf, enabled_tfs)
        else:
            try:
                window = int(adv.get("window", 14) or 14)
                ddof = int(adv.get("ddof", 1) or 1)

                if window < 2:
                    logger.debug("[ADV-VOL] window < 2 (window=%s); skipping", window)
                elif ddof < 0 or ddof >= window:
                    logger.debug("[ADV-VOL] invalid ddof (ddof=%s window=%s); skipping", ddof, window)
                else:
                    vol = VolatilityEstimators.compute_all(out, window=window, ddof=ddof)
                    out["vol_rs_bps"] = vol.vol_rs_bps
                    out["vol_gk_bps"] = vol.vol_gk_bps
                    out["vol_yz_bps"] = vol.vol_yz_bps
                    out["vol_atr_bps"] = vol.vol_atr_bps
                    out["vol_std_bps"] = vol.vol_std_bps
                    cols += ["vol_rs_bps", "vol_gk_bps", "vol_yz_bps", "vol_atr_bps", "vol_std_bps"]
            except Exception:
                logger.exception("[ADV-VOL] compute_all failed (tf=%s)", tf)
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    return out
