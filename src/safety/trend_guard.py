from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import logging
import time

import numpy as np
import pandas as pd

try:  # Optional acceleration if available
    import talib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    talib = None

try:  # Optional indicator backend
    import pandas_ta as pta  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pta = None

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    is_vetoed: bool
    reason: str
    meta_data: Dict[str, Any] = field(default_factory=dict)


class TrendGuard:
    """
    Fast Trend Validation (Squeeze + Breakout + Slope) guard.

    This module is designed to be a veto layer before risk sizing.
    It is self-calibrated using recent history so the thresholds adapt
    to each symbol/timeframe.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "enabled": True,
        "default_timeframe": "5m",
        "lookback_bars": 500,
        "min_history_bars": 200,
        "update_every_bars": 25,
        "update_every_seconds": 300,
        "bb_period": 20,
        "bb_std": 2.0,
        "bbw_squeeze_quantile": 0.20,
        "bbw_expand_quantile": 0.80,
        "bbw_expand_lookback": 20,
        "squeeze_lookback": 20,
        "slope_ema_period": 50,
        "slope_lookback": 5,
        "slope_quantile": 0.85,
        "slope_use_atr": True,
        "slope_atr_period": 14,
        "min_body_ratio": 0.0,
        "apply_to_strategies": [],
        "apply_to_signal_types": [],
        "allow_missing_signal_type": True,
    }

    REQUIRED_COLS = ("open", "high", "low", "close", "volume")

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(self.DEFAULT_CONFIG)
        if isinstance(config, dict):
            self.config.update(config)

        self.enabled = bool(self.config.get("enabled", True))
        self._threshold_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self._apply_to_strategies = {
            str(s).lower() for s in (self.config.get("apply_to_strategies") or [])
        }
        self._apply_to_signal_types = {
            str(s).upper() for s in (self.config.get("apply_to_signal_types") or [])
        }

    def should_check(self, strategy_name: str, signal: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False

        if self._apply_to_strategies:
            if str(strategy_name or "").lower() not in self._apply_to_strategies:
                return False

        if self._apply_to_signal_types:
            signal_type = ""
            if isinstance(signal, dict):
                signal_type = str(signal.get("signal_type") or signal.get("strategy_type") or "").upper()
            if not signal_type:
                return bool(self.config.get("allow_missing_signal_type", True))
            if signal_type not in self._apply_to_signal_types:
                return False

        return True

    def resolve_timeframe(self, signal: Optional[Dict[str, Any]] = None) -> str:
        if isinstance(signal, dict):
            tf = signal.get("timeframe")
            if isinstance(tf, str) and tf.strip():
                return tf.strip()
        return str(self.config.get("default_timeframe", "5m"))

    def check_veto(
        self,
        symbol: str,
        side: str,
        current_candle: Optional[Any],
        dataframe: pd.DataFrame,
        timeframe: Optional[str] = None,
    ) -> GuardResult:
        if not self.enabled:
            return GuardResult(False, "trend_guard_disabled", {})

        if dataframe is None or not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            return GuardResult(False, "trend_guard_no_data", {"symbol": symbol})

        if not self._has_required_columns(dataframe):
            return GuardResult(False, "trend_guard_missing_columns", {"symbol": symbol})

        tf = str(timeframe or self.resolve_timeframe({"timeframe": timeframe}))
        df = self._slice_history(dataframe)
        if len(df) < int(self.config.get("min_history_bars", 200)):
            return GuardResult(False, "trend_guard_insufficient_history", {"bars": len(df)})

        thresholds = self._get_or_recalculate_thresholds(symbol, tf, df)
        if not thresholds or not self._thresholds_valid(thresholds):
            return GuardResult(False, "trend_guard_uncalibrated", {"symbol": symbol, "timeframe": tf})

        features = self._compute_features(df)
        last = features["last"]
        if last is None:
            return GuardResult(False, "trend_guard_invalid_last", {"symbol": symbol, "timeframe": tf})

        bbw = last["bbw"]
        bbw_ratio = last["bbw_ratio"]
        slope = last["slope"]
        upper = last["upper"]
        lower = last["lower"]
        close = last["close"]
        body_ratio = last["body_ratio"]

        squeeze_recent = features["bbw"].tail(int(self.config["squeeze_lookback"])).le(
            thresholds["bbw_squeeze"]
        ).any()

        breakout_dir = None
        min_body_ratio = float(self.config.get("min_body_ratio", 0.0) or 0.0)
        body_ok = True if min_body_ratio <= 0 else body_ratio >= min_body_ratio

        if (
            squeeze_recent
            and close > upper
            and bbw_ratio >= thresholds["bbw_expand"]
            and slope >= thresholds["slope_up"]
            and body_ok
        ):
            breakout_dir = "up"
        elif (
            squeeze_recent
            and close < lower
            and bbw_ratio >= thresholds["bbw_expand"]
            and slope <= -thresholds["slope_dn"]
            and body_ok
        ):
            breakout_dir = "down"

        side_norm = self._normalize_side(side)
        vetoed = False
        reason = "trend_guard_pass"

        if breakout_dir == "up" and side_norm == "short":
            vetoed = True
            reason = "trend_guard_veto_short_breakout_up"
        elif breakout_dir == "down" and side_norm == "long":
            vetoed = True
            reason = "trend_guard_veto_long_breakout_down"

        meta = {
            "symbol": symbol,
            "timeframe": tf,
            "side": side_norm,
            "squeeze_recent": bool(squeeze_recent),
            "breakout_dir": breakout_dir,
            "bbw": bbw,
            "bbw_ratio": bbw_ratio,
            "bbw_squeeze_thr": thresholds["bbw_squeeze"],
            "bbw_expand_thr": thresholds["bbw_expand"],
            "slope": slope,
            "slope_up_thr": thresholds["slope_up"],
            "slope_dn_thr": thresholds["slope_dn"],
            "close": close,
            "upper": upper,
            "lower": lower,
            "body_ratio": body_ratio,
        }

        return GuardResult(vetoed, reason, meta)

    def _has_required_columns(self, df: pd.DataFrame) -> bool:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            logger.debug("TrendGuard missing columns: %s", missing)
            return False
        return True

    def _slice_history(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.config.get("lookback_bars", 500))
        extra = int(self.config.get("bb_period", 20)) + int(self.config.get("slope_ema_period", 50))
        extra += int(self.config.get("slope_lookback", 5)) + int(self.config.get("slope_atr_period", 14))
        total = max(lookback, int(self.config.get("min_history_bars", 200))) + extra + 5
        return df.tail(total).copy()

    def _get_or_recalculate_thresholds(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, float]:
        key = (str(symbol), str(timeframe))
        state = self._threshold_cache.get(key)
        now = time.time()
        update_every_bars = int(self.config.get("update_every_bars", 25))
        update_every_seconds = int(self.config.get("update_every_seconds", 300))
        needs_update = state is None

        if state is not None:
            last_len = int(state.get("last_len", 0))
            last_ts = float(state.get("updated_at", 0.0))
            if update_every_bars > 0 and len(df) - last_len >= update_every_bars:
                needs_update = True
            if update_every_seconds > 0 and (now - last_ts) >= update_every_seconds:
                needs_update = True

        if not needs_update and state:
            return dict(state.get("thresholds") or {})

        thresholds = self._calculate_dynamic_thresholds(df)
        if not thresholds and state:
            return dict(state.get("thresholds") or {})

        if thresholds:
            self._threshold_cache[key] = {
                "thresholds": thresholds,
                "updated_at": now,
                "last_len": len(df),
            }
        return thresholds

    def _calculate_dynamic_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        features = self._compute_features(df)
        bbw = features["bbw"].dropna()
        slope = features["slope"].dropna()

        if bbw.empty or slope.empty:
            return {}

        q_squeeze = float(self.config.get("bbw_squeeze_quantile", 0.20))
        q_expand = float(self.config.get("bbw_expand_quantile", 0.80))
        q_slope = float(self.config.get("slope_quantile", 0.85))

        bbw_squeeze = float(np.nanquantile(bbw.values, q_squeeze))

        expand_lb = int(self.config.get("bbw_expand_lookback", 20))
        bbw_med = float(np.nanmedian(bbw.tail(expand_lb).values))
        if not np.isfinite(bbw_med) or bbw_med <= 0:
            bbw_expand = float(np.nanquantile(bbw.values, q_expand))
        else:
            bbw_ratio = (bbw / bbw_med).replace([np.inf, -np.inf], np.nan).dropna()
            bbw_expand = float(np.nanquantile(bbw_ratio.values, q_expand)) if not bbw_ratio.empty else bbw_squeeze

        slope_pos = slope[slope > 0.0]
        slope_neg = slope[slope < 0.0]
        slope_up = float(np.nanquantile(slope_pos.values, q_slope)) if not slope_pos.empty else np.nan
        slope_dn = float(np.nanquantile((-slope_neg).values, q_slope)) if not slope_neg.empty else np.nan

        return {
            "bbw_squeeze": bbw_squeeze,
            "bbw_expand": bbw_expand,
            "slope_up": slope_up,
            "slope_dn": slope_dn,
        }

    @staticmethod
    def _thresholds_valid(thresholds: Dict[str, float]) -> bool:
        keys = ("bbw_squeeze", "bbw_expand", "slope_up", "slope_dn")
        for k in keys:
            val = thresholds.get(k)
            if val is None or not np.isfinite(float(val)):
                return False
        if thresholds["slope_up"] <= 0 or thresholds["slope_dn"] <= 0:
            return False
        if thresholds["bbw_squeeze"] <= 0 or thresholds["bbw_expand"] <= 0:
            return False
        return True

    def _compute_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")

        upper, mid, lower = self._bbands(close)
        bbw = (upper - lower) / mid.replace(0, np.nan)
        bbw = bbw.replace([np.inf, -np.inf], np.nan)

        ema_period = int(self.config.get("slope_ema_period", 50))
        slope_lookback = int(self.config.get("slope_lookback", 5))
        ema = self._ema(close, ema_period)
        ema_shift = ema.shift(slope_lookback)

        if bool(self.config.get("slope_use_atr", True)):
            atr = self._atr(df, int(self.config.get("slope_atr_period", 14)))
            denom = (atr * float(slope_lookback)).replace(0, np.nan)
        else:
            denom = (close * float(slope_lookback)).replace(0, np.nan)

        slope = (ema - ema_shift) / denom
        slope = slope.replace([np.inf, -np.inf], np.nan)

        med = float(np.nanmedian(bbw.tail(int(self.config.get("bbw_expand_lookback", 20))).values))
        if not np.isfinite(med) or med <= 0:
            bbw_ratio = bbw * 0.0 + np.nan
        else:
            bbw_ratio = (bbw / med).replace([np.inf, -np.inf], np.nan)

        body = (close - open_).abs()
        rng = (high - low).replace(0, np.nan)
        body_ratio = (body / rng).replace([np.inf, -np.inf], np.nan)

        last = None
        if len(df) > 0:
            last = {
                "close": float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan,
                "upper": float(upper.iloc[-1]) if np.isfinite(upper.iloc[-1]) else np.nan,
                "lower": float(lower.iloc[-1]) if np.isfinite(lower.iloc[-1]) else np.nan,
                "bbw": float(bbw.iloc[-1]) if np.isfinite(bbw.iloc[-1]) else np.nan,
                "bbw_ratio": float(bbw_ratio.iloc[-1]) if np.isfinite(bbw_ratio.iloc[-1]) else np.nan,
                "slope": float(slope.iloc[-1]) if np.isfinite(slope.iloc[-1]) else np.nan,
                "body_ratio": float(body_ratio.iloc[-1]) if np.isfinite(body_ratio.iloc[-1]) else np.nan,
            }

        return {
            "close": close,
            "upper": upper,
            "lower": lower,
            "mid": mid,
            "bbw": bbw,
            "bbw_ratio": bbw_ratio,
            "slope": slope,
            "last": last,
        }

    def _bbands(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        period = int(self.config.get("bb_period", 20))
        std = float(self.config.get("bb_std", 2.0))

        if talib is not None:
            upper, mid, lower = talib.BBANDS(
                close.values, timeperiod=period, nbdevup=std, nbdevdn=std, matype=0
            )
            return (
                pd.Series(upper, index=close.index),
                pd.Series(mid, index=close.index),
                pd.Series(lower, index=close.index),
            )

        if pta is not None:
            bb = pta.bbands(close, length=period, std=std)
            if isinstance(bb, pd.DataFrame) and not bb.empty:
                upper_col = next((c for c in bb.columns if c.startswith("BBU_")), None)
                mid_col = next((c for c in bb.columns if c.startswith("BBM_")), None)
                lower_col = next((c for c in bb.columns if c.startswith("BBL_")), None)
                if upper_col and mid_col and lower_col:
                    return bb[upper_col], bb[mid_col], bb[lower_col]

        mid = close.rolling(window=period, min_periods=period).mean()
        std_series = close.rolling(window=period, min_periods=period).std()
        upper = mid + std_series * std
        lower = mid - std_series * std
        return upper, mid, lower

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        period = max(int(period), 1)
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        period = max(int(period), 1)
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _normalize_side(side: str) -> str:
        val = str(side or "").lower()
        if val in ("buy", "long"):
            return "long"
        if val in ("sell", "short"):
            return "short"
        return ""
