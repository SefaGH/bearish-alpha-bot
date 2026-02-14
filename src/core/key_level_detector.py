from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from core.indicators import atr as calc_atr
from core.resistance_band import Band, compute_band


@dataclass(frozen=True)
class KeyLevels:
    symbol: str
    timeframe: str
    ts_ms: int
    price: float
    nearest_resistance: Optional[Band]
    nearest_support: Optional[Band]
    distance_to_resistance_bps: Optional[float]
    distance_to_support_bps: Optional[float]
    position_in_range: Optional[float]
    range_width_atr: Optional[float]
    touch_count_resistance: int
    touch_count_support: int
    state: str
    reason: str
    meta: Dict[str, Any]


def _coerce_finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _extract_ts_ms(df: Optional[pd.DataFrame]) -> int:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    try:
        ts = df.index[-1]
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return int(ts.timestamp() * 1000)
    except Exception:
        pass
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def band_to_dict(band: Optional[Band]) -> Optional[Dict[str, Any]]:
    if band is None:
        return None
    return {
        "level": float(band.level),
        "band_low": float(band.band_low),
        "band_high": float(band.band_high),
        "method": str(band.method),
        "timeframe": str(band.timeframe),
        "meta": dict(band.meta or {}),
    }


def key_levels_to_dict(levels: Optional[KeyLevels]) -> Optional[Dict[str, Any]]:
    if levels is None:
        return None
    return {
        "symbol": str(levels.symbol),
        "timeframe": str(levels.timeframe),
        "ts_ms": int(levels.ts_ms),
        "price": float(levels.price),
        "nearest_resistance": band_to_dict(levels.nearest_resistance),
        "nearest_support": band_to_dict(levels.nearest_support),
        "distance_to_resistance_bps": levels.distance_to_resistance_bps,
        "distance_to_support_bps": levels.distance_to_support_bps,
        "position_in_range": levels.position_in_range,
        "range_width_atr": levels.range_width_atr,
        "touch_count_resistance": int(levels.touch_count_resistance),
        "touch_count_support": int(levels.touch_count_support),
        "state": str(levels.state),
        "reason": str(levels.reason),
        "meta": dict(levels.meta or {}),
    }


class KeyLevelDetector:
    DEFAULT_CONFIG: Dict[str, Any] = {
        "method": "smc",  # smc | kmeans
        "pivot_left": 5,
        "pivot_right": 3,
        "lookback_bars": 200,
        "band_pct": 0.005,
        "smc_cluster_pct": 0.01,
        "min_cluster_n": 2,
        "kmin": 2,
        "kmax": 8,
        "random_state": 42,
        "touch_proximity_bps": 30.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, market_data_pipeline: Any = None) -> None:
        self.config: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        if isinstance(config, dict):
            self.config.update(config)
        self.market_data_pipeline = market_data_pipeline

    async def detect_levels(
        self,
        *,
        symbol: str,
        price: float,
        timeframe: str = "1h",
        limit: int = 300,
        market_data_pipeline: Any = None,
    ) -> KeyLevels:
        pipeline = market_data_pipeline if market_data_pipeline is not None else self.market_data_pipeline
        if pipeline is None or not hasattr(pipeline, "get_latest_ohlcv"):
            return self._empty_levels(
                symbol=symbol,
                timeframe=timeframe,
                price=price,
                reason="missing_market_data_pipeline",
            )
        try:
            df = await pipeline.get_latest_ohlcv(symbol, timeframe, limit=limit, include_forming=False)
        except Exception:
            df = None
        return self.detect_from_df(symbol=symbol, timeframe=timeframe, df=df, price=price)

    def detect_from_df(
        self,
        *,
        symbol: str,
        timeframe: str,
        df: Optional[pd.DataFrame],
        price: float,
    ) -> KeyLevels:
        px = _coerce_finite_float(price)
        if px is None or px <= 0:
            return self._empty_levels(
                symbol=symbol,
                timeframe=timeframe,
                price=0.0,
                reason="invalid_price",
            )
        if not isinstance(df, pd.DataFrame) or df.empty:
            return self._empty_levels(
                symbol=symbol,
                timeframe=timeframe,
                price=float(px),
                reason="missing_dataframe",
            )

        df_used = self._as_closed_df(df)
        if df_used.empty:
            return self._empty_levels(
                symbol=symbol,
                timeframe=timeframe,
                price=float(px),
                reason="empty_after_closed_only",
            )

        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(set(df_used.columns)):
            return self._empty_levels(
                symbol=symbol,
                timeframe=timeframe,
                price=float(px),
                reason="missing_required_columns",
            )

        cfg = self.config
        resistance = compute_band(
            df=df_used,
            timeframe=str(timeframe),
            price=float(px),
            side="buy",
            method=str(cfg.get("method", "smc") or "smc"),
            pivot_left=int(cfg.get("pivot_left", 5) or 5),
            pivot_right=int(cfg.get("pivot_right", 3) or 3),
            lookback_bars=int(cfg.get("lookback_bars", 200) or 200),
            band_pct=float(cfg.get("band_pct", 0.005) or 0.005),
            smc_cluster_pct=float(cfg.get("smc_cluster_pct", 0.01) or 0.01),
            min_cluster_n=int(cfg.get("min_cluster_n", 2) or 2),
            kmin=int(cfg.get("kmin", 2) or 2),
            kmax=int(cfg.get("kmax", 8) or 8),
            random_state=int(cfg.get("random_state", 42) or 42),
        )
        support = compute_band(
            df=df_used,
            timeframe=str(timeframe),
            price=float(px),
            side="sell",
            method=str(cfg.get("method", "smc") or "smc"),
            pivot_left=int(cfg.get("pivot_left", 5) or 5),
            pivot_right=int(cfg.get("pivot_right", 3) or 3),
            lookback_bars=int(cfg.get("lookback_bars", 200) or 200),
            band_pct=float(cfg.get("band_pct", 0.005) or 0.005),
            smc_cluster_pct=float(cfg.get("smc_cluster_pct", 0.01) or 0.01),
            min_cluster_n=int(cfg.get("min_cluster_n", 2) or 2),
            kmin=int(cfg.get("kmin", 2) or 2),
            kmax=int(cfg.get("kmax", 8) or 8),
            random_state=int(cfg.get("random_state", 42) or 42),
        )

        dist_res_bps = self._distance_to_level_bps(price=float(px), level=resistance.level, direction="up") if resistance else None
        dist_sup_bps = self._distance_to_level_bps(price=float(px), level=support.level, direction="down") if support else None

        atr_value = self._latest_atr(df_used)
        pos_in_range = None
        range_width_atr = None
        if resistance is not None and support is not None:
            range_width = float(resistance.level) - float(support.level)
            if math.isfinite(range_width) and range_width > 0:
                pos = (float(px) - float(support.level)) / range_width
                pos_in_range = float(max(0.0, min(1.0, pos)))
                if atr_value is not None and atr_value > 0:
                    range_width_atr = float(range_width / atr_value)

        touch_prox_bps = float(cfg.get("touch_proximity_bps", 30.0) or 30.0)
        touch_res = self._count_touches(df_used, float(resistance.level), touch_prox_bps) if resistance else 0
        touch_sup = self._count_touches(df_used, float(support.level), touch_prox_bps) if support else 0

        state = "ok" if (resistance is not None or support is not None) else "unknown"
        reason = "ok" if state == "ok" else "levels_not_found"
        return KeyLevels(
            symbol=str(symbol),
            timeframe=str(timeframe),
            ts_ms=_extract_ts_ms(df_used),
            price=float(px),
            nearest_resistance=resistance,
            nearest_support=support,
            distance_to_resistance_bps=dist_res_bps,
            distance_to_support_bps=dist_sup_bps,
            position_in_range=pos_in_range,
            range_width_atr=range_width_atr,
            touch_count_resistance=int(touch_res),
            touch_count_support=int(touch_sup),
            state=state,
            reason=reason,
            meta={
                "touch_proximity_bps": touch_prox_bps,
                "has_resistance": bool(resistance is not None),
                "has_support": bool(support is not None),
            },
        )

    @staticmethod
    def _as_closed_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        includes_forming = False
        try:
            includes_forming = bool(getattr(df, "attrs", {}).get("includes_forming", False))
        except Exception:
            includes_forming = False
        if includes_forming and len(df) >= 2:
            return df.iloc[:-1].copy()
        return df.copy()

    @staticmethod
    def _latest_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        try:
            if "atr" in df.columns:
                atr_val = _coerce_finite_float(df["atr"].iloc[-1])
                if atr_val is not None and atr_val > 0:
                    return float(atr_val)
        except Exception:
            pass
        try:
            atr_series = calc_atr(df, period=max(2, int(period)))
            if isinstance(atr_series, pd.Series) and not atr_series.empty:
                atr_val = _coerce_finite_float(atr_series.iloc[-1])
                if atr_val is not None and atr_val > 0:
                    return float(atr_val)
        except Exception:
            return None
        return None

    @staticmethod
    def _distance_to_level_bps(*, price: float, level: float, direction: str) -> Optional[float]:
        if price <= 0:
            return None
        lvl = _coerce_finite_float(level)
        if lvl is None or lvl <= 0:
            return None
        if direction == "up":
            return float((float(lvl) - float(price)) / float(price) * 10000.0)
        if direction == "down":
            return float((float(price) - float(lvl)) / float(price) * 10000.0)
        return None

    @staticmethod
    def _count_touches(df: pd.DataFrame, level: float, touch_proximity_bps: float) -> int:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return 0
        if level <= 0:
            return 0
        if "high" not in df.columns or "low" not in df.columns:
            return 0
        count = 0
        tolerance = float(abs(level) * (float(touch_proximity_bps) / 10000.0))
        if tolerance <= 0:
            return 0
        highs = pd.to_numeric(df["high"], errors="coerce").fillna(float("nan")).to_numpy(dtype=float)
        lows = pd.to_numeric(df["low"], errors="coerce").fillna(float("nan")).to_numpy(dtype=float)
        for i in range(len(highs)):
            hi = highs[i]
            lo = lows[i]
            if not (math.isfinite(hi) and math.isfinite(lo)):
                continue
            if abs(float(hi) - float(level)) <= tolerance or abs(float(lo) - float(level)) <= tolerance:
                count += 1
        return int(count)

    def _empty_levels(
        self,
        *,
        symbol: str,
        timeframe: str,
        price: float,
        reason: str,
    ) -> KeyLevels:
        return KeyLevels(
            symbol=str(symbol),
            timeframe=str(timeframe),
            ts_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            price=float(price),
            nearest_resistance=None,
            nearest_support=None,
            distance_to_resistance_bps=None,
            distance_to_support_bps=None,
            position_in_range=None,
            range_width_atr=None,
            touch_count_resistance=0,
            touch_count_support=0,
            state="unknown",
            reason=str(reason),
            meta={},
        )
