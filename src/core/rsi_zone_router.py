from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.indicators import rsi as calc_rsi


class RsiZone(str, Enum):
    OVERSOLD = "OVERSOLD"
    TRANSITION_LOW = "TRANSITION_LOW"
    MR = "MR"
    TRANSITION_HIGH = "TRANSITION_HIGH"
    OVERBOUGHT = "OVERBOUGHT"


@dataclass
class RsiZoneSnapshot:
    symbol: str
    ts_ms: int
    rsi_slow: float
    rsi_fast: Optional[float]
    mode: str
    ob_threshold: float
    str_threshold: float
    zone: RsiZone
    transition_width: float
    version: str
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
    ts_val = None
    try:
        ts_val = df.index[-1]
    except Exception:
        ts_val = None
    try:
        if isinstance(ts_val, pd.Timestamp):
            dt = ts_val.to_pydatetime()
        elif isinstance(ts_val, datetime):
            dt = ts_val
        else:
            return int(datetime.now(timezone.utc).timestamp() * 1000)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return int(datetime.now(timezone.utc).timestamp() * 1000)


def _extract_latest_rsi(df: Optional[pd.DataFrame], *, period: int = 14) -> Optional[float]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    candidate: Optional[float] = None
    try:
        if "rsi" in df.columns:
            raw = _coerce_finite_float(df["rsi"].iloc[-1])
            if raw is not None:
                candidate = raw
    except Exception:
        candidate = None
    if candidate is None:
        try:
            if "close" in df.columns:
                close_series = pd.to_numeric(df["close"], errors="coerce")
                rsi_series = calc_rsi(close_series, period=max(2, int(period)))
                if isinstance(rsi_series, pd.Series) and not rsi_series.empty:
                    raw = _coerce_finite_float(rsi_series.iloc[-1])
                    if raw is not None:
                        candidate = raw
        except Exception:
            candidate = None
    if candidate is None:
        return None
    if candidate < 0.0 or candidate > 100.0:
        return None
    return float(candidate)


def _normalize_thresholds(ob_thr: float, str_thr: float, cfg: Dict[str, Any]) -> Tuple[float, float, bool]:
    thresholds_cfg = cfg.get("thresholds", {}) if isinstance(cfg.get("thresholds"), dict) else {}
    ob_floor = _coerce_finite_float(thresholds_cfg.get("ob_floor"))
    ob_cap = _coerce_finite_float(thresholds_cfg.get("ob_cap"))
    str_floor = _coerce_finite_float(thresholds_cfg.get("str_floor"))
    str_cap = _coerce_finite_float(thresholds_cfg.get("str_cap"))
    min_gap = _coerce_finite_float(thresholds_cfg.get("min_gap"))

    ob_floor = 10.0 if ob_floor is None else float(ob_floor)
    ob_cap = 45.0 if ob_cap is None else float(ob_cap)
    str_floor = 55.0 if str_floor is None else float(str_floor)
    str_cap = 90.0 if str_cap is None else float(str_cap)
    min_gap = 8.0 if min_gap is None else float(min_gap)

    if ob_cap < ob_floor:
        ob_cap = ob_floor
    if str_cap < str_floor:
        str_cap = str_floor
    min_gap = max(1e-6, float(min_gap))

    ob = float(max(ob_floor, min(ob_cap, ob_thr)))
    st = float(max(str_floor, min(str_cap, str_thr)))

    min_gap_applied = False
    if (st - ob) < min_gap:
        mid = (ob + st) / 2.0
        ob = mid - (min_gap / 2.0)
        st = mid + (min_gap / 2.0)
        ob = float(max(ob_floor, min(ob_cap, ob)))
        st = float(max(str_floor, min(str_cap, st)))
        if (st - ob) < min_gap:
            st = min(str_cap, ob + min_gap)
            if st < str_floor:
                st = str_floor
                ob = max(ob_floor, st - min_gap)
        min_gap_applied = True
    return float(ob), float(st), bool(min_gap_applied)


def compute_effective_thresholds(
    *,
    symbol: str,
    regime_data: Optional[Dict[str, Any]],
    ob_strategy: Any,
    str_strategy: Any,
    router_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    regime_payload = regime_data if isinstance(regime_data, dict) else {}
    ob_thr = None
    st_thr = None

    if ob_strategy is not None:
        try:
            getter = getattr(ob_strategy, "get_symbol_specific_threshold", None)
            if callable(getter):
                ob_thr = _coerce_finite_float(getter(symbol))
        except Exception:
            ob_thr = None
        if ob_thr is None:
            try:
                adaptive_getter = getattr(ob_strategy, "get_adaptive_rsi_threshold", None)
                if callable(adaptive_getter):
                    ob_thr = _coerce_finite_float(adaptive_getter(regime_payload))
            except Exception:
                ob_thr = None

    if str_strategy is not None:
        try:
            getter = getattr(str_strategy, "get_symbol_specific_threshold", None)
            if callable(getter):
                st_thr = _coerce_finite_float(getter(symbol))
        except Exception:
            st_thr = None
        if st_thr is None:
            try:
                adaptive_getter = getattr(str_strategy, "get_adaptive_rsi_threshold", None)
                if callable(adaptive_getter):
                    st_thr = _coerce_finite_float(adaptive_getter(regime_payload))
            except Exception:
                st_thr = None

    if ob_thr is None:
        ob_thr = 35.0
    if st_thr is None:
        st_thr = 65.0

    ob, st, min_gap_applied = _normalize_thresholds(float(ob_thr), float(st_thr), router_cfg)
    return {
        "ob_threshold": float(ob),
        "str_threshold": float(st),
        "min_gap_applied": bool(min_gap_applied),
    }


def _classify_zone(*, rsi: float, ob_threshold: float, str_threshold: float, width: float) -> RsiZone:
    if rsi <= ob_threshold:
        return RsiZone.OVERSOLD
    if rsi < (ob_threshold + width):
        return RsiZone.TRANSITION_LOW
    if rsi >= str_threshold:
        return RsiZone.OVERBOUGHT
    if rsi > (str_threshold - width):
        return RsiZone.TRANSITION_HIGH
    return RsiZone.MR


def _zone_group(zone: RsiZone) -> str:
    if zone == RsiZone.OVERSOLD:
        return "oversold"
    if zone == RsiZone.OVERBOUGHT:
        return "overbought"
    if zone == RsiZone.MR:
        return "mr"
    return "transition"


def resolve_zone(
    *,
    symbol: str,
    rsi_slow: float,
    rsi_fast: Optional[float],
    ts_ms: int,
    thresholds: Dict[str, Any],
    router_cfg: Dict[str, Any],
) -> RsiZoneSnapshot:
    source_cfg = router_cfg.get("source", {}) if isinstance(router_cfg.get("source"), dict) else {}
    transition_cfg = router_cfg.get("transition", {}) if isinstance(router_cfg.get("transition"), dict) else {}
    mode = str(source_cfg.get("mode", "slow_only") or "slow_only").strip().lower()
    if mode not in {"slow_only", "consensus"}:
        mode = "slow_only"
    width = _coerce_finite_float(transition_cfg.get("width"))
    if width is None or width < 0:
        width = 5.0

    ob_threshold = float(thresholds.get("ob_threshold", 35.0))
    str_threshold = float(thresholds.get("str_threshold", 65.0))
    slow_zone = _classify_zone(
        rsi=float(rsi_slow),
        ob_threshold=ob_threshold,
        str_threshold=str_threshold,
        width=float(width),
    )
    chosen_zone = slow_zone
    consensus_status = "slow_only"
    fast_zone = None

    if mode == "consensus" and rsi_fast is not None and math.isfinite(float(rsi_fast)):
        fast_zone = _classify_zone(
            rsi=float(rsi_fast),
            ob_threshold=ob_threshold,
            str_threshold=str_threshold,
            width=float(width),
        )
        slow_group = _zone_group(slow_zone)
        fast_group = _zone_group(fast_zone)
        if slow_group == fast_group and slow_group in {"oversold", "mr", "overbought"}:
            chosen_zone = slow_zone
            consensus_status = "aligned"
        else:
            consensus_status = "mismatch_transition"
            if float(rsi_slow) < ob_threshold + width:
                chosen_zone = RsiZone.TRANSITION_LOW
            elif float(rsi_slow) > str_threshold - width:
                chosen_zone = RsiZone.TRANSITION_HIGH
            else:
                mid = (ob_threshold + str_threshold) / 2.0
                chosen_zone = RsiZone.TRANSITION_LOW if float(rsi_slow) <= mid else RsiZone.TRANSITION_HIGH

    meta = {
        "slow_zone": slow_zone.value,
        "fast_zone": fast_zone.value if isinstance(fast_zone, RsiZone) else None,
        "consensus_status": consensus_status,
        "min_gap_applied": bool(thresholds.get("min_gap_applied", False)),
    }
    return RsiZoneSnapshot(
        symbol=str(symbol),
        ts_ms=int(ts_ms),
        rsi_slow=float(rsi_slow),
        rsi_fast=float(rsi_fast) if rsi_fast is not None else None,
        mode=mode,
        ob_threshold=float(ob_threshold),
        str_threshold=float(str_threshold),
        zone=chosen_zone,
        transition_width=float(width),
        version="rsi_zone_router_v1",
        meta=meta,
    )


def build_rsi_zone_snapshot(
    *,
    symbol: str,
    df_slow: Optional[pd.DataFrame],
    df_fast: Optional[pd.DataFrame],
    regime_data: Optional[Dict[str, Any]],
    ob_strategy: Any,
    str_strategy: Any,
    router_cfg: Dict[str, Any],
) -> Optional[RsiZoneSnapshot]:
    if not isinstance(router_cfg, dict):
        return None
    if not bool(router_cfg.get("enabled", False)):
        return None

    thresholds = compute_effective_thresholds(
        symbol=str(symbol),
        regime_data=regime_data if isinstance(regime_data, dict) else {},
        ob_strategy=ob_strategy,
        str_strategy=str_strategy,
        router_cfg=router_cfg,
    )
    rsi_slow = _extract_latest_rsi(df_slow, period=14)
    if rsi_slow is None:
        return None
    rsi_fast = _extract_latest_rsi(df_fast, period=14)
    ts_ms = _extract_ts_ms(df_slow)
    return resolve_zone(
        symbol=str(symbol),
        rsi_slow=float(rsi_slow),
        rsi_fast=float(rsi_fast) if rsi_fast is not None else None,
        ts_ms=int(ts_ms),
        thresholds=thresholds,
        router_cfg=router_cfg,
    )


def snapshot_to_dict(snapshot: Optional[RsiZoneSnapshot]) -> Optional[Dict[str, Any]]:
    if snapshot is None:
        return None
    out = asdict(snapshot)
    out["zone"] = snapshot.zone.value
    return out


def _extract_zone(snapshot: Any) -> Optional[str]:
    if snapshot is None:
        return None
    if isinstance(snapshot, RsiZoneSnapshot):
        return snapshot.zone.value
    if isinstance(snapshot, dict):
        raw = snapshot.get("zone")
        try:
            if raw is None:
                return None
            zone = str(raw).strip().upper()
        except Exception:
            return None
        if zone:
            return zone
    return None


def _normalize_strategy_name(name: Any) -> str:
    try:
        return str(name or "").strip().lower()
    except Exception:
        return ""


def is_strategy_allowed(
    strategy_name: Any,
    side: Any,
    snapshot: Any,
    router_cfg: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    del side  # side-aware constraints are intentionally out-of-scope for v1.

    cfg = router_cfg if isinstance(router_cfg, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return True, "rsi_router.disabled"

    zone = _extract_zone(snapshot)
    if not zone:
        return True, "rsi_router.snapshot_missing"

    transition_cfg = cfg.get("transition", {}) if isinstance(cfg.get("transition"), dict) else {}
    no_trade_new_entry = bool(transition_cfg.get("no_trade_new_entry", True))
    if no_trade_new_entry and zone in {RsiZone.TRANSITION_LOW.value, RsiZone.TRANSITION_HIGH.value}:
        return False, "rsi_router.transition_no_trade"

    normalized = _normalize_strategy_name(strategy_name)
    ob_names = {"adaptive_ob", "oversold_bounce"}
    str_names = {"adaptive_str", "short_the_rip", "adaptive_short_the_rip"}
    mr_names = {"mean_reversion", "mr"}

    if normalized in ob_names:
        if zone != RsiZone.OVERSOLD.value:
            return False, "rsi_router.zone_mismatch"
        return True, "rsi_router.allowed"

    if normalized in str_names:
        if zone != RsiZone.OVERBOUGHT.value:
            return False, "rsi_router.zone_mismatch"
        return True, "rsi_router.allowed"

    if normalized in mr_names:
        if zone != RsiZone.MR.value:
            return False, "rsi_router.zone_mismatch"
        return True, "rsi_router.allowed"

    return True, "rsi_router.strategy_unknown_allow"

