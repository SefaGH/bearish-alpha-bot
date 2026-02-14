from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.key_level_detector import KeyLevelDetector, key_levels_to_dict


class LevelZone(str, Enum):
    AT_LEVEL = "AT_LEVEL"
    IN_RANGE = "IN_RANGE"
    BREAKOUT_UP_CONFIRMED = "BREAKOUT_UP_CONFIRMED"
    BREAKOUT_DOWN_CONFIRMED = "BREAKOUT_DOWN_CONFIRMED"
    UNKNOWN = "UNKNOWN"


@dataclass
class LevelZoneSnapshot:
    symbol: str
    ts_ms: int
    price: float
    mode: str
    zone: LevelZone
    version: str
    primary_timeframe: str
    zones_by_timeframe: Dict[str, str]
    key_levels_by_timeframe: Dict[str, Dict[str, Any]]
    meta: Dict[str, Any]


def _coerce_finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _extract_price(df: Optional[pd.DataFrame]) -> Optional[float]:
    if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
        return None
    try:
        close = _coerce_finite_float(df["close"].iloc[-1])
    except Exception:
        close = None
    if close is None or close <= 0:
        return None
    return float(close)


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


def _normalize_timeframes(value: Any) -> List[str]:
    default = ["15m", "1h"]
    if isinstance(value, str):
        out = [part.strip() for part in value.split(",") if part.strip()]
        return out if out else default
    if isinstance(value, (list, tuple)):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out if out else default
    return default


def _normalize_mode(source_cfg: Dict[str, Any]) -> str:
    mode = str(source_cfg.get("mode", "consensus") or "consensus").strip().lower()
    if mode not in {"single_tf", "consensus"}:
        mode = "consensus"
    return mode


def _volume_breakout_confirmed(df: pd.DataFrame, min_volume_mult: float) -> bool:
    if min_volume_mult <= 0:
        return True
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "volume" not in df.columns:
        return False
    vol_series = pd.to_numeric(df["volume"], errors="coerce").dropna()
    if len(vol_series) < 20:
        return False
    vol_now = _coerce_finite_float(vol_series.iloc[-1])
    vol_ma = _coerce_finite_float(vol_series.tail(20).mean())
    if vol_now is None or vol_ma is None or vol_ma <= 0:
        return False
    return bool(float(vol_now) >= float(vol_ma) * float(min_volume_mult))


def _breakout_confirmed(
    *,
    closes: pd.Series,
    level: Optional[float],
    min_close_bars: int,
    direction: str,
) -> bool:
    if level is None or not math.isfinite(float(level)):
        return False
    if closes is None or closes.empty:
        return False
    bars = max(int(min_close_bars), 1)
    if len(closes) < bars:
        return False
    tail = pd.to_numeric(closes.tail(bars), errors="coerce").dropna()
    if len(tail) < bars:
        return False
    if direction == "up":
        return bool((tail > float(level)).all())
    if direction == "down":
        return bool((tail < float(level)).all())
    return False


def _classify_single_timeframe(
    *,
    df: Optional[pd.DataFrame],
    key_levels: Dict[str, Any],
    zones_cfg: Dict[str, Any],
    breakout_cfg: Dict[str, Any],
) -> Tuple[LevelZone, str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return LevelZone.UNKNOWN, "missing_df"
    if not isinstance(key_levels, dict):
        return LevelZone.UNKNOWN, "missing_key_levels"
    if str(key_levels.get("state") or "unknown").lower() != "ok":
        return LevelZone.UNKNOWN, f"levels_state:{key_levels.get('state')}"

    near_level_bps = float(zones_cfg.get("near_level_bps", 50.0) or 50.0)
    min_close_bars = int(breakout_cfg.get("min_close_bars", 2) or 2)
    min_volume_mult = float(breakout_cfg.get("min_volume_mult", 1.5) or 1.5)

    close_series = pd.to_numeric(df["close"], errors="coerce").dropna() if "close" in df.columns else pd.Series(dtype=float)

    resistance = key_levels.get("nearest_resistance") if isinstance(key_levels.get("nearest_resistance"), dict) else {}
    support = key_levels.get("nearest_support") if isinstance(key_levels.get("nearest_support"), dict) else {}
    resistance_level = _coerce_finite_float(resistance.get("level"))
    support_level = _coerce_finite_float(support.get("level"))

    # Fallback levels for breakout checks when nearest-level lookup does not return
    # a usable reference on one side of current price.
    if min_close_bars > 0 and isinstance(df, pd.DataFrame) and len(df) > min_close_bars:
        if resistance_level is None and "high" in df.columns:
            try:
                hist_high = pd.to_numeric(df["high"].iloc[:-min_close_bars], errors="coerce").dropna()
                if not hist_high.empty:
                    resistance_level = _coerce_finite_float(hist_high.max())
            except Exception:
                resistance_level = None
        if support_level is None and "low" in df.columns:
            try:
                hist_low = pd.to_numeric(df["low"].iloc[:-min_close_bars], errors="coerce").dropna()
                if not hist_low.empty:
                    support_level = _coerce_finite_float(hist_low.min())
            except Exception:
                support_level = None

    breakout_up = _breakout_confirmed(
        closes=close_series,
        level=resistance_level,
        min_close_bars=min_close_bars,
        direction="up",
    )
    breakout_down = _breakout_confirmed(
        closes=close_series,
        level=support_level,
        min_close_bars=min_close_bars,
        direction="down",
    )

    if breakout_up and _volume_breakout_confirmed(df, min_volume_mult):
        return LevelZone.BREAKOUT_UP_CONFIRMED, "breakout_up_confirmed"
    if breakout_down and _volume_breakout_confirmed(df, min_volume_mult):
        return LevelZone.BREAKOUT_DOWN_CONFIRMED, "breakout_down_confirmed"

    dist_res = _coerce_finite_float(key_levels.get("distance_to_resistance_bps"))
    dist_sup = _coerce_finite_float(key_levels.get("distance_to_support_bps"))
    near_res = dist_res is not None and abs(float(dist_res)) <= float(near_level_bps)
    near_sup = dist_sup is not None and abs(float(dist_sup)) <= float(near_level_bps)
    if near_res or near_sup:
        return LevelZone.AT_LEVEL, "near_level"

    pos_in_range = _coerce_finite_float(key_levels.get("position_in_range"))
    if pos_in_range is not None and 0.0 <= float(pos_in_range) <= 1.0:
        return LevelZone.IN_RANGE, "in_range"

    return LevelZone.UNKNOWN, "zone_unresolved"


def _resolve_consensus_zone(evals: List[Tuple[str, LevelZone, str]], mode: str) -> Tuple[LevelZone, Dict[str, Any]]:
    if not evals:
        return LevelZone.UNKNOWN, {"consensus_status": "no_timeframe_eval"}

    if mode == "single_tf":
        tf, zone, reason = evals[0]
        return zone, {"consensus_status": "single_tf", "primary_reason": reason, "primary_timeframe": tf}

    zones = [item[1] for item in evals if item[1] != LevelZone.UNKNOWN]
    if not zones:
        return LevelZone.UNKNOWN, {"consensus_status": "all_unknown"}

    if any(z == LevelZone.AT_LEVEL for z in zones):
        return LevelZone.AT_LEVEL, {"consensus_status": "at_level_present"}

    unique = set(zones)
    if len(unique) == 1:
        return zones[0], {"consensus_status": "aligned"}

    if LevelZone.BREAKOUT_UP_CONFIRMED in unique and LevelZone.BREAKOUT_DOWN_CONFIRMED in unique:
        return LevelZone.AT_LEVEL, {"consensus_status": "breakout_conflict"}

    if LevelZone.BREAKOUT_UP_CONFIRMED in unique and LevelZone.BREAKOUT_DOWN_CONFIRMED not in unique:
        return LevelZone.BREAKOUT_UP_CONFIRMED, {"consensus_status": "breakout_up_bias"}

    if LevelZone.BREAKOUT_DOWN_CONFIRMED in unique and LevelZone.BREAKOUT_UP_CONFIRMED not in unique:
        return LevelZone.BREAKOUT_DOWN_CONFIRMED, {"consensus_status": "breakout_down_bias"}

    if LevelZone.IN_RANGE in unique:
        return LevelZone.IN_RANGE, {"consensus_status": "mixed_to_in_range"}

    return LevelZone.UNKNOWN, {"consensus_status": "unresolved"}


def build_level_zone_snapshot(
    *,
    symbol: str,
    price: Optional[float],
    dfs_by_timeframe: Dict[str, Optional[pd.DataFrame]],
    router_cfg: Optional[Dict[str, Any]],
) -> Optional[LevelZoneSnapshot]:
    cfg = router_cfg if isinstance(router_cfg, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return None

    source_cfg = cfg.get("source", {}) if isinstance(cfg.get("source"), dict) else {}
    levels_cfg = cfg.get("levels", {}) if isinstance(cfg.get("levels"), dict) else {}
    zones_cfg = cfg.get("zones", {}) if isinstance(cfg.get("zones"), dict) else {}
    breakout_cfg = cfg.get("breakout", {}) if isinstance(cfg.get("breakout"), dict) else {}

    timeframes = _normalize_timeframes(source_cfg.get("timeframes"))
    mode = _normalize_mode(source_cfg)
    detector = KeyLevelDetector(levels_cfg)

    evals: List[Tuple[str, LevelZone, str]] = []
    levels_by_tf: Dict[str, Dict[str, Any]] = {}
    zones_by_tf: Dict[str, str] = {}
    resolved_price = _coerce_finite_float(price)
    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for tf in timeframes:
        df = dfs_by_timeframe.get(tf) if isinstance(dfs_by_timeframe, dict) else None
        if not isinstance(df, pd.DataFrame) or df.empty:
            zones_by_tf[str(tf)] = LevelZone.UNKNOWN.value
            continue

        if resolved_price is None:
            resolved_price = _extract_price(df)
        if resolved_price is None or resolved_price <= 0:
            zones_by_tf[str(tf)] = LevelZone.UNKNOWN.value
            continue

        levels = detector.detect_from_df(
            symbol=symbol,
            timeframe=str(tf),
            df=df,
            price=float(resolved_price),
        )
        level_dict = key_levels_to_dict(levels) or {}
        levels_by_tf[str(tf)] = level_dict
        zone, reason = _classify_single_timeframe(
            df=df,
            key_levels=level_dict,
            zones_cfg=zones_cfg,
            breakout_cfg=breakout_cfg,
        )
        zones_by_tf[str(tf)] = zone.value
        evals.append((str(tf), zone, reason))
        ts_ms = max(int(ts_ms), int(_extract_ts_ms(df)))

    if resolved_price is None or resolved_price <= 0:
        return None

    zone, consensus_meta = _resolve_consensus_zone(evals, mode)
    primary_timeframe = str(timeframes[0]) if timeframes else "n/a"
    if evals:
        primary_timeframe = str(evals[0][0])

    return LevelZoneSnapshot(
        symbol=str(symbol),
        ts_ms=int(ts_ms),
        price=float(resolved_price),
        mode=mode,
        zone=zone,
        version="level_zone_router_v1",
        primary_timeframe=primary_timeframe,
        zones_by_timeframe=dict(zones_by_tf),
        key_levels_by_timeframe=dict(levels_by_tf),
        meta={
            "timeframes": list(timeframes),
            "eval_count": int(len(evals)),
            **consensus_meta,
        },
    )


def snapshot_to_dict(snapshot: Optional[LevelZoneSnapshot]) -> Optional[Dict[str, Any]]:
    if snapshot is None:
        return None
    out = asdict(snapshot)
    out["zone"] = snapshot.zone.value
    return out


def _extract_zone(snapshot: Any) -> Optional[str]:
    if snapshot is None:
        return None
    if isinstance(snapshot, LevelZoneSnapshot):
        return snapshot.zone.value
    if isinstance(snapshot, dict):
        raw = snapshot.get("zone")
        try:
            zone = str(raw).strip().upper() if raw is not None else ""
        except Exception:
            zone = ""
        return zone or None
    return None


def _normalize_strategy_name(name: Any) -> str:
    try:
        return str(name or "").strip().lower()
    except Exception:
        return ""


def _normalize_symbol(value: Any) -> str:
    try:
        return str(value or "").strip().upper()
    except Exception:
        return ""


def _normalize_canary_symbol_token(token: Any) -> Optional[str]:
    if isinstance(token, str):
        out = token.strip()
        return out if out else None
    if isinstance(token, dict) and len(token) == 1:
        key, value = next(iter(token.items()))
        key_str = str(key).strip() if key is not None else ""
        if not key_str:
            return None
        if value is None:
            return key_str
        value_str = str(value).strip()
        if not value_str:
            return key_str
        if ":" in key_str:
            return key_str
        if "/" in key_str:
            return f"{key_str}:{value_str}"
    return None


def _parse_canary_symbols(value: Any) -> List[str]:
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for token in value:
            norm = _normalize_canary_symbol_token(token)
            if norm:
                out.append(norm)
        return out
    return []


def _resolve_rollout_mode(snapshot: Any, cfg: Dict[str, Any]) -> str:
    rollout_cfg = cfg.get("rollout", {}) if isinstance(cfg.get("rollout"), dict) else {}
    mode = str(rollout_cfg.get("mode", "enforce") or "enforce").strip().lower()
    if mode in {"off", "disabled"}:
        return "off"
    if mode not in {"observe", "enforce"}:
        mode = "enforce"

    canary_tokens = _parse_canary_symbols(rollout_cfg.get("canary_symbols"))
    if not canary_tokens:
        return mode
    if "*" in canary_tokens:
        return mode

    symbol = ""
    if isinstance(snapshot, LevelZoneSnapshot):
        symbol = snapshot.symbol
    elif isinstance(snapshot, dict):
        symbol = str(snapshot.get("symbol") or "")
    symbol_norm = _normalize_symbol(symbol)
    allowed = {_normalize_symbol(token) for token in canary_tokens}
    if symbol_norm and symbol_norm in allowed:
        return mode
    return "off"


def is_strategy_allowed(
    strategy_name: Any,
    side: Any,
    snapshot: Any,
    router_cfg: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    del side  # Side split is intentionally deferred for v1.

    cfg = router_cfg if isinstance(router_cfg, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return True, "level_router.disabled"

    rollout_mode = _resolve_rollout_mode(snapshot, cfg)
    if rollout_mode == "off":
        return True, "level_router.rollout_out_of_scope"

    def _deny_or_observe(reason_code: str) -> Tuple[bool, str]:
        if rollout_mode == "observe":
            return True, "level_router.observe_would_block"
        return False, reason_code

    zone = _extract_zone(snapshot)
    if not zone:
        return True, "level_router.snapshot_missing"
    if zone == LevelZone.UNKNOWN.value:
        return True, "level_router.unknown_fail_open"

    zones_cfg = cfg.get("zones", {}) if isinstance(cfg.get("zones"), dict) else {}
    no_trade_new_entry = bool(zones_cfg.get("no_trade_new_entry", True))
    if no_trade_new_entry and zone == LevelZone.AT_LEVEL.value:
        return _deny_or_observe("level_router.at_level")

    normalized = _normalize_strategy_name(strategy_name)
    ob_names = {"adaptive_ob", "oversold_bounce"}
    str_names = {"adaptive_str", "short_the_rip", "adaptive_short_the_rip"}
    mr_names = {"mean_reversion", "mr"}

    if normalized in ob_names:
        if zone in {LevelZone.IN_RANGE.value, LevelZone.BREAKOUT_UP_CONFIRMED.value}:
            return True, "level_router.allowed"
        return _deny_or_observe("level_router.zone_mismatch")

    if normalized in str_names:
        if zone in {LevelZone.IN_RANGE.value, LevelZone.BREAKOUT_DOWN_CONFIRMED.value}:
            return True, "level_router.allowed"
        return _deny_or_observe("level_router.zone_mismatch")

    if normalized in mr_names:
        if zone == LevelZone.IN_RANGE.value:
            return True, "level_router.allowed"
        return _deny_or_observe("level_router.zone_mismatch")

    return True, "level_router.strategy_unknown_allow"
