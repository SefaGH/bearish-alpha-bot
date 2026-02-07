from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class SmartEntryDecision:
    applied: bool
    reason: str


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _safe_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    return None


def _side_to_direction(side: str) -> Optional[str]:
    s = str(side or "").lower().strip()
    if s in {"buy", "long"}:
        return "LONG"
    if s in {"sell", "short"}:
        return "SHORT"
    return None


def apply_smart_entry_policy(
    *,
    signal: Dict[str, Any],
    execution_params: Dict[str, Any],
    policy_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], SmartEntryDecision]:
    """Inject Smart Entry params into the signal + execution_params.

    Safety principles:
    - If required inputs are missing (ATR/entry/side), do not place LIMIT.
    - If volatility is below threshold, force MARKET (even if global config is limit).
    - Do not override explicit upstream order_type/limit_price unless policy_cfg.force_override is true.

    Expected policy_cfg shape:
      {
        "enabled": True,
        "volatility_threshold_bps": 5.0,
        "params": {
          "LONG": {"atr_multiplier": 0.90, "timeout_seconds": 300, "gate_bps": 5.0},
          "SHORT": {"atr_multiplier": 0.85, "timeout_seconds": 240, "gate_bps": 12.0},
        },
        "force_override": False
      }
    """

    enabled = bool(policy_cfg.get("enabled"))
    if not enabled:
        return signal, execution_params, SmartEntryDecision(applied=False, reason="disabled")

    force_override = bool(policy_cfg.get("force_override", False))

    # Respect explicit upstream instructions by default.
    explicit_order_type = execution_params.get("order_type") or execution_params.get("type")
    explicit_limit = signal.get("limit_price") or signal.get("execution_price")
    if not force_override and (explicit_order_type or explicit_limit):
        return signal, execution_params, SmartEntryDecision(applied=False, reason="explicit_execution_override")

    direction = _side_to_direction(str(signal.get("side") or ""))
    if direction is None:
        # Unknown side => safest is market.
        execution_params["order_type"] = "market"
        return signal, execution_params, SmartEntryDecision(applied=False, reason="unknown_side_force_market")

    entry = _safe_float(signal.get("entry"))
    if entry is None or entry <= 0:
        execution_params["order_type"] = "market"
        return signal, execution_params, SmartEntryDecision(applied=False, reason="missing_entry_force_market")

    force_market_on_missing_atr = _safe_bool(policy_cfg.get("force_market_on_missing_atr"))
    if force_market_on_missing_atr is None:
        force_market_on_missing_atr = True
    force_market_on_low_vol = _safe_bool(policy_cfg.get("force_market_on_low_vol"))
    if force_market_on_low_vol is None:
        force_market_on_low_vol = True
    extreme_market_ban = bool(_safe_bool(policy_cfg.get("extreme_market_ban")) or False)

    meta = signal.get("meta") if isinstance(signal.get("meta"), dict) else {}
    vol_tel = meta.get("vol_telemetry") if isinstance(meta.get("vol_telemetry"), dict) else {}
    vol_block = signal.get("volatility") if isinstance(signal.get("volatility"), dict) else {}
    bucket = str(
        signal.get("volume_bucket")
        or meta.get("volume_bucket")
        or vol_tel.get("bucket")
        or vol_block.get("bucket")
        or ""
    ).upper().strip()
    is_extreme_bucket = bucket == "EXTREME"

    def _inject_conservative_limit(reason: str) -> Tuple[Dict[str, Any], Dict[str, Any], SmartEntryDecision]:
        fallback_timeout_s = _safe_float(
            policy_cfg.get("fallback_timeout_seconds")
            or policy_cfg.get("fallback_timeout_s")
            or policy_cfg.get("fallback_limit_timeout_seconds")
        )
        if fallback_timeout_s is None or fallback_timeout_s <= 0:
            fallback_timeout_s = 30.0
        signal["limit_price"] = float(entry)
        signal["_execution_price_locked"] = True
        execution_params["order_type"] = "limit"
        execution_params["timeout_seconds"] = float(fallback_timeout_s)
        execution_params.setdefault("poll_interval_s", 1.0)
        execution_params["market_fallback"] = False
        execution_params["market_fallback_on_timeout_enabled"] = False
        if is_extreme_bucket:
            execution_params["disable_market_fallback_on_extreme_bucket"] = True
        return signal, execution_params, SmartEntryDecision(applied=False, reason=reason)

    atr = _safe_float(signal.get("atr"))
    if atr is None:
        sizing_meta = signal.get("sizing_meta")
        if isinstance(sizing_meta, dict):
            atr = _safe_float(sizing_meta.get("atr"))

    if atr is None or atr <= 0:
        if (not force_market_on_missing_atr) or (extreme_market_ban and is_extreme_bucket):
            reason = "missing_atr_conservative_limit"
            if extreme_market_ban and is_extreme_bucket:
                reason = "missing_atr_conservative_limit_extreme"
            return _inject_conservative_limit(reason)
        execution_params["order_type"] = "market"
        return signal, execution_params, SmartEntryDecision(applied=False, reason="missing_atr_force_market")

    vol_bps = (atr / entry) * 10000.0
    thr_bps = _safe_float(policy_cfg.get("volatility_threshold_bps"))
    thr_bps = 5.0 if thr_bps is None else float(thr_bps)

    if vol_bps < thr_bps:
        if (not force_market_on_low_vol) or (extreme_market_ban and is_extreme_bucket):
            reason = f"low_vol_conservative_limit:{vol_bps:.2f}<{thr_bps:.2f}"
            if extreme_market_ban and is_extreme_bucket:
                reason = f"low_vol_conservative_limit_extreme:{vol_bps:.2f}<{thr_bps:.2f}"
            return _inject_conservative_limit(reason)
        # Low vol => force MARKET.
        execution_params["order_type"] = "market"
        # Ensure we don't accidentally pass a stale limit price.
        signal.pop("limit_price", None)
        signal.pop("execution_price", None)
        return signal, execution_params, SmartEntryDecision(applied=False, reason=f"low_vol_force_market:{vol_bps:.2f}<{thr_bps:.2f}")

    params_block = policy_cfg.get("params")
    if not isinstance(params_block, dict):
        return signal, execution_params, SmartEntryDecision(applied=False, reason="missing_params_block")

    dir_params = params_block.get(direction)
    if not isinstance(dir_params, dict):
        return signal, execution_params, SmartEntryDecision(applied=False, reason=f"missing_direction_params:{direction}")

    k = _safe_float(dir_params.get("atr_multiplier"))
    timeout_s = _safe_float(dir_params.get("timeout_seconds") or dir_params.get("timeout_s") or dir_params.get("time_in_force_seconds"))
    gate_bps = _safe_float(dir_params.get("gate_bps") or dir_params.get("max_chase_bps"))

    if k is None or k <= 0:
        return signal, execution_params, SmartEntryDecision(applied=False, reason="invalid_k")

    if timeout_s is None or timeout_s <= 0:
        return signal, execution_params, SmartEntryDecision(applied=False, reason="invalid_timeout")

    if gate_bps is None or gate_bps < 0:
        return signal, execution_params, SmartEntryDecision(applied=False, reason="invalid_gate")

    if direction == "LONG":
        limit_price = entry - (k * atr)
    else:
        limit_price = entry + (k * atr)

    if limit_price <= 0:
        return signal, execution_params, SmartEntryDecision(applied=False, reason="invalid_limit_price")

    # Inject limit configuration.
    signal["limit_price"] = float(limit_price)
    signal["_execution_price_locked"] = True

    execution_params["order_type"] = "limit"
    execution_params["timeout_seconds"] = float(timeout_s)
    execution_params["max_chase_bps"] = float(gate_bps)
    execution_params["market_fallback"] = True
    execution_params.setdefault("poll_interval_s", 1.0)
    for key in (
        "market_fallback_on_timeout_enabled",
        "disable_market_fallback_on_extreme_bucket",
        "disable_market_fallback_on_fast_move",
    ):
        parsed = _safe_bool(policy_cfg.get(key))
        if parsed is not None:
            execution_params[key] = parsed
    if extreme_market_ban and is_extreme_bucket:
        execution_params["market_fallback_on_timeout_enabled"] = False
        execution_params["disable_market_fallback_on_extreme_bucket"] = True

    signal.setdefault("smart_entry_meta", {})
    if isinstance(signal["smart_entry_meta"], dict):
        signal["smart_entry_meta"].update(
            {
                "applied": True,
                "direction": direction,
                "vol_bps": float(vol_bps),
                "threshold_bps": float(thr_bps),
                "k": float(k),
                "timeout_seconds": float(timeout_s),
                "gate_bps": float(gate_bps),
                "limit_price": float(limit_price),
            }
        )

    return signal, execution_params, SmartEntryDecision(applied=True, reason="applied")
