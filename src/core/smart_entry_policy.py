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

    atr = _safe_float(signal.get("atr"))
    if atr is None:
        sizing_meta = signal.get("sizing_meta")
        if isinstance(sizing_meta, dict):
            atr = _safe_float(sizing_meta.get("atr"))

    if atr is None or atr <= 0:
        execution_params["order_type"] = "market"
        return signal, execution_params, SmartEntryDecision(applied=False, reason="missing_atr_force_market")

    vol_bps = (atr / entry) * 10000.0
    thr_bps = _safe_float(policy_cfg.get("volatility_threshold_bps"))
    thr_bps = 5.0 if thr_bps is None else float(thr_bps)

    if vol_bps < thr_bps:
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
