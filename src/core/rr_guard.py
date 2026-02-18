"""
Shared R/R gate helpers used by pre-fill and post-fill checks.
"""

from typing import Any, Dict, Optional

RR_REASON_BELOW_1 = "rr_below_1"
RR_REASON_BELOW_REQUIRED = "rr_below_required"
DEFAULT_RR_FLOOR = 1.0


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_rr_required(rr_required: Any) -> Optional[float]:
    normalized = _safe_float(rr_required)
    if normalized is None or normalized <= 0:
        return None
    return normalized


def normalize_rr_floor(rr_floor: Any = DEFAULT_RR_FLOOR) -> float:
    normalized = _safe_float(rr_floor)
    if normalized is None or normalized <= 0:
        return DEFAULT_RR_FLOOR
    return normalized


def evaluate_rr_gate(
    rr_actual: Any,
    *,
    rr_required: Any = None,
    rr_required_source: Optional[str] = None,
    rr_floor: Any = DEFAULT_RR_FLOOR,
    action_on_fail: str = "early_exit",
) -> Dict[str, Any]:
    normalized_rr_actual = _safe_float(rr_actual)
    if normalized_rr_actual is not None and normalized_rr_actual <= 0:
        normalized_rr_actual = None

    normalized_rr_required = normalize_rr_required(rr_required)
    normalized_rr_floor = normalize_rr_floor(rr_floor)

    action = "keep"
    reason_code = None
    if normalized_rr_actual is not None:
        if normalized_rr_actual < normalized_rr_floor:
            action = action_on_fail
            reason_code = RR_REASON_BELOW_1
        elif normalized_rr_required is not None and normalized_rr_actual < normalized_rr_required:
            action = action_on_fail
            reason_code = RR_REASON_BELOW_REQUIRED

    return {
        "rr_actual": normalized_rr_actual,
        "rr_required": normalized_rr_required,
        "rr_required_source": rr_required_source if normalized_rr_required is not None else None,
        "rr_floor": normalized_rr_floor,
        "action": action,
        "reason_code": reason_code,
    }


def build_prefill_rr_reason_code(reason_code: Optional[str]) -> Optional[str]:
    if not reason_code:
        return None
    return f"risk.rr.pre_fill.{reason_code}"

