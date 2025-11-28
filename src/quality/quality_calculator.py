"""Quality calculator: computes a numeric quality score and detailed breakdown.
Returns a dict:
{
  "value": float,
  "components": { "ml_component": 0.12, ... },
  "reason": ["ml_component_below_threshold", ...]
}
"""
from typing import Dict, Any

DEFAULT_FALLBACK = {
    "ml_component": 0.10,
    "volume_component": 0.05,
    "momentum_component": 0.05,
    "spread_component": 0.05,
}

WEIGHTS = {
    "ml_component": 0.60,
    "volume_component": 0.20,
    "momentum_component": 0.15,
    "spread_component": 0.05,
}

MIN_COMPONENT_THRESHOLDS = {
    "ml_component": 0.10,
    "volume_component": 0.10,
    "momentum_component": 0.05,
    "spread_component": 0.01,
}

def _normalize(value, min_v, max_v):
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if max_v == min_v:
        return 0.0
    return max(0.0, min(1.0, (v - min_v) / (max_v - min_v)))


def compute_quality(features: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    features expected keys (any may be missing):
      - ml_component (0..1)
      - volume_component (raw, normalized here)
      - momentum_component (raw)
      - spread_component (raw)
    """
    components: Dict[str, float] = {}
    reasons = []

    # ML component: assume already normalized 0..1 from ML pipeline; fallback if missing
    ml_raw = features.get("ml_component")
    if ml_raw is None:
        components["ml_component"] = DEFAULT_FALLBACK["ml_component"]
        reasons.append("ml_component_missing_fallback_used")
        if logger:
            logger.debug("quality_input_missing - component=ml_component")
    else:
        try:
            components["ml_component"] = float(ml_raw)
        except Exception:
            components["ml_component"] = DEFAULT_FALLBACK["ml_component"]
            reasons.append("ml_component_parse_error")
            if logger:
                logger.debug(f"quality_input_parse_error - component=ml_component raw={ml_raw}")

    # volume_component: typical raw scale 0..2 => normalize 0..1 using clamp
    vol_raw = features.get("volume_component")
    if vol_raw is None:
        components["volume_component"] = DEFAULT_FALLBACK["volume_component"]
        reasons.append("volume_component_missing_fallback_used")
        if logger:
            logger.debug("quality_input_missing - component=volume_component")
    else:
        n = _normalize(vol_raw, 0.0, 2.0)
        components["volume_component"] = n if n is not None else DEFAULT_FALLBACK["volume_component"]

    # momentum_component: normalize roughly -5..+5 -> 0..1
    mom_raw = features.get("momentum_component")
    if mom_raw is None:
        components["momentum_component"] = DEFAULT_FALLBACK["momentum_component"]
        reasons.append("momentum_component_missing_fallback_used")
        if logger:
            logger.debug("quality_input_missing - component=momentum_component")
    else:
        n = _normalize(mom_raw, -5.0, 5.0)
        components["momentum_component"] = n if n is not None else DEFAULT_FALLBACK["momentum_component"]

    # spread_component: normalize 0..1
    sp_raw = features.get("spread_component")
    if sp_raw is None:
        components["spread_component"] = DEFAULT_FALLBACK["spread_component"]
        reasons.append("spread_component_missing_fallback_used")
        if logger:
            logger.debug("quality_input_missing - component=spread_component")
    else:
        n = _normalize(sp_raw, 0.0, 1.0)
        components["spread_component"] = n if n is not None else DEFAULT_FALLBACK["spread_component"]

    # Clamp and round components
    for k in list(components.keys()):
        v = components[k]
        try:
            components[k] = round(float(v), 4)
        except Exception:
            components[k] = 0.0

    # Weighted sum
    quality_value = 0.0
    for k, w in WEIGHTS.items():
        quality_value += components.get(k, 0.0) * w
    quality_value = round(quality_value, 4)

    # If near-zero, append human-readable reasons based on thresholds
    if quality_value <= 0.001:
        if components.get("ml_component", 0.0) < MIN_COMPONENT_THRESHOLDS["ml_component"]:
            reasons.append("ml_component_below_threshold")
        if components.get("volume_component", 0.0) < MIN_COMPONENT_THRESHOLDS["volume_component"]:
            reasons.append("volume_insufficient")
        if components.get("momentum_component", 0.0) < MIN_COMPONENT_THRESHOLDS["momentum_component"]:
            reasons.append("momentum_weak")
        if components.get("spread_component", 0.0) < MIN_COMPONENT_THRESHOLDS["spread_component"]:
            reasons.append("spread_unfavorable")

    # Deduplicate reasons
    reasons = list(dict.fromkeys(reasons))

    out = {
        "value": quality_value,
        "components": components,
        "reason": reasons,
    }

    if logger:
        logger.debug(f"quality_computed - {out}")

    return out
