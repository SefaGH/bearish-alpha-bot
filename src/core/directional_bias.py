from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _coerce_finite_float(value: Any, *, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _normalize_side(side: Any) -> str:
    try:
        raw = str(side or "").strip().lower()
    except Exception:
        raw = ""
    if raw in {"buy", "long"}:
        return "long"
    if raw in {"sell", "short"}:
        return "short"
    return ""


def _extract_level_zone_snapshot(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(signal, dict):
        return None
    direct = signal.get("level_zone_snapshot")
    if isinstance(direct, dict):
        return direct
    meta = signal.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("level_zone_snapshot"), dict):
        return meta.get("level_zone_snapshot")
    return None


def _normalize_symbol(value: Any) -> str:
    try:
        return str(value or "").strip().upper()
    except Exception:
        return ""


def _parse_canary_symbols(value: Any) -> list[str]:
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    return []


def _resolve_rollout_mode(signal: Dict[str, Any], cfg: Dict[str, Any]) -> str:
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

    symbol_norm = _normalize_symbol(signal.get("symbol"))
    allowed = {_normalize_symbol(token) for token in canary_tokens}
    if symbol_norm and symbol_norm in allowed:
        return mode
    return "off"


def compute_directional_bias_adjustment(signal: Dict[str, Any], cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg if isinstance(cfg, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "applied": False,
            "delta": 0.0,
            "bias_score": 0.0,
            "confidence": 0.0,
            "zone": None,
            "reason": "directional_bias.disabled",
        }

    mode = str(cfg.get("mode", "quality_adjust_only") or "quality_adjust_only").strip().lower()
    if mode not in {"quality_adjust_only", "off", "disabled"}:
        mode = "quality_adjust_only"
    if mode in {"off", "disabled"}:
        return {
            "enabled": True,
            "applied": False,
            "delta": 0.0,
            "bias_score": 0.0,
            "confidence": 0.0,
            "zone": None,
            "reason": "directional_bias.mode_disabled",
        }

    rollout_mode = _resolve_rollout_mode(signal, cfg)
    if rollout_mode == "off":
        return {
            "enabled": True,
            "applied": False,
            "delta": 0.0,
            "bias_score": 0.0,
            "confidence": 0.0,
            "zone": None,
            "reason": "directional_bias.rollout_out_of_scope",
        }

    snapshot = _extract_level_zone_snapshot(signal)
    if not isinstance(snapshot, dict):
        return {
            "enabled": True,
            "applied": False,
            "delta": 0.0,
            "bias_score": 0.0,
            "confidence": 0.0,
            "zone": None,
            "reason": "directional_bias.snapshot_missing",
        }

    zone = str(snapshot.get("zone") or "").strip().upper()
    max_delta = abs(_coerce_finite_float(cfg.get("max_quality_delta"), default=0.08) or 0.08)
    weight = abs(_coerce_finite_float(cfg.get("weight"), default=0.10) or 0.10)
    at_level_penalty = abs(_coerce_finite_float(cfg.get("at_level_penalty"), default=0.05) or 0.05)

    bias_score = 0.0
    confidence = 0.0
    if zone == "BREAKOUT_UP_CONFIRMED":
        bias_score = 0.8
        confidence = 0.9
    elif zone == "BREAKOUT_DOWN_CONFIRMED":
        bias_score = -0.8
        confidence = 0.9
    elif zone == "IN_RANGE":
        bias_score = 0.0
        confidence = 0.4
    elif zone == "AT_LEVEL":
        bias_score = 0.0
        confidence = 0.2
    elif zone == "UNKNOWN":
        bias_score = 0.0
        confidence = 0.0

    side = _normalize_side(signal.get("side"))

    # AT_LEVEL is an uncertainty regime; apply a small deterministic quality penalty.
    if zone == "AT_LEVEL":
        delta = -min(max_delta, at_level_penalty)
        if rollout_mode == "observe":
            return {
                "enabled": True,
                "applied": False,
                "delta": 0.0,
                "would_delta": float(delta),
                "bias_score": float(bias_score),
                "confidence": float(confidence),
                "zone": zone,
                "reason": "directional_bias.observe_only",
            }
        return {
            "enabled": True,
            "applied": bool(abs(delta) > 1e-12),
            "delta": float(delta),
            "bias_score": float(bias_score),
            "confidence": float(confidence),
            "zone": zone,
            "reason": "directional_bias.at_level_penalty",
        }

    alignment = 0
    if side == "long":
        alignment = 1 if bias_score > 0 else (-1 if bias_score < 0 else 0)
    elif side == "short":
        alignment = 1 if bias_score < 0 else (-1 if bias_score > 0 else 0)

    raw_delta = float(abs(bias_score) * confidence * weight * alignment)
    delta = max(-max_delta, min(max_delta, raw_delta))

    if rollout_mode == "observe":
        return {
            "enabled": True,
            "applied": False,
            "delta": 0.0,
            "would_delta": float(delta),
            "bias_score": float(bias_score),
            "confidence": float(confidence),
            "zone": zone or None,
            "reason": "directional_bias.observe_only",
        }

    reason = "directional_bias.no_effect"
    if alignment == 1 and abs(delta) > 0:
        reason = "directional_bias.aligned_boost"
    elif alignment == -1 and abs(delta) > 0:
        reason = "directional_bias.countertrend_penalty"
    elif zone == "UNKNOWN":
        reason = "directional_bias.unknown_zone"

    return {
        "enabled": True,
        "applied": bool(abs(delta) > 1e-12),
        "delta": float(delta),
        "bias_score": float(bias_score),
        "confidence": float(confidence),
        "zone": zone or None,
        "reason": reason,
    }
