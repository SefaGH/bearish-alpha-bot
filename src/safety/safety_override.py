from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


@dataclass
class GuardResult:
    is_vetoed: bool
    reason: str
    meta_data: Dict[str, Any] = field(default_factory=dict)


class SafetyOverride:
    """
    Adaptive Strategy Safety Override (2-out-of-3 gate).

    Activates only when a strategy reports that it lowered its threshold:
      current_threshold < base_threshold by at least aggressive_threshold_delta_min.

    It is designed to run inside StrategyCoordinator as a modular veto layer.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "enabled": True,
        "apply_to_strategies": [],
        "apply_to_sides": ["sell", "short"],
        "aggressive_threshold_delta_min": 0.1,
        "min_passes": 2,
        "fail_closed_on_insufficient_context": True,
        "gates": {
            "trend": {
                "enabled": True,
                "rsi_floor": 65.0,
            },
            "volume": {
                "enabled": True,
                "low_bucket": "LOW",
                "require_bearish_if_low": True,
                "volume_confirm_mult": 1.0,
            },
            "resistance": {
                "enabled": True,
                "max_distance_bps": 25.0,
            },
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(self.DEFAULT_CONFIG)
        if isinstance(config, dict):
            self.config.update(config)

        gates = self.config.get("gates", {})
        if isinstance(gates, dict):
            merged = dict(self.DEFAULT_CONFIG.get("gates") or {})
            for k, v in gates.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            self.config["gates"] = merged

        self.enabled = bool(self.config.get("enabled", True))
        self._apply_to_strategies: Set[str] = {
            str(s).lower() for s in (self.config.get("apply_to_strategies") or []) if str(s).strip()
        }
        self._apply_to_sides: Set[str] = {
            str(s).lower() for s in (self.config.get("apply_to_sides") or []) if str(s).strip()
        }

    def should_check(self, strategy_name: str, signal: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False
        if self._apply_to_strategies and str(strategy_name or "").lower() not in self._apply_to_strategies:
            return False
        if self._apply_to_sides and isinstance(signal, dict):
            side = str(signal.get("side") or "").lower()
            if side and side not in self._apply_to_sides:
                return False
        return True

    def check_veto(self, strategy_name: str, signal: Dict[str, Any]) -> GuardResult:
        if not self.should_check(strategy_name, signal):
            return GuardResult(False, "safety_override_skip", {"enabled": False})

        meta = signal.get("meta") if isinstance(signal, dict) else None
        meta = meta if isinstance(meta, dict) else {}

        thr = meta.get("adaptive_threshold") if isinstance(meta.get("adaptive_threshold"), dict) else {}
        base_thr = thr.get("base_threshold")
        cur_thr = thr.get("current_threshold")

        base_thr_f = _as_float(base_thr)
        cur_thr_f = _as_float(cur_thr)
        if base_thr_f is None or cur_thr_f is None:
            return GuardResult(
                False,
                "safety_override_no_threshold_meta",
                {"base_threshold": base_thr, "current_threshold": cur_thr},
            )

        delta = base_thr_f - cur_thr_f
        delta_min = float(self.config.get("aggressive_threshold_delta_min", 0.1) or 0.1)
        aggressive = bool(delta >= delta_min and cur_thr_f < base_thr_f)
        if not aggressive:
            return GuardResult(
                False,
                "safety_override_inactive",
                {"base_threshold": base_thr_f, "current_threshold": cur_thr_f, "delta": delta},
            )

        snapshot = meta.get("safety_snapshot") if isinstance(meta.get("safety_snapshot"), dict) else {}

        gates_cfg = self.config.get("gates", {}) if isinstance(self.config.get("gates"), dict) else {}

        gate_results: Dict[str, Dict[str, Any]] = {}
        passes = []
        fails = []
        na = []

        # Gate A: Trend Respect (EMA stack)
        if bool((gates_cfg.get("trend") or {}).get("enabled", True)):
            res = self._gate_trend(snapshot, gates_cfg.get("trend") or {})
            gate_results["trend"] = res
            code = res.get("code") or "trend"
            (passes if res["pass"] else (na if res["na"] else fails)).append(code)

        # Gate B: Volume / candle confirmation
        if bool((gates_cfg.get("volume") or {}).get("enabled", True)):
            res = self._gate_volume(signal=signal, snapshot=snapshot, cfg=gates_cfg.get("volume") or {})
            gate_results["volume"] = res
            code = res.get("code") or "volume"
            (passes if res["pass"] else (na if res["na"] else fails)).append(code)

        # Gate C: Resistance proximity
        if bool((gates_cfg.get("resistance") or {}).get("enabled", True)):
            res = self._gate_resistance(snapshot, gates_cfg.get("resistance") or {})
            gate_results["resistance"] = res
            code = res.get("code") or "resistance"
            (passes if res["pass"] else (na if res["na"] else fails)).append(code)

        applicable = 3 - len(na)
        min_passes = int(self.config.get("min_passes", 2) or 2)
        fail_closed = bool(self.config.get("fail_closed_on_insufficient_context", True))

        if applicable < min_passes and fail_closed:
            meta_out = {
                "reason": "safety_override.insufficient_context",
                "aggressive": True,
                "base_threshold": base_thr_f,
                "current_threshold": cur_thr_f,
                "delta": delta,
                "score": f"{len(passes)}/{applicable}",
                "required": min_passes,
                "passes": passes,
                "fails": fails,
                "na": na,
                "gates": gate_results,
            }
            return GuardResult(True, "safety_override.insufficient_context", meta_out)

        is_blocked = len(passes) < min_passes
        reason = "safety_override.blocked" if is_blocked else "safety_override.pass"
        meta_out = {
            "reason": reason,
            "aggressive": True,
            "base_threshold": base_thr_f,
            "current_threshold": cur_thr_f,
            "delta": delta,
            "score": f"{len(passes)}/{applicable}",
            "required": min_passes,
            "passes": passes,
            "fails": fails,
            "na": na,
            "gates": gate_results,
        }
        return GuardResult(is_blocked, reason, meta_out)

    @staticmethod
    def _gate_trend(snapshot: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
        close = _as_float(snapshot.get("close"))
        ema21 = _as_float(snapshot.get("ema21"))
        ema50 = _as_float(snapshot.get("ema50"))
        rsi = _as_float(snapshot.get("rsi"))

        if close is None or ema21 is None or ema50 is None or rsi is None:
            return {"pass": False, "na": True, "code": "trend_missing", "meta": {"missing": True}}

        stack_up = bool(close > ema21 > ema50)
        rsi_floor = float(cfg.get("rsi_floor", 65.0) or 65.0)
        if stack_up:
            ok = bool(rsi >= rsi_floor)
            return {
                "pass": ok,
                "na": False,
                "code": "trend_ok" if ok else "trend_mismatch",
                "meta": {"stack_up": True, "rsi_floor": rsi_floor},
            }
        return {"pass": True, "na": False, "code": "trend_ok", "meta": {"stack_up": False, "rsi_floor": rsi_floor}}

    @staticmethod
    def _gate_volume(*, signal: Dict[str, Any], snapshot: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
        low_bucket = str(cfg.get("low_bucket", "LOW"))
        side_bucket = str(signal.get("volume_bucket") or snapshot.get("volume_bucket") or "").upper()
        require_bearish = bool(cfg.get("require_bearish_if_low", True))
        vol_mult = float(cfg.get("volume_confirm_mult", 1.0) or 1.0)

        if not side_bucket:
            # If bucket isn't known, try raw volume vs ma.
            side_bucket = ""

        if side_bucket and side_bucket != str(low_bucket).upper():
            return {
                "pass": True,
                "na": False,
                "code": "volume_ok",
                "meta": {"volume_bucket": side_bucket, "low_bucket": low_bucket},
            }

        open_ = _as_float(snapshot.get("candle_open"))
        close = _as_float(snapshot.get("candle_close"))
        vol = _as_float(snapshot.get("volume"))
        vol_ma = _as_float(snapshot.get("volume_ma20"))

        bearish_ok = None
        if require_bearish:
            if open_ is None or close is None:
                bearish_ok = None
            else:
                bearish_ok = bool(close < open_)
        else:
            bearish_ok = True

        vol_ok = None
        if vol is None or vol_ma is None or vol_ma <= 0:
            vol_ok = None
        else:
            vol_ok = bool(vol >= (vol_ma * vol_mult))

        if bearish_ok is None and vol_ok is None:
            return {
                "pass": False,
                "na": True,
                "code": "volume_missing",
                "meta": {"volume_bucket": side_bucket or None, "missing": True, "low_bucket": low_bucket},
            }

        ok = bool((bearish_ok is True) or (vol_ok is True))
        return {
            "pass": ok,
            "na": False,
            "code": "volume_ok" if ok else "no_volume_confirm",
            "meta": {
                "volume_bucket": side_bucket or None,
                "low_bucket": low_bucket,
                "require_bearish": require_bearish,
                "volume_confirm_mult": vol_mult,
                "bearish_candle": bearish_ok,
                "volume_ok": vol_ok,
            },
        }

    @staticmethod
    def _gate_resistance(snapshot: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
        dist_bps = _as_float(snapshot.get("resistance_distance_bps"))
        if dist_bps is None:
            return {"pass": False, "na": True, "code": "resistance_missing", "meta": {"missing": True}}
        max_bps = float(cfg.get("max_distance_bps", 25.0) or 25.0)
        ok = bool(dist_bps >= 0.0 and dist_bps <= max_bps)
        return {
            "pass": ok,
            "na": False,
            "code": "resistance_ok" if ok else "resistance_far",
            "meta": {"max_distance_bps": max_bps},
        }


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        fv = float(v)
        if fv != fv:  # NaN
            return None
        return fv
    except Exception:
        return None
