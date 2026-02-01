from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.core.signal_intents import INTENT_CLOSE


class SignalIntegrityGuard:
    """Sinyal bütünlük kontrolü ve price deviation guard."""

    def __init__(self, config: Dict[str, Any], market_data_pipeline: Any):
        self.config = (config or {}).get("signals", {}).get("integrity_guard", {})
        self.market_data_pipeline = market_data_pipeline
        self.enabled = bool(self.config.get("enabled", False))

    async def validate(self, signal: Dict[str, Any], current_position: Any = None) -> Dict[str, Any]:
        """
        Sinyali doğrula ve aksiyon belirle.

        Returns:
            Dict with keys: valid(bool), action(str), reason(str), metadata(dict)
        """
        if not self.enabled:
            return {"valid": True, "action": "pass", "reason": "disabled"}

        if not self.market_data_pipeline:
            return {"valid": True, "action": "pass", "reason": "no_market_data_pipeline"}

        staleness_result = self._check_staleness(signal)
        if not staleness_result.get("valid", True):
            return staleness_result

        impulse_result = self._check_impulse_veto(signal)
        if not impulse_result.get("valid", True):
            return self._determine_action_based_on_intent(signal, impulse_result, current_position)

        deviation_result = await self._check_price_deviation(signal)
        if not deviation_result.get("valid", True):
            return self._determine_action_based_on_intent(signal, deviation_result, current_position)

        return {"valid": True, "action": "pass", "reason": "valid"}

    def _check_staleness(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        max_staleness_ms = int(self.config.get("max_staleness_ms", 10000) or 10000)

        meta = signal.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        price_meta = meta.get("price_meta")
        if not isinstance(price_meta, dict):
            price_meta = {}
        candle_ts = price_meta.get("candle_close_ts")
        if candle_ts:
            try:
                now = datetime.now(timezone.utc)
                if isinstance(candle_ts, datetime) and candle_ts.tzinfo is None:
                    candle_ts = candle_ts.replace(tzinfo=timezone.utc)
                staleness_ms = (now - candle_ts).total_seconds() * 1000
            except Exception:
                staleness_ms = 0
            if staleness_ms > max_staleness_ms:
                return {
                    "valid": False,
                    "action": "reject",
                    "reason": f"stale_candle_{staleness_ms:.0f}ms",
                    "metadata": {"staleness_ms": staleness_ms},
                }

        signal_ts = signal.get("timestamp")
        if signal_ts is not None:
            try:
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                ts_ms = int(float(signal_ts))
                signal_age_ms = max(0, now_ms - ts_ms)
                if signal_age_ms > max_staleness_ms:
                    return {
                        "valid": False,
                        "action": "reject",
                        "reason": f"stale_signal_{signal_age_ms:.0f}ms",
                        "metadata": {"signal_age_ms": signal_age_ms},
                    }
            except Exception:
                pass

        return {"valid": True}

    async def _check_price_deviation(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        max_deviation_pct = float(self.config.get("max_deviation_pct", 0.001) or 0.001)
        atr_guard_enabled = bool(self.config.get("atr_guard_enabled", False))
        atr_guard_mult = self._safe_float(self.config.get("atr_guard_mult"), 0.5)
        spread_buffer_bps = self._safe_float(self.config.get("spread_buffer_bps"), 0.0)
        min_gap_bps_fallback = self._safe_float(self.config.get("min_gap_bps_fallback"), 0.0)

        meta = signal.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        price_meta = meta.get("price_meta")
        if not isinstance(price_meta, dict):
            price_meta = {}
        reference_price = price_meta.get("price_used") or signal.get("entry")
        try:
            reference_price = float(reference_price) if reference_price is not None else None
        except (TypeError, ValueError):
            reference_price = None
        if not reference_price or reference_price <= 0:
            return {"valid": True, "reason": "no_reference_price"}

        symbol = signal.get("symbol")
        tf = signal.get("timeframe") or "1m"
        current_price = await self.market_data_pipeline.get_latest_price(str(symbol), timeframe=str(tf))
        try:
            current_price = float(current_price) if current_price is not None else None
        except (TypeError, ValueError):
            current_price = None

        if not current_price or current_price <= 0:
            return {
                "valid": False,
                "action": "reject",
                "reason": "integrity_data_unavailable",
            }

        deviation_pct = abs(current_price - reference_price) / reference_price
        gap_bps = deviation_pct * 10000.0

        allowed_gap_bps = max_deviation_pct * 10000.0
        atr_bps = None
        if atr_guard_enabled:
            atr_val = self._safe_float(signal.get("atr") or signal.get("atr_value"))
            if atr_val is None:
                vol_meta = meta.get("vol_telemetry") if isinstance(meta.get("vol_telemetry"), dict) else {}
                atr_bps = self._safe_float(vol_meta.get("atr_bps"))
            else:
                atr_bps = (atr_val / reference_price) * 10000.0

            if atr_bps is not None and atr_bps > 0:
                allowed_gap_bps = max(allowed_gap_bps, (atr_guard_mult * atr_bps) + spread_buffer_bps)
            elif min_gap_bps_fallback > 0:
                allowed_gap_bps = max(allowed_gap_bps, min_gap_bps_fallback)

        if gap_bps > allowed_gap_bps:
            reason = "price_moved_fast" if atr_guard_enabled else "price_deviation"
            return {
                "valid": False,
                "action": "deviation_detected",
                "reason": reason,
                "metadata": {
                    "reason_code": reason,
                    "deviation_pct": deviation_pct,
                    "gap_bps": gap_bps,
                    "threshold_bps": allowed_gap_bps,
                    "allowed_gap_bps": allowed_gap_bps,
                    "reference_price": reference_price,
                    "current_price": current_price,
                    "max_deviation_pct": max_deviation_pct,
                    "atr_guard_enabled": atr_guard_enabled,
                    "atr_guard_mult": atr_guard_mult,
                    "atr_bps": atr_bps,
                    "spread_buffer_bps": spread_buffer_bps,
                    "min_gap_bps_fallback": min_gap_bps_fallback,
                },
            }

        return {"valid": True}

    def _check_impulse_veto(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Optional impulse/shock veto based on signal meta (strategy-provided)."""
        enabled = bool(self.config.get("impulse_guard_enabled", False))
        if not enabled:
            return {"valid": True}

        meta = signal.get("meta")
        if not isinstance(meta, dict):
            return {"valid": True}

        impulse_meta = meta.get("impulse_guard")
        if not isinstance(impulse_meta, dict):
            return {"valid": True}

        if not impulse_meta.get("enabled", True):
            return {"valid": True}

        is_shock = bool(impulse_meta.get("is_shock_move"))
        trade_dir = str(impulse_meta.get("trade_dir") or "").lower().strip()
        candle_dir = str(impulse_meta.get("candle_dir") or "").lower().strip()
        require_opposite = bool(impulse_meta.get("require_opposite", True))

        if is_shock and trade_dir and candle_dir and (not require_opposite or trade_dir != candle_dir):
            meta_out = dict(impulse_meta)
            meta_out.setdefault("reason_code", "impulse_shock")
            return {
                "valid": False,
                "action": "reject",
                "reason": "impulse_shock",
                "metadata": meta_out,
            }

        return {"valid": True}

    def _determine_action_based_on_intent(
        self,
        signal: Dict[str, Any],
        result: Dict[str, Any],
        current_position: Any = None,
    ) -> Dict[str, Any]:
        intent = self._infer_intent(signal, current_position)
        reason = result.get("reason") or "integrity_guard_reject"
        metadata = result.get("metadata", {})

        if intent in {"entry", "reentry", "scale_in"}:
            return {
                "valid": False,
                "action": "reject",
                "reason": reason,
                "metadata": metadata,
            }

        if intent == "reverse":
            return {
                "valid": True,
                "action": "convert_reverse_to_close",
                "reason": "integrity_reverse_to_close",
                "metadata": {
                    **(metadata or {}),
                    "original_reason": reason,
                    "original_intent": "reverse",
                    "new_intent": INTENT_CLOSE,
                },
            }

        return result

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return default
            val = float(value)
            if not (val == val):  # NaN check
                return default
            return val
        except (TypeError, ValueError):
            return default

    def _infer_intent(self, signal: Dict[str, Any], position: Any) -> str:
        if not position:
            return "entry"

        signal_side = str(signal.get("side") or "").strip().lower()
        position_side = getattr(position, "side", None)
        if position_side is None and isinstance(position, dict):
            position_side = position.get("side")
        position_side = str(position_side or "").strip().lower()

        if signal_side and position_side and signal_side == position_side:
            return "scale_in" if self._should_scale_in(position, signal) else "exit_same_side"

        return "reverse"

    def _should_scale_in(self, position: Any, signal: Dict[str, Any]) -> bool:
        return False
