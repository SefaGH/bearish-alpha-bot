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

        meta = signal.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        price_meta = meta.get("price_meta")
        if not isinstance(price_meta, dict):
            price_meta = {}
        reference_price = price_meta.get("price_used") or signal.get("entry")
        if not reference_price:
            return {"valid": True, "reason": "no_reference_price"}

        symbol = signal.get("symbol")
        tf = signal.get("timeframe") or "1m"
        current_price = await self.market_data_pipeline.get_latest_price(str(symbol), timeframe=str(tf))

        if not current_price:
            return {
                "valid": False,
                "action": "reject",
                "reason": "integrity_data_unavailable",
            }

        deviation_pct = abs(current_price - reference_price) / reference_price

        if deviation_pct > max_deviation_pct:
            return {
                "valid": False,
                "action": "deviation_detected",
                "reason": f"price_deviation_{deviation_pct:.4f}",
                "metadata": {
                    "deviation_pct": deviation_pct,
                    "reference_price": reference_price,
                    "current_price": current_price,
                    "max_allowed": max_deviation_pct,
                },
            }

        return {"valid": True}

    def _determine_action_based_on_intent(
        self,
        signal: Dict[str, Any],
        deviation_result: Dict[str, Any],
        current_position: Any = None,
    ) -> Dict[str, Any]:
        intent = self._infer_intent(signal, current_position)

        if intent in {"entry", "reentry", "scale_in"}:
            return {
                "valid": False,
                "action": "reject",
                "reason": f"integrity_price_deviation_{intent}",
                "metadata": deviation_result.get("metadata", {}),
            }

        if intent == "reverse":
            return {
                "valid": True,
                "action": "convert_reverse_to_close",
                "reason": "integrity_reverse_to_close",
                "metadata": {
                    **(deviation_result.get("metadata") or {}),
                    "original_intent": "reverse",
                    "new_intent": INTENT_CLOSE,
                },
            }

        return deviation_result

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
