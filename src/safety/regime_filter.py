from typing import Any, Dict

from src.safety.regime_detector import MarketRegimeDetector


class RegimeFilter:
    """Stratejileri rejime göre filtreler."""

    def __init__(self, config: Dict[str, Any], market_data_pipeline: Any):
        self.detector = MarketRegimeDetector(config, market_data_pipeline)
        self.config = (config or {}).get("signals", {}).get("regime_filter", {})
        self.enabled = bool(self.config.get("enabled", False))

    async def validate(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"valid": True, "action": "pass", "reason": "disabled"}

        strategy = signal.get("strategy_name") or signal.get("strategy")
        symbol = signal.get("symbol")
        tf = signal.get("timeframe") or "1h"
        if not strategy or not symbol:
            return {"valid": True, "action": "pass", "reason": "missing_strategy_or_symbol"}

        regime = await self.detector.detect_regime(str(symbol), timeframe=str(tf))
        allowed, reason, weight = self.detector.is_strategy_allowed(str(strategy), regime)

        if not allowed:
            return {
                "valid": False,
                "action": "reject",
                "reason": f"regime_veto_{reason}",
                "metadata": {"regime": regime, "strategy": strategy},
            }

        meta = signal.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["regime"] = regime
        signal["meta"] = meta

        signal["regime_weight"] = float(weight)

        min_weight = float(self.config.get("min_regime_weight", 0.3) or 0.3)
        if weight < min_weight:
            return {
                "valid": False,
                "action": "reject",
                "reason": f"low_regime_weight_{weight:.2f}",
                "metadata": {
                    "regime": regime,
                    "weight": weight,
                    "min_required": min_weight,
                },
            }

        return {
            "valid": True,
            "action": "pass",
            "reason": f"allowed_in_{regime.get('label', 'unknown')}",
            "metadata": {"regime": regime, "weight": weight},
        }
