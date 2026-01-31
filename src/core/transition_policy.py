from typing import Any, Dict


class PositionTransitionPolicy:
    """Cross-strategy reversal politikalarını uygular."""

    def __init__(self, config: Dict[str, Any]):
        self.config = (config or {}).get("signals", {}).get("transition_policy", {})
        self.enabled = bool(self.config.get("enabled", False))

        self.strategy_families = {
            "adaptive_ob": "trend_following",
            "adaptive_str": "trend_following",
            "mean_reversion": "mean_reversion",
        }

        self.policy_matrix = self._build_policy_matrix()

    def _build_policy_matrix(self) -> Dict:
        return {
            ("trend_following", "mean_reversion", "opposite", "reverse"): {
                "action": "convert_to_close",
                "reason": "cross_strategy_trend_to_counter",
                "allow_force": False,
            },
            ("mean_reversion", "trend_following", "opposite", "reverse"): {
                "action": "allow",
                "reason": "counter_to_trend_allowed",
                "min_profit_pct": 0.5,
            },
            ("trend_following", "trend_following", "opposite", "reverse"): {
                "action": "allow",
                "reason": "same_family_reverse",
            },
            ("mean_reversion", "mean_reversion", "opposite", "reverse"): {
                "action": "allow",
                "reason": "same_family_reverse",
            },
            ("*", "*", "same", "entry"): {
                "action": "scale_in_check",
                "reason": "same_direction_scale_in",
            },
        }

    def evaluate(self, current_position: Any, incoming_signal: Dict[str, Any], inferred_intent: str) -> Dict[str, Any]:
        if not self.enabled or not current_position:
            return {"allowed": True, "action": "allow", "reason": "no_position_or_disabled"}

        from_family = self._strategy_family(getattr(current_position, "strategy", None) or _safe_get(current_position, "strategy"))
        to_family = self._strategy_family(
            incoming_signal.get("strategy_name") or incoming_signal.get("strategy")
        )

        incoming_side = str(incoming_signal.get("side") or "").strip().lower()
        position_side = _safe_get(current_position, "side")
        if position_side is None and hasattr(current_position, "side"):
            position_side = getattr(current_position, "side")
        position_side = str(position_side or "").strip().lower()

        direction = "same" if incoming_side == position_side else "opposite"

        meta = incoming_signal.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        if meta.get("force_reverse_allowed", False):
            return {"allowed": True, "action": "allow", "reason": "force_reverse_flagged"}

        policy_key = (from_family, to_family, direction, inferred_intent)
        policy = self.policy_matrix.get(policy_key)
        if not policy:
            wildcard_key = ("*", "*", direction, inferred_intent)
            policy = self.policy_matrix.get(wildcard_key)

        if not policy:
            return {"allowed": True, "action": "allow", "reason": "no_policy_found_default_allow"}

        action = policy["action"]
        if action == "allow":
            if "min_profit_pct" in policy:
                pnl = _safe_get(current_position, "unrealized_pnl_pct")
                try:
                    pnl_val = float(pnl)
                except Exception:
                    pnl_val = 0.0
                if pnl_val < float(policy["min_profit_pct"]):
                    return {
                        "allowed": False,
                        "action": "convert_to_close",
                        "reason": f"insufficient_profit_{pnl_val:.2f}%",
                        "metadata": {"required": policy["min_profit_pct"]},
                    }
            return {"allowed": True, "action": "allow", "reason": policy["reason"]}

        if action == "convert_to_close":
            return {
                "allowed": False,
                "action": "convert_to_close",
                "reason": policy["reason"],
                "metadata": {
                    "from_family": from_family,
                    "to_family": to_family,
                    "original_intent": inferred_intent,
                },
            }

        if action == "reject":
            return {"allowed": False, "action": "reject", "reason": policy["reason"]}

        return {"allowed": True, "action": "allow", "reason": "policy_default"}

    def _strategy_family(self, strategy_name: Any) -> str:
        if not strategy_name:
            return "unknown"
        return self.strategy_families.get(str(strategy_name), "unknown")


def _safe_get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return None
