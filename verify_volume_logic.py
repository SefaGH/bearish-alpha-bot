import asyncio
import logging
import sys
import time
import types
from typing import Any, Dict, Optional

# Ensure local src/ is importable
sys.path.append("src")

from core.strategy_coordinator import StrategyCoordinator, PrioritySignalQueue  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verify_volume_logic")


def _build_coordinator() -> StrategyCoordinator:
    """Create a StrategyCoordinator instance with minimal mocks/stubs."""
    coord: StrategyCoordinator = object.__new__(StrategyCoordinator)  # bypass __init__

    # Minimal config for a single strategy
    coord.config = {"strategies": {"test_strategy": {}}}
    coord.portfolio_manager = types.SimpleNamespace(cfg={})
    coord.risk_manager = types.SimpleNamespace()
    coord.market_data_pipeline = None

    # Fresh dual-stage queue
    queue_cfg = {"max_pending_per_symbol": 10, "max_queue_depth": 20}
    coord.signal_queue = PrioritySignalQueue(queue_cfg, logger)
    coord.active_signals = {}
    coord.signal_history = []
    coord._signal_history_lookup = {}

    # Tracking stats (keys used in the flow)
    coord.processing_stats = {
        "total_signals": 0,
        "accepted_signals": 0,
        "rejected_signals": 0,
        "duplicate_rejections": 0,
        "queue_rejections": 0,
        "last_signal_time": None,
    }

    coord.ml_integration = False  # Skip ML branch

    # --- Stubbed methods (sync/async) to isolate volume policy path ---
    def _validate_signal_format(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        return {"valid": True}

    async def _enrich_signal(self, strategy_name: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        # Return the signal as-is, ensuring expected keys exist
        sig = dict(signal)
        sig.setdefault("queue_meta", {})
        sig.setdefault("strategy_name", strategy_name)
        return sig

    async def _assess_signal_risk(self, signal: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
        return {"acceptable": True, "reason": "", "position_size": 1.0, "notional": 0, "metrics": {}}

    async def _enhance_signal_with_ml(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return signal

    def _route_signal(self, signal: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    async def _check_signal_conflicts(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        return {"has_conflict": False, "conflicting_signals": []}

    def validate_duplicate(self, signal: Dict[str, Any], strategy_name: str):
        return True, None

    def _compute_signal_quality(self, signal: Dict[str, Any]):
        return {"value": 1.0}

    def emit_signal_breakdown(self, *args, **kwargs):
        return None

    def _add_signal_history_entry(self, *args, **kwargs):
        return None

    # Deterministic signal id
    _sig_counter = {"v": 0}

    def _generate_signal_id(self, *args, **kwargs):
        _sig_counter["v"] += 1
        return f"sig-{_sig_counter['v']}"

    def _determine_intent(self, signal: Dict[str, Any], strategy_name: str):
        return signal.get("intent") or signal.get("side") or "entry"

    # Bind stubs
    coord._validate_signal_format = types.MethodType(_validate_signal_format, coord)
    coord._enrich_signal = types.MethodType(_enrich_signal, coord)
    coord._assess_signal_risk = types.MethodType(_assess_signal_risk, coord)
    coord._enhance_signal_with_ml = types.MethodType(_enhance_signal_with_ml, coord)
    coord._route_signal = types.MethodType(_route_signal, coord)
    coord._check_signal_conflicts = types.MethodType(_check_signal_conflicts, coord)
    coord.validate_duplicate = types.MethodType(validate_duplicate, coord)
    coord._compute_signal_quality = types.MethodType(_compute_signal_quality, coord)
    coord.emit_signal_breakdown = types.MethodType(emit_signal_breakdown, coord)
    coord._add_signal_history_entry = types.MethodType(_add_signal_history_entry, coord)
    coord._generate_signal_id = types.MethodType(_generate_signal_id, coord)
    coord._determine_intent = types.MethodType(_determine_intent, coord)

    return coord


async def scenario_defer(coord: StrategyCoordinator):
    """Scenario A: Low volume + tight stop -> defer."""
    signal = {
        "symbol": "BTCUSDT",
        "side": "long",
        "entry": 100.0,
        "stop": 99.9,  # 0.10% distance
        "target": 103.0,
        "priority": 100,
        "volume_bucket": "LOW",
        "queue_meta": {},
    }
    result = await coord.process_strategy_signal("test_strategy", signal)
    waiting_len = len(coord.signal_queue._waiting_room)  # type: ignore
    logger.info("Scenario A (Defer): result=%s, waiting_room=%d", result, waiting_len)
    return result, waiting_len, signal


async def scenario_rescue(coord: StrategyCoordinator, signal: Dict[str, Any]):
    """Scenario B: Deferred signal comes back under LOW volume -> rescue path."""
    rescue_signal = dict(signal)
    rescue_signal.setdefault("queue_meta", {})["is_deferred"] = True
    result = await coord.process_strategy_signal("test_strategy", rescue_signal)
    enriched = result.get("enriched_signal") if isinstance(result, dict) else {}
    logger.info(
        "Scenario B (Rescue): status=%s, stop_loss_pct=%s, exec_params=%s",
        result.get("status") if isinstance(result, dict) else None,
        enriched.get("stop_loss_pct") if isinstance(enriched, dict) else None,
        enriched.get("execution_params") if isinstance(enriched, dict) else None,
    )
    return result, enriched


async def scenario_busy_loop_prevention():
    """Scenario C: Waiting room item should not be popped early."""
    queue = PrioritySignalQueue({}, logger)
    payload = {
        "signal_id": "future",
        "signal": {"symbol": "ETHUSDT", "side": "long"},
        "risk_assessment": {},
        "routing": {},
    }
    await queue.put(payload, process_after=time.time() + 100)  # far in the future

    try:
        await queue.get(timeout=0.2)
        popped = True
    except asyncio.TimeoutError:
        popped = False

    remaining_waiting = len(queue._waiting_room)  # type: ignore
    logger.info(
        "Scenario C (Busy-loop prevention): popped=%s, waiting_room=%d",
        popped,
        remaining_waiting,
    )
    return popped, remaining_waiting


async def main():
    coord = _build_coordinator()

    # Scenario A: defer path
    res_a, waiting_len, base_signal = await scenario_defer(coord)

    # Scenario B: rescue path for deferred signal
    res_b, enriched_b = await scenario_rescue(coord, base_signal)

    # Scenario C: waiting_room should not be popped prematurely
    popped_c, waiting_len_c = await scenario_busy_loop_prevention()

    print("\n=== Summary ===")
    print(f"Scenario A -> status: {res_a.get('status') if isinstance(res_a, dict) else res_a}, waiting_room: {waiting_len}")
    print(
        f"Scenario B -> status: {res_b.get('status') if isinstance(res_b, dict) else res_b}, "
        f"stop_loss_pct: {enriched_b.get('stop_loss_pct') if isinstance(enriched_b, dict) else None}, "
        f"execution_params: {enriched_b.get('execution_params') if isinstance(enriched_b, dict) else None}"
    )
    print(f"Scenario C -> popped_early: {popped_c}, waiting_room: {waiting_len_c}")


if __name__ == "__main__":
    asyncio.run(main())
