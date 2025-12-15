import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.signal_intents import INTENT_SCALE_IN
from src.core.strategy_coordinator import StrategyCoordinator


@pytest.mark.asyncio
async def test_strategy_coordinator_scale_in_uses_computed_quality_before_risk():
    pm = MagicMock()
    pm.cfg = {
        "pyramiding": {"enabled": True, "min_scale_in_quality": 0.8},
        "volume_analyzer": {"enabled": False},
        "risk": {"queue": {"max_pending_per_symbol": 2}},
    }
    pm.performance_monitor = None
    pm.get_open_positions_for_symbol.return_value = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "strategy_name": "alpha",
            "entry_price": 100.0,
            "unrealized_pnl_pct": 0.01,
            "entry_time": 1,
        }
    ]
    pm.get_open_positions.return_value = {}

    rm = MagicMock()

    captured = {}

    async def fake_assess(signal, strategy_name):
        captured["quality_score"] = signal.get("quality_score")
        captured["intent"] = signal.get("intent")
        return {"acceptable": True, "position_size": 1.0, "notional": 10.0, "metrics": {}}

    coordinator = StrategyCoordinator(
        pm,
        rm,
        config={
            "pyramiding": {"enabled": True, "min_scale_in_quality": 0.8},
            "volume_analyzer": {"enabled": False},
            "risk": {"queue": {"max_pending_per_symbol": 2, "max_queue_depth": 5}},
        },
    )
    coordinator.ml_integration = MagicMock()
    coordinator.ml_integration.get_ml_context = AsyncMock(
        return_value={"consensus_score": 0.95, "regime": "bullish", "regime_confidence": 0.9, "quality_score": 0.95}
    )
    coordinator._assess_signal_risk = fake_assess
    coordinator._compute_signal_quality = lambda s: s.update(
        {"quality_score": 0.91, "quality_breakdown": {"value": 0.91, "components": {}, "reason": []}}
    ) or {"value": 0.91, "components": {}, "reason": []}

    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry": 101.0,
        "stop": 99.0,
        "target": 105.0,
        "strategy_name": "alpha",
        "features": {"momentum": 5.0, "spread": 0.01, "volume_score": 2.0},
    }

    result = await coordinator.process_strategy_signal("alpha", signal)

    assert result["status"] == "accepted"
    assert captured["intent"] == INTENT_SCALE_IN
    assert captured["quality_score"] is not None
    assert captured["quality_score"] > 0.8
    assert result["enriched_signal"]["quality_score"] == captured["quality_score"]
