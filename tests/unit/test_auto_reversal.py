from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.signal_intents import INTENT_ENTRY, INTENT_REVERSE
from src.core.strategy_coordinator import StrategyCoordinator


def _make_coordinator(*, allow_auto_reversal: bool, open_position_side: str):
    pm = MagicMock()
    pm.cfg = {"signals": {"allow_auto_reversal": allow_auto_reversal}}
    pm.exchange_clients = {}
    pm.get_strategy_allocation.return_value = 0.1
    pm.performance_monitor = None

    pm.get_open_positions_for_symbol.return_value = [
        {
            "position_id": "pos_1",
            "symbol": "BTC/USDT",
            "side": open_position_side,
            "entry_price": 100.0,
            "size": 1.0,
        }
    ]

    rm = MagicMock()

    coordinator = StrategyCoordinator(
        portfolio_manager=pm,
        risk_manager=rm,
        market_data_pipeline=None,
        config={
            "volume_analyzer": {"enabled": False},
            "strategies": {},
            "risk": {"queue": {}},
        },
    )

    coordinator._determine_intent = MagicMock(return_value=INTENT_ENTRY)
    coordinator._validate_signal_format = MagicMock(return_value={"valid": True})
    coordinator._enrich_signal = AsyncMock(side_effect=lambda strat, sig: dict(sig))
    coordinator.validate_duplicate = MagicMock(return_value=(True, "OK"))

    # Simulate conflict with open position; accept conflict resolution
    coordinator._check_signal_conflicts = AsyncMock(
        return_value={
            "has_conflict": True,
            "conflicts": ["opposite_to_position"],
            "conflicting_signals": [
                {
                    "position_id": "pos_1",
                    "position": {"position_id": "pos_1", "symbol": "BTC/USDT", "side": open_position_side},
                    "conflict_type": "opposite_to_position",
                }
            ],
        }
    )
    coordinator.resolve_signal_conflicts = AsyncMock(return_value={"action": "accept", "reason": "ok", "winner": {}})

    coordinator._assess_signal_risk = AsyncMock(
        return_value={"acceptable": True, "position_size": 1.0, "notional": 100.0, "metrics": {}}
    )
    coordinator._route_signal = MagicMock(return_value={})
    coordinator._generate_signal_id = MagicMock(return_value="sig_reverse")
    coordinator._compute_signal_quality = MagicMock(return_value={})
    coordinator.emit_signal_breakdown = MagicMock()

    coordinator.signal_queue = MagicMock()
    coordinator.signal_queue.put = AsyncMock(return_value=(True, None))

    return coordinator


@pytest.mark.asyncio
async def test_auto_reversal_tags_reverse_intent_when_enabled():
    coordinator = _make_coordinator(allow_auto_reversal=True, open_position_side="buy")

    base_signal = {
        "symbol": "BTC/USDT",
        "side": "sell",
        "entry": 100.0,
        "stop": 101.0,
        "target": 98.0,
        "timeframe": "5m",
    }

    result = await coordinator.process_strategy_signal("alpha", dict(base_signal))
    assert result["status"] == "accepted"

    queued_payload = coordinator.signal_queue.put.call_args[0][0]
    queued_signal = queued_payload["signal"]
    assert queued_signal["intent"] == INTENT_REVERSE
    assert queued_signal["reverse_from_position_id"] == "pos_1"


@pytest.mark.asyncio
async def test_auto_reversal_does_not_tag_when_disabled():
    coordinator = _make_coordinator(allow_auto_reversal=False, open_position_side="buy")

    base_signal = {
        "symbol": "BTC/USDT",
        "side": "sell",
        "entry": 100.0,
        "stop": 101.0,
        "target": 98.0,
        "timeframe": "5m",
    }

    result = await coordinator.process_strategy_signal("alpha", dict(base_signal))
    assert result["status"] == "accepted"

    queued_payload = coordinator.signal_queue.put.call_args[0][0]
    queued_signal = queued_payload["signal"]
    assert queued_signal.get("intent") != INTENT_REVERSE

