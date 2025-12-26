import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.signal_intents import INTENT_ENTRY
from src.core.strategy_coordinator import StrategyCoordinator


def _make_coordinator(*, queued: bool, reason: str | None):
    pm = MagicMock()
    pm.cfg = {}

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

    coordinator._enrich_signal = AsyncMock()
    coordinator.validate_duplicate = MagicMock(return_value=(True, "OK"))
    coordinator._check_signal_conflicts = AsyncMock(return_value={"has_conflict": False})
    coordinator._assess_signal_risk = AsyncMock(
        return_value={"acceptable": True, "position_size": 1.0, "notional": 100.0, "metrics": {}}
    )
    coordinator._route_signal = MagicMock(return_value={})
    coordinator._generate_signal_id = MagicMock(return_value="sig_455")
    coordinator._compute_signal_quality = MagicMock(return_value={})
    coordinator.emit_signal_breakdown = MagicMock()

    coordinator.signal_queue = MagicMock()
    coordinator.signal_queue.put = AsyncMock(return_value=(queued, reason))

    return coordinator


@pytest.mark.asyncio
async def test_process_strategy_signal_logs_enqueued_only_on_queue_acceptance(caplog):
    coordinator = _make_coordinator(queued=True, reason=None)

    base_signal = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "entry": 100.0,
        "stop": 90.0,
        "target": 110.0,
        "timeframe": "5m",
        "priority": 2,
    }

    coordinator._enrich_signal.return_value = dict(base_signal)

    caplog.set_level(logging.INFO)
    result = await coordinator.process_strategy_signal("alpha", dict(base_signal))

    assert result["status"] == "accepted"
    assert " ENQUEUED | " in caplog.text
    assert " REJECTED | " not in caplog.text


@pytest.mark.asyncio
async def test_process_strategy_signal_logs_rejected_only_on_queue_rejection(caplog):
    coordinator = _make_coordinator(queued=False, reason="queue_limit")

    base_signal = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "entry": 100.0,
        "stop": 90.0,
        "target": 110.0,
        "timeframe": "5m",
        "priority": 2,
    }

    coordinator._enrich_signal.return_value = dict(base_signal)

    caplog.set_level(logging.INFO)
    result = await coordinator.process_strategy_signal("alpha", dict(base_signal))

    assert result["status"] == "rejected"
    assert "queue_limit" in (result.get("reason") or "")
    assert " REJECTED | reason=queue_limit " in caplog.text
    assert " ENQUEUED | " not in caplog.text
