import asyncio
import time

import pytest

from src.core.strategy_coordinator import StrategyCoordinator


class DummyPortfolioManager:
    def get_current_equity(self):
        return 1000.0


@pytest.mark.asyncio
async def test_queue_ttl_expiry_triggers_discard_active_signal(monkeypatch):
    t = 0.0

    def fake_time():
        return t

    monkeypatch.setattr(time, "time", fake_time)

    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    signal_id = "signal_1"
    coordinator.active_signals[signal_id] = {"signal": {"signal_id": signal_id}, "status": "active", "timestamp": None}

    ok, _, _ = await coordinator.signal_queue.put(
        {
            "signal_id": signal_id,
            "signal": {"symbol": "BTC/USDT:USDT", "intent": "entry", "priority": 1},
        }
    )
    assert ok is True
    assert signal_id in coordinator.active_signals

    t = 6.0

    ok, _, _ = await coordinator.signal_queue.put(
        {
            "signal_id": "signal_2",
            "signal": {"symbol": "BTC/USDT:USDT", "intent": "entry", "priority": 1},
        }
    )
    assert ok is True

    await asyncio.sleep(0)
    assert signal_id not in coordinator.active_signals
