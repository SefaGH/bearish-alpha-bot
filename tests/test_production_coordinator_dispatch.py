import os
import sys
from typing import Any, cast

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator  # type: ignore


class DummyEngine:
    def __init__(self, should_drain):
        self.should_drain = should_drain
        self.drain_calls = 0
        self.forwarded_payloads = []

    async def trigger_coordinator_drain(self, timeout: float = 0.0) -> bool:
        self.drain_calls += 1
        return self.should_drain

    async def _forward_signal_from_coordinator(self, payload):
        self.forwarded_payloads.append(payload)
        return True


class DummyQueue:
    def __init__(self):
        self.requeued = []

    async def requeue(self, payload):
        self.requeued.append(payload)


class DummyCoordinator:
    def __init__(self, payload=None):
        self._payload = payload
        self.calls = 0
        self.signal_queue = DummyQueue()

    async def try_dispatch_next(self, timeout: float = 0.0):
        self.calls += 1
        return self._payload


def _make_coordinator_instance():
    return object.__new__(ProductionCoordinator)


@pytest.mark.asyncio
async def test_nudge_strategy_dispatch_prefers_engine_helper():
    instance = _make_coordinator_instance()
    instance.strategy_coordinator = cast(Any, DummyCoordinator())
    instance.trading_engine = cast(Any, DummyEngine(should_drain=True))

    await ProductionCoordinator._nudge_strategy_dispatch(instance)

    assert instance.trading_engine.drain_calls == 1
    assert instance.strategy_coordinator.calls == 0


@pytest.mark.asyncio
async def test_nudge_strategy_dispatch_forwards_payload_when_engine_cannot_drain():
    payload = {'signal_id': 'sig-nudge', 'signal': {'symbol': 'BTC/USDT:USDT'}}
    instance = _make_coordinator_instance()
    instance.strategy_coordinator = cast(Any, DummyCoordinator(payload=payload))
    instance.trading_engine = cast(Any, DummyEngine(should_drain=False))

    await ProductionCoordinator._nudge_strategy_dispatch(instance)

    assert instance.strategy_coordinator.calls == 1
    assert instance.trading_engine.forwarded_payloads == [payload]
    assert instance.strategy_coordinator.signal_queue.requeued == []
