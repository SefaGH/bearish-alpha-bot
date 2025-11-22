from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.live_trading_engine import LiveTradingEngine


@pytest.mark.asyncio
async def test_trigger_coordinator_drain_routes_signal_into_engine_queue():
    payload = {
        'signal_id': 'sig-1',
        'signal': {'symbol': 'BTC/USDT:USDT'},
    }
    dispatcher = AsyncMock(return_value=payload)
    coordinator = SimpleNamespace(
        signal_queue=SimpleNamespace(qsize=lambda: 0),
        try_dispatch_next=dispatcher,
    )

    engine = LiveTradingEngine(strategy_coordinator=coordinator)

    drained = await engine.trigger_coordinator_drain(timeout=0.0)

    assert drained is True
    assert dispatcher.await_args.kwargs['timeout'] == 0.0

    routed_signal = await engine.signal_queue.get()
    assert routed_signal['signal_id'] == 'sig-1'
    assert routed_signal['from_coordinator'] is True


@pytest.mark.asyncio
async def test_trigger_coordinator_drain_returns_false_when_no_payload():
    dispatcher = AsyncMock(return_value=None)
    coordinator = SimpleNamespace(
        signal_queue=SimpleNamespace(qsize=lambda: 0),
        try_dispatch_next=dispatcher,
    )

    engine = LiveTradingEngine(strategy_coordinator=coordinator)

    drained = await engine.trigger_coordinator_drain(timeout=0.0)

    assert drained is False
    assert engine.signal_queue.empty()
