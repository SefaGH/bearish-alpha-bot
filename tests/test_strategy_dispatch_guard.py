from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.strategy_coordinator import StrategyCoordinator


def _make_coordinator(can_open_side_effect=None, fallback_side_effect=None):
    portfolio = SimpleNamespace(get_current_equity=lambda: 1000.0)
    risk_manager = SimpleNamespace()

    if can_open_side_effect is not None:
        risk_manager.can_open_new_position = Mock(side_effect=can_open_side_effect)
    else:
        risk_manager.can_open_new_position = None

    fallback = fallback_side_effect or [(True, "ok")]
    risk_manager.has_execution_capacity = Mock(side_effect=fallback)

    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio,
        risk_manager=risk_manager,
        config={'risk': {'queue': {}}}
    )
    return coordinator, risk_manager


@pytest.mark.asyncio
async def test_try_dispatch_waits_for_capacity_then_releases():
    coordinator, risk_manager = _make_coordinator([
        (False, "max open positions", {'portfolio_heat': 0.09}),
        (True, "ok", {'portfolio_heat': 0.02}),
    ])

    payload = {
        'signal_id': 'sig-1',
        'signal': {'symbol': 'BTC/USDT:USDT'},
        'risk_assessment': {'metrics': {'portfolio_heat': 0.5}}
    }
    await coordinator.signal_queue.put(payload)

    result = await coordinator.try_dispatch_next(timeout=0.5)

    assert result is payload
    assert risk_manager.can_open_new_position.call_count == 2
    assert result['risk_assessment']['metrics']['portfolio_heat'] == 0.02


@pytest.mark.asyncio
async def test_try_dispatch_returns_none_when_timeout_hits():
    coordinator, risk_manager = _make_coordinator([
        (False, "max open positions", {'portfolio_heat': 0.12})
    ])
    payload = {
        'signal_id': 'sig-2',
        'signal': {'symbol': 'ETH/USDT:USDT'}
    }
    await coordinator.signal_queue.put(payload)

    result = await coordinator.try_dispatch_next(timeout=0.1)

    assert result is None
    assert coordinator.signal_queue.qsize() == 1
    assert risk_manager.can_open_new_position.call_count >= 1


@pytest.mark.asyncio
async def test_try_dispatch_falls_back_to_legacy_capacity_checks():
    coordinator, risk_manager = _make_coordinator(
        can_open_side_effect=None,
        fallback_side_effect=[(False, "max open positions"), (True, "ok")]
    )

    payload = {
        'signal_id': 'sig-legacy',
        'signal': {'symbol': 'SOL/USDT:USDT'}
    }
    await coordinator.signal_queue.put(payload)

    result = await coordinator.try_dispatch_next(timeout=0.5)

    assert result is payload
    assert risk_manager.has_execution_capacity.call_count == 2