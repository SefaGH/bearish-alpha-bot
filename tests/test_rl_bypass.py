import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.strategy_coordinator import StrategyCoordinator


class DummyMLIntegration:
    async def enhance_strategy_signal(self, symbol, signal, current_price):
        return {}


def make_coordinator(config=None):
    portfolio_manager = MagicMock()
    portfolio_manager.cfg = {}
    risk_manager = MagicMock()
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        config=config or {}
    )
    coordinator.ml_integration = DummyMLIntegration()
    return coordinator


def attach_state_stub(coordinator):
    async def _fake_state(self, symbol, price):
        return np.zeros(82, dtype=float)

    coordinator._extract_rl_state = types.MethodType(_fake_state, coordinator)


class DummyRLAgent:
    def __init__(self, q_values):
        self.q_values = q_values
        self.training_mode = False
        self._inference_locked = False

    def set_inference_mode(self):
        self._inference_locked = True

    def get_action_with_meta(self, state_features, **kwargs):
        idx = int(np.argmax(self.q_values))
        meta = {
            'raw_q_values': list(self.q_values),
            'probabilities': [0.33, 0.33, 0.34],
            'best_probability': 0.34,
            'epsilon': 0.0
        }
        return idx, meta


@pytest.mark.asyncio
async def test_bypass_on_frozen_model():
    coordinator = make_coordinator()
    attach_state_stub(coordinator)
    coordinator.rl_agent = DummyRLAgent([-0.009, -0.009, -0.009])

    signal = {
        'side': 'sell',
        'entry': 100.0,
        'symbol': 'BTC/USDT',
        'strategy_name': 'TestStrategy'
    }

    enhanced = await coordinator._enhance_signal_with_ml(signal)

    assert enhanced['rl_bypassed'] is True
    assert enhanced['rl_bypass_reason'] == 'frozen_model'
    assert enhanced['rl_recommendation'] == 'sell'
    assert coordinator.rl_telemetry['bypass_count'] == 1


@pytest.mark.asyncio
async def test_normal_rl_when_variance_ok():
    coordinator = make_coordinator()
    attach_state_stub(coordinator)
    coordinator.rl_agent = DummyRLAgent([0.5, 0.2, 0.3])

    signal = {
        'side': 'buy',
        'entry': 100.0,
        'symbol': 'ETH/USDT',
        'strategy_name': 'TestStrategy'
    }

    enhanced = await coordinator._enhance_signal_with_ml(signal)

    assert enhanced['rl_bypassed'] is False
    assert enhanced['rl_recommendation'] == 'buy'
    assert coordinator.rl_telemetry['bypass_count'] == 0
