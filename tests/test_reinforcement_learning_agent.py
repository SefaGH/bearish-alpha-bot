import numpy as np
import pytest

from src.ml.reinforcement_learning import TradingRLAgent, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
else:  # pragma: no cover - torchless environments fall back to mock agent
    torch = None


def _build_agent(**extra):
    if TORCH_AVAILABLE:
        base_config = {'training_mode': True, 'epsilon_start': 1.0}
        base_config.update(extra)
        return TradingRLAgent(state_size=4, action_size=3, config=base_config)
    return TradingRLAgent(state_size=4, action_size=3, training_mode=True, epsilon_start=1.0, **extra)


def test_set_inference_mode_forces_zero_epsilon():
    agent = _build_agent()
    agent.set_inference_mode()
    assert agent.training_mode is False
    assert agent.epsilon == 0.0
    assert getattr(agent, '_inference_locked', False) is True


def test_inference_lock_disables_training_flag():
    agent = _build_agent()
    agent.set_inference_mode()
    action, meta = agent.get_action_with_meta(np.zeros(agent.state_size), training=True)
    assert meta['training_mode'] is False
    assert meta['epsilon'] == agent.epsilon
    assert action in {0, 1, 2}


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch runtime required for bias test")
def test_regime_bias_skips_when_confidence_low():
    agent = _build_agent(min_regime_confidence_for_bias=0.8, regime_bias_strength=2.0)
    q_values = torch.zeros((1, agent.action_size))
    adjusted, meta = agent._apply_regime_bias(q_values, {
        'predicted_regime': 'bullish',
        'confidence': 0.5
    })
    assert torch.equal(adjusted, q_values)
    assert meta['bias_applied'] is False
    assert meta['effective_bias'] == 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch runtime required for bias test")
def test_regime_bias_scales_with_confidence():
    agent = _build_agent(
        min_regime_confidence_for_bias=0.2,
        regime_bias_strength=2.0,
        max_regime_bias=3.0
    )
    q_values = torch.zeros((1, agent.action_size))
    adjusted, meta = agent._apply_regime_bias(q_values, {
        'predicted_regime': 'bearish',
        'confidence': 0.9
    })
    assert meta['bias_applied'] is True
    expected_bias = min(2.0 * 0.9, 3.0)
    assert meta['effective_bias'] == pytest.approx(expected_bias, rel=1e-5)
    assert adjusted[0, 2].item() == pytest.approx(expected_bias, rel=1e-5)
