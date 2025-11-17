import numpy as np

from src.ml.reinforcement_learning import TradingRLAgent, TORCH_AVAILABLE


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
