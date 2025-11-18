"""
Unit tests to verify RL action mapping consistency across the stack.

Ensures that the canonical action mapping (0=HOLD, 1=BUY, 2=SELL)
is respected by both the RL environment and agent.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Import the modules under test
from src.ml.rl_trading_env import RLTradingEnv, ACTION_LABELS as ENV_ACTION_LABELS
from src.ml.reinforcement_learning import TradingRLAgent, ACTION_LABELS as AGENT_ACTION_LABELS


def test_action_labels_consistency():
    """Verify that ACTION_LABELS are identical in env and agent modules."""
    assert ENV_ACTION_LABELS == AGENT_ACTION_LABELS, \
        f"Action labels mismatch: env={ENV_ACTION_LABELS}, agent={AGENT_ACTION_LABELS}"
    assert ENV_ACTION_LABELS == ['HOLD', 'BUY', 'SELL'], \
        f"Expected ['HOLD', 'BUY', 'SELL'], got {ENV_ACTION_LABELS}"


def test_env_action_mapping_hold():
    """Verify that action 0 (HOLD) does not trigger a trade in the environment."""
    # Create minimal test data
    features_df = pd.DataFrame(np.random.randn(10, 5), columns=[f'f{i}' for i in range(5)])
    raw_df = pd.DataFrame({
        'close': [100.0] * 10,
        'open': [99.0] * 10,
        'high': [101.0] * 10,
        'low': [98.0] * 10,
        'volume': [1000.0] * 10
    })
    
    env = RLTradingEnv(features_df=features_df, raw_df=raw_df, initial_balance=10000.0)
    initial_state = env.reset()
    
    initial_balance = env.balance
    initial_position = env.position
    
    # Execute action 0 (HOLD)
    next_state, reward, done, info = env.step(0)
    
    # Balance and position should remain unchanged (allowing for floating point)
    assert abs(env.balance - initial_balance) < 1e-6, \
        f"HOLD action changed balance: {initial_balance} -> {env.balance}"
    assert abs(env.position - initial_position) < 1e-6, \
        f"HOLD action changed position: {initial_position} -> {env.position}"


def test_env_action_mapping_buy():
    """Verify that action 1 (BUY) triggers a buy in the environment."""
    features_df = pd.DataFrame(np.random.randn(10, 5), columns=[f'f{i}' for i in range(5)])
    raw_df = pd.DataFrame({
        'close': [100.0] * 10,
        'open': [99.0] * 10,
        'high': [101.0] * 10,
        'low': [98.0] * 10,
        'volume': [1000.0] * 10
    })
    
    env = RLTradingEnv(features_df=features_df, raw_df=raw_df, initial_balance=10000.0)
    env.reset()
    
    initial_balance = env.balance
    
    # Execute action 1 (BUY)
    next_state, reward, done, info = env.step(1)
    
    # Balance should decrease, position should increase
    assert env.balance < initial_balance, \
        f"BUY action did not reduce balance: {initial_balance} -> {env.balance}"
    assert env.position > 0, \
        f"BUY action did not increase position: {env.position}"


def test_env_action_mapping_sell():
    """Verify that action 2 (SELL) triggers a sell in the environment."""
    features_df = pd.DataFrame(np.random.randn(10, 5), columns=[f'f{i}' for i in range(5)])
    raw_df = pd.DataFrame({
        'close': [100.0] * 10,
        'open': [99.0] * 10,
        'high': [101.0] * 10,
        'low': [98.0] * 10,
        'volume': [1000.0] * 10
    })
    
    env = RLTradingEnv(features_df=features_df, raw_df=raw_df, initial_balance=10000.0)
    env.reset()
    
    # First buy to have a position
    env.step(1)  # BUY
    position_after_buy = env.position
    balance_after_buy = env.balance
    
    # Now sell
    next_state, reward, done, info = env.step(2)  # SELL
    
    # Position should decrease to 0, balance should increase
    assert env.position < position_after_buy, \
        f"SELL action did not reduce position: {position_after_buy} -> {env.position}"
    assert env.balance > balance_after_buy, \
        f"SELL action did not increase balance: {balance_after_buy} -> {env.balance}"


def test_agent_action_semantics():
    """Verify that agent returns actions in the correct range and uses correct labels."""
    # Create a minimal agent for testing
    state_size = 10
    action_size = 3
    
    config = {
        'training_mode': False,
        'epsilon_inference': 0.0,  # Deterministic for testing
        'gamma': 0.99,
        'batch_size': 32,
        'learning_rate': 0.001,
        'buffer_size': 10000,
        'target_update_freq': 100,
    }
    
    agent = TradingRLAgent(state_size=state_size, action_size=action_size, config=config)
    agent.set_inference_mode(epsilon=0.0)
    
    # Create a dummy state
    state = np.random.randn(state_size).astype(np.float32)
    
    # Get action
    action, meta = agent.get_action_with_meta(state, training=False)
    
    # Action should be in valid range [0, 1, 2]
    assert action in [0, 1, 2], f"Invalid action: {action}"
    
    # Meta should contain probabilities
    assert 'probabilities' in meta, "Meta missing 'probabilities'"
    assert len(meta['probabilities']) == 3, \
        f"Expected 3 probabilities, got {len(meta['probabilities'])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
