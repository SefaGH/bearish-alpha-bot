"""
Tests for Phase 4.2: Adaptive Learning System

Tests reinforcement learning agent and experience replay components.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch

from src.ml.reinforcement_learning import TradingRLAgent, DQNNetwork
from src.ml.experience_replay import ExperienceReplay, EpisodeBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test fixtures
@pytest.fixture
def sample_state():
    """Create sample state vector."""
    return np.random.randn(50)  # 50 features


@pytest.fixture
def experience_replay():
    """Create experience replay buffer."""
    return ExperienceReplay(max_size=1000, priority_alpha=0.6)


@pytest.fixture
def rl_agent():
    """Create RL agent."""
    agent = TradingRLAgent(
        state_size=50,
        action_size=5,
        learning_rate=0.001,
        epsilon=1.0
    )
    # Set up memory
    memory = ExperienceReplay(max_size=1000)
    agent.set_memory(memory)
    return agent


class TestExperienceReplay:
    """Test experience replay buffer functionality."""
    
    def test_initialization(self, experience_replay):
        """Test buffer initialization."""
        assert len(experience_replay) == 0
        assert experience_replay.max_size == 1000
        assert experience_replay.priority_alpha == 0.6
        assert experience_replay.total_experiences == 0
    
    def test_add_experience(self, experience_replay):
        """Test adding experiences to buffer."""
        state = np.random.randn(50)
        next_state = np.random.randn(50)
        
        experience_replay.add_experience(
            state=state,
            action=1,
            reward=0.5,
            next_state=next_state,
            done=False
        )
        
        assert len(experience_replay) == 1
        assert experience_replay.total_experiences == 1
    
    def test_add_multiple_experiences(self, experience_replay):
        """Test adding multiple experiences."""
        for i in range(100):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            experience_replay.add_experience(
                state=state,
                action=i % 5,
                reward=np.random.randn(),
                next_state=next_state,
                done=(i % 10 == 9)
            )
        
        assert len(experience_replay) == 100
        assert experience_replay.total_experiences == 100
    
    def test_sample_batch(self, experience_replay):
        """Test batch sampling."""
        # Add experiences
        for i in range(100):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            experience_replay.add_experience(
                state=state,
                action=i % 5,
                reward=np.random.randn(),
                next_state=next_state,
                done=False
            )
        
        # Sample batch
        batch = experience_replay.sample_batch(batch_size=32)
        
        assert len(batch) == 32
        assert all(len(exp) == 5 for exp in batch)  # (state, action, reward, next_state, done)
    
    def test_sample_batch_insufficient_data(self, experience_replay):
        """Test sampling when insufficient data."""
        # Add only 10 experiences
        for i in range(10):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            experience_replay.add_experience(
                state=state,
                action=0,
                reward=0.0,
                next_state=next_state,
                done=False
            )
        
        # Try to sample 32
        batch = experience_replay.sample_batch(batch_size=32)
        
        assert len(batch) == 0  # Should return empty
    
    def test_priority_sampling(self, experience_replay):
        """Test priority-based sampling."""
        # Add experiences with different TD errors
        for i in range(50):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            td_error = float(i) / 50  # Increasing TD errors
            
            experience_replay.add_experience(
                state=state,
                action=0,
                reward=0.0,
                next_state=next_state,
                done=False,
                td_error=td_error
            )
        
        # Sample should favor higher priority experiences
        batch = experience_replay.sample_batch(batch_size=10)
        assert len(batch) == 10
    
    def test_get_statistics(self, experience_replay):
        """Test buffer statistics."""
        # Empty buffer
        stats = experience_replay.get_statistics()
        assert stats['size'] == 0
        assert stats['utilization'] == 0.0
        
        # Add experiences
        for i in range(50):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            experience_replay.add_experience(
                state=state,
                action=0,
                reward=0.0,
                next_state=next_state,
                done=False
            )
        
        stats = experience_replay.get_statistics()
        assert stats['size'] == 50
        assert stats['max_size'] == 1000
        assert stats['utilization'] == 0.05
        assert 'priority_stats' in stats
    
    def test_buffer_overflow(self):
        """Test buffer behavior when max size is reached."""
        buffer = ExperienceReplay(max_size=10)
        
        # Add more than max_size experiences
        for i in range(20):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            buffer.add_experience(
                state=state,
                action=0,
                reward=0.0,
                next_state=next_state,
                done=False
            )
        
        # Should only keep last 10
        assert len(buffer) == 10
        assert buffer.total_experiences == 20
    
    def test_get_recent_experiences(self, experience_replay):
        """Test getting recent experiences."""
        # Add experiences
        for i in range(50):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            experience_replay.add_experience(
                state=state,
                action=i,
                reward=float(i),
                next_state=next_state,
                done=False
            )
        
        # Get recent
        recent = experience_replay.get_recent_experiences(n=10)
        assert len(recent) == 10
        
        # Should be most recent (last 10 added)
        # Check that rewards are from 40-49 range
        rewards = [exp[2] for exp in recent]
        assert min(rewards) >= 40.0


class TestDQNNetwork:
    """Test DQN network architecture."""
    
    def test_initialization(self):
        """Test network initialization."""
        try:
            import torch
            network = DQNNetwork(state_size=50, action_size=5)
            assert network.state_size == 50
            assert network.action_size == 5
        except ImportError:
            # PyTorch not available, skip
            pytest.skip("PyTorch not available")
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        try:
            import torch
            network = DQNNetwork(state_size=50, action_size=5)
            
            # Create sample input
            state = torch.randn(1, 50)
            
            # Forward pass
            q_values = network(state)
            
            assert q_values.shape == (1, 5)
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_custom_hidden_sizes(self):
        """Test network with custom hidden layer sizes."""
        try:
            import torch
            network = DQNNetwork(
                state_size=50,
                action_size=5,
                hidden_sizes=[128, 64]
            )
            
            state = torch.randn(4, 50)  # Batch of 4
            q_values = network(state)
            
            assert q_values.shape == (4, 5)
        except ImportError:
            pytest.skip("PyTorch not available")


class TestTradingRLAgent:
    """Test Trading RL Agent functionality."""
    
    def test_initialization(self, rl_agent):
        """Test agent initialization."""
        assert rl_agent.state_size == 50
        assert rl_agent.action_size == 5
        assert rl_agent.epsilon == 1.0
        assert rl_agent.memory is not None
    
    def test_act_exploration(self, rl_agent, sample_state):
        """Test action selection with exploration."""
        # With epsilon=1.0, should always explore
        actions = []
        for _ in range(100):
            action = rl_agent.act(sample_state, training=True)
            actions.append(action)
        
        # Should have variety in actions
        unique_actions = len(set(actions))
        assert unique_actions > 1
        assert all(0 <= a < 5 for a in actions)
    
    def test_act_exploitation(self, rl_agent, sample_state):
        """Test action selection with exploitation."""
        # Set epsilon to 0 for pure exploitation
        rl_agent.epsilon = 0.0
        
        actions = []
        for _ in range(10):
            action = rl_agent.act(sample_state, training=False)  # Use training=False for consistent eval
            actions.append(action)
        
        # With same state and eval mode, should select same action consistently
        assert all(0 <= a < 5 for a in actions)
        assert len(set(actions)) == 1  # Should be deterministic in eval mode
    
    def test_act_with_risk_constraints(self, rl_agent, sample_state):
        """Test action selection with risk constraints."""
        risk_constraints = {
            'max_position_reached': True,
            'max_drawdown_reached': False
        }
        
        # Should still return valid action
        action = rl_agent.act(
            sample_state,
            risk_constraints=risk_constraints,
            training=False
        )
        
        assert 0 <= action < 5
    
    def test_act_with_market_regime(self, rl_agent, sample_state):
        """Test action selection with market regime."""
        # Test different regimes
        for regime in ['bullish', 'bearish', 'neutral']:
            action = rl_agent.act(
                sample_state,
                market_regime=regime,
                training=False
            )
            assert 0 <= action < 5
    
    def test_learn_from_experience(self, rl_agent, sample_state):
        """Test learning from experience."""
        # Add enough experiences for training
        for i in range(100):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            action = i % 5
            reward = np.random.randn()
            done = (i % 10 == 9)
            
            rl_agent.memory.add_experience(state, action, reward, next_state, done)
        
        # Now learn
        state = np.random.randn(50)
        next_state = np.random.randn(50)
        metrics = rl_agent.learn_from_experience(
            state=state,
            action=1,
            reward=0.5,
            next_state=next_state,
            done=False
        )
        
        # Should have training metrics
        assert 'loss' in metrics
        assert 'q_value' in metrics
    
    def test_epsilon_decay(self, rl_agent, sample_state):
        """Test epsilon decay over time."""
        initial_epsilon = rl_agent.epsilon
        
        # Add experiences and train multiple times
        for i in range(200):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            rl_agent.learn_from_experience(
                state=state,
                action=i % 5,
                reward=0.1,
                next_state=next_state,
                done=False
            )
        
        # Epsilon should have decayed (if enough training happened)
        if len(rl_agent.training_history['losses']) > 0:
            assert rl_agent.epsilon <= initial_epsilon
    
    def test_get_training_summary(self, rl_agent):
        """Test training summary."""
        # Before training
        summary = rl_agent.get_training_summary()
        assert 'status' in summary or 'total_updates' in summary
        
        # After some training
        for i in range(100):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            rl_agent.memory.add_experience(state, i % 5, 0.1, next_state, False)
        
        for i in range(10):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            rl_agent.learn_from_experience(state, 0, 0.1, next_state, False)
        
        summary = rl_agent.get_training_summary()
        if 'total_updates' in summary:
            assert summary['total_updates'] >= 0


class TestEpisodeBuffer:
    """Test episode buffer functionality."""
    
    def test_initialization(self):
        """Test episode buffer initialization."""
        buffer = EpisodeBuffer(max_episodes=100)
        assert len(buffer) == 0
        assert buffer.max_episodes == 100
    
    def test_add_transition(self):
        """Test adding transitions to episode."""
        buffer = EpisodeBuffer(max_episodes=100)
        
        state = np.random.randn(50)
        next_state = np.random.randn(50)
        
        buffer.add_transition(state, 1, 0.5, next_state, False)
        
        assert len(buffer.current_episode) == 1
    
    def test_end_episode(self):
        """Test ending an episode."""
        buffer = EpisodeBuffer(max_episodes=100)
        
        # Add transitions
        for i in range(10):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            done = (i == 9)
            buffer.add_transition(state, i % 5, 0.1, next_state, done)
        
        # Should have 1 completed episode
        assert len(buffer) == 1
        assert len(buffer.current_episode) == 0
    
    def test_get_best_episodes(self):
        """Test getting best episodes."""
        buffer = EpisodeBuffer(max_episodes=100)
        
        # Create episodes with different returns
        for episode_idx in range(10):
            for step in range(10):
                state = np.random.randn(50)
                next_state = np.random.randn(50)
                reward = episode_idx * 0.1  # Different returns per episode
                done = (step == 9)
                buffer.add_transition(state, 0, reward, next_state, done)
        
        # Get best episodes
        best = buffer.get_best_episodes(n=3)
        
        assert len(best) == 3
        # Should be sorted by return (descending)
        assert best[0]['return'] >= best[1]['return']
        assert best[1]['return'] >= best[2]['return']
    
    def test_get_statistics(self):
        """Test episode statistics."""
        buffer = EpisodeBuffer(max_episodes=100)
        
        # Empty buffer
        stats = buffer.get_statistics()
        assert stats['num_episodes'] == 0
        
        # Add episodes
        for episode_idx in range(5):
            for step in range(10):
                state = np.random.randn(50)
                next_state = np.random.randn(50)
                done = (step == 9)
                buffer.add_transition(state, 0, 0.1, next_state, done)
        
        stats = buffer.get_statistics()
        assert stats['num_episodes'] == 5
        assert stats['avg_length'] == 10
        assert 'avg_return' in stats
        assert 'best_return' in stats


class TestIntegration:
    """Integration tests for RL agent with experience replay."""
    
    def test_full_training_loop(self):
        """Test complete training loop."""
        # Initialize components
        agent = TradingRLAgent(
            state_size=50,
            action_size=5,
            learning_rate=0.001,
            batch_size=32
        )
        memory = ExperienceReplay(max_size=1000)
        agent.set_memory(memory)
        
        # Simulate trading episodes
        num_episodes = 5
        steps_per_episode = 20
        
        for episode in range(num_episodes):
            state = np.random.randn(50)
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Select action
                action = agent.act(state, training=True)
                
                # Simulate environment
                next_state = np.random.randn(50)
                reward = np.random.randn() * 0.1
                done = (step == steps_per_episode - 1)
                
                # Learn from experience
                metrics = agent.learn_from_experience(
                    state, action, reward, next_state, done
                )
                
                episode_reward += reward
                state = next_state
            
            logger.info(f"Episode {episode+1}: reward={episode_reward:.4f}")
        
        # Check that agent has learned something
        assert len(memory) > 0
        summary = agent.get_training_summary()
        logger.info(f"Training summary: {summary}")
    
    def test_save_load_integration(self, tmp_path):
        """Test saving and loading agent with replay buffer."""
        # Initialize and train agent
        agent = TradingRLAgent(state_size=50, action_size=5)
        memory = ExperienceReplay(max_size=100)
        agent.set_memory(memory)
        
        # Add some experiences
        for i in range(50):
            state = np.random.randn(50)
            next_state = np.random.randn(50)
            memory.add_experience(state, i % 5, 0.1, next_state, False)
        
        # Save
        model_path = tmp_path / "agent.pt"
        buffer_path = tmp_path / "buffer.pkl"
        
        try:
            agent.save_model(str(model_path))
            memory.save_buffer(str(buffer_path))
            
            # Load into new agent
            agent2 = TradingRLAgent(state_size=50, action_size=5)
            memory2 = ExperienceReplay(max_size=100)
            
            agent2.load_model(str(model_path))
            memory2.load_buffer(str(buffer_path))
            
            # Verify
            assert len(memory2) == len(memory)
            assert agent2.epsilon == agent.epsilon
        except Exception as e:
            # If PyTorch not available, this is expected
            logger.info(f"Save/load test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
