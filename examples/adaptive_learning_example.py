#!/usr/bin/env python3
"""
Example: Phase 4.2 Adaptive Learning System

Demonstrates the complete workflow of the reinforcement learning system
for trading strategy optimization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from typing import Dict, Tuple

from src.ml.reinforcement_learning import TradingRLAgent
from src.ml.experience_replay import ExperienceReplay, EpisodeBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTradingEnvironment:
    """Mock trading environment for demonstration."""
    
    def __init__(self):
        self.current_price = 100.0
        self.position = 0.0  # 0 = no position, >0 = long, <0 = short
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.current_price = 100.0
        self.position = 0.0
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current market state."""
        # Simplified state: price change, position, volatility, etc.
        price_change = (self.current_price - 100.0) / 100.0
        state = np.array([
            price_change,
            self.position,
            np.sin(self.step_count / 10.0),  # Mock cyclical pattern
            np.random.randn() * 0.1,  # Noise
        ] + [np.random.randn() * 0.1 for _ in range(46)])  # Fill to 50 features
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action and return (next_state, reward, done).
        
        Actions:
        0: Buy (go long)
        1: Hold
        2: Sell (go short)
        """
        self.step_count += 1
        
        # Simulate price movement
        price_change = np.random.randn() * 0.02
        self.current_price *= (1 + price_change)
        
        # Execute action
        reward = 0.0
        if action == 0:  # Buy
            if self.position <= 0:  # Open long or close short
                reward = price_change * (1 - abs(self.position))  # Reward for correct direction
                self.position = 1.0
            else:
                reward = -0.01  # Small penalty for redundant action
        elif action == 2:  # Sell
            if self.position >= 0:  # Open short or close long
                reward = -price_change * (1 + abs(self.position))  # Reward for correct direction
                self.position = -1.0
            else:
                reward = -0.01
        else:  # Hold
            if self.position != 0:
                reward = price_change * self.position  # Reward for holding position
            else:
                reward = -0.005  # Small penalty for not being in market
        
        # Get next state
        next_state = self._get_state()
        
        # Check if done
        done = (self.step_count >= self.max_steps) or (abs(self.current_price - 100.0) > 20.0)
        
        return next_state, reward, done


def example_basic_training():
    """Example 1: Basic training loop."""
    print("\n" + "="*60)
    print("Example 1: Basic Training Loop")
    print("="*60 + "\n")
    
    # Initialize components
    agent = TradingRLAgent(
        state_size=50,
        action_size=3,  # buy, hold, sell
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    memory = ExperienceReplay(max_size=10000, priority_alpha=0.6)
    agent.set_memory(memory)
    
    env = MockTradingEnvironment()
    
    # Training loop
    num_episodes = 10
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(100):
            # Select action
            action = agent.act(state, training=True)
            
            # Execute in environment
            next_state, reward, done = env.step(action)
            
            # Learn from experience
            metrics = agent.learn_from_experience(
                state, action, reward, next_state, done
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        logger.info(
            f"Episode {episode+1:2d}: "
            f"Reward={episode_reward:7.4f}, "
            f"Epsilon={agent.epsilon:.4f}, "
            f"Buffer={len(memory)}"
        )
    
    # Training summary
    summary = agent.get_training_summary()
    logger.info(f"\nTraining Summary: {summary}")
    
    # Buffer statistics
    buffer_stats = memory.get_statistics()
    logger.info(f"Buffer Statistics: {buffer_stats}")
    
    return agent, memory, episode_rewards


def example_risk_aware_trading():
    """Example 2: Risk-aware action selection."""
    print("\n" + "="*60)
    print("Example 2: Risk-Aware Trading")
    print("="*60 + "\n")
    
    agent = TradingRLAgent(state_size=50, action_size=3)
    memory = ExperienceReplay(max_size=1000)
    agent.set_memory(memory)
    
    # Simulate different risk scenarios
    state = np.random.randn(50)
    
    # Scenario 1: Normal conditions
    logger.info("Scenario 1: Normal market conditions")
    action = agent.act(state, risk_constraints=None, training=False)
    logger.info(f"  Selected action: {action}")
    
    # Scenario 2: Max position reached
    logger.info("\nScenario 2: Max position reached")
    risk_constraints = {'max_position_reached': True}
    action = agent.act(state, risk_constraints=risk_constraints, training=False)
    logger.info(f"  Selected action: {action} (should avoid aggressive buys)")
    
    # Scenario 3: Max drawdown reached
    logger.info("\nScenario 3: Max drawdown reached")
    risk_constraints = {'max_drawdown_reached': True}
    action = agent.act(state, risk_constraints=risk_constraints, training=False)
    logger.info(f"  Selected action: {action} (should be conservative)")


def example_regime_aware_trading():
    """Example 3: Market regime-aware trading."""
    print("\n" + "="*60)
    print("Example 3: Market Regime-Aware Trading")
    print("="*60 + "\n")
    
    agent = TradingRLAgent(state_size=50, action_size=3)
    memory = ExperienceReplay(max_size=1000)
    agent.set_memory(memory)
    
    state = np.random.randn(50)
    
    # Test different regimes
    for regime in ['bullish', 'neutral', 'bearish']:
        logger.info(f"Regime: {regime}")
        action = agent.act(state, market_regime=regime, training=False)
        logger.info(f"  Selected action: {action}\n")


def example_episode_analysis():
    """Example 4: Episode buffer and analysis."""
    print("\n" + "="*60)
    print("Example 4: Episode Analysis")
    print("="*60 + "\n")
    
    episode_buffer = EpisodeBuffer(max_episodes=100)
    env = MockTradingEnvironment()
    
    # Simulate episodes with different performance
    for episode_idx in range(5):
        state = env.reset()
        
        for step in range(20):
            action = np.random.choice([0, 1, 2])
            next_state, reward, done = env.step(action)
            
            episode_buffer.add_transition(state, action, reward, next_state, done)
            
            state = next_state
            if done:
                break
    
    # Analyze episodes
    stats = episode_buffer.get_statistics()
    logger.info(f"Episode Statistics:")
    logger.info(f"  Episodes: {stats['num_episodes']}")
    logger.info(f"  Avg Length: {stats['avg_length']:.1f}")
    logger.info(f"  Avg Return: {stats['avg_return']:.4f}")
    if 'best_return' in stats:
        logger.info(f"  Best Return: {stats['best_return']:.4f}")
        logger.info(f"  Worst Return: {stats['worst_return']:.4f}")
    
    # Get best episodes
    best_episodes = episode_buffer.get_best_episodes(n=3)
    logger.info(f"\nTop 3 Episodes:")
    for i, ep in enumerate(best_episodes):
        logger.info(f"  #{i+1}: Return={ep['return']:.4f}, Length={ep['length']}")


def example_model_persistence():
    """Example 5: Save and load models."""
    print("\n" + "="*60)
    print("Example 5: Model Persistence")
    print("="*60 + "\n")
    
    import tempfile
    import os
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'agent.pt')
    buffer_path = os.path.join(temp_dir, 'buffer.pkl')
    
    # Train agent
    logger.info("Training agent...")
    agent = TradingRLAgent(state_size=50, action_size=3)
    memory = ExperienceReplay(max_size=1000)
    agent.set_memory(memory)
    
    # Add some experiences
    for i in range(100):
        state = np.random.randn(50)
        next_state = np.random.randn(50)
        memory.add_experience(state, i % 3, 0.1, next_state, False)
    
    # Save
    logger.info(f"Saving model to {model_path}")
    agent.save_model(model_path)
    memory.save_buffer(buffer_path)
    
    # Load into new agent
    logger.info("Loading model into new agent...")
    agent2 = TradingRLAgent(state_size=50, action_size=3)
    memory2 = ExperienceReplay(max_size=1000)
    
    agent2.load_model(model_path)
    memory2.load_buffer(buffer_path)
    agent2.set_memory(memory2)
    
    logger.info(f"✓ Model loaded successfully")
    logger.info(f"  Epsilon: {agent2.epsilon:.4f}")
    logger.info(f"  Buffer size: {len(memory2)}")
    
    # Cleanup
    os.remove(model_path)
    os.remove(buffer_path)
    os.rmdir(temp_dir)


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# Phase 4.2: Adaptive Learning System - Examples")
    print("#"*60)
    
    try:
        # Run examples
        example_basic_training()
        example_risk_aware_trading()
        example_regime_aware_trading()
        example_episode_analysis()
        example_model_persistence()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
