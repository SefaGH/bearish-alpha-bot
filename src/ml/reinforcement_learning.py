"""
Reinforcement Learning Engine for Trading Strategy Optimization.

Implements Deep Q-Network (DQN) agent for continuous strategy improvement
through interaction with the trading environment.
"""

import numpy as np
from collections import deque
import random
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. RL agent will use mock implementation.")


if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network for trading action value estimation."""
        
        def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = None):
            """
            Initialize DQN network.
            
            Args:
                state_size: Dimension of state space
                action_size: Number of possible actions
                hidden_sizes: List of hidden layer sizes (default: [256, 128, 64])
            """
            super().__init__()
            
            self.state_size = state_size
            self.action_size = action_size
            
            if hidden_sizes is None:
                hidden_sizes = [256, 128, 64]
            
            # Build network layers
            layers = []
            prev_size = state_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, action_size))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, state):
            """
            Forward pass through network.
            
            Args:
                state: State tensor
                
            Returns:
                Q-values for each action
            """
            return self.network(state)


    class TradingRLAgent:
        """Reinforcement Learning agent for trading strategy optimization."""
        
        def __init__(self, state_size: int, action_size: int, 
                     learning_rate: float = 0.001,
                     gamma: float = 0.99,
                     epsilon: float = 1.0,
                     epsilon_decay: float = 0.995,
                     epsilon_min: float = 0.01,
                     batch_size: int = 64,
                     target_update_freq: int = 10):
            """
            Initialize Trading RL Agent.
            
            Args:
                state_size: Dimension of state space (market features)
                action_size: Number of possible actions
                learning_rate: Learning rate for optimizer
                gamma: Discount factor for future rewards
                epsilon: Initial exploration rate
                epsilon_decay: Decay rate for epsilon
                epsilon_min: Minimum exploration rate
                batch_size: Batch size for training
                target_update_freq: Frequency to update target network
            """
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq
            self.update_counter = 0
            
            # Initialize Q-networks
            self.q_network = DQNNetwork(state_size, action_size)
            self.target_network = DQNNetwork(state_size, action_size)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            
            # Optimizer
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
            
            # Experience replay (imported separately)
            self.memory = None  # Will be set externally
            
            # Training metrics
            self.training_history = {
                'losses': [],
                'q_values': [],
                'rewards': []
            }
            
            logger.info(f"Initialized TradingRLAgent with state_size={state_size}, action_size={action_size}")
        
        def set_memory(self, memory):
            """Set experience replay buffer."""
            self.memory = memory
        
        def act(self, state: np.ndarray, market_regime: str = None, 
                risk_constraints: Dict = None, training: bool = True) -> int:
            """
            Select action based on current state using epsilon-greedy policy.
            
            Args:
                state: Current market state features
                market_regime: Current market regime (e.g., 'bullish', 'neutral', 'bearish')
                risk_constraints: Risk management constraints
                training: Whether in training mode (applies epsilon-greedy)
                
            Returns:
                Selected action index
            """
            # Epsilon-greedy exploration during training
            if training and random.random() < self.epsilon:
                # Random exploration
                action = random.randrange(self.action_size)
                logger.debug(f"Exploration: random action {action}")
                return action
            
            # Exploitation: use Q-network
            # Set to eval mode to disable dropout
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                
                # Apply risk constraints if provided
                if risk_constraints:
                    q_values = self._apply_risk_constraints(q_values, risk_constraints)
                
                # Apply market regime bias if provided
                if market_regime:
                    q_values = self._apply_regime_bias(q_values, market_regime)
                
                action = q_values.argmax().item()
                logger.debug(f"Exploitation: selected action {action} with Q-value {q_values[0][action]:.4f}")
            
            # Back to train mode for gradient updates
            if training:
                self.q_network.train()
            
            return action
        
        def _apply_risk_constraints(self, q_values: torch.Tensor, 
                                   risk_constraints: Dict) -> torch.Tensor:
            """
            Apply risk management constraints to Q-values.
            
            Args:
                q_values: Raw Q-values from network
                risk_constraints: Dictionary with risk limits
                
            Returns:
                Adjusted Q-values
            """
            # Penalize risky actions based on constraints
            q_adjusted = q_values.clone()
            
            # Example: if max_position_reached, penalize buy actions
            if risk_constraints.get('max_position_reached', False):
                # Assume actions 0-2 are buy-related, 3-5 are sell/hold
                q_adjusted[0, :3] -= 100  # Large penalty
            
            # Example: if max_drawdown_reached, penalize aggressive actions
            if risk_constraints.get('max_drawdown_reached', False):
                q_adjusted[0, [0, 2]] -= 50  # Penalize aggressive buy/sell
            
            return q_adjusted
        
        def _apply_regime_bias(self, q_values: torch.Tensor, 
                              market_regime: str) -> torch.Tensor:
            """
            Apply market regime bias to action selection.
            
            Args:
                q_values: Raw Q-values from network
                market_regime: Current market regime
                
            Returns:
                Adjusted Q-values with regime bias
            """
            q_adjusted = q_values.clone()
            
            # Boost certain actions based on regime
            if market_regime == 'bullish':
                # Favor long positions in bullish regime
                q_adjusted[0, 0] += 5  # Boost buy action
            elif market_regime == 'bearish':
                # Favor short positions in bearish regime
                q_adjusted[0, 2] += 5  # Boost sell action
            elif market_regime == 'neutral':
                # Favor hold/wait in neutral regime
                q_adjusted[0, 1] += 3  # Boost hold action
            
            return q_adjusted
        
        def learn_from_experience(self, state: np.ndarray, action: int, 
                                 reward: float, next_state: np.ndarray, 
                                 done: bool) -> Dict[str, float]:
            """
            Learn from trading experience using Q-learning with experience replay.
            
            Args:
                state: Previous state
                action: Action taken
                reward: Reward received
                next_state: New state after action
                done: Whether episode is done
                
            Returns:
                Training metrics
            """
            # Store experience in replay buffer
            if self.memory is not None:
                self.memory.add_experience(state, action, reward, next_state, done)
            
            metrics = {'loss': 0.0, 'q_value': 0.0}
            
            # Only train if we have enough experiences
            if self.memory is None or len(self.memory.buffer) < self.batch_size:
                return metrics
            
            # Sample batch from replay buffer
            batch = self.memory.sample_batch(self.batch_size)
            if not batch:
                return metrics
            
            # Extract batch components
            states = torch.FloatTensor(np.array([exp[0] for exp in batch]))
            actions = torch.LongTensor([exp[1] for exp in batch])
            rewards = torch.FloatTensor([exp[2] for exp in batch])
            next_states = torch.FloatTensor(np.array([exp[3] for exp in batch]))
            dones = torch.FloatTensor([exp[4] for exp in batch])
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Target Q-values using target network (Double DQN)
            with torch.no_grad():
                # Select best actions using main network
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                # Evaluate using target network
                next_q_values = self.target_network(next_states).gather(1, next_actions)
                target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
            
            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update target network periodically
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                logger.debug(f"Target network updated at step {self.update_counter}")
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Track metrics
            metrics = {
                'loss': loss.item(),
                'q_value': current_q_values.mean().item(),
                'epsilon': self.epsilon
            }
            
            self.training_history['losses'].append(metrics['loss'])
            self.training_history['q_values'].append(metrics['q_value'])
            self.training_history['rewards'].append(rewards.mean().item())
            
            return metrics
        
        def save_model(self, path: str):
            """Save model weights."""
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_history': self.training_history
            }, path)
            logger.info(f"Model saved to {path}")
        
        def load_model(self, path: str):
            """Load model weights."""
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.training_history = checkpoint.get('training_history', self.training_history)
            logger.info(f"Model loaded from {path}")
        
        def get_training_summary(self) -> Dict[str, Any]:
            """Get training summary statistics."""
            if not self.training_history['losses']:
                return {'status': 'no_training_data'}
            
            return {
                'total_updates': len(self.training_history['losses']),
                'avg_loss': np.mean(self.training_history['losses'][-100:]),
                'avg_q_value': np.mean(self.training_history['q_values'][-100:]),
                'avg_reward': np.mean(self.training_history['rewards'][-100:]),
                'current_epsilon': self.epsilon,
                'exploration_rate': f"{self.epsilon:.2%}"
            }

else:
    # Mock implementation when PyTorch is not available
    class DQNNetwork:
        """Mock DQN network (PyTorch not available)."""
        
        def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = None):
            self.state_size = state_size
            self.action_size = action_size
            logger.info("Initialized mock DQNNetwork (PyTorch not available)")
    
    
    class TradingRLAgent:
        """Mock Trading RL Agent (PyTorch not available)."""
        
        def __init__(self, state_size: int, action_size: int, **kwargs):
            self.state_size = state_size
            self.action_size = action_size
            self.epsilon = 1.0
            self.memory = None
            self.training_history = {'losses': [], 'q_values': [], 'rewards': []}
            logger.info("Initialized mock TradingRLAgent (PyTorch not available)")
        
        def set_memory(self, memory):
            """Set experience replay buffer."""
            self.memory = memory
        
        def act(self, state: np.ndarray, market_regime: str = None, 
                risk_constraints: Dict = None, training: bool = True) -> int:
            """Mock action selection - returns random action."""
            return random.randrange(self.action_size)
        
        def learn_from_experience(self, state: np.ndarray, action: int, 
                                 reward: float, next_state: np.ndarray, 
                                 done: bool) -> Dict[str, float]:
            """Mock learning - does nothing."""
            return {'loss': 0.0, 'q_value': 0.0, 'epsilon': self.epsilon}
        
        def save_model(self, path: str):
            """Mock save."""
            logger.info(f"Mock save to {path}")
        
        def load_model(self, path: str):
            """Mock load."""
            logger.info(f"Mock load from {path}")
        
        def get_training_summary(self) -> Dict[str, Any]:
            """Mock training summary."""
            return {'status': 'mock_mode', 'pytorch_available': False}
