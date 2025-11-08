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

# =========================================================================
# === YENİ EKLENECEK BÖLÜM: Experience Replay Buffer ===
# =========================================================================
class ExperienceReplay:
    """
    A replay buffer for storing and sampling experiences for the RL agent.
    Crucial for stabilizing the learning process by breaking correlations in data.
    """
    def __init__(self, buffer_size: int):
        """
        Initializes the experience replay buffer.

        Args:
            buffer_size (int): The maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=buffer_size)
        logger.info(f"Initialized ExperienceReplay buffer with max_size={buffer_size}")

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size: int) -> List:
        """
        Samples a random batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            A list of experience tuples, or an empty list if not enough samples.
        """
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """
        Returns the current number of experiences in the buffer.
        """
        return len(self.buffer)
# =========================================================================
# === YENİ BÖLÜM SONU ===
# =========================================================================

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
        
        def _get_default_config(self) -> Dict[str, Any]:
            """Provides default values for the agent's configuration."""
            return {
                'learning_rate': 0.0001,
                'gamma': 0.99,
                'buffer_size': 100000,
                'batch_size': 64,
                'target_update_freq': 10,
                'training_mode': False,
                'hold_confidence_threshold': 0.60,
                'epsilon_inference': 0.01,
                'epsilon_start': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'regime_bias_strength': 5.0,
                'risk_penalty_strength': 100.0,
            }
        
        def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
            """
            Initialize Trading RL Agent using a configuration dictionary.

            Args:
                state_size (int): Dimension of state space (market features).
                action_size (int): Number of possible actions.
                config (Dict[str, Any]): Configuration dictionary, typically from the
                                         'reinforcement_learning' section of the YAML config.
            """
            # Merge provided config with defaults for robustness
            default_config = self._get_default_config()
            self.config = {**default_config, **config}

            self.state_size = state_size
            self.action_size = action_size
            
            # Core Algorithm Parameters from config
            self.gamma = self.config['gamma']
            self.batch_size = self.config['batch_size']
            self.target_update_freq = self.config['target_update_freq']
            
            # Behavior and Mode Parameters from config
            self.training_mode = self.config.get('training_mode', False)
            self.hold_confidence_threshold = self.config.get('hold_confidence_threshold', 0.60)
            
            # Epsilon values from config
            self.epsilon = self.config.get('epsilon_start', 1.0) if self.training_mode else self.config.get('epsilon_inference', 0.01)
            self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
            self.epsilon_min = self.config.get('epsilon_min', 0.01)
            
            # === DEBUG: Log epsilon initialization ===
            logger.info(f"🎯 Epsilon Initialization:")
            logger.info(f"   training_mode:      {self.training_mode}")
            logger.info(f"   epsilon_start:      {self.config.get('epsilon_start', 'NOT SET')}")
            logger.info(f"   epsilon_inference:  {self.config.get('epsilon_inference', 'NOT SET')}")
            logger.info(f"   epsilon (selected): {self.epsilon:.4f}")
            logger.info(f"   epsilon_decay:      {self.epsilon_decay:.4f}")
            logger.info(f"   epsilon_min:        {self.epsilon_min:.4f}")
            
            if self.training_mode and self.epsilon != 1.0:
                logger.error("="*70)
                logger.error("❌ EPSILON INITIALIZATION ERROR!")
                logger.error("="*70)
                logger.error(f"   Expected: 1.0 (training mode)")
                logger.error(f"   Got:      {self.epsilon:.4f}")
                logger.error(f"   This will prevent exploration during training!")
                logger.error("="*70)
            
            # Bias strengths from config
            self.regime_bias_strength = self.config.get('regime_bias_strength', 5.0)
            self.risk_penalty_strength = self.config.get('risk_penalty_strength', 100.0)

            self.update_counter = 0

            # Initialize Q-networks
            self.q_network = DQNNetwork(state_size, action_size)
            self.target_network = DQNNetwork(state_size, action_size)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            
            # Optimizer
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])
            
            # Experience replay
            self.memory = ExperienceReplay(self.config['buffer_size'])
            
            # Training metrics
            self.training_history = {
                'losses': [],
                'q_values': [],
                'rewards': []
            }
            
            logger.info(f"Initialized TradingRLAgent with state_size={state_size}, action_size={action_size}")
            logger.info(f"RL Agent Config: training_mode={self.training_mode}, hold_threshold={self.hold_confidence_threshold}, regime_bias={self.regime_bias_strength}")
        
        def set_memory(self, memory):
            """Set experience replay buffer."""
            self.memory = memory
        
        def act(self, state: np.ndarray, market_regime: str = None, 
                risk_constraints: Dict = None, training: bool = False) -> int:
            """
            Select action based on current state using epsilon-greedy policy.
    
            Args:
                state (np.ndarray): The current state from the environment.
                market_regime (str, optional): The current market regime to apply bias.
                risk_constraints (Dict, optional): Any risk constraints to apply.
                training (bool, optional): If True, enables exploration (epsilon-greedy). 
                                           If False, uses exploitation-only mode. Defaults to False.
            """
            if state is None:
                logger.warning("RL Agent received None state, defaulting to HOLD (1).")
                return 1

            if training and random.random() < self.epsilon:
                action = random.randrange(self.action_size)
                logger.debug(f"🤖 [RL-ACT] Exploration: Selected random action -> {['BUY', 'HOLD', 'SELL'][action]}")
                return action
            
            with torch.no_grad():
                self.q_network.eval()
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                raw_q_values = self.q_network(state_tensor)
                
                adjusted_q_values = raw_q_values.clone()
                if risk_constraints:
                    adjusted_q_values = self._apply_risk_constraints(adjusted_q_values, risk_constraints)
                if market_regime:
                    adjusted_q_values = self._apply_regime_bias(adjusted_q_values, market_regime)
    
                probabilities = torch.softmax(adjusted_q_values, dim=1).squeeze().cpu().numpy()
                best_action = int(np.argmax(probabilities))
                best_prob = probabilities[best_action]
                
                # "Uncertain HOLD" kontrolü - Sadece canlı modda (training=False) çalışmalı
                if not training and best_action == 1 and best_prob < self.hold_confidence_threshold:
                    sorted_indices = np.argsort(probabilities)[::-1]
                    second_best_action = int(sorted_indices[1])
                    logger.warning(
                        f"🤖 [RL-OVERRIDE] Agent uncertain on HOLD (prob: {best_prob:.2f} < {self.hold_confidence_threshold}). "
                        f"Overriding with 2nd choice: {['BUY', 'HOLD', 'SELL'][second_best_action]}"
                    )
                    return second_best_action
    
                # Eğitim devam ediyorsa modeli tekrar train moduna al
                if training:
                    self.q_network.train()
    
                return best_action
        
        def _apply_risk_constraints(self, q_values: torch.Tensor, 
                                   risk_constraints: Dict) -> torch.Tensor:
            """Apply risk management constraints to Q-values using configurable penalty."""
            q_adjusted = q_values.clone()
            penalty = self.risk_penalty_strength

            if risk_constraints.get('max_position_reached', False):
                q_adjusted[0, 0] -= penalty  # Penalize BUY
            
            if risk_constraints.get('max_drawdown_reached', False):
                q_adjusted[0, [0, 2]] -= penalty / 2  # Penalize both BUY and SELL
            
            return q_adjusted
        
        def _apply_regime_bias(self, q_values: torch.Tensor, 
                              market_regime: str) -> torch.Tensor:
            """Apply market regime bias to Q-values using configurable strength."""
            q_adjusted = q_values.clone()
            bias = self.regime_bias_strength

            if market_regime == 'bullish':
                q_adjusted[0, 0] += bias      # Boost BUY
            elif market_regime == 'bearish':
                q_adjusted[0, 2] += bias      # Boost SELL
            elif market_regime == 'neutral':
                q_adjusted[0, 1] += bias / 2  # Slightly boost HOLD
            
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
            
            metrics = {'loss': 0.0, 'q_value': 0.0, 'epsilon': self.epsilon}
            
            # Only train if we have enough experiences
            if self.memory is None:
                logger.debug("⚠️ learn_from_experience: memory is None, skipping training")
                return metrics
            
            buffer_size = len(self.memory.buffer)
            if buffer_size < self.batch_size:
                # Log only occasionally to avoid spam (every 10th call)
                if hasattr(self, '_learn_call_count'):
                    self._learn_call_count += 1
                else:
                    self._learn_call_count = 1
                
                if self._learn_call_count % 10 == 0:
                    logger.debug(f"⚠️ Buffer not full yet: {buffer_size}/{self.batch_size} samples, skipping training")
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
            old_epsilon = self.epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Log epsilon decay (only first time and every 50 times)
            if not hasattr(self, '_epsilon_decay_count'):
                self._epsilon_decay_count = 0
                logger.info(f"✅ First successful learning! Epsilon decay started:")
                logger.info(f"   Epsilon: {old_epsilon:.4f} → {self.epsilon:.4f}")
                logger.info(f"   Buffer: {len(self.memory.buffer)}/{self.memory.buffer.maxlen} samples")
            
            self._epsilon_decay_count += 1
            if self._epsilon_decay_count % 50 == 0:
                logger.info(f"📊 Learning update #{self._epsilon_decay_count}: Epsilon = {self.epsilon:.4f}")
            
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
            """
            Load model weights from a checkpoint file.
            Includes error handling for missing files.
            """
            try:
                checkpoint = torch.load(path)
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.training_history = checkpoint.get('training_history', self.training_history)
                # Başarı durumunda net bir log mesajı ekle
                logger.info(f"✅ RL Agent model loaded successfully from {path}")
            except FileNotFoundError:
                # Dosya bulunamazsa hata logu yazdır
                logger.error(f"❌ RL Agent model file not found at {path}. Agent will use untrained weights.")
            except Exception as e:
                # Diğer olası hatalar için detaylı log yazdır
                logger.error(f"❌ Failed to load RL Agent model from {path}: {e}", exc_info=True)
        
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
