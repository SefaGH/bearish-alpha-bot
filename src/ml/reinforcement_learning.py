"""
Reinforcement Learning Engine for Trading Strategy Optimization.

Implements Deep Q-Network (DQN) agent for continuous strategy improvement
through interaction with the trading environment.

Action Mapping Convention:
    0 -> HOLD: No trade action
    1 -> BUY:  Open or increase long position
    2 -> SELL: Close position or open short
"""

import numpy as np
from collections import deque
import random
import logging
import math
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Canonical action mapping used throughout the RL stack
ACTION_LABELS = ['HOLD', 'BUY', 'SELL']

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
        
        def __init__(
            self,
            state_size: int,
            action_size: int,
            hidden_sizes: List[int] = None,
            *,
            learnable_head_scale: bool = False,
            initial_head_scale: float = 1.0,
            head_scale_lower: float = 0.1,
        ):
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

            scale_value = float(initial_head_scale)
            self.head_scale_min = float(max(head_scale_lower, 1e-6))

            if learnable_head_scale:
                delta = max(scale_value - self.head_scale_min, 1e-6)
                softplus_inverse = float(math.log(math.expm1(delta)))
                self.head_scale_raw = nn.Parameter(
                    torch.tensor([softplus_inverse], dtype=torch.float32)
                )
            else:
                frozen_scale = max(scale_value, self.head_scale_min)
                self.register_buffer(
                    "head_scale_buffer",
                    torch.tensor([frozen_scale], dtype=torch.float32),
                )

        def forward(self, state):
            """
            Forward pass through network.
            
            Args:
                state: State tensor
                
            Returns:
                Q-values for each action
            """
            q_values = self.network(state)
            if (
                hasattr(self, "head_scale_raw")
                or hasattr(self, "head_scale_buffer")
            ):
                q_values = q_values * self.head_scale
            return q_values

        @property
        def head_scale(self) -> torch.Tensor:
            if hasattr(self, "head_scale_raw"):
                return self.head_scale_min + F.softplus(self.head_scale_raw)
            return getattr(self, "head_scale_buffer")


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
                'regime_bias_strength': 2.0,
                'max_regime_bias': 3.0,
                'min_regime_confidence_for_bias': 0.6,
                'risk_penalty_strength': 100.0,
                'reward_clip_enabled': False,
                'reward_clip_min': -10.0,
                'reward_clip_max': 10.0,
                'reward_scale': 1.0,
                'gradient_clip_norm': 1.0,
                'output_scale': 1.0,
                'head_scale_learnable': False,
                'initial_head_scale': 1.0,
                'head_scale_min_multiplier': 0.1,
            }
        
        def __init__(self, state_size: int = None, action_size: int = 3, config: Dict[str, Any] = None):
            """
            Initialize Trading RL Agent with manifest-driven state size.

            Args:
                state_size (int, optional): Dimension of state space. If None, loads from manifest.
                action_size (int): Number of possible actions (default: 3).
                config (Dict[str, Any]): Configuration dictionary from YAML config.
            """
            # Merge provided config with defaults for robustness
            default_config = self._get_default_config()
            self.config = {**default_config, **(config or {})}

            # Load state size from manifest if not provided
            if state_size is None:
                from .manifest_manager import ManifestManager
                manifest_mgr = ManifestManager()
                bundle_path = self.config.get('active_bundle', 'artifacts/legacy')
                try:
                    manifest = manifest_mgr.load_manifest(bundle_path)
                    state_size = manifest.get('rl_state_size', manifest.get('feature_count', 42))
                    logger.info(f"✅ RL Agent using state_size={state_size} from manifest")
                except Exception as e:
                    logger.warning(f"Failed to load manifest for RL agent: {e}")
                    state_size = 42  # Fallback
            
            self.state_size = state_size
            self.action_size = action_size
            
            # Core Algorithm Parameters from config
            self.gamma = self.config['gamma']
            self.batch_size = self.config['batch_size']
            self.target_update_freq = self.config['target_update_freq']
            self.reward_clip_enabled = bool(self.config.get('reward_clip_enabled', False))
            self.reward_clip_min = float(self.config.get('reward_clip_min', -10.0))
            self.reward_clip_max = float(self.config.get('reward_clip_max', 10.0))
            self.reward_scale = float(self.config.get('reward_scale', 1.0))
            self.gradient_clip_norm = self.config.get('gradient_clip_norm', None)
            self.output_scale = float(self.config.get('output_scale', 1.0))
            self.head_scale_learnable = bool(self.config.get('head_scale_learnable', False))
            self.initial_head_scale = float(self.config.get('initial_head_scale', 1.0))
            self.head_scale_min_multiplier = float(
                self.config.get('head_scale_min_multiplier', 0.1)
            )
            if self.head_scale_min_multiplier <= 0:
                self.head_scale_min_multiplier = 1e-6
            
            # Behavior and Mode Parameters from config
            self.training_mode = self.config.get('training_mode', False)
            self._inference_locked = False
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
            self.regime_bias_strength = self.config.get(
                'regime_bias_strength',
                self.config.get('regime_bias', 2.0)
            )
            self.max_regime_bias = self.config.get('max_regime_bias', 3.0)
            self.min_regime_confidence_for_bias = self.config.get('min_regime_confidence_for_bias', 0.6)
            self.risk_penalty_strength = self.config.get('risk_penalty_strength', 100.0)

            self.update_counter = 0

            # Initialize Q-networks
            network_kwargs = {
                'learnable_head_scale': self.head_scale_learnable,
                'initial_head_scale': self.initial_head_scale,
                'head_scale_lower': self.head_scale_min_multiplier,
            }
            self.q_network = DQNNetwork(state_size, action_size, **network_kwargs)
            self.target_network = DQNNetwork(state_size, action_size, **network_kwargs)
            self.target_network.load_state_dict(self.q_network.state_dict())
            if (
                self.head_scale_learnable
                and hasattr(self.target_network, 'head_scale_raw')
                and isinstance(self.target_network.head_scale_raw, nn.Parameter)
            ):
                self.target_network.head_scale_raw.requires_grad_(False)
            self.target_network.eval()
            
            # Optimizer (includes learnable head scale if configured)
            self._reset_optimizer(self._collect_optimizer_parameters(), self.config['learning_rate'])
            
            # Experience replay
            self.memory = ExperienceReplay(self.config['buffer_size'])
            
            # Training metrics
            self.training_history = {
                'losses': [],
                'q_values': [],
                'rewards': []
            }
            self.head_only_mode = False
            
            logger.info(f"Initialized TradingRLAgent with state_size={state_size}, action_size={action_size}")
            logger.info(f"RL Agent Config: training_mode={self.training_mode}, hold_threshold={self.hold_confidence_threshold}, regime_bias={self.regime_bias_strength}")
        
        def set_inference_mode(self, epsilon: float = 0.0) -> None:
            """Force deterministic inference behavior (used in live trading)."""
            epsilon = max(0.0, float(epsilon))
            self.training_mode = False
            self.config['training_mode'] = False
            self.epsilon = epsilon
            self.epsilon_decay = 0.0
            self.epsilon_min = 0.0
            self._inference_locked = True
            logger.info("🔒 RL Agent locked to inference mode (epsilon=%.4f)", self.epsilon)
        
        def set_memory(self, memory):
            """Set experience replay buffer."""
            self.memory = memory
        
        def act(self, state: np.ndarray, market_regime: str = None, 
                risk_constraints: Dict = None, training: bool = False) -> int:
            """Compatibility wrapper that returns only the action."""
            action, _ = self.get_action_with_meta(
                state,
                market_regime=market_regime,
                risk_constraints=risk_constraints,
                training=training
            )
            return action

        def get_action_with_meta(self, state: np.ndarray, market_regime: Any = None,
                                  risk_constraints: Dict = None, training: bool = False) -> Tuple[int, Dict[str, Any]]:
            """Select an action and expose diagnostics for downstream logging."""
            inference_locked = getattr(self, '_inference_locked', False)
            effective_training = bool(training and not inference_locked)
            regime_label = market_regime
            if isinstance(market_regime, dict):
                regime_label = (
                    market_regime.get('predicted_regime')
                    or market_regime.get('regime')
                    or market_regime.get('label')
                    or 'neutral'
                )
            meta: Dict[str, Any] = {
                'training_mode': effective_training,
                'epsilon': self.epsilon,
                'market_regime': regime_label,
                'risk_constraints_applied': bool(risk_constraints)
            }

            if state is None:
                logger.warning("RL Agent received None state, defaulting to HOLD (1).")
                meta['reason'] = 'missing_state'
                return 1, meta

            if effective_training and random.random() < self.epsilon:
                action = random.randrange(self.action_size)
                meta['exploration'] = True
                meta['probabilities'] = None
                logger.debug(f"🤖 [RL-ACT] Exploration: Selected random action -> {ACTION_LABELS[action]}")
                return action, meta
            
            with torch.no_grad():
                self.q_network.eval()
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                raw_q_values = self._scale_q(self.q_network(state_tensor))
                
                adjusted_q_values = raw_q_values.clone()
                if risk_constraints:
                    adjusted_q_values = self._apply_risk_constraints(adjusted_q_values, risk_constraints)
                adjusted_q_values, bias_meta = self._apply_regime_bias(adjusted_q_values, market_regime)

                probabilities = torch.softmax(adjusted_q_values, dim=1).squeeze().cpu().numpy()
                best_action = int(np.argmax(probabilities))
                best_prob = float(probabilities[best_action])

                meta.update({
                    'exploration': False,
                    'probabilities': probabilities.tolist(),
                    'raw_q_values': raw_q_values.squeeze().cpu().tolist(),
                    'adjusted_q_values': adjusted_q_values.squeeze().cpu().tolist(),
                    'best_probability': best_prob,
                    'regime_confidence': bias_meta.get('regime_confidence', 0.0),
                    'bias_applied': bias_meta.get('bias_applied', False),
                    'effective_bias': bias_meta.get('effective_bias', 0.0),
                    'regime_confidence_threshold': bias_meta.get('confidence_threshold'),
                    'regime_label': bias_meta.get('regime_label'),
                })
                
                # "Uncertain HOLD" kontrolü - Sadece canlı modda (training=False) çalışmalı
                if not training and best_action == 1 and best_prob < self.hold_confidence_threshold:
                    sorted_indices = np.argsort(probabilities)[::-1]
                    second_best_action = int(sorted_indices[1])
                    meta['override'] = {
                        'reason': 'low_hold_confidence',
                        'original_probability': best_prob,
                        'threshold': self.hold_confidence_threshold
                    }
                    logger.warning(
                        f"🤖 [RL-OVERRIDE] Agent uncertain on HOLD (prob: {best_prob:.2f} < {self.hold_confidence_threshold}). "
                        f"Overriding with 2nd choice: {ACTION_LABELS[second_best_action]}"
                    )
                    best_action = second_best_action

                # Eğitim devam ediyorsa modeli tekrar train moduna al
                if effective_training:
                    self.q_network.train()

                return best_action, meta
        
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
        
        def _apply_regime_bias(
            self,
            q_values: torch.Tensor,
            market_regime: Any,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            """Apply adaptive regime bias scaled by regime confidence."""
            q_adjusted = q_values.clone()
            bias_meta: Dict[str, Any] = {
                'bias_applied': False,
                'effective_bias': 0.0,
                'regime_confidence': 0.0,
                'regime_label': 'neutral',
                'confidence_threshold': self.min_regime_confidence_for_bias,
            }

            if market_regime is None:
                return q_adjusted, bias_meta

            if isinstance(market_regime, dict):
                regime_label = (
                    market_regime.get('predicted_regime')
                    or market_regime.get('regime')
                    or market_regime.get('label')
                    or 'neutral'
                )
                confidence_raw = market_regime.get('confidence', 0.0)
                try:
                    regime_confidence = float(confidence_raw)
                except (TypeError, ValueError):
                    regime_confidence = 0.0
            else:
                regime_label = str(market_regime)
                regime_confidence = 0.0

            bias_meta['regime_label'] = regime_label
            bias_meta['regime_confidence'] = regime_confidence

            if regime_confidence < self.min_regime_confidence_for_bias:
                logger.debug(
                    "🔒 [RL-BIAS] Skipping regime bias - confidence %.3f below threshold %.3f",
                    regime_confidence,
                    self.min_regime_confidence_for_bias,
                )
                return q_adjusted, bias_meta

            effective_bias = min(self.regime_bias_strength * regime_confidence, self.max_regime_bias)
            bias_meta['bias_applied'] = True
            bias_meta['effective_bias'] = effective_bias

            if regime_label == 'bullish':
                q_adjusted[0, 0] += effective_bias      # Boost BUY
            elif regime_label == 'bearish':
                q_adjusted[0, 2] += effective_bias      # Boost SELL
            elif regime_label == 'neutral':
                q_adjusted[0, 1] += effective_bias * 0.5  # Slightly boost HOLD
            elif regime_label == 'volatile':
                q_adjusted[0, 1] -= effective_bias * 0.3  # Reduce HOLD in volatile markets
            else:
                q_adjusted[0, 1] += effective_bias * 0.25  # Conservative bias for unknown regimes

            logger.debug(
                "🎯 [RL-BIAS] Applied adaptive bias | regime=%s | confidence=%.2f | effective_bias=%.2f",
                regime_label,
                regime_confidence,
                effective_bias,
            )

            return q_adjusted, bias_meta
        
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
            rewards = self._transform_rewards(rewards)
            
            # Current Q-values
            scaled_current_q = self._scale_q(self.q_network(states))
            current_q_values = scaled_current_q.gather(1, actions.unsqueeze(1))
            
            # Target Q-values using target network (Double DQN)
            with torch.no_grad():
                # Select best actions using main network
                next_actions = self._scale_q(self.q_network(next_states)).argmax(1, keepdim=True)
                # Evaluate using target network
                next_q_values = self._scale_q(self.target_network(next_states)).gather(1, next_actions)
                target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
            
            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.gradient_clip_norm)
            self.optimizer.step()
            
            # Update target network periodically
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                logger.debug(f"Target network updated at step {self.update_counter}")
            
            # Log first successful learning event
            if not hasattr(self, '_epsilon_decay_count'):
                self._epsilon_decay_count = 0
                logger.info(f"✅ First successful learning! Network weights updated")
                logger.info(f"   Buffer: {len(self.memory.buffer)}/{self.memory.buffer.maxlen} samples")
                logger.info(f"   Loss: {loss.item():.4f}")
            
            self._epsilon_decay_count += 1
            if self._epsilon_decay_count % 50 == 0:
                logger.info(f"📊 Learning update #{self._epsilon_decay_count}: Loss = {loss.item():.4f}")
            
            # Track metrics
            td_error = target_q_values - current_q_values
            metrics = {
                'loss': loss.item(),
                'q_value': scaled_current_q.mean().item(),
                'epsilon': self.epsilon,
                'td_error': float(td_error.abs().mean().item())
            }
            
            self.training_history['losses'].append(metrics['loss'])
            self.training_history['q_values'].append(metrics['q_value'])
            self.training_history['rewards'].append(rewards.mean().item())
            self.training_history.setdefault('td_errors', []).append(metrics.get('td_error', 0.0))
            
            return metrics

        def _scale_q(self, q: torch.Tensor) -> torch.Tensor:
            scaled_q = q
            if self.output_scale != 1.0:
                scaled_q = scaled_q * self.output_scale
            return scaled_q

        def reinit_last_layer(self) -> bool:
            linear = self._find_last_linear()
            if linear is None:
                return False
            nn.init.kaiming_normal_(linear.weight, a=0, mode='fan_in')
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            return True

        def scale_last_layer(self, factor: float) -> bool:
            linear = self._find_last_linear()
            if linear is None:
                return False
            linear.weight.data.mul_(factor)
            if linear.bias is not None:
                linear.bias.data.mul_(factor)
            return True

        def get_last_layer_stats(self) -> Dict[str, float]:
            linear = self._find_last_linear()
            if linear is None:
                return {'std': 0.0, 'mean': 0.0}
            weight = linear.weight.detach().cpu()
            bias = linear.bias.detach().cpu() if linear.bias is not None else None
            stats = {
                'weight_std': float(weight.std()),
                'weight_mean': float(weight.mean()),
            }
            if bias is not None:
                stats['bias_std'] = float(bias.std())
                stats['bias_mean'] = float(bias.mean())
            return stats

        def get_head_scale_value(self) -> float:
            head_scale_attr = getattr(self.q_network, 'head_scale', None)
            if isinstance(head_scale_attr, torch.Tensor):
                return float(head_scale_attr.detach().cpu().item())
            return float(self.initial_head_scale)

        def _find_last_linear(self):
            for module in reversed(list(self.q_network.network)):
                if isinstance(module, nn.Linear):
                    return module
            return None

        def _reset_optimizer(self, parameters, lr: float, extra_params: Optional[List[nn.Parameter]] = None):
            params = list(parameters)
            if extra_params:
                params.extend(extra_params)
            if not params:
                logger.warning("No parameters provided to optimizer reset.")
                return
            self.optimizer = torch.optim.Adam(params, lr=lr)

        def _collect_optimizer_parameters(self) -> List[nn.Parameter]:
            return [p for p in self.q_network.parameters() if p.requires_grad]

        def reset_optimizer(self, lr: Optional[float] = None) -> None:
            target_lr = lr or self.config.get('learning_rate', 1e-4)
            parameters = self._collect_optimizer_parameters()
            self._reset_optimizer(parameters, target_lr)

        def enable_head_only_training(self, head_lr: float) -> bool:
            linear = self._find_last_linear()
            if linear is None:
                return False
            for param in self.q_network.parameters():
                param.requires_grad = False
            for param in linear.parameters():
                param.requires_grad = True
            if (
                self.head_scale_learnable
                and hasattr(self.q_network, 'head_scale_raw')
                and isinstance(self.q_network.head_scale_raw, nn.Parameter)
            ):
                self.q_network.head_scale_raw.requires_grad = True
            self._reset_optimizer(self._collect_optimizer_parameters(), head_lr)
            self.head_only_mode = True
            return True

        def _transform_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
            if self.reward_clip_enabled:
                rewards = rewards.clamp(self.reward_clip_min, self.reward_clip_max)
            if self.reward_scale != 1.0:
                rewards = rewards * self.reward_scale
            return rewards
        
        def decay_epsilon(self):
            """
            Decay epsilon for exploration-exploitation trade-off.
            Should be called once per episode (not per step).
            """
            if self.epsilon > self.epsilon_min:
                old_epsilon = self.epsilon
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)  # Ensure we don't go below min
                logger.debug(f"Epsilon decayed: {old_epsilon:.4f} → {self.epsilon:.4f}")
        
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
        
        def _migrate_head_scale_checkpoint(self, module_state: Dict[str, torch.Tensor]) -> None:
            if not module_state:
                return
            keys = [k for k in list(module_state.keys()) if 'head_scale_' in k]
            eps_value = 1e-6
            for key in keys:
                tensor = module_state[key]
                target_key = None
                scale_tensor = None

                if 'head_scale_log' in key:
                    target_key = key.replace('head_scale_log', 'head_scale_raw')
                    tensor_data = torch.as_tensor(tensor).detach()
                    scale_tensor = torch.exp(tensor_data)
                elif 'head_scale_alpha' in key and 'head_scale_raw' not in module_state:
                    target_key = key.replace('head_scale_alpha', 'head_scale_raw')
                    tensor_data = torch.as_tensor(tensor).detach()
                    scale_tensor = torch.clamp(1.0 + tensor_data, min=self.head_scale_min_multiplier)

                if target_key is None or scale_tensor is None:
                    continue

                module_state.pop(key, None)
                scale_tensor = scale_tensor.to(dtype=torch.float32)
                min_offset = scale_tensor.new_tensor(self.head_scale_min_multiplier)
                min_delta = scale_tensor.new_tensor(eps_value)
                delta = torch.clamp(scale_tensor - min_offset, min=min_delta)
                raw_tensor = torch.log(torch.expm1(delta))
                module_state[target_key] = raw_tensor

        def load_model(self, path: str):
            """
            Load model weights from a checkpoint file with dimension compatibility checking.
            Includes error handling for missing files and dimension mismatches.
            """
            try:
                checkpoint = torch.load(path, map_location='cpu')
                
                # Check dimension compatibility before loading
                if 'q_network' in checkpoint:
                    # Try to extract first layer dimensions
                    try:
                        first_layer_key = next(k for k in checkpoint['q_network'].keys() if 'network.0.weight' in k)
                        weight_shape = checkpoint['q_network'][first_layer_key].shape
                        model_input_size = weight_shape[1]
                        
                        if model_input_size != self.state_size:
                            logger.warning(
                                f"⚠️ RL model dimension mismatch: model expects {model_input_size}, "
                                f"but agent has state_size={self.state_size}"
                            )
                            logger.warning("Skipping model load - dimension mismatch. Agent will use untrained weights.")
                            return
                    except (StopIteration, KeyError):
                        # Can't determine dimensions, proceed with caution
                        logger.warning("Cannot determine model dimensions, attempting load anyway...")
                
                # Ensure compatibility with legacy head scale checkpoints
                self._migrate_head_scale_checkpoint(checkpoint.get('q_network', {}))
                self._migrate_head_scale_checkpoint(checkpoint.get('target_network', {}))

                # Load model
                missing, unexpected = self.q_network.load_state_dict(checkpoint['q_network'], strict=False)
                if missing:
                    logger.warning("Missing keys while loading q_network: %s", missing)
                if unexpected:
                    logger.warning("Unexpected keys while loading q_network: %s", unexpected)

                tgt_missing, tgt_unexpected = self.target_network.load_state_dict(checkpoint['target_network'], strict=False)
                if tgt_missing:
                    logger.warning("Missing keys while loading target_network: %s", tgt_missing)
                if tgt_unexpected:
                    logger.warning("Unexpected keys while loading target_network: %s", tgt_unexpected)

                optimizer_state = checkpoint.get('optimizer')
                if optimizer_state:
                    try:
                        self.optimizer.load_state_dict(optimizer_state)
                    except ValueError as opt_err:
                        logger.warning(
                            "Optimizer state incompatible with current parameters; resetting optimizer. Details: %s",
                            opt_err,
                        )
                        self.reset_optimizer()
                else:
                    logger.warning("Checkpoint missing optimizer state; using freshly initialized optimizer.")
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.training_history = checkpoint.get('training_history', self.training_history)
                logger.info(f"✅ RL Agent model loaded successfully from {path} (state_size={self.state_size})")
            except FileNotFoundError:
                logger.error(f"❌ RL Agent model file not found at {path}. Agent will use untrained weights.")
            except Exception as e:
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
        
        def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = None, **kwargs):
            self.state_size = state_size
            self.action_size = action_size
            logger.info("Initialized mock DQNNetwork (PyTorch not available)")
    
    
    class TradingRLAgent:
        """Mock Trading RL Agent (PyTorch not available)."""
        
        def __init__(self, state_size: int, action_size: int, **kwargs):
            self.state_size = state_size
            self.action_size = action_size
            self.epsilon = kwargs.get('epsilon_start', 1.0)
            self.training_mode = kwargs.get('training_mode', False)
            self._inference_locked = False
            self.memory = None
            self.training_history = {'losses': [], 'q_values': [], 'rewards': []}
            logger.info("Initialized mock TradingRLAgent (PyTorch not available)")
        
        def set_inference_mode(self, epsilon: float = 0.0) -> None:
            self.training_mode = False
            self.epsilon = max(0.0, float(epsilon))
            self._inference_locked = True
            logger.info("🔒 Mock RL Agent locked to inference mode (epsilon=%.4f)", self.epsilon)

        def set_memory(self, memory):
            """Set experience replay buffer."""
            self.memory = memory
        
        def act(self, state: np.ndarray, market_regime: str = None, 
                risk_constraints: Dict = None, training: bool = True) -> int:
            """Mock action selection - returns random action."""
            action, _ = self.get_action_with_meta(state, market_regime, risk_constraints, training)
            return action

        def get_action_with_meta(self, state: np.ndarray, market_regime: str = None,
                                  risk_constraints: Dict = None, training: bool = True) -> Tuple[int, Dict[str, Any]]:
            inference_locked = getattr(self, '_inference_locked', False)
            effective_training = bool(training and not inference_locked)
            meta = {
                'training_mode': effective_training,
                'market_regime': market_regime,
                'epsilon': self.epsilon,
                'mock_mode': True
            }
            if state is None:
                meta['reason'] = 'missing_state'
                return 1, meta
            action = random.randrange(self.action_size)
            meta['exploration'] = effective_training and self.epsilon > 0
            return action, meta
        
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

        def reset_optimizer(self, lr: Optional[float] = None) -> None:
            logger.info("Mock reset optimizer (lr=%s)", lr)
