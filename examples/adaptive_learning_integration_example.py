#!/usr/bin/env python3
"""
Integration Example: Phase 4.2 with Existing Trading Bot

Demonstrates how to integrate the adaptive learning system with:
- Phase 4.1: ML Regime Prediction
- Phase 3.2: Risk Management
- Phase 2: Market Intelligence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
from typing import Dict

from src.ml.reinforcement_learning import TradingRLAgent
from src.ml.experience_replay import ExperienceReplay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingBotWithAdaptiveLearning:
    """
    Trading bot enhanced with adaptive learning capabilities.
    
    Integrates:
    - RL Agent for strategy optimization
    - Risk management constraints
    - Market regime awareness
    """
    
    def __init__(self, config: Dict):
        """Initialize bot with adaptive learning."""
        # Initialize RL agent
        self.rl_agent = TradingRLAgent(
            state_size=config.get('state_size', 50),
            action_size=config.get('action_size', 5),
            learning_rate=config.get('learning_rate', 0.001),
            epsilon=config.get('epsilon', 1.0)
        )
        
        # Initialize experience replay
        self.memory = ExperienceReplay(
            max_size=config.get('replay_buffer_size', 100000),
            priority_alpha=config.get('priority_alpha', 0.6)
        )
        self.rl_agent.set_memory(self.memory)
        
        # Configuration
        self.config = config
        self.training_mode = config.get('training_mode', True)
        
        # State tracking
        self.last_state = None
        self.last_action = None
        
        logger.info(f"Initialized TradingBotWithAdaptiveLearning")
        logger.info(f"  State size: {config.get('state_size', 50)}")
        logger.info(f"  Action space: {config.get('action_size', 5)}")
        logger.info(f"  Training mode: {self.training_mode}")
    
    def extract_market_state(self, price_data: pd.DataFrame, 
                            portfolio_state: Dict,
                            regime_prediction: Dict = None) -> np.ndarray:
        """
        Extract state features from market data.
        
        Args:
            price_data: OHLCV data with indicators
            portfolio_state: Current portfolio state
            regime_prediction: Optional regime prediction from Phase 4.1
            
        Returns:
            State vector
        """
        features = []
        
        # Price-based features
        if len(price_data) > 0:
            latest = price_data.iloc[-1]
            
            # Normalized price change
            if 'close' in price_data.columns and len(price_data) > 1:
                price_change = (latest['close'] - price_data.iloc[-2]['close']) / price_data.iloc[-2]['close']
                features.append(price_change)
            else:
                features.append(0.0)
            
            # Technical indicators (normalized)
            for indicator in ['rsi', 'macd', 'volume']:
                if indicator in latest:
                    if indicator == 'rsi':
                        # RSI is already 0-100, normalize to [-1, 1]
                        features.append((latest[indicator] - 50) / 50)
                    elif indicator == 'volume':
                        # Log-scale volume change
                        if len(price_data) > 1:
                            vol_change = np.log(latest[indicator] / (price_data.iloc[-2][indicator] + 1e-6))
                            features.append(np.clip(vol_change, -1, 1))
                        else:
                            features.append(0.0)
                    else:
                        # Normalize by std
                        features.append(latest[indicator] / (price_data[indicator].std() + 1e-6))
                else:
                    features.append(0.0)
        else:
            features.extend([0.0] * 4)
        
        # Portfolio state features
        features.append(portfolio_state.get('position_count', 0) / 10.0)  # Normalize by max positions
        features.append(portfolio_state.get('drawdown', 0.0))
        features.append(portfolio_state.get('total_risk', 0.0))
        
        # Regime prediction features (if available)
        if regime_prediction and 'probabilities' in regime_prediction:
            probs = regime_prediction['probabilities']
            features.append(probs.get('bullish', 0.33))
            features.append(probs.get('neutral', 0.33))
            features.append(probs.get('bearish', 0.33))
        else:
            features.extend([0.33, 0.33, 0.33])  # Neutral assumption
        
        # Pad to state_size
        target_size = self.config.get('state_size', 50)
        while len(features) < target_size:
            features.append(0.0)
        
        return np.array(features[:target_size])
    
    def get_risk_constraints(self, portfolio_state: Dict) -> Dict:
        """
        Extract risk constraints from portfolio state.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Risk constraints for RL agent
        """
        max_positions = self.config.get('max_positions', 10)
        max_drawdown = self.config.get('max_drawdown', 0.15)
        max_portfolio_heat = self.config.get('max_portfolio_heat', 0.10)
        
        return {
            'max_position_reached': portfolio_state.get('position_count', 0) >= max_positions,
            'max_drawdown_reached': portfolio_state.get('drawdown', 0.0) >= max_drawdown,
            'portfolio_heat_high': portfolio_state.get('total_risk', 0.0) >= max_portfolio_heat
        }
    
    def select_action(self, price_data: pd.DataFrame,
                     portfolio_state: Dict,
                     regime_prediction: Dict = None) -> Dict:
        """
        Select trading action using RL agent.
        
        Args:
            price_data: OHLCV data
            portfolio_state: Portfolio state
            regime_prediction: Optional regime prediction
            
        Returns:
            Action recommendation
        """
        # Extract state
        state = self.extract_market_state(price_data, portfolio_state, regime_prediction)
        
        # Get risk constraints
        risk_constraints = self.get_risk_constraints(portfolio_state)
        
        # Get current regime
        current_regime = None
        if regime_prediction and 'predicted_regime' in regime_prediction:
            current_regime = regime_prediction['predicted_regime']
        
        # Select action
        action = self.rl_agent.act(
            state=state,
            market_regime=current_regime,
            risk_constraints=risk_constraints,
            training=self.training_mode
        )
        
        # Store for learning
        self.last_state = state
        self.last_action = action
        
        # Map action to trading decision
        action_map = {
            0: {'type': 'buy', 'strength': 'strong', 'size_multiplier': 1.0},
            1: {'type': 'buy', 'strength': 'weak', 'size_multiplier': 0.5},
            2: {'type': 'hold', 'strength': 'neutral', 'size_multiplier': 0.0},
            3: {'type': 'sell', 'strength': 'weak', 'size_multiplier': 0.5},
            4: {'type': 'sell', 'strength': 'strong', 'size_multiplier': 1.0}
        }
        
        return {
            'action_idx': action,
            'decision': action_map.get(action, action_map[2]),
            'state': state,
            'regime': current_regime,
            'risk_constraints': risk_constraints
        }
    
    def learn_from_trade(self, trade_result: Dict,
                        price_data: pd.DataFrame,
                        portfolio_state: Dict,
                        regime_prediction: Dict = None):
        """
        Learn from trade execution result.
        
        Args:
            trade_result: Trade execution result with PnL
            price_data: Current OHLCV data
            portfolio_state: Current portfolio state
            regime_prediction: Current regime prediction
        """
        if self.last_state is None or self.last_action is None:
            return
        
        # Calculate reward from trade result
        reward = self._calculate_reward(trade_result)
        
        # Get next state
        next_state = self.extract_market_state(price_data, portfolio_state, regime_prediction)
        
        # Episode done?
        done = trade_result.get('done', False)
        
        # Learn
        metrics = self.rl_agent.learn_from_experience(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        if metrics.get('loss', 0) > 0:
            logger.debug(f"Learning: loss={metrics['loss']:.6f}, "
                        f"Q-value={metrics['q_value']:.4f}, "
                        f"epsilon={metrics['epsilon']:.4f}")
        
        # Update state
        self.last_state = next_state
    
    def _calculate_reward(self, trade_result: Dict) -> float:
        """Calculate reward from trade result."""
        # Base reward on PnL
        pnl = trade_result.get('pnl', 0.0)
        
        # Normalize PnL to reasonable range
        reward = np.tanh(pnl * 10.0)  # Tanh keeps in [-1, 1]
        
        # Add penalties/bonuses
        if trade_result.get('violated_risk', False):
            reward -= 0.5  # Penalty for risk violations
        
        if trade_result.get('good_execution', False):
            reward += 0.1  # Bonus for good execution
        
        return reward
    
    def get_training_summary(self) -> Dict:
        """Get training metrics."""
        return self.rl_agent.get_training_summary()
    
    def save_model(self, path: str):
        """Save trained model."""
        self.rl_agent.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        self.rl_agent.load_model(path)
        logger.info(f"Model loaded from {path}")


def main():
    """Demonstration of integrated trading bot."""
    print("\n" + "="*60)
    print("Adaptive Learning Integration Example")
    print("="*60 + "\n")
    
    # Initialize bot
    config = {
        'state_size': 50,
        'action_size': 5,
        'learning_rate': 0.001,
        'epsilon': 1.0,
        'training_mode': True,
        'max_positions': 10,
        'max_drawdown': 0.15,
        'max_portfolio_heat': 0.10
    }
    
    bot = TradingBotWithAdaptiveLearning(config)
    
    # Simulate trading with learning
    logger.info("\nSimulating trading episodes with adaptive learning...\n")
    
    for episode in range(5):
        # Mock data
        price_data = pd.DataFrame({
            'close': [100 + i + np.random.randn() for i in range(100)],
            'rsi': [50 + np.random.randn() * 10 for _ in range(100)],
            'macd': [np.random.randn() * 0.5 for _ in range(100)],
            'volume': [1000 + np.random.randn() * 100 for _ in range(100)]
        })
        
        portfolio_state = {
            'position_count': np.random.randint(0, 5),
            'drawdown': np.random.rand() * 0.05,
            'total_risk': np.random.rand() * 0.05
        }
        
        regime_prediction = {
            'predicted_regime': np.random.choice(['bullish', 'neutral', 'bearish']),
            'probabilities': {
                'bullish': np.random.rand(),
                'neutral': np.random.rand(),
                'bearish': np.random.rand()
            }
        }
        
        # Select action
        action_result = bot.select_action(price_data, portfolio_state, regime_prediction)
        
        logger.info(f"Episode {episode + 1}:")
        logger.info(f"  Regime: {action_result['regime']}")
        logger.info(f"  Decision: {action_result['decision']['type']} "
                   f"({action_result['decision']['strength']})")
        
        # Simulate trade result
        trade_result = {
            'pnl': np.random.randn() * 0.01,
            'done': False,
            'violated_risk': False,
            'good_execution': np.random.rand() > 0.5
        }
        
        # Learn from trade
        bot.learn_from_trade(trade_result, price_data, portfolio_state, regime_prediction)
    
    # Get summary
    logger.info("\nTraining Summary:")
    summary = bot.get_training_summary()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ Integration example completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
