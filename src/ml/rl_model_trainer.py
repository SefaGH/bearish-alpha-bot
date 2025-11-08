"""
Reinforcement Learning (RL) Model Trainer

This module contains the trainer for the RL agent. It orchestrates the
training process by running episodes in the RLTradingEnv, collecting
experiences, and triggering the agent's learning steps.

Author: SefaGH
Date: 2025-11-02
"""

import numpy as np
import pandas as pd
import torch
import os
import logging
from collections import deque
from typing import Optional

from .rl_trading_env import RLTradingEnv
from .reinforcement_learning import TradingRLAgent, ExperienceReplay

logger = logging.getLogger(__name__)

class RLModelTrainer:
    """
    Orchestrates the training of the TradingRLAgent.
    """
    def __init__(self,
                 agent: TradingRLAgent,
                 env: RLTradingEnv,
                 experience_replay: ExperienceReplay,
                 model_save_path: str = 'data/models',
                 model_name: str = 'rl_agent.pth'):
        """
        Initializes the RL Model Trainer.
        """
        self.agent = agent
        self.env = env
        self.experience_replay = experience_replay
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.training_history = []  # Store episode metrics
        
        # Agent'ın hafızasını (experience_replay) ayarla
        self.agent.set_memory(self.experience_replay)
        
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
            logger.info(f"Created directory for saving models: {self.model_save_path}")
        
        # =====================================================================
        # ENHANCED CONFIGURATION LOGGING
        # =====================================================================
        
        logger.info("="*70)
        logger.info("🤖 RL AGENT TRAINING CONFIGURATION")
        logger.info("="*70)
        
        # Training Mode
        training_mode_icon = "🎓" if self.agent.training_mode else "🔒"
        logger.info(f"{training_mode_icon} Training Mode:     {self.agent.training_mode}")
        
        if not self.agent.training_mode:
            logger.warning("⚠️  WARNING: Agent is in INFERENCE mode!")
            logger.warning("⚠️  Agent will NOT learn during this training session")
            logger.warning("⚠️  This is likely a configuration error")
        
        # Exploration Strategy
        logger.info(f"🎯 Exploration Strategy (Epsilon-Greedy):")
        logger.info(f"   Initial Epsilon:   {self.agent.epsilon:.4f}")
        
        # Log epsilon parameters if available
        if hasattr(self.agent, 'epsilon_start'):
            logger.info(f"   Epsilon Start:     {self.agent.epsilon_start:.4f}")
        if hasattr(self.agent, 'epsilon_decay'):
            logger.info(f"   Epsilon Decay:     {self.agent.epsilon_decay:.4f}")
        if hasattr(self.agent, 'epsilon_min'):
            logger.info(f"   Epsilon Min:       {self.agent.epsilon_min:.4f}")
        
        # Learning Parameters
        logger.info(f"📚 Learning Parameters:")
        logger.info(f"   Learning Rate:     {self.agent.learning_rate}")
        logger.info(f"   Gamma (discount):  {self.agent.gamma}")
        logger.info(f"   Batch Size:        {self.agent.batch_size}")
        
        # Memory Configuration
        logger.info(f"💾 Experience Replay:")
        logger.info(f"   Buffer Capacity:   {self.experience_replay.capacity}")
        logger.info(f"   Current Size:      {len(self.experience_replay.memory)}")
        
        logger.info("="*70)


    def train(self,
              num_episodes: int,
              batch_size: int = 64, # Bu parametre artık doğrudan ajan tarafından kullanılıyor
              save_every: int = 10,
              checkpoint_path: Optional[str] = None):
        """
        Runs the main training loop for the specified number of episodes.
        """
        # === GÜNCELLEME: Doğru metod adı 'load_model' ===
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.agent.load_model(checkpoint_path)
            logger.info(f"Resumed training from checkpoint: {checkpoint_path}")

        scores = deque(maxlen=100)
        latest_metrics = {}

        for e in range(1, num_episodes + 1):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # === GÜNCELLEME: 'training=True' parametresi eklendi ===
                action = self.agent.act(state, training=True)
                
                next_state, reward, done, info = self.env.step(action)
                
                # === GÜNCELLEME: Deneyim ekleme ve öğrenme mantığı düzeltildi ===
                # Ajanın kendi içindeki öğrenme metodu her adımda çağrılır.
                # Bu metod hem deneyimi ekler hem de yeterli veri birikince öğrenir.
                latest_metrics = self.agent.learn_from_experience(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward

            scores.append(total_reward)
            avg_score = np.mean(scores)

            # === GÜNCELLEME: Loglamaya 'Loss' eklendi ===
            logger.info(
                f"Episode {e}/{num_episodes} | "
                f"Total Reward: {total_reward:.4f} | "
                f"Avg Reward (last 100): {avg_score:.4f} | "
                f"PnL: {info.get('pnl', 0):.2f} | "
                f"Epsilon: {self.agent.epsilon:.4f} | "
                f"Loss: {latest_metrics.get('loss', 0):.4f}"
            )
            
            # Store episode metrics
            self.training_history.append({
                'episode': e,
                'total_reward': total_reward,
                'avg_reward': avg_score,
                'pnl': info.get('pnl', 0),
                'epsilon': self.agent.epsilon,
                'loss': latest_metrics.get('loss', 0)
            })

            # === GÜNCELLEME: Doğru metod adı 'save_model' ===
            if e % save_every == 0:
                full_path = os.path.join(self.model_save_path, self.model_name)
                self.agent.save_model(full_path)
                logger.info(f"💾 Model checkpoint saved to {full_path}")

        logger.info("✅ RL model training completed.")
        # === GÜNCELLEME: Doğru metod adı 'save_model' ===
        full_path = os.path.join(self.model_save_path, 'rl_agent_final.pth') # Final modelini farklı kaydet
        self.agent.save_model(full_path)
        logger.info(f"💾 Final RL model saved to {full_path}")
        
        # Save training metrics
        self._save_training_metrics()
    
    def _save_training_metrics(self):
        """Save RL training history to CSV file."""
        if not self.training_history:
            logger.info("No RL training history to save.")
            return
        
        try:
            # Create logs directory if it doesn't exist
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            # Validate and normalize training history entries
            normalized_history = []
            for entry in self.training_history:
                normalized_entry = {
                    'episode': entry.get('episode', 0),
                    'total_reward': entry.get('total_reward', 0.0),
                    'avg_reward': entry.get('avg_reward', 0.0),
                    'pnl': entry.get('pnl', 0.0),
                    'epsilon': entry.get('epsilon', 0.0),
                    'loss': entry.get('loss', 0.0)
                }
                normalized_history.append(normalized_entry)
            
            # Save as CSV
            df = pd.DataFrame(normalized_history)
            csv_path = os.path.join(log_dir, 'rl_training_metrics.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Saved RL training metrics: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save RL training metrics: {e}", exc_info=True)
