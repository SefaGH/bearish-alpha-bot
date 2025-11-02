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

        Args:
            agent (TradingRLAgent): The RL agent to be trained.
            env (RLTradingEnv): The trading environment for simulation.
            experience_replay (ExperienceReplay): The buffer to store and sample experiences.
            model_save_path (str): Directory to save the trained model.
            model_name (str): Filename for the saved model.
        """
        self.agent = agent
        self.env = env
        self.experience_replay = experience_replay
        self.model_save_path = model_save_path
        self.model_name = model_name
        
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
            logger.info(f"Created directory for saving models: {self.model_save_path}")

    def train(self,
              num_episodes: int,
              batch_size: int = 64,
              save_every: int = 10,
              checkpoint_path: Optional[str] = None):
        """
        Runs the main training loop for the specified number of episodes.

        Args:
            num_episodes (int): The total number of episodes to run for training.
            batch_size (int): The number of experiences to sample from the replay buffer for each learning step.
            save_every (int): Frequency (in episodes) to save the model checkpoint.
            checkpoint_path (Optional[str]): Path to a model checkpoint to continue training from.
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)
            logger.info(f"Resumed training from checkpoint: {checkpoint_path}")

        scores = deque(maxlen=100) # Stores the total rewards for the last 100 episodes

        for e in range(1, num_episodes + 1):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Agent chooses an action
                action = self.agent.act(state)
                
                # Environment executes the action
                next_state, reward, done, info = self.env.step(action)
                
                # Store the experience in the replay buffer
                self.experience_replay.add(state, action, reward, next_state, done)
                
                # Agent learns from a batch of experiences
                if len(self.experience_replay) > batch_size:
                    self.agent.learn(self.experience_replay.sample(batch_size))
                
                state = next_state
                total_reward += reward

            scores.append(total_reward)
            avg_score = np.mean(scores)

            logger.info(
                f"Episode {e}/{num_episodes} | "
                f"Total Reward: {total_reward:.4f} | "
                f"Avg Reward (last 100): {avg_score:.4f} | "
                f"PnL: {info.get('pnl', 0):.2f} | "
                f"Epsilon: {self.agent.epsilon:.4f}"
            )

            # Save the model periodically
            if e % save_every == 0:
                full_path = os.path.join(self.model_save_path, self.model_name)
                self.agent.save(full_path)
                logger.info(f"💾 Model checkpoint saved to {full_path}")

        logger.info("✅ RL model training completed.")
        # Save the final model
        full_path = os.path.join(self.model_save_path, self.model_name)
        self.agent.save(full_path)
        logger.info(f"💾 Final RL model saved to {full_path}")
