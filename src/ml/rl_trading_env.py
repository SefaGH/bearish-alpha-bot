"""
Reinforcement Learning Trading Environment

This module defines the trading environment for the RL agent, following a
gym-like interface (step, reset). It simulates trading on historical data,
calculates rewards, and provides the state for the agent.

Author: SefaGH
Date: 2025-11-02
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class RLTradingEnv:
    """
    A trading environment for a Reinforcement Learning agent.
    """
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, fee: float = 0.0006):
        """
        Initializes the trading environment.

        Args:
            df (pd.DataFrame): DataFrame containing historical OHLCV and feature data.
            initial_balance (float): The starting balance for each episode.
            fee (float): The trading fee per transaction.
        """
        if df.empty:
            raise ValueError("DataFrame for trading environment cannot be empty.")
            
        self.df = df
        self.initial_balance = initial_balance
        self.fee = fee
        
        # State dimensions: price data features + portfolio state (1 for balance, 1 for position)
        self.state_dim = len(df.columns) + 2 
        self.action_dim = 3  # 0: Hold, 1: Buy, 2: Sell

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Resets the environment to the initial state for a new episode.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        self._current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Amount of asset held
        self.position_value = 0.0
        self.total_pnl = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Constructs the state array for the current step.

        The state includes market data and portfolio status.
        """
        market_state = self.df.iloc[self._current_step].values
        
        # Portfolio state: normalized balance and position
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.position 
        ])
        
        return np.concatenate([market_state, portfolio_state])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take (0: Hold, 1: Buy, 2: Sell).

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: 
            A tuple containing (next_state, reward, done, info).
        """
        if self.done:
            raise ValueError("step() called after episode is done.")

        self._current_step += 1
        if self._current_step >= len(self.df) - 1:
            self.done = True

        current_price = self.df['close'].iloc[self._current_step]
        
        # --- Execute Action ---
        if action == 1:  # Buy
            if self.balance > 0:
                # Buy with full available balance
                amount_to_buy = (self.balance / current_price) * (1 - self.fee)
                self.position += amount_to_buy
                self.balance = 0.0
        elif action == 2:  # Sell
            if self.position > 0:
                # Sell all holdings
                self.balance += self.position * current_price * (1 - self.fee)
                self.position = 0.0

        # --- Calculate Reward ---
        # Reward is the change in total portfolio value from the previous step
        new_portfolio_value = self.balance + (self.position * current_price)
        previous_portfolio_value = self.initial_balance + self.total_pnl
        
        reward = (new_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # Update total PnL
        self.total_pnl = new_portfolio_value - self.initial_balance
        
        # Penalty for holding a position to encourage closing trades
        if self.position > 0:
            reward -= 0.0001
            
        # Check for catastrophic loss
        if new_portfolio_value < self.initial_balance * 0.5:
            self.done = True
            reward -= 1.0 # Heavy penalty for losing 50% of capital

        next_state = self._get_state()
        info = {'step': self._current_step, 'pnl': self.total_pnl}

        return next_state, reward, self.done, info
