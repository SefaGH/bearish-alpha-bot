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
    # === GÜNCELLEME: __init__ metodu artık iki DataFrame alıyor: features_df ve raw_df ===
    def __init__(self, features_df: pd.DataFrame, raw_df: pd.DataFrame, initial_balance: float = 10000.0, fee: float = 0.0006):
        """
        Initializes the trading environment.

        Args:
            features_df (pd.DataFrame): DataFrame containing ONLY the engineered features for the agent's state.
            raw_df (pd.DataFrame): DataFrame containing the raw OHLCV data, must include 'close'.
            initial_balance (float): The starting balance for each episode.
            fee (float): The trading fee per transaction.
        """
        if features_df.empty or raw_df.empty:
            raise ValueError("DataFrames for trading environment cannot be empty.")
        
        # İki DataFrame'in de aynı sayıda satıra sahip olduğundan emin ol
        if len(features_df) != len(raw_df):
            raise ValueError(f"Features DataFrame (len: {len(features_df)}) and Raw DataFrame (len: {len(raw_df)}) must have the same length.")

        self.features_df = features_df
        self.raw_df = raw_df # Ham veriyi sakla
        self.initial_balance = initial_balance
        self.fee = fee
        
        # State dimensions: ONLY features. Portfolio state is removed.
        self.state_dim = len(features_df.columns) 
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
        self.total_pnl = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Constructs the state array for the current step.

        The state now consists ONLY of the market features.
        """
        market_state = self.features_df.iloc[self._current_step].values
        return market_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take (0: Hold, 1: Buy, 2: Sell).

        Returns:
            A tuple containing (next_state, reward, done, info).
        """
        if self.done:
            raise ValueError("step() called after episode is done.")

        self._current_step += 1
        # === GÜNCELLEME: Bitiş kontrolü özelliklerin sayısına göre yapılır ===
        if self._current_step >= len(self.features_df) - 1:
            self.done = True

        # === GÜNCELLEME: Fiyat, ham veriyi içeren raw_df'den alınır ===
        current_price = self.raw_df['close'].iloc[self._current_step]
        
        # --- Execute Action ---
        if action == 1:  # Buy
            if self.balance > 0:
                amount_to_buy = (self.balance / current_price) * (1 - self.fee)
                self.position += amount_to_buy
                self.balance = 0.0
        elif action == 2:  # Sell
            if self.position > 0:
                self.balance += self.position * current_price * (1 - self.fee)
                self.position = 0.0

        # --- Calculate Reward ---
        new_portfolio_value = self.balance + (self.position * current_price)
        previous_portfolio_value = self.initial_balance + self.total_pnl
        
        # Ödül, portföy değerindeki oransal değişimdir
        # Sıfıra bölünmeyi önlemek için küçük bir epsilon değeri eklenir
        if previous_portfolio_value != 0:
            reward = (new_portfolio_value - previous_portfolio_value) / abs(previous_portfolio_value)
        else:
            reward = 0.0
        
        self.total_pnl = new_portfolio_value - self.initial_balance
        
        # Pozisyon tutmak için küçük bir ceza
        if self.position > 0:
            reward -= 0.0001
            
        # Büyük kayıp için ağır ceza
        if new_portfolio_value < self.initial_balance * 0.5:
            self.done = True
            reward -= 1.0

        next_state = self._get_state()
        info = {'step': self._current_step, 'pnl': self.total_pnl}

        return next_state, reward, self.done, info
