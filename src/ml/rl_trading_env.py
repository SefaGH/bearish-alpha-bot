"""
Reinforcement Learning Trading Environment

This module defines the trading environment for the RL agent, following a
gym-like interface (step, reset). It simulates trading on historical data,
calculates rewards, and provides the state for the agent.

Action Mapping Convention:
    0 -> HOLD: No trade action
    1 -> BUY:  Open or increase long position  
    2 -> SELL: Close position or open short

Author: SefaGH
Date: 2025-11-02
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Canonical action mapping - must match reinforcement_learning.py
ACTION_LABELS = ['HOLD', 'BUY', 'SELL']

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

        # === YENİ: Pozisyon oranı ve reward stabilizasyon parametreleri ===
        # position_fraction: 0.0 → flat, 1.0 → full long
        self.position_fraction: float = 0.0

        # Reward clip & scale + trade penalty
        # Şimdilik sabit; istenirse ml.reinforcement_learning bloğundan da okunabilir
        self.reward_clip_enabled: bool = True
        self.reward_clip_min: float = -1.0
        self.reward_clip_max: float = 1.0
        self.reward_scale: float = 1.0

        # Trade penalty: her adımda portföyün trade_fraction kadarını
        # döndürmek reward'dan trade_penalty_alpha * trade_fraction kadar düşer.
        self.trade_penalty_alpha: float = 0.001
        
        logger.info(f"✅ RLTradingEnv initialized with action mapping: {dict(enumerate(ACTION_LABELS))}")

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

        # === YENİ: pozisyon oranını sıfırla ===
        self.position_fraction = 0.0

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

        # --- Portföy değeri (aksiyondan önce) ---
        portfolio_value_before = self.balance + (self.position * current_price)

        # --- Aksiyon → hedef pozisyon oranı (kademeli pozisyon) ---
        # 0 -> 0.0 (flat)
        # 1 -> 0.5 (yarım long)
        # 2 -> 1.0 (full long)
        TARGETS = {
            0: 0.0,
            1: 0.5,
            2: 1.0,
        }
        target_fraction = TARGETS.get(int(action), 0.0)

        delta_fraction = target_fraction - self.position_fraction
        trade_fraction = abs(delta_fraction)  # 0.0–1.0 arası

        # --- Execute Action (kısmi trade) ---
        # BUY: pozisyon artır (delta_fraction > 0)
        if delta_fraction > 0 and portfolio_value_before > 0:
            # Hedefe göre portföyün delta_fraction kadarı kadar notional al
            notional_to_buy = portfolio_value_before * delta_fraction
            # Elimizdeki balance bundan küçükse clamp et
            notional_to_buy = min(notional_to_buy, self.balance)

            if notional_to_buy > 0:
                amount_to_buy = (notional_to_buy / current_price) * (1 - self.fee)
                self.position += amount_to_buy
                # Fee'yi notional içinden karşıladığımızı varsayıyoruz
                self.balance -= notional_to_buy

        # SELL: pozisyon azalt (delta_fraction < 0)
        elif delta_fraction < 0 and portfolio_value_before > 0:
            notional_to_sell = portfolio_value_before * (-delta_fraction)
            # Elde satılabilir en fazla notional:
            max_notional_we_can_sell = self.position * current_price
            notional_to_sell = min(notional_to_sell, max_notional_we_can_sell)

            if notional_to_sell > 0:
                amount_to_sell = notional_to_sell / current_price
                self.balance += notional_to_sell * (1 - self.fee)
                self.position -= amount_to_sell

        # HOLD (delta_fraction == 0) ise hiçbir şey yapma

        # Hedef pozisyon oranına ulaştığını varsay
        self.position_fraction = target_fraction

        # --- Calculate Reward ---
        new_portfolio_value = self.balance + (self.position * current_price)
        previous_portfolio_value = self.initial_balance + self.total_pnl
        
        # Ödül, portföy değerindeki oransal değişimdir
        if previous_portfolio_value != 0:
            base_reward = (new_portfolio_value - previous_portfolio_value) / abs(previous_portfolio_value)
        else:
            base_reward = 0.0
        
        self.total_pnl = new_portfolio_value - self.initial_balance

        # Eski "pozisyon tutma cezası"nı kaldırıyoruz; onun yerine trade penalty kullanacağız.
        # if self.position > 0:
        #     base_reward -= 0.0001

        # --- Trade penalty: ne kadar büyük trade yaptıysan o kadar küçük ceza ---
        trade_penalty = self.trade_penalty_alpha * trade_fraction
        reward = base_reward - trade_penalty

        # Büyük kayıp için ağır ceza (stop-out mantığı)
        if new_portfolio_value < self.initial_balance * 0.5:
            self.done = True
            reward -= 1.0

        # --- Reward clip & scale ---
        if self.reward_clip_enabled:
            reward = max(self.reward_clip_min, min(self.reward_clip_max, reward))

        reward *= self.reward_scale

        next_state = self._get_state()
        info = {
            'step': self._current_step,
            'pnl': self.total_pnl,
            'portfolio_value_before': portfolio_value_before,
            'portfolio_value_after': new_portfolio_value,
            'base_reward': base_reward,
            'trade_fraction': trade_fraction,
            'trade_penalty': trade_penalty,
            'position_fraction': self.position_fraction,
            'action_label': ACTION_LABELS[int(action)] if 0 <= int(action) < len(ACTION_LABELS) else 'UNKNOWN',
        }

        return next_state, reward, self.done, info
