import logging
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces

from .rl_trading_env import RLTradingEnv, ACTION_LABELS

logger = logging.getLogger(__name__)


class RLTradingEnvGym(gym.Env):
    """
    Gym-uyumlu RL Trading ortamı.

    - İçeride mevcut RLTradingEnv'i kullanır (state, reward, trade mantığı aynı).
    - Sadece Gym'in beklediği API'yi (spaces, reset, step) sağlar.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        features_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        initial_balance: float = 10000.0,
    ):
        super().__init__()

        if features_df.empty or raw_df.empty:
            raise ValueError("features_df/raw_df boş olamaz.")
        if len(features_df) != len(raw_df):
            raise ValueError(
                f"Len mismatch: Features ({len(features_df)}) vs Raw ({len(raw_df)})"
            )

        self._base_env = RLTradingEnv(
            features_df=features_df,
            raw_df=raw_df,
            config=config or {},
            initial_balance=initial_balance,
            idle_cost=(config or {}).get("idle_cost", 0.0),
        )

        self.state_dim = self._base_env.state_dim
        self.n_actions = self._base_env.action_dim  # şu an 2 (0: flat, 1: full long)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._last_info: Dict[str, Any] = {}

        logger.info(
            "✅ RLTradingEnvGym Initialized | state_dim=%d, action_dim=%d",
            self.state_dim,
            self.n_actions,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Gym reset: (obs, info) döner."""
        super().reset(seed=seed)
        obs = self._base_env.reset()
        self._last_info = {
            "portfolio_value": self._base_env._get_portfolio_value(
                float(self._base_env.raw_df["close"].iloc[self._base_env._current_step])
            ),
            "pnl": self._base_env.total_pnl,
        }
        return obs.astype(np.float32), dict(self._last_info)

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Gym step:
        - action: int (0 veya 1)
        - dönen: (obs, reward, terminated, truncated, info)
        """
        next_state, reward, done, info = self._base_env.step(int(action))

        terminated = bool(done)
        truncated = False  # Şimdilik ayrıca max-steps truncation yok

        self._last_info = dict(info)

        obs = next_state.astype(np.float32)
        reward = float(reward)

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Basit console render."""
        if mode != "human":
            return
        info = self._last_info or {}
        step = info.get("step", getattr(self._base_env, "_current_step", 0))
        pv = info.get("portfolio_value", 0.0)
        pos_frac = info.get("position_fraction", 0.0)
        pnl = info.get("pnl", 0.0)
        print(f"t={step} | PV={pv:.2f} | PosFrac={pos_frac:.2f} | PnL={pnl:.2f}")
