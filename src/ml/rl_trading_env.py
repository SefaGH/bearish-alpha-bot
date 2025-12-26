import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from src.ml.ppo.observation_spec import (
    DEFAULT_EXTRA_FEATURE_NAMES,
    ObservationSpec,
    build_observation,
    compute_price_extras,
    spec_from_feature_columns,
)
from src.ml.ppo.deterministic_scaler import DeterministicScaler

logger = logging.getLogger(__name__)

# Aksiyon Etiketleri: Hedef Pozisyon Oranları
ACTION_LABELS = ['TARGET_0.0', 'TARGET_1.0']


class RLTradingEnv:
    """
    Gelişmiş RL Ticaret Ortamı (Benchmark Karşılaştırmalı Reward ile).

    - Aksiyonlar: 0 → FLAT (0.0), 1 → FULL LONG (1.0)
    - Reward: Bot portföyü log-getirisi - Buy&Hold benchmark log-getirisi
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        config: Optional[Dict] = None,
        initial_balance: float = 10000.0,
        idle_cost: float = 0.0,
        observation_spec: Optional[ObservationSpec] = None,
    ):
        if features_df.empty or raw_df.empty:
            raise ValueError("DataFrames cannot be empty.")

        if len(features_df) != len(raw_df):
            raise ValueError(
                f"Len mismatch: Features ({len(features_df)}) vs Raw ({len(raw_df)})"
            )

        self.features_df = features_df
        self.raw_df = raw_df
        self.initial_balance = initial_balance
        if observation_spec:
            self.observation_spec = observation_spec
        else:
            extra_names = DEFAULT_EXTRA_FEATURE_NAMES if len(features_df.columns) == 82 else []
            self.observation_spec = spec_from_feature_columns(features_df.columns, extra_feature_names=extra_names)
        self._scaler = DeterministicScaler(spec=self.observation_spec, log_every=0)

        config = config or {}
        self.fee = config.get("fee_pct", 0.0006)

        # State: features + [position_fraction, normalized_pv]
        self.state_dim = self.observation_spec.obs_dim
        self.action_dim = 2  # 0: flat, 1: full long

        self.position_fraction: float = 0.0

        # Reward clip/scale (relaxed range for hybrid reward)
        self.reward_clip_enabled = config.get("reward_clip_enabled", True)
        self.reward_clip_min = config.get("reward_clip_min", -5.0)
        self.reward_clip_max = config.get("reward_clip_max", 5.0)
        self.reward_scale = config.get("reward_scale", 1.0)

        # Trade penalty / idle cost = 0 (benchmark-based reward ile gereksiz)
        self.trade_penalty_alpha = 0.0
        cfg_idle = config.get("idle_cost", None)
        self.idle_cost = float(cfg_idle) if cfg_idle is not None else float(idle_cost)

        logger.info(
            "✅ RLTradingEnv Initialized (Benchmark-aware). "
            "state_dim=%d, action_dim=%d, idle_cost=%s",
            self.state_dim,
            self.action_dim,
            self.idle_cost,
        )
        self.reset()

    def reset(self) -> np.ndarray:
        """Environment'ı sıfırlar."""
        self._current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_pnl = 0.0
        self.done = False
        self.position_fraction = 0.0

        # --- Benchmark: Buy & Hold BTC ---
        # Başlangıçta tüm sermaye ile BTC alıp episode boyunca hiç satmayan portföy.
        first_price = float(self.raw_df["close"].iloc[0])
        # Fee'yi de hesaba katarak alınabilecek BTC adedi
        self.bench_position = (self.initial_balance / (first_price * (1.0 + self.fee)))
        self.bench_pv = self.bench_position * first_price
        self.bench_prev_pv = self.bench_pv

        return self._get_state()

    def _get_portfolio_value(self, price: float) -> float:
        return self.balance + (self.position * price)

    def _get_state(self) -> np.ndarray:
        feature_row = self.features_df.iloc[self._current_step]
        current_price = float(self.raw_df["close"].iloc[self._current_step])
        portfolio_value = self._get_portfolio_value(current_price)

        normalized_pv = (
            portfolio_value / self.initial_balance
            if self.initial_balance > 0
            else 0.0
        )

        tail = {
            "position_fraction": self.position_fraction,
            "normalized_pv": normalized_pv,
        }
        extra_values = {}
        if getattr(self.observation_spec, "extra_feature_names", None):
            extra_arr = compute_price_extras(self.raw_df.iloc[: self._current_step + 1])
            extra_values = {
                name: float(extra_arr[i]) for i, name in enumerate(self.observation_spec.extra_feature_names)
            }
        row_dict = feature_row.to_dict()
        row_dict.update(extra_values)
        row_dict.update(tail)
        close_price = float(current_price)
        if self._scaler:
            return self._scaler.transform(row_dict, close_price)
        return build_observation(
            self.observation_spec,
            feature_row,
            tail_values=tail,
            extra_values=extra_values,
        )

    def _calculate_reward_legacy(self, bot_log_ret: float, bench_log_ret: float) -> float:
        """Previous reward: purely benchmark-relative."""
        return bot_log_ret - bench_log_ret

    def _calculate_reward(self, action: int, bot_log_ret: float, bench_log_ret: float) -> float:
        """
        Hybrid reward:
        - Absolute profit dominates (70%)
        - Benchmark-relative term keeps buy&hold as a reference (30%)
        - Idle penalty discourages staying flat forever
        """
        idle_penalty = -0.00005 if int(action) == 0 else 0.0
        absolute_profit = 0.7 * bot_log_ret
        benchmark_relative = 0.3 * (bot_log_ret - bench_log_ret)
        return absolute_profit + benchmark_relative + idle_penalty

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise ValueError("step() called after episode is done.")

        # 1) Hedef pozisyon oranı (0: 0.0, 1: 1.0)
        TARGETS = {0: 0.0, 1: 1.0}
        target_fraction = TARGETS.get(int(action), 0.0)

        # 2) Zamanı ilerlet, fiyatı al
        self._current_step += 1
        if self._current_step >= len(self.features_df) - 1:
            self.done = True

        current_price = float(self.raw_df["close"].iloc[self._current_step])

        # 3) İşlem öncesi PV'ler
        prev_bot_pv = self._get_portfolio_value(current_price)
        prev_bench_pv = self.bench_pv

        # 4) Bot portföyü için trade uygula
        delta_fraction = target_fraction - self.position_fraction
        trade_fraction = abs(delta_fraction)

        if trade_fraction > 1e-6 and prev_bot_pv > 0:
            if delta_fraction > 0:  # ALIM
                notional_to_buy = prev_bot_pv * delta_fraction
                notional_to_buy = min(notional_to_buy, self.balance)
                if notional_to_buy > 0:
                    amount_to_buy = (notional_to_buy / (1.0 + self.fee)) / current_price
                    self.position += amount_to_buy
                    self.balance -= notional_to_buy
            elif delta_fraction < 0:  # SATIŞ
                notional_to_sell = prev_bot_pv * trade_fraction
                max_sell_notional = self.position * current_price
                notional_to_sell = min(notional_to_sell, max_sell_notional)
                if notional_to_sell > 0:
                    amount_to_sell = notional_to_sell / current_price
                    revenue = notional_to_sell * (1.0 - self.fee)
                    self.position -= amount_to_sell
                    self.balance += revenue

        self.position_fraction = target_fraction

        # 5) Bot & benchmark portföylerini güncelle
        new_bot_pv = self._get_portfolio_value(current_price)

        # Benchmark PV (buy & hold)
        self.bench_pv = self.bench_position * current_price
        new_bench_pv = self.bench_pv

        # 6) Reward components
        if prev_bot_pv > 0 and new_bot_pv > 0:
            bot_log_ret = float(np.log(new_bot_pv / prev_bot_pv))
        else:
            bot_log_ret = 0.0

        if prev_bench_pv > 0 and new_bench_pv > 0:
            bench_log_ret = float(np.log(new_bench_pv / prev_bench_pv))
        else:
            bench_log_ret = 0.0

        self.total_pnl = new_bot_pv - self.initial_balance
        reward = self._calculate_reward(action, bot_log_ret, bench_log_ret)

        # Stop-out: bot PV çok düşerse epizodu bitir
        if new_bot_pv < self.initial_balance * 0.5:
            self.done = True
            reward = -1.0

        if self.reward_clip_enabled:
            reward = max(self.reward_clip_min, min(self.reward_clip_max, reward))
        reward *= self.reward_scale

        # Benchmark previous PV'yi güncelle
        self.bench_prev_pv = new_bench_pv

        next_state = self._get_state()

        info = {
            "step": self._current_step,
            "pnl": self.total_pnl,
            "portfolio_value": new_bot_pv,
            "benchmark_value": new_bench_pv,
            "position_fraction": self.position_fraction,
            "reward": float(reward),
            "action": int(action),
        }

        return next_state, float(reward), self.done, info
