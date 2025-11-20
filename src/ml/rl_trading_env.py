import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Aksiyon Etiketleri: Hedef Pozisyon Oranları
# Eski: ['TARGET_0.0', 'TARGET_0.5', 'TARGET_1.0']
# Yeni: Sadece FLAT (0.0) ve FULL LONG (1.0)
ACTION_LABELS = ['TARGET_0.0', 'TARGET_1.0']


class RLTradingEnv:
    """
    Gelişmiş RL Ticaret Ortamı (Final Version - State Augmented).

    Özellikler:
    1. Target Position Logic: Aksiyonlar hedef pozisyon oranını belirler (0.0, 1.0).
    2. Reward Shaping: Minimal – temel olarak portföy log-getirisi kullanılır.
    3. State Augmentation: State vektörü, piyasa verilerine ek olarak portföy durumunu (pozisyon oranı + normalize PV) içerir.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        config: Optional[Dict] = None,
        initial_balance: float = 10000.0,
        idle_cost: float = 0.0,
    ):
        """
        Environment başlatıcı.
        """
        if features_df.empty or raw_df.empty:
            raise ValueError("DataFrames cannot be empty.")

        if len(features_df) != len(raw_df):
            raise ValueError(
                f"Len mismatch: Features ({len(features_df)}) vs Raw ({len(raw_df)})"
            )

        self.features_df = features_df
        self.raw_df = raw_df
        self.initial_balance = initial_balance

        # Config ve Parametreler
        config = config or {}
        self.fee = config.get("fee_pct", 0.0006)

        # --- STATE DIMENSION ---
        # Feature Sayısı + 2 Ekstra Bilgi (Pozisyon Oranı + Normalize Portföy Değeri)
        self.state_dim = len(features_df.columns) + 2

        # Action space: 2 discrete aksiyon (0: flat, 1: full long)
        self.action_dim = 2

        # Durum Değişkenleri
        self.position_fraction: float = 0.0

        # Reward Parametreleri
        self.reward_clip_enabled = config.get("reward_clip_enabled", True)
        self.reward_clip_min = config.get("reward_clip_min", -1.0)
        self.reward_clip_max = config.get("reward_clip_max", 1.0)
        self.reward_scale = config.get("reward_scale", 1.0)

        # Trade penalty ve idle cost'u şimdilik kapatıyoruz (FinRL benzeri sade reward için)
        self.trade_penalty_alpha = 0.0

        cfg_idle = config.get("idle_cost", None)
        if cfg_idle is not None:
            self.idle_cost = float(cfg_idle)
        else:
            self.idle_cost = float(idle_cost)

        logger.info(
            "✅ RLTradingEnv Initialized. Mode: Target Position (0.0/1.0) + Augmented State. "
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
        return self._get_state()

    def _get_portfolio_value(self, price: float) -> float:
        """Yardımcı Fonksiyon: Güncel portföy değerini hesaplar (Nakit + Varlık)."""
        return self.balance + (self.position * price)

    def _get_state(self) -> np.ndarray:
        """
        State vektörünü oluşturur.
        İçerik: [Market Features (N)] + [Position Fraction (1)] + [Normalized PV (1)]
        """
        market_state = self.features_df.iloc[self._current_step].values.astype(
            np.float32
        )

        current_price = float(self.raw_df["close"].iloc[self._current_step])
        portfolio_value = self._get_portfolio_value(current_price)

        normalized_pv = (
            portfolio_value / self.initial_balance
            if self.initial_balance > 0
            else 0.0
        )

        portfolio_state = np.array(
            [self.position_fraction, normalized_pv], dtype=np.float32
        )

        return np.concatenate([market_state, portfolio_state])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Bir zaman adımı ilerletir."""
        if self.done:
            raise ValueError("step() called after episode is done.")

        # 1. Hedef Pozisyonu Belirle (Discrete 0/1 → 0.0 / 1.0)
        TARGETS = {0: 0.0, 1: 1.0}
        target_fraction = TARGETS.get(int(action), 0.0)

        # 2. Adım İlerlet ve Fiyat Al
        self._current_step += 1
        if self._current_step >= len(self.features_df) - 1:
            self.done = True

        current_price = float(self.raw_df["close"].iloc[self._current_step])

        # 3. İşlem Öncesi Değerler
        portfolio_value_before = self._get_portfolio_value(current_price)

        # Ne kadar değişim gerekiyor?
        delta_fraction = target_fraction - self.position_fraction
        trade_fraction = abs(delta_fraction)

        # 4. İşlemi Uygula (delta fraction üzerinden al/sat)
        if trade_fraction > 1e-6 and portfolio_value_before > 0:
            if delta_fraction > 0:  # ALIM: pozisyonu artır
                notional_to_buy = portfolio_value_before * delta_fraction
                notional_to_buy = min(
                    notional_to_buy, self.balance
                )  # Bakiye kontrolü

                if notional_to_buy > 0:
                    # Fee'yi fiyat tarafında uygula
                    amount_to_buy = (notional_to_buy / (1.0 + self.fee)) / current_price
                    self.position += amount_to_buy
                    self.balance -= notional_to_buy

            elif delta_fraction < 0:  # SATIŞ: pozisyonu azalt
                notional_to_sell = portfolio_value_before * trade_fraction
                max_sell_notional = self.position * current_price
                notional_to_sell = min(notional_to_sell, max_sell_notional)

                if notional_to_sell > 0:
                    amount_to_sell = notional_to_sell / current_price
                    revenue = notional_to_sell * (1.0 - self.fee)

                    self.position -= amount_to_sell
                    self.balance += revenue

        # Güncel hedef pozisyon oranını yaz
        self.position_fraction = target_fraction

        # 5. Reward Hesapla (log-return)
        new_portfolio_value = self._get_portfolio_value(current_price)
        previous_portfolio_value = self.initial_balance + self.total_pnl

        if previous_portfolio_value > 0 and new_portfolio_value > 0:
            base_reward = float(
                np.log(new_portfolio_value / previous_portfolio_value)
            )
        else:
            base_reward = 0.0

        self.total_pnl = new_portfolio_value - self.initial_balance
        reward = base_reward

        # Batış Cezası (Stop-out)
        if new_portfolio_value < self.initial_balance * 0.5:
            self.done = True
            reward = -1.0

        # Clip & Scale
        if self.reward_clip_enabled:
            reward = max(self.reward_clip_min, min(self.reward_clip_max, reward))
        reward *= self.reward_scale

        next_state = self._get_state()

        info = {
            "step": self._current_step,
            "pnl": self.total_pnl,
            "portfolio_value": new_portfolio_value,
            "position_fraction": self.position_fraction,
            "reward": reward,
            "action": int(action),
        }

        return next_state, float(reward), self.done, info
