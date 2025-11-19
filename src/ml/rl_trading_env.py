import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Aksiyon Etiketleri: Hedef Pozisyon Oranları
ACTION_LABELS = ['TARGET_0.0', 'TARGET_0.5', 'TARGET_1.0']

class RLTradingEnv:
    """
    Gelişmiş RL Ticaret Ortamı (Final Version - State Augmented).
    
    Özellikler:
    1. Target Position Logic: Aksiyonlar hedef pozisyon oranını belirler (0, 0.5, 1.0).
    2. Reward Shaping: İşlem cezası (penalty) ve kırpma (clipping) uygulanır.
    3. State Augmentation: State vektörü, piyasa verilerine ek olarak portföy durumunu (pozisyon oranı + bakiye durumu) içerir.
    """
    
    def __init__(self, features_df: pd.DataFrame, raw_df: pd.DataFrame, config: Optional[Dict] = None, initial_balance: float = 10000.0):
        """
        Environment başlatıcı.
        """
        if features_df.empty or raw_df.empty:
            raise ValueError("DataFrames cannot be empty.")
        
        if len(features_df) != len(raw_df):
            raise ValueError(f"Len mismatch: Features ({len(features_df)}) vs Raw ({len(raw_df)})")

        self.features_df = features_df
        self.raw_df = raw_df
        self.initial_balance = initial_balance
        
        # Config ve Parametreler
        config = config or {}
        self.fee = config.get('fee_pct', 0.0006)
        
        # --- STATE DIMENSION UPDATE ---
        # Feature Sayısı + 2 Ekstra Bilgi (Pozisyon Oranı + Normalize Portföy Değeri)
        # Bu, modelin giriş boyutunu artırır.
        self.state_dim = len(features_df.columns) + 2
        self.action_dim = 3 

        # Durum Değişkenleri
        self.position_fraction: float = 0.0
        
        # Reward Parametreleri
        self.reward_clip_enabled = config.get('reward_clip_enabled', True)
        self.reward_clip_min = config.get('reward_clip_min', -1.0)
        self.reward_clip_max = config.get('reward_clip_max', 1.0)
        self.reward_scale = config.get('reward_scale', 1.0)
        self.trade_penalty_alpha = config.get('trade_penalty_alpha', 0.001)

        self.idle_cost = float(idle_cost)
        
        logger.info(f"✅ RLTradingEnv Initialized. Mode: Target Position + Augmented State. Dim: {self.state_dim}")
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
        # 1. Market Features
        market_state = self.features_df.iloc[self._current_step].values.astype(np.float32)
        
        # 2. Portfolio Features (Ajanın "Neredeyim?" sorusunun cevabı)
        current_price = float(self.raw_df['close'].iloc[self._current_step])
        portfolio_value = self._get_portfolio_value(current_price)
        
        # Normalize edilmiş portföy değeri (Başlangıca göre oran, örn: 1.05 = %5 kâr)
        normalized_pv = portfolio_value / self.initial_balance if self.initial_balance > 0 else 0.0
        
        portfolio_state = np.array([self.position_fraction, normalized_pv], dtype=np.float32)
        
        # 3. Vektörleri Birleştir
        return np.concatenate([market_state, portfolio_state])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Bir zaman adımı ilerletir."""
        if self.done:
            raise ValueError("step() called after episode is done.")

        # 1. Hedef Pozisyonu Belirle
        TARGETS = {0: 0.0, 1: 0.5, 2: 1.0}
        target_fraction = TARGETS.get(int(action), 0.0)

        # 2. Adım İlerlet ve Fiyat Al
        self._current_step += 1
        if self._current_step >= len(self.features_df) - 1:
            self.done = True

        current_price = float(self.raw_df['close'].iloc[self._current_step])

        # 3. İşlem Öncesi Değerler
        portfolio_value_before = self._get_portfolio_value(current_price)
        
        # Ne kadar değişim gerekiyor?
        delta_fraction = target_fraction - self.position_fraction
        trade_fraction = abs(delta_fraction)
        
        # 4. İşlemi Uygula (Simülasyon)
        # Sadece anlamlı bir değişim varsa işlem yap (floating point hatasını önlemek için > 1e-6)
        if trade_fraction > 1e-6 and portfolio_value_before > 0:
            if delta_fraction > 0: # ALIM (Mevcut orandan daha yükseğe çık)
                notional_to_buy = portfolio_value_before * delta_fraction
                notional_to_buy = min(notional_to_buy, self.balance) # Bakiye kontrolü (fee için kabaca)
                
                # Fee düşüldükten sonra net alım
                if notional_to_buy > 0:
                    cost_with_fee = notional_to_buy # Toplam harcanacak para
                    amount_to_buy = (notional_to_buy / (1 + self.fee)) / current_price
                    
                    self.position += amount_to_buy
                    self.balance -= cost_with_fee
                    
            elif delta_fraction < 0: # SATIŞ (Mevcut orandan daha düşüğe in)
                notional_to_sell = portfolio_value_before * abs(delta_fraction)
                max_sell_notional = self.position * current_price
                notional_to_sell = min(notional_to_sell, max_sell_notional) # Pozisyon kontrolü
                
                if notional_to_sell > 0:
                    amount_to_sell = notional_to_sell / current_price
                    revenue = notional_to_sell * (1 - self.fee) # Fee düşülmüş gelir
                    
                    self.position -= amount_to_sell
                    self.balance += revenue

        # Durumu güncelle
        self.position_fraction = target_fraction

        # 5. Reward Hesapla
        new_portfolio_value = self._get_portfolio_value(current_price)
        previous_portfolio_value = self.initial_balance + self.total_pnl
        
        # a. Base Reward: Yüzdesel portföy değişimi
        if previous_portfolio_value != 0:
            base_reward = (new_portfolio_value - previous_portfolio_value) / abs(previous_portfolio_value)
        else:
            base_reward = 0.0
            
        self.total_pnl = new_portfolio_value - self.initial_balance

        # b. Penalty & Shaping
        # Sık işlem yapmayı cezalandır (Churn Penalty)
        trade_penalty = self.trade_penalty_alpha * trade_fraction
        reward = base_reward - trade_penalty

        # HOLD / idle cezası:
        if trade_fraction < 1e-6 and self.idle_cost > 0.0:
            reward -= self.idle_cost

        # c. Batış Cezası (Stop-out)
        if new_portfolio_value < self.initial_balance * 0.5:
            self.done = True
            reward = -1.0 

        # d. Clip & Scale
        if self.reward_clip_enabled:
            reward = max(self.reward_clip_min, min(self.reward_clip_max, reward))
        reward *= self.reward_scale

        next_state = self._get_state()
        
        info = {
            'step': self._current_step,
            'pnl': self.total_pnl,
            'portfolio_value': new_portfolio_value,
            'position_fraction': self.position_fraction,
            'reward': reward,
            'action': int(action)
        }

        return next_state, reward, self.done, info
