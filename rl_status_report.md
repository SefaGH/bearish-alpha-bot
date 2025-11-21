# RL Trading Ajanı – Durum Raporu

_Tarih_: 2025-11-20  
_Sorumlu_: `SefaGH/bearish-alpha-bot` projesi – RL bileşeni

---

## 1. Genel Amaç

Bu dokümanın amacı, projedeki **RL (Reinforcement Learning) trading ajanı** ile ilgili:

- Şu ana kadar neler yapıldığını,
- Hangi aşamada olduğumuzu,
- Ne yapmaya çalıştığımızı ve nasıl yaptığımızı

özetleyen kalıcı bir **durum raporu** sunmaktır.

Ana hedef:

- BTC/USDT 1h verisi üzerinde çalışan,
- Piyasa koşullarına göre **flat vs full long** pozisyon kararları verebilen,
- Klasik stratejilere (özellikle **buy-and-hold BTC**) karşı **daha iyi risk/getiri profili** sunmayı hedefleyen,
- Uzun vadede canlı trade pipeline’ına entegre edilebilir bir **PPO tabanlı RL ajanı** geliştirmek.

Bu ajan, mevcut ML ve strateji katmanlarının (ör. GEMMA, strateji sinyalleri, risk motoru) üzerinde, **nihai onay/veto** mekanizması olarak konumlanacak.

---

## 2. Veri Hazırlama ve Dataset Pipeline’ı

### 2.1. Dataset oluşturma script’i

**Dosya**:  
```text
scripts/prepare_rl_training_data.py
```

Bu script şu adımları gerçekleştirir:

1. **OHLCV verisini yükler**
   - Borsadan:  
     `src/core/ccxt_client.py` üzerinden:
     ```python
     from src.core.ccxt_client import CcxtClient
     ```
     ve içinde:
     ```python
     client.fetch_ohlcv_bulk(args.symbol, args.timeframe, target_limit=args.candles)
     ```
   - veya dosyadan:
     ```bash
     --input-file <csv/parquet path>
     ```

2. **Feature engineering** uygular  
   **82 feature**’lık GEMMA/ML ile uyumlu feature matrisi üretir:

   ```python
   from src.ml.feature_engineering import FeatureEngineeringPipeline

   feature_engine = FeatureEngineeringPipeline(config=config)
   features = feature_engine.extract_features(price_df.copy(), mode="price")
   ```

3. **Zaman bazlı train/val/test split** oluşturur  

   - Argümanlar:
     ```bash
     --train-ratio 0.7
     --val-ratio 0.15
     --min-split 512
     ```
   - Split hesaplama:
     ```python
     @dataclass(frozen=True)
     class DatasetSplits:
         train: slice
         val: slice
         test: slice
     ```

4. **Datasetleri .npz olarak kaydeder**  

   Örnek çıktı dosyaları:

   - `data/training/BTC_USDT_USDT_1h_train.npz`
   - `data/training/BTC_USDT_USDT_1h_val.npz`
   - `data/training/BTC_USDT_USDT_1h_test.npz`
   - Metadata:
     ```text
     data/training/BTC_USDT_USDT_1h_metadata.json
     ```

Metadata içeriğinde; sembol, timeframe, exchange, toplam satır sayısı ve her split’in başlangıç/bitiş tarihleri tutulur:

```json
{
  "symbol": "BTC/USDT:USDT",
  "timeframe": "1h",
  "exchange": "bingx",
  "rows": 13730,
  "splits": {
    "train": {
      "rows": 9611,
      "start": "2024-04-27 10:00:00",
      "end": "2025-06-01 20:00:00"
    },
    "test": {
      "rows": 2060,
      "start": "2025-08-26 16:00:00",
      "end": "2025-11-20 11:00:00"
    }
  }
}
```

### 2.2. Tarih kontrollü veri filtresi

Script’e aşağıdaki parametreler eklendi:

```bash
--start-date YYYY-MM-DD
--end-date YYYY-MM-DD
```

İlgili fonksiyon:

```python
def _apply_date_filter(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if start_date:
        start_ts = pd.to_datetime(start_date)
        df = df.loc[df.index >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        df = df.loc[df.index <= end_ts]
    return df
```

Bu sayede:

- Dataset’in tarih aralığı, borsanın en son X mumu ile sınırlı kalmadan,
- Gerekirse spesifik dönemlere (örn. sadece 2022 ayı, ya da 2019–2024 arası) kısıtlanabiliyor.

Şu an kullanılan dataset (örnek run):

- Toplam satır: 13 730,
- Train: 2024-04-27 → 2025-06-01,
- Test: 2025-08-26 → 2025-11-20,

yani zaten **boğa + ayı karışık rejimleri** içeren bir pencere.

---

## 3. RL Environment Tasarımı

Ana env dosyaları:

```text
src/ml/rl_trading_env.py
src/ml/rl_trading_env_gym.py
```

### 3.1. State tasarımı – `RLTradingEnv`

**Dosya**:  
```text
src/ml/rl_trading_env.py
```

State:

- `features_df`: Geniş feature matrisi (ör. 82 feature),
- Buna ek olarak 2 portföy feature’ı:
  - `position_fraction` (0.0 veya 1.0),
  - `normalized_pv` (portfolio_value / initial_balance).

Kod:

```python
self.state_dim = len(features_df.columns) + 2
```

```python
def _get_state(self) -> np.ndarray:
    market_state = self.features_df.iloc[self._current_step].values.astype(np.float32)

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
```

Bu, FinRL env’lerindeki şu şemaya benziyor:

- Cash,
- Fiyatlar,
- Pozisyon,
- Indikatörler.

Biz ise fiyat/indikatorlar zaten feature engineering içinde olduğu için ek olarak sadece “pozisyon oranı” ve “normalize PV” kullanıyoruz.

### 3.2. Action space redesign (3 → 2 aksiyon)

İlk versiyon:

```python
ACTION_LABELS = ['TARGET_0.0', 'TARGET_0.5', 'TARGET_1.0']
self.action_dim = 3
```

- PPO ajanı burada sık sık:
  - Ya hep `TARGET_0.5`,
  - Ya da hep `TARGET_0.0` seçip, hiç trade yapmıyordu.

Bunun sebebi:

- 0.5 orta pozisyon “konforlu” bir kaçış noktasıydı,
- 0.0 (cash) ise özellikle ayı trendlerinde çok cazipti.

Şu an:

```python
ACTION_LABELS = ['TARGET_0.0', 'TARGET_1.0']
self.action_dim = 2  # 0: flat, 1: full long
```

`step()` içinde aksiyon:

```python
TARGETS = {0: 0.0, 1: 1.0}
target_fraction = TARGETS.get(int(action), 0.0)

delta_fraction = target_fraction - self.position_fraction
trade_fraction = abs(delta_fraction)
```

- `delta_fraction > 0` → alım,
- `delta_fraction < 0` → satış,
- İşlem büyüklüğü, mevcut portföy değerine göre hesaplanıyor,
- Fee (`self.fee`) fiyat/amount üzerinden yansıtılıyor.

Bu değişiklik sonrası:

- Ajan **iki aksiyonu da** kullanmaya başladı (`unique_actions: [0, 1]`),
- `num_trades` > 0 oldu (örneğin 18 trade içeren run’lar),
- “Hep flat / hep 0.5” saplanmaları kalktı.

### 3.3. Reward tasarımının evrimi

#### 3.3.1. İlk reward – yüzde PV değişimi + penalty’ler

Başlangıçta:

```python
base_reward = (new_pv - prev_pv) / abs(prev_pv)
trade_penalty = trade_penalty_alpha * trade_fraction
reward = base_reward - trade_penalty
if trade_fraction < 1e-6 and idle_cost > 0:
    reward -= idle_cost
```

Bu yapı:

- Ajanı, özellikle ayı rejimlerinde:
  - “Hep flat kal, hiç risk alma” politikasına yönlendiriyordu.
- Sonuç:  
  `num_trades = 0`, `unique_actions` genellikle tek değer.

#### 3.3.2. Log-return bazlı reward (FinRL standardına yakın)

Sonra FinRL env’lerine benzer şekilde:

```python
if previous_portfolio_value > 0 and new_portfolio_value > 0:
    base_reward = np.log(new_portfolio_value / previous_portfolio_value)
else:
    base_reward = 0.0
reward = base_reward
```

- `trade_penalty_alpha = 0.0`,
- `idle_cost = 0.0`.

Bu değişiklikten sonra:

- Agent hareketlenmeye başladı:
  - `unique_actions: [0, 1]`,
  - `num_trades > 0`.
- Ancak:
  - `total_return` hâlâ negatif (örn. ~−13%),
  - `max_drawdown` yüksek (örn. ~−29%).

Yani ajan artık **piyasaya girip çıkıyor**, ama ayı trendlerinde pozisyonu azaltmayı/yok etmeyi yeterince öğrenmemişti.

#### 3.3.3. Benchmark karşılaştırmalı reward (buy-and-hold BTC’ye göre)

FinRL-Crypto’daki `CryptoEnvAlpaca` env’inden şu fikir alındı:

```python
reward = (delta_bot - delta_eqw) * norm_reward
```

Yani:

- Reward, bot portföyü ile **benchmark (equal-weight)** portföy arasındaki fazladan getiriyi ölçüyor.

Bunu tek asset (BTC) için şu şekilde uyarladık:

- Benchmark = **buy-and-hold BTC**:
  - Episode başında tüm sermaye ile BTC alıp,
  - Hiç satmayan, hep long kalan hayali portföy.

`reset()` içinde:

```python
first_price = float(self.raw_df["close"].iloc[0])
self.bench_position = self.initial_balance / (first_price * (1.0 + self.fee))
self.bench_pv = self.bench_position * first_price
self.bench_prev_pv = self.bench_pv
```

`step()` içinde:

```python
prev_bot_pv = self._get_portfolio_value(current_price)
prev_bench_pv = self.bench_pv

# Bot için trade sonrası PV:
new_bot_pv = self._get_portfolio_value(current_price)

# Benchmark (buy & hold) PV güncellemesi:
self.bench_pv = self.bench_position * current_price
new_bench_pv = self.bench_pv

if prev_bot_pv > 0 and new_bot_pv > 0:
    bot_log_ret = np.log(new_bot_pv / prev_bot_pv)
else:
    bot_log_ret = 0.0

if prev_bench_pv > 0 and new_bench_pv > 0:
    bench_log_ret = np.log(new_bench_pv / prev_bench_pv)
else:
    bench_log_ret = 0.0

base_reward = bot_log_ret - bench_log_ret
reward = base_reward
```

Bu sayede:

- Reward > 0 ise:
  - Ajan, aynı adımda **HODL BTC’ye göre daha iyi** getiri sağlamış demektir.
- Reward < 0 ise:
  - Buy-and-hold BTC’yi **daha kötü** performe etmiştir.

Stop-out ve clipping:

```python
if new_bot_pv < self.initial_balance * 0.5:
    self.done = True
    reward = -1.0

if self.reward_clip_enabled:
    reward = max(self.reward_clip_min, min(self.reward_clip_max, reward))
reward *= self.reward_scale
```

`info` sözlüğünde artık benchmark PV de yer alıyor:

```python
info = {
    "step": self._current_step,
    "pnl": self.total_pnl,
    "portfolio_value": new_bot_pv,
    "benchmark_value": new_bench_pv,
    "position_fraction": self.position_fraction,
    "reward": float(reward),
    "action": int(action),
}
```

---

## 4. Gym Wrapper ve PPO Training

**Dosya**:  
```text
src/ml/rl_trading_env_gym.py
```

Amaç:  
`RLTradingEnv`’i Stable Baselines3 PPO’nun beklediği Gym API’sine sarmak.

Önemli noktalar:

```python
self._base_env = RLTradingEnv(...)
self.state_dim = self._base_env.state_dim
self.n_actions = self._base_env.action_dim  # şu an 2

self.observation_space = spaces.Box(
    low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
)
self.action_space = spaces.Discrete(self.n_actions)
```

`step` imzası:

```python
def step(self, action: int):
    next_state, reward, done, info = self._base_env.step(int(action))
    terminated = bool(done)
    truncated = False
    return next_state.astype(np.float32), float(reward), terminated, truncated, info
```

Train/eval script’leri (özet):

- Eğitim:

  ```bash
  python scripts/train_ppo_agent.py \
    --dataset data/training/BTC_USDT_USDT_1h_train.npz \
    --model-dir data/checkpoints \
    --timesteps 300000
  ```

- Değerlendirme:

  ```bash
  python scripts/evaluate_ppo_agent.py \
    --model data/checkpoints/ppo_trading_agent.zip \
    --dataset data/training/BTC_USDT_USDT_1h_test.npz
  ```

Çıktı dosyaları:

- `data/training/ppo_eval_summary.json`
- `data/training/ppo_eval_equity_curve.csv`

Önceki bir run (benchmark öncesi, sadece log-return reward ile):

```json
{
  "steps": 2059,
  "initial_balance": 10000.0,
  "final_portfolio_value": 8657.6180,
  "total_pnl": -1342.38,
  "total_return": -0.1342,
  "max_drawdown": -0.2947,
  "num_trades": 18,
  "position_changes": 18,
  "unique_actions": [0, 1]
}
```

Yorum:

- Artık ajan gerçekten:
  - Hem flat hem long pozisyona gidip geliyor,
  - 18 trade ile trend yakalamaya çalışıyor.
- Ancak:
  - Boğa dönemlerinde kazanç var ama,
  - Ayı rejimlerine geçişte pozisyonu azaltmakta geç kalıyor,
  - Sonuç: yüksek DD, negatif toplam getiri.

Bu noktada benchmark-aware reward’a geçtik; şu anda training bu yeni reward ile devam ediyor.

---

## 5. Şu Anda Hangi Aşamadayız?

Özet:

1. **Dataset**:
   - BTC/USDT 1h, ~13k bar, 2024-04–2025-11 arası,
   - Train/val/test zaman bazlı düzgün split,
   - Tarih filtresi isteğe bağlı.

2. **Env**:
   - State: features + pozisyon oranı + normalize PV,
   - Action: Discrete 2 → flat vs full long,
   - Reward:
     - Ajanın getirisini, **buy-and-hold BTC benchmark’ına göre** ölçen log-return farkı.

3. **Ajan davranışı**:
   - Tek aksiyona saplanan/sıfır trade atan ajan dönemini geçtik,
   - Artık:
     - Trade yapıyor,
     - Trendleri kovalıyor,
     - Risk yönetimi tarafında iyileştirmeye açık.

4. **Hedef**:
   - HODL BTC’yi:
     - İmkân varsa **net getiri olarak geçmek**,
     - En azından **aynı getiri civarında, daha düşük DD** ile “daha az riskli” hale getirmek.

---

## 6. Sonraki Adımlar (Plan)

Benchmark-aware reward devredeyken:

1. Yeni `ppo_eval_summary.json` ve `ppo_eval_equity_curve.csv` çıktılarında:
   - `total_return`,
   - `max_drawdown`,
   - `num_trades`,
   - `unique_actions`,
   - Bot vs benchmark PV eğrileri (gerekirse ayrı çizerek),

incelenecek.

2. Eğer hâlâ:
   - DD çok yüksek,
   - Benchmark’ın çok altında sonuçlar varsa,

şu iyileştirmeler gündeme gelecek:

- **Risk-aware shaping**:
  - Max drawdown veya kısa vadeli volatiliteye küçük bir ceza katsayısı ekleme.
- **PPO hyperparam tuning**:
  - `gamma`, `learning_rate`, `ent_coef`, `n_steps` gibi parametrelerle hafif oynamalar.

3. Uzun vadede:

- Env’in multi-asset’e genişletilmesi (FinRL `PortfolioOptimizationEnv` tarzı),
- RL ajanının canlı/paper trading loop’una entegrasyonu:
  - Sinyal/strateji + ML + RL karar katmanını aynı pipeline’da çalıştırma.

---

## 7. İlgili Dosyalar Özeti

- Dataset:
  - `scripts/prepare_rl_training_data.py`
  - `data/training/BTC_USDT_USDT_1h_*.npz`
  - `data/training/BTC_USDT_USDT_1h_metadata.json`
- RL env:
  - `src/ml/rl_trading_env.py`
  - `src/ml/rl_trading_env_gym.py`
- PPO training/eval:
  - `scripts/train_ppo_agent.py`
  - `scripts/evaluate_ppo_agent.py`
- Diğer (bağlı bileşenler):
  - `src/core/ccxt_client.py`
  - `src/ml/feature_engineering.py`
  - `config/config.example.yaml` (RL config parametreleri, fee, reward clip, idle_cost vb.)

Bu dosya, RL ajanının evrimini ve mevcut tasarım kararlarını özetleyen kalıcı bir referans olarak güncellenebilir.
