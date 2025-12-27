# PPO Training Playbook

Bu rehber, PPO tabanlı trading ajanını yeniden eğitmek, değerlendirmek ve canlıya almak için adım adım bir referans sunar. Amaç, Flat Bias sorununu gideren “Balanced Model” sürecini tekrar edilebilir ve denetlenebilir kılmaktır.

## 1) Eğitim Öncesi Hazırlık (Prerequisites)
- **Veri seti oluşturma:** `scripts/build_ppo_dataset_from_live_pipeline.py` (veya proje içinde eşdeğer builder) ile canlı pipeline’a uyumlu `.npz` veri seti üretin. Çıktılar tipik olarak `data/training/` altına kaydolur ve şu alanları içerir: `features (N,82) float32`, `prices (N,5) float32`, `feature_columns`, `price_columns`, `timestamps`, `symbol`, `timeframe`, `metadata` (örn. `initial_balance`, `fee_pct`). Mevcut örnekler: `data/training/BTC_USDT_USDT_1h_liveparity_{train,val,test}.npz`.
- **Gözlem sözleşmesi:** `artifacts/ppo/ppo_trading_agent.obs_spec.json` veya datasetle beraber gelen `.obs_spec.json` dosyası 89 boyutlu gözlemi (82 özellik + 5 fiyat ekstrası + 2 kuyruk) tanımlar. Eğitim ve inference’ta aynı spec kullanılmalıdır.
- **Çekirdek ayarlar:**  
  - `src/ml/rl_trading_env.py` → `idle_penalty=-5e-5` (denge için), hybrid reward (0.7 * absolute log return + 0.3 * benchmark-relative).  
  - `scripts/train_ppo_agent.py` → PPO hiperparametreleri: `ent_coef=0.02`, `learning_rate=2.5e-4`, `gamma` SB3 varsayılanı (0.99) ile uyumlu. Reward clipping `[-5,5]` (VecNormalize ile birlikte).  
  - VecNormalize zorunlu: hem eğitim hem inference aynı `.vecnormalize.pkl` ile çalışmalı.
- **Bağımlılıklar:** Python 3.11, Stable-Baselines3 (PPO), Gym (compat layer uyarıları görülebilir; SB3 otomatik sarıyor).

## 2) Eğitim Süreci (Training Pipeline)
- **Komut (örnek):**  
  ```bash
  python scripts/train_ppo_agent.py \
    --dataset data/training/BTC_USDT_USDT_1h_liveparity_train.npz \
    --model-dir artifacts/ppo \
    --model-name ppo_trading_agent \
    --timesteps 300000 \
    --obs-spec data/training/BTC_USDT_USDT_1h_liveparity.obs_spec.json
  ```
- **İzlenecek sinyaller (terminal/TensorBoard):**  
  - `entropy_loss`: Çok hızlı 0’a inmesi erken çöküş (flat bias) riskidir; 0.5–0.7 seviyeleri sağlıklı keşfe işaret eder.  
  - `explained_variance`: Negatif değerler value tahminlerinin kötü olduğunu gösterir; zamanla 0–0.5’e yaklaşması beklenir.  
  - `approx_kl` / `clip_fraction`: Aşırı sıklıkla yüksek KL, politikanın zorlandığını gösterir.  
  - `reward` trendi: Sürekli 0 veya klips limitlerinde kalıyorsa gözlem/ödül hatasını araştırın.
- **Süre:** Küçük doğrulamalar için 20–50k timesteps; üretim modeli için 200k–500k+ timesteps (veri büyüklüğüne ve donanıma göre).

## 3) Çıktılar ve Artifact Yönetimi (Outputs)
- Eğitim sonunda `model-dir` altında üretilen dosyalar:  
  - `{model-name}.zip` (SB3 PPO ağırlıkları)  
  - `{model-name}.vecnormalize.pkl` (gözlem istatistikleri)  
  - `{model-name}.obs_spec.json` (gözlem şeması; değişmediyse aynı kalır)  
- Varsayılan konum: `artifacts/ppo/`.

## 4) Değerlendirme ve Seçim (Evaluation)
- **Komut (örnek):**  
  ```bash
  python scripts/evaluate_ppo_agent.py \
    --model artifacts/ppo/ppo_trading_agent.zip \
    --dataset data/training/BTC_USDT_USDT_1h_liveparity_test.npz \
    --obs-spec data/training/BTC_USDT_USDT_1h_liveparity.obs_spec.json \
    --output-summary data/training/ppo_eval_summary.json \
    --output-equity-curve data/training/ppo_eval_equity_curve.csv
  ```
- **Başarı kriterleri (öneri):**  
  - Pozitif toplam getiri ve Buy&Hold’a kıyasla üstün/benzer performans.  
  - Makul risk: max drawdown kabul edilebilir aralıkta (ör. < -10%).  
  - İşlem çeşitliliği: `unique_actions` hem 0 hem 1; `num_trades` sıfır değil, aşırı da değil (veri rejimine göre).  
  - Sharpe (>1.0 tercih) ve entropy’nin tamamen çökmediğini doğrulayan dağılım.  
  - VecNormalize’nin yüklendiğini ve obs_spec uyumunu loglardan teyit edin.

## 5) Canlıya Alma (Deployment)
- **Standart dosya adları:** Onaylanan modeli `artifacts/ppo/` altına şu isimlerle kopyalayın:  
  - `ppo_trading_agent.zip`  
  - `ppo_trading_agent.vecnormalize.pkl`  
  - `ppo_trading_agent.obs_spec.json`
- **Versiyonlama:** Mevcut dosyaları yedekleyin (örn. `ppo_trading_agent_prev.*`) taşıma öncesi `Copy-Item`/`cp` ile kopyalayarak.
- **Hızlı adaptör testi:**  
  - `src/tools/ppo_observation_parity_check.py --model artifacts/ppo/ppo_trading_agent.zip --dataset <test_npz> --index -1`  
  - Beklenti: obs_dim hizalı (89), vecnorm yüklü, `p_long/p_flat` 0–1 aralığında. Hata veya uyumsuzluk yoksa canlı adapter yükleyebilir.

## 6) Operasyonel Notlar
- Çevrim içi/Shadow modda `[PPO-INIT]` ve `[PPO-DEBUG]` loglarıyla yükleme ve dağılımı doğrulayın: `obs_norm_present=True`, makul `obs_clip_frac`, değişen `p_long`.
- **Runtime toggles (PPO):** `ppo_include_forming` ile forming candle dahil/harici kontrolü yapılır (varsayılan: closed-only). Cache hit log sıklığı `PPO_CACHE_LOG_INTERVAL_SEC` env değişkeniyle yönetilir (varsayılan: 300s).
- Downtrend verilerinde kısa satış desteği yok; flat kalmak beklenen davranış olabilir. Bullish rejim testi için yükseliş dönemli dataset veya artırılmış idle_penalty kullanarak aksiyon çeşitliliği gözlenebilir.

Bu playbook, Balanced Model eğitim akışını kalıcı hale getirmek ve yeni geliştiricilerin aynı süreci güvenle tekrarlamasını sağlamak için hazırlanmıştır. Model davranışı değiştiğinde veya yeni veri rejimleri eklendiğinde metrik eşiklerini güncelleyin ve değerlendirme/taşıma adımlarını yeniden çalıştırın.
