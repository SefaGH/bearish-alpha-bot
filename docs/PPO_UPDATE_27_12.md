Aşağıdaki doküman, PPO entegrasyonu için bugüne kadar yaptığımız işleri, mevcut durumu ve “paper run” öncesi/sonrası operasyonel kontrolleri tek bir **playbook** halinde toplar. Konu başlıkları, **dosya adları**, **container path’leri**, **config parametreleri** ve **log’da görülmesi gereken satırlar** ile birlikte verilmiştir.

---

## 1) Amaç ve kapsam

**Amaç:** PPO (Stable-Baselines3 PPO) ajanını live bot akışında “signal veto/confirm” katmanı olarak güvenli şekilde devreye almak.

**Kapsam:**

* Observation schema (feature list) tekilleştirme (spec)
* Deterministic scaling (training ↔ live parity)
* VecNormalize (obs normalization) yükleme ve guard’lar
* Health-guard (fail-closed) + clipping regresyonlarının giderilmesi
* Action/score indeksleme ve semantik doğrulama
* Paper run öncesi checklist + izleme metrikleri

---

## 2) Repo/artefakt haritası (kritik dosyalar)

### Kod

* `src/ml/ppo/deterministic_scaler.py`
  **DeterministicScaler**: Spec tabanlı, isim→transform map ile deterministik scaling.
* `src/ml/adapters/ppo_trading_adapter.py`
  **PPOTradingAdapter**: Model + spec + vecnorm yükleme, state üretimi, health guard, score/action üretimi.
* `scripts/build_ppo_dataset_from_live_pipeline.py`
  Live pipeline ile aynı feature üretim hattından PPO dataset üretimi (train/val/test split + metadata).
* `scripts/train_ppo_agent.py`
  PPO eğitim script’i (SB3 PPO). Eğitim çıktısı olarak `.zip` + `.obs_spec.json` + `.vecnormalize.pkl`.
* `src/tools/ppo_observation_parity_check.py`
  Aynı index için training obs ↔ live obs **parite** kontrolü + policy çıktısı kıyası.

### Artefaktlar (container içi)

Container’da şu dizin kritik:

* `/app/artifacts/ppo/`

Sende mevcut envanter (paylaştığın `ls -la /app/artifacts/ppo` çıktısına göre):

* `ppo_trading_agent.zip`
* `ppo_trading_agent.obs_spec.json`
* `ppo_trading_agent.vecnormalize.pkl`
* Versiyonlu: `ppo_trading_agent_v13.zip / .obs_spec.json / .vecnormalize.pkl`
* Smoke: `ppo_trading_agent_v13_smoke.*`
* Diğer varyantlar: `_aggressive`, `_balanced`, `_debug`, `_prev`

---

## 3) Observation schema (spec) – “tek kaynak” prensibi

### 3.1 Obs boyutu

Senin parity check çıktıların net:

* **Training_obs_len = 89**
* **Live_obs_len = 89**

Bu 89 şu şekilde oluşuyor (v13 spec’ine göre):

* `feature_names`: **82**
* `extra_feature_names`: **5**
* `tail_names`: **2**
* Toplam: **82 + 5 + 2 = 89**

### 3.2 Spec dosyası (deploy’de referans)

Model path’in baz adına göre otomatik türetiliyor. Adapter şu mantıkla arıyor:

* Model: `.../ppo_trading_agent_v13.zip`
* Spec:  `.../ppo_trading_agent_v13.obs_spec.json`
* VecNorm: `.../ppo_trading_agent_v13.vecnormalize.pkl`

Dolayısıyla **config’te hangi `.zip` seçiliyse**, yanında aynı baz adla bu iki dosya da **mutlaka** olmalı.

### 3.3 Spec içeriği (v13 örneği)

`extra_feature_names`:

* `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`

`tail_names`:

* `market_phase`, `position_ratio`

> Not: 82 ana feature’ın tam listesi spec JSON içinde. Operasyonel olarak “spec dosyası tek gerçek” yaklaşımını koruyoruz; feature listesi değişirse **dataset yeniden üretilir + PPO yeniden eğitilir**.

---

## 4) Bugüne kadar yapılan PPO çalışmaları (özet)

### 4.1 “Training ↔ Live observation parity” problemi çözüldü

* Parity check çıktılarında:

  * **Vector stats (mean/std/min/max)** birebir aynı
  * **Top Diff Indices**: diff `+0.000000`
  * **Policy Stats**: `p_flat`, `p_long`, `entropy`, `logits` training vs live neredeyse birebir (float yuvarlama farkı dışında)

Bu, en kritik hedef olan **“aynı input → aynı model kararı”** garantisini pratikte sağladığınız anlamına gelir.

### 4.2 Clipping/regresyon problemi çözüldü

Eski loglarda PPO health satırlarında “clip_mean” yüksek görünüyordu (örnek format: `clip_mean: 0.79` gibi).
Yeni logda aynı metrik:

* `clip_mean: 0.0`

Bu, **DeterministicScaler + VecNormalize** hattının “input’ları clip obs sınırlarına vurmadan” çalıştığını doğruluyor.

### 4.3 “Puanlama/indeksleme” mantık hatası düzeltildi

Paylaştığın analizde tespit edilen kritik nokta şuydu: PPO çıktısında `[p_flat, p_long]` varken skorun yanlış indeksten alınması riski.

Güncel adapter kodu tarafında (düzeltme sonrası) doğru semantik:

* `p_long = dist.distribution.probs[1]`
* `action_int` (0/1) ile karar: 1 ise LONG

Bu düzeltme yapılmadan **model LONG derken adapter LOW score görüp veto** edebiliyordu.

---

## 5) Mevcut durum (2025-12-27 run log gözlemleri)

### 5.1 Yükleme / init tarafı

Yeni logda PPO adapter init satırları görünüyor (kısaltılmış formatla da olsa):

* `PPOTradingAdapter initialized ... enabled=True ... cfg.symbols=['BTC/USDT:USDT'] ...`
* `Using cached PPO model ... /app/artifacts/ppo/ppo_trading_agent.zip`
* `Loaded obs_spec: ... /app/artifacts/ppo/ppo_trading_agent.obs_spec.json`

Bu, **model_path → spec_path türetme** zincirinin çalıştığını gösterir.

### 5.2 Yeni açık nokta: “health_guard_low_variance” false-positive

Yeni logda düzenli aralıklarla şu uyarı görünüyor (satır içeriği kısaltılmış):

* `... Reasons: ['health_guard_low_variance'] | Stats: {'p_long_std': 0.0, 'clip_mean': 0.0}`

Bu şu anlama gelir:

* Clipping yok (iyi).
* Ancak health window içinde `p_long` varyansı **0** çıktığı için guard devreye giriyor.

**Muhtemel neden (en sık):** 1h timeframe’de yeni bar gelmeden çok sık evaluation yapıyorsanız, observation değişmiyor → `p_long` da değişmiyor → std=0 → guard tetikleniyor. Bu “gerçek arıza” değil, “ölçüm yaklaşımı” problemi.

**Paper run için kritik etkisi:**
Adapter “fail-closed” mantığında fallback skor (çoğunlukla `ppo_fallback_score=0.5`) döndürürse, `ppo_conf_threshold=0.60` ile birlikte PPO sürekli “yetersiz güven” sayılıp sinyalleri veto edebilir.

---

## 6) Konfigürasyon: PPO ile ilgili parametreler ve anlamları

Senin `config/config.example.yaml` grep çıktına göre PPO alanı:

`ml.reinforcement_learning` altında:

* `enabled` (ML_RL_ENABLED)
* `ppo_enabled` (ML_RL_PPO_ENABLED)
* `ppo_symbols` (ML_RL_PPO_SYMBOLS) = `BTC/USDT:USDT`
* `ppo_timeframe` (ML_RL_PPO_TIMEFRAME) = `"1h"`
* `ppo_model_path` (ML_RL_PPO_MODEL) = `"artifacts/ppo/ppo_trading_agent.zip"`
* `ppo_fallback_score` (ML_RL_PPO_FALLBACK) = `0.5`
* `ppo_rr_down_mult` (ML_RL_PPO_RR_DOWN)
* `ppo_rr_up_mult` (ML_RL_PPO_RR_UP)
* `ppo_position_base` (ML_RL_PPO_POS_BASE)
* `ppo_position_bonus` (ML_RL_PPO_POS_BONUS)
* `ppo_lookback_bars` (ML_RL_PPO_LOOKBACK_BARS) = `240`
* `ppo_lookback_windows` (ML_RL_PPO_LOOKBACK_WINDOWS) = `[12,24,48,96]`
* `ppo_conf_threshold` (ML_RL_PPO_CONF_THRESHOLD) = `0.60`
* `ppo_min_margin` (ML_RL_PPO_MIN_MARGIN) = `0.0`
* `ppo_health_min_std` (ML_RL_PPO_HEALTH_MIN_STD) = `0.001`
* `ppo_health_window` (ML_RL_PPO_HEALTH_WINDOW) = `30`
* `ppo_health_clip_frac_limit` (ML_RL_PPO_HEALTH_CLIP_LIMIT) = `0.30`
* `ppo_require_vecnorm` (ML_RL_PPO_REQUIRE_VECNORM) = `true`

### Paper run için önerilen güvenli geçiş ayarı (özellikle low_variance nedeniyle)

* `ppo_health_min_std`: **0.0** (veya çok daha düşük)
  Amaç: “bar değişmediği için std=0” durumunda PPO’nun kendini kilitlemesini engellemek.
* `ppo_health_window`: **bar-bazlı** çalışana kadar anlamı sınırlı. (Aşağıda kalıcı çözüm var.)

---

## 7) Paper run öncesi “net” checklist

### 7.1 Config’te model_path nerede?

* Dosya: `config/<kullandığın_live_config>.yaml`
* Alan: `ml.reinforcement_learning.ppo_model_path`

Örnek:

* `ppo_model_path: "artifacts/ppo/ppo_trading_agent.zip"`

> Not: `config.example.yaml` içinde `training_mode: true` görünüyor. Live için bunun **false** olması gerekir (ya config’te değiştirin ya da env var ile override edin: `ML_RL_TRAINING_MODE=false`).

### 7.2 Container’da hangi path’e kopyalanmalı?

Config’teki relative path `/app` kökünden çözülüyor. Yani:

* `artifacts/ppo/ppo_trading_agent.zip` → **/app/artifacts/ppo/ppo_trading_agent.zip**

Aynı baz adla **yan dosyalar** da şart:

* `/app/artifacts/ppo/ppo_trading_agent.obs_spec.json`
* `/app/artifacts/ppo/ppo_trading_agent.vecnormalize.pkl`

#### Eğer v13’ü aktif edeceksen (en temiz yöntem)

**Seçenek A – config’i v13’e çevir:**

* `ppo_model_path: "artifacts/ppo/ppo_trading_agent_v13.zip"`

Bu durumda adapter şunları arar:

* `ppo_trading_agent_v13.obs_spec.json`
* `ppo_trading_agent_v13.vecnormalize.pkl`

**Seçenek B – “active alias” mantığı (önerilir)**
Config’i değiştirmeden “aktif model”i alias’a bağla:

```bash
cd /app/artifacts/ppo
cp -f ppo_trading_agent_v13.zip ppo_trading_agent.zip
cp -f ppo_trading_agent_v13.obs_spec.json ppo_trading_agent.obs_spec.json
cp -f ppo_trading_agent_v13.vecnormalize.pkl ppo_trading_agent.vecnormalize.pkl
```

(İstersen `cp` yerine `ln -sf` ile symlink de kullanabilirsin.)

### 7.3 Start loglarında hangi satırlar görülmeli?

Paper run’da (tercihen INFO/DEBUG seviyesinde) şu imzaları arayın:

**Yükleme / init**

* `PPOTradingAdapter initialized ... enabled=True ...`
* `Loaded ... ppo_trading_agent*.zip`
* `Loaded obs_spec: ... ppo_trading_agent*.obs_spec.json`
* `Loaded VecNormalize ... ppo_trading_agent*.vecnormalize.pkl`
* `DeterministicScaler initialized ... features=89` (DEBUG/INFO seviyesine bağlı)

**Sağlık / clipping**

* `clip_mean: 0.0` civarı (senin yeni logda böyle)
* Eğer görürsen: `clip_mean` yükselmesi veya `obs_clip_high` gibi reason’lar → scaling regresyonu alarmı

**Karar**

* `Action:` ve `Score:` (ör. `p_long`) tutarlı olmalı:

  * Action=1 (LONG) iken Score düşük görünüyorsa → indeksleme/semantik tekrar kontrol
  * Score yüksek iken Action=0 görünüyorsa → model/policy mapping tekrar kontrol

---

## 8) Paper run sırasında izleme playbook’u (operasyonel)

### 8.1 Minimum izleme metrikleri

* **PPO load status:** model/spec/vecnorm yüklendi mi?
* **clip_mean:** 0’a yakın mı?
* **p_long_std:** (bar-bazlı ölçüme geçene kadar) “0” görürseniz bunun bar değişmediği için olabileceğini not edin.
* **Decision rate:** PPO sürekli veto mu ediyor?
* **Latency:** PPO inference süreleri (özellikle tek çekirdekte) – spike var mı?

### 8.2 “health_guard_low_variance” için kısa vadeli önlem

Paper run’ı bloke etmemesi için (geçici):

* `ppo_health_min_std=0.0` (env override ile hızlı)
* Alternatif: `ppo_conf_threshold` biraz aşağı çekmek veya `ppo_fallback_score` yükseltmek (daha riskli; ilk seçenek daha temiz)

### 8.3 Kalıcı çözüm (önerilen teknik düzeltme)

Health guard’ın `p_long` history’si **evaluation bazlı** değil, **bar değişimi bazlı** tutulmalı:

* History’ye `p_long` ekleme koşulu: `last_closed_ts` değiştiğinde ekle
* Böylece `health_window=30` gerçekten “son 30 bar” olur
* `p_long_std=0` yalnızca model gerçekten sabit çıktı üretiyorsa alarm olur

---

## 9) Eğitim / dataset üretimi – işletim kuralları

### 9.1 Dataset üretimi (live pipeline ile aynı hat)

* Script: `scripts/build_ppo_dataset_from_live_pipeline.py`
* Üretilenler:

  * `..._train.npz`, `..._val.npz`, `..._test.npz`
  * `...metadata.json`
  * `...obs_spec.json` (schema snapshot)

Kural:

* Feature manifest/spec değiştiyse → **dataset yeniden üretilir** → **PPO yeniden eğitilir**.

### 9.2 Eğitim

* Script: `scripts/train_ppo_agent.py`
* Çıktılar (örnek):

  * `ppo_trading_agent_v13.zip`
  * `ppo_trading_agent_v13.obs_spec.json`
  * `ppo_trading_agent_v13.vecnormalize.pkl`

Kural:

* Model `.zip` tek başına deploy edilmez; **spec + vecnorm** birlikte deploy edilir.

### 9.3 Parity check (release gate)

* Script: `src/tools/ppo_observation_parity_check.py`
* Gate kriterleri:

  * obs len: **89/89**
  * Top diffs: ~0
  * Policy stats: training vs live aynı (minör float farkı tolerans)

Senin çıktılar bu gate’i geçti; bu şu anki entegrasyonun en güçlü doğrulaması.

---

## 10) “Son durum” değerlendirmesi ve next steps

### Mevcut durum

* **Parity:** Geçti (çok güçlü sinyal).
* **Clipping:** Eski regresyon çözüldü (`clip_mean` artık 0).
* **Index bug:** Düzeltildi (action/score semantiği doğru).
* **Açık nokta:** `health_guard_low_variance` — şu an paper run’da gereksiz veto üretebilir.

### Bundan sonra dikkat edilmesi gerekenler

1. **Active model seçimi netleştir:**
   `ppo_trading_agent.zip` mi aktif, yoksa `ppo_trading_agent_v13.zip` mi? Config/alias mutlaka tutarlı olsun.
2. **Spec/vecnorm eşleşmesi:**
   `.zip` ile aynı baz ad + aynı versiyon spec/vecnorm deploy edilmezse “sessiz” davranış bozulur.
3. **Health guard kalibrasyonu:**
   Low_variance kontrolünü bar-bazlı hale getir (kalıcı çözüm). Paper run için min_std’yi geçici 0’a çek.
4. **training_mode:**
   Live’da `training_mode=false` olduğundan emin ol (env override dahil).
5. **Log seviyeleri:**
   Paper run’da en azından INFO, tercihen PPO adapter için DEBUG açıp ilk 1-2 saat “karar izlerini” yakala.

---
