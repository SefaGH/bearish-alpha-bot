# Kaçırılan Fırsat Analizi — BTC/USDT 19 Şubat 2026

**Tarih:** 2026-02-19  
**Zaman Dilimi:** ~14:30 – 14:55 UTC  
**Sembol:** BTC/USDT:USDT (BingX Perpetual)  
**Grafik TF:** 5m  
**Container:** bearish-bot (bearishalphabot.azurecr.io/bearish-bot:manual-20260219-v1)  
**Analiz Zamanı:** 2026-02-19 15:00 UTC  

---

## 1. Piyasa Durumu (Grafik Özeti)

| Metrik | Değer |
|--------|-------|
| Yatay Direnç | ~67,050 |
| Konsolidasyon Bandı (06:00–12:00) | 66,250 – 67,050 |
| Breakdown Başlangıcı | ~12:30 UTC |
| Sert Düşüş Noktası (Fırsat) | ~14:35 UTC |
| Fiyat Düşüş Hedefi | ~65,598 |
| Toplam Hareket | ~67,050 → 65,598 (≈ %2.2 drop) |
| 14:30–14:55 Arası Hareket | ~66,600 → 65,598 (≈ %1.5 drop) |

**Yorum:** Fiyat yatay direnci (~67,050) kıramayıp konsolidasyon bandının altına sert kırılma yapmıştır. 14:35 civarında momentum hızlanmış ve ciddi bir short/sell fırsatı oluşmuştur. Bot bu hareketten faydalanamamıştır.

---

## 2. Bot Durumu — Genel Bakış

| Metrik | Değer |
|--------|-------|
| Bot Status | Running (Up ~1 hour) |
| Data Feed | WebSocket aktif, gap yok |
| İterasyon Aralığı | ~30s per loop |
| Aktif Semboller | 1 (BTC/USDT:USDT) |
| Üretilen Sinyal | **0** |
| Açılan Pozisyon | **0** |
| Hata Sayısı | 0 |

Bot teknik olarak çalışıyordu, veri akışı sağlamdı, ancak **hiçbir trade sinyali üretilemedi.**

---

## 3. Kök Neden Analizi — 3 Katmanlı Blokaj

### 3.1 RSI Router — 14:30–14:57 Arasında Deterministik Blokaj

**Durum (doğrulandı):** 14:30–14:57 aralığında `adaptive_ob` ve `adaptive_str` her döngüde RSI Router tarafından engellendi.

- `adaptive_ob` `transition_no_trade`: **52/52**
- `adaptive_str` `transition_no_trade`: **52/52**
- Zone dağılımı (bu pencere): **TRANSITION_LOW = 104/104**
- `consensus_status`: **mismatch_transition = 104/104**

**Log Örneği (14:35:04):**
```
[RSI-ROUTER] Skip | symbol=BTC/USDT:USDT | strategy=adaptive_ob
  reason=rsi_router.transition_no_trade
  zone=TRANSITION_LOW
  rsi_level=32.17 | rsi_slow=32.17 | rsi_fast=38.90
  ob_threshold=32.00 | str_threshold=55.00
  consensus_status=mismatch_transition
```

**Sorunun Detayı (bu pencere):**

- `rsi_slow` bu pencerede **32.17’de sabit** kaldı.
- `ob_threshold=32.00`, `str_threshold=55.00`; bu yüzden sınıflama sürekli `TRANSITION_LOW` oldu.
- `rsi_fast` 39.77’den 58.86’ya çıksa da consensus mismatch devam ettiği için zone değişmedi.
- Sonuç: `adaptive_ob` ve `adaptive_str` için yeni girişler her iterasyonda bloklandı.

**RSI Zaman Çizelgesi:**

| Zaman | rsi_level | rsi_slow | rsi_fast | Zone |
|-------|-----------|----------|----------|------|
| 13:27 | 41.06 | 41.06 | 25.59 | TRANSITION_LOW |
| ~13:50 | 34.20 | 34.20 | 25.60 | TRANSITION_LOW |
| ~14:10 | 30.24 | 30.24 | 34.59 | OVERSOLD |
| ~14:27 | 32.17 | 32.17 | 39.77 | TRANSITION_LOW |
| 14:35 | 32.17 | 32.17 | 38.90 | TRANSITION_LOW |
| 14:55 | 32.17 | 32.17 | 58.86 | TRANSITION_LOW |

**Kritik Nüans:** Sadece `transition_no_trade` kuralını kaldırmak tek başına yeterli değil.  
`adaptive_ob` yalnız `OVERSOLD`, `adaptive_str` yalnız `OVERBOUGHT` zone'unda çalıştığı için bu kez `zone_mismatch` blokajı devreye girebilir.

**Sprint-3 Sonrası Davranış Güncellemesi (2026-02-19):**
- `rsi_zone_router` içine hedefli `transition.shock_override` eklendi.
- Override sadece `transition` zonunda, canary sembolde, izinli stratejilerde ve `shock_state + min_score + min_adx` koşulları sağlanırsa çalışır.
- `enforce` modunda strateji `rsi_router.transition_shock_override` ile geçer.
- `observe` modunda blokaj korunur ve `rsi_router.observe_would_override_transition` reason'ı loglanır.

**Zone Dağılımı (Tüm Çalışma Süresi):**

| Zone | Sayı | Oran |
|------|------|------|
| TRANSITION_LOW | 236 | %67 |
| OVERSOLD | 112 | %32 |
| TRANSITION_HIGH | 4 | %1 |

Bot çalışma süresinin **%99'unda** ya OVERSOLD ya da TRANSITION_LOW'daydı — hiç NEUTRAL veya tradeable bir zona geçemedi.

---

### 3.2 MeanReversion Stratejisi — Dynamic Z Veto (52/52)

RSI Router'ı geçebilen tek strateji `mean_reversion` idi ve **52 kez işlendi**.  
Ancak her seferinde **Dynamic Z veto** aldı.

**Log Örneği (14:35:36):**
```
[MeanReversion] Dynamic Z veto BTC/USDT:USDT: z=1.27 required=1.90 adx=56.54
[MeanReversion] Cycle complete for BTC/USDT:USDT. Action: HOLD
```

**Z-Score Zaman Çizelgesi (14:30–14:57):**

| Zaman | Z-Score | Gereken Eşik | ADX | Sonuç |
|-------|---------|--------------|-----|-------|
| 14:30:16 | 0.47 | 1.90 | 58.78 | **VETO** |
| 14:30:48 | 0.55 | 1.90 | 57.09 | **VETO** |
| 14:35:04 | 0.44 | 1.90 | 57.09 | **VETO** |
| 14:35:36 | 1.27 | 1.90 | 56.54 | **VETO** |
| 14:36:08 | 1.23 | 1.90 | 56.54 | **VETO** |
| 14:36:40 | 1.23 | 1.90 | 56.54 | **VETO** |
| 14:37:12 | 1.20 | 1.90 | 56.54 | **VETO** |
| 14:54:17 | 0.73 | 1.90 | 47.22 | **VETO** |
| 14:55:21 | 1.43 | 1.90 | 45.20 | **VETO** |
| 14:56:25 | 1.20 | 1.90 | 45.20 | **VETO** |

**Sorunun Detayı:**
- Z-score **hiçbir zaman 1.90 eşiğine ulaşamadı**. En yüksek değer **1.43** (14:55).
- ADX aralığı **45.20–58.78** olduğu için `required=1.90` sabit kaldı.
- Bu pencerede:
  - `z >= 1.25`: **3/52**
  - `z >= 1.20`: **9/52**
  - `z >= 1.90`: **0/52**

**Kritik Nüans:** MR tarafında sadece Z eşiğini düşürmek de tek başına yeterli değil.  
Strateji giriş için ayrıca band dışı fiyat koşulu arıyor (`price < lower` veya `price > upper`).  
Ek olarak kodda `high_adx_z_threshold` için alt sınır `1.60` olarak zorlanıyor; yani konfigürasyonla 1.20 gibi seviyelere inmek şu an mümkün değil.

---

### 3.3 ML Modelleri — Bearish Sinyali Yok

#### Regime Predictor
```
Prediction: neutral (confidence: 0.29)
```
- Tüm dönem boyunca **sürekli "neutral"** tahmini.
- Güven skoru sadece 0.29 — çok düşük ve kararsız.

#### Gemma ML Modeli
```
[GEMMA] conf_p50=0.976 conf_p95=0.991 class_counts={'bearish': 0, 'neutral': 19, 'bullish': 0}
```
- **Tek bir bearish çıkış bile üretmedi.**
- 19/19 tahmin "neutral" — model sert düşüşü algılayamadı.

#### PPO Modeli
```
[PPO-CACHE] hit sym=BTC/USDT tf=1h last_candle_ts=2026-02-19T13:00:00+00:00 age_s=2275.0
```
- PPO 1h cache kullanıyordu ve son mum **13:00 kapanışından** geliyordu.
- 14:35'te cache yaşı **~38 dakika** ile eski veri kullanılıyordu.
- RL telemetri: `PPO samples=2 | avg_score=0.000 | long=0 | flat=2` — model sürekli "flat" kararı verdi.

---

### 3.4 Shock Dedektörü — Tespit Etti Ama Etkisiz

ShockScore düzgün çalıştı ve düşüşü doğru tespit etti:

| Zaman | Shock Durumu | ShockScore |
|-------|-------------|------------|
| 14:30:16 | DISARMED | 0.08 |
| 14:30:48 | DISARMED | 0.03 |
| 14:35:04 | **ARMED** | **0.68** |
| 14:35:36 | **ARMED** | **0.64** |
| 14:36:08 | **ARMED** | **0.76** |
| 14:36:40 | **ARMED** | **0.76** |
| 14:54:17 | **ARMED** | **0.53** |
| 14:55:21 | **ARMED** | **0.81** |
| 14:56:25 | **ARMED** | **0.89** |

**Olay Anındaki Sorun:** `Shock=ARMED` durumu **trade tetikleme mekanizmasına bağlı değildi** ve yalnız log amaçlıydı.  
**Sprint-3 Sonrası:** Shock sinyali artık RSI Router kararına kontrollü şekilde bağlandı (`transition.shock_override`).

---

## 4. Blokaj Akış Diyagramı

```
Sert düşüş başladı (14:30–14:35)
  │
  ├─→ adaptive_ob stratejisi
  │     └─→ RSI Router: zone=TRANSITION_LOW, mismatch_transition → ❌ BLOKE
  │
  ├─→ adaptive_str stratejisi
  │     └─→ RSI Router: zone=TRANSITION_LOW, mismatch_transition → ❌ BLOKE
  │
  └─→ mean_reversion stratejisi
        ├─→ RSI Router: geçti ✅
        ├─→ Level Router: geçti ✅ (zone=AT_LEVEL → IN_RANGE)
        └─→ Dynamic Z-Score: z=0.44–1.43 < required=1.90 → ❌ VETO
  
  Ek Filtreler:
  ├─→ Regime Predictor: neutral (conf=0.29) → bearish sinyali yok
  ├─→ Gemma: 0 bearish / 19 neutral → bearish sinyali yok
  ├─→ PPO: flat (score=0.000) → trade yok
  └─→ Shock Detector: ARMED (0.68–0.89) → 🟡 (OLAY ANINDA) KULLANILMIYOR

  Sonuç: 0 sinyal üretildi, 0 trade açıldı ❌
```

---

## 5. Öneriler

### 5.1 Kritik — RSI Router İçin Hedefli Geçiş Override (Global Değil)
**Öncelik:** 🔴 Yüksek  
**Sorun:** `transition_no_trade` blokajı doğru tespit edildi ama bunu global gevşetmek riskli.  
**Uygulanan Çözüm (Sprint-3):**
- `strategies.rsi_zone_router.transition.shock_override` bloğu eklendi.
- `mode`: `enforce | observe | off`, `canary_symbols`, `allow_strategies`, `state`, `min_score`, `min_adx` parametreleri ile kontrollü rollout sağlandı.
- `production_coordinator` hem main loop hem `dispatch_strategy` recheck yolunda `shock_state`, `shock_score`, `regime_adx` alanlarını RSI snapshot'a taşıyor.
- Böylece sadece hedefli koşulda `adaptive_str` gibi stratejiler transition bölgesinde geçebiliyor; global gevşetme yok.

**Yeni Reason Kodları:**
- `rsi_router.transition_shock_override` (enforce geçiş)
- `rsi_router.observe_would_override_transition` (observe telemetri)

### 5.2 Kritik — MeanReversion Z Eşiği ve Kod Sınırı Refactor
**Öncelik:** 🔴 Yüksek  
**Sorun:** `required=1.90` bu pencere için aşırı yüksek; ayrıca kodda alt sınır `1.60` hard-limit.  
**Çözüm:**
- `high_adx_z_threshold` hard floor (`max(1.60, ...)`) konfigürasyon kontrollü hale getirilmeli.
- A/B canary ile kademeli dene (`1.90 -> 1.60 -> 1.30`) ve ek telemetri ekle:
  - `z_pass_but_no_entry`
  - `entry_blocked_by_band`
  - `entry_blocked_by_regime_policy`

### 5.3 Yüksek — Shock için Ayrı Momentum/Breakdown Short Yolu
**Öncelik:** 🟠 Yüksek  
**Sorun:** İncelenen hareket mean-reversion karakterinden çok continuation/breakdown karakterinde.  
**Çözüm:** `Shock=ARMED` penceresinde MR/OB/STR’i zorlamak yerine ayrı bir kısa-yönlü momentum yolu ekle:
- Trigger: `Shock=ARMED` + kırılım teyidi + minimum momentum/volume koşulu
- Risk: kısa süreli cooldown, tek-yön guard, sıkı stop
- Rollout: yalnız canary sembol ve düşük allocation ile başlat

### 5.4 Orta — Gemma Modeli Bearish Sınıf Kalibrasyonu
**Öncelik:** 🟡 Orta  
**Sorun:** Gemma modeli tüm dönemde tek bir bearish çıkış üretmedi (0/19).  
**Çözüm:** Model eğitim verisinde bearish örneklerin yeterliliğini kontrol et. Sınıf dengesizliği varsa yeniden eğitim veya threshold kalibrasyonu yap.

### 5.5 Orta — PPO Cache Stratejisi
**Öncelik:** 🟡 Orta  
**Sorun:** PPO modeli 1h cache kullanıyor, sert hareketlerde 30+ dakika eski veriyle karar veriyor.  
**Çözüm:** Shock=ARMED olduğunda PPO cache'ini invalidate edip yeniden hesaplama tetiklemek. Veya daha düşük TF (15m) bazlı bir PPO versiyonu eklemek.

### 5.6 Düşük — Consensus Mismatch Mantığını Gözden Geçir
**Öncelik:** 🟢 Düşük  
**Sorun:** `consensus_status=mismatch_transition` tüm dönem boyunca trade'i engelledi. Trend piyasalarında (ADX>50) consensus gereksiz yere kısıtlayıcı.  
**Çözüm:** ADX ve shock koşuluna bağlı, loglanabilir ve geri alınabilir bir gevşetme politikası uygula; global bypass kullanma.

---

## 6. Sonuç

Bot teknik olarak sorunsuz çalışıyordu (veri akışı, WebSocket, iterasyon döngüsü). Ancak **aşırı muhafazakar filtre zinciri** nedeniyle net bir bearish fırsatı değerlendiremedi. Ana sorunlar:

1. **Yavaş/kapalı-mum bazlı slow RSI** hızlı piyasa hareketlerinde geç tepki veriyor
2. **Çok yüksek Z-score eşiği** (1.90) ve kod içi alt limit (1.60) esnekliği azaltıyor
3. **Shock Detector çıktısı** olay anında trade kararlarına entegre değildi (Sprint-3 ile hedefli olarak entegre edildi)
4. **ML modelleri** (Gemma, Regime, PPO) bearish koşulları algılayamıyor

Bu analiz, bearish bot'un **reaktivite** ve **sinyal üretim kapasitesini** artırmak için öncelikli iyileştirme alanlarını ortaya koymaktadır.

## 7. Teknik Uygulama Durumu ve Backlog (Güncel)

**Konfig Uygulama Notu:** Bu projede strategy-seviyesi detay ayarlar Azure App Configuration üzerinden değil, deployment sırasında yüklenen bot config dosyası üzerinden yönetilir.

### 7.1 Sprint-1 (Düşük Risk, Config-Only Canary)

1. MR high-ADX Z eşiğini mevcut kod limitine kadar düşür:
- `strategies.mean_reversion.high_adx_z_threshold: 1.60`
- Beklenen etki: `Dynamic Z veto` oranının düşmesi.

2. Shock sırasında short veto engelini kaldır:
- `strategies.mean_reversion.regime_policy.shock.short_mode: off`
- Not: `disabled` yerine `off` kullan; `disabled` aktif veto üretir.

3. Güvenlik:
- Sadece canary sembolde (`BTC/USDT:USDT`) ve düşük allocation ile başlat.
- `transition_no_trade` global olarak kapatılmasın.

### 7.2 Sprint-2 (Kod Değişikliği: MR Eşik Esnekliği + Telemetri)

1. `high_adx_z_threshold` alt limitini konfigüre edilebilir yap:
- Dosya: `src/strategies/mean_reversion.py`
- Mevcut davranış: `max(1.60, high_adx_z_threshold)`
- Hedef: alt limitin config ile yönetilmesi (default davranış korunarak).

2. MR gate telemetrisi ekle:
- Dosya: `src/strategies/mean_reversion.py`
- Yeni telemetri alanları:
- `dynamic_z_passed`
- `entry_candidate_long`
- `entry_candidate_short`
- `entry_blocked_by_band`
- `entry_blocked_by_regime_policy`

3. Config şema ve örnek config güncelle:
- Dosya: `config/config.example.yaml`
- Dosya: `src/config/schema.py`

4. Testler:
- Dosya: `tests/unit/test_mean_reversion_high_adx_z_threshold.py`
- Komut: `pytest tests/unit/test_mean_reversion_high_adx_z_threshold.py`
- Komut: `pytest tests/test_mr_controller.py`

### 7.3 Sprint-3 (Kod Değişikliği: Hedefli RSI Shock Override) — Durum: ✅ Tamamlandı

1. Kod entegrasyonu tamamlandı:
- Dosya: `src/core/rsi_zone_router.py`
- Dosya: `src/core/production_coordinator.py`
- Mantık: `Shock=ARMED + min_shock_score + min_adx + canary_symbol + allow_strategies` sağlanırsa transition blokajı hedefli override edilir.
- Recheck (`dispatch_strategy`) ve ana döngü aynı kuralı kullanır.

2. Konfigürasyon ve validasyon tamamlandı:
- Dosya: `config/config.example.yaml`
- Dosya: `src/config/schema.py`
- Aktif blok: `strategies.rsi_zone_router.transition.shock_override`

3. Test kapsamı tamamlandı:
- Dosya: `tests/unit/test_rsi_zone_router.py`
- Dosya: `tests/unit/test_config_rsi_zone_router_validation.py`
- Dosya: `tests/unit/test_production_rsi_shock_override_dispatch.py`
- Çalıştırılan komut: `pytest tests/unit/test_rsi_zone_router.py tests/unit/test_config_rsi_zone_router_validation.py tests/unit/test_production_rsi_shock_override_dispatch.py -q`
- Sonuç: `38 passed`

### 7.4 Sprint-4 (Yeni Strateji Yolu: Shock Breakdown Short) — Durum: 🟡 Kodlandı, Canary Rollout Bekliyor

1. Ayrı strategy ekle (MR/OB/STR’den bağımsız):
- Dosya: `src/strategies/shock_breakdown_short.py` (uygulandı)
- Trigger: `Shock=ARMED`, breakdown teyidi, minimum momentum/volume.

2. Risk korumaları:
- Cooldown, tek-yön guard, sıkı stop, max hold.

3. Entegrasyon:
- Dosya: `scripts/live_trading_launcher.py` (strateji init/register)
- Dosya: `src/core/production_coordinator.py` (ana döngüde strateji çalıştırma)
- Dosya: `src/core/rsi_zone_router.py` (`shock_breakdown_short` için transition istisnası)
- Canary rollout ile devreye al.

### 7.5 Canary Rollout ve Kabul Kriterleri

1. Faz-A (Observe, 12 saat):
- Yeni kararlar sadece loglansın, trade açılmasın.

2. Faz-B (Canary Enforce, 24 saat):
- Sadece `BTC/USDT:USDT`, düşük allocation.
- Kill-switch koşulları: risk limit ihlali, ani artan churn, beklenmeyen tekrar sinyal.

3. Faz-C (Kademeli Yayılım):
- 1 sembolden 3 sembole, sonra kademeli genişleme.

4. Başarı ölçütleri:
- `Dynamic Z veto` oranında anlamlı düşüş.
- Trade sayısı artarken risk metriklerinin bozulmaması.
- `same_signal_repeat_rate` ve erken çıkış oranlarında kontrolsüz artış olmaması.

### 7.6 Operasyonel Rollout Checklist (Sprint-3 + Sprint-4 Sonrası)

1. Preflight (Deploy Öncesi)
- [ ] `strategies.rsi_zone_router.transition.shock_override.enabled=true`
- [ ] `mode=observe` ile başlanacak (ilk fazda enforce yok).
- [ ] `canary_symbols` yalnız `BTC/USDT:USDT` olarak set edildi.
- [ ] `allow_strategies` yalnız hedef stratejileri içeriyor (`adaptive_str` vb.).
- [ ] `min_score` ve `min_adx` risk komitesi tarafından onaylandı.
- [ ] Test paketi geçti (RSI override): `pytest tests/unit/test_rsi_zone_router.py tests/unit/test_config_rsi_zone_router_validation.py tests/unit/test_production_rsi_shock_override_dispatch.py -q` (güncel sonuç: `38 passed`)
- [ ] `strategies.shock_breakdown_short.enabled=true`
- [ ] `strategies.shock_breakdown_short.rollout.mode=observe`
- [ ] `strategies.shock_breakdown_short.rollout.canary_symbols` yalnız `BTC/USDT:USDT` olarak set edildi.
- [ ] Test paketi geçti (Sprint-4): `pytest tests/unit/test_shock_breakdown_short_strategy.py tests/unit/test_config_shock_breakdown_short_validation.py -q`

2. Faz-A (Observe, 12 saat)
- [ ] `rsi_router.observe_would_override_transition` event sayımı aktif izleniyor.
- [ ] `transition_no_trade` blok sayımı düşmeden önce davranış farklılaşması sadece log seviyesinde doğrulandı.
- [ ] Beklenmeyen sinyal patlaması yok (`same_signal_repeat_rate`, churn, false trigger).
- [ ] `shock_breakdown_short_observe` event sayımı aktif izleniyor.
- [ ] `shock_breakdown_short` adaylarında cooldown/tekrar sinyal davranışı beklenen limitte.

3. Faz-B (Canary Enforce, 24 saat)
- [ ] `mode=enforce` yalnız canary sembolde açıldı.
- [ ] `rsi_router.transition_shock_override` reason kodu beklenen sıklıkta görülüyor.
- [ ] PnL dağılımı, win-rate, max adverse excursion ve erken çıkış oranları takip ediliyor.
- [ ] Kill-switch koşulları için on-call kişi ve karar yetkisi net.
- [ ] `strategies.shock_breakdown_short.rollout.mode=enforce` yalnız canary sembolde açıldı.
- [ ] `strategy.shock_breakdown_short.entry` reason kodu beklenen sıklıkta görülüyor.

4. Rollback Planı
- [ ] Anlık geri dönüş: `mode=off` veya `enabled=false`.
- [ ] Gerekirse `canary_symbols` boşaltılarak override kapsamı sıfırlanır.
- [ ] Rollback sonrası 30-60 dakika boyunca `transition_no_trade` ve sinyal hacmi normalleşmesi doğrulanır.
- [ ] Incident notu ve öğrenimler backlog'a işlenir.
- [ ] Sprint-4 acil geri dönüş: `strategies.shock_breakdown_short.enabled=false` veya `rollout.mode=off`.

---

*Rapor otomatik olarak oluşturulmuştur. bearish-bot container logları ve TradingView grafiğinden derlenmiştir.*
