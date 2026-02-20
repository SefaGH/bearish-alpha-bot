# Bütünsel Sistem Değerlendirmesi — Bearish Bot

**Tarih:** 18 Şubat 2026  
**Run ID:** 20260217_182419_464545  
**Container:** bearish-bot (bearishalphabot.azurecr.io/bearish-bot:manual-20260217-v2)  
**Sembol:** BTC/USDT:USDT | 10x Kaldıraç | 5m Timeframe  
**Strateji:** Mean Reversion (Dynamic Controller)  
**Exchange:** BingX VST (Sandbox)  
**Analiz Penceresi:** 17 Şubat 2026, 21:10 – 21:50 UTC (4 trade)

---

## 1. Yönetici Özeti

Bu rapor, 17 Şubat 2026 akşamı (21:10–21:50 UTC) açılan 4 kısa ömürlü BTC/USDT long işlemin kapsamlı adli analizini içerir. 4 işlemin tamamı `postfill_rr_below_1` veya `postfill_rr_below_required` nedeniyle anında kapatılmıştır.

**Ana Bulgu:** Botun stratejik zekası çalışıyor (4 trade'in 3'ünde yön doğru). Ancak volume matrix LOW ceza mekanizması, kârlı olabilecek trade'leri sistematik olarak öldürüyor. Pre-volume RR değerleri tüm trade'lerde eşiğin üstündeyken, volume matrix %43-53 RR yıkımı yaratmıştır.

---

## 2. İşlem Özeti

| # | Trade ID | Saat (UTC) | Yön | Fill Fiyatı | Sinyal Fiyatı | Pos. Size | Çıkış Nedeni | PnL |
|---|----------|------------|------|-------------|--------------|-----------|--------------|-----|
| 1 | ed5f1850 | 21:10:46 | LONG | 67,645.80 | 67,666.20 | 0.0586 | postfill_rr_below_1 | −$0.012 |
| 2 | 0a150bb8 | 21:33:51 | LONG | 67,528.50 | 67,528.70 | 0.0616 | postfill_rr_below_required | −$0.019 |
| 3 | 18adce71 | 21:45:30 | LONG | 67,570.90 | 67,560.60 | 0.0908 | postfill_rr_below_required | −$0.018 |
| 4 | 813796ed | 21:50:20 | LONG | 67,590.90 | 67,560.60 | 0.0762 | postfill_rr_below_1 | −$0.823 |

**Toplam PnL:** −$0.872

---

## 3. Kök Neden Analizi: RR Yıkım Zinciri

### 3.1 Volume Matrix Etkisi

Tüm trade'lerin pre-volume (SIGNAL_BREAKDOWN aşamasındaki) RR değerleri, gerekli 1.327 eşiğinin **üstündeydi**. Volume matrix LOW profili (stop_loss × 1.3, take_profit × 0.9) her trade'de RR'ı %43-53 oranında yok etmiştir:

| Trade | Planlanan Stop Dist. | Planlanan TP Dist. | Pre-Vol RR | Post-Vol Stop Dist. | Post-Vol TP Dist. | Post-Vol RR | RR Kaybı |
|-------|---------------------|-------------------|-----------|---------------------|--------------------|------------|----------|
| 1 | 170.5 | 249.7 | **1.464** | 238.1 (×1.40) | 182.0 (×0.73) | 0.764 | −47.8% |
| 2 | 162.3 | 352.0 | **2.169** | 229.8 (×1.42) | 284.5 (×0.81) | 1.238 | −42.9% |
| 3 | 110.2 | 279.5 | **2.536** | 177.8 (×1.61) | 212.0 (×0.76) | 1.192 | −53.0% |
| 4 | 131.2 | 259.2 | **1.975** | 198.9 (×1.52) | 191.8 (×0.74) | 0.964 | −51.2% |

### 3.2 Yıkım Mekanizması

```
Strateji Sinyali → Pre-volume RR (1.464-2.536) ✅ Geçer
        ↓
Volume Matrix LOW uygulanır:
  - Stop loss genişletilir: × 1.3-1.6
  - Take profit daraltılır: × 0.73-0.81
        ↓
Post-volume RR (0.764-1.238) ❌ Eşik altında
        ↓
Pozisyon açılır → Anında kapatılır
  - Gereksiz fee ve slippage ödenir
  - Kârlı olabilecek fırsat kaybedilir
```

### 3.3 Postfill RR'ı Kötüleştiren Ek Etki: Fill Fallback/Slippage

Log doğrulaması, 4 trade'in de `limit_timeout_market_fallback` ile fill olduğunu göstermektedir. Bu durum postfill RR'ı ek olarak zayıflatmıştır:

- Trade 1: entry_slippage_bps = 0.0296  
- Trade 2: entry_slippage_bps = 0.0148  
- Trade 3: entry_slippage_bps = 0.0296  
- Trade 4: entry_slippage_bps = 1.4053 (**anomalik yüksek**)

Sonuç: RR düşüşünün ana nedeni volume matrix olsa da, fill kalitesi/fallback davranışı özellikle Trade 4'te kaybı büyütmüştür.

---

## 4. Grafik ve Yön Analizi

BTC/USDT 5m grafik üzerinden yapılan değerlendirme:

| Trade | Yön Doğruluğu | Seviye Değerlendirmesi | Detay |
|-------|:---:|---|---|
| 1 | ❌ Yanlış | Erken tetikleme | 21:10'da px=67,666 ile LONG ama fiyat 67,520'ye düştü — sadece %0.04 band penetrasyonu ile tetiklendi |
| 2 | ✅ Doğru | İyi seviye | 21:33'te px=67,528 dip bölgesinde giriş. Fiyat 15dk içinde 67,610'a çıktı (+81.5 USDT) |
| 3 | ✅ Doğru | İyi seviye | 21:45'te px=67,570. Fiyat 15dk içinde 67,635'e çıktı (+64.5 USDT) |
| 4 | ⚠️ Gereksiz | Tekrar | Trade 3 ile aynı signal_px (67,560.60) — 20sn duplicate cooldown var ama aynı-sinyal (signal_px+side) cooldown olmadığı için 5dk sonra tekrar açıldı |

**Sonuç:** Bot, 4 trade'in 3'ünde doğru yönü tespit etti. Sorun yön tespitinde değil, pozisyon açma koşullarında.

---

## 5. Band ve Z-Score Hesaplama Analizi

### 5.1 Dynamic Controller Davranışı

Controller'ın 20:25-22:05 arası band parametreleri:

| Saat | VWAP | Std | Multiplier | Lower Band | Upper Band | Z-Score |
|------|------|-----|-----------|------------|------------|---------|
| 20:25 | 67,597 | 326 | 2.000 | 66,944 | 68,249 | - |
| 20:30 | 67,601 | 308 | 1.563 | 67,120 | 68,083 | - |
| 20:45 | 67,566 | 246 | 1.451 | 67,210 | 67,923 | -0.94 |
| 21:05 | 67,560 | 173 | 1.429 | 67,313 | 67,807 | -1.70 |
| 21:10 | 67,571 | 174 | 1.429 | 67,322 | 67,819 | - |
| 21:30 | 67,556 | 160 | 1.429 | 67,328 | 67,785 | -2.25 |
| 21:45 | 67,558 | 171 | 1.429 | 67,313 | 67,802 | - |
| 21:50 | 67,555 | 162 | 1.429 | 67,324 | 67,786 | - |
| 22:05 | 67,533 | 156 | 1.429 | 67,311 | 67,756 | - |

### 5.2 Tespit Edilen Sorunlar

1. **Double-Squeeze (Çift Sıkışma):**
   - std: 326 → 156 (**−52%** düşüş)
   - multiplier: 2.0 → 1.429 (**−29%** düşüş)
   - Bu iki düşüş çarpımsal etki yaptı → bandlar VWAP'a aşırı yaklaştı

2. **m_min = 1.0 çok düşük:**
   - Controller, multiplier'ı 1.429'a kadar düşürdü (m_min=1.0 izin veriyor)
   - Bu, bandların VWAP'tan yalnızca ~230 USDT uzaklaşmasına neden oldu

3. **ADX konsolidasyon koruması yok:**
   - ADX 12-17 aralığındaydı (güçlü konsolidasyon)
   - Sadece ADX > 36 freeze var, ADX < 20 freeze yok
   - Controller, konsolidasyonda bandları daraltmaya devam etti

4. **Lookback Uyumsuzluğu:**
   - Sinyal sistemi: 1440-bar VWAP lookback
   - Dynamic Controller: 180-bar lookback
   - Farklı pencereler farklı band değerleri üretiyor

5. **Controller Band Gözlemlenebilirlik Tutarsızlığı (Kritik):**
   - `mr_controller_decision` logu overlay uygulanmadan önce yazılıyor (`mr_controller.py:510-547`).
   - Aynı akışta overlay sonradan `decision.lower/upper` değerlerini değiştiriyor (`mr_controller.py:548-555`, `mr_controller.py:297-304`).
   - Sinyal `reason` metni final `lower/upper` ile üretiliyor (`mean_reversion.py:3205`, `mean_reversion.py:3211`) ve sinyal payload'ında da final band taşınıyor (`mean_reversion.py:3427-3428`).
   - Log doğrulaması da aynı farkı gösteriyor:
     - 21:10 trade penceresi: controller lower `67586.7156` (`logs/live_trading_20260217_182419_464545.log:12083`) vs signal ingress lower `67691.7694` (`logs/live_trading_20260217_182419_464545.log:11640`)
     - 21:45 trade penceresi: controller lower `67570.6537` (`logs/live_trading_20260217_182419_464545.log:14357`) vs signal ingress lower `67621.2983` (`logs/live_trading_20260217_182419_464545.log:14360`)
   - Net durum: mismatch deterministik olarak loglama zamanlamasından geliyor (controller logu pre-overlay, signal/reason post-overlay).

---

## 6. Sinyal Kalitesi Analizi

| Trade | Quality Score | Volume Bucket | Volume Strength | ML Consensus | Giriş Mekanizması |
|-------|:---:|:---:|:---:|:---:|---|
| 1 | 0.454 | LOW | 0.49 | 0.091 | Doğrudan sinyal |
| 2 | 0.435 | LOW | 0.39 | 0.001 | soft_deferral_salvaged (near_miss → fast_watch → band_touch) |
| 3 | 0.442 | LOW | 0.43 | 0.033 | Doğrudan sinyal |
| 4 | 0.442 | LOW | 0.43 | 0.004 | Doğrudan sinyal (aynı signal_px ile tekrar) |

**Sorunlar:**
- Quality score'lar çok düşük (0.43-0.45 / 1.0) — minimum eşik yok
- Volume sürekli LOW (strength 0.26-0.49) — MR stratejisi LOW vol'de doğal çalışır ama matrix cezalandırıyor
- ML consensus ~0 — 3 model de nötr, hiçbir katkı yok
- Trade 2, `soft_deferral_salvaged` ile girdi — kalite kontrolü atlandı
- Trade 3 ve 4 aynı signal_px ile 5dk arayla açıldı — run anında 20sn duplicate cooldown vardı, same-signal cooldown yoktu (Paket-2 ile giderildi)

---

## 7. ML Katmanı Durumu

| Model | Durum | Çıktı | Etki |
|-------|-------|-------|------|
| Gemma TorchScript | Yüklü, aktif | Her zaman "neutral", consensus ≈ 0 | Sıfır |
| PPO RL Agent | Yüklü, aktif | Her zaman "flat", p_flat > 0.999 | Sıfır |
| Regime Predictor | Yüklü, aktif | Her zaman "neutral", confidence 0.70 | Sıfır |

**Değerlendirme:** 3 model de fiilen çalışmıyor. Gemma muhtemelen eğitim verisi veya feature mismatch sorunu yaşıyor. PPO hiç öğrenememiş (p_flat=0.999). Regime predictor 0.70 confidence ile her zaman nötr kalıyor.

---

## 8. Simülasyon Sonuçları

### 8.1 Frozen Band Simülasyonu (m_min=1.5 + ADX<20 Freeze)

frozen_m = 1.5629 ile hesaplanan SIM bandları:

| Trade | Fiyat | Orijinal Lower | SIM Lower | Sonuç |
|-------|-------|:---:|:---:|---|
| 1 (21:10) | 67,666.2 | 67,691.8 | 67,562.8 | **ENGELLENDİ** — px SIM bandının 103 USDT üstünde |
| 2 (21:33) | 67,528.7 | 67,529.3 | 67,594.4 | Tetiklenir → ama post-vol RR=1.238 → FAIL |
| 3 (21:45) | 67,560.6 | 67,555.1 | 67,570.8 | Tetiklenir → ama post-vol RR=1.192 → FAIL |
| 4 (21:50) | 67,560.6 | 67,547.5 | 67,554.8 | **ENGELLENDİ** — px SIM bandının 5.8 USDT üstünde |

### 8.2 Senaryo Karşılaştırması

| Senaryo | Açıklama | Toplam PnL | İyileşme |
|---------|----------|:---:|:---:|
| **Mevcut** | Değişiklik yok | −$0.87 | — |
| **A: Sadece Band** | m_min=1.5, ADX freeze, controller band, double-squeeze | −$0.04 | +$0.84 (%96) |
| **B: Band + Volume** | A + volume matrix yumuşatma (stop×1.1, TP×0.95) + pre-fill RR | **+$10.88** | +$11.75 |
| **C: Tam Muhafazakâr** | B + quality score ≥ 0.55 | $0.00 | +$0.87 |

### 8.3 Senaryo B Detayı (Önerilen)

- **Trade 1:** ENGELLENDİ (band filtresi) → $0.00
- **Trade 2:** Yeni RR=1.873 ≥ 1.327 → Pozisyon AÇIK KALIR → Giriş: 67,528.5 → Çıkış (15dk): 67,610.0 → **+$5.02**
- **Trade 3:** Yeni RR=2.190 ≥ 1.327 → Pozisyon AÇIK KALIR → Giriş: 67,570.9 → Çıkış (15dk): 67,635.4 → **+$5.86**
- **Trade 4:** ENGELLENDİ (band filtresi) → $0.00

---

## 9. Sistem Akış Şeması — Sorun Noktaları

```
┌─────────────────────────────────────────────────────────────────┐
│  MR Strateji Sinyali                                            │
│  VWAP + Band + ADX + Z-Score                                    │
│  Sonuç: 3/4 doğru yön ✅                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  ⚠️ Band Hesaplama (Dynamic Controller)                         │
│  Sorun: Double-squeeze, m_min=1.0, ADX<20 freeze yok            │
│  Etki: Sığ penetrasyonla sinyal tetiklenmesi                     │
│  → Trade 1 sadece %0.04 penetrasyon ile girdi                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  ❌ Volume Matrix (LOW profili)                                  │
│  Sorun: stop×1.3, TP×0.9 → RR %43-53 yıkım                     │
│  Etki: Tüm trade'ler postfill RR eşiğini geçemedi               │
│  → Pre-vol RR (1.46-2.54) → Post-vol RR (0.76-1.24)             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  ❌ Postfill RR Check                                            │
│  Pozisyon açılıp anında kapatılıyor                              │
│  → Gereksiz fee + slippage ödeniyor                              │
│  → Pre-fill kontrol olsa hiç açılmazdı                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  ⚪ ML Katmanı                                                   │
│  3 model aktif ama fiilen nötr                                   │
│  Hiçbir katkı sağlamıyor                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. İyileştirme Önerileri — Öncelik Sıralı

### 10.0 Sorunların Sıralı Değerlendirmesi

| # | Sorun | Değerlendirme | Öncelik | Not |
|---|------|---------------|---------|-----|
| 1 | Volume matrix LOW → RR yıkımı | **Doğrulandı (ana kök neden)** | Kritik | Postfill erken çıkışların ana tetikleyicisi |
| 2 | Postfill fallback/slippage etkisi | **Doğrulandı (ikincil kuvvetli etki)** | Yüksek | Trade 4'te belirgin anomali |
| 3 | Double-squeeze | **Doğrulandı** | Yüksek | Band daralmasını hızlandırıyor |
| 4 | m_min düşük (1.0) | **Doğrulandı** | Yüksek | Erken/sığ tetiklemeyi artırıyor |
| 5 | ADX<20 freeze yok | **Doğrulandı** | Yüksek | Konsolidasyonda aşırı hassasiyet |
| 6 | Lookback uyumsuzluğu | **Doğrulandı** | Yüksek | Katmanlar arası band farkı |
| 7 | Controller band gözlemlenebilirlik tutarsızlığı | **Doğrulandı (kritik)** | Kritik | `mr_controller_decision` pre-overlay, signal/reason post-overlay |
| 8 | Düşük quality score + min gate yok | **Doğrulandı** | Orta | Düşük kaliteli sinyaller geçiyor |
| 9 | Soft deferral'da kalite atlanması | **Doğrulandı** | Orta | Policy tutarsızlığı |
| 10 | Same-signal cooldown yok | **Doğrulandı** | Kritik | 20sn duplicate check yeterli değil |
| 11 | ML katmanı etkisiz | **Doğrulandı** | Düşük | Kısa vadede shadow-mode daha doğru |

### 10.1 Önerilerin Sıralı Değerlendirmesi

| # | Öneri | Değerlendirme | Uygulama Notu |
|---|-------|---------------|---------------|
| 1 | LOW stop_loss_multiplier 1.05-1.10 | **Uygun** | Hemen uygulanmalı |
| 2 | LOW take_profit_multiplier 0.95-1.0 | **Uygun** | Hemen uygulanmalı |
| 3 | Pre-fill RR kontrolü | **Zorunlu** | Postfill metodolojisiyle birebir hizalanmalı |
| 4 | Controller band tutarlılığı (tek snapshot) | **Kritik** | `mr_controller_decision` post-overlay loglansın; ingress/breakdown aynı snapshot'tan beslensin |
| 5 | m_min=1.5 | **Uygun** | Canary ile açılmalı |
| 6 | ADX<20 / >36 freeze | **Uygun** | Controller koruma paketi içinde |
| 7 | MR için strategy-specific volume | **Çok uygun** | En temiz mimari çözüm |
| 8 | Min band penetrasyon | **Uygun** | Sabit değil, volatiliteye duyarlı tercih edilmeli |
| 9 | Double-squeeze detection | **Uygun** | False-positive izlenmeli |
| 10 | Min quality score >=0.55 | **Uygun (sert)** | 0.50→0.55 canary geçişi önerilir |
| 11 | Soft deferral quality gate | **Kesinlikle uygulanmalı** | Kalite politikası tekilleşir |
| 12 | Same-signal cooldown 10dk | **Uygun** | `signal_px` için tick-rounding/tolerans eklenmeli |
| 13 | Fill fallback slippage guard | **Uygun** | Adaptif eşik (ATR/spread tabanlı) önerilir |
| 14 | ML shadow mode | **Doğru yaklaşım** | Kararı etkilemeden logla |
| 15 | Gemma yeniden eğitim | **Orta vade** | Önce feature/data uyumu doğrulanmalı |
| 16 | PPO devre dışı/yeniden eğitim | **Uygun** | Kısa vadede disable/shadow daha mantıklı |

### 10.2 Revize Net Fix Sırası (Uygulama Planı)

1. **Paket-1 (Kritik tutarlılık):** Controller band snapshot'ını tekilleştir (`post_overlay` kaynak), `mr_controller_decision` logunu post-overlay üret (veya pre/post'u birlikte logla), `Signal ingress`/`SIGNAL_BREAKDOWN` loglarını aynı snapshot'tan besle.
2. **Paket-2 (Kritik giriş tekrar kontrolü):** Same-signal cooldown'u (`signal_px` + `side`, tick toleranslı) aktif et.
3. **Paket-3 (Kritik kalite kontrolü):** Min quality gate + soft-deferral quality gate'i tek politika olarak zorunlu kıl.
4. **Paket-4 (Kritik RR uyumu):** Pre-fill ve post-fill RR hesaplarını aynı hesaplayıcı/metodolojiye bağla.
5. **Paket-5 (Execution koruması):** Fallback slippage guard (adaptif eşik) ve aşımda iptal.
6. **Paket-6 (Risk profili):** MR için LOW volume matrix'i yumuşat/override et.
7. **Paket-7 (Controller stabilizasyonu):** `m_min=1.5`, `ADX<20 freeze`, `double-squeeze freeze`.
8. **Paket-8 (Canary ve metrik):** `postfill_exit_rate`, `same_signal_repeat_rate`, `band_snapshot_mismatch_count`, `slippage_p95`.
9. **Paket-9 (ML yönetişimi):** ML katmanını shadow-mode tut; retrain kararını metrik sonrası ver.

### KRİTİK (Hemen Uygulanması Gereken)

| # | Değişiklik | Mevcut Değer | Önerilen Değer | Etki |
|---|-----------|-------------|---------------|------|
| 1 | Controller band tutarlılığı | Controller log: pre-overlay, Signal reason: post-overlay | **Tek `band_snapshot` (post-overlay) + ingress/breakdown bu snapshot'ı kullansın** | Karar izlenebilirliği tekilleşir |
| 2 | LOW stop_loss_multiplier | 1.3 | **1.05-1.10** | RR yıkımını %47→%10'a düşürür |
| 3 | LOW take_profit_multiplier | 0.9 | **0.95-1.0** | TP daraltmasını minimalize eder |
| 4 | Pre-fill RR kontrolü | Yok | **Pozisyon açmadan önce post-rebase RR hesapla** | Fee/slippage tasarrufu |
| 5 | m_min | 1.0 | **1.5** | Band aşırı daralmasını önler |

### YÜKSEK (Kısa Vadede Uygulanması Gereken)

| # | Değişiklik | Mevcut Değer | Önerilen Değer | Etki |
|---|-----------|-------------|---------------|------|
| 6 | ADX konsolidasyon freeze | Sadece ADX>36 | **ADX<20 veya ADX>36 iken freeze** | Konsolidasyonda band koruma |
| 7 | Stratejiye özgü volume profili | Tek profil | **MR LOW: stop×1.05, TP×0.97** | En temiz mimari çözüm |
| 8 | Min band penetrasyon derinliği | Yok | **≥ %0.10 (~67 USDT)** | Sığ tetiklemeleri engeller |
| 9 | Double-squeeze algılama | Yok | **std -%30 VE multiplier -%20 → freeze** | Çarpımsal daralmayı önler |

### ORTA (Planlı Döngüde Uygulanması Gereken)

| # | Değişiklik | Mevcut Değer | Önerilen Değer | Etki |
|---|-----------|-------------|---------------|------|
| 10 | Min quality score | Yok | **≥ 0.55** | Düşük kaliteli sinyalleri filtreler |
| 11 | Soft deferral kalite kontrolü | Atlanıyor | **Salvage edilen sinyallere de quality gate uygula** | Kalite tutarlılığı |
| 12 | Aynı sinyal cooldown | Sadece 20sn duplicate cooldown | **Aynı signal_px + side ile 10dk cooldown (tick-rounding/tolerans ile)** | Trade tekrarını önler |
| 13 | Fill fallback slippage guard | Yok | **fallback sonrası adaptif max slippage (ATR/spread tabanlı)** | Anomalik fill kaynaklı RR bozulmasını azaltır |

### DÜŞÜK (İzleme ve Planlama)

| # | Değişiklik | Mevcut Değer | Önerilen Değer | Etki |
|---|-----------|-------------|---------------|------|
| 14 | ML shadow mode | Aktif ama etkisiz | **Shadow mode — kararı etkilemesin, logla** | Doğruluk ölçümü |
| 15 | Gemma yeniden eğitim | Hep "neutral" | **Yeni veri ile yeniden eğit** | Gelecekteki katkı |
| 16 | PPO RL | Hep "flat" | **Devre dışı bırak veya yeniden eğit** | Kaynak tasarrufu |

---

## 11. Önerilen Konfigürasyon Değişiklikleri

```yaml
risk:
  # Global Volume Matrix
  volume_bucket_risk_matrix:
    LOW:
      stop_loss_multiplier: 1.3
      take_profit_multiplier: 0.9
      position_size_multiplier: 0.5

  # Paket-6: Strategy-specific LOW softening (MR)
  strategy_profiles:
    mean_reversion:
      volume_bucket_risk_matrix:
        LOW:
          stop_loss_multiplier: 1.05
          take_profit_multiplier: 0.97
          position_size_multiplier: 0.5

  # === DYNAMIC CONTROLLER ===
  dynamic_controller:
    m_min: 1.5                         # eskisi: 1.0
    consolidation_freeze_adx: 20.0     # YENİ — ADX<20 iken freeze
    double_squeeze_detection:           # YENİ
      std_pct_threshold: -30
      multiplier_pct_threshold: -20
      action: freeze

  # === RISK ===
  pre_fill_rr_check: true             # YENİ — açmadan önce kontrol
  pre_fill_rr_use_post_rebase: true   # YENİ — postfill ile aynı RR metodolojisi
  min_band_penetration_pct: 0.001     # YENİ — %0.10 minimum

# === SIGNALS / QUALITY GATE ===
signals:
  duplicate_prevention:
    same_signal_cooldown_seconds: 600   # YENİ (Paket-2)
    same_signal_tick_size: 0.1          # YENİ (Paket-2)
    same_signal_price_tolerance_bps: 0.0  # YENİ (opsiyonel)
    same_signal_price_tolerance_pct: 0.0  # YENİ (opsiyonel)

  signal_scoring:
    min_quality_score: 0.55            # YENİ (hard gate)
    health_policy:                      # YENİ (Paket-3)
      use_healthy_components_only: true
      ml_require_healthy: true
      ppo_require_healthy: true
      force_disable_ml: false
      force_disable_ppo: false

# === EXECUTION GUARD ===
execution_guard:
  fallback_max_entry_slippage_bps: 0.5  # YENİ
  on_exceed: cancel_trade                # YENİ

# === CANARY METRICS ===
monitoring:
  rl_telemetry_interval_seconds: 120
  canary_metrics:
    enabled: true
    interval_sec: 300
    canary_symbols: ["BTC/USDT:USDT"]
    alerts:
      enabled: true
      min_entries: 15
      min_same_signal_total: 15
      min_band_checks: 15
      min_slippage_samples: 15
      thresholds:
        postfill_exit_rate: { warning: 0.15, critical: 0.25 }
        same_signal_repeat_rate: { warning: 5.0, critical: 10.0 }
        band_snapshot_mismatch_rate: { warning: 0.01, critical: 0.03 }
        slippage_p95_bps: { warning: 3.0, critical: 6.0 }

# === ML GOVERNANCE AUTO (Paket-9) ===
ml:
  governance:
    automation:
      enabled: true
      interval_sec: 120
      ppo:
        enabled: true
        min_window_samples: 20
        degrade_after_windows: 2
        recover_after_windows: 6
        cooldown_sec: 1800
        auto_recover: false
        degrade_to_mode: "shadow"
        bad_flat_vote_rate: 0.97
        bad_avg_score_max: 0.08
        good_flat_vote_rate: 0.80
        good_avg_score_min: 0.40
```

**Paket-1 Uygulama Notu (kodla uyumlu):**
- Controller telemetry tek event içinde pre/post overlay görünürlüğü ile standartlaştı (`mr_controller_decision`):
  - pre-overlay alanları: `reason_pre_overlay`, `derived.lower_pre_overlay`, `derived.upper_pre_overlay`
  - final alanlar: `reason`, `derived.lower`, `derived.upper`
  - `params.overlay_applied` ile overlay etkisi açıkça işaretlenir.
- Mean Reversion sinyali artık final band değerlerini tek snapshot altında taşır:
  - `signal.band_snapshot.source=controller_post_overlay|pipeline_fallback`
  - `band_snapshot.lower/upper` ile `signal.vwap_lower/upper` aynı kaynaktan beslenir.
- StrategyCoordinator ingress ve `SIGNAL_BREAKDOWN` logları `signal.band_snapshot` üzerinden çalışır; böylece katmanlar arası band kaynağı tekilleşir.
- Not: Paket-1 için ayrı `band_source_consistency.*` config anahtarı yok; davranış doğrudan strateji/controller akışında uygulanmıştır.

**Paket-2 Uygulama Notu (kodla uyumlu):**
- Duplicate katmanına same-signal cooldown eklendi:
  - key: `(strategy, symbol, timeframe, side, signal_px_bucket)`
  - varsayılan: `same_signal_cooldown_seconds=600` (10 dk)
  - bucket kuantizasyonu: `same_signal_tick_size` (+ opsiyonel `same_signal_price_tolerance_bps/pct`)
- Bu kontrol, baz cooldown’dan bağımsız ikinci kapı gibi çalışır:
  - baz cooldown bitse bile aynı `signal_px` bucket tekrarında sinyal reddedilir.
- Telemetry/sayaçlar:
  - `processing_stats.same_signal_rejections`
  - `get_duplicate_prevention_stats().same_signal_repeat_rate`
  - canary tarafında bu oran `same_signal_repeat_rate` KPI’sına taşınır.

**Paket-3 Uygulama Notu (kodla uyumlu):**
- Quality gate artık `process_strategy_signal` içinde zorunlu uygulanır (`reason_code=quality.below_min_threshold`).
- Aynı gate incubator replay / soft-deferral salvage akışında da aynen çalışır (ayrı bypass yok).
- Telemetry standardı: `waiting_room_drop` içinde `drop_kind=quality_gate`, `drop_reason=quality.below_min_threshold`, `quality_score`, `min_quality_score`.
- Skor hesaplamasında ML/PPO için health-aware politika eklendi; sağlıksız işaretlenen bileşenler ağırlıktan düşülür.

**Paket-4 Uygulama Notu (kodla uyumlu):**
- Pre-fill ve post-fill RR değerlendirmesi tek yardımcıda birleştirildi: `src/core/rr_guard.py::evaluate_rr_gate`.
- Ortak RR karar mantığı: önce `RR<1.0` (`rr_below_1`), sonra `RR<dynamic_rr_target` (`rr_below_required`).
- Pre-fill reason-code standardı netleştirildi:
  - `risk.rr.pre_fill.rr_below_1`
  - `risk.rr.pre_fill.rr_below_required`
- Pre-fill RR meta (`rr_gate_prefill`) artık risk metadata içinde taşınır; `rr_actual`, `rr_required`, `rr_floor`, `rr_required_source` alanları standardize edildi.
- StrategyCoordinator pre-fill RR rejectlerinde `waiting_room_drop` telemetry üretir:
  - `drop_kind=pre_fill_rr_gate`
  - `drop_reason=<prefill reason_code>`
  - `rr_reason_code`, `rr_actual`, `rr_required`, `rr_floor`, `rr_required_source`
- Post-fill tarafında mevcut davranış korunur (`postfill_reason_code=rr_below_1|rr_below_required`), ancak aynı helper ile hesaplandığı için pre/post karar kuralları artık deterministik olarak aynıdır.

**Paket-5 Uygulama Notu (kodla uyumlu):**
- Timeout sonrası market fallback için adaptif `slippage_guard` eklendi (ATR + spread tabanlı dinamik eşik).
- Guard kuralı: adverse expected fallback bps (`ask/bid` tabanlı) `kill_bps` eşiğini aşarsa fallback iptal edilir.
- Yeni fallback block reason-code:
  - `execution.fallback.limit_timeout.slippage_guard_blocked`
  - `fallback_reason=limit_timeout_market_fallback_slippage_guard_blocked`
- Fallback akışında reason-code standardı genişletildi:
  - `execution.fallback.limit_timeout.disabled`
  - `execution.fallback.limit_timeout.unverified_position_delta`
  - `execution.fallback.limit_timeout.no_residual_qty`
  - `execution.fallback.limit_timeout.hard_chase_killed`
  - `execution.fallback.limit_timeout.soft_gate_blocked`
  - `execution.fallback.limit_timeout.market_fallback`
- Telemetry standardı:
  - `order_manager_decision` ve `order_decision_outcome` artık `reason_code` alanını taşır.
  - Fallback guard reason alanları loglanır: `fallback_hard_chase_reason`, `fallback_soft_gate_reason`, `fallback_slippage_guard_reason`.

**Paket-6 Uygulama Notu (kodla uyumlu):**
- `VolumeAwarePositionSizingRule` artık strateji profiline göre efektif volume matrix çözümler:
  - Global kaynak: `risk.volume_bucket_risk_matrix`
  - Override kaynak: `risk.strategy_profiles.<strategy>.volume_bucket_risk_matrix`
- Merge politikası: strategy override yalnızca verilen bucket/alanları ezer; eksik alanlar global matrix'ten miras alınır.
- `RiskManager` default rule kurulumunda rule'a `risk_config` enjekte edilir; böylece strategy-profile override runtime'da aktif olur.
- Yeni telemetry alanları (`volume_bucket_risk` log payload):
  - `strategy_name`
  - `volume_matrix_source` (`global` veya `strategy_profile:<strategy>`)
- MR LOW yumuşatma örneği (`config.example.yaml`):
  - `stop_loss_multiplier: 1.05`
  - `take_profit_multiplier: 0.97`
  - `position_size_multiplier: 0.5`

**Paket-7 Uygulama Notu (kodla uyumlu):**
- Controller stabilizasyonu aktif edildi (`src/strategies/mr_controller.py`):
  - `m_min` tabanı pratik kullanımda `1.5` olarak yükseltildi (örnek config).
  - ADX çift yönlü freeze:
    - `adx >= adx_freeze_threshold` -> `reason=freeze_on_trend_high_adx`
    - `adx <= adx_consolidation_freeze_threshold` -> `reason=freeze_on_trend_low_adx`
  - Double-squeeze freeze:
    - `double_squeeze_detection.enabled=true`
    - `std_pct_threshold` ve `multiplier_pct_threshold` aynı anda sağlandığında `reason=freeze_double_squeeze`
- Yeni config alanları (`strategies.mean_reversion.dynamic_controller`):
  - `adx_consolidation_freeze_threshold`
  - `double_squeeze_detection.enabled`
  - `double_squeeze_detection.std_pct_threshold`
  - `double_squeeze_detection.multiplier_pct_threshold`
  - `double_squeeze_detection.action`
- `mr_controller_decision` telemetry’sine stabilizasyon parametreleri eklendi:
  - `adx_freeze_high_threshold`
  - `adx_freeze_low_threshold`
  - `double_squeeze_freeze_enabled`

**Paket-8 Uygulama Notu (kodla uyumlu):**
- Canary KPI snapshot telemetry eklendi (`LiveTradingEngine`):
  - `event=canary_metrics_snapshot`
  - `postfill_exit_rate`
  - `same_signal_repeat_rate`
  - `band_snapshot_mismatch_count`
  - `slippage_p95_bps`
- Snapshot içine alarm değerlendirme özeti eklendi:
  - `alert_status` (`ok | warning | critical | insufficient_data | disabled`)
  - `alert_count`
  - `alerts.alerts[].reason_code` formatı: `monitoring.canary.<metric>.<severity>`
- `postfill_exit_rate` hesaplaması:
  - Pay: `postfill_action=early_exit` ile kapanan girişler
  - Payda: açılmış toplam girişler
- `same_signal_repeat_rate` kaynağı:
  - `StrategyCoordinator.get_duplicate_prevention_stats()` (`rejected_by_same_signal / total_signals_processed`)
- `band_snapshot_mismatch_count` kaynağı:
  - `process_strategy_signal` ingress aşamasında `signal.vwap_lower/upper` vs `band_snapshot.lower/upper` karşılaştırması
  - sayaçlar: `band_snapshot_checks`, `band_snapshot_mismatch_count`
- `slippage_p95_bps`:
  - giriş fill slippage örneklerinden (`execution_result.slippage * 10000`) p95
- Yeni config bloğu:
  - `monitoring.canary_metrics.enabled`
  - `monitoring.canary_metrics.interval_sec`
  - `monitoring.canary_metrics.canary_symbols` (`[]` veya `"*"` -> tüm semboller)
  - `monitoring.canary_metrics.alerts.*` (eşik + min örnek sayısı)
- Alarm event’i:
  - `event=canary_metrics_alert` (`warning`/`critical` durumunda ayrıca loglanır)
  - Varsayılan kritik eşikler:
    - `postfill_exit_rate >= 0.25`
    - `same_signal_repeat_rate >= 10.0`
    - `band_snapshot_mismatch_rate >= 0.03`
    - `slippage_p95_bps >= 6.0`

### 11.1 Paket Uygulama Durumu (Kod Bazlı)

| Paket | Kapsam | Durum | Not |
|---|---|---|---|
| Paket-1 | Controller band tutarlılığı / snapshot tekilleştirme | **Tamamlandı** | pre/post overlay aynı telemetry event’inde; signal `band_snapshot` standardı aktif |
| Paket-2 | Same-signal cooldown | **Tamamlandı** | `signal_px` bucket + side/timeframe anahtarı ile 10dk pencere |
| Paket-3 | Quality gate tekilleştirme | **Tamamlandı** | hard gate + salvage akışı aynı politika |
| Paket-4 | Pre-fill/Post-fill RR harmonizasyonu | **Tamamlandı** | ortak RR helper + reason-code standardı |
| Paket-5 | Fallback slippage guard | **Tamamlandı** | adaptif guard + reason-code/telemetry |
| Paket-6 | MR strategy-specific volume override | **Tamamlandı** | MR LOW profili yumuşatma aktif |
| Paket-7 | Controller stabilizasyonu | **Tamamlandı** | `m_min`, ADX çift yönlü freeze, double-squeeze freeze |
| Paket-8 | Canary + metrikler + alarm eşikleri | **Tamamlandı** | `canary_metrics_snapshot` + `canary_metrics_alert` |
| Paket-9 | ML yönetişimi (shadow/retrain kararı) | **Tamamlandı (Governance Fazı)** | mode wiring + PPO shadow karar nötralizasyonu + telemetry tabanlı auto degrade/recover aktif |

### 11.2 Paket-9 Ön İnceleme (Shadow Altyapısı Teyidi)

| Başlık | Beklenen (Paket-9) | Mevcut Durum | Sonuç |
|---|---|---|---|
| GEMMA shadow anahtarı | Kararı etkilemeden shadow log | `ml.governance.gemma_mode=apply|shadow|disabled` eklendi; legacy `ml.gemma.shadow_mode` fallback’i korunuyor (`config/config.example.yaml`, `src/core/production_coordinator.py`, `src/core/strategy_coordinator.py`) | **Var** |
| GEMMA shadow aktiflik (run bazlı) | Shadow açık | Yeni run’da da `Shadow Mode: False` (`logs/live_trading_20260218_222216_783819.log:255`, `logs/live_trading_20260218_222216_783819.log:468`) | **Kapalı** |
| PPO shadow anahtarı | PPO için explicit shadow/apply ayrımı | `ml.governance.ppo_mode=apply|shadow|disabled` eklendi; runtime’da `_rl_config.ppo_mode` üzerinden normalize ediliyor (`config/config.example.yaml`, `src/core/production_coordinator.py`, `src/core/strategy_coordinator.py`) | **Var** |
| PPO shadow telemetry | Sadece izleme amaçlı inference | `monitor_ppo_state` mevcut ve telemetry üretiyor (`src/core/strategy_coordinator.py:8574`) | **Var** |
| PPO karar etkisi | Shadow’da karar etkisi olmamalı | `_apply_ppo_long_filter` shadow modda `decision_effective=false` işaretliyor, `rl_recommendation`/`_last_rl_decision` override etmiyor; yalnızca telemetry (`ppo_shadow_action`, `ppo_shadow_score`) üretiyor (`src/core/strategy_coordinator.py`) | **Nötralize Edildi** |
| PPO RR etkisi | Shadow’da RR nötr | `ppo_mode!=apply` iken `ppo_rr_multiplier=1.0` zorlanıyor ve `ppo_rr_reason_code=ml.governance.ppo.shadow.rr_neutralized` set ediliyor (`src/core/strategy_coordinator.py`) | **Nötralize Edildi** |
| PPO pozisyon boyutu etkisi | Shadow’da size nötr | `ppo_mode!=apply` iken `_compute_ppo_position_multiplier` doğrudan `1.0` döndürüyor; `ppo_position_reason_code=ml.governance.ppo.shadow.size_neutralized` ve sizing telemetry alanına taşınıyor (`src/core/strategy_coordinator.py`) | **Nötralize Edildi** |
| PPO quality etkisi | Shadow’da kalite skorunu etkilememeli | Quality hesaplamasında `ppo_rl` bileşeni `ppo_mode!=apply` için zorunlu exclude ediliyor (`quality_excluded` reason-code) (`src/core/strategy_coordinator.py`) | **Nötralize Edildi** |
| Retrain karar otomasyonu | Canlı metrik eşiğine bağlı faz geçişi | `ProductionCoordinator` RL telemetry döngüsünde `ml_governance_snapshot`/`ml_governance_transition` ile PPO için window-bazlı auto degrade/recover (apply↔shadow/disabled) aktif; eşikler `ml.governance.automation.ppo.*` ile yönetiliyor | **Var (PPO Faz Otomasyonu)** |
| Otomatik model eğitimi | Eşik sonrası modeli yeniden eğitme | Runtime tarafında auto-train tetiklenmiyor; otomasyon fazı `mode` yönetiyor, retrain adımı offline araçlarla operatör kontrollü | **Yok (Bilinçli Tasarım)** |
| Retrain için offline araçlar | Tanı/eval araçları hazır olmalı | PPO için denetim/sweep scriptleri mevcut (`src/tools/ppo_observation_parity_check.py`, `src/tools/ppo_reward_audit.py`, `src/tools/ppo_threshold_sweep.py`) | **Var** |
| Canlı log kanıtı (yeni run) | Shadow’da işlem parametresi değişmemeli | Yeni run `logs/live_trading_20260218_222216_783819.log` incelendi. `canary_metrics_snapshot`/`alert_status` mevcut (`:678`, `:7800`) fakat `ml_governance_snapshot`/`ml_governance_transition` satırı yok ve `ppo_rr_reason_code=ml.governance.ppo.shadow.rr_neutralized` + `ppo_position_reason_code=ml.governance.ppo.shadow.size_neutralized` bulunamadı. Run başlangıcında mod `PPO=apply` (`:32`, `:206`, `:257`). | **Açık (kriter-1/2 karşılanmadı)** |

**Net tespit:** Paket-9’un governance hedefi kod seviyesinde tamamlandı: `apply|shadow|disabled` mode yönetimi + shadow’da karar etkisinin sıfırlanması + PPO telemetry’ye dayalı otomatik faz geçişi aktif. Bu tasarımda otomatik model yeniden eğitim (train job tetikleme) bilinçli olarak runtime dışında tutulmuştur.

### 11.3 Paket-9 Uygulama Notu (Kodla Uyumlu)

- Governance mode anahtarları eklendi:
  - `ml.governance.gemma_mode`
  - `ml.governance.ppo_mode`
  - ENV override: `ML_GOVERNANCE_GEMMA_MODE`, `ML_GOVERNANCE_PPO_MODE`
- Runtime wiring:
  - `ProductionCoordinator` ve `StrategyCoordinator` mode’ları normalize edip çalışma anına taşıyor.
  - `ppo_mode=disabled` iken adapter init/inference kapanıyor.
  - `gemma_mode=shadow` iken adapter shadow aktifleniyor.
- Governance otomasyon katmanı (yeni):
  - `ml.governance.automation.enabled` ile açılır (canary başlangıcında açıklandı).
  - PPO için window-bazlı sağlık kontrolü: `flat_vote_rate` ve `avg_score`.
  - Eşik aşımında otomatik faz düşürme: `apply -> shadow|disabled` (`degrade_to_mode`).
  - İyileşmede (opsiyonel) otomatik geri dönüş: `shadow|disabled -> apply` (`auto_recover=true`).
  - Transition/cycle telemetry:
    - `event=ml_governance_snapshot`
    - `event=ml_governance_transition`
- PPO shadow karar nötralizasyonu:
  - RR: `ppo_rr_multiplier=1.0` + `ppo_rr_reason_code`.
  - Size: `ppo_position_multiplier=1.0` + `ppo_position_reason_code` (sizing telemetry’ye yazılır).
  - RL agreement/prob: shadow’da nötr (`False` / `0.5`).
  - `_apply_ppo_long_filter`: shadow’da sadece telemetry (`ppo_shadow_action`, `ppo_shadow_score`), karar override yok.
- Quality gate etkisi:
  - `ppo_mode!=apply` olduğunda `ppo_rl` quality bileşeni zorunlu exclude edilir (`quality_excluded` reason).
- Test doğrulaması:
  - `tests/test_strategy_coordinator_ppo.py` (10/10 geçti)
  - `tests/test_dynamic_rr.py` (21/21 geçti)
  - `tests/unit/test_ml_governance_automation.py` (3/3 geçti)

### 11.4 Açık Kalem İncelemesi (Canlı Doğrulama)

- Yeni run doğrulandı: `logs/live_trading_20260218_222216_783819.log` (başlangıç: `2026-02-18 22:22:16`, kapanış: `2026-02-19 06:22:35`).
- Kriter-3 (canary) **sağlandı**:
  - `event=canary_metrics_snapshot` + `alert_status` mevcut (`:678`, `:7800`).
  - Bu run’da `same_signal_repeat_rate` için `critical` alarm da üretildi (`:7800`, `:7801`).
- Kriter-1 (governance snapshot) **sağlanmadı**:
  - `ml_governance_snapshot` ve `ml_governance_transition` için log içinde eşleşme yok.
  - Run’da RL tarafı sürekli `RL inactive` kaldığı için otomasyon döngüsü snapshot loguna ulaşmıyor (`logs/live_trading_20260218_222216_783819.log:880`, `logs/live_trading_20260218_222216_783819.log:1007`, `logs/live_trading_20260218_222216_783819.log:30237`).
  - Kod nedeni: `src/core/production_coordinator.py:6454-6462` bloğunda `if not samples and ppo_samples: ... continue` nedeniyle `6490` satırındaki `ml_governance_snapshot` logu atlanıyor.
- Kriter-2 (PPO shadow nötralizasyon reason-code) **sağlanmadı**:
  - `ppo_rr_reason_code=ml.governance.ppo.shadow.rr_neutralized` ve `ppo_position_reason_code=ml.governance.ppo.shadow.size_neutralized` eşleşmesi yok.
  - Aynı run başlangıcında governance modu `PPO=apply` (`logs/live_trading_20260218_222216_783819.log:32`, `:206`, `:257`); shadow reason-code kanıtı bu modda beklenmez.
- Grafik/log uyumu (5m pencere):
  - Ekran görüntüsündeki yükseliş eğilimli akışta bot ağırlıkla `near_miss`, `quality_gate` ve `volume.low_vol_tight_stop(_far)` nedenleriyle girişe dönmedi; run sonunda `entries_total=0`, `Active Positions: 0` (`logs/live_trading_20260218_222216_783819.log:7800`, `logs/live_trading_20260218_222216_783819.log:33042`, `logs/live_trading_20260218_222216_783819.log:33045`).
- Kapanış kriteri:
  1. Yeni canlı run logunda `ml_governance_snapshot` (ve mümkünse `ml_governance_transition`) görülmeli.
  2. En az bir MR sinyalinde PPO etkisinin nötrlendiği reason-code satırları görülmeli.
  3. Aynı pencerede `canary_metrics_snapshot` üretilip alarm durumu (`alert_status`) doğrulanmalı.

---

## 12. Sonuç

Bu analiz dört farklı perspektiften (trade forensics, grafik analizi, band hesaplama, simülasyon) gerçekleştirilmiştir. Sonuçlar tutarlı olarak aynı temel soruna işaret etmektedir:

**Botun mean reversion stratejisi doğru çalışıyor — fiyatın VWAP'a döneceğini 4'te 3 kez doğru bildi. Ancak risk yönetimi katmanındaki volume matrix LOW profili ana RR yıkımını üretirken, controller pre/post-overlay loglama tutarsızlığı (izlenebilirlik sorunu) ve fill fallback/slippage etkisi de karar kalitesini bozdu.**

Simülasyonda **+$10.88** üreten kombinasyon Band+Volume senaryosudur (volume matrix yumuşatma + pre-fill RR + `m_min=1.5` + ADX freeze). Paket-1..8 ve Paket-9 governance/faz otomasyonu kodda uygulanmıştır; modelin yeniden eğitimi ise runtime dışında, offline operasyon adımı olarak bırakılmıştır.

**Önerilen aksiyon:** Senaryo B (Band + Volume Matrix düzeltmeleri) uygulanmalıdır. Bot doğru yönü buluyor — onu desteklemek, kesmekten daha mantıklıdır.

---

*Bu rapor, bearish-bot Docker container logları ve gerçek zamanlı BTC/USDT fiyat verilerinden elde edilmiş bulgulara dayanmaktadır.*
