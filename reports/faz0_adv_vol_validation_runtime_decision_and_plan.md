# Faz-0 Doğrulama + Runtime Karar Notu + Faz-1/2 Uygulama Taslağı

Bu doküman, **advanced volatility (vol_*) entegrasyonunun fail-closed (production-safe)** çalıştığını kanıtlar, son runtime oturumu üzerinden **karar notu** çıkarır ve **Faz-1/Faz-2** için minimal PR taslağı sunar.

> Kapsam notu: Bu çalışma **stop/target modelini değiştirmez**. Sadece doğrulama + karar notu + plan.

---

## A) Faz-0 Doğrulama (Fail-Closed Smoke Tests)

### A1. Test kapsamı
Yeni unit test dosyası: [tests/unit/test_advanced_volatility_smoke.py](tests/unit/test_advanced_volatility_smoke.py)

Bu testler şunları kanıtlar:

- **Enabled=false ⇒ vol_* kolonları eklenmez**
- **Enabled=true + timeframe mismatch ⇒ vol_* kolonları eklenmez**
- **Enabled=true + timeframe match ⇒ vol_* kolonları eklenir**
- **attrs.timeframe yokken allow_without_timeframe=false ⇒ fail-closed**
- **allow_without_timeframe=true ⇒ timeframe olmadan da compute edebilir** (bilinçli risk/opsiyon)
- **Guard’lar çalışır:**
  - `window < 2` ⇒ skip
  - `ddof >= window` ⇒ skip
- **Geçersiz OHLC (<=0) ⇒ crash yok, vol_* tamamen NaN**

### A2. Kanıt (test koşumu)
Windows için repo standardı: `\.pytest.cmd`

- Komut: `\.pytest.cmd tests\unit\test_advanced_volatility_smoke.py -q`
- Sonuç: **8 passed**

### A3. Timeframe gating mekanizması (kanıt)
Bu test paketi, `df.attrs["timeframe"]` set edildiğinde ve allowlist ile eşleştiğinde `vol_*` kolonlarının gerçekten üretildiğini gösterir.

Bu, üretimdeki mevcut akışla uyumlu:
- Market data pipeline tarafında timeframe `df.attrs["timeframe"]` üzerinden taşınıyor.
- `add_indicators()` advanced_volatility blok’u allowlist üzerinden koşuyor.

---

## B) Runtime Karar Notu (20260119_231242_313193)

Referans session raporu: [reports/session_20260119_231242_313193.md](reports/session_20260119_231242_313193.md)

### B1. Gözlenen gerçekler
- Oturum özeti: **6 trade**, realized PnL **-$1.4765**, tamamı **Chop** fazında.
- Signal funnel: Chop’ta çok sayıda signal var; çoğu `pyramiding_disabled_for_strategy` ve az sayıda `volume_gating` ile reject.
- Timeline tablosunda **Vol sütunu = LOW** (rejim etiketi) görünüyor.

### B2. Telemetri boşluğu (karar verdiren kritik eksik)
Bu run log / session report içinde **`vol_rs_bps / vol_gk_bps / vol_yz_bps / vol_atr_bps / vol_std_bps` sayısal telemetrisi görünmüyor**.

Sonuç: **Runtime log bazlı “estimator karşılaştırması” (RS/GK/YZ vs Std/ATR) bu oturumdan doğrudan üretilemez.**

### B3. Bu oturumdan çıkarılabilen güvenli karar
- Advanced volatility hook’un production’ı kırmadığı (fail-closed) doğrulandı.
- Bu oturumun performans problemi, vol estimator seçimi ile ilişkilendirilebilecek telemetriye sahip değil.
- Estimator seçimi için karar dayanağı hâlâ Faz‑0 offline kıyas raporlarına dayanmalı (mevcut: [reports/vol_estimator_compare_20251227_220109_746903_vs_20260118_202844_781510.md](reports/vol_estimator_compare_20251227_220109_746903_vs_20260118_202844_781510.md)).

### B4. Minimal telemetri önerisi (kod değişikliği değil)
Runtime karar verebilmek için minimum gereksinim:
- En azından `SIGNAL_BREAKDOWN` veya `TRADE_CLOSED.entry_metadata.entry_indicators` içine **son bar `vol_*` snapshot** eklenmeli.
- Snapshot yanında `timeframe`, `window`, `ddof` ve “hangi estimator baz alındı” alanı olmalı.

Bu sayede:
- Run log üzerinden estimator dağılımları (p50/p90), micro-stop oranı, “k_needed_to15” gibi metrikler runtime’da da raporlanabilir.

---

## C) Faz-1 / Faz-2 Minimal PR Taslağı

### Faz-1: Stop scale kalibrasyonu (telemetri + offline doğrulama)
Hedef: micro-stop (<15 bps) yoğunluğunu kontrol etmek için **stop scale’ı kalibre etmek**, ama production default’unu değiştirmeden.

- **Feature flag**: `mr.stop_scale_calibration.enabled=false` (default kapalı)
- **Kalibrasyon mantığı** (öneri):
  - Seçilen vol ölçümü üzerinden `$k$` çarpanı ile `stop_bps = max(stop_floor_bps, k * vol_bps)`
  - `$k$` için Faz‑0 raporundan p50/p75/p90 bazlı “k_needed_to15” türetimi
- **Çıkış kriteri**:
  - Unit/integration test: stop_bps alt/üst bound’lara uyuyor
  - Log/telemetry: her trade için stop_bps, vol_bps, k, floor, clamp sebepleri yazılıyor
- **Rollback**:
  - Flag kapat ⇒ eski stop davranışı

### Faz-2: LOW rejimi politikası (gating / davranış)
Hedef: LOW vol rejiminde “tepkisiz chop” içinde yanlış girişleri azaltmak veya stop/TP davranışını stabilize etmek.

Minimal seçenekler (tek tek flag’lenmeli):
- **LOW rejiminde ek sinyal filtresi**: örn. min z-score / min band excursion / min momentum
- **LOW rejiminde trade aralığı**: aynı yönde tekrar giriş cooldown
- **LOW rejiminde stop floor artırma** (riskli; Faz‑2’de, ölçümlü)

Çıkış kriteri:
- Session bazında: win-rate, avg loss, time_exit oranı, micro-stop oranı değişimi
- Yan etki kontrolü: fırsat kaçırma (signals→executed trade oranı)

### En sonda (opsiyonel) gate relaxation
Risk notu: allowlist genişletme / `allow_without_timeframe=true` gibi gevşetmeler sadece en sonda ve ölçümlü yapılmalı.
- Risk: yanlış timeframe’da compute (özellikle multi-TF pipeline) ⇒ yanlış vol scale ⇒ stop/target sapması

---

## Ek: Pratik doğrulama checklist
- `advanced_volatility.enabled=false` iken üretimde hiçbir `vol_*` beklenmemeli.
- `enabled=true` iken sadece allowlist timeframe’larda `vol_*` üretilmeli.
- `window/ddof` yanlışsa compute skip + crash yok.
- OHLC <= 0 gibi bozuk data crash yapmamalı; vol_* NaN olmalı.
