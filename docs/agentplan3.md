Aşağıda, iki planını **tek bir nihai dokümana** birleştirip; bizim log/grafik tespitlerini de içine gömerek **Codex agent’ın doğrudan uygulayabileceği** şekilde detaylandırdım.

> Not: Daha önce yüklediğin bazı dosyaların süresi dolmuş olabilir; bu nedenle “yeniden doğrulama / tekrar parse” istersek log/görseli yeniden yüklemen gerekebilir. Bu nihai planı hazırlamak için şu an ek dosyaya ihtiyaç yok.

---

# Nihai Plan: Bağlama Duyarlı, Çok Modlu Trader (Minimal Müdahale Fazı)

## A) Hedef ve Tasarım İlkeleri

**Hedef:** Botu “işlem açmayan” hale getirmeden, **rejime uygun sağlıklı risk** aldırmak.

**Tespitlerden gelen ana gereksinimler:**

1. **TREND-DOWN** penceresinde MR-LONG “falling knife” üretiyor → **rejim + confirm + churn freni** şart.
2. **TREND-DOWN** bacağında MR tek başına fırsatı kaçırıyor → **Trend-Follow SHORT** şart.
3. **SHOCK/flush sonrası rebound**: ADX veto güvenli ama fırsat kaçırıyor → **Capitulation Overlay (ARM/CONFIRM) + küçük boy** şart.
4. **Doğru SHORT**’lar erken kapanıyor → **ATR/vol tabanlı trailing + 2 kademeli çıkış** şart.
5. **execution**: pending/active_exists + timeout→market edge’i yiyor → **supersede + timeout→cancel + dynamic gate** şart.
6. Exit reason / fee-lock / trailing karışıyor → **telemetri standardı** şart.

---

# B) Sistem Mimarisi (Mevcut Yapı Üzerinden)

```
[MarketRegimeAnalyzer + AdaptiveOB Shock Gate] → [Strategy/Risk Policy Router] → [Execution] → [Exit/PM]
                 ↓                                 ↓                     ↓           ↓
     RANGE / TREND_DOWN / SHOCK (türetilmiş)        MR / OB Overlay        smart-entry  trailing/fee-lock
```

## Rejimler ve aktif modüller (türetilmiş kaynaklar)

1. **RANGE**

   * Mean Reversion (Normal) + normal risk

2. **TREND_DOWN**

   * Mean Reversion LONG: **disabled veya confirmed_only + küçük boy**
   * (Trend-Follow SHORT **bu fazda yok**; ayrı ekleme planında)

3. **SHOCK (Capitulation)**

   * Normal MR-LONG: **disabled**
   * **Capitulation Overlay: ARM/CONFIRM ile küçük boy long (AdaptiveOB shock gate üzerinden)**

> Önemli: SHOCK’ta “işlem yok” değil; “kör MR-LONG yok, teyitli overlay var”.

---

# C) Uygulama Yol Haritası (Bu Faz: sadece 1–2) + net görev listesi

## 1. Faz (Şimdi) — Stop the Bleeding: MR Policy + Shock Overlay (P0)

### C1) Rejim Kaynağı Birleştirme (Yeni modül yok)

**Not:** Yeni `regime_classifier.py` eklenmez.

**Kaynaklar:**

* **TREND/RANGE**: `MarketRegimeAnalyzer` çıktısı (`regime_data`)
* **SHOCK**: `adaptive_ob` dyn gate (`meta.shock_state`, `meta.shock_score`)

**Türetilmiş etiketler (başlangıç):**

**TREND_DOWN**

* `regime_data.trend == bearish` **ve** `trend_strength (ADX) >= threshold`

**SHOCK**

* `meta.shock_state == ARMED` **veya** `shock_score >= threshold`

**RANGE**

* Diğerleri

**Config (öneri):**

```yaml
regime:
  enabled: true
  trend_down:
    adx_floor: 25
  shock:
    state: "ARMED"
    min_score: 0.60
```

**Log şartı:** her sinyal döngüsünde

* `[REGIME] trend=... adx=... shock_state=... shock_score=...`

---

### C2) MR-LONG Rejim Politikası (mean_reversion.py)

**Amaç:** TREND_DOWN’da kör long zincirini kes; RANGE’te MR akışını bozma.

#### Policy

* **RANGE:** MR normal (mevcut kurallar)
* **TREND_DOWN:** MR-LONG = `disabled` (varsayılan)
  Alternatif mod (isteğe bağlı): `confirmed_only`

  * şartlar:

    * `breach_bps <= -20`
    * `reclaim_confirmed = true` (1m veya 5m kapanış band içine)
    * `rising_adx` varsa size düşür (`*0.5`)
  * size_mult: 0.5 (başlangıç)
* **SHOCK:** normal MR-LONG = disabled (overlay’e bırak)

**Config:**

```yaml
strategies:
  mean_reversion:
    regime_policy:
      range:
        enabled: true
      trend_down:
        long_mode: "disabled"        # disabled | confirmed_only
        min_breach_bps: -20
        require_reclaim: true
        size_mult: 0.5
        rising_adx_floor: 20
        rising_adx_size_mult: 0.5
      shock:
        long_mode: "disabled"
```

#### Stop sonrası akıllı fren (churn guard)

* aynı yönde 2 stop / 15 dk:

  * `cooldown=15dk` **veya** `require_confirm=true` (daha iyi)
* 3 stop / 30 dk:

  * `size_mult=0.25 + require_confirm=true`

**Config:**

```yaml
risk:
  loss_streak:
    enabled: true
    window_minutes: 30
    rules:
      - stops: 2
        cooldown_minutes: 15
      - stops: 3
        size_mult: 0.25
        require_confirm: true
```

**Kabul kriteri (bu faz sonunda):**

* 30m downtrend + dump penceresinde (senin 02:47–03:17 benzeri) MR-LONG “ilk breach” ile arka arkaya açılmıyor.
* RANGE’te MR trade frekansı ciddi düşmüyor.

---

### C3) Capitulation Reversal Overlay (SHOCK modunda) — **AdaptiveOB üzerinden**

**Yeni dosya yok:** `adaptive_ob` dyn gate + reversal confirmation + risk profili ile uygulanır.

#### State machine

`OFF → ARMED → CONFIRMED(entry) → COOLDOWN`

**ARM:**

* `regime == SHOCK`
* `breach_atr >= 0.8`
* `max_arm_minutes=10`
* `low_watermark_px` takip

**CONFIRM (entry):** en az biri

* `close_1m > lower_band` **ve** `reclaim_hold_seconds >= 20`
* veya `bounce_from_low >= 0.25 * ATR_1m`
* veya 1m’de 2 bar “higher-low + higher-close”

**Risk (öneri):**

* size_mult: `0.3`
* stop: `low_watermark - 0.25*ATR_1m`
* exit: `%50 @ 0.3R`, kalan runner `ATR trailing (1.5x)`

**Cooldown:** 20 dk

**Config (öneri):**

```yaml
strategies:
  adaptive_ob:
    shock:
      arm_score_threshold: 1.0
    armed:
      ttl_s: 600
      cooldown_s: 1200
    # overlay risk profile (MR karşıtı küçük boy)
    size_mult: 0.3
```

**Log şartı:**

* `[CAP] state=ARMED breach_atr=... low=...`
* `[CAP] state=CONFIRMED reason=... entry=... stop=...`

---

**Not:** Trend‑follow short bu fazda **iptal** (sonradan eklenecek).

---

## 2. Faz (Beklemede) — Exit & Execution İnce Ayarı (P0)

### C5) Execution düzeltmeleri (core/execution_engine.py)

#### 1) Dynamic gate (vol’a bağlı)

* `gate_bps = clamp(max(base, k*vol_std_bps), min, max)`
* öneri: base=12, k=0.25, max=60

#### 2) Pending supersede + active_exists çözümü

* limit pending 45 sn dolmadıysa:

  * cancel + new_signal (ancak `quality_score` daha iyiyse)
* timeout→market **MR’de kapalı**, timeout→cancel
* trend-follow için (opsiyonel): momentum varsa küçük boy market izinli (feature flag)

**Config:**

```yaml
execution:
  smart_entry:
    pending_supersede_seconds: 45
    timeout_cancel_seconds: 60
    gate:
      base_bps: 12
      k_vol: 0.25
      min_bps: 12
      max_bps: 60
```

#### 3) STOP_HIT_BEFORE_ENTRY abort sonrası

* aynı dedupe key’e **tek retry** (güncel stop/entry) veya “supersede allow”

**Kabul kriteri:**

* “active_exists” drop oranı düşüyor.
* timeout→market kaynaklı kötü fill azalıyor.

---

### C6) Exit yönetimi (core/position_manager.py)

#### 1) Fee-lock R tabanlı

* `fee_lock_r = 0.5R` (başlangıç)

#### 2) Trailing ATR tabanlı + rejime göre genişlik

* RANGE: 1.0x ATR
* TREND: 1.8x ATR (runner için)
* SHOCK: 2.2x ATR (boy küçük)

#### 3) 2 kademeli çıkış

* %50 @ 0.3R, runner trailing

#### 4) Slippage cap (stop)

* `max_slippage_bps = 25`
* stop-limit / reduce-only fallback

#### 5) Exit reason standardı

* tek reason: `STOP_LOSS | TRAILING | FEE_LOCK | TP_PARTIAL | MANUAL`
* karışık log (stop+trailing aynı anda) engellenecek.

**Config:**

```yaml
exits:
  partial_tp_r: 0.3
  fee_lock_r: 0.5
  trailing:
    range_atr_mult: 1.0
    trend_atr_mult: 1.8
    shock_atr_mult: 2.2
  slippage:
    max_slippage_bps: 25
    stop_limit_enabled: true
```

**Kabul kriteri:**

* Doğru short’larda realize PnL artar (trend taşıma).
* Fee-lock “kârdayken zarara dönüş” patolojisi azalır.

---

## 3. Faz (Beklemede) — Entegrasyon + Dinamik Optimizasyon + Regresyon (P1)

### C7) Dynamic Z decay (kilitlenme çözümü)

* SHOCK sonrası `required_z` zamanla düşsün (tau 10–15 dk)
* veya ADX düştükçe required_z de düşsün

**Config:**

```yaml
mean_reversion:
  dynamic_z:
    decay:
      enabled: true
      tau_minutes: 12
      min_required_z: 1.2
```

### C8) Golden-window regresyon / ölçümleme

**Metrikler:**

1. Net PnL
2. Rejim başına trade sayısı ve dağılım
3. Win rate + Avg R
4. Max loss streak / churn
5. MFE vs realized (exit verimi)
6. TREND_DOWN coverage (trend-follow trade sayısı)
7. SHOCK ARMED→CONFIRMED oranı

---

# D) Codex Agent için “Tek Sayfa Görev Listesi” (Bu Faz)

1. Rejim kaynağını birleştir (MarketRegimeAnalyzer + adaptive_ob shock gate) ve `[REGIME]` loglarını standardize et.
2. MR içine `regime_policy` uygula (TREND_DOWN long disabled/confirmed_only, SHOCK long disabled).
3. `adaptive_ob` shock gate üzerinden küçük boy “capitulation overlay” kurgusunu uygula.

> Diğer görevler beklemede (execution/exit/churn/trend-follow).

---

# E) “Sağlıklı Risk”in pratik tanımı (bu planla)

* RANGE’te MR **işlem üretmeye devam eder**.
* TREND_DOWN’da risk **trend yönüne kayar** (trend-follow short), trende karşı risk **kalite/confirm ile** sınırlanır.
* SHOCK’ta bot “susmaz”; **teyitli** ve **küçük boy** bir dip-reversal riski alır.

---
