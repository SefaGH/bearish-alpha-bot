# BTC/USDT Kaçan Fırsat Analizi — 18 Şubat 2026, 19:30 UTC

## Özet

18 Şubat 2026 tarihinde saat 19:30 civarında BTC/USDT paritesinde güçlü bir oversold bounce fırsatı oluştu. Fiyat ~69,200'den ~66,250 seviyesine düştü (~%4.3 düşüş) ve 19:25-19:35 arasında lokal dip oluşturup bounce başlattı. `adaptive_ob` stratejisi bu fırsatı **birden fazla yapısal sorun** nedeniyle değerlendiremedi.

---

## 1. Grafik Gözlemi

- **Parite:** BTCUSDT · 5m · BingX
- **Düşüş:** ~69,200 → ~66,250 (17 Şubat akşamından 18 Şubat 19:30'a kadar)
- **Bounce Bölgesi:** ~66,250 – 66,400 (19:25-19:35 UTC)
- **Bounce Sonrası:** Fiyat kısa vadede ~66,800'lere toparlandı
- **Destek:** 67,050 horizontal destek kırıldıktan sonra 65,839 seviyesine kadar gerileme geldi

---

## 2. Bot Loglarından Kronoloji

### 2a. Trend Penalty Aktivasyonu (19:00:24 UTC)

```
Trend penalty state | symbol=BTC/USDT:USDT
  active=True | reason=turned_on_gap_ge_on
  ema_fast=67265.94 | ema_mid=67510.35
  ema_gap_pct=0.003620 | gap_on=0.002000 | gap_off=0.001500
  base_thr=32.00 | effective_thr=27.00
  extreme_bypass=False
```

**Yorum:** EMA fast, EMA mid'in sadece **%0.36** altına düştü. Hysteresis on-threshold (0.20%) aşıldığı için trend penalty aktifleşti ve RSI threshold'u **32.00 → 27.00**'ye çekildi (sabit 5.0 puanlık penaltı).

**Ek not (Downtrend Veto):** `adaptive_ob` içinde ADX>30 ve fiyat<EMA50 koşulunda threshold'u 25.0'a zorlayan ek bir kural bulunuyor (`src/strategies/adaptive_ob.py`, "Downtrend Veto" bloğu). Bu olayda loglanan effective threshold 27.00 olduğundan bu veto devreye girmemiş görünüyor.

### 2b. Fırsat Penceresi (19:25 – 19:35 UTC)

| Zaman (UTC) | RSI (Hybrid) | Threshold | Delta | Shock State | Sonuç |
|-------------|-------------|-----------|-------|-------------|-------|
| 19:25:30 | 27.62 | 27.00 | +0.62 | DISARMED | No Signal |
| 19:26:02 | 27.88 | 27.00 | +0.88 | DISARMED | No Signal |
| 19:26:34 | 27.58 | 27.00 | +0.58 | DISARMED | No Signal |
| 19:27:06 | 27.92 | 27.00 | +0.92 | DISARMED | No Signal |
| **19:27:38** | **27.30** | **27.00** | **+0.30** | **ARMED** | **No Signal** |
| 19:28:10 | 27.48 | 27.00 | +0.48 | ARMED | No Signal |
| 19:28:42 | 27.77 | 27.00 | +0.77 | ARMED | No Signal |
| 19:29:14 | 27.80 | 27.00 | +0.80 | ARMED | No Signal |
| 19:29:46 | 28.04 | 27.00 | +1.04 | ARMED | No Signal |
| 19:30:18 | 28.16 | 27.00 | +1.16 | ARMED | No Signal |
| 19:30:50 | 27.99 | 27.00 | +0.99 | ARMED | No Signal |

**En yakın geçiş:** 19:27:38'de RSI 27.30 ile threshold'a **sadece 0.30 puan** kaldı. Tam bu anda shock ARMED geçişi de gerçekleşti (shock_score=0.61). Tüm koşullar neredeyse hizalanmıştı.

Log kanıtı:
```
2026-02-18 19:27:38 - [strategies.adaptive_ob] - INFO -
  🚫 [ADAPTIVE_OB/BTC/USDT:USDT] No Signal: RSI (27.30) is above the threshold (27.00).
```

### 2c. Bounce Sonrası RSI Consensus Blokajı (20:00+ UTC)

Bounce başladıktan sonra fast RSI hızla yükselirken slow RSI geride kaldı:

```
[RSI-ROUTER] Skip | symbol=BTC/USDT:USDT | strategy=adaptive_ob
  reason=rsi_router.transition_no_trade
  zone=TRANSITION_LOW
  rsi_slow=31.07 | rsi_fast=52.24
  ob_threshold=32.00 | str_threshold=55.00
  consensus_status=mismatch_transition
```

Bu blokaj **en az 1 saat boyunca** sürdü ve tüm stratejileri (adaptive_ob, adaptive_str, mean_reversion) devre dışı bıraktı.

---

## 3. Tespit Edilen Yapısal Sorunlar

### Sorun #1: Sabit Trend RSI Penaltısı (KRİTİK)

**Konum:** `src/strategies/adaptive_ob.py` — satır 1584  
**Config:** `trend_confirmation_rsi_penalty: 5.0` (hardcoded default)

**Mekanizma:**
```python
ema_trend_penalty = float(self.strategy_config.get('trend_confirmation_rsi_penalty', 5.0))
new_threshold = max(min_adaptive_rsi, adaptive_rsi_threshold - ema_trend_penalty)
# base: 32.0 - 5.0 = 27.0
```

**Problem:**
- EMA gap %0.36 olduğunda da, %3.0 olduğunda da **aynı 5.0 puanlık sabit penaltı** uygulanıyor.
- Hafif downtrend'lerde (bounce potansiyeli yüksek) threshold gereksiz yere derine çekiliyor.
- Bu fırsatta: RSI 27.30'a kadar düştü ama 27.00 eşiği aşılamadı. **Penaltı 3.0 olsaydı** threshold 29.00 olacak ve sinyal üretilecekti.

**Karşıolgusal (Counterfactual) Analiz:**

| Penaltı Tipi | Değer | Threshold | 19:27:38 RSI=27.30 | Sonuç |
|-------------|-------|-----------|---------------------|-------|
| Mevcut (sabit) | 5.0 | 27.00 | > threshold | ❌ Kaçırıldı |
| Oransal (önerilen) | ~1.8 | 30.20 | < threshold | ✅ RSI filtresi geçilirdi* |
| Sabit (hafifletilmiş) | 3.0 | 29.00 | < threshold | ✅ RSI filtresi geçilirdi* |

\* Not: RSI filtresinin geçilmesi, tek başına kesin trade anlamına gelmez; fiyat/EMA, persistency, risk ve diğer guardrail kontrolleri ayrıca geçilmelidir.

### Sorun #2: `mismatch_extreme_override` min_penetration Çok Yüksek (ORTA)

**Konum:** `src/core/rsi_zone_router.py` — `_resolve_mismatch_extreme_override()`  
**Config:** `min_penetration: 2.0`

**Mekanizma:**
```python
low_trigger = bool(
    low_side_enabled
    and float(rsi_slow) <= (float(ob_threshold) - min_penetration)
)
# rsi_slow=31.07 <= (32.00 - 2.0) = 30.00 → False (31.07 > 30.00)
```

**Problem:**
- `rsi_slow=31.07` zaten `ob_threshold=32.00`'ün altında (oversold bölgede).
- Ama override'ın devreye girmesi için `31.07 <= 30.00` olması gerekiyor → **trigger etmiyor**.
- Sonuç: `consensus_status=mismatch_transition` → `zone=TRANSITION_LOW` → `no_trade_new_entry=true` → **tüm stratejiler bloklandı**.
- `min_penetration: 1.0` olsaydı: `31.07 <= 31.00` → Hâlâ trigger etmez ama çok yakın.
- `min_penetration: 0.0` olsaydı: `31.07 <= 32.00` → Override devreye girer, zone=OVERSOLD olurdu → **adaptive_ob çalışabilirdi**.

### Sorun #3: Consensus Mode Fast/Slow RSI Doğal Ayrışmayı Cezalandırıyor (DÜŞÜK-ORTA)

**Konum:** `src/core/rsi_zone_router.py` — `resolve_zone()` fonksiyonu

**Problem:**
- Bounce sonrası fast RSI (5m) doğal olarak hızla yükselirken slow RSI (30m) geride kalır.
- Bu **her başarılı bounce'ta** olacak bir davranıştır — algoritmik bir sorun değil, piyasa fiziğidir.
- Mevcut consensus mantığı bu durumu "mismatch" olarak değerlendirip TRANSITION_LOW'a düşürür.
- `no_trade_new_entry=true` kuralı nedeniyle TRANSITION zone'unda hiçbir yeni pozisyon açılamaz.
- Sonuç: Bot, bounce'ı fark etse bile bounce'tan faydalanacak ikinci bir fırsat oluşursa onu da kaçırır.

---

## 4. Çözüm Önerileri

### Öneri 1: Oransal Trend RSI Penaltısı (Sorun #1 için — ÖNCELİK: YÜKSEK)

**Mevcut kod** (`adaptive_ob.py:1584`):
```python
ema_trend_penalty = float(self.strategy_config.get('trend_confirmation_rsi_penalty', 5.0))
new_threshold = max(min_adaptive_rsi, adaptive_rsi_threshold - ema_trend_penalty)
```

**Önerilen değişiklik:**
```python
# Oransal penaltı: EMA gap büyüklüğüne göre ölçeklendir
ema_trend_penalty_max = float(self.strategy_config.get('trend_confirmation_rsi_penalty', 5.0))
ema_trend_penalty_scale = float(self.strategy_config.get('trend_confirmation_penalty_scale', 1000.0))

if ema_gap_pct is not None:
    # gap=%0.36 → 0.0036*1000=3.6 → min(5.0,3.6)=3.6 → threshold=32-3.6=28.4
    # gap=%0.50 → 0.005*1000=5.0 → min(5.0,5.0)=5.0 → threshold=32-5.0=27.0
    # gap=%0.20 → 0.002*1000=2.0 → min(5.0,2.0)=2.0 → threshold=32-2.0=30.0
    ema_trend_penalty = min(ema_trend_penalty_max, ema_gap_pct * ema_trend_penalty_scale)
else:
    ema_trend_penalty = ema_trend_penalty_max

new_threshold = max(min_adaptive_rsi, adaptive_rsi_threshold - ema_trend_penalty)
```

**Etki:** Bu fırsatta EMA gap %0.36 idi → penalty 3.6 → threshold 28.4. RSI 27.30 < 28.4 olacağı için RSI filtresi geçilirdi; nihai sinyal üretimi için diğer kontrollerin de sağlanması gerekir.

**Config parametreleri:**
```yaml
oversold_bounce:
  trend_confirmation_rsi_penalty: 5.0        # Max penaltı cap'i (mevcut)
  trend_confirmation_penalty_scale: 1000.0   # Ölçeklendirme faktörü (yeni)
  # Formula: penalty = min(max_penalty, ema_gap_pct * scale)
```

### Öneri 2: `min_penetration` Değerini Düşür (Sorun #2 için — ÖNCELİK: ORTA)

**Mevcut config:**
```yaml
mismatch_extreme_override:
  enabled: true
  low_side_enabled: true
  min_penetration: 2.0  # RSI slow, ob_threshold'un 2.0 puan altında olmalı
```

**Önerilen config:**
```yaml
mismatch_extreme_override:
  enabled: true
  low_side_enabled: true
  min_penetration: 0.0  # RSI slow ob_threshold'un altında olması yeterli
```

**Gerekçe:** Slow RSI zaten ob_threshold'un (32.00) altındaysa, piyasa fiilen oversold bölgededir. Ek penetrasyon gerekliliği, consensus mismatch durumunda gereksiz blokajlara yol açıyor.

**Alternatif (daha muhafazakar):** `min_penetration: 0.5` — En azından bir miktar derinlik teyidi ister ama 2.0'a göre çok daha az kısıtlayıcıdır.

### Öneri 3: Bounce Sonrası Fast/Slow Ayrışma Toleransı (Sorun #3 için — ÖNCELİK: DÜŞÜK-ORTA)

Bu sorun, Öneri 2 ile büyük ölçüde çözülür. `min_penetration` düşürüldüğünde, slow RSI oversold'dayken fast RSI yukarı çıksa bile `mismatch_extreme_override` devreye girip zone'u OVERSOLD olarak koruyacaktır.

Eğer ek bir mekanizma istenirse:
- **Directional bias:** Fast RSI'ın slow RSI'dan yukarı ayrışması (bounce sinyali) durumunda, mismatch'i TRANSITION yerine slow_zone'a (OVERSOLD) fallback ettiren bir "recovery_mode" eklenebilir.
- Config: `rsi_zone_router.transition.bounce_recovery_mode: true`

---

## 5. Risk Değerlendirmesi

| Değişiklik | Risk | Etki |
|-----------|------|------|
| Oransal penaltı | Düşük — max cap hâlâ 5.0, güçlü downtrend'lerde mevcut davranış korunur | Hafif trend'lerde daha esnek threshold |
| min_penetration: 0.0 | Orta — Zone override daha sık devreye girecek | Consensus mismatch'te daha az blokaj, potansiyel olarak daha fazla false positive sinyal |
| Bounce recovery mode | Orta — Yeni mekanizma, kapsamlı test gerektirir | İkincil fırsatların yakalanması |

---

## 6. Sonuç

Bu fırsat **0.30 RSI puanıyla** kaçırıldı. Kök neden, **sabit trend penaltısının** hafif downtrend koşullarında orantısız şekilde sıkıştırmasıdır. Öneri 1 (oransal penaltı) uygulansaydı bu olayda en azından RSI giriş filtresi daha yüksek olasılıkla geçilirdi. Öneri 2 ile birlikte uygulandığında, bounce sonrası süresiz blokaj sorunu da çözülmüş olur.

**Tahmini iyileştirme:** Benzer hafif-downtrend bounce fırsatlarının yakalanma oranı %30-50 artardı (backtesting ile doğrulanmalı).

---

## 7. Sınırlamalar ve Doğrulama Notları

- Bu repoda 18 Şubat 2026 19:30 UTC olayına ait ham container log dosyası bulunmadığı için zaman çizelgesi ve sayıların birincil kaynağı bu dokümandaki log alıntılarıdır.
- Kod davranışları repository kaynak kodundan doğrulanmıştır (`adaptive_ob` trend penalty + Downtrend Veto, `rsi_zone_router` mismatch/transition gate).
- Bu nedenle karşıolgusal sonuçlar "kesin trade olurdu" şeklinde değil, "RSI filtresini geçme olasılığı artardı" şeklinde yorumlanmalıdır.

---

*Analiz: GitHub Copilot — 18 Şubat 2026, 21:00 UTC*  
*Veri Kaynağı: Bu dokümanda alıntılanmış bearish-bot Docker container logları, kaynak kodu*
