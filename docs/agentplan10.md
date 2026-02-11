## Fırsatı Kaçırmamak İçin Öneriler

Sorunu 3 katmanda ele alıyorum: **SL dayanıklılığı**, **Shock mekanizması**, **re-entry zekası**.

---

### 1. STOP-LOSS GENİŞLETME — Kök Neden Çözümü

Bu işlemde SL ($67,323) sadece **$193 (~0.29%)** uzaktaydı. BTC'de 5m TF'de bu bir noise aralığı. SL vurulmasaydı, TP ($66,892) 50 dk sonra tutacaktı.

**Öneri: ATR-bazlı minimum SL tabanı**

```yaml
mean_reversion:
  risk:
    # Mevcut (çok dar)
    min_stop_pct: 0.001
    # Önerilen: ATR'nin belirli katı olarak SL tabanı
    min_stop_atr_mult: 1.5       # SL en az 1.5×ATR olsun
    # ATR ~$149 olduğunda → min SL = $224 (~0.33%)
    # Bu, noise spike'larından korur ama R:R'ı çok bozmaz
```

**Neden?** Girişteki ATR $149 idi. 1.5×ATR = $224. Mevcut SL $193 bunu karşılamıyor — eğer bu kural olsaydı SL $67,354'e genişler ve **o spike'da vurulmazdı** (tepe $67,344 idi). Trade hayatta kalır, 50 dk sonra kâr ederdi.

---

### 2. SHOCK BYPASS — "Aynı Yön" İstisnası

Kodda Shock `adaptive_ob.py` de DISARMED→ARMED→COOLDOWN→DISARMED döngüsü çalışıyor. Mevcut yapıda **herhangi bir** volatilite yüksekliği tüm stratejileri durduruyor. Ama SL sonrası **aynı yönde** yeniden giriş çoğu zaman doğru harekettir.

**Öneri: `shock_bypass_same_direction` parametresi**

```yaml
strategies:
  mean_reversion:
    regime_policy:
      shock:
        # Mevcut davranış: shock ARMED iken long engelli
        long_mode: "disabled"
        # YENİ: Eğer son kapanan trade AYNI YÖN ise ve z-score yüksekse bypass
        same_direction_bypass:
          enabled: true
          min_z_score: 2.0          # Sadece güçlü sinyallerde bypass
          min_quality_score: 0.65   # Kalite filtresi
          max_time_since_sl_s: 300  # SL'den sonraki 5 dk içinde tekrar girebilir
          require_rejection_conf: true  # Rejection confirmation zorunlu
```

**Mantık:** 12:25'te z=2.67 (çok güçlü) ve yön SHORT (son kapanan trade de SHORT idi). Bu senaryoda Shock bypass'ı mantıklı çünkü piyasa tezini doğruluyor — sadece giriş noktası erken/dardı.

---

### 3. SHOCK TTL DİNAMİKLEŞTİRME

Mevcut yapıda Shock TTL sabit (`ttl_s`, `cooldown_s` config'den). Büyük bir flash crash ile küçük bir SL exit **aynı süre** Shock uyguluyor.

**Öneri: Score'a göre kademeli TTL**

```yaml
adaptive_ob:
  dyn_fast_gate:
    armed:
      # Mevcut sabit TTL yerine kademeli
      ttl_tiers:
        - max_score: 0.5    # Düşük shock → kısa kilit
          ttl_s: 120
          cooldown_s: 60
        - max_score: 0.75   # Orta shock
          ttl_s: 300
          cooldown_s: 180
        - max_score: 1.0    # Yüksek shock → uzun kilit
          ttl_s: 600
          cooldown_s: 300
```

Bu durumda score=0.66 orta tier'a düşerdi → TTL 300s (5 dk), COOLDOWN 180s (3 dk). Toplam 8 dk. Mevcut 14 dk yerine 8 dk'da çözülür ve z=2.67 fırsatı hâlâ yakalanabilirdi.

---

### 4. RE-ENTRY GUARD İYİLEŞTİRME

Kodda zaten `reentry_guard` mekanizması var (`mean_reversion.py:467`). Ama bu sadece **LONG** girişe VWAP reclaim şartı koyuyor. SHORT tarafı için eşdeğeri yok.

**Öneri: İki yönlü reentry guard + hızlı temizleme**

```yaml
mean_reversion:
  reentry_guard:
    enabled: true
    require_vwap_reclaim_after_stop: true
    # YENİ: SHORT tarafı - SL sonrası band dışına çıkınca guard temizlensin
    short_side:
      enabled: true
      clear_on_band_breach: true   # Fiyat tekrar üst bant üstüne çıkınca guard temizle
      clear_on_z_threshold: 2.0    # Veya z>2.0 olunca doğrudan temizle
```

---

### 5. "SL → SHOCK → KAÇIRMA" DÖNGÜSÜ KIRICI

Bu en kritik yapısal sorun. Kendi SL exit'i shock tetikliyor → sonraki fırsatı engelliyor.

**Öneri: SL-kaynaklı Shock'u ayrı sınıflandır**

```python
# adaptive_ob.py shock state machine'ine ek:
# SL çıkışından kaynaklanan küçük volatiliteyi "internal_sl" olarak işaretle
# Bu tür shock'lar:
# 1. Daha kısa TTL (60-120s)
# 2. Aynı yön sinyallerini bloke ETMEsin
# 3. Sadece karşı yön sinyallerini ertelisin (falling-knife protection)
```

Bunu implemente etmek için `strategy_coordinator.py:1147`'deki TRADE_CLOSED handler'dan shock kaynağını (`exit_reason: stop_loss`) `adaptive_ob`'a iletmek gerekir. Şu an shock sadece price-move-based hesaplanıyor, ama kaynağı bilinmiyor.

---

### 6. PRATİK QUICK-WIN: Min R:R Artırma

Bu tradenin R:R'ı 1.23 idi (minimum 1.215). Bu çok düşük. Hızlı bir config değişikliği:

```yaml
mean_reversion:
  risk:
    min_rr_ratio: 1.8    # Mevcut ~1.2'den yükselt
    # Bu sayede:
    # - Dar SL'li, düşük ödüllü tradeler otomatik reddedilir
    # - SL genişletme > TP daraltma denklemi kendini zorlar
    # - Daha az ama daha kaliteli girişler
```

---

### Öncelik Sıralaması

| Öncelik | Değişiklik | Etki | Zorluk |
|---------|-----------|------|--------|
| **1** | **ATR-bazlı min SL tabanı** | Bu trade kurtarılırdı | Config+küçük kod |
| **2** | **Min R:R yükseltme (1.2→1.8)** | Düşük kaliteli girişler engellenir | Sadece config |
| **3** | **Shock TTL kademelendirme** | 12:25 fırsatı yakalanırdı | Orta kod değişikliği |
| **4** | **SL-kaynaklı Shock ayrıştırma** | Döngü kırılır | Orta-büyük kod |
| **5** | **Same-direction bypass** | Tekrar giriş mümkün olur | Orta kod |
| **6** | **Bidi reentry guard** | SHORT tarafı da korunur | Küçük kod |

**Hemen uygulanabilecek 2 değişiklik** (1) ve (2) — ikisi birlikte bu trade'in hem gereksiz SL'ye takılmasını engellerdi, hem de böyle dar R:R'lı girişleri filtreleyerek kaliteyi artırırdı.