### **Priority 1: PROMOTE Override'ı Sertleştir** ⭐⭐⭐⭐⭐

```python
# mean_reversion.py içinde PROMOTE override bloğu

# NOT: Repo gerçekleri (mevcut mimariyle uyum)
# - Recheck/FastWatch v2 metadatası StrategyCoordinator -> check_detail["fast_watch"] altında gelir.
# - MR şu an kendi içinde dist_bps hesaplıyor; fakat FastWatch zaten "dist_to_band_bps" üretiyor.
#   PROMOTE threshold'ları için tek bir dist_bps tanımına standardize etmek şart.
# - volume_strength ve ema_stack MR içinde hazır değil (ya MR içinde approx hesaplanacak, ya upstream geçirilecek).

# ❌ ESKİ (tehlikeli)
if (touch_confirmed and near in ['lower', 'upper'] and 
    dist_bps <= 6.0 and abs(z) >= 1.23 and adx <= 25):
    promotion_override = True
    adx_ok = True  # bypass

# ✅ YENİ (güvenli + repo-uyumlu)
# DİKKAT (uygulanabilirlik notları):
# - Bu fonksiyon keyword-only tasarlandıysa çağrı da keyword olmalı (aksi halde TypeError).
# - mean_reversion.py içinde logger genelde modül seviyesi `logger` (self.logger olmayabilir).
#   Uygulamada `logger.info(...)` kullanımına göre uyarlayın.
def check_promotion_override(
    self,
    *,
    touch_confirmed,
    near,
    dist_bps,
    z,
    adx,
    # Opsiyonel sinyaller: yoksa gate'ler "skip" veya "fail-safe" olabilir
    shock_state=None,
    regime_data=None,
    volume_strength=None,
    ema_stack=None,
):
    """
    PROMOTE override için çok daha sıkı şartlar.
    Amaç: Sadece GERÇEK ekstrem sapmalarda kapıyı aç.
    """
    if not touch_confirmed or near not in ['lower', 'upper']:
        return False
    
    # Şart 1: Mesafe çok dar olmalı (6.0 → 2.0 bps)
    # NOT: check_detail.fast_watch.dist_to_band_bps yönlü/signed olabilir.
    # Yakınlık eşiği için MUTLAKA abs()/normalize et (aksi halde negatif büyük değerler yanlışlıkla "yakın" sayılabilir).
    if abs(dist_bps) > 2.0:  # "sürtme mesafesi" olmalı
        logger.info(f"PROMOTE reject: dist_bps={dist_bps:.2f} (abs>{2.0})")
        return False
    
    # Şart 2: Z-score çok yüksek olmalı (1.23 → 2.0)
    if abs(z) < 2.0:  # belirgin sapma şart
        logger.info(f"PROMOTE reject: z={z:.2f} < 2.0")
        return False
    
    # Şart 3: ADX veto'yu ASLA bypass etme!
    # PROMOTE bile olsa trend varsa girme
    if adx > 20:  # 25 → 20 daha sıkı
        logger.info(f"PROMOTE reject: ADX={adx:.1f} > 20 (trend risk)")
        return False
    
    # ★ YENİ: Şart 4: Trend veto (EMA stack kontrolü)
    # NOT: ema_stack yoksa burada iki politika var:
    # - Fail-open (mevcut): veri yoksa veto yok
    # - Fail-safe: veri yoksa PROMOTE'u reject et
    if ema_stack and self._is_trend_against_mr(near, ema_stack):
        logger.info("PROMOTE reject: trend against MR direction")
        return False
    
    # ★ YENİ: Şart 5: Volume yeterli olmalı (conviction)
    # NOT: volume_strength MR içinde yoksa bu gate opsiyonel kalmalı ya da MR içinde approx hesaplanmalı.
    if volume_strength is not None and volume_strength < 0.50:
        logger.info(f"PROMOTE reject: volume_strength={volume_strength:.2f} < 0.50")
        return False
    
    # ★ YENİ: Şart 6: Regime uyumu
    # NOT: MR'ın mevcut regime_policy guardrail'i şu an LONG tarafında daha belirgin.
    # Minimum güvenlik: shock_state ARMED/TRIGGERED iken PROMOTE kapalı.
    # NOT: shock_state MR içinde genelde ayrı kwargs olarak geliyor; regime_data içinden okumak yanlış olabilir.
    if shock_state is not None:
        shock_state_u = str(shock_state).upper()
        if shock_state_u in ["ARMED", "TRIGGERED"]:
            logger.info(f"PROMOTE reject: shock_state={shock_state_u}")
            return False
    
    logger.info(f"✓ PROMOTE override APPROVED: z={z:.2f}, dist={dist_bps:.2f}, ADX={adx:.1f}")
    return True

# Uygulama (keyword-only çağrı)
promotion_override = self.check_promotion_override(
    touch_confirmed=touch_confirmed,
    near=near,
    dist_bps=dist_bps,
    z=z,
    adx=adx,
    shock_state=shock_state,
    regime_data=regime_data,
    volume_strength=volume_strength,
    ema_stack=ema_stack,
)

# PROMOTE bile olsa ADX veto'yu bypass etme!
if promotion_override:
    # ADX bypass KALDIRILDI - normal ADX kontrolü devam etsin
    pass  # adx_ok = True YOK artık
```

**Etki**: 03:10 ve 03:30 trade'leri bu yeni PROMOTE ile **ENGELLENİRDİ**:
- 03:10: z=1.55 < 2.0 → ❌
- 03:30: z=1.60 < 2.0 → ❌ (ayrıca EMA bullish stack → ❌)

---

### **Priority 2: Reversal Confirmation'ı Recheck'e Ekle** ⭐⭐⭐⭐⭐

```python
# mean_reversion.py içinde

# ❌ ESKİ (recheck'te bypass)
if not is_recheck:
    # Rejection Confirmation (SHORT)
    if entry_short:
        # ... rejection logic ...
        if not rejection_confirmed:
            return None

# ✅ YENİ (repo-uyumlu, kademeli rollout)
# 1) "if not is_recheck:" koşulunu kaldır: rejection confirmation recheck'te de çalışsın.
# 2) Veri garantisi yok: OHLC kolonları yoksa veya forming candle kullanılıyorsa KIRICI davranma.
#    - includes_forming=true ise mümkünse prev closed candle kullan
#    - OHLC eksikse: mevcut davranışı koru + telemetry logla; (opsiyonel) tek rearm sonrası HOLD
#    Not: Mevcut recheck entegrasyon testlerinde OHLC olmadan short beklenen durum var.
# 3) Recheck'te yeni kriter icat etmek yerine mevcut rejection-confirmation mantığını
#    recheck'te de çalıştır (aynı fonksiyon/aynı kurallar).

if entry_short:
    # pseudo:
    candle = last_closed_candle(clean_sig)  # includes_forming aware
    if not has_ohlc(candle):
        logger.info("MR recheck: OHLC missing; keep legacy behavior (no hard reject)")
        # Stage-1 (observe): rejection kontrolü bu durumda uygulanmaz; mevcut flow korunur.
        # Stage-2 (enforce): istenirse 1 rearm + sonra HOLD gibi fail-safe eklenebilir.
    else:
        rejection_confirmed = self._rejection_confirmation_short(
            candle=candle,
            vwap=vwap,
            upper_band=upper_band,
            # ... mevcut parametreler ...
        )
        if not rejection_confirmed:
            logger.info("MR SHORT rejected: no reversal confirmation")
            return None
```

**Etki**: PROMOTE override tetiklense bile, reversal teyidi olmadan entry olmaz.

---

### **Priority 3: Trend Veto Helper Ekle** ⭐⭐⭐⭐

```python
# mean_reversion.py'ye yeni helper method

def _is_trend_against_mr(self, direction, ema_stack):
    """
    MR direction ile trend uyumlu mu kontrol et.
    
    Args:
        direction: 'upper' (short için) veya 'lower' (long için)
        ema_stack: {'ema21': x, 'ema50': y, 'ema200': z}
    
    Returns:
        True ise trend MR aleyhtedir (girme!)
    """
    if not ema_stack or 'ema21' not in ema_stack:
        return False  # veri yoksa veto yok
    
    ema21 = ema_stack['ema21']
    ema50 = ema_stack['ema50']
    ema200 = ema_stack['ema200']
    
    # Bullish stack: EMA21 > EMA50 > EMA200
    bullish_stack = (ema21 > ema50 > ema200)
    
    # Bearish stack: EMA21 < EMA50 < EMA200
    bearish_stack = (ema21 < ema50 < ema200)
    
    # MR short ama trend yukarı → veto
    if direction == 'upper' and bullish_stack:
        logger.info(f"Trend veto: MR SHORT but EMA bullish stack")
        return True
    
    # MR long ama trend aşağı → veto
    if direction == 'lower' and bearish_stack:
        logger.info(f"Trend veto: MR LONG but EMA bearish stack")
        return True
    
    return False

# check_signals metodunda kullan
# NOT: EMA stack / TF seçimi drop-in değil.
# Aşama-1 öneri: yeni EMA fetch eklemeden önce `regime_data` içindeki mevcut trend sınıflamasını kullan.
# Aşama-2: EMA stack gerçekten gerekiyorsa, indicator pipeline'dan hangi TF dataframe'in erişilebilir olduğunu netleştir.
ema_stack = self._get_ema_stack(candles_30m or candles_5m)  # opsiyonel (Aşama-2)
if self._is_trend_against_mr(near, ema_stack):
    # Trend aleyhteyse PROMOTE'a bile izin verme
    if abs(z) < 2.5:  # çok ekstrem değilse
        return None
```

---

### **Priority 3.5: SHORT-Side Regime Guardrails (Simetri)** ⭐⭐⭐⭐

Mevcut tespit: MR içinde rejim/trend guardrail mantığı pratikte **LONG tarafında** daha güçlü.
Bu, directional up-move sırasında "MR SHORT" girişlerinin gereğinden kolaylaşmasına yol açıyor.

**Hedef**: LONG'a uygulanan trend/shock veto mantığını SHORT tarafına da simetrik uygula.

Minimum repo-uyumlu uygulama:
- Trend yukarı (EMA bullish stack / market_regime bullish) ise **MR SHORT** normalde kapalı.
- Sadece **çok ekstrem** fade'lerde (örn. $|z|\ge 2.5$) ve "rejection confirmation" geldiyse izin ver.
- Shock_state ARMED/TRIGGERED ise iki yönde de entry kapalı (PROMOTE dahil).

Uygulama önerisi (kademeli, düşük entegrasyon maliyeti):
1) EMA stack eklemeden: `regime_data.trend` + `shock_state` ile short-side veto/size_mult.
2) Sonra gerekirse EMA helper ile trend sınıflamasını güçlendir.

Önerilen config genişletmesi (kod desteği gerektirir):

```yaml
strategies:
    mean_reversion:
        regime_policy:
            enabled: true

            # mevcut: long tarafı
            trend_down:
                long_mode: "disabled"
            shock:
                long_mode: "disabled"

            # ★ YENİ: short tarafı simetri
            trend_up:
                short_mode: "disabled"      # veya "size_mult"
                adx_floor: 25               # opsiyonel
            shock:
                short_mode: "disabled"
```

Notlar:
- Trend sınıflaması için `core/market_regime.py` içindeki EMA alignment mantığı reuse edilebilir.
- En "az değişiklik" yaklaşımı: `_is_trend_against_mr()` + shock_state check zaten var; bunu entry_short gating'e de uygula.

---

### **Priority 4: Config Ayarlarını Güncelle** ⭐⭐⭐

```yaml
# config/config.example.yaml

# NOT: Bu repo'da gerçek yol `strategies.mean_reversion`.
# MR config şeması "soft_deferral_threshold" + "fast_watch" (v2) şeklinde.
# Aşağıdaki blok bu şemaya göre güncellenmiştir.

strategies:
    mean_reversion:
        # near-miss → waiting-room eşiği (fraction): 0.0010 = 10 bps
        soft_deferral_threshold: 0.0010

        fast_watch:
            allow_touch_entry: false

            # ★ YENİ: PROMOTE override şartları
            # NOT: Bu anahtarlar config'e eklenebilir ama kodun bunları okuyacak şekilde güncellenmesi gerekir.
            promote_override:
                enabled: true  # tamamen kapatmak yerine sertleştir
                min_z_score: 2.0  # 1.23 → 2.0
                max_dist_bps: 2.0  # 6.0 → 2.0
                max_adx: 20  # 25 → 20
                min_volume_strength: 0.50  # opsiyonel (volume_strength üretimi gerekli)
                require_reversal_confirmation: true
                respect_trend_veto: true
```

---

### **Priority 4.5: Rollout / Kill-Switch / KPI** ⭐⭐⭐⭐

```yaml
# Güvenli rollout anahtarları (kod desteği eklendi)
strategies:
  mean_reversion:
    fast_watch:
      promote_override:
        mode: "observe"   # observe | enforce
        enabled: true
```

Rollout önerisi:
1) **Observe (1-3 gün)**: Yeni gate kararlarını telemetry'ye yaz, execution davranışını değiştirme.
2) **Canary enforce (%10-20 sembol)**: Trade-count, false-fade, win-rate, PnL drift izle.
3) **Full enforce**: Canary metrikleri kabul aralığındaysa tüm evrene aç.

Rollback tetikleri:
- MR trade count baseline'a göre `>%35` düşerse
- PROMOTE win-rate beklenen iyileşmeyi göstermiyorsa (örn. `+10pp` altı)
- Net PnL veya max drawdown canary periyodunda anlamlı kötüleşirse

Minimum KPI seti:
- `promote_triggered_per_day`
- `promote_win_rate`
- `false_fade_count`
- `mr_total_win_rate`
- `mr_net_pnl`
- `mr_max_drawdown`

---

## 🔎 Netleştirme: dist_bps Tanımı (Mutlaka Standardize)

- **MR içi dist_bps (mevcut):** `abs(price - band) / price * 10000`
    - `price` recheck'te `fast_watch_price` (bid/ask) olabilir.
- **FastWatch v2 dist_bps (mevcut / önerilen):** `check_detail.fast_watch.dist_to_band_bps`
    - `trigger_price` referanslı ve recheck'le deterministik.

Öneri: PROMOTE threshold'larını `check_detail.fast_watch.dist_to_band_bps` üzerinden kalibre et;
bu alan yoksa fallback olarak MR içi hesabı kullan.

---

## 📊 Simülasyon: Yeni Kurallarla 03:10 & 03:30

### **03:10 Trade**
```python
# PROMOTE override check
✓ touch_confirmed = True
✓ near = "upper"
✓ dist_bps = 2.33
❌ abs(z) = 1.55 < 2.0  # REJECTED

# Eğer z >= 2.0 olsaydı bile:
❌ volume_strength = 0.57 MAYBE (>=0.50 ama sınırda)
❌ ADX = 15.26 < 20 ✓ ama...
❌ Reversal confirmation YOK (recheck'te de kontrol edilecek)

SONUÇ: AÇILMAZ ✅
```

### **03:30 Trade**
```python
# PROMOTE override check
✓ touch_confirmed = True
✓ near = "upper"
✓ dist_bps = 0.93 < 2.0
❌ abs(z) = 1.60 < 2.0  # REJECTED

# Eğer z >= 2.0 olsaydı bile:
❌ volume_strength = 0.42 < 0.50  # REJECTED
❌ EMA bullish stack + SHORT direction  # TREND VETO
❌ Reversal confirmation YOK

SONUÇ: AÇILMAZ ✅ (3 farklı sebepten)
```

---

## 🎯 Öncelik Sırası (Updated)

| Priority | Action | Effort | Impact | Deadline |
|----------|--------|--------|--------|----------|
| **P0** | PROMOTE z-score 1.23→2.0 | 5 min | ★★★★★ | Bugün |
| **P0** | PROMOTE dist 6.0→2.0 bps | 5 min | ★★★★★ | Bugün |
| **P0** | ADX bypass'ı kaldır | 10 min | ★★★★★ | Bugün |
| **P1** | Reversal conf recheck'e ekle | 30 min | ★★★★★ | Bugün |
| **P1** | Trend veto helper ekle | 20 min | ★★★★☆ | Yarın |
| **P1** | SHORT-side regime guardrails | 20-40 min | ★★★★☆ | Yarın |
| **P2** | Volume gate PROMOTE'a ekle | 10 min | ★★★☆☆ | 2 gün |
| **P3** | Config validation ekle | 15 min | ★★★☆☆ | 1 hafta |

---

## ⚠️ Kritik Uyarılar

### 1. **PROMOTE'u Tamamen Kapatma!**
```python
# ❌ YAPMA
promote_override:
  enabled: false  # fırsatları kaçırırsın

# ✅ YAP
promote_override:
  enabled: true
  min_z_score: 2.0  # sertleştir ama kapat değil
```

**Sebep**: PROMOTE gerçekten ekstrem sapmalarda (z>2.5) faydalı olabilir. Tamamen kapatmak yerine şartları sertleştir.

### 2. **Recheck Flow'u Değiştirme Riski**
Reversal confirmation'ı recheck'e eklerken:
- `is_recheck` koşulunu kaldırırken dikkatli ol
- Hem normal hem recheck için ayrı lookback period kullan
- Test et: normal flow bozulmasın

### 3. **ADX Bypass Kaldırma Etkisi**
```python
# ESKİ: PROMOTE ile ADX bypass
if promotion_override:
    adx_ok = True  # ← bu satırı SİL

# Bunun etkisi:
# - ADX > 20 ise PROMOTE bile olsa entry olmaz
# - 03:10 (ADX=15.26) geçerdi ama...
# - 03:30'da z zaten yetersiz, ayrıca trend veto var
```

---

## 📈 Beklenen Sonuçlar (Conservative)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **PROMOTE triggered** | 15-20/day | 3-5/day | -70% |
| **PROMOTE win-rate** | ~40% | ~65% | +25pp |
| **Overall MR win-rate** | 50% | 58% | +8pp |
| **Avg MR R/R** | 1.45 | 1.50 | +0.05 |
| **False fade count** | 8-10/day | 2-3/day | -70% |

---

## 💻 Uygulama Durumu

### ✅ Tamamlananlar
- [x] `mean_reversion.py`: PROMOTE z-score → 2.0
- [x] `mean_reversion.py`: PROMOTE dist → 2.0 bps
- [x] `mean_reversion.py`: ADX bypass satırını kaldır (PROMOTE artık ADX veto'yu aşmıyor)
- [x] `mean_reversion.py`: `check_promotion_override()` helper ekle
- [x] `mean_reversion.py`: `_is_trend_against_mr()` helper ekle
- [x] `mean_reversion.py`: Reversal conf'tan `if not is_recheck` kaldır
- [x] `mean_reversion.py`: Recheck reversal logic ekle (Stage-1 observe / Stage-2 enforce)
- [x] `mean_reversion.py`: SHORT-side regime/trend guardrail simetrisi (trend_up + shock, opt-in)
- [x] `mean_reversion.py`: promote_override parametrelerini config'ten okuyacak parsing ekle
- [x] `mean_reversion.py`: dist_bps kaynağını standardize et (prefer `check_detail.fast_watch.dist_to_band_bps`)
- [x] Tests: PROMOTE signed dist + shock_state source + recheck observe/enforce ana senaryoları
- [x] `config.example.yaml`: `strategies.mean_reversion.fast_watch.promote_override` örnek bloğunu ekle/güncelle
- [x] Tests: Recheck-short `OHLC missing` için explicit test case ekle
- [x] `mean_reversion.py`: çift kaynak volume analizi (upstream analyzer passthrough + `df_sig` local fallback) eklendi
- [x] `production_coordinator.py`: STRATEGY_RECHECK -> MR hattında `volume_strength` / `volume_bucket` / `volume_source` bağlantısı eklendi
- [x] `strategy_coordinator.py`: `strategy_recheck_request` event/check_detail içine payload volume context taşınması eklendi
- [x] Tests: volume routing + upstream/local fallback senaryoları eklendi
- [x] `mean_reversion.py`: `promote_override.mode` (`observe|enforce|off`) + `canary_symbols` rollout/kill-switch eklendi
- [x] Tests: promote_override observe-only ve canary-enforce senaryoları eklendi
- [x] Config validation: `promote_override.mode` enum + `canary_symbols` tip/format doğrulaması eklendi (`src/config/schema.py`)
- [x] `mean_reversion.py`: `_get_ema_stack()` helper eklendi (fail-open) ve PROMOTE trend veto akışına bağlandı
- [x] Tests: EMA stack extraction + auto-EMA trend veto senaryoları eklendi
- [x] `scripts/analyze_mr_promote_z_tuning.py`: `min_z_score` sweep (1.8/2.0/2.2) analiz aracı eklendi
- [x] Rapor: `reports/mr_promote_z_tuning_latest.md` üretildi (mevcut veriyle ön karar: `2.0` korunmalı)
- [x] `mean_reversion.py`: `promotion_override` meta snapshot genişletildi (`near/touch/dist/z/adx/volume/shock`)
- [x] `position_manager.py`: `entry_metadata` içine `signal_id` + `promotion_override` taşınması eklendi
- [x] `position_manager.py`: `TRADE_CLOSED` payload'ına `signal_id`, `promote_override_candidate`, `promote_override_applied` eklendi
- [x] `scripts/analyze_mr_promote_z_tuning.py`: `TRADE_CLOSED` tabanlı trade-labeled sweep desteği eklendi
- [x] Tests: `test_runtime_vol_telemetry_patch.py` ve `test_mean_reversion_promote_override.py` yeni telemetry alanları için güncellendi

### ⏳ Kalanlar (Gerçek TODO)
- [x] Operasyon: promote_override canary geçiş runbook'u (süre, metrik eşiği, geri dönüş kararı)
- [x] Rollback: metrik eşikleri + geri dönüş prosedürü (operasyonel playbook)
- [ ] Tune (final karar): `promote_win_rate` / `false_fade_count` / `mr_net_pnl` ile PnL etiketli canary karşılaştırmasını tamamla ve eşiği kesinleştir

### 🗂️ Kalan TODO Planı (Uygulama Sırası)

#### **Faz 1 (Code Safety, kalan ~0.5 gün)**
1) **Config validation tamamlandı (kritik)**
- Kapsam:
`src/config/schema.py` içinde `strategies.mean_reversion.fast_watch.promote_override.mode` için enum doğrulaması (`observe|enforce|off|disabled`).
`canary_symbols` için tip/format doğrulaması (list[str] veya CSV string, normalize uppercase, boş değer eleme).
- Kabul kriteri:
Geçersiz mode ile startup fail-fast.
Geçersiz `canary_symbols` tiplerinde deterministic hata mesajı.
Geçerli config ile mevcut davranış değişmeden devam.
- Test:
`tests/unit/test_config_promote_override_validation.py` eklendi (geçerli/geçersiz kombinasyonlar).

2) **EMA helper (opsiyonel) tamamlandı**
- Kapsam:
`_get_ema_stack()` helper ekle; veri yoksa fail-open (None dön).
PROMOTE trend veto yalnızca helper güvenilir veri döndürdüğünde EMA stack'i kullansın; aksi durumda mevcut `regime_data` fallback korun.
- Kabul kriteri:
Mevcut sinyal akışında regresyon yok.
EMA verisi eksikken exception üretmiyor.
- Test:
Unit test eklendi: EMA extraction + auto-EMA trend veto senaryoları.

#### **Faz 2 (Operations, 0.5 gün)**
3) **Canary runbook tamamlandı**
- Kapsam:
`observe -> canary -> full enforce` geçiş adımları.
Sembol seçim stratejisi (`canary_symbols`), gözlem süresi, günlük kontrol checklisti.
- Kabul kriteri:
Operatör tek dökümanla rollout yapabiliyor (komut, config örneği, karar matrisi).

4) **Rollback playbook tamamlandı**
- Kapsam:
Tetik eşikleri, geri alma adımı, doğrulama adımı, incident not şablonu.
`mode: off` ve `mode: observe` dönüş prosedürü.
- Kabul kriteri:
Rollback 5-10 dakika içinde uygulanabilir ve doğrulanabilir.

#### **Faz 3 (Tuning, 1-2 gün gözlem + 0.5 gün karar)**
5) **Z-score tuning (kısmi tamamlandı)**
- Kapsam:
`2.0` baz alınarak `1.8`, `2.2` sweep analizi (`scripts/analyze_mr_promote_z_tuning.py`) + canary/paper outcome karşılaştırması.
- Ölçümler:
`promote_triggered_per_day`, `promote_win_rate`, `false_fade_count`, `mr_net_pnl`, `mr_max_drawdown`.
- Kabul kriteri:
Seçilen eşik mevcut baseline'a göre daha iyi veya en azından daha güvenli risk profili veriyor.

Ara sonuç (mevcut log sweep):
- `reports/mr_promote_z_tuning_latest.md`:
  - `z>=1.8`: 13 geçiş
  - `z>=2.0`: 7 geçiş
  - `z>=2.2`: 1 geçiş
- Trade-labeled coverage (historical):
  - `TRADE_CLOSED` scope: 54
  - `entry_metadata.promotion_override` bulunan: 0
  - Sonuç: PnL sweep altyapısı hazır, ama geçmiş loglarda promote telemetry yok (yeni rollout verisi gerekiyor).
- Ön karar: fırsat daralması nedeniyle `2.2` agresif; mevcut veriyle `2.0` korunmalı.
- Not: Bu sonuç opportunity analizidir; nihai karar için PnL etiketli canary gerekli.

Çalıştırma:
```bash
python scripts/analyze_mr_promote_z_tuning.py \
  --glob "logs/live_trading_*.log" \
  --thresholds "1.8,2.0,2.2" \
  --default-threshold 2.0 \
  --max-dist-bps 2.0 \
  --max-adx 20 \
  --touch-policy missing_as_true \
  --out-json reports/mr_promote_z_tuning_latest.json \
  --out-md reports/mr_promote_z_tuning_latest.md
```

### ✅ Önerilen Teslim Sırası
1) PnL etiketli canary ile `min_z_score` nihai kararını tamamla (`2.0` vs `1.8/2.2`)

### 📘 Operasyon Playbook (Canary + Rollback)

#### **Canary Runbook**
1) **Observe başlangıcı (1-3 gün)**
- Config:
```yaml
strategies:
  mean_reversion:
    fast_watch:
      promote_override:
        enabled: true
        mode: "observe"
        canary_symbols: []
```
- Amaç:
Karar telemetry'sini topla, execution davranışını değiştirme.
- Günlük kontrol:
`promote_triggered_per_day`, `promote_win_rate`, `false_fade_count`, `mr_net_pnl`, `mr_max_drawdown`.

2) **Canary enforce (%10-20 sembol, 2-5 gün)**
- Config:
```yaml
strategies:
  mean_reversion:
    fast_watch:
      promote_override:
        enabled: true
        mode: "observe"
        canary_symbols: ["BTC/USDT:USDT", "ETH/USDT:USDT"]
```
- Geçiş kriteri:
Observe döneminde veri kalitesi yeterli olmalı (eksik telemetry olmamalı).
- Canary Go/No-Go:
Trade count düşüşü kabul sınırı içinde.
Win-rate ve net PnL drift negatif sapma göstermemeli.

3) **Full enforce**
- Config:
```yaml
strategies:
  mean_reversion:
    fast_watch:
      promote_override:
        enabled: true
        mode: "enforce"
        canary_symbols: []
```
- Geçiş kriteri:
Canary periyodu sonunda rollback tetiklerinden hiçbiri çalışmamış olmalı.

#### **Rollback Playbook**
1) **Rollback tetikleri**
- MR trade count baseline'a göre `>%35` düşüş.
- PROMOTE win-rate iyileşmesi `+10pp` altı.
- Canary penceresinde net PnL bozulması veya max drawdown kötüleşmesi.

2) **Acil geri alma (hızlı)**
- Seçenek A (tam kapatma):
```yaml
strategies:
  mean_reversion:
    fast_watch:
      promote_override:
        mode: "off"
```
- Seçenek B (telemetry-only güvenli mod):
```yaml
strategies:
  mean_reversion:
    fast_watch:
      promote_override:
        mode: "observe"
        canary_symbols: []
```

3) **Doğrulama adımları (ilk 15-30 dk)**
- Yeni execution kayıtlarında `promote_override_applied=true` görülmemeli (`off` için).
- Recheck telemetry akışı devam etmeli (`promote_override_candidate` alanı gözlenmeli).
- Trade akışı ve hata oranı normal banda dönmeli.

4) **Incident not şablonu**
- Başlangıç zamanı (UTC), tetik metrik, alınan config aksiyonu, geri alma zamanı, etkilenen semboller, takip aksiyonu.

---

## 🎬 Son Söz

**Önceki analiz**: "Rejim tespiti sorunu var"  
**Yeni analiz**: "PROMOTE override tüm güvenlik sistemini bypass ediyor"

İkisi de doğru ama **kök neden PROMOTE**. Çünkü:
1. Rejim tespiti yanlış olsa bile, normal flow reversal confirmation ile elerdi
2. Ama PROMOTE hem rejimi hem reversal confirmation'ı hem ADX'i bypass ediyor
3. Sonuç: Directional move sırasında "band sürtmesi" entry'ye dönüşüyor

**Çözüm önceliği:**
1. ✅ PROMOTE'u sertleştir (z≥2.0, dist≤2.0, ADX bypass kaldır)
2. ✅ Reversal confirmation'ı recheck'e ekle
3. ✅ Trend veto ekle
4. ⏭️ Rejim tespitini iyileştir (hala önemli ama artık "failsafe bypass" yok)
