## 🔍 BANT BELİRLEME YAPISI — KOD İNCELEME RAPORU

### Mimari Genel Bakış

Bant sistemi **3 katmanlı** çalışıyor:

| Katman | Dosya | İşlev |
|--------|-------|-------|
| **1. Pipeline** | indicators.py | Statik VWAP + rolling std hesaplama |
| **2. Controller** | mr_controller.py `_compute_band_multiplier()` | Adaptive multiplier (quantile targeting) |
| **3. Overlay** | mr_controller.py `_apply_adaptive_overlay()` | Volume + slope shift düzeltmesi |

---

### Aktif Konfigürasyon

```yaml
dynamic_controller:
  enabled: true
  target_outside_pct: 0.15        # Hedef: fiyatın %15'i bantların dışında
  abs_z_window: 500               # Z-score tarihçe penceresi
  warmup_samples: 50              # Isınma örnekleri
  update_interval_sec: 60         # Güncelleme aralığı
  m_min: 1.0 / m_max: 3.0        # Multiplier sınırları
  min_m_change: 0.05              # Min değişim eşiği
  dynamic_lookback: enabled, static=180, min=60, max=300
adaptive_settings:
  volume_weight: 1.0              # Tam volume etkisi
  volume_mult_min: 0.8 / max: 1.5
  slope_shift_mult: 1.0
```

---

### Tespit Edilen 6 Yapısal Problem

---

#### ⛔ Problem 1: std Hesaplama Metodolojisi Yanlış

indicators.py L133 ve mr_controller.py `_compute_vwap_and_std` satırlarında:

```python
# indicators.py L133:
vwap_std = out["close"].rolling(window=lookback, min_periods=lookback // 2).std()

# mr_controller.py _compute_vwap_and_std:
std = float(np.nanstd(close, ddof=1))  # close price std
```

**Sorun**: `std`, close fiyatlarının standart sapmasıdır — **fiyatın VWAP'tan sapmasının değil**. Klasik VWAP bantlarında std, `deviation_from_vwap = |close - vwap|` serisinden hesaplanmalıdır.

**Etkisi**: Trend sırasında fiyat sabit yönde ilerlediğinde, `close.std()` fiyatın kapsadığı **aralığı** ölçer. Fiyat VWAP etrafında dar toplanmış olsa bile, eğer 180 bar önceki fiyat bugünkünden çok farklıysa std şişer. Bu, bantların gereksiz yere genişlemesine neden olur.

**Öneri**: std hesaplamasını "deviation from VWAP" metoduna geçirin:

```python
# Mevcut (yanlış):
std = np.nanstd(close, ddof=1)

# Önerilen:
deviation = close - vwap_values  # her bar'ın VWAP'tan sapması
std = np.nanstd(deviation, ddof=1)
```

Bu değişiklik, trend sırasında std'nin çok daha kontrollü genişlemesini sağlar çünkü fiyat VWAP'ı takip ederken sapma düşük kalır.

---

#### ⛔ Problem 2: Çift Şişme (Double Inflation)

`_compute_band_multiplier()` metodunda:

```python
def _compute_band_multiplier(self, state):
    q = 1.0 - target          # 0.85
    hist = np.asarray(list(state.abs_z_hist))
    m_raw = float(np.quantile(hist, q))  # 85. yüzdelik z-score
    m_clamped = min(max(m_raw, m_min), m_max)
```

**Sorun**: `abs_z = |price - vwap| / std`. Yönlü hareket sırasında:
1. `|price - vwap|` büyür → abs_z yükselir → quantile yükselir → **multiplier artar**
2. Aynı anda `std` de büyür (close.rolling.std daha fazla varyans yakalar)

Bant genişliği formülü: `BW = multiplier × std`

Her iki terim de aynı anda büyüdüğü için bant genişliği **üstel** olarak genişler. 11 Şubat'ta 15 dakikada $691 → $2,007 (3x) genişleme gözlemlendi.

**Öneri**: Multiplier'ı _normalize edilmiş_ bir metriğe bağlayın:

```python
# 1. Opsiyon: std değişim oranını kompanse et
std_ratio = current_std / ema_of_std  # std ne kadar şişti?
compensation_factor = 1.0 / max(std_ratio, 1.0)  # sadece şişme sırasında düzelt
m_compensated = m_raw * compensation_factor

# 2. Opsiyon: Multiplier değişim hızını sınırla (ramp limiter)
max_increase_per_update = 0.10  # her 60s'de max %10 artış
m_new = min(m_raw, m_prev + max_increase_per_update)
```

---

#### ⛔ Problem 3: Yavaş Daralma (Slow Contraction)

Log verilerinden: 19:15'ten 21:15'e kadar 2 saat boyunca `outside_rate` sürekli %10-13 (hedef %15'in altında), ama multiplier sadece 2.01 → 1.85'e düştü (%8 azalma).

**Kök Neden (kodda)**:

```python
self._abs_z_window = 500  # 500 değerlik deque
```

Spike sırasında giren yüksek z-score'lar deque'de 500 güncelleme boyunca kalır. 1 güncelleme/dk = **~8.3 saat** flush süresi. Bu süre boyunca 85. yüzdelik stubbornly yüksek kalır.

**min_m_change=0.05** da mikro-düzeltmeleri engeller.

**Öneri**: Asimetrik adaptasyon mekanizması ekleyin:

```python
def _compute_band_multiplier(self, state):
    m_prev = state.last_band_multiplier
    m_raw = float(np.quantile(hist, q))
    
    outside_rate = self._current_outside_pct(state.abs_z_hist, m_prev)
    
    if outside_rate is not None and outside_rate < self._target_outside_pct:
        # Bantlar çok geniş — HIZLI daralt
        undershoot = self._target_outside_pct - outside_rate
        contraction_boost = 1.0 + (undershoot / self._target_outside_pct) * 2.0
        # Örnek: outside=%10, target=%15 → undershoot=%5 → boost=1.67
        m_raw = m_prev - abs(m_prev - m_raw) * contraction_boost
        m_raw = max(m_raw, self._m_min)
    
    # Ayrıca: kısa pencereli (son 60 bar) quantile ile uzun pencereli 
    # quantile arasında ağırlıklı ortalama al
    recent_hist = list(state.abs_z_hist)[-60:]
    m_recent = float(np.quantile(recent_hist, q))
    m_raw = 0.6 * m_recent + 0.4 * m_raw  # yakın tarih ağırlıklı
```

Alternatif olarak **abs_z_window'u 500 → 200'e düşürün** (flush süresi 8.3h → 3.3h).

---

#### ⛔ Problem 4: Container Restart Sonrası Bellek Kaybı

`hydrate_symbol_history()` metodu:

```python
def hydrate_symbol_history(self, symbol, df_vwap):
    subset = df_vwap.tail(required_samples)  # warmup_samples=50
    for _, row in subset.iterrows():
        z_score = (close - vwap) / vwap_std
        state.abs_z_hist.append(abs(z_score))
```

**Sorun**:
- Sadece 50 bar hidrasyonlanır ama pipeline VWAP 1440 bar lookback kullanır
- Container restart'ta multiplier=2.0'dan başlar (statik default)
- İlk 50 z-score ile quantile hesaplanır → çoğunlukla düşük z'ler → multiplier 1.0'a düşer
- Ardından gerçek volatilite gelince 1.0 → 1.81'e sıçrar (11 Şubat'taki gözlem)
- İlk 5-10 dakikada yanlış bantlarla çalışılır

**Öneri**:

```python
# 1. Hidrasyon miktarını artır
self._warmup_samples = max(200, self._abs_z_window // 2)

# 2. Ilk multiplier için pipeline bantlarını referans al
def hydrate_symbol_history(self, symbol, df_vwap):
    # Mevcut z-scoreleri doldur...
    count = ...
    
    # Yeterli veri varsa, başlangıç multiplier'ını pipeline'dan türet
    if count >= 50:
        m_bootstrap = float(np.quantile(
            [abs(x) for x in state.abs_z_hist], 
            1.0 - self._target_outside_pct
        ))
        m_bootstrap = min(max(m_bootstrap, self._m_min), self._m_max)
        state.last_band_multiplier = m_bootstrap
        logger.info(f"[MRController] Bootstrap mult={m_bootstrap:.3f} from {count} samples")

# 3. Isınma süresi boyunca konservatif mod
def _compute_band_multiplier(self, state):
    if len(state.abs_z_hist) < 100:  # warmup bitmeden
        # Statik değere yakın tut, radikal değişim yapma
        m_raw = max(m_raw, self._static_band_multiplier * 0.8)
```

---

#### ⚠️ Problem 5: Freeze-on-Trend ADX Etkileşimi

```python
if self._freeze_on_trend and adx >= self._adx_freeze_threshold:  # 36.0
    should_update = False
    reason = "freeze_on_trend"
```

**Sorun**: ADX 36+ olduğunda multiplier donduruluyor. Ama güçlü trend sırasında bantların genişlemesi gerekebilir (aksi halde fiyat sürekli bantların dışında kalır ve false signal üretir). Freeze mantıklı gibi görünse de, trend **bittiğinde** (ADX düşünce) multiplier eski (dar) değerden kalıyor ve ani bir güncelleme ile sıçrama yapabiliyor.

**Öneri**: Freeze yerine **yavaşlatma** (damping) uygulayın:

```python
if adx >= self._adx_freeze_threshold:
    # Tamamen dondurma yerine değişim hızını yavaşlat
    damping = 0.3  # %30 hızında güncelle
    m_new = m_prev + (m_raw - m_prev) * damping
else:
    m_new = m_raw
```

---

#### ⚠️ Problem 6: Volume Overlay Çok Agresif

```python
volume_weight: 1.0         # %100 ağırlık
volume_mult_max: 1.5       # +%50 genişletme
```

**Sorun**: Base multiplier zaten 2.0 iken, volume spike'ta `2.0 × 1.5 = 3.0` (m_max'a dayandı). Bu, Problem 2 (çift şişme) ile birleşince bantları aşırı geniş yapıyor.

**Öneri**:
```yaml
volume_weight: 0.5          # 1.0 → 0.5 (yarı etki)
volume_mult_max: 1.2         # 1.5 → 1.2 (max %20 genişleme)
```

---

### Öncelik Sıralaması

| # | Problem | Etkisi | Zorluk | Öncelik |
|---|---------|--------|--------|---------|
| 1 | std metodolojisi yanlış | Bantların temelini etkiler | Orta | **P0 — KRİTİK** |
| 2 | Çift şişme | 3x genişleme, kaçırılan fırsatlar | Orta | **P0 — KRİTİK** |
| 3 | Yavaş daralma | 2+ saat boyunca geniş bantlar | Düşük | **P1 — YÜKSEK** |
| 4 | Restart bellek kaybı | İlk 10 dk yanlış bantlar | Düşük | **P1 — YÜKSEK** |
| 5 | Freeze-on-trend sıçrama | ADX geçişlerinde instabilite | Düşük | **P2 — ORTA** |
| 6 | Volume overlay agresif | Spike'larda aşırı genişleme | Config | **P2 — ORTA** |

---

### Hemen Uygulanabilir Config Değişiklikleri (Kod değişikliği gerektirmez)

```yaml
dynamic_controller:
  abs_z_window: 200          # 500 → 200 (flush 8h → 3.3h)
  warmup_samples: 150        # 50 → 150 (daha güvenilir bootstrap)
  m_max: 2.5                 # 3.0 → 2.5 (üst sınırı daralt)
  min_m_change: 0.03         # 0.05 → 0.03 (daha hassas ayarlama)

adaptive_settings:
  volume_weight: 0.5         # 1.0 → 0.5
  volume_mult_max: 1.2       # 1.5 → 1.2
```

Bu 6 config değişikliği **kod değiştirmeden** 11 Şubat'taki problemlerin büyük kısmını hafifletir. Ama asıl çözüm **Problem 1 (std metodolojisi)** ve **Problem 2 (çift şişme kompansasyonu)** için kod değişikliği gerektirir.