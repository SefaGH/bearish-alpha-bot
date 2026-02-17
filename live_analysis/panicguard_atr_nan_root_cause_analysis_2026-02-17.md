# PanicGuard ATR=NaN Kök Neden Analizi

**Tarih:** 2026-02-17  
**Olay Zamanı:** 15:26:20 UTC  
**Sembol:** BTC/USDT:USDT (BingX)  
**Etkilenen Modül:** `src/core/strategy_coordinator.py` → `_compute_panic_state()`  
**Severity:** High — Doğru hesaplama yapılamaması nedeniyle sinyal haksız yere bloke edildi

---

## 1. Olay Özeti

Saat 15:26:20'de `adaptive_ob` stratejisi tüm base koşullarını geçerek bir **BUY sinyali** üretti:

- RSI: 26.9 ≤ 27.0 (eşik altı ✅)
- Persistency: 2 sample, 32s geçti ✅
- Risk/Reward: 1.50 ✅
- Smart Recovery TP: $67,502 ✅
- Volume bucket: HIGH ✅

Sinyal `StrategyCoordinator`'a iletildi ve **PanicGuard** tarafından reddedildi:

```
REJECTED (PanicGuard): panic_veto_no_reversal
  bucket=HIGH tf=5m drop=0.003159 atr=nan bear_body=0.787
  ema_gap_atr=None strict_extreme=False
  missing=rsi_hook,bull_candle_or_reclaim
```

Fiyat sonrasında $66,670'den $67,500+'ya bounce yaptı — sinyal kârlı olacaktı (~$770+/BTC).

---

## 2. Kök Neden Zinciri

### 2.1 Veri Akışı (Data Flow)

```
_compute_panic_state()
  │
  ├─ crash_cfg.get("panic_lookback_bars", 3) → limit = 3
  │
  ├─ market_data_pipeline.get_latest_ohlcv(symbol, tf="5m", limit=3, include_forming=True)
  │   │
  │   ├─ limit_override = 3  (limit parametresi dolu olduğu için)
  │   │
  │   ├─ limit_ws = limit_override or (ema_slow + warmup + safety)
  │   │            = 3 or (200 + ...)
  │   │            = 3                    ← ❌ BUG NOKTASI
  │   │
  │   ├─ ws_collector.get_latest_ohlcv(limit=3) → sadece 3 bar döner
  │   │
  │   └─ add_indicators(df_3_bars)
  │       │
  │       └─ atr(df, period=14) → ewm(min_periods=14) 
  │                              → 3 bar < 14 min_periods
  │                              → ATR = NaN (tüm satırlar)
  │
  ├─ last.get("atr") → float(NaN) = NaN
  │
  ├─ meta["atr"] = NaN                    ← NaN kaydediliyor
  ├─ meta["atr_pct"] = NaN / close = NaN  ← NaN propagasyonu
  ├─ meta["ema_fast_gap_atr"] = None      ← ATR > 0 guard'ı engeller
  │
  ├─ high_atr = False   (NaN >= threshold → False)
  ├─ ema_gap_panic = False (gap_atr = None)
  │
  └─ is_panic_state = fast_drop OR high_atr OR bearish_body OR ema_gap_panic
                     = False    OR False    OR True          OR False
                     = True  ← SADECE bearish_body ile tetiklendi
```

### 2.2 Sorunun Kaynağı

| Katman | Dosya | Satır | Sorun |
|--------|-------|-------|-------|
| **Çağıran** | `strategy_coordinator.py` | ~1015 | `limit = crash_cfg.get("panic_lookback_bars", 3)` → sadece 3 bar istiyor |
| **Pipeline** | `market_data_pipeline.py` | ~925 | `limit_ws = limit_override or (...)` → `limit_override=3` olduğu için warmup buffer **atlanıyor** |
| **Collector** | `stream_data_collector.py` | ~355 | `all_candles[-limit:]` → son 3 bar döndürülüyor |
| **Indicators** | `indicators.py` | ~47 | `ewm(min_periods=14)` → 3 < 14 → NaN |

**Temel neden:** `_compute_panic_state()` OHLCV verisi isterken `limit=3` kullanıyor. Bu, pipeline'ın indicator warmup buffer'ını bypass ediyor. ATR(14) hesaplamak için minimum 14 bar gerekli, ancak sadece 3 bar çekiliyor.

### 2.3 NaN'ın Dolaylı Etkileri

ATR=NaN olduğunda PanicGuard'ın 4 tetikleyicisinden 2'si devre dışı kalıyor:

| Tetikleyici | ATR Bağımlılığı | ATR=NaN ile Durum |
|-------------|-----------------|-------------------|
| `fast_drop` | Yok | ✅ Normal çalışır |
| `high_atr` | **Doğrudan** (`atr_pct = atr/close`) | ❌ NaN >= threshold → False → devre dışı |
| `bearish_body` | Yok | ✅ Normal çalışır |
| `ema_gap_panic` | **Doğrudan** (`gap = (ema-close)/atr`) | ❌ ATR > 0 guard → None → devre dışı |

Bu senaryoda `bearish_body=0.787` tek başına `is_panic_state=True` tetikledi. Ancak ATR bilgisi olsaydı, PanicGuard daha **informed** bir karar verebilirdi — örneğin, bearish body oranının ATR'ye göre normal mi yoksa anormal mi olduğunu değerlendirebilirdi.

---

## 3. Doğrulama

Containerda çalıştırılan test:

```python
# ATR with 3 bars, period=14
>>> atr_calc(df_3_bars, period=14)
[NaN, NaN, NaN]

# ATR with 20 bars, period=14
>>> atr_calc(df_20_bars, period=14)
200.0  ← Normal değer
```

Log'daki `atr=nan` çıktısı bu hesaplamanın doğrudan sonucudur.

---

## 4. Çözüm Önerisi

### Seçenek A: `_compute_panic_state` İçinde Limit Düzeltmesi (Önerilen)

`_compute_panic_state()` metodu, `panic_lookback_bars` değerini doğrudan `get_latest_ohlcv`'ye limit olarak geçmek yerine, ATR + EMA-fast warmup'ını kapsayan **dinamik yeterli bar** istemeli:

```python
# MEVCUT (HATALI):
lookback = int(crash_cfg.get("panic_lookback_bars", 3) or 3)
lookback = max(3, lookback)

# ÖNERİLEN DÜZELTME:
ind_cfg = market_data_pipeline.config.get("indicators", {})
atr_period = int(ind_cfg.get("atr_period", 14) or 14)
ema_fast_period = int(ind_cfg.get("ema_fast", 21) or 21)
indicator_warmup = max(atr_period, ema_fast_period) + 2
limit = max(lookback, indicator_warmup)
```

`+2` tamponu, hybrid/forming merge varyantlarında son satır göstergelerinin güvenli taşınması için bırakılır.  
Default değerlerle minimum limit tipik olarak `max(lookback, 23)` olur (EMA-fast=21 nedeniyle).

**Etki:** Minimal. WS collector zaten ~3000+ bar bellekte tutuyor, 3 yerine ~23 bar okumak performans farkı yaratmaz.

### Seçenek B: NaN Guard'ı Ekleme (Ek Koruma)

Sadece `meta["atr"]` atamasını değil, karar akışında kullanılan ATR/EMA değerlerinin tamamını finite guard ile normalize etme:

```python
import math

def _finite_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None

atr_val = _finite_float(last.get("atr"))
ema_fast_val = _finite_float(last.get("ema_fast"))
meta["atr"] = atr_val
meta["ema_fast"] = ema_fast_val
```

Ek olarak `high_atr` hesaplaması raw `last.get("atr")` yerine normalize `atr_val` üzerinden yapılmalı; aksi halde NaN yine `atr_pct` tarafına sızabilir.

### Seçenek C: Config Seviyesinde `panic_lookback_bars` Değerini Artırma (Geçici Çözüm)

Config'de `panic_lookback_bars: 20` ayarlanabilir. Ancak bu kök nedeni çözmez — config'e güvenmek kırılgan bir çözümdür. Birisi config'i 3'e geri çekerse sorun tekrarlar.

### Önerilen Uygulama Planı

1. **Seçenek A + B birlikte** uygulanmalı
2. Seçenek A kök nedeni ortadan kaldırır (ATR + EMA-fast için yeterli bar çekilir)
3. Seçenek B NaN/Inf değerlerin karar hesaplarına sızmasını engeller
4. Değişiklikler `strategy_coordinator.py` içinde `_compute_panic_state()` metodunda limit + hesaplama akışını birlikte günceller (~15-25 satır)

---

## 5. Etkilenen Dosyalar

| Dosya | Değişiklik Tipi |
|-------|-----------------|
| `src/core/strategy_coordinator.py` → `_compute_panic_state()` | Limit hesaplaması + NaN guard |

---

## 6. Risk Değerlendirmesi

| Risk | Seviye | Açıklama |
|------|--------|----------|
| Performans etkisi | Düşük | WS collector bellekte ~3000+ bar tutuyor, 3→~23 bar okumak ihmal edilebilir |
| Yan etki | Düşük | Sadece `_compute_panic_state` etkilenir, diğer stratejiler bu metodu kullanmaz |
| Geriye dönük uyumluluk | Sıfır | Config anahtarları değişmiyor, davranış düzeliyor |
| Düzeltme yapılmazsa risk | Yüksek | Her PanicGuard çağrısında ATR=NaN → eksik bilgiyle karar verme devam eder |

---

## 7. Kaçırılan Fırsat Etkisi (Bu Olay)

| Metrik | Değer |
|--------|-------|
| Giriş fiyatı (15:26:20) | $66,731.60 |
| Olası TP | $67,502.54 |
| Olası kâr | ~$770.94 / BTC (+1.16%) |
| Stop loss | $66,217.64 (hiç tetiklenmezdi, fiyat min $66,670'de kaldı) |
| Gerçekleşen sonuç | Bounce $67,500+'ya ulaştı — TP isabetliydi |
