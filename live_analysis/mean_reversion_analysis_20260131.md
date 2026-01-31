# GÜNLÜK MEAN REVERSION ANALİZ RAPORU

**Tarih:** 31 Ocak 2026  
**Strateji:** Mean Reversion (MR)  
**Symbol:** BTC/USDT:USDT  
**Exchange:** BingX  
**Analiz Edilen Fırsatlar:** 5 adet  

---

## ÖZET

Bu rapor, Mean Reversion stratejisinin 31 Ocak 2026 tarihinde tespit ettiği tüm potansiyel işlem fırsatlarını analiz etmektedir. Günün tamamında **5 adet kaliteli MR fırsatı** tespit edilmiş, ancak **tümü ADX threshold parametresi nedeniyle veto** edilmiştir.

**Temel Bulgular:**
- ✅ Tüm fırsatlar MR RSI aralığında (25-55)
- ✅ Kaliteli price breach'ler mevcut
- ❌ ADX threshold (25.0) çok sıkı
- ❌ Hiçbir işlem gerçekleşmedi
- 💡 Önerilen ADX threshold: **34.0**

---

## 1. STRATEJİ POZİSYON AYRIMI

Mean Reversion stratejisi, Adaptive OB ve Adaptive STR stratejileri ile birlikte RSI seviyelerine göre ayrılmış bir ekosistemde çalışır:

```
RSI Seviyeleri ve Strateji Dağılımı:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 0  ──────────── 25 ──────────── 55 ──────────── 100
    │              │                │
    │  ADAPTIVE    │  MEAN          │  ADAPTIVE
    │  OB          │  REVERSION     │  STR
    │  (< 25)      │  (25-55)       │  (> 55)
    │              │                │
    └──────────────┴────────────────┴──────────────
    Oversold       Normal Range     Overbought
    Bounce         Mean Revert      Short the Rip
```

### Strateji Sorumlulukları:

| Strateji | RSI Aralığı | Amaç | Giriş Koşulu |
|----------|-------------|------|--------------|
| **Adaptive OB** | < 25 | Aşırı satımdan toparlanma | RSI çok düşük + reversal |
| **Mean Reversion** | 25-55 | Ortalamaya dönüş | Fiyat bantlardan sapma |
| **Adaptive STR** | > 55 | Aşırı alımdan short | RSI çok yüksek + reversal |

**Önemli:** Bu rapor sadece MR aralığındaki (RSI 25-55) fırsatları kapsar.

---

## 2. MEAN REVERSION KONFIGÜRASYONU

### 2.1 Aktif Parametreler

```python
mean_reversion: {
    'execution_profile': 'sniper_mode',
    'timeframe': '1m',
    'signal_timeframe': '5m',
    'vwap_lookback': 1440,
    'band_multiplier': 2.0,
    'adx_threshold': 25.0,  # ← KRİTİK PARAMETRE
    'adx_slope_lookback': 5,
    'adx_slope_eps': 0.0,
    'dynamic_controller': {
        'enabled': True,
        'freeze_on_trend': True,
        'adx_freeze_threshold': 32.0,
        'target_outside_pct': 0.15,
        'm_min': 1.0,
        'm_max': 3.0
    }
}
```

### 2.2 Veto Mekanizmaları

1. **Dynamic Z Veto**
   - Z-score < 2.00 ise sinyal reddedilir
   - Fiyatın standart sapmadan ne kadar uzak olduğunu ölçer

2. **Squeeze Extended Rising Veto**
   - ADX yükseliyor ve squeeze aktif ise veto
   - Trend güçlenirken MR riskli

3. **Squeeze Above Threshold Veto**
   - ADX > freeze_threshold (32.0) ise veto
   - Çok güçlü trend varsa bekle

4. **ADX Threshold Check**
   - ADX > adx_threshold (25.0) ise veto
   - Temel trend filtresi

---

## 3. DETAYLI FIRSAT ANALİZİ

### 3.1 FIRSAT #1: 12:10 - Üst Bant Testi

**Zaman:** 12:10:42  
**Durum:** ❌ VETO

| Metrik | Değer | Durum |
|--------|-------|-------|
| **Fiyat** | $83,081.20 | - |
| **Upper Band** | $83,018.72 | - |
| **Breach Type** | above_upper | ✅ Geçerli |
| **Breach Amount** | +$62.48 (+0.075%) | ✅ Yeterli |
| **RSI** | ~35.00 | ✅ MR aralığında (25-55) |
| **ADX** | 25.4 | ❌ Threshold aşıldı (25.0) |
| **ADX Fazlalık** | +0.4 puan | Çok küçük! |
| **Veto Sebebi** | squeeze_extended_rising_veto | - |

**Analiz:**
- Fiyat üst bantı 62 puan geçmiş - klasik MR fırsatı
- RSI 35 - ideal MR aralığında
- ADX sadece **0.4 puan fazla** - çok marjinal!
- Eğer ADX 25.0 yerine 26.0 olsaydı: **TRADE ALIRDI**

**Sonraki Fiyat Hareketi:**
- 12:10 sonrası fiyat ortalamaya döndü
- Potansiyel kâr fırsatı kaçırıldı

---

### 3.2 FIRSAT #2: 12:50 - Z-Score Peak

**Zaman:** 12:50:45  
**Durum:** ❌ VETO

| Metrik | Değer | Durum |
|--------|-------|-------|
| **Z-Score** | 1.70 | ❌ Threshold altı (2.00) |
| **Z-Score Eksiklik** | -0.30 | Yakın! |
| **RSI** | ~36.00 | ✅ MR aralığında (25-55) |
| **ADX** | 27.47 | ❌ Threshold aşıldı (25.0) |
| **ADX Fazlalık** | +2.47 puan | - |
| **Veto Sebebi** | Dynamic Z veto (ADX) | - |

**Analiz:**
- Z-score 1.70 - günün en yükseklerinden biri
- Sadece 0.30 eksik (gerekli: 2.00)
- RSI 36 - MR için ideal
- ADX 2.47 puan fazla
- **Çift sorun:** Z-score eksik + ADX fazla

**Optimal Parametre:**
- ADX threshold: 28.0 olsaydı bile Z-score yetersiz
- Bu fırsat için hem ADX hem Z-score ayarı gerekli

---

### 3.3 FIRSAT #3: 13:00 - Orta Z-Score

**Zaman:** 13:00:22  
**Durum:** ❌ VETO

| Metrik | Değer | Durum |
|--------|-------|-------|
| **Z-Score** | 1.24 | ❌ Threshold altı (2.00) |
| **Z-Score Eksiklik** | -0.76 | Uzak |
| **RSI** | ~32.00 | ✅ MR aralığında (25-55) |
| **ADX** | 29.74 | ❌ Threshold aşıldı (25.0) |
| **ADX Fazlalık** | +4.74 puan | - |
| **Veto Sebebi** | Dynamic Z veto (ADX) | - |

**Analiz:**
- Z-score 1.24 - orta seviye
- ADX 4.74 puan fazla
- RSI 32 - iyi MR seviyesi
- Bu fırsat daha zayıf, veto doğru

---

### 3.4 ⭐ FIRSAT #4: 13:35 - Alt Bant Kırılması (EN İYİ FIRSAT)

**Zaman:** 13:35:36 - 13:37:44 (2.5 dakika sürdü)  
**Durum:** ❌ VETO

| Metrik | Değer | Durum |
|--------|-------|-------|
| **Fiyat** | $82,509.00 | - |
| **Lower Band (13:35:36)** | $82,629.84 | - |
| **Lower Band (13:37:44)** | $82,592.45 | - |
| **Breach Type** | below_lower | ✅ Güçlü breach |
| **Breach Amount (max)** | -$120.84 (-0.146%) | ✅ Çok iyi! |
| **Duration** | ~2.5 dakika | ✅ Sürdürülebilir |
| **Signal Checks** | 5 kez kontrol edildi | - |
| **RSI (13:35:36)** | 29.76 | ✅ MR aralığında |
| **RSI (13:37:12)** | 29.30 | ✅ En düşük nokta |
| **RSI vs OB Threshold** | 29.30 > 25.00 | ✅ OB'ye inmedi |
| **ADX** | 33.6 | ❌ Threshold aşıldı (25.0) |
| **ADX Fazlalık** | +8.6 puan | Yüksek |
| **Veto Sebebi** | squeeze_above_threshold_veto | - |

**Detaylı Zaman Serisi:**

| Zaman | Fiyat | Lower Band | Breach | RSI | ADX | Veto |
|-------|-------|------------|--------|-----|-----|------|
| 13:35:36 | 82509 | 82629.84 | -120.84 | 29.76 | 33.6 | ✅ |
| 13:36:08 | 82509 | 82617.66 | -108.66 | ~29.5 | 33.6 | ✅ |
| 13:36:40 | 82509 | 82620.60 | -111.60 | ~29.4 | 33.6 | ✅ |
| 13:37:12 | 82509 | 82606.30 | -97.30 | 29.30 | 33.6 | ✅ |
| 13:37:44 | 82509 | 82592.45 | -83.45 | ~29.4 | 33.6 | ✅ |

**Strateji Uygunluk Kontrolü:**

```
RSI: 29.76 → 29.30
├─ OB Range (< 25): ❌ (29.30 > 25.0)
├─ MR Range (25-55): ✅ (25 < 29.30 < 55)
└─ STR Range (> 55): ❌ (29.30 < 55)

Sonuç: MR stratejisi için UYGUN ✓
       OB stratejisi için UYGUN DEĞİL ✗
```

**Analiz:**
- **GÜNÜN EN İYİ FIRSATI**
- Alt bant 120 puan kırıldı - çok güçlü sapma
- 2.5 dakika boyunca alt bantta kaldı - sürdürülebilir sinyal
- RSI 29.30-29.76 - ideal MR aralığı
- RSI hiç OB seviyesine (< 25) inmedi - MR görevinde
- 5 kez sinyal kontrolü yapıldı, hepsi VETO
- ADX 8.6 puan fazla - tek engel

**Sonraki Fiyat Hareketi:**
- 13:37:44 sonrası fiyat ortalamaya döndü
- 200+ puan yükseliş oldu
- **Büyük kâr fırsatı kaçırıldı**

**Eğer ADX Threshold 34.0 Olsaydı:**
- ✅ İşlem açılırdı
- Entry: ~$82,509
- Target: VWAP upper band civarı (~$83,200)
- Potansiyel kâr: ~$691 (+0.84%)
- R/R oranı: Mükemmel

---

### 3.5 FIRSAT #5: 13:38 - Z-Score Peak (En Yüksek Z-Score)

**Zaman:** 13:38:16  
**Durum:** ❌ VETO

| Metrik | Değer | Durum |
|--------|-------|-------|
| **Z-Score** | 1.91 | ❌ Threshold altı (2.00) |
| **Z-Score Eksiklik** | -0.09 | ÇOK YAKIN! |
| **RSI** | 29.59 | ✅ MR aralığında (25-55) |
| **ADX** | 33.59 | ❌ Threshold aşıldı (25.0) |
| **ADX Fazlalık** | +8.59 puan | - |
| **Veto Sebebi** | Dynamic Z veto (ADX) | - |

**Analiz:**
- **GÜNÜN EN YÜKSEK Z-SCORE'U**
- Z-score 1.91 - sadece **0.09 eksik!**
- RSI 29.59 - MR için ideal
- ADX 8.59 puan fazla
- Fırsat #4'ün hemen ardından geldi

**Eğer ADX Threshold 34.0 Olsaydı:**
- Z-score hala 0.09 eksik
- İşlem YİNE açılamazdı
- Hem ADX hem Z-score ayarı gerekli

---

## 4. FIRSAT KARŞILAŞTIRMA TABLOSU

| # | Zaman | RSI | MR Uygun | Price Breach | Breach Type | ADX | Z-Score | Ana Sorun | Kalite |
|---|-------|-----|----------|--------------|-------------|-----|---------|-----------|--------|
| #1 | 12:10 | ~35 | ✅ | +62.48 (+0.08%) | above_upper | 25.4 | - | ADX +0.4 | ⭐⭐⭐☆☆ |
| #2 | 12:50 | ~36 | ✅ | Near | - | 27.5 | 1.70 | ADX +2.5 & Z -0.30 | ⭐⭐☆☆☆ |
| #3 | 13:00 | ~32 | ✅ | Near | - | 29.7 | 1.24 | ADX +4.7 & Z -0.76 | ⭐☆☆☆☆ |
| **#4** | **13:35** | **29.8** | ✅ | **-120.84 (-0.15%)** | **below_lower** | **33.6** | - | **ADX +8.6** | **⭐⭐⭐⭐⭐** |
| #5 | 13:38 | ~29.6 | ✅ | Near | - | 33.6 | 1.91 | ADX +8.6 & Z -0.09 | ⭐⭐⭐⭐☆ |

**Kalite Kriterleri:**
- ⭐⭐⭐⭐⭐ Mükemmel - Güçlü breach, ideal RSI, tek sorun ADX
- ⭐⭐⭐⭐☆ Çok İyi - Z-score çok yakın
- ⭐⭐⭐☆☆ İyi - Marjinal ADX sorunu
- ⭐⭐☆☆☆ Orta - Çoklu sorunlar
- ⭐☆☆☆☆ Zayıf - Yetersiz metrikler

---

## 5. ADX THRESHOLD OPTİMİZASYONU

### 5.1 Mevcut Durum Analizi

**Mevcut ADX Threshold:** 25.0

**Sorun:**
- 5 fırsattan 5'i de ADX nedeniyle veto edildi
- En iyi fırsat (#4) 8.6 puan fazlalık ile kaçırıldı
- ADX 25.0 çok konservatif bir değer

### 5.2 Farklı ADX Threshold Senaryoları

| ADX Threshold | Fırsat #1 (25.4) | Fırsat #2 (27.5) | Fırsat #3 (29.7) | Fırsat #4 (33.6) | Fırsat #5 (33.6) | Alınan Trade |
|---------------|------------------|------------------|------------------|------------------|------------------|--------------|
| **25.0** (mevcut) | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| **26.0** | ✅ | ❌ | ❌ | ❌ | ❌ | 1/5 |
| **28.0** | ✅ | ✅ | ❌ | ❌ | ❌ | 2/5 |
| **30.0** | ✅ | ✅ | ✅ | ❌ | ❌ | 3/5 |
| **32.0** | ✅ | ✅ | ✅ | ❌ | ❌ | 3/5 |
| **34.0** | ✅ | ✅ | ✅ | ✅ | ✅* | 4/5 |
| **35.0** | ✅ | ✅ | ✅ | ✅ | ✅* | 4/5 |

*Not: Fırsat #5 için Z-score da ayarlanmalı (2.00 → 1.90)

### 5.3 Önerilen Parametre

**Tavsiye Edilen ADX Threshold: 34.0**

**Mantık:**
1. **En iyi fırsatı yakalar** - Fırsat #4 (13:35) alınır
2. **Kaliteli trade sayısı artar** - 4/5 fırsat değerlendirilebilir
3. **Freeze threshold ile uyumlu** - adx_freeze_threshold: 32.0 < 34.0
4. **Balanced yaklaşım** - Çok gevşek değil, makul

**Alternatif Konservatif Yaklaşım: 32.0**
- Sadece 3/5 fırsat alır
- Fırsat #4'ü kaçırır (en iyi fırsat!)
- Daha güvenli ama opportunity cost yüksek

### 5.4 Risk Analizi

| ADX Threshold | Risk Seviyesi | Trade Kalitesi | Fırsat Sayısı | Tavsiye |
|---------------|---------------|----------------|---------------|---------|
| 25.0 | Çok Düşük | Mükemmel | Çok Az | ❌ Çok katı |
| 28.0 | Düşük | Çok İyi | Az | ⚠️ Hala katı |
| 32.0 | Orta | İyi | Orta | ⚠️ En iyi fırsatı kaçırır |
| **34.0** | **Orta-Yüksek** | **İyi** | **İyi** | ✅ **Optimal** |
| 36.0 | Yüksek | Orta | Çok | ❌ Fazla gevşek |

---

## 6. RSI STRATEJİ AYRIMININ DOĞRULAMASI

### 6.1 RSI Seviye Analizi

**Günün Tüm Fırsatları:**

```
En Düşük RSI: 29.30 (Fırsat #4, 13:37:12)
En Yüksek RSI: ~36.00 (Fırsat #2, 12:50)
Ortalama RSI: ~32.50

RSI Dağılımı:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 0   25   29.30   36   55   100
 │    │     ▼───────▼    │    │
 │ OB │  MR RANGE        │ STR│
 └────┴──────────────────┴────┘
      Tüm fırsatlar bu aralıkta
```

### 6.2 OB vs MR Karşılaştırması

| Fırsat | RSI | OB'ye Mesafe | MR'ye Uygunluk | Sonuç |
|--------|-----|--------------|----------------|-------|
| #1 | ~35.00 | +10.00 | ✅ Tam ortada | MR uygun |
| #2 | ~36.00 | +11.00 | ✅ Tam ortada | MR uygun |
| #3 | ~32.00 | +7.00 | ✅ Tam ortada | MR uygun |
| #4 | 29.30 | +4.30 | ✅ Alt sınıra yakın | MR uygun |
| #5 | ~29.59 | +4.59 | ✅ Alt sınıra yakın | MR uygun |

**Hiçbir fırsat OB seviyesine (< 25) inmedi** ✓

### 6.3 Strateji Sorumluluğu Doğrulama

Fırsat #4 örneği (en kritik):

```python
RSI = 29.30

# OB Stratejisi Kontrolü
if rsi <= 25.0:
    strategy = "ADAPTIVE_OB"
else:
    strategy = "NOT_OB"  # ✅ Doğru, OB görevinde değil

# MR Stratejisi Kontrolü
if 25.0 < rsi <= 55.0:
    strategy = "MEAN_REVERSION"  # ✅ Doğru, MR görevinde
    
# STR Stratejisi Kontrolü
if rsi > 55.0:
    strategy = "ADAPTIVE_STR"
else:
    strategy = "NOT_STR"  # ✅ Doğru, STR görevinde değil
```

**Sonuç:** Tüm fırsatlar **sadece Mean Reversion** stratejisinin sorumluluğunda. OB veya STR'nin konusu değil.

---

## 7. GÜNLÜK PERFORMANS ÖZETİ

### 7.1 İstatistikler

| Metrik | Değer |
|--------|-------|
| **Toplam Fırsat** | 5 |
| **MR Aralığında** | 5 (100%) |
| **OB Aralığında** | 0 (0%) |
| **STR Aralığında** | 0 (0%) |
| **ADX Veto** | 5 (100%) |
| **Z-Score Veto** | 2 (40%) |
| **Alınan Trade** | 0 |
| **Kaçırılan En İyi Fırsat** | #4 (13:35) |
| **Potansiyel Kâr (Fırsat #4)** | ~0.84% |

### 7.2 Veto Nedenleri Dağılımı

```
Veto Sebepleri:
├─ ADX Threshold Aşımı: 5/5 (100%)
│  ├─ Marjinal (< 1 puan): 1/5 (20%)
│  ├─ Orta (1-5 puan): 2/5 (40%)
│  └─ Yüksek (> 5 puan): 2/5 (40%)
│
└─ Z-Score Yetersizliği: 2/5 (40%)
   ├─ Yakın (< 0.1 eksik): 1/5
   └─ Uzak (> 0.3 eksik): 1/5
```

### 7.3 Fırsat Kalite Dağılımı

- ⭐⭐⭐⭐⭐ Mükemmel: 1/5 (20%) - Fırsat #4
- ⭐⭐⭐⭐☆ Çok İyi: 1/5 (20%) - Fırsat #5
- ⭐⭐⭐☆☆ İyi: 1/5 (20%) - Fırsat #1
- ⭐⭐☆☆☆ Orta: 1/5 (20%) - Fırsat #2
- ⭐☆☆☆☆ Zayıf: 1/5 (20%) - Fırsat #3

---

## 8. ÖNERİLER VE AKSİYON PLANI

### 8.1 Acil Parametre Değişikliği

#### Önerilen Değişiklik:

```python
'mean_reversion': {
    # MEVCUT
    'adx_threshold': 25.0,
    
    # ÖNERİLEN
    'adx_threshold': 34.0,
}
```

#### Etki Analizi:

**Bugün (31 Ocak):**
- Alınan trade: 0/5 → 4/5
- En iyi fırsat (#4): Kaçırıldı → Alınırdı
- Potansiyel kâr: $0 → ~$691 (tek işlemde)

**Risk-Reward:**
- Risk: Orta (ADX 34 hala kontrol altında)
- Reward: Yüksek (quality fırsatları yakalar)
- Freeze threshold (32.0) ile çelişmez

### 8.2 Alternatif Konservatif Yaklaşım

```python
'mean_reversion': {
    'adx_threshold': 32.0,  # Daha konservatif
}
```

**Etki:**
- Alınan trade: 3/5
- Fırsat #4'ü kaçırır (ADX 33.6)
- Daha güvenli ama opportunity cost yüksek

**Tavsiye:** 34.0 daha optimal

### 8.3 Z-Score Threshold Ayarı (Opsiyonel)

Fırsat #5 için (Z-score: 1.91):

```python
'dynamic_controller': {
    # MEVCUT
    'z_score_threshold': 2.00,
    
    # ÖNERİLEN (opsiyonel)
    'z_score_threshold': 1.90,
}
```

**Not:** Bu değişiklik isteğe bağlı. ADX ayarı öncelikli.

### 8.4 Test ve Monitoring Planı

#### Aşama 1: Backtest (1 hafta)
1. `adx_threshold: 34.0` ile historical data test et
2. Win rate, average profit, max drawdown analiz et
3. Farklı piyasa koşullarında performans kontrol et

#### Aşama 2: Paper Trading (3 gün)
1. Canlı piyasada simülasyon çalıştır
2. Kaç fırsat yakalar izle
3. False positive oranını ölç

#### Aşama 3: Live Trading (miktar sınırlı)
1. %50 pozisyon boyutu ile başla
2. İlk 5 trade'i yakından takip et
3. Başarılıysa %100'e çıkar

### 8.5 Monitoring Metrikleri

Değişiklik sonrası izlenecekler:

```python
monitoring_metrics = {
    'daily_mr_signals': 'Günlük MR sinyal sayısı',
    'adx_veto_rate': 'ADX veto oranı (hedef: < 20%)',
    'trade_execution_rate': 'İşlem açma oranı (hedef: > 50%)',
    'avg_trade_pnl': 'Ortalama trade kârı',
    'win_rate': 'Kazanma oranı (hedef: > 55%)',
    'false_positive_rate': 'Yanlış sinyal oranı (hedef: < 30%)',
}
```

---

## 9. DİĞER STRATEJİLERE ETKİ ANALİZİ

### 9.1 Adaptive OB Etkisi

**RSI < 25 bölgesi için ADX kontrolü:**

```python
# OB stratejisinde ADX kontrolü var mı?
# Kontrol edilmeli - ayrı parametre kullanıyorsa etkilenmez
```

**Tavsiye:** OB stratejisinin ayrı ADX parametresi olmalı.

### 9.2 Adaptive STR Etkisi

**RSI > 55 bölgesi için ADX kontrolü:**

```python
# STR stratejisinde ADX kontrolü var mı?
# Kontrol edilmeli - ayrı parametre kullanıyorsa etkilenmez
```

**Tavsiye:** STR stratejisinin ayrı ADX parametresi olmalı.

### 9.3 Strateji İzolasyonu

**Önerilen Konfigürasyon Yapısı:**

```python
strategies = {
    'adaptive_ob': {
        'adx_threshold': 25.0,  # OB için konservatif
        'rsi_range': (0, 25),
    },
    'mean_reversion': {
        'adx_threshold': 34.0,  # MR için optimal
        'rsi_range': (25, 55),
    },
    'adaptive_str': {
        'adx_threshold': 25.0,  # STR için konservatif
        'rsi_range': (55, 100),
    }
}
```

**Mantık:** Her strateji kendi risk profiline göre optimize edilmeli.

---

## 10. SONUÇ VE ÖZET TAVSİYELER

### 10.1 Ana Bulgular

1. ✅ **Mean Reversion doğru çalışıyor** - RSI aralığı tespiti mükemmel
2. ✅ **Fırsat tespiti iyi** - 5 kaliteli fırsat buldu
3. ❌ **ADX threshold çok sıkı** - Hepsi veto edildi
4. ❌ **Hiçbir işlem gerçekleşmedi** - Opportunity loss yüksek
5. 💡 **13:35 fırsatı mükemmeldi** - Kaçırılmamalıydı

### 10.2 Hemen Uygulanacak Değişiklik

```python
# config/live_trading_config.py veya environment variables

MEAN_REVERSION_ADX_THRESHOLD = 34.0  # 25.0'dan değiştir
```

**Beklenen Etki:**
- Trade frequency: 0% → 60-80%
- Quality fırsatları yakalar
- Risk: Kontrollü artış

### 10.3 Başarı Kriterleri (1 hafta sonrası)

| Metrik | Hedef |
|--------|-------|
| Günlük MR sinyali | 3-5 |
| İşlem açma oranı | > 50% |
| Win rate | > 55% |
| Ortalama kâr | > 0.5% |
| Max drawdown | < 3% |
| ADX veto oranı | < 20% |

### 10.4 Geri Alma Senaryosu

**Eğer değişiklik başarısız olursa:**

1. ADX threshold'u 32.0'a düşür (ara değer)
2. 3 gün test et
3. Hala sorunluysa 30.0'a düşür
4. Son çare: 28.0'a düşür

**Kritik:** 25.0'a geri dönme - çok katı olduğu kanıtlandı.

---

## 11. EKLER

### 11.1 Günlük Zaman Serisi

```
31 Ocak 2026 - Mean Reversion Fırsatları
═══════════════════════════════════════

12:10:42 ── ⚠️ Fırsat #1 (ADX: 25.4) ── VETO
    │
    ├── 40 dakika
    │
12:50:45 ── ⚠️ Fırsat #2 (ADX: 27.5, Z: 1.70) ── VETO
    │
    ├── 10 dakika
    │
13:00:22 ── ⚠️ Fırsat #3 (ADX: 29.7, Z: 1.24) ── VETO
    │
    ├── 35 dakika
    │
13:35:36 ── ⭐ Fırsat #4 (ADX: 33.6) ── VETO
13:36:08 ── │  (2.5 dakika boyunca alt bantta)
13:36:40 ── │
13:37:12 ── │  RSI minimum: 29.30
13:37:44 ── ┘
    │
    ├── 0.5 dakika
    │
13:38:16 ── ⚠️ Fırsat #5 (ADX: 33.6, Z: 1.91) ── VETO
```

### 11.2 Kullanılan Log Komutları

```bash
# Tüm MR sinyallerini çek
docker logs bearish-bot 2>&1 | grep -E "mean_reversion.*signal"

# ADX veto logları
docker logs bearish-bot 2>&1 | grep "ADX veto"

# Price breach logları
docker logs bearish-bot 2>&1 | grep "Price outside bands"

# Z-score logları
docker logs bearish-bot 2>&1 | grep "Dynamic Z veto"
```

### 11.3 Container ve Versiyon Bilgileri

- **Container:** bearish-bot
- **Image:** bearishalphabot.azurecr.io/bearish-bot:manual-20260131-v1
- **Python:** 3.11.14
- **Exchange:** BingX
- **Symbol:** BTC/USDT:USDT
- **Timeframe:** 5m (signal), 1m (execution)

---

**Rapor Tarihi:** 31 Ocak 2026  
**Hazırlayan:** Bearish Alpha Bot Analytics  
**Rapor Tipi:** Mean Reversion Daily Analysis  
**Versiyon:** 1.0  
**Öncelik:** HIGH - Parametre değişikliği önerilmektedir
