### 1. Adım: Kör Noktayı Aydınlat (Audit & Log Fix)

*Öncelik: Hemen (Kodun içine girip değiştirmen gereken ilk yer)*

Analizdeki JSON yapısı mükemmel. Bunu `mr_controller.py` içine şu mantıkla yerleştir:

```python
# --- MEVCUT DURUM (YANLIŞ) ---
# log(pre_values)
# values = apply_overlay(values)
# return values

# --- OLMASI GEREKEN (DOĞRU) ---
# 1. Ham bantları hesapla
pre_lower, pre_upper = calculate_bands(...)

# 2. Overlay uygula (Düzeltilmiş mantıkla)
post_lower, post_upper, meta_data = self._apply_adaptive_overlay(...)

# 3. İKİSİNİ BİRLİKTE LOGLA (Tek satırda her şeyi gör)
log_payload = {
    "event": "mr_controller_decision",
    "pre_overlay": {"lower": pre_lower, "upper": pre_upper},
    "post_overlay": {
        "lower": post_lower, 
        "upper": post_upper, 
        "vol_multiplier": meta_data['vol_multiplier'],
        "vol_ratio": meta_data['vol_ratio']
    },
    "decision_price": current_price
}
logger.info(log_payload)

# 4. Sinyale ekle (Backtest ve Audit için)
signal["mr_controller"].update({
    "lower_post": post_lower,
    "upper_post": post_upper,
    "vol_multiplier": meta_data['vol_multiplier']
})

```

### 2. Adım: Kanamayı Durdur (Clamp & Hybrid Logic)

*Öncelik: Hemen (Bu, parayı koruyan kısımdır)*

Analizdeki **Seçenek A + C (Hibrid)** önerisini şu şekilde koda dökmelisin. Bu, botun sığ piyasada intihar etmesini engeller.

```python
def _apply_adaptive_overlay(self, ...):
    # Mevcut hacim oranı
    vol_ratio = current_15s_volume / avg_volume
    
    # 1. GÜVENLİK KİLİDİ (CLAMP): Asla 1.0'ın altına inme
    # Hacim düşükse bantları daraltma, olduğu gibi bırak.
    vol_multiplier = max(1.0, vol_ratio) 
    
    # 2. EK FİLTRE (OPSİYONEL AMA ÖNERİLİR)
    # Eğer piyasa "ölü" ise (örn: ortalamanın %30'u), işlem kalitesi (Z-score) çok yüksek olmalı.
    if vol_ratio < 0.3:
        required_z_score = 2.5 # Çok seçici ol
    else:
        required_z_score = 1.5 # Normal davran
        
    return m_final, required_z_score

```

### 3. Adım: Titremeyi Engelle (Smoothing - Seviye 2)

*Öncelik: Bu hafta içi*

Analizdeki en sofistike dokunuş burası. Hacim verisi anlık olarak çok sıçrar (bir mumda 100, diğerinde 1000 olabilir). Bu da bantların "titremesine" (jitter) yol açar. Bunu engellemek için **EWMA (Exponential Weighted Moving Average)** kullanmak profesyonel çözümdür.

```python
# Basit ortalama yerine, son değerlere ağırlık veren yumuşatma
# alpha=0.3 demek, yeni verinin etkisi %30, geçmişin etkisi %70 olsun demektir.
smoothed_vol_ratio = (current_vol_ratio * 0.3) + (previous_vol_ratio * 0.7)

# Multiplier hesabında bu yumuşatılmış değeri kullan
vol_multiplier = max(1.0, smoothed_vol_ratio)

```

### Özet Karar

Paylaştığın son analiz **%100 onaylandı**.

1. **Log yapısını** analizdeki JSON formatına çevir.
2. **`max(1.0, ...)` kuralını** hemen ekle.
3. Hafta sonuna doğru **EWMA (smoothing)** ekleyerek botun asabiyetini al.