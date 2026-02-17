# Risk/Reward Dynamic Target Uyumsuzluk Analizi

**Tarih:** 2026-02-17  
**Sembol:** BTC/USDT:USDT  
**Zaman Aralığı:** 5m  
**İncelenen Olay:** 15:34:21 UTC — adaptive_ob LONG sinyali R/R filtresi tarafından reddedildi  
**Sonuç:** Gerçek R/R = 1.50, Dinamik Hedef = 1.90 → **REJECTİON** (Fırsat kaçırıldı)

---

## 1. Olay Özeti

Adaptive_ob stratejisi BTC/USDT:USDT için 15:34:21'de bir **LONG** sinyali üretti. Sinyal aşağıdaki parametrelerle risk değerlendirmesine gönderildi:

| Parametre | Değer |
|-----------|-------|
| Entry | 66,698.70 |
| Stop Loss | 66,180.67 |
| Take Profit (ATR 1.80x) | 67,475.75 |
| Risk | 0.8% (518.03 USDT) |
| Reward | 1.2% (777.05 USDT) |
| Gerçek R/R | **1.50** |
| Dinamik R/R Hedefi | **1.90** |

**Sonuç:** `🚫 [RiskRewardRatioRule] REJECTED: Risk/reward ratio 1.50 is below dynamic target 1.90`

Fiyat sonrasında 67,500+ seviyesine çıkmış ve TP'ye ulaşmıştır. **Bu fırsat doğru bir şekilde değerlendirilmeliydi.**

---

## 2. Dinamik R/R Formülü

```
dynamic_target_pre_ppo = (base_rr - relaxation + tightening) × regime_adjustment
dynamic_target = dynamic_target_pre_ppo × ppo_rr_multiplier
final = CLAMP(max(dynamic_target, strategy_floor), lower_bound, upper_bound)
```

**Kaynak:** `src/core/risk_rules.py` → `_calculate_dynamic_target()` (satır 702+)

---

## 3. Gerçek Hesaplama Akışı (Log Verileri)

```
Base=1.50 - Relax=0.17 + Tight=0.14 × Regime(mult=1.0, weight=1.00)=1.00 = 1.46 × PPO(1.30) = 1.90 → Final=1.90
```

### Girdi Değerleri:

| Parametre | Değer | Kaynak | Sorunlu mu? |
|-----------|-------|--------|:-----------:|
| base_target_rr | 1.50 | Config default | ✅ Normal |
| ml_confidence | 0.50 | Fallback (ML yok) | ⚠️ |
| rl_is_agree | False | PPO HOLD action | ⚠️ |
| rl_action_prob | 0.00 | PPO score | ⚠️ |
| regime_name | `{'symbol': 'BTC/USDT:USDT', ...}` (DICT) | Bug! | ❌ BUG |
| regime_confidence | 0.30 | Fallback (gerçek: 0.186) | ❌ BUG |
| regime_weight | 1.00 | Default (hard-reject bypass) | ❌ TASARIM HATASI |
| ppo_rr_multiplier | 1.30 | Inactive PPO → rr_up_mult | ❌ TASARIM HATASI |
| volume_strength | 0.85 | EXTREME (v1'de kullanılmaz) | ✅ Normal |
| momentum_strength | 0.44 | (v1'de kullanılmaz) | ✅ Normal |

---

## 4. Tespit Edilen 4 Sorun

### 4.1 🔴 SORUN #1: PPO Multiplier — İnaktif PPO'dan 1.30x Çarpan (KRİTİK)

**Konum:** `src/core/strategy_coordinator.py` satır 9261-9266

```python
ppo_rr_multiplier = 1.0
if side in ('buy', 'long') and 'ppo_long_score' in signal:
    score = float(signal['ppo_long_score'])
    ppo_rr_multiplier = (
        self.ppo_multipliers['rr_up_mult'] if score < 0.5 else self.ppo_multipliers['rr_down_mult']
    )
```

**Config değerleri** (`strategy_coordinator.py` satır 762-763):
```python
'rr_up_mult': float(rl_cfg.get('ppo_rr_up_mult', 1.3)),   # PPO karşı → %30 artır
'rr_down_mult': float(rl_cfg.get('ppo_rr_down_mult', 0.9)), # PPO onay → %10 azalt
```

**Ne oldu:**
1. PPO adapter aktif ancak **eğitilmemiş/inaktif** (action=HOLD, score=0.00, tüm örnekler flat)
2. Log: `🤖 [PPO-DECISION] BTC/USDT:USDT | Action: HOLD | Score: 0.00 | Conf: 1.00`
3. Log: `📈 [RL-TELEMETRY] RL inactive | PPO samples=5 | avg_score=0.000 | long=0 | flat=5`
4. `ppo_long_score = 0.00` → sinyale ekleniyor → `score < 0.5` → `rr_up_mult = 1.3` uygulanıyor
5. Sonuç: R/R hedefi %30 şişiriliyor (1.46 × 1.30 = 1.90)

**Neden yanlış:**
- PPO aktif olarak "karşı çıkmıyor" — hiç çalışmıyor, eğitilmemiş
- `flat=5` → 5 örnekten 5'i HOLD → model sadece default "hiçbir şey yapma" davranışı gösteriyor
- İnaktif bir model'in "disagree" sinyali vermesi mantıksal olarak yanlış
- Bu, sinyalin geçebileceği ile reddedilmesi arasındaki fark (1.46 vs 1.90)

**Etki:** Dinamik hedefi 1.46'dan 1.90'a çıkardı → **tek başına ret nedeni**

---

### 4.2 🔴 SORUN #2: regime_name Dict Bug

**Konum:** `src/core/strategy_coordinator.py` satır 9086

```python
signal['regime_name'] = str(ml_context.get('regime', 'neutral'))
```

**Ne oldu:**
1. `get_ml_context()` → `context["regime"] = regime` (satır 600, `strategy_integration.py`)
2. `regime` = `predict_regime_transition()` sonucu → **tam dict objesi**:
   ```python
   {'symbol': 'BTC/USDT:USDT', 'horizon': '1h', 'predicted_regime': 'bearish',
    'probabilities': {...}, 'confidence': 0.186, 'quality_score': 0.70, ...}
   ```
3. `str(dict)` → `"{'symbol': 'btc/usdt:usdt', ...}"` (dict'in string hali)
4. `risk_rules.py` satır 773: `regime_name = signal.get('regime_name', 'neutral').lower()`
5. `regime_mults.get(regime_name, 1.0)` → **eşleşme yok** → `regime_mult = 1.0` (default)

**Olması gereken:**
- `regime_name = 'bearish'` → `regime_mult = 0.9`
- `regime_adjustment = 1.0 + (0.9 - 1.0) × 1.0 = 0.90`

**Gerçekleşen:**
- `regime_name = "{'symbol': ...}"` → `regime_mult = 1.0`
- `regime_adjustment = 1.0 + (1.0 - 1.0) × 1.0 = 1.00`

**Etki:** Bearish rejimde hedef %10 düşürülmeliydi (×0.9), ama düşürülmedi (×1.0)

**Çözüm:**
```python
# BUGGY:
signal['regime_name'] = str(ml_context.get('regime', 'neutral'))

# FIX:
regime_raw = ml_context.get('regime')
if isinstance(regime_raw, dict):
    signal['regime_name'] = str(regime_raw.get('predicted_regime', 'neutral'))
else:
    signal['regime_name'] = str(regime_raw or 'neutral')
```

---

### 4.3 🟡 SORUN #3: regime_confidence Fallback Hatası

**Konum:** `src/core/strategy_coordinator.py` satır 9088

```python
signal['regime_confidence'] = float(ml_context.get('regime_confidence', 0.3))
```

**Ne oldu:**
1. `ml_context` dict yapısı:
   ```python
   {"is_healthy": True, "prediction": {...}, "regime": {FULL_DICT}, "reason": ""}
   ```
2. `ml_context` dict'inde `'regime_confidence'` anahtarı **yok** → fallback `0.3` kullanılıyor
3. Gerçek confidence `0.186` → regime dict içinde `regime['confidence']` olarak mevcut
4. Dinamik R/R hesabında `regime_conf = 0.30` kullanıldı (gerçek: 0.186)

**Etki:**
- Tightening hesabı: `0.2 × (1 - regime_conf) × regime_weight`
- Gerçek değerle: `0.2 × (1 - 0.186) × 1.0 = 0.163` (daha fazla tightening)
- Fallback ile: `0.2 × (1 - 0.30) × 1.0 = 0.140` (daha az tightening)
- Bu özel durumda fallback **daha düşük** hedef verdi (0.02 fark)
- Ancak bu tesadüfi — farklı confidence değerlerinde ters yönde etki edebilir

**Çözüm:**
```python
# BUGGY:
signal['regime_confidence'] = float(ml_context.get('regime_confidence', 0.3))

# FIX:
regime_raw = ml_context.get('regime')
if isinstance(regime_raw, dict):
    signal['regime_confidence'] = float(regime_raw.get('confidence', 0.3))
else:
    signal['regime_confidence'] = 0.3
```

---

### 4.4 🟡 SORUN #4: regime_weight Default Gap (Tasarım Hatası)

**Akış Çelişkisi:**

```
strategy_integration.py (satır 144-152):
  regime_conf = 0.186
  regime_weight = _calculate_regime_weight(0.186)
  → 0.186 < 0.30 (hard_reject) → return None
  → "Regime ignored" → regime_weight SİNYALE EKLENMEDİ

risk_rules.py (satır 773):
  regime_weight = signal.get('regime_weight', 1.0)  ← DEFAULT 1.0!
  → Tam güvenle tightening uygulandı
```

**Çelişki:**
- `strategy_integration.py`: "Confidence çok düşük (0.186 < 0.30), regime tahminini tamamen YOKSAY"
- `risk_rules.py`: `regime_weight=1.0` → "Regime tahminine TAM GÜVEN, maksimum tightening uygula"
- Bu iki davranış birbirinin tam tersi

**Neden oluyor:**
- `strategy_integration` hard-reject durumunda `regime_weight` alanını sinyale hiç eklemiyor
- `risk_rules.py` bu alanı bulamayınca `1.0` (full confidence) olarak default alıyor
- Sonuç: düşük güvenilirlikli regime tahmini → **paradoks olarak maksimum tightening etkisi**

**Etki:**
- `tightening = 0.2 × (1 - 0.30) × 1.0 = 0.14` (tam ağırlıkla uygulandı)
- Doğru davranış: `regime_weight = 0` veya çok düşük değer → tightening ≈ 0

---

## 5. Senaryo Analizi: "What-If"

Her fix kombinasyonu ile dinamik hedefin nasıl değişeceği:

### Mevcut Durum (Tüm bug'lar aktif):
```
relaxation = 0.17
tightening = 0.14
regime_adjustment = 1.0
pre_ppo = (1.50 - 0.17 + 0.14) × 1.0 = 1.47
target = 1.47 × 1.30 = 1.91 → Final=1.90
Sinyal R/R: 1.50 < 1.90 → ❌ REJECTED
```

### Senaryo A: Sadece PPO fix (inaktif → multiplier=1.0):
```
target = 1.47 × 1.0 = 1.47
Sinyal R/R: 1.50 > 1.47 → ✅ PASSED
```

### Senaryo B: Sadece regime_name fix (bearish → mult=0.9):
```
regime_adjustment = 1.0 + (0.9-1.0) × 1.0 = 0.90
pre_ppo = 1.47 × 0.9 = 1.323
target = 1.323 × 1.30 = 1.72
Sinyal R/R: 1.50 < 1.72 → ❌ REJECTED
```

### Senaryo C: PPO + regime_name fix:
```
pre_ppo = 1.47 × 0.9 = 1.323
target = 1.323 × 1.0 = 1.32
Sinyal R/R: 1.50 > 1.32 → ✅ PASSED
```

### Senaryo D: Tüm fix'ler (PPO + regime_name + regime_conf + regime_weight):
```
relaxation = 0.17
tightening = 0.2 × (1 - 0.186) × 0.0 = 0.0  (conf < hard_reject → weight=0)
regime_adjustment = 1.0 + (0.9-1.0) × 0.0 = 1.0
pre_ppo = (1.50 - 0.17 + 0.0) × 1.0 = 1.33
target = 1.33 × 1.0 = 1.33
Sinyal R/R: 1.50 > 1.33 → ✅ PASSED
```

### Özet Tablo:

| Senaryo | Hedef | Sonuç | Kök Neden Fix'i |
|---------|-------|-------|----------------|
| Mevcut (buggy) | 1.90 | ❌ REJECTED | - |
| A: PPO fix | 1.47 | ✅ PASSED | En kritik |
| B: Regime fix | 1.72 | ❌ REJECTED | Tek başına yetmez |
| C: PPO + Regime | 1.32 | ✅ PASSED | İdeal |
| D: Tüm fix'ler | 1.33 | ✅ PASSED | Tam çözüm |

---

## 6. Kök Neden Zinciri

```
                    PPO Model İnaktif
                    (score=0.00, flat=5)
                          │
                          ▼
                  ppo_long_score = 0.00
                  (sinyale ekleniyor)
                          │
                          ▼
                  score < 0.5 → rr_up_mult = 1.30
                  (PPO "karşı çıkıyor" gibi)
                          │
                          ▼
        ┌─────────────────┴──────────────────┐
        │                                     │
   regime_name = str(dict)              regime_weight = 1.0
   (string eşleşme başarısız)           (hard-reject default gap)
   regime_mult = 1.0 (not 0.9)         tightening = tam kuvvetle
        │                                     │
        └─────────────┬───────────────────────┘
                      │
                      ▼
            pre_ppo = 1.47 × 1.0 = 1.47
            (olması gereken: 1.33-1.47 × 0.9)
                      │
                      ▼
            target = 1.47 × 1.30 = 1.90
            (olması gereken: 1.33-1.47)
                      │
                      ▼
          R/R 1.50 < Target 1.90 → REJECTED
          (olması gereken: 1.50 > 1.33-1.47 → PASSED)
```

---

## 7. Önerilen Çözümler

### 7.1 PPO Multiplier — İnaktif PPO Koruması (ÖNCELİK: KRİTİK)

**Dosya:** `src/core/strategy_coordinator.py` satır 9260-9267

```python
# MEVCUT:
ppo_rr_multiplier = 1.0
if side in ('buy', 'long') and 'ppo_long_score' in signal:
    score = float(signal['ppo_long_score'])
    ppo_rr_multiplier = (
        self.ppo_multipliers['rr_up_mult'] if score < 0.5 
        else self.ppo_multipliers['rr_down_mult']
    )

# ÖNERİLEN FIX:
ppo_rr_multiplier = 1.0
if side in ('buy', 'long') and 'ppo_long_score' in signal:
    score = float(signal['ppo_long_score'])
    ppo_meta = signal.get('ppo_meta', {})
    ppo_reason = ppo_meta.get('reason', '')
    
    # İnaktif/eğitilmemiş PPO ise multiplier uygulanmasın
    if ppo_reason in ('adapter_unavailable', 'unsupported_symbol', 'disabled'):
        ppo_rr_multiplier = 1.0
    elif score == 0.0 and ppo_reason != 'active':
        # Score tam 0.00 ise model muhtemelen inaktif
        ppo_rr_multiplier = 1.0
        logger.info("⚠️ [PPO-RR] Inactive PPO detected (score=0.00), skipping multiplier")
    else:
        ppo_rr_multiplier = (
            self.ppo_multipliers['rr_up_mult'] if score < 0.5
            else self.ppo_multipliers['rr_down_mult']
        )
```

### 7.2 regime_name Dict Fix (ÖNCELİK: YÜKSEK)

**Dosya:** `src/core/strategy_coordinator.py` satır 9086-9088

```python
# MEVCUT:
signal['regime_name'] = str(ml_context.get('regime', 'neutral'))
signal['regime_confidence'] = float(ml_context.get('regime_confidence', 0.3))

# ÖNERİLEN FIX:
regime_raw = ml_context.get('regime')
if isinstance(regime_raw, dict):
    signal['regime_name'] = str(regime_raw.get('predicted_regime', 'neutral'))
    signal['regime_confidence'] = float(regime_raw.get('confidence', 0.3))
else:
    signal['regime_name'] = str(regime_raw or 'neutral')
    signal['regime_confidence'] = 0.3
```

### 7.3 regime_weight Default Gap Fix (ÖNCELİK: ORTA)

**Dosya:** `src/core/risk_rules.py` satır 773

```python
# MEVCUT:
regime_weight = _safe_float(signal.get('regime_weight', 1.0), 1.0)

# ÖNERİLEN FIX (Seçenek A — Yoksa regime_conf'dan türet):
regime_weight_raw = signal.get('regime_weight')
if regime_weight_raw is not None:
    regime_weight = _safe_float(regime_weight_raw, 1.0)
else:
    # regime_weight yoksa, regime_conf'dan hesapla (strategy_integration mantığı)
    if regime_conf < 0.30:
        regime_weight = 0.0  # Hard reject → efekt sıfır
    elif regime_conf < 0.60:
        regime_weight = regime_conf / 0.60
    else:
        regime_weight = 1.0
    logger.debug(
        "[Dynamic R/R] regime_weight fallback: derived %.2f from regime_conf=%.2f",
        regime_weight, regime_conf
    )
```

---

## 8. Etkilenen Dosyalar

| Dosya | Satırlar | Sorun |
|-------|----------|-------|
| `src/core/strategy_coordinator.py` | 9261-9266 | PPO inaktif iken 1.3x çarpan |
| `src/core/strategy_coordinator.py` | 9086-9088 | regime dict → str dönüşüm hatası |
| `src/ml/strategy_integration.py` | 564-607 | `get_ml_context` regime'i dict olarak döndürüyor |
| `src/core/risk_rules.py` | 773 | regime_weight default=1.0 gap |
| `src/config/risk_config.py` | 665-760 | Konfigürasyon doğru, kod yanlış okuyor |

---

## 9. İlişkili Log Kayıtları (Kronolojik)

```
15:34:20 [ml.regime_predictor] Predicting regime transition for BTC/USDT:USDT with 1h horizon
15:34:20 [ml.regime_predictor] Prediction: bearish (confidence: 0.19)
15:34:20 [core.production_coordinator] [REGIME] BTC/USDT:USDT | Trend=NEUTRAL | ADX=16.2 | Vol=normal | Momentum=sideways | Shock=ARMED
15:34:21 [ml.strategy_integration] 🧠 Regime for BTC/USDT:USDT ignored (Conf: 0.19 < 0.30 hard reject threshold)
15:34:21 [core.strategy_coordinator] 🤖 [PPO-DECISION] BTC/USDT:USDT | Action: HOLD | Score: 0.00 | Conf: 1.00
15:34:21 [core.strategy_coordinator] 📊 [Signal Enriched] ML=0.50, RL_agree=False, Regime=DICT(!) (0.30), Vol=0.85, PPO_RR=1.30
15:34:21 [core.risk_rules] 📊 [Dynamic R/R Calc] Base=1.50 - Relax=0.17 + Tight=0.14 × Regime(DICT, mult=1.0, weight=1.00)=1.00 = 1.46 × PPO(1.30) = 1.90
15:34:21 [core.risk_rules] 🚫 REJECTED: Risk/reward ratio 1.50 < dynamic target 1.90
15:34:21 [core.risk_manager] 🚫 Position REJECTED by RiskRewardRatioRule
15:34:21 [core.strategy_coordinator] 🛡️ [ADAPTIVE_OB] REJECTED (Risk Check)
```

---

## 10. Sonuç

Bu olay, **4 bağımsız sorunun birlikte çalışarak** dinamik R/R hedefini 1.33-1.47'den 1.90'a şişirdiğini göstermektedir:

1. **PPO Multiplier (×1.30)**: İnaktif PPO'nun aktif "karşı çıkış" olarak değerlendirilmesi — **tek başına sinyalin reddine yol açtı**
2. **regime_name Dict Bug**: Regime çarpanı hatalı hesaplanıyor (1.0 yerine 0.9 olmalı)
3. **regime_confidence Fallback**: Gerçek 0.186 yerine 0.30 kullanılıyor
4. **regime_weight Default Gap**: Hard-reject → tam ağırlık paradoksu

**En acil fix:** PPO Multiplier — inaktif model algılama mekanizması eklenmelidir. Bu tek fix bile sinyalin geçmesini sağlardı (hedef: 1.47 < sinyal R/R: 1.50).

---

*Analiz: 2026-02-17 | Container: bearish-bot | Image: bearishalphabot.azurecr.io/bearish-bot:manual-20260217-v1*
