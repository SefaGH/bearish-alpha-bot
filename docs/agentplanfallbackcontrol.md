Tüm veriler hazır. Şimdi analizi derleyeyim. 

Completed: *Propose direction-confirmation checks* (4/4)

---

## pos_BTC/USDT:USDT_1770894460 — Fallback Market Emir Analizi

### 1. İşlem Kronolojisi

| Zaman | Olay | Detay |
|-------|------|-------|
| **11:03:33** | Sinyal üretildi | adaptive_str **SHORT**, entry=$67,978.10, RSI=61.90 (eşik=55.00) |
| **11:03:33** | SMART-ENTRY limit hesaplandı | `limit = $67,978 + 0.85×$399 = $68,317.40` |
| **11:03:33** | Limit emir borsaya gönderildi | $68,317.40 SHORT SELL LIMIT, timeout=240s, gate=12bps |
| **11:04:07** | **GUARD RED #1** | RSI rollover check FAIL: `rsi_prev=72.99, rsi_now=75.40` — RSI hala yükseliyor! |
| **11:04:38** | **GUARD RED #2** | Aynı RSI guard (rollover defer count=2) |
| **11:05:10** | **GUARD RED #3** | Aynı RSI guard (rollover defer count=3) |
| **11:05:42** | **PANIC VETO** | 5m hacim spike: `current=1204, avg20=388` (3.1x) |
| **11:06:16** | **PANIC VETO** | Aynı hacim spike |
| **11:06:49** | **PANIC VETO** | Aynı hacim spike |
| **11:07:22** | **PANIC VETO** | Aynı hacim spike |
| **11:07:39** | Timeout doldu (246s) | Chase gate: `deviation=11.74bps < gate=12.00bps` → GEÇTİ |
| **11:07:40** | **MARKET FALLBACK** | Fill @ $67,903.70 — **SIFIR yeniden doğrulama** |
| **11:10:45** | BREAKEVEN_LOCK | Stop → $67,890.12 |
| **11:17:16** | TRADE_CLOSED | trailing_stop, exit=$67,894.50 |

### 2. Sonuç

| Metrik | Değer |
|--------|-------|
| Entry | $67,903.70 |
| Exit | $67,894.50 |
| P&L | **+$0.31 (+0.014%)** |
| R:R planlanan | 2.00 |
| R:R gerçekleşen | **0.02** |
| MFE | 0.253% |
| MAE | -0.045% |
| Süre | 9.6 dakika |

### 3. Kritik Sorun: Kör Fallback

Sinyal **11:03:33**'te üretildi. Ama limit emir beklerken, **her 30 saniyede bir** strateji tekrar değerlendirme yaptı ve **7 kere üst üste "HAYIR"** dedi:

**İlk 3 red (11:04–11:05):** RSI Guard
```
rsi_prev=72.99  rsi_now=75.40   → RSI hala YÜKSELİYOR, rollover YOK
```
Strateji emniyet mekanizması: "RSI tepeden dönmedikçe SHORT yapma"

**Sonraki 4 red (11:05–11:07):** Panic Veto
```
5m volume: current=1204.31  avg20=388.46  (3.1x spike)
```
Strateji emniyet mekanizması: "Anormal hacim varken girme"

Ama `order_manager.py`'daki fallback kodu **bunlardan hiçbirini bilmeden** market emri verdi çünkü:

```python
# lines 1273-1276: Chase gate — SADECE fiyat sapmasına bakar
if deviation_bps is not None and deviation_bps <= 0:
    max_chase_bps = ...
if deviation_bps > max_chase_bps:
    return ABORT   # → Bu tetiklenmedi çünkü 11.74 < 12.00

# lines 1284-1300: Fallback kontrolleri — SADECE config flaglarına bakar
fallback_enabled = True  # default
# extreme bucket? → Hayır, HIGH bucket
# fast_move? → Hayır, flag set edilmemiş
# ⚠️ Strateji re-validasyonu? → MEVCUT DEĞİL!
```

### 4. Mevcut Fallback Akışı (Sorunlu)

```
Timeout → Cancel limit → Chase gate kontrolü → Config flag kontrolü → MARKET EMİR
           ↓                    ↓                       ↓                    ↓
     Sadece borsa        Sadece fiyat            Sadece bucket         Strateji'nin
      cancel race        sapması (bps)          ve flag check        fikri SORULMAZ
```

### 5. Nihai Karar (Bu Faz): Soft Gate-Only

Bu fazda **hard veto uygulanmayacak**.  
Timeout sonrası market fallback kararı için yalnızca **esnek soft gate (min 2/3 pass)** kullanılacak.

```
Timeout → Cancel limit → Chase gate → Config flag → Position-delta check
       → ⭐ Soft Gate (2/3) → MARKET (allow) / ABORT (block)
```

### 6. Uygulanan Model (Kodda)

`soft gate` üç kontrolden oluşur:

1. `edge_preserved`
   - Fallback fiyatında hesaplanan anlık RR, minimum eşiğin üstünde mi?
2. `direction_continuity`
   - Referans girişe göre ters hareket belirlenen bps eşiğini aşıyor mu?
3. `execution_quality`
   - Anlık spread (bid/ask) market fallback için kabul edilebilir mi?

Karar:
- `min_passes=2` ise en az 2 gate PASS olmalı.
- Yetersiz veri (NA) durumunda `fail_closed_on_insufficient_context=false` ise sistem işlem açmayı tamamen durdurmaz (risk almayı sürdürür).

### 7. Güvenli Mimari Notu

`execution_params` alanı exchange `params` ile birleştiği için callable/function taşımak güvenli değildir.  
Bu nedenle fallback soft-gate bağlamı `order_request["_internal"]` içinde taşınır.

### 8. Kod Değişiklikleri

1. `src/core/order_manager.py`
   - `_collect_fallback_soft_gate_cfg(...)` eklendi.
   - `_evaluate_fallback_soft_gate(...)` eklendi.
   - Limit timeout fallback akışında market emrinden hemen önce soft gate çalışır.
   - Block durumunda:
     - `reason=ABORT:FALLBACK_SOFT_GATE:<reason>`
     - `fallback_reason=limit_timeout_market_fallback_soft_gate_blocked`

2. `src/core/live_trading_engine.py`
   - `order_request["_internal"]` içine fallback soft-gate policy enjekte edilir.
   - Varsayılan scope: `adaptive_str` (config ile genişletilebilir).

3. `config/config.example.yaml`
   - `order_manager` altında soft gate ayarları eklendi:
     - `fallback_soft_gate_enabled`
     - `fallback_soft_gate_apply_to_strategies`
     - `fallback_soft_gate_min_passes`
     - `fallback_soft_gate_rr_min`
     - `fallback_soft_gate_max_adverse_bps`
     - `fallback_soft_gate_max_spread_bps`
     - `fallback_soft_gate_fail_closed_on_insufficient_context`

### 9. SafetyOverride ile Konumlandırma

| Başlık | SafetyOverride | Timeout Soft Gate |
|--------|----------------|-------------------|
| Aşama | Pre-trade (signal accept) | Execution-time (timeout fallback) |
| Mantık | 2/3 gate | 2/3 gate |
| Amaç | Aşırı agresif sinyali elemek | Kör market fallback'i azaltmak |

Yani fallback katmanı da SafetyOverride gibi çoğunluk bazlıdır; ancak hard veto eklenmediği için botun risk alma kapasitesi korunur.

### 10. İzleme Kriterleri (Aşırı Korumacılığı Önlemek İçin)

1. `fallback_block_rate` uzun süre çok yüksekse kurallar fazla serttir.
2. `fallback_allowed_expectancy` negatifse kurallar fazla gevşektir.
3. `post-fallback stop-out rate` düşmüyorsa gate kalitesi yetersizdir.

Bu fazın hedefi: fallback kalitesini artırırken botu pasifleştirmemek.
