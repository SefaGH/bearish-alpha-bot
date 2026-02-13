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

### 5. Nihai Karar (Bu Faz): Soft Gate Ağırlıklı + Hard Chase Kill-Switch

Bu fazda karar yükü **soft gate** tarafında olacak.  
`hard chase` akışta kalacak ama yalnızca **katastrofik fiyat kovalamayı kesen son emniyet freni** rolünde çalışacak.

```
Timeout → Cancel/Position verification
       → Hard chase kill-switch (katastrofik durum)
       → ⭐ Soft Gate (min 3/4 pass)
       → MARKET (allow) / ABORT (block)
```

### 6. Uygulanan Model (Kodda)

`soft gate` dört kontrolden oluşur:

1. `edge_preserved`
   - Fallback fiyatında hesaplanan anlık RR, minimum eşiğin üstünde mi?
2. `direction_continuity`
   - Referans girişe göre ters hareket belirlenen bps eşiğini aşıyor mu?
3. `execution_quality`
   - Anlık spread (bid/ask) market fallback için kabul edilebilir mi?
4. `peak_distance`
   - Limit bekleme penceresindeki ekstrem fiyat ile timeout anındaki mevcut fiyat arasındaki fark kabul edilebilir mi?
   - `short`: `wait_window_max -> current`
   - `long`: `wait_window_min -> current`

Karar:
- `min_passes=3` ise en az 3 gate PASS olmalı.
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
   - `order_manager` altında timeout fallback ayarları eklendi:
     - `fallback_hard_chase_enabled`
     - `fallback_hard_chase_floor_bps`
     - `fallback_hard_chase_min_bps`
     - `fallback_hard_chase_max_bps`
     - `fallback_hard_chase_atr_k`
     - `fallback_hard_chase_spread_m`
     - `fallback_soft_gate_enabled`
     - `fallback_soft_gate_apply_to_strategies`
     - `fallback_soft_gate_min_passes`
     - `fallback_soft_gate_rr_min`
     - `fallback_soft_gate_max_adverse_bps`
     - `fallback_soft_gate_max_spread_bps`
     - `fallback_soft_gate_max_peak_distance_bps`
     - `fallback_soft_gate_fail_closed_on_insufficient_context`

### 9. SafetyOverride ile Konumlandırma

| Başlık | SafetyOverride | Timeout Soft Gate |
|--------|----------------|-------------------|
| Aşama | Pre-trade (signal accept) | Execution-time (timeout fallback) |
| Mantık | 3/3 gate | Hard kill-switch + 3/4 gate |
| Amaç | Aşırı agresif sinyali elemek | Kör market fallback'i azaltmak |

Yani fallback katmanı SafetyOverride gibi çoğunluk bazlı karar mantığı kullanırken, hard chase sadece katastrofik senaryolarda devreye giren son fren olarak konumlanır.

### 10. İzleme Kriterleri (Aşırı Korumacılığı Önlemek İçin)

1. `fallback_block_rate` uzun süre çok yüksekse kurallar fazla serttir.
2. `fallback_allowed_expectancy` negatifse kurallar fazla gevşektir.
3. `post-fallback stop-out rate` düşmüyorsa gate kalitesi yetersizdir.

Bu fazın hedefi: fallback kalitesini artırırken botu pasifleştirmemek.

### 11. Nihai Akış

1. `timeout` sonrası limit emir iptal edilir, `position verification` ile kısmi/dolmuş durumlar temizlenir.
2. `hard_chase_kill_switch` çalışır:
   - Primary filtre değildir.
   - Sadece aşırı kötü fiyat kovalamada ABORT üretir.
3. `soft_gate` ana karar katmanıdır:
   - `edge_preserved`
   - `direction_continuity`
   - `execution_quality`
   - `peak_distance`
4. Karar kuralı: en az `3/4 PASS` ise market fallback, aksi durumda ABORT.
5. NA bağlamında `fail_closed_on_insufficient_context=false` ile bot tamamen pasifleşmez; kontrollü risk alma korunur.

### 12. Uygulama Patch Planı

1. `src/core/order_manager.py` — Hard chase'i kill-switch olarak konumlandır
   - Timeout sonrası, cancel/verification tamamlandıktan hemen sonra çalıştır.
   - Eşik yapısı:
     - Statik taban: `hard_chase_floor_bps`
     - Opsiyonel dinamik katkı: `atr_bps * k + spread_bps * m`
     - Clamp: `hard_chase_min_bps` - `hard_chase_max_bps` (ör. `20-60`)
   - Sadece katastrofik durumda `ABORT:HARD_CHASE_KILL:<value>` üret.
   - Primary kalite filtresi gibi davranmamalı.

2. `src/core/order_manager.py` — Soft gate'i ana karar katmanı olarak sabitle
   - Sıralama: `hard_chase_kill_switch -> soft_gate`.
   - `min_passes=3` varsayılanını koru.
   - 4 gate:
     - `edge_preserved`
     - `direction_continuity`
     - `execution_quality`
     - `peak_distance` (wait-window ekstremine göre)
   - Block reason: `ABORT:FALLBACK_SOFT_GATE:<reason>`.

3. `src/core/live_trading_engine.py` — Soft gate/hard chase policy aktarımı
   - `order_request["_internal"]` içine fallback policy ekle.
   - Strategy scope default: `adaptive_str`.
   - Runtime bağlamı (signal ref, state snapshot, entry ref) tutarlı taşınsın.

4. `config/config.example.yaml` — Operasyonel ayarları netleştir
   - Soft gate:
     - `fallback_soft_gate_min_passes: 3`
     - `fallback_soft_gate_rr_min`
     - `fallback_soft_gate_max_adverse_bps`
     - `fallback_soft_gate_max_spread_bps`
     - `fallback_soft_gate_max_peak_distance_bps`
   - Hard chase kill-switch:
     - `fallback_hard_chase_enabled`
     - `fallback_hard_chase_floor_bps`
     - `fallback_hard_chase_min_bps`
     - `fallback_hard_chase_max_bps`
     - `fallback_hard_chase_atr_k`
     - `fallback_hard_chase_spread_m`

5. `tests/test_execution_backend.py` — Kabul testleri
   - Limit timeout + katastrofik sapma: hard chase ABORT.
   - Limit timeout + normal sapma: hard chase geçer, karar soft gate'e kalır.
   - Soft gate `3/4` pass: market fallback ALLOW.
   - Soft gate `<3/4`: ABORT.
   - `peak_distance` gate PASS/FAIL senaryoları.
   - NA bağlamında `fail_closed_on_insufficient_context=false` davranışı.

6. Doğrulama ve rollout
   - Test komutu: `pytest -q tests/test_execution_backend.py -k soft_gate`
   - Gözlem metrikleri:
     - `fallback_block_rate`
     - `hard_chase_abort_rate`
     - `post_fallback_expectancy`
   - 1 haftalık izleme sonrası eşik ince ayarı yap.

### 13. Operasyon Checklist (Canlı İzleme)

1. Günlük takip (seans sonu)
   - `hard_chase_abort_rate = hard_chase_abort / fallback_attempt`
   - `fallback_block_rate = soft_gate_block / fallback_attempt`
   - `post_fallback_expectancy = avg(pnl_after_fallback)`

2. Hedef davranış
   - `hard_chase_abort_rate` düşük kalmalı (kill-switch nadir çalışmalı).
   - `fallback_block_rate` orta bantta olmalı (ne kör izin ne aşırı blok).
   - `post_fallback_expectancy` 0 üstünde kalmalı.

3. Aksiyon matrisi
   - `hard_chase_abort_rate` yüksekse:
     - `fallback_hard_chase_floor_bps` artır
     - veya `fallback_hard_chase_max_bps` artır
   - `fallback_block_rate` yüksekse:
     - `fallback_soft_gate_max_adverse_bps` veya `fallback_soft_gate_max_peak_distance_bps` artır
   - `post_fallback_expectancy` negatifse:
     - `fallback_soft_gate_rr_min` artır
     - `fallback_soft_gate_max_spread_bps` düşür

4. Haftalık kalibrasyon disiplini
   - Her hafta yalnızca 1-2 parametre değiştir.
   - Değişiklik sonrası en az 3 gün veri biriktirmeden yeni ayar yapma.
   - Kararı tek trade ile değil, toplu dağılım (MAE/MFE + expectancy) ile ver.
