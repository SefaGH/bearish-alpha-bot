# Mean Reversion Watchers: FAST_PRICE_WATCH & MICRO_GATE_WATCH

Bu doküman, **Mean Reversion (MR)** stratejisi ile entegre çalışan iki izleme mekanizmasının (watcher) *amaçlarını*, *uçtan uca akışlarını*, *state/TTL/dedupe kurallarını*, *telemetri event sözlüğünü* ve *tuning / troubleshooting* rehberini anlatır.

## 0) Kısa özet

MR stratejisi iki tip “bekleme” ihtiyacı üretir:

1) **Near-miss (fiyat çok yakın ama henüz bandı kırmadı/temas değil)**  
   → **FAST_PRICE_WATCH** (3–10s aralıklı, TTL ~30s, re-arm ile persistence)

2) **Gate fail (örn. volume.low_vol_tight_stop) fakat gate kısa sürede düzelebilir**  
   → **MICRO_GATE_WATCH** (timer-only, max_checks=2, TTL ~20–25s, loop içi mikro kontrol)

Her iki mekanizma da:
- aynı `pending_id` üzerinde çalışır,
- TTL uzatmaz,
- dedupe çakışmalarında deterministik davranır,
- post-mortem için yapılandırılmış telemetri üretir.

---

## 1) Bileşenler ve sorumluluklar

### ProductionCoordinator
- Stratejileri periyodik çalıştırır (pratikte ~30s loop).
- Stratejiden gelen sinyali StrategyCoordinator’a gönderir.
- Recheck (soft deferral) çağrılarını yürütür.
- Recheck sonucunu `soft_deferral_recheck_outcome` ile loglar.

### MeanReversion Strategy
- VWAP bandları / z-score / ADX gibi metriklerle `SIGNAL` veya `HOLD` kararı üretir.
- Near-miss durumunda “hemen trade etme, izle” semantiği doğurur.
- Recheck context’te:
  - **FAST_PRICE_WATCH fiyatını** (varsa) hesaplarda kullanır (`px_source=fast_watch`)
  - `mr_recheck_eval` telemetrisi üretir.
  - Smart persistence için `rearm_recommended` işaretleyebilir.

### StrategyCoordinator
- Gating (volume/risk/concurrency) uygular.
- Deferral türüne göre watcher başlatır:
  - Near-miss → FAST_PRICE_WATCH
  - low_vol_tight_stop → MICRO_GATE_WATCH
- Watch tick döngülerini yürütür, outcome üretir, recheck talep eder.
- Expiry anında “price:null” boşluğunu önlemek için **last_known_price** imputasyonu yapar.

---

## 2) FAST_PRICE_WATCH: Near-miss fiyat izlemesi

### 2.1 Amaç
Fiyat band eşiğine (upper/lower) **çok yakın** olduğunda:
- ana strateji loop’unu bekletmeden,
- kısa aralıklarla fiyatı kontrol edip,
- temas/kırılım koşulu oluşunca stratejiyi **recheck** ile doğrulamak.

### 2.2 Temel parametreler
FAST_PRICE_WATCH item state tipik alanlar:
- `pending_id`, `dedupe_key`, `symbol`, `side`
- `near`: `"upper"` veya `"lower"`
- `trigger_price`: ilgili band seviyesi
- `eps_bps`: tolerans (örn. 10 bps)
- `ttl_ms`: toplam izleme süresi (örn. 30000ms)
- `watch_interval_ms`: ilk interval (örn. 3000ms)
- `max_checks`: TTL içinde maksimum check sayısı
- `max_rearms`: Smart persistence için maksimum rearm sayısı
- `rearm_count`: kaç defa rearm edildi
- `last_known_price`, `last_known_ts_ms`: telemetri için cache

### 2.3 Çalışma akışı (State machine)

MR near-miss
|
| waiting_room_add (refresh_policy=FAST_PRICE_WATCH, ttl=30s)
v
FAST_WATCH_ACTIVE
|
| tick -> price fetch OK
| - last_known_price güncellenir
| - band_touch eval
|
+--> if NOT triggered: schedule next tick (interval/backoff) until TTL
|
+--> if triggered:
emit fast_watch_outcome(outcome=triggered, price=..)
emit strategy_recheck_request(check_detail.fast_watch.price=..)
call MR recheck
|
+--> MR returns HOLD + rearm_recommended=true
| -> fast_watch_rearm (same pending_id, TTL not extended)
|
+--> MR returns SIGNAL (or HOLD no rearm)
-> final outcome (no_signal or signal replay)
-> drop (fast_watch_final) or convert to execution
|
+--> if TTL/max_checks:
emit fast_watch_outcome(outcome=expired, price imputed if needed)
waiting_room_drop(drop_reason=expired/max_checks)


### 2.4 Smart Persistence (Re-arming)
Recheck sonucu `HOLD` ise ve `primary_gate_reason="in_band"` olup “hala yakın” görünüyorsa:
- MR `rearm_recommended=true`, `rearm_reason=still_near` üretir.
- Coordinator item’ı düşürmez, **aynı pending_id** ile rearm eder.
- **TTL uzatılmaz** (remaining TTL korunur).
- Interval backoff uygulanır (örn. 3s → 4.5s → 6.75s → 10.125s).

Rearm limiti dolunca:
- MR `rearm_recommended=false`, `rearm_reason=rearm_limit`
- Coordinator `waiting_room_drop(drop_reason=max_rearms)`.

### 2.5 Data consistency (Phase 6.1)
Recheck context’te MR:
- `check_detail.fast_watch.price` varsa **tüm hesaplarda** onu `px` olarak kullanır.
- `px_source="fast_watch"` loglar.
- `market_price` ayrıca debug amaçlı loglanır (karşılaştırma).

---

## 3) MICRO_GATE_WATCH: Gate bazlı mikro izleme (low_vol_tight_stop)

### 3.1 Amaç
`volume.low_vol_tight_stop` gibi “şu an trade etme” gate’leri:
- bazen 10–20 saniyede değişebilir (veya yeni tick ile koşul düzelebilir),
- fakat 300s bekletmek gereksizdir (zaten ~30s strateji loop var).

MICRO_GATE_WATCH:
- loop içi mikro kontrol yapar (timer-only),
- en fazla 2 check ile “değişti mi?” bakar,
- değişmediyse düşürür.

### 3.2 Parametreler
- `max_checks = 2` (sabit)
- `ttl_ms ~ 20000–25000` (loop içi)
- `interval_policy = timer_only`
- `near_pass_filter`: gate’in “geçmeye yakın” olmadığı durumlarda direkt drop
- `last_known_price`, `last_known_ts_ms`: recheck fiyatı ve expiry telemetri için

### 3.3 Çalışma akışı (State machine)

Signal -> gate fail: volume.low_vol_tight_stop
|
| attach micro_watch params (ttl~25s, max_checks=2)
v
MICRO_GATE_WATCH_ACTIVE
|
| tick#1 (t≈10s): gate re-eval (+ live price)
| - if far from pass -> drop immediately (near_pass_filter)
| - if clears -> replay REPRICE_AND_RESIZE
| - else schedule tick#2
|
| tick#2 (t≈20s): gate re-eval
| - if clears -> replay REPRICE_AND_RESIZE
| - else drop (max_checks)
|
+--> if TTL: expiry outcome (last_known_price imputation)


### 3.4 Dedupe davranışı (micro-watch aktifken)
Aynı `dedupe_key` ile yeni sinyal gelirse:
- micro-watch aktifse **incoming sinyal drop edilir**
- micro-watch state “latest snapshot” ile güncellenebilir ama **TTL uzatılmaz**
- telemetri: `micro_gate_watch_dedupe_drop_incoming`

---

## 4) Telemetri event sözlüğü (Glossary)

Aşağıdaki event’ler debug ve post-mortem için “yük taşıyan” event’lerdir.

### 4.1 waiting_room_add
Deferral/izleme başlatıldığında.

Örnek alanlar:
- `pending_id`, `dedupe_key`, `symbol`, `side`
- `reason_code` (örn. `strategy.mean_reversion.near_miss`, `volume.low_vol_tight_stop`)
- `refresh_policy` (`FAST_PRICE_WATCH`, `REPRICE_AND_RESIZE`, `MICRO_GATE_WATCH`)
- `ttl_seconds`, `watch_interval_ms`, `max_checks`, `near`, `trigger_price`

### 4.2 fast_watch_outcome
FAST_PRICE_WATCH tick sonucu.

Kritik alanlar:
- `outcome`: `triggered | expired`
- `expire_reason`: `hit | ttl | max_checks`
- `price`: (cache fail olursa imputed)
- `price_imputed`, `imputed_from`, `last_price_age_ms`
- `checks_done`, `rearm_count`, `max_rearms`, `remaining_ttl_ms`

### 4.3 strategy_recheck_request
Recheck tetiklendiğinde.

Kritik alanlar:
- `check_detail.fast_watch.price` (veya micro watch price)
- `condition_data.lower/upper/vwap/adx/...` (snapshot)
- `pending_id`, `dedupe_key`, `intent=soft_deferral`

### 4.4 mr_recheck_eval
MR recheck kararı.

Kritik alanlar:
- `action`: `HOLD | SIGNAL`
- `gate_reasons`, `primary_gate_reason`
- `dist_to_trigger_bps`, `eps_bps`
- `px_source`: `fast_watch | market_price | micro_gate_watch`
- `fast_watch_price` (varsa)
- `market_price` (debug)
- `rearm_recommended`, `rearm_reason`

### 4.5 soft_deferral_recheck_outcome
ProductionCoordinator’ın recheck sonucunu özetlemesi.

- `outcome`: `rearmed | no_signal | signal`
- `final_reason`: `still_near | rearm_limit | ...`
- `attempt` (kaçıncı recheck)
- `rearm_fast_watch` bool

### 4.6 fast_watch_rearm / soft_deferral_rearm
Rearm olduğunda.

- `rearm_count`
- `interval_ms` (bir sonraki check)
- `remaining_ttl_ms`
- `reason` (still_near vb.)

### 4.7 waiting_room_drop
İtem düşürüldüğünde.

Kritik alanlar:
- `drop_kind`: `fast_watch | fast_watch_final | micro_gate_watch`
- `drop_reason`: `expired | max_rearms | max_checks | far_from_pass`
- `remaining_ttl_ms`

---

## 5) Tuning rehberi (pratik)

### FAST_PRICE_WATCH (near-miss)
Önerilen başlangıç:
- `ttl_ms = 30000`
- `watch_interval_ms = 3000`
- `eps_bps = 10`
- `max_rearms = 1..3` (piyasa koşuluna göre)
- backoff: 1.5x (3s → 4.5s → 6.75s → 10.125s)
- TTL uzatma **yok**

Amaç:
- “bandı saniyelerle kaçırma” riskini azaltmak
- ama “sonsuz recheck”e girmemek

### MICRO_GATE_WATCH (low_vol_tight_stop)
Önerilen başlangıç:
- `max_checks = 2`
- `ttl_ms = 20000..25000`
- tick planı: t≈10s ve t≈20s (timer-only)
- near_pass_filter: “geçmeye yakın değilse drop”
- TTL uzatma **yok**
- dedupe çakışmasında incoming drop

Amaç:
- “300s bekleme kampı”na dönmeden,
- loop içi mikro doğrulama.

---

## 6) Troubleshooting (Hızlı teşhis)

### Soru: “Niye HOLD oldu, oysa fiyat kırmış gibi?”
Kontrol listesi:
1) `strategy_recheck_request.check_detail.fast_watch.price` var mı?
2) `mr_recheck_eval.px_source` ne?
   - `fast_watch` olmalı (Phase 6.1 sonrası)
3) `dist_to_trigger_bps` pozitif mi? (kırılım yönüne göre)

### Soru: “Expiry’de price null görünüyor”
Beklenen:
- `fast_watch_outcome.price` null olmamalı (imputation)
- `price_imputed=true`, `imputed_from=last_known_price` alanlarını ara

### Soru: “Yeni sinyal geldi ama işlem olmadı”
Dedupe olasılığı:
- micro-watch aktifken incoming sinyal drop edilir
- event: `micro_gate_watch_dedupe_drop_incoming`

---

## 7) İnvariant’lar (değişmez kurallar)

1) **Rearm aynı `pending_id` üzerinde yapılır**  
2) **TTL uzatılmaz** (remaining TTL korunur)  
3) **Recheck fiyat kaynağı deterministik olmalıdır**  
   - recheck varsa: watch price > market snapshot  
4) **Dedupe çakışmalarında “kamp” yok**  
   - micro-watch aktifken incoming drop / state update (TTL yok)

---

## 8) Referans: Kod konumları (yüksek seviye)

- `src/strategies/mean_reversion.py`
  - recheck context: `px_source` seçimi (fast_watch/micro_gate_watch)
  - `mr_recheck_eval` telemetrisi
  - `rearm_recommended` mantığı

- `src/core/strategy_coordinator.py`
  - FAST_PRICE_WATCH tick & rearm
  - MICRO_GATE_WATCH tick & near_pass_filter
  - last_known_price/ts state + expiry imputation
  - dedupe collision handling (micro-watch aktifken drop)

- `src/core/production_coordinator.py`
  - recheck yürütme + `soft_deferral_recheck_outcome`

---

## 9) Örnek “iyi” akışlar (pattern)

### FAST watch + rearm + TTL bitmeden karar
- waiting_room_add (FAST_PRICE_WATCH)
- fast_watch_outcome(triggered)
- strategy_recheck_request
- mr_recheck_eval(HOLD, rearm_recommended=true, px_source=fast_watch)
- fast_watch_rearm (rearm_count=1, remaining_ttl_ms düşer)
- fast_watch_outcome(triggered)
- mr_recheck_eval(SIGNAL) **veya** no_signal + drop

### MICRO gate watch clears on second check
- gate fail (low_vol_tight_stop) → micro watch start
- tick#1: not clear, near-pass OK → schedule tick#2
- tick#2: clear → REPRICE_AND_RESIZE replay
- execute pipeline continues

---

## 10) Notlar / Tasarım gerekçeleri

- “FAST” adı yalnızca near-miss fiyat izlemesi içindir; gate izlemesi ayrı (micro).
- Candle boundary clamp (next 5m close) **micro watch** için gereksiz gecikme yaratır; timer-only seçilmiştir.
- Ana strateji loop’u zaten 30s olduğu için micro watch TTL’leri loop içi kalacak şekilde tasarlanmıştır.

---