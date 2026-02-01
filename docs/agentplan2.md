#### **MODÜL 1: RİSK YÖNETİMİ & EXECUTION (Teknik Düzeltme)**

**Hedef:** Entry-Ref Bug'ı düzeltmek ve Stop geometrisini korumak.
**Uygulama noktası notu:** Fill fiyatı sistemde `execution_result.avg_price` olarak geliyor ve pozisyon oluşturma `src/core/position_manager.py` içindeki `open_position()` üzerinden yapılıyor. Rebase mantığı için doğal yer burasıdır.

**1. Dinamik Stop/TP Hesaplaması (Math Fix)**

* **Düzeltme:** Bps değerlerini doğrudan çarpan olarak kullanma, fraction'a çevir. Ayrıca "Config"deki sabit değer yerine, sinyal anında hesaplanan **hedef oranı** (ratio) kullan.
* **Logic:**
```python
# Adım 1: Sinyal anında hedeflenen mesafeyi (Ratio) kaydet.
# Not: signal_price için tercih sırası: price_meta.price_used -> entry_raw -> entry
# Örn: Signal: 100, Stop: 99 ise Ratio = 0.01
target_stop_ratio = abs(signal_price - signal_stop_price) / signal_price
target_tp_ratio = abs(signal_price - signal_tp_price) / signal_price

# Adım 2: Fill geldikten sonra bu oranı uygula
# fill_price: 100.5 (Slippage yemiş)
if side == SHORT:
    real_stop_price = fill_price * (1 + target_stop_ratio)
    real_tp_price = fill_price * (1 - target_tp_ratio)
else: # LONG
    real_stop_price = fill_price * (1 - target_stop_ratio)
    real_tp_price = fill_price * (1 + target_tp_ratio)

```



**2. Emir Güncelleme Mekanizması (Cancel/Replace)**

* **Düzeltme:** Borsada `update_order` yoksa, eskiyi iptal edip yenisini kurmalıyız. Race-condition (yarış durumu) riskini yönetmeliyiz.
* **Logic:**
```python
def adjust_risk_orders(position, new_stop, new_tp, order_manager):
    # Mevcut risk emri ID’leri position üzerinden takip edilmeli
    risk_ids = [
        position.get("native_hard_stop_order_id"),
        position.get("native_trailing_stop_order_id"),
    ]
    for oid in filter(None, risk_ids):
        try:
            order_manager.cancel_order(oid, position["exchange"])
        except OrderNotFound:
            pass  # Zaten dolmuş veya iptal olmuş

    # Yeni emirleri reduce-only olarak gir
    order_manager.place_stop_loss(position["symbol"], new_stop, reduce_only=True)
    order_manager.place_take_profit(position["symbol"], new_tp, reduce_only=True)

```



**3. Slippage Guard (İki Katmanlı Koruma)**

* **Düzeltme:** İşlem açıldıktan sonra kapatmak pahalıdır. Önce girmemeye çalış, girdiysen ve çok kötüyse abort et.
* **Katman A (Pre-Trade):** Mevcut `SignalIntegrityGuard` zaten fiyat sapması kontrolü yapıyor (`signals.integrity_guard.max_deviation_pct`). Bu katmanı ATR/spread tabanlı eşik ile **genişlet**.
* **Not (Guard Birleştirme):** Slippage + Impuls kontrollerini **tek guard katmanında** topla ve tek tip `reason_code/telemetri` standardı kullan (örn. SignalIntegrityGuard genişletme veya yeni MarketGuard).
```python
current_price = market_data_pipeline.get_latest_price(...)
gap_bps = abs(current_price - signal_price) / signal_price * 10000.0
atr_bps = (atr / signal_price) * 10000.0 if atr else None

# Öneri: max_deviation_pct ile birleşik eşik
max_gap_bps = max(cfg_max_dev_pct * 10000, (atr_bps or fallback_bps) * 0.5)
if gap_bps > max_gap_bps:
    return ABORT_TRADE ("Price moved too fast before entry")

```


* **Katman B (Post-Fill RR Re-validation):**
* Eğer fill sonrası yeni Stop/TP hesaplandığında **Risk/Reward Oranı (RR)** 1.0'in altına düşüyorsa (slippage kâr payını yediyse), işlemi kapat veya stop'u başa başa çek.

**TP1 / Partial Close (Reduce-Only) – Ön Koşullar**
* Bu plan **One-Way Mode** varsayımıyla anlatılır. Hedge Mode kullanılıyorsa `positionSide` + “close/open” semantiklerini ayrıca tasarlayın.
* Reduce-Only parametresi **resmi API tablosundan** doğrulanmalı (parametre adı ve boolean serileştirme).
* TP1 fill sonrası **kalan qty’ye göre** SL/TP/Trailing emirleri yeniden boyutlandırılmalı (aksi halde fazla kapatma/flip riski).
* Detaylar için: `docs/reduce_only.md`



---

#### **MODÜL 2: STRATEJİ MANTIĞI (Impuls & Teyit)**

**Hedef:** İmpuls mumuna "kafa atmayı" engellemek.
**Not:** `trend_guard` halihazırda MR ve adaptive_str için aktif ve “breakout up → short veto” yapıyor. Yeni Impuls Veto ya TrendGuard’a entegre edilmeli **ya da slippage guard ile tek bir guard katmanında birleştirilmeli** (tek tip reason_code/telemetri).

**4. Impuls Veto (Gelişmiş)**

* **Logic:** Sadece tek muma değil, son hareketin şiddetine bak. MR varsayılanı `signal_close` olduğu için **son kapanmış mum** üzerinden hesapla.
```python
# Son mumun gövdesi ATR'nin 1.5 katı MI? VEYA
# Son 2 mumun toplam hareketi ATR'nin 2.5 katı MI?
body_size = abs(close - open)
is_shock_move = (body_size > atr_5m * 1.5)

if is_shock_move and (trade_direction != candle_direction):
    return VETO ("Impulse/Shock Move Detected")

```



**5. Rejection Confirmation (Teyit)**

* **Logic:** "Kırmızı Mum" şartı baki, ama "Band İçine Dönüş" şartını esnetebilirsin (Wick Rejection). Band değerlerini `vwap_upper/vwap_lower` üzerinden, **aynı kapanmış mum** ile eşleştir.
```python
# Short Teyidi:
has_red_candle = close < open
# Fiyat bandın altına indi Mİ veya Üst Fitil çok mu uzun?
rejected_from_band = (close < upper_band) or (high_wick_size > body_size * 0.8)

if not (has_red_candle and rejected_from_band):
    return WAIT ("No clear rejection yet")

```



---

#### **MODÜL 3: CHURN ÖNLEME (Yeni Eklenenler)**

**Hedef:** Üst üste stop olmayı (Trade A -> B -> C serisini) engellemek.
**Not:** `StrategyCoordinator` içinde stop-loss sonrası cooldown/reversal guard zaten var (`_stop_loss_reversal_required`, `_strategy_cooldowns`). Yeni kural eklerken bu mekanizmaları genişletmek daha güvenli.

**6. Cooldown & Reset**

* **Logic:**
```python
# Global state veya DB'de tutulacak
last_stop_time = get_last_stop_time(pair)

# Kural 1: Stop olduktan sonra 15 dakika (3 bar) işlem açma.
if (current_time - last_stop_time) < timedelta(minutes=15):
    return BLOCKED ("Cooldown Phase")

# Kural 2 (Opsiyonel): Fiyat Bollinger Orta Bandına değmeden tekrar aynı yöne işlem açma.

```



---

### **GELİŞTİRİCİ/AGENT İÇİN NET KOMUT (Copy-Paste)**

Aşağıdaki metni Agent'a vererek süreci başlatabilirsin:

> **GÖREV: Trading Bot "Episode C" Düzeltme Paketi**
> Analizler sonucunda botta hem teknik (execution) hem stratejik eksikler tespit edilmiştir. Aşağıdaki değişiklikleri 3 fazda uygulayın.
> **FAZ 1: Teknik Altyapı (Risk & Execution)**
> 1. **Birim Düzeltmesi:** Tüm BPS hesaplamalarında `bps / 10000.0` dönüşümünü garantiye al (config’deki pct değerleriyle karıştırma).
> 2. **Dinamik Stop/TP (Rebase):** `position_manager.open_position()` içinde, `execution_result.avg_price` (fill) ile sinyalde saklanan `target_stop_ratio/target_tp_ratio` oranlarını uygula. Config’deki sabit bps’i fill fiyatına direkt ekleme.
> 3. **Cancel/Replace:** `order_manager` üzerinden risk emirlerini **ID bazlı** iptal edip yeniden gir (reduce-only). `OrderNotFound` idempotent kabul edilmeli.
> 4. **Pre-Trade Slippage Guard:** Mevcut `SignalIntegrityGuard` fiyat sapması kontrolünü ATR/spread tabanlı eşik ile genişlet; `price_meta.price_used` ve `market_data_pipeline.get_latest_price()` kullan.
> 5. **Guard Birleştirme:** Slippage + Impuls kontrollerini tek guard katmanında topla; `reason_code` ve telemetri formatını ortaklaştır.
> 
> 
> **FAZ 2: Strateji Filtreleri (`mean_reversion.py`)**
> 1. **Impuls Veto:** Eğer son kapanmış mumun gövdesi `1.5 * ATR`'den büyükse ve işlem ters yöndeyse sinyali blokla. TrendGuard ile çakışıyorsa tek bir yerde toplamak tercih edilir.
> 2. **Rejection Teyidi:** Short için `Close < Open` (Kırmızı Mum) VE (`Close < vwap_upper` VEYA `Güçlü Üst Fitil`) şartı ara. Değerlendirme “closed-only” üzerinden yapılmalı.
> 
> 
> **FAZ 3: Güvenlik (Churn Prevention)**
> 1. **Cooldown:** Bir paritede Stop Loss gerçekleşirse, o paritede aynı yönde işlem açmak için 15 dakika bekleme süresi (Cooldown) ekle.
> 2. **RR Check:** Fill sonrası Stop/TP hesaplandığında, eğer Risk/Reward oranı 1.0'in altına düşmüşse işlemi "Early Exit" ile kapat.
> 
> 
EK NOT (Kritik Pimler):

target_stop_ratio ve target_tp_ratio rebase edildikten sonra min mesafe clamp uygulayın. **Birimleri karıştırma:**
`min_stop_ratio = max(min_stop_pct, 0.8*atr_pct, spread_buffer_bps/10000.0)`
(`atr_pct = atr/price`, `min_stop_pct` config’de zaten fraction olarak tutuluyor.)

Slippage ölçümünde last yerine mümkünse mid=(bid+ask)/2 kullanın; eğer ATR yoksa guard için **sabit bps fallback** koyun.

RR<1.0 durumunda otomatik market close’u hard slippage koşuluna bağlayın (RR<1.0 AND slippage_bps > hard_slip_bps), aksi halde TP’yi yakın banda çekme opsiyonu tercih edilir. (Not: **partial reduce** altyapısı mevcut değilse ayrı geliştirme gerekir.)

TP1/partial close için reduce-only akışı, mod varsayımı (One-Way vs Hedge) ve kalan qty’ye göre risk emirlerini resize gereksinimi `docs/reduce_only.md` içinde netleştirildi.

Cooldown state’i (symbol,strategy,side) bazında tutulmalı; reset olarak “mid band touch” veya “N bar geçişi” şartı eklenmeli. Mevcut `StrategyCoordinator` stop-loss guard mekanizması genişletilebilir.

Aşağıdaki “kabul kriterleri” seti, senin **Final Uygulama Planı**nın üç modülünü (Risk/Execution, Strateji Filtreleri, Churn Önleme) **ölçülebilir** hale getirir:

1. **Telemetri alanları** (logda neyi basacağız),
2. **Golden-window regression eşikleri** (hangi pencerede ne bekliyoruz),
3. **Test / CI kabul kriterleri** (geçti-kaldı netliği).

## 1) Telemetri alanları: “Başarılı sayılması için logda neyi görmeliyiz?”

Aşağıdaki alanları **tek bir SSOT formatında** (JSON ya da key=value) basmanı öneririm. En kritik amaç: **signal_price → fill_price sapması** ve **stop/TP rebase sonrası geometrinin korunması**.

### A) Sinyal telemetrisi (mean_reversion sinyal üretimi)

Her sinyal için:

* `signal_id`, `symbol`, `strategy`, `side`, `tf` (signal_timeframe)
* **Fiyat referansları**

  * `signal_price` (sinyalin kullandığı fiyat – closed/forming net yaz; mevcut `price_meta.price_used` ile eşleştir)
  * `mid_price_at_signal` (bid/ask varsa)
  * `ticker_age_ms`
* **Band metrikleri**

  * `upper_band`, `mid_band`, `lower_band`, `px_vs_upper_bps` (MR için `vwap_upper/vwap/vwap_lower`)
* **Volatilite / rejim**

  * `atr_bps` (atr/price*10000; yoksa `null` + `atr_missing=true`)
  * `spread_bps` (bid-ask’tan)
  * `adx`, `ema_slope`, `momentum_strength` (varsa)
* **Impuls metriği**

  * `candle_body_bps`, `candle_range_bps`
  * `body_atr_mult`, `range_atr_mult`
  * `sum2_range_atr_mult`
  * `candle_dir` (up/down)
* **Rejection teyidi**

  * `prev_red` (close<open)
  * `close_back_inside_band` (close<upper_band)
  * `upper_wick_ratio` (upper_wick/body)
* **Karar**

  * `decision=allow|veto|wait`
  * `reason_code` (örn `impulse_veto`, `no_rejection`, `cooldown`, `pretrade_gap_abort`)

Önerilen log etiketi:
`[MR-SIGNAL] ...` ve veto/wait durumlarında `[MR-VETO]` / `[MR-WAIT]`.

---

### B) Pre-trade slippage guard telemetrisi (order submit öncesi)

Her “allow” sinyalinde order göndermeden önce:

* `current_mid` (tercihen mid), yoksa `last`
* `gap_bps = abs(current_mid - signal_price)/signal_price*10000`
* `max_allowed_gap_bps = max(cfg_max_dev_pct*10000, 0.5*atr_bps + spread_buffer_bps)` (ATR yoksa fallback)
* `pretrade_action=send|abort`
* `abort_reason=price_moved_fast` (abort ise)
* `order_intent=market|limit|smart_entry`

Log etiketi:
`[SLIPPAGE-GUARD-PRE] ...`

**Kabul için logda görmek istediğimiz:**
Abort olduysa **“order sent” olmamalı** (order_id üretmemeli).

---

### C) Fill sonrası rebase telemetrisi (position_manager)

Fill geldiğinde:

* **Fiyatlar**

  * `fill_price` (execution_result.avg_price), `signal_price`, `slippage_bps`
* **Planlanan mesafeler**

  * `target_stop_ratio`, `target_tp_ratio`
  * `planned_stop_dist_bps = target_stop_ratio*10000`
  * `planned_tp_dist_bps = target_tp_ratio*10000`
* **Clamp sonrası gerçek mesafeler**

  * `min_stop_bps_applied`, `min_tp_bps_applied`
  * `effective_stop_dist_bps`, `effective_tp_dist_bps`
* **Rebased emirler**

  * `rebased_stop_price`, `rebased_tp_price`
  * `rr_planned`, `rr_effective` (rebased sonrası)
* **Risk emir yönetimi**

  * `risk_orders_action=cancel_replace`
  * `cancelled_count`, `placed_count`
  * `reduce_only=true`
  * `replace_latency_ms`
* **Post-fill karar**

  * `postfill_action=keep|early_exit|partial_reduce|tp_adjust`
  * `reason_code=rr_below_1|hard_slippage|stop_invalid_side` vb.

Log etiketi:
`[RISK-REBASE] ...` ve devamında `[RISK-ORDERS] ...`

---

### D) Cooldown / churn telemetrisi

Stop-loss gerçekleştiğinde:

* `stop_event=true`, `stop_time`, `symbol`, `strategy`, `side`
* `cooldown_until`, `cooldown_minutes`
* `reset_condition_enabled=true|false`
* `attempt_counter_in_impulse` (varsa)

Cooldown blokladığında:

* `blocked=true`, `blocked_reason=cooldown`
* `cooldown_remaining_s`

Log etiketi:
`[CHURN-COOLDOWN] ...`

---

### E) TP1 / partial close telemetrisi (reduce-only)

* `tp1_order_placed`: `reduceOnly=true`, `qty`, `side`, `order_id`
* `tp1_filled`: `filled_qty`, `remaining_position_qty`
* `risk_orders_resized`: `new_qty` (kalan pozisyona göre)
* Not: `reduceOnly=true` logu olmadan “partial close” kabul edilmez.

---

## 2) Golden-window regression: pencere tanımı ve eşikler

Amaç: Episode C benzeri “shock move + MR fade” senaryosunda **davranışın düzelmesini** otomatik doğrulamak.

### A) Golden window seti (öneri)

1. **episode_c_shock_window**: 04:10–04:25 (UTC)

   * Beklenen: impuls veto + rejection teyidi + slippage guard + cooldown davranışı
2. **dip_recovery_window**: 01:45–02:15

   * Beklenen: MR long’lar allowed; stop geometrisi tutarlı; fee-lock/trailing davranışı stabil
3. **range_chop_window**: 02:15–03:30

   * Beklenen: churn limitleri, cooldown/attempt limiter çalışıyor mu

> Pencereleri istersen `windows.yaml` gibi bir dosyada tut; şu an repo içinde mevcut bir “windows.yaml” yok.

---

### B) Episode C için “pass/fail” eşikleri (asıl kritik)

Bu pencerede **kârlılık** değil **doğru davranış** ölçüyoruz:

**Eşik-1 — Impuls anında ters yön MR girişleri**

* `mr_countertrend_entries_during_shock == 0`

  * Tanım: `body_atr_mult >= 1.5` veya `sum2_range_atr_mult >= 2.5` iken *ters yönde* trade açılmamalı.

**Eşik-2 — Rejection teyidi olmadan MR short açılmaması**

* Eğer MR short açıldıysa:

  * `rejection_confirmed == true` olmalı (red candle ve/veya wick rejection koşulu)
* Tercihen: `mr_short_without_rejection == 0`

**Eşik-3 — Stop geometrisi bozulmasın (fill sonrası)**

* Kabul kuralı (trade bazında):

  * `effective_stop_dist_bps >= max(min_stop_bps, 0.8*atr_5m_bps, spread_buffer_bps)`
* Golden window için:

  * `stop_geom_violation_count == 0`

**Eşik-4 — Stop “wrong side” / fallback 1% gibi anomaliler yok**

* `stop_invalid_side_count == 0`
* `fallback_stop_applied_count == 0`
  (Fallback oluyorsa bu artık bug/detektör başarısı sayılır; ama hedef “fallback gerekmeyecek kadar rebase doğru”.)

**Eşik-5 — Pre-trade gap abort çalışıyor**

* Shock sırasında `gap_bps > threshold` olan sinyallerde:

  * `pretrade_abort_count >= 1` (Episode C’de beklenir)
  * ve `aborted_signal_has_order_id == 0`

**Eşik-6 — Stop sonrası re-entry churn kontrolü**

* Stop-loss olduysa:

  * cooldown süresince `same_symbol_strategy_side_entries == 0`

**Eşik-7 — TP1 reduce-only kanıtı (varsa)**

* TP1 tetikleniyorsa:
  * `tp1_order_placed.reduceOnly == true`
  * `risk_orders_resized == true` (kalan qty’ye göre)

---

### C) Dip recovery (01:45–02:15) için eşikler

* `trade_open_count > 0` (fırsat kaçmasın)
* `effective_stop_dist_bps` alt sınır kuralı yine **0 ihlal**
* `slippage_bps` yüksekse pretrade abort ya da postfill aksiyon loglanmalı:

  * `high_slippage_unhandled_count == 0`

Opsiyonel (kârı erken kesme konusu için ölçüm):

* `fee_lock_armed_bps` ve `exit_reason=trailing_stop` oranını izleyin:

  * Eğer aşırı artarsa, bu ayrı tuning konusu.

---

## 3) CI / test kabul kriterleri

### A) Unit test (zorunlu)

1. **Birim doğruluğu**

* bps→fraction dönüşümleri
* `slippage_bps` hesabı doğru mu

2. **Rebase matematiği**

* SHORT/LONG için stop/tp doğru yönde mi
* `effective_stop_dist_bps` clamp çalışıyor mu

3. **RR re-validation**

* `rr_effective` hesap doğru mu
* `RR<1` aksiyonu doğru branch’e gidiyor mu

### B) Integration test (fake exchange / sim)

* Cancel/replace idempotency:

  * “OrderNotFound” yakalanınca sistem bozulmuyor
  * Rate-limit/backoff senaryosunda “placed_count” doğru
* Reduce-only doğrulama
* “Abort edilmiş” sinyalde order submit yok

### C) Golden regression (CI job) geçiş kriteri

* `episode_c_shock_window`: tüm Eşik-1..6 **PASS**
* Diğer pencereler: stop geometry ihlali **0**, anomaly **0**

---

## 4) “Başarılı” çıktının logda görünmesi gereken 6 imza satır

CI’da grep ile bile kontrol edilebilecek şekilde:

1. `[MR-VETO] reason=impulse_shock ... decision=block` (Episode C’de en az 1 kez)
2. `[SLIPPAGE-GUARD-PRE] action=abort gap_bps=... threshold_bps=...`
3. `[RISK-REBASE] fill_price=... target_stop_ratio=... effective_stop_dist_bps=...`
4. `[RISK-ORDERS] action=cancel_replace cancelled=... placed=... reduce_only=true`
5. `[POSTFILL-RR] rr_effective=... action=early_exit|partial_reduce|tp_adjust` (tetikleniyorsa)
6. `[CHURN-COOLDOWN] blocked=true remaining_s=...` (stop sonrası)

---
