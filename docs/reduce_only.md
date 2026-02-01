Aşağıdaki “reduce-only ile partial close (TP1)” yaklaşımı **BingX’in reduce-only davranışıyla uyumlu** ve bot tarafında TP1/partial çıkışı “ters yöne pozisyon açmadan” güvenli yapmanın en temiz yoludur.

## 1) “Reduce-Only” ne yapar?

BingX’in **One-Way Mode** açıklamasında Reduce-Only, **mevcut pozisyonu azaltmak/kapatmak** için kullanılan bir işaret (toggle) olarak geçiyor:

* One-Way Mode’da aynı kontratta long+short aynı anda tutulamaz; aksi yöndeki emir **pozisyonu kapatıp** kalan varsa ters pozisyon açabilir.
* **Reduce-Only işaretliyse**, ters yöndeki emir **yalnızca pozisyonu kapatır**, **ters pozisyon açmaz** (örnek: 10 BTC long + Reduce-Only ile 10 BTC short emri → long kapanır, pozisyon sıfırlanır). ([BingX Help Center][1])
* Ayrıca BingX aynı metinde, bu “position mode” ayarının **app/web/API/3rd-party** tarafında birlikte uygulanacağını not ediyor. ([BingX Help Center][1])

**Buradan çıkan kritik sonuç:** Reduce-Only, “exit-only” davranışı sağlayarak **ters pozisyon açma riskini** (özellikle hızlı hareketlerde) sistematik biçimde düşürür.

## 2) Reduce-Only ile “Partial Close (TP1)” nasıl yapılır?

Mantık basit:

* Elinde örn. **1.0 BTC LONG** var.
* TP1’de %30 çıkmak istiyorsun → **0.30 BTC’lik** bir **SELL** emri gönderirsin.
* Bu emri **Reduce-Only** işaretlersin → emir **sadece pozisyonu 1.0 → 0.70** düşürür; **0.30’ın üzerinde** fill olursa ters pozisyon açmaya çalışmaz (exit-only). ([BingX Help Center][1])

Aynı şey SHORT için ters yönde uygulanır (SHORT pozisyonu azaltmak için BUY reduce-only).

> BingX’in kendi açıklamasında “close the corresponding amount directly or tick ‘Reduce Only’ to close the corresponding amount” ifadesi geçiyor; bu, Reduce-Only’nin **miktar bazlı kapatmayı (partial/total)** hedeflediğini netleştiriyor. ([BingX Help Center][1])

## 3) Hangi modda nasıl düşünmelisin? (One-Way vs Hedge)

BingX’in açıklaması iki mod arasında operasyonel farkı da net çiziyor:

* **One-Way Mode:** Kapatmak için “Reduce Only” ile “corresponding amount” kapatma davranışı var. ([BingX Help Center][1])
* **Hedge Mode:** Pozisyon kapatma için arayüzde “Open/Close” seçimi vurgulanıyor; yanlış seçim ters pozisyon açar. ([BingX Help Center][1])

Bot açısından pratik karşılık:

* One-Way modda reduce-only **özellikle değerli**, çünkü ters emirler aksi halde pozisyonu kapatıp ters yöne çevirebilir.
* Hedge modda ise “hangi positionSide/close semantiği” kullanıyorsun kısmını **netleştirmen gerekir** (aksi halde reduce-only niyetiyle “open” davranışı tetiklenebilir).

**Mod Varsayımı (Plan İçin Netlik):**
Bu doküman TP1/partial close akışını **One-Way Mode** varsayımıyla anlatır. Hedge Mode’da uygulanacaksa `positionSide` + `close` semantiklerinin ayrı bir “Hedge Mode” alt‑akışıyla açıkça tanımlanması gerekir.

## 4) API tarafında “Place order” nerede, churn açısından maliyeti ne?

**Not (Doğrulama):** Bu bölümdeki rate‑limit ve endpoint iddiaları **zamanla değişebilir**; uygulama öncesinde resmi dokümandan teyit edilmelidir.

**Neden önemli?** TP1/partial close uygulayınca genelde şunlar gelir:

* TP1 reduce-only emri
* Kalan pozisyon için SL/Trailing emirlerinin **qty** güncellemesi (çoğu borsada **cancel/replace**)
* (Opsiyonel) TP2 emirleri

Bu, “order placement/cancel” çağrı sayısını artırır. Rate limit artışı yardımcı olur ama yine de:

* **tek fonksiyonda batch/seri cancel‑replace** yapıp,
* “değişmediyse dokunma” (idempotent) mantığıyla churn’ü sınırlamak iyi olur.

## 5) SDK/örneklerde reduce-only’nin API parametre adı

BingX’in Help Center metni “Reduce-Only” kavramını net anlatıyor (davranış ve kullanım). ([BingX Help Center][1])
API dokümanındaki **parametre adı** ise genelde `reduceOnly` olarak geçiyor; bunu pratikte çeşitli istemciler/SDK’lar bu isimle expose ediyor (ör. bir BingX PHP client örneğinde `->reduceOnly()` builder’ı var). ([GitHub][3])

> Not: Bu son madde (parametre adı) **resmi help center değil**, topluluk/SDK tarafı. Resmi doğrulama için Place Order dokümanındaki parametre tablosunda “reduceOnly” alanını görmek idealdir.

## Bot tasarımına “TP1/partial close” ekleyeceksen 3 kritik teknik not

1. **Reduce-Only + qty = kalan pozisyonu bozmasın**
   TP1 fill olduktan sonra kalan pozisyon için çalışan SL/Trailing emirlerinin **miktarını** mutlaka kalan qty’ye çek (aksi halde fazla kapatma denemeleri, reject, gereksiz cancel/replace döngüsü).

2. **Close-all ile partial’ı karıştırma**
   Bazı borsalarda “closePosition=true” (tam kapat) ile “reduceOnly” aynı anda gelir; partial için **close-all mantığını devreye sokma**.

3. **Episode C gibi hızlı spike’larda**
   Reduce-Only, “ters tarafa döndürme” riskini azaltır ama **kötü fill/slippage** sorununu tek başına çözmez. Senin planındaki **fill-ref stop/TP düzeltmesi + slippage guard** hâlâ gerekli (reduce-only bunun yerine geçmez, tamamlar).

Aşağıya iki şeyi birlikte koyuyorum:

1. **BingX’te “reduce-only partial close” (TP1) nasıl çalışır / API’de neyi hedeflemeliyiz?**
2. **Senin “Final Uygulama Planı” için kabul kriterleri + telemetri alanları + golden-window regression eşikleri** (“başarılı sayılması için logda neyi görmeliyiz?”)

---

## 1) BingX’te reduce-only partial close (TP1) — araştırma özeti

### Reduce-Only’nin anlamı (mekanik)

BingX’in One-Way Mode açıklamasında, **reduce-only ile ters yönde emir girdiğinde bunun yeni pozisyon açmak yerine mevcut pozisyonu kapatma/azaltma amaçlı davrandığı** net şekilde anlatılıyor: örnekte “10 BTC long varken reduce-only seçilip 10 BTC short girilirse long kapanır.” ([BingX Help Center][1])
Bu davranış, TP1 için aradığımız temel garanti: **kısmi çıkış emri yanlışlıkla pozisyon “flip” ettirmesin.**

### API tarafında reduceOnly / closePosition alanları

BingX futures order endpointlerinde **`reduceOnly` ve `closePosition` parametrelerinin var olduğu** (ve boolean serileştirme hassasiyeti) çok sayıda client implementasyonunda açıkça görülüyor. Örn. BingX.Net sürüm notlarında “PlaceOrderAsync reduceOnly/closePosition düzeltildi” gibi kayıtlar var. ([GitHub][2])

> Not: Bu ortamda resmi “Place Order” dokümanındaki parametre tablosu okunmadı; bu yüzden “reduceOnly” alanını **resmi tablo üzerinden** ayrıca doğrulamak önerilir.

### Rate‑limit etkisi (TP1 + cancel/replace beraber düşünülmeli)

Bu bölümdeki rate‑limit ifadesi **zamanla değişebilir**; uygulama öncesi resmi kaynakla teyit edilmelidir.
TP1/partial exit ekleyince **emir sayın artacak** (entry + SL/TP + TP1 reduceOnly + (muhtemel) risk emirlerini yeniden boyutlandırma cancel/replace). Dolayısıyla kabul kriterlerine **“order-ops bütçesi”** koymak şart.

### TP1 için pratik “doğru uygulama” kuralı

* **TP1 tetiklenince**: mevcut pozisyonun *ters yönüne* **reduce-only** bir emir gönder; `qty = position_qty * tp1_fraction`.
* **TP1 dolunca**: kalan qty için **SL/TP (ve trailing)** emirlerini **kalan miktara** göre yeniden kur (aksi halde fazla reduce edip istenmeden flat/flip riski).
* One-Way Mode’da reduce-only, flip riskini ciddi azaltır. ([BingX Help Center][1])

---

## 2) “Final Uygulama Planı” için kabul kriterleri (telemetri + golden-window regression)

Aşağıyı, agent’a “Definition of Done / Acceptance Criteria” olarak direkt verebilirsin.

### A) Telemetri (log alanları) — “başarılı sayılması için logda neyi görmeliyiz?”

Minimum set:

#### A1) Execution / risk-adjust telemetrisi (fill sonrası geometri kanıtı)

Her pozisyon için (entry fill anında) tek satır “kanıt log”:

* `trade_id / position_id / strategy / symbol / side`
* `signal_price`, `signal_stop_price`, `signal_tp_price`
* `fill_price`
* `target_stop_ratio`, `target_tp_ratio`  *(senin planındaki ratio)*
* `real_stop_price`, `real_tp_price` *(fill’e göre hesaplanan)*
* `slippage_bps` = |fill - signal| / signal * 10000
* `effective_stop_dist_bps`, `effective_tp_dist_bps` *(fill’e göre)*
* `rr_planned` (signal üzerinden), `rr_after_fill` (fill üzerinden)
* `risk_orders_action` = `cancel_replace|create|skip`
* `order_ops_count` (bu trade’in toplam order operasyonu)

**Kabul:** Episode C ve benzeri pencerelerde, stop mesafesi “mikro-bps”e düşmemeli; ratio korunmalı (aşağıda eşikler).

#### A2) Slippage Guard telemetrisi (iki katman)

* Pre-trade: `slippage_guard_pretrade outcome=abort|pass gap_bps atr_5m_bps threshold_bps`
* Post-fill: `rr_revalidation outcome=keep|early_exit|move_to_be rr_after_fill`

**Kabul:** “abort” veya “early_exit” olduğu her olayda gerekçe + metrikler loglanmalı (sonradan regression ölçebilelim).

#### A3) Strateji veto / teyit telemetrisi

* `impulse_veto` için: `body_to_atr`, `sum2_to_atr`, `direction_mismatch`, `outcome=veto|pass`
* `rejection_confirmation` için: `has_red`, `rejected_from_band`, `outcome=wait|confirm`

**Kabul:** Episode C’de stop yediren girişlerde, artık ya **veto** ya da **wait** görülmeli (en azından “kapanış/ret teyidi yokken entry” azalsın).

#### A4) Cooldown / churn telemetrisi

* `cooldown_block symbol side remaining_sec reason=post_stop`

**Kabul:** Stop olduktan sonra aynı yönde 15 dk içinde entry denenirse **block** logu görmeliyiz.

#### A5) TP1 / partial close telemetrisi (reduce-only kanıtı)

* `tp1_triggered` anında: `tp1_price`, `tp1_fraction`, `planned_reduce_qty`
* `tp1_order_placed`: `reduceOnly=true`, `qty`, `side`, `order_id`
* `tp1_filled`: `filled_qty`, `remaining_position_qty`
* Ardından: `risk_orders_resized` (kalan qty’ye göre)

**Kabul:** TP1’de **reduceOnly=true** loglanmadan “partial close” sayılmaz. (Bu, flip riskini test edilebilir hale getirir.) ([BingX Help Center][1])

---

### B) Golden-window regression eşikleri (Episode C odaklı)

> Pencere: **Episode C (04:10–04:25)**
> Hedef: O iki stop-loss’un kök nedenlerini (teyit eksikliği / stop sıkılığı / slippage) “log üstünden” düzelttiğimizi kanıtlamak.

#### B1) Stop geometri korunumu (Entry-Ref fix’in “kanıtı”)

* `stop_ratio_error = abs(real_stop_ratio - target_stop_ratio)`
  **Eşik:** p95 `stop_ratio_error` ≤ **0.10 * target_stop_ratio**
  (yani fill sonrası stop, niyet edilen ratio’dan %10’dan fazla sapmasın)

* `effective_stop_dist_bps` için alt sınır:
  **Eşik:** `effective_stop_dist_bps ≥ max(10 bps, 0.25 * atr_5m_bps)`
  (Episode C’deki “2–3 bps stop” gibi saçmalıkları otomatik yakalar)

#### B2) Slippage Guard davranışı

* `gap_bps > 0.5 * atr_5m_bps` olduğunda:
  **Eşik:** pre-trade `ABORT_TRADE` oranı **%100** (bu koşul gerçekleştiyse girmemeli)

* Post-fill RR:
  **Eşik:** `rr_after_fill < 1.0` olduğunda **%100** “early_exit veya BE-adjust” aksiyonu loglanmalı (senin planınla uyumlu).

#### B3) Episode C stop-loss sayısı / hız metriği

* Episode C penceresinde:
  **Eşik (sert):** `stop_loss_count == 0`
  **Eşik (yumuşak, fallback):** `stop_loss_count <= 1` ve stop olan trade’de `impulse_veto` veya `rejection_wait` en az 1 kez görünmüş olmalı (yani “kör entry” kalmasın).

#### B4) Churn kontrolü

* Stop gerçekleştiyse:
  **Eşik:** 15 dk içinde aynı yönde yeni entry **0** (cooldown block logu beklenir)

#### B5) API/Emir operasyon bütçesi (rate-limit ve stabilite)

Rate‑limit değerleri değişebilir; bu yüzden aşağıdaki eşikler **korumacı öneri** olarak görülmelidir:

* Trade başına `order_ops_count` (entry + risk emirleri + tp1 + resize)
  **Eşik:** ortalama ≤ **6**, p95 ≤ **10**
* “rate-limit / too frequent” hatası:
  **Eşik:** **0 adet** (golden window’da)

---

## 3) Bu kabul kriterleri “senin tespit ettiğin sorunlara çözüm üretir mi?”

Evet — **doğru hedeflere vuruyor**, özellikle:

* Episode C’deki “stop geometrisi bozuldu / çok dar stop” kök nedenine doğrudan çözüm: **fill sonrası ratio ile stop/TP yeniden kurma** + telemetriyle kanıtlama.
* “Piyasa kaydı, trade artık valid değil” → **pre-trade slippage abort** + “post-fill RR re-validation”.
* “Impulse mumuna kafa atma / teyitsiz entry” → **impulse veto + rejection confirmation**.
* “Üst üste stop (A→B→C)” → **cooldown**.
* TP1 isteniyorsa: reduce-only partial close, One-Way Mode mantığıyla flip riskini düşürür. ([BingX Help Center][1])

En kritik ek not: TP1’i eklediğinde **mutlaka** “kalan pozisyon qty’ye göre risk emirlerini resize” et. Aksi halde TP1’le pozisyonu küçültürken SL/TP emirleri eski miktarla kalır → istemeden fazla kapatma / risk yönetimi sapması.

[1]: https://bingx.com/en/support/articles/23233089254169-perpetualfuturesone-waymodeexplained "BingX Help Center — Perpetual Futures | One-Way Mode Explained"
[2]: https://github.com/JKorf/BingX.Net "GitHub - JKorf/BingX.Net"
[3]: https://github.com/tigusigalpa/bingx-php "tigusigalpa/bingx-php"
