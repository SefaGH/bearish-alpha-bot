## Risk Engine & Size Planner El Kitabı

### 1. Sistem ne işe yarar?

Bu sistem, **her bir trade için pozisyon büyüklüğünü**, hesap sermayesini, risk iştahını ve limitleri dikkate alarak:

1. Önce **stratejinin ürettiği sinyal üzerinden risk-bazlı “ham” pozisyonu** hesaplar.
2. Sonra bu ham pozisyonu, **global limitler (yüzde, notional, sermaye, risk/heat)** ile **tek bir merkezde (Size Planner)** kısar.
3. Son olarak, **risk kuralları** ve **sermaye/margin kontrolleri** üzerinden geçirip, trade’i:

   * Açmaya izin verir, veya
   * Net, loglanmış bir gerekçeyle reddeder.

Amaç:

* “1$ risk istiyorum ama 199$’lık pozisyon açıldı” gibi patolojik durumları bitirmek,
* Küçük hesaplarda “dev pozisyon” denemelerini güvenli şekilde kırpmak,
* Tüm davranışı **konfigürasyon + loglar üzerinden açıklanabilir** hale getirmek.

---

## 2. Temel Bileşenler ve Kavramlar

### 2.1 Konfigürasyon katmanı

Ana kaynak: `config.example.yaml` + ENV + Azure App Configuration.

Özet akış:

1. **YAML**: Varsayılan değerler.
2. **ENV override**: `CAPITAL_USDT`, `PER_TRADE_RISK_PCT`, `RISK_MIN_STOP_PCT`, vb.
3. **Azure App Configuration**:

   * Önemli nested key örnekleri:

     * `risk.size_planner_enabled` (planner flag)
     * `risk.per_trade_risk_pct`
     * `risk.max_position_size`
     * `risk.max_notional_pct_per_trade`
     * `risk.max_position_notional_usd`
     * `risk.min_stop_pct`
   * Artık nested key’ler (`risk.xxx`) doğru şekilde parse edilip merge ediliyor.

**Öncelik genel olarak:**

* App Config → ENV → YAML
* Sadece `RISK_SIZE_PLANNER_ENABLED` için özel kural:

  * ENV (`RISK_SIZE_PLANNER_ENABLED`) **>** config (`risk.size_planner_enabled`) **>** yoksa default (shadow).

### 2.2 RiskConfiguration

`RiskConfiguration` şunları yapar:

* **Sermaye ve risk yüzdeleri:**

  * `equity_usd` (örn. `CAPITAL_USDT`)
  * `per_trade_risk_pct` → fraksiyon (0.01 = %1)
  * `computed_max_risk_usd = equity * per_trade_risk_pct`
  * `daily_loss_limit_pct` → günlük kayıp limiti (USD’e çevrilir)
  * `max_drawdown_pct` → maksimum çekilme limiti (USD’e çevrilir)

* **Stop & notional:**

  * `min_stop_pct`

    * YAML / App Config / ENV’den gelir.
    * `>1` ise yüzde kabul edilip `/100` ile normalize edilir (50 → 0.5).
  * `min_notional_threshold`

    * Default ≈ 5 USDT; küçük notional “toz” işlemleri engellemek için kullanılır.

* **Büyüklük limitleri:**

  * `max_position_size` → pozisyon büyüklüğü limiti (fraction of equity; 0.10 = %10).
  * `max_notional_pct_per_trade` → trade başına notional çarpanı (equity * multiple; 0.75 = %75, 10.0 = 10x).
  * `max_position_notional_usd`

    * Öncelik: explicit USD > computed (equity * max_notional_pct_per_trade) > None.

Bunlar `RiskManager` içinde `risk_limits` sözlüğüne taşınır.

### 2.3 AdvancedPositionSizing (APS)

APS’in görevi: **stratejiden gelen stop & sinyal üzerinden risk-bazlı ham notional hesaplamak.**

Girdi:

* `equity_usd`
* `per_trade_risk_pct` (fraksiyon)
* `risk_usd_cap` (opsiyonel üst limit)
* `min_stop_pct`
* Volatilite multipliers (low_vol_multiplier, baseline vs.)
* Sinyalin entry ve stop fiyatları

Çıktı:

* `raw_notional` (risk bazlı ham pozisyon, stop floor uygulanmış)
* `effective_risk_usd` (risk_usd_cap ile kısıtlanmış gerçek risk)
* `effective_stop_pct` (floor sonrası stop yüzdesi)
* `floor_triggered` (min_stop devreye girdi mi?)

Önemli: APS, **risk matematiği** ve **stop/volatilite** işinden sorumlu.
Global limitlerden (max size %, max notional, sermaye, heat) sorumlu değil.

### 2.4 Size Planner (RiskManager içinde, Option B)

**Size Planner = Global boyut yöneticisi.**

Planner, APS’ten gelen `raw_notional`’ı alıp şu limitlerle tek seferde “clip” eder:

Kullanılan değerler:

* `equity` (toplam özsermaye)
* `available_balance` (serbest bakiye)
* `max_position_size`
* `max_position_notional_usd` / `max_notional_pct_per_trade`
* `min_notional_threshold`
* `max_portfolio_risk_usd` & `current_open_risk_usd`
* `leverage`
* `price`

Kaplar:

1. **Size % cap**
   `cap_size_pct = equity * max_position_size`
2. **Notional cap**
   `cap_notional = max_position_notional_usd`
   veya `equity * max_notional_pct_per_trade`
   yoksa `∞`
3. **Sermaye / margin cap (capital cap)**
   `cap_capital = compute_max_affordable_notional(available_balance, leverage, 0.95)`
   (CapitalLimitRule ile aynı formül, shared helper)
4. **Portfolio heat cap**

   * `cap_heat = max_portfolio_risk_usd - current_open_risk_usd`
   * Risk birimi: USD (APS’in kullandığı risk USD ile aynı)

Planner hesapları:

```text
planned_notional = min(
    raw_notional,
    cap_size_pct,
    cap_notional,
    cap_capital,
    cap_heat (varsa)
)
```

**Min notional enforcement (tek nokta):**

* Eğer `planned_notional < min_notional_threshold` ise:

  * Trade **reddedilir** (clip edilmez).
  * Reason kodu: `REJECT_TOO_SMALL_AFTER_CAP`
  * Eğer asıl sebep heat ise: `portfolio_heat_exhausted`.

**Policy: `position_size_policy`**

* `clip` (default):

  * Size & notional kapları trade’i **çekerek** uygular (10’dan büyükse 10’a indir).
  * Capital/heat kapları da clip davranışıyla çalışır (min_notional üzerindeyse).
* `reject`:

  * Size kaynaklı kap (size_pct / max_notional) bind ediyorsa: **direct reject** (clip yok).
  * Capital/heat kapları yine clip olarak davranır (kasıtlı olarak daha az agresif).

Planner çıktısı: `PlannedSizeResult`

* `planned_notional`, `planned_qty`
* `capped_by_size_pct`, `capped_by_max_notional`, `capped_by_capital`, `capped_by_heat`
* `below_min_notional`
* `reason` (ör: `REJECT_TOO_SMALL_AFTER_CAP`, `portfolio_heat_exhausted`, None)

### 2.5 Risk kuralları (guard rails)

Planner’dan sonra trade, **Risk kuralları**na gider:

* `CapitalLimitRule`

  * Aynı affordability helper’ı kullanır.
* `PositionSizeRule`

  * Artık **guard rail**:

    * Planner aktifken, **signal.notional**’ı kullanır.
    * Planner’ın cap ettiği notional > size caps ise bu bir bug; o durumda anomaly log basar.
* `PortfolioHeatRule`

  * Planner ile aynı `compute_portfolio_open_risk_usd` helper’ını kullanır.
* Diğer kurallar:

  * Max drawdown
  * Risk/Reward ratio
  * Günlük trade limiti vb.

### 2.6 Execution

* Planner **aktif** ise:

  * Strategy risk assessment’ten gelen `final_notional` / `final_position_size`,
    execution öncesinde sinyalin içine yazılır.
  * LiveTradingEngine, PPO / multipliers ile **boyutu tekrar değiştirmez**.
  * `ENQUEUED ... Size: $X` log’undaki X, planner’ın `planned_notional` değeridir.
* Planner **shadow/kapalı** ise:

  * Legacy sizing + multipliers devrededir; planner sadece log/telemetry üretir.

---

## 3. Loglar ve Etiketler

Agent’ın bakacağı kritik log etiketleri:

1. **Planner flag (startup)**

   * `[RISK-PLANNER-FLAG] size_planner_flag_resolved`
   * İçerik:

     * `env_value`
     * `config_value`
     * `resolved_mode` = `active` / `shadow`
2. **Strategy path**

   * `[RISK-PLANNER] strategy_path`
   * Önemli alanlar:

     * `symbol`
     * `mode` (`active` / `shadow`)
     * `raw_notional`
     * `final_notional`
     * `notional_delta_abs`
     * `notional_delta_ratio`
3. **Planner kararı**

   * `[RISK-PLANNER] size_planner.decision`
   * Alanlar:

     * `raw_notional`, `planned_notional`, `planned_qty`
     * `capped_by_size_pct`, `capped_by_max_notional`, `capped_by_capital`, `capped_by_heat`
     * `below_min_notional`
     * `reason`
     * `shadow_mode`
4. **Risk validation path**

   * `[RISK-PLANNER] validate_path`
   * Planner modu + rule sonuçları.
5. **Anomali log**

   * `[RISK-PLANNER] anomaly_position_size_rule`
   * Planner aktifken PositionSizeRule hâlâ “exceeds max” vs. görürse tetiklenir.
6. **Health check**

   * `RiskManager health check PASSED/FAILED`
7. **ENQUEUED logları**

   * `ENQUEUED [Signal] ... Size: $X`
   * Buradaki X, gerçek execution notional’ı.

---

## 4. Operasyonel Checklist (AI Agent için)

### 4.1 Deployment / yeni versiyon sonrası

1. **Flag doğrulama**

   * `python scripts/print_risk_planner_flag.py` çıktısını kontrol et:

     * `env_value`
     * `config_value`
     * `resolved_mode`
   * Logda `[RISK-PLANNER-FLAG] ... resolved_mode=...` satırını ara; betik ve runtime aynı şeyi söylüyor mu?

2. **RiskManager health**

   * Startup’ta:

     * `RiskManager health check PASSED`
     * `config_max_position_notional_usd`

       * None veya >0 olmalı.
     * `config_min_stop_pct`, `config_min_notional` >0.

3. **Risk config özeti**

   * Sermaye (`equity_usd`)
   * `per_trade_risk_pct`
   * `max_position_size`
   * `max_notional_pct_per_trade` / `max_position_notional_usd`
   * `min_stop_pct`
   * `min_notional_threshold`
   * `max_portfolio_risk_usd` (varsa)
   * Bunlar, beklenen/istenen risk profili ile uyumlu mu?

### 4.2 Paper run / kısa canlı run sonrası

Her sinyal için (özellikle ilk 1–3 sinyal):

1. **Strategy path kontrolü**

   * `[RISK-PLANNER] strategy_path`:

     * `mode`:

       * `active` → Planner cap ediyor ve execution’a kararını aktarıyor.
       * `shadow` → Planner sadece izliyor; legacy sizing devrede.
     * `raw_notional` vs `final_notional`:

       * Genelde `final_notional <= raw_notional` olmalı.
       * Çok büyük fark varsa (örneğin 333 → 10), bu bilerek (caps) mi, yoksa aşırı mı?

2. **Planner kararı**

   * `[RISK-PLANNER] size_planner.decision`:

     * Hangi cap bind etmiş?

       * Küçük hesap için sık pattern: `capped_by_size_pct=True`.
       * Sermaye yetersizliğinde: `capped_by_capital=True`.
       * Heat limitine vurduysa: `capped_by_heat=True` + `reason="portfolio_heat_exhausted"`.
     * `below_min_notional=True` ise:

       * `reason="REJECT_TOO_SMALL_AFTER_CAP"` beklenir.
       * Trade açılmaz; bu normaldir ama çok sık oluyorsa optimizasyon ihtiyacı var demektir.

3. **Execution hizası**

   * `ENQUEUED ... Size: $X` satırını,
     aynı sinyal için `planned_notional` ile kıyasla:

     * Planner aktifse → `X ≈ planned_notional` olmalı.
   * Order result loglarında da order değeri bu civarda olmalı.

4. **Risk kural sonucu**

   * `PositionSizeRule` loglarını tara:

     * Planner **active** iken:

       * Normalde **REJECT** etmemeli.
       * Eğer REJECT ediyorsa:

         * Aynı zamanda `[RISK-PLANNER] anomaly_position_size_rule` logu görülmeli.
         * Bu bir bug / investigate-case olarak işaretlenmeli.
   * `CapitalLimitRule` / `PortfolioHeatRule`:

     * Nadiren reject etmesi normal; özellikle hesap çok küçük / risk ayarı agresif ise.
     * Sık sık capital/heat reject görüyorsan, risk ayarları fazla agresif demektir.

---

## 5. Optimizasyon Playbook’u

AI agent’ın, log + config üzerinden önerebileceği optimizasyonlar:

### 5.1 Risk bütçesi (per trade)

**Parametreler:**

* `risk.per_trade_risk_pct` (App Config) / `PER_TRADE_RISK_PCT` (ENV)
* `risk.risk_usd_cap` / `RISK_USD_CAP`

**Gözlem → Aksiyon örnekleri:**

* Eğer:

  * `raw_notional` çok küçük (ör. 2–3 USDT),
  * `planned_notional` da `min_notional_threshold` sebebiyle sık sık `REJECT_TOO_SMALL_AFTER_CAP`,

  → Ajan şunları önerebilir:

  * `per_trade_risk_pct`’i hafifçe artır (0.01 → 0.015 gibi),
  * veya `min_notional_threshold`’i düşür (borsa min notional’ı izin veriyorsa),
  * veya stop/volatilite ayarlarını (min_stop_pct) yeniden kalibre et.

* Eğer:

  * `raw_notional` sürekli çok büyük (ör. equity 100, raw 400+),
  * Planner sürekli `capped_by_size_pct` veya `capped_by_capital`,

  → Öneri:

  * `per_trade_risk_pct`’i düşür (0.01 → 0.005),
  * `risk_usd_cap` tanımlayarak tek trade riskinin üst limitini düşür.

### 5.2 Büyüklük limitleri (size caps)

**Parametreler:**

* `risk.max_position_size` / `MAX_POSITION_SIZE_PCT`
* `risk.max_notional_pct_per_trade` / `MAX_NOTIONAL_PCT_PER_TRADE`
* `risk.max_position_notional_usd`

**Gözlem → Aksiyon:**

* Planner loglarında:

  * Her sinyalde `capped_by_size_pct=True` ve **daha yüksek risk istiyorsun**:

    * `max_position_size`’ı artır (0.10 → 0.2 gibi).
* Eğer:

  * Çok büyük notional önerileri var, ama sistem bunları `max_position_notional_usd` ile kırpıyor ve sen daha fazla risk istemiyorsun:

    * `max_position_notional_usd`’i sabit kalarak bırak (bu zaten güvenlik kapağı).
* Eğer capital reject’leri çok sık:

  * `max_position_size` ve `max_notional_pct_per_trade` değerleri, sermayeye göre fazla agresif olabilir → düşür.

### 5.3 Stop floor ve min_notional

**Parametreler:**

* `risk.min_stop_pct` / `RISK_MIN_STOP_PCT`
* `risk.min_notional_threshold` (şimdilik internal; ileride dışarı açılabilir)

**Gözlem → Aksiyon:**

* Eğer:

  * Stoplar çok sık `floor_triggered=True`,
  * `raw_notional` 100+ gibi büyük değerlere gidiyor,
  * Planner sürekli aşağı kırpıyor,

  → Öneri:

  * `min_stop_pct`’i biraz artırarak **stopları genişlet**; böylece `raw_notional` nominalde küçülür.
* Eğer `REJECT_TOO_SMALL_AFTER_CAP` sayısı yüksek:

  * Min notional’ı (threshold) yeniden düşün:

    * Borsa min notional’ı 5 ise, threshold’u belki 3–4 gibi daha düşük bir değere çekebilirsin (likiditeye göre).

### 5.4 Heat ve circuit breaker

**Parametreler:**

* `max_portfolio_risk_usd` (App Config üzerinden risk kısmı)
* `risk.daily_loss_limit_pct`
* `risk.daily_max_trades`

**Gözlem → Aksiyon:**

* Sık `portfolio_heat_exhausted`:

  * Günlük / toplam risk limitlerin çok sık devreye giriyor → belki risk per trade çok yüksek.
* Sık günlük kayıp limiti (/ circuit breaker) tetikleniyorsa:

  * `daily_loss_limit_pct`’i yeniden kalibre et,
  * veya stratejinin kalitesi / SL mesafeleri gözden geçirilmeli (bu kısım artık strateji optimizasyonu).

---

## 6. Karlılık Odaklı Geliştirme Önerileri

Bu bölüm, risk altyapısı oturduktan sonra **kârlılığı artırmak** için atılacak adımların çerçevesi.

### 6.1 Dinamik risk ayarı

* `per_trade_risk_pct` ve/veya `risk_usd_cap`’i:

  * Volatiliteye göre (yüksek vol → daha düşük risk, düşük vol → biraz daha yüksek),
  * Regime’e göre (trend modu vs range modu),
  * Strateji güven skoruna göre (yüksek güvenli sinyallerde daha yüksek risk) dinamik hale getirmek.

### 6.2 Planner telemetrisi ile tuning

* Planner loglarından:

  * `notional_delta_ratio` dağılımı,
  * Hangi cap’in ne sıklıkta bind ettiği,
  * `REJECT_TOO_SMALL_AFTER_CAP` ve `portfolio_heat_exhausted` frekansı
    gibi istatistikleri toplayıp:
  * Sistem çok sık “frene mi basıyor”?
    → Risk ayarı planınıza göre çok agresif/temkinli mi?
  * Kullanılmayan risk kapasitesi var mı?
    → Günlük / toplam risk limitleri sürekli kullanılmadan mı kalıyor?

Bu analiz sonucu agent şunları önerebilir:

* Risk per trade’i hafif artırmak veya azaltmak,
* Max size / notional limitlerini sermayeye göre yeniden ölçeklendirmek,
* Belirli semboller için (ör. BTC vs altcoin) farklı caps kullanmak.

### 6.3 Strateji & execution iyileştirmeleri

Risk sistemi sağlam; bundan sonrası daha fazla strateji tarafı:

* **SL / TP optimizasyonu**:

  * Stop mesafesi ve `min_stop_pct` ilişkisini gözleyerek,
  * Fazla sık floor’a takılıyorsa, stratejinin “too tight stop” üretimi azaltılmalı.
* **Position scaling / partial close**:

  * Planner şu an tek-shot pozisyon için çalışıyor.
  * Gelecekte kademeli giriş/çıkış (scale-in, scale-out) modelleri eklenebilir.
* **Market microstructure / slipaj**:

  * `planned_notional`’a karşılık gelen fill fiyatları ve realized PnL üzerinden,
  * Slipaj ve komisyon sonrası net getiriyi ölçüp,
  * Spread geniş zamanlarda risk azaltma / trade pas geçme mantığı eklenebilir.

---

## 7. AI Agent için Kullanım Rehberi

Bu el kitabı, bir AI agent’ın şu patternle çalışmasını hedefliyor:

1. **Girdi:**

   * Son run’a ait log dosyası (özellikle `[RISK-PLANNER-*]`, `ENQUEUED`, `RiskManager health`, `PositionSizeRule`, `CapitalLimitRule` satırları).
   * Mevcut konfig (özellikle risk kısmı ve App Config override’ları).

2. **Agent adımları:**

   1. Planner flag & health check → doğru ortamda mıyız?
   2. İlk birkaç sinyal için:

      * raw vs final notional → caps davranışı
      * ENQUEUED size hizası
      * Anomali log var mı?
   3. Reddedilen trade’lerin nedenlerine bak:

      * `REJECT_TOO_SMALL_AFTER_CAP`
      * `portfolio_heat_exhausted`
      * CapitalLimitRule reject
   4. Bu el kitabındaki **Optimizasyon Playbook’u**na göre:

      * Hangi parametrelerin hangi yönde ayarlanabileceğini öner.
   5. Son olarak, kullanıcıya:

      * “Şu an sistem X şekilde davranıyor (ör: her trade’i 10$’a clip’liyor).”
      * “Eğer daha agresif olmak istersen şu parametreleri şöyle şöyle artır/düşür.”
      * “Eğer çok sık şu nedenle reddediyorsan, Y ayarını revize et.”
        şeklinde somut aksiyon listesi sun.

3. **Çıktı:**

   * Net teşhis (diagnostic)
   * Parametre bazlı öneriler
   * Gerekirse “önce şu paper senaryoları ile test et” şeklinde küçük test planı.
