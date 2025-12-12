# Risk Motoru El Kitabı (Sprint 1 Sonrası)

## 1. Amaç ve Kapsam

Bu el kitabı:

* Yeni kurulan **risk yönetimi mimarisini** özetler,
* Azure App Configuration + ENV + YAML üzerinden **hangi parametrelerin nasıl çalıştığını** açıklar,
* Operatörün (sen veya takım arkadaşların) botu çalıştırırken:

  * Neye bakması gerektiğini,
  * Hangi logları kontrol etmesi gerektiğini,
  * Hangi durumlarda alarm sayılması gerektiğini
    net bir şekilde tarif eder.

Bu doküman **Sprint 1 Risk Refactor** sonrasındaki durumu anlatır ve “paper mode” / gerçek mod fark etmeksizin risk motorunun davranışını kapsar.

---

## 2. Yüksek Seviye Mimari

Risk sistemi kabaca üç katmandan oluşuyor:

1. **LiveTradingConfiguration**

   * Kaynaklar:

     * `config.example.yaml`
     * ENV değişkenleri
     * Azure App Configuration
   * Görevleri:

     * Üç kaynağı **öncelik sırasına göre** birleştirmek:

       * App Config > ENV > YAML
     * Risk ile ilgili alanları normalize etmek:

       * `risk.per_trade_risk_pct` → fraction (0.01 = %1)
       * `risk.daily_loss_limit_pct` → fraction
       * `risk.max_notional_pct_per_trade` → fraction
     * İnsan-okur özet oluşturmak:

       * “Capital: 100 USDT, Risk Per Trade: 1.00% (1.00 USDT max risk)” gibi.

2. **RiskConfiguration**

   * `LiveTradingConfiguration`’dan gelen **merged config** ile çalışır.
   * Görevleri:

     * Percent / fraction karışıklığını giderip **tek bir semantik** sağlamak:

       * `per_trade_risk_pct` / `daily_loss_limit_pct` → fraction (0.01 = %1)
       * `min_stop_pct`:

         * ≤ 1 ise fraction (`0.003` = %0.3)
         * > 1 ise yüzde kabul edilip /100’e bölünür (`3` → `0.03` = %3)
     * USD cinsinden risk değerlerini hesaplamak:

       * `max_risk_per_trade_usd` (ör: 1.00 USDT)
       * `daily_loss_limit_usd` (ör: 2.00 USDT)
       * `max_drawdown_usd` (ör: 10.00 USDT)
       * `computed_max_notional_usd` = `equity * max_notional_pct_per_trade`
     * Bu değerleri `RiskManager`’a geçmek üzere hazır hale getirmek.

3. **RiskManager + AdvancedPositionSizing**

   * `RiskConfiguration`’dan gelen risk limitlerini kullanır.
   * Görevleri:

     * Her yeni trade için:

       1. **Boyut hesaplama (sizing)**:

          * Stop floor + per-trade USD risk + min notional
       2. **Clamp (clip)**:

          * `max_position_notional_usd` sınırına göre pozisyonu kesmek
       3. **Risk kuralları**:

          * `CapitalLimitRule`, `PositionSizeRule`, günlük trade limiti vb.
       4. **Auto-resize fallback**:

          * Bakiye yetersizliğinde pozisyonu “paran kadar” küçültmeyi denemek
     * **Health check**:

       * `run_health_check()` ile kritik parametreleri ve bağımlılıkları doğrulamak
       * Startup’ta health durumunu log’lamak.

---

## 3. Konfigürasyon Akışı (Kaynak → Runtime)

### 3.1. Öncelik Zinciri

Risk ile ilgili tüm ayarlar için **öncelik sırası**:

1. **Azure App Configuration**
2. **ENV değişkenleri**
3. **YAML (`config.example.yaml`)**

LiveTradingConfiguration şu adımları uygular:

1. YAML’ı yükler.
2. YAML içindeki `# Override with:` yorumlarına göre ENV mapping’i çıkarır.
3. ENV override’larını uygular.
4. Azure App Config’ten değerleri alır, nested yapıya çevirir ve en son bunları merge eder.
5. Risk alanlarını normalize eder (fraction/percent vs.).
6. Bu merge edilmiş config’i `RiskConfiguration`’a geçirir.

### 3.2. Kritik Risk Key’leri ve Semantik

Aşağıdaki tablo en önemli risk parametrelerini özetler:

| Kavram                 | YAML / App Config key                   | ENV                          | Semantik                                                            |
| ---------------------- | --------------------------------------- | ---------------------------- | ------------------------------------------------------------------- |
| Sermaye                | `risk.equity_usd`                       | `CAPITAL_USDT`               | Toplam sermaye (USDT)                                               |
| Trade başına risk (%)  | `risk.per_trade_risk_pct`               | `PER_TRADE_RISK_PCT`         | Fraction: `0.01` = %1                                               |
| Stop floor (%)         | `risk.min_stop_pct`                     | `RISK_MIN_STOP_PCT`          | ≤1 → fraction, >1 → `%`/100                                         |
| Max notional (%)       | `risk.max_notional_pct_per_trade`       | `MAX_NOTIONAL_PCT_PER_TRADE` | Fraction: `0.75` = sermayenin %75’i                                 |
| Max notional (USD)     | `max_position_notional_usd`             | `MAX_POSITION_NOTIONAL_USD`  | Opsiyonel; set edilirse hard USD clamp                              |
| Min notional threshold | (şu an YAML/AppConfig’te exposed değil) | (ENV’de de yok)              | Internal default: 5.0 USDT; `min_notional`/`min_notional_threshold` |

> Not: `per_trade_risk_pct` ve `max_notional_pct_per_trade` **her yerde fraction** olarak çalışır (0.01 = %1, 0.75 = %75). Eğer App Config’te/ENV’de `1` yazarsan, kod bunu `%1` değil `%100` kabul eder, sonra normalize eder. Bu yüzden **0.01, 0.02, 0.75** gibi değerler kullanmak en temiz yaklaşımdır.

---

## 4. Trade Bazında Risk Akışı (Adım Adım)

Bir sinyal geldiğinde risk motoru kabaca şu adımları izler:

1. **Sinyal → Aday Pozisyon**

   * Strateji, sembol, yön (long/short), giriş fiyatı ve stop fiyatı ile bir “candidate position” üretir.

2. **RiskManager.size_and_validate_position()**

   1. `AdvancedPositionSizing` ile **pozisyon boyutunu hesapla**:

      * Girdi:

        * `entry_price`
        * `stop_price`
        * `risk_amount_usd` (≈ `equity * per_trade_risk_pct`, max `max_risk_per_trade_usd`)
        * `min_stop_pct`
        * `min_notional_threshold`
      * Stop floor:

        * `raw_stop_dist = abs(entry - stop)`
        * `min_stop_dist = entry * min_stop_pct`
        * `effective_stop_dist = max(raw_stop_dist, min_stop_dist)`
        * Eğer floor devreye girdiyse `floor_triggered = True` log’lanır.
      * Pozisyon büyüklüğü:

        * `qty = risk_amount_usd / effective_stop_dist`
        * `notional = qty * entry_price`
      * Min notional:

        * Eğer `notional < min_notional_threshold` ise `ValueError` atılır → trade kibarca reddedilir.
   2. **Clamp (clip)**:

      * Eğer `risk_limits['max_position_notional_usd']` set ise (ör: 75 USDT):

        * Pozisyon notional’ı bu limitin üstündeyse scale edilir:

          * `scale_factor = max_notional / current_notional`
          * `new_qty = qty * scale_factor`
          * `new_notional = max_notional`
        * Log: “Position clipped to limits … 1512 → 75 USD” gibi.
   3. **Risk kuralları**:

      * `CapitalLimitRule` (bakiye/kaldıraç yeter mi),
      * `PositionSizeRule` (max_position_size, max_portfolio_risk),
      * Günlük trade limiti (`daily_max_trades`),
      * vb.
   4. **Auto-resize fallback**:

      * Eğer `CapitalLimitRule` margin yetersizliği nedeniyle block ediyorsa:

        * `available_balance * leverage * 0.95` üzerinden **maksimum alınabilir notional** hesaplanır.
        * Bu değer `min_notional_threshold`’un üzerindeyse pozisyon yeniden scale edilir.
        * Log: “Position auto-resized due to capital limits …”.
        * Risk kuralları **sadece bir kez daha** çalıştırılır.
      * Eğer `max_affordable < min_notional_threshold` ise:

        * Trade `resize_failed = True` ile reddedilir (sistem çökmeksizin).

3. **Sonuç**

   * `ValidationResult` ile döner:

     * `blocked` (True/False)
     * `clipped`
     * `resized` / `resize_failed`
     * `floor_triggered`
   * Trade açılırsa execution katmanına geçer; reddedilirse stratejiye/log’a yansır.

---

## 5. Operatör İçin “Önce Konfigürasyon” Check-list’i

Botu açmadan önce (veya yeni deploy sonrası) aşağıdaki adımları uygula:

### 5.1. Azure App Configuration Değerlerini Kontrol Et

En kritik key’ler:

* `capital_usdt` (veya ENV: `CAPITAL_USDT`)
* `risk.per_trade_risk_pct` (veya `PER_TRADE_RISK_PCT`):

  * Örnek: `0.01` → sermayenin %1’i
* `risk.min_stop_pct` (veya `RISK_MIN_STOP_PCT`):

  * Örnek: `0.003` → %0.3 stop floor
* `risk.max_notional_pct_per_trade`:

  * Örnek: `0.75` → sermayenin %75’i kadar notional clamp
* Opsiyonel: `max_position_notional_usd`:

  * Örnek: `75` → her trade için max 75 USDT notional

Bu değerleri App Config portalından kontrol et; fraction semantiğini unutma.

### 5.2. Paper Mode Başlangıç Loglarını Kontrol Et

Bot paper mode’da açıldığında aşağıdaki log satırlarını özellikle kontrol et:

1. **Config özet** (LiveTradingConfiguration):

   * “Capital: 100.00 USDT”
   * “Risk Per Trade: 1.00% (1.00 USDT max risk)”
   * “Max Notional Per Trade: 75.00 USDT”
     Bu özet **niyet ettiğin konfig ile uyumlu olmalı**.

2. **RiskConfiguration USD hesapları**:

   * “Per-Trade Risk: 1.00% = $1.00”
   * “Daily Loss Limit: 2.00% = $2.00”
   * “Max Drawdown: 10.0% = $10.00”
     Eğer burada 0.01 USD gibi anlamsız küçük değerler görürsen, fraction/percent mapping’inde sorun var demektir.

3. **RiskManager risk_limits**:

   * `max_position_notional_usd`: 75.0 (veya sen ne istiyorsan)
   * `max_risk_per_trade_usd`: 1.0
   * `min_stop_pct`: 0.003
   * `min_notional_threshold` ve `min_notional`: 5.0
     Bu değerler runtime davranışını doğrudan belirlediği için mutlaka gözünle doğrula.

4. **Health check**:

   * “RiskManager health check PASSED”
   * Startup’ta “RiskManager startup health … HEALTHY” satırı.
     Eğer “FAILED” veya `UNHEALTHY` görürsen, botu gerçek modda çalıştırma; önce config ve bağımlılıkları düzelt.

---

## 6. Çalışma Sırasında İzlenmesi Gereken Loglar

Bot çalışırken (paper ya da live):

### 6.1. Normal Davranışta Görmeyi Beklediğin Log Tipleri

* **Stop floor tetiklenmesi**:

  * Mesaj içeriğinde `floor_triggered: true` veya benzeri.
  * Bu, stop mesafesinin fazla dar olduğu durumlarda devreye girmeli.
* **Clip (max notional)**:

  * “Position clipped to limits … original_notional → clipped_notional”
  * Örnek: “1512 → 75 USD”
* **Auto-resize (margin sınırlı)**:

  * “Position auto-resized due to capital limits …”
  * İçinde:

    * `original_notional`
    * `new_notional` (ör: 950 USD bantları)
    * `available_balance`, `leverage`
* **PositionSizing reddi**:

  * “Position sizing rejected trade … Notional X < minimum Y”
  * Bu durumda:

    * Sistem crash olmaz,
    * Trade sadece “çok küçük” olduğu için reddedilir.

Bu loglar, sistemin tasarladığımız gibi davrandığını gösterir.

### 6.2. Alarm Kabul Edilmesi Gereken Durumlar

Aşağıdakiler “inceleme gerektiren” durumlar:

1. **Health check FAILED**

   * “RiskManager health check FAILED” veya `status: "UNHEALTHY"`
     → Yapılacaklar:
   * Log detayında hangi key’in `ok: False` olduğuna bak.
   * Örn: `config_min_stop_pct` invalid vs.
   * App Config / ENV / YAML değerlerini düzelt.

2. **Sürekli `CapitalLimitRule` block’ları, resize olmadan**

   * Çok sayıda “CapitalLimitRule blocking … insufficient margin” mesajı, **hiç auto-resize log’u olmadan**
     → Auto-resize tetikleme koşulları çalışmıyor olabilir (örn. mesaj formatı değişmiş).
   * Log örnekleri ile birlikte koda tekrar bakmak gerekir.

3. **Sürekli `resize_failed: true`**

   * “Auto-resize failed: balance too low …” ve her trade’de block
     → Bu durumda:
   * Ya sermaye çok küçük,
   * Ya `min_notional_threshold` gereğinden yüksek,
   * Ya kaldıraç çok düşük.
     → Parametreleri gözden geçir (özellikle equity, leverage ve min_notional_threshold).

4. **Özet log ile runtime davranışı uyumsuz**

   * Özet “1.00 USDT max risk” diyor ama trace log’larda `max_risk_per_trade_usd` bambaşka bir değer (ör: 0.01) görünüyorsa
     → Fraction / percent mapping’inde kaçak vardır; config zinciri yeniden incelenmelidir.

---

## 7. Tipik Operasyon Akışı (Pratik Rehber)

### 7.1. Yeni Deploy Sonrası

1. Yeni imaj / tag ile container’ı yayınla.
2. Azure App Config ayarlarını gözden geçir:

   * sermaye, per_trade_risk, min_stop_pct, max_notional_pct.
3. Botu **paper mode**’da başlat.
4. İlk 1–2 dakikalık logları kontrol et:

   * Config summary,
   * RiskConfiguration USD hesapları,
   * RiskManager risk_limits,
   * Health check PASS.
5. Paper run’ı en az 20–30 dakika çalıştır;

   * Mümkünse trade üretecek piyasa koşullarında (veya test sinyali ile).
6. Trade içeren logları incele:

   * stop floor tetikleniyor mu?
   * clamp çalışıyor mu?
   * auto-resize gerektiğinde devreye giriyor mu?

Her şey yolunda ise aynı imajı **live mode** için de kullanabilirsin.

---

## 8. Sonraki İterasyonlar İçin Notlar (Sprint 2+)

Bu el kitabı **Risk davranışını** Sprint 1 sonrası için stabil hale getiriyor. Bir sonraki adımlar (şu an yapılması zorunlu değil):

* Prometheus metrikleri:

  * stop floor trigger sayısı,
  * auto-resize sayısı,
  * risk validation süresi,
  * margin error oranı.
* HTTP health endpoint (`/health/risk-manager`):

  * Logic App / dış izleme için kullanılabilir.
* Shadow mode / canary rollout:

  * Yeni risk kurallarını önce sadece log’layıp, sonra küçük bir oranla gerçek karar verdirerek ilerlemek.
* Exchange adapter ile `normalize_quantity`:

  * Borsanın `min_qty`, `step_size` kurallarına göre miktar yuvarlama.

---

## 9. Özet

* Artık **stop floor**, **min notional** ve **max notional clamp** ile risk yönetimi deterministik ve korunmuş durumda.
* **Per-trade risk** fraction semantiğiyle (0.01 = %1) uçtan uca tutarlı.
* `max_position_notional_usd` hem hesaplanıp hem de gerçekten enforce ediliyor.
* **Health check** startup’ta çalışıyor ve testlerle güvence altında.
* Operatör, sadece:

  * Doğru App Config değerlerini girdiğinden,
  * Başlangıç loglarında konfig & health satırlarını gördüğünden,
  * Çalışma sırasında stop floor / clamp / auto-resize loglarını makul aralıkta izlediğinden
    emin olarak sistemi güvenle işletebilir.
