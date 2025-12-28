# 📄 ADR-004: Mean Reversion Stratejisi İçin "Tight Stop Deferral" ve "Stop Rewrite" Davranışı

* **Tarih:** 28 Aralık 2025
* **Durum:** ✅ Kabul Edildi (Accepted / Design Choice)
* **Kapsam:** Mean Reversion Stratejisi, Risk Manager, Execution Engine

### 1. Bağlam ve Gözlem (Context)

Mean Reversion (MR) stratejisinin entegrasyonu sırasında, sinyal üretiminden işlem açılışına kadar olan süreçte aşağıdaki döngü tespit edilmiştir:

1. **Sinyal Üretimi:** Strateji, teknik olarak geçerli (VWAP band ihlali) bir sinyal üretir.
2. **Risk Filtrelemesi (Defer):** `StrategyCoordinator` ve `RiskManager`, sinyalin teknik stop mesafesini çok dar (örn. <%0.15) ve hacmi yetersiz bulursa sinyali reddetmek yerine "Defer" (erteleme) mekanizmasıyla kuyruğa atar.
3. **Yürütme (Execution & Rewrite):** Sinyal kuyruktan çıkıp veya yeni bir sinyalle `LiveTradingEngine`'e ulaştığında, `Guardrail` (Güvenlik) mekanizmaları devreye girer.
4. **Sonuç:** `LiveTradingEngine`, risk modülünün "çok dar" bulduğu stop seviyesini, minimum güvenlik seviyesine (örn. %1 veya volatilite tabanlı buffer) **genişleterek (rewrite)** işlemi açar.

### 2. Sorun/Kısıt (Problem)

Risk modülü, işlem anında `LiveTradingEngine` tarafından uygulanacak olan "Genişletilmiş Stop" (Guardrail Stop) kuralından habersizdir. Bu nedenle, işlem aslında güvenli bir stop ile açılacak olsa bile, Risk Modülü ham sinyalin dar stop'una bakarak sinyali "riskli/gereksiz" olarak etiketleyip ertelemektedir (Throttle/Queue).

Bu durum bir yazılım hatası (bug) değil, iki modül arasındaki "Gerçeklik Kaynağı" (Source of Truth) uyumsuzluğudur.

### 3. Karar (Decision)

Bu davranışın **Mevcut Haliyle Korunmasına (WontFix)** karar verilmiştir.

Sistemi yeniden tasarlamak (Risk modülüne Execution kurallarını öğretmek veya MR için özel bypass kanalları açmak) yerine, mevcut güvenlik katmanlarının (Guardrails) nihai kararı verip işlemi güvenli bir şekilde açması "yeterli" kabul edilmiştir.

### 4. Gerekçe (Rationale)

* **Sistem Bütünlüğü:** Mevcut `StrategyCoordinator` ve `Queue` yapısı genel sistem sağlığı için kritik öneme sahiptir. MR için özel bir istisna (bypass) yaratmak, mimariyi karmaşıklaştıracaktır.
* **Güvenlik:** "Stop Rewrite" davranışı istenen bir güvenlik önlemidir. Botun çok dar stoplarla işlem açıp anında stop olmasını engellemektedir.
* **Operasyonel Risk:** Canlıya geçiş (Go-Live) aşamasında köklü bir refactoring yapmak, çalışan sistemi bozma riski taşır.

### 5. Sonuçlar ve Riskler (Consequences)

* **Pozitif:** İşlemler her zaman güvenli bir stop mesafesiyle (Guardrail garantisiyle) açılır.
* **Negatif (Risk):** "Defer" mekanizması nedeniyle sinyal üretimi ile işlemin gerçekleşmesi arasında zaman farkı (Latency) oluşabilir. Çok hızlı piyasa hareketlerinde MR stratejisi "fırsat penceresini" kaçırabilir.
* **İzleme:** Loglarda "Deferring signal" uyarılarının ardından işlemin başarıyla açıldığı görülecektir; bu bir hata değil, beklenen akıştır.

---

### 🏁 Son Durum: CANLIYA HAZIR (PRODUCTION READY)

Bu dokümantasyon ile birlikte, Mean Reversion entegrasyonuna dair açıkta kalan teknik belirsizlik kalmamıştır. Sistem; sinyal üretimi, risk kontrolü (kısıtlamalara rağmen) ve emir iletimi konularında **başarılı ve kararlı** çalışmaktadır.

**Konu kapatılmıştır.** 🚀