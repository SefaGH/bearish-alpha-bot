# Bearish Alpha Bot – Mimarî, Flow ve Lifecycle
> `/docs/architecture.md` — Son güncelleme: 2025-11-10

Bu belge, **Bearish Alpha Bot (v3.2)** mimarisini, **uçtan uca iş akışını (flow)**, **yaşam döngüsünü (lifecycle)** ve
**katmanlı + merkezi yönetim** modelini **rakam/parametre detayına girmeden** kavramsal düzeyde anlatır. Görseller:
- [Mimari Blok Diyagramı](./bearish_alpha_architecture.png)
- [Lifecycle / Swimlane](./bearish_alpha_lifecycle.png)

---

## 1) Mimari Genel Bakış (Katmanlar)

### A. Orkestrasyon / Operasyon (Merkezi beyin)
- Tüm alt sistemleri **başlatır/durdurur**, **sıralar** ve **birbirine bağlar**.
- Sistem **durumunu** (initialized/running), **zamanlayıcıları**, **health‑check** ve **graceful shutdown** süreçlerini yönetir.
- **Tek entegrasyon noktasıdır**: Strateji, ML/RL, Veri/WS ve Yürütme katmanları birbirine **doğrudan değil**, **merkez üzerinden** konuşur.

### B. Konfigürasyon (Single Source of Truth)
- **YAML + ENV override** ile **tip‑güvenli etkin konfigürasyon** (effective config) derlenir.
- Tüm katmanların politika/parametreleri **merkezden** dağıtılır; değişiklikler **tek noktadan** yönetilir.
- **Final config özeti** ile tanı koymayı kolaylaştırır.

### C. Borsa & İletişim (Exchange/WS)
- **CCXT istemci**: emir/bakiye/market-verisi için senkron API.
- **BingX WebSocket + WS‑Manager**: çoklu zaman dilimi abonelikleri, bağlantı yaşam döngüsü, yeniden bağlanma ve sağlık doğrulama.
- **StreamDataCollector**: akışı **tamponlar**, **throttle/backpressure** uygular, üst katmanlara **temiz veri** sağlar.

### D. Veri & Özellik Mühendisliği
- Ham OHLCV akışından **özellik vektörü (state)** üretir; **çoklu TF hizalama**, **normalizasyon** ve **pencere** işlemleri.
- Üst katmanlara **tutarlı** ve **deterministik** bir veri yüzeyi sunar.

### E. ML / Zekâ Katmanı (Zenginleştirme & Filtreleme)
- **Fiyat tahmini** (ensemble), **rejim/trend tahmini** (ensemble) çıktılarını strateji sinyaliyle **birleştirir**.
- **AI‑Enhanced Adapter** ham strateji sinyalini rejim/tahmin bağlamıyla **ağırlıklandırır/filtreler**.
- **RL Ajanı** son kalite kapısıdır: sinyali **onayla / beklet / reddet** kararını verir.

### F. Strateji Katmanı
- Modüler stratejiler (örn. `adaptive_ob`, `adaptive_str`) ortak strateji arayüzüne uyar.
- Girdi: özellik vektörleri (+ rejim ipuçları). Çıktı: BUY/SELL/FLAT içeren **ham sinyal** (+ opsiyonel TP/SL/RR önerileri).
- Stratejiler **“ince”** tutulur; davranış/politikalar **merkezî konfigden** gelir.

### G. Yürütme & Pozisyon Yönetimi
- **Execution kapısı**: yalnızca **tüm filtreleri geçen** sinyaller **emre** dönüşür (örn. market/IOC).
- **Duplicate/cooldown** kuralları, **günlük/oturum risk** politikaları, **pozisyon izleme** ve **emir yaşam döngüsü** burada uygulanır.

> **Görsel:** [Mimari Blok Diyagramı](./bearish_alpha_architecture.png)

---

## 2) Flow (Uçtan Uca Sinyal Boru Hattı)

1. **Veri girişi** → WS‑Manager akışları açar, Collector tamponlar ve düzenler.  
2. **Özellik üretimi** → çoklu TF hizalanmış, normalize **state** oluşturulur.  
3. **Ham sinyal (Strateji)** → kural tabanlı tetikleyicilerle BUY/SELL/FLAT üretilir.  
4. **Filtreleme / zenginleştirme (ML/RL)**  
   - **Duplicate‑prevention**: çok yakın/tekrarlı sinyaller elenir.  
   - **Rejim/Trend**: güven düşükse **yoksay/soft‑weight**, uygunsa **destekle**.  
   - **RL Ajanı**: **go / hold / reject**.  
5. **Yürütme** → yalnızca **kapılardan geçen** sinyaller emre dönüşür.  
6. **Geri besleme** → emir/pozisyon/latency/telemetri metrikleri üst katmanlara ve orkestrasyona akar.

---

## 3) Lifecycle (Yaşam Döngüsü)

### A. Başlatma / Hazırlık
- **Launcher/CI** argümanları çözer (mod/süre/log).  
- **Konfig derle** (YAML + ENV) → **etkin konfig**.  
- **Çekirdeklerin açılışı (sırayla):**  
  1) Exchange/WS → abonelikler + *priming*,  
  2) Veri/özellik hattı hazır,  
  3) ML bileşenleri yüklenir + **health‑check**,  
  4) Stratejiler merkezî kayıt noktasına eklenir.  
- **Pre‑flight checklist** → bağlantı, veri tazeliği, hazır olma, yürütme korumaları.  
- **READY** → orkestratör başlatır.

### B. Çalışma (Runtime)
- **Ana döngü (merkez zamanlayıcı):** veri → özellik → strateji → ML/RL → (geçerse) yürütme.  
- **Paralel işler:** ML tahmin güncelleyicisi, yürütme/pozisyon izleme, heartbeat/watcher.  
- **Telemetri:** standartlaştırılmış log ve metrik akışı.

### C. Kapanış (Graceful Shutdown)
- **Yeni sinyal alımını kes**, ana döngüyü durdur.  
- **Açık pozisyonları güvenli kapat / doğrula**.  
- **WS ve exchange bağlantılarını** sırayla kapat.  
- **Artefakt/rapor** topla ve temiz çıkış yap.

> **Görsel:** [Lifecycle / Swimlane](./bearish_alpha_lifecycle.png)

---

## 4) Katmanlı & Merkezi Yönetim (Yönetişim Modeli)

### Merkezî ilkeler
- **Tek entegrasyon noktası:** Orkestratör; yatay kısa devre yok.  
- **Tek konfig kaynağı:** politika/parametreler merkezden; dağıtım deterministik.  
- **Sert sıraya uyum:** açılış ve kapanış **bağımlılık zinciri**ne göre.  
- **İzlenebilirlik:** her geçişte standart log; post‑session analiz kolay.  
- **Hata izolasyonu:** istisnalar katman sınırında yakalanır; karar (retry/skip/stop) merkezde verilir.

### Bileşenler arası kontrol akışı
- **Orkestratör →** WS‑Manager, Feature/ML, Stratejiler, Execution’ı **başlatır/durdurur**, **zamanlar** ve **denetler**.  
- **Stratejiler →** yalnızca **StrategyCoordinator**’a yazar; Execution’a **doğrudan** gitmez.  
- **AI‑Enhanced/RL →** strateji sinyalini **değerlendirir**; Execution’a geçiş için **izin/ret** verir.  
- **Execution →** pozisyon/emir yönetir; durum/olayları merkeze **geri bildirir**.

---

## 5) Depolama ve Kullanım

- Bu dosyayı **`/docs/architecture.md`** olarak ekleyin.  
- Görselleri aynı klasörde tutun:  
  - `./bearish_alpha_architecture.png`  
  - `./bearish_alpha_lifecycle.png`  
- README’den “Daha Fazla Bilgi → Architecture” bağlantısı vermeniz önerilir.

---

## 6) Değişiklik Geçmişi (Changelog)

- 2025-11-10: İlk mimarî döküm ve görseller eklendi.
