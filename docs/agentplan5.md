# **PROJE YÖNERGESİ: Algoritmik Ticaret Botu için ML Tabanlı "Risk Gatekeeper" Entegrasyonu**

### **1. Proje Özeti ve Hedef**

Mevcut çalışan bir algoritmik ticaret botumuz (Python, Azure VM, Docker) bulunmaktadır. Bu bot şu anda kural tabanlı (Mean Reversion vb.) sinyaller üretmektedir. Hedefimiz, bu botun "karar mekanizmasını" değiştirmeden, sinyalleri filtreleyen bir **"Risk Gatekeeper" (Meta-Controller)** katmanı eklemektir.

Bu katman, **Meta-Labeling** (Marcos Lopez de Prado yaklaşımı) yöntemini kullanarak, botun ürettiği sinyallerin başarı olasılığını tahmin eden bir **Machine Learning (XGBoost/LightGBM)** modeli olacaktır.

**Temel Hedef:**

* Botun "Signal Generator" olarak kalması.
* ML modelinin "Gatekeeper" (Kapı Bekçisi) olarak devreye girip düşük olasılıklı işlemleri (False Positives) engellemesi.
* Sistemin **Azure Machine Learning** ve **Docker** üzerinde mikro servis mimarisiyle çalışması.

---

### **2. Mevcut Altyapı ve Kısıtlar**

* **Platform:** Azure Virtual Machine (Ubuntu)
* **Deployment:** Docker Compose
* **Dil:** Python (Bot ve yeni servis için)
* **Veri:** Botun logları (`TRADE_CLOSED`, `RISK-REBASE` vb. JSON formatında)

---

### **3. Teknik Mimari ve Uygulama Planı**

Agent olarak senden aşağıdaki 3 ana bileşeni tasarlamanı ve kodlamanı bekliyorum:

#### **A. Veri Boru Hattı (Data Pipeline & Logging)**

Mevcut botun loglama mekanizmasını zenginleştirerek eğitim verisi (Training Dataset) oluşturmalıyız.

* **Görev:** Botun kaynak kodunda `TRADE_CLOSED` logunun içeriğini güncelle.
* **Eklenecek Alanlar:**
* `primary_signal_features`: Botun sinyal üretirken kullandığı indikatörler (RSI, Bollinger Band %B, vb.).
* `market_state`: Volatilite (`vol_atr_bps`), Spread, Funding Rate.
* `risk_parameters`: `tp_ratio` (Hedef Kâr Oranı), `final_stop_ratio` (Stop Loss Oranı), `rr_required`.
* `regime_fingerprint`: O anki piyasa rejimi (Trend/Range).


* **Çıktı:** Zenginleştirilmiş JSONL formatında log dosyası.

#### **B. Model Eğitimi (Training Pipeline - Azure ML)**

Loglardan öğrenen bir ML modeli geliştirilecek.

* **Algoritma:** XGBoost veya LightGBM (Tabular veri için).
* **Etiketleme (Labeling) Stratejisi:**
* **Triple Barrier Method:** Her işlem için dinamik etiketleme yapılacak.
* `Label 1 (Başarılı)`: Eğer işlem `Take Profit` ile kapandıysa VEYA `PnL > Hedeflenen TP` ise.
* `Label 0 (Başarısız)`: Eğer işlem `Stop Loss` ile kapandıysa VEYA `PnL <= 0` ise.


* **Validasyon:** Standart `K-Fold` **kullanılmayacak**. Finansal veri sızıntısını (leakage) önlemek için **Purged K-Fold** veya **Embargo TimeSeriesSplit** kullanılacak.
* **Azure Entegrasyonu:** Eğitim scripti Azure ML Studio üzerinde çalıştırılabilir şekilde tasarlanmalı.

#### **C. Gatekeeper Servisi (Inference API - Sidecar Pattern)**

Eğitilen modeli canlıda çalıştıracak mikro servis.

* **Teknoloji:** FastAPI + ONNX Runtime (Hız için).
* **Mimari:** Mevcut `docker-compose.yml` dosyasına `risk-gatekeeper` adında yeni bir servis olarak eklenecek.
* **İletişim:**
1. Bot, sinyal ürettiğinde bu servise `POST /evaluate` isteği atacak (Feature JSON ile).
2. Servis, modeli çalıştırıp bir olasılık skoru (`probability`) ve karar (`trade_permission: True/False`) dönecek.
3. Eğer `probability > 0.65` (eşik değer) ise bot işlem açacak, aksi halde reddedecek.

---

### **4. Teslimat Gereksinimleri**

Lütfen aşağıdaki dosyaları ve kod bloklarını hazırla:

1. **`logger_patch.py`**: Botun loglama fonksiyonuna eklenecek kod parçası.
2. **`train_model.py`**: Log dosyasını okuyan, Triple Barrier etiketlemesi yapan, Purged CV ile XGBoost eğiten ve modeli `.onnx` veya `.json` olarak kaydeden Python scripti.
3. **`gatekeeper_service/`**: FastAPI uygulaması (main.py, Dockerfile).
4. **`docker-compose.override.yml`**: Yeni servisi sisteme ekleyen konfigürasyon.

**Önemli Not:** Kodları yazarken "Fail-Safe" prensibine dikkat et. Eğer Gatekeeper servisi çökerse veya zaman aşımına (timeout) uğrarsa, bot "Güvenli Mod"da (işlem açmadan veya varsayılan kurallarla) devam etmeli, sistem kilitlenmemeli.

#### 1. Veri Boru Hattı (Data Pipeline) Değişikliği

Önceki planda "ayrı bir CSV logger yap" demiştim. Analiz diyor ki: *"Gerek yok, senin `TRADE_CLOSED` logun zaten bir hazine sandığı."*

**Yapılacak:** Loglama mekanizmanı sadeleştir.

* **Eski Plan:** `Bot -> Gatekeeper -> Trade -> CSV`
* **Yeni Plan:** `Bot -> Trade -> Log Dosyası (JSONL) -> Python Script (Parser) -> Eğitim`

Botun kodunda şu küçük değişikliği yap (**Madde 5'teki öneri**):
`TRADE_CLOSED` olayını loglarken, `RISK-REBASE` logundaki şu değerleri de içine göm:

* `tp_ratio`
* `final_stop_ratio`
* `floor_selected`
* `rsi_at_entry` (Kesinlikle null olmamalı!)
* `regime_at_entry`

#### 2. Etiketleme (Labeling) Mantığı: "Seçenek A"

Python ile yazacağın `labeling.py` scriptini şu mantığa göre güncelle. Artık dışarıdan bir `sl` veya `tp` parametresi almayacak, her satırın (işlemin) kendi hedefini kullanacak.

```python
def get_labels_from_logs(df_logs):
    """
    df_logs: TRADE_CLOSED loglarından parse edilmiş DataFrame
    """
    labels = []
    
    for index, row in df_logs.iterrows():
        # Gerçekleşen Sonuç (Logdan geliyor)
        exit_reason = row['exit_reason'] # 'stop_loss', 'take_profit', 'time_limit'
        pnl = row['pnl_pct']
        
        # Meta-Labeling Mantığı:
        # Eğer işlem Take Profit ile kapandıysa veya PnL > Hedeflenen Kâr ise = 1
        # Eğer işlem Stop Loss olduysa veya PnL < 0 ise = 0
        
        target_tp = row['tp_ratio'] 
        
        if exit_reason == 'take_profit' or pnl >= target_tp:
            label = 1
        elif exit_reason == 'stop_loss' or pnl <= 0: # Breakeven altı başarısız sayılır
            label = 0
        else:
            # Zaman doldu (Time Limit) veya manuel kapatıldı
            # Eğer kârdayken kapandıysa 1, zarardaysa 0 diyebiliriz
            # Veya Triple Barrier mantığıyla "dikey bariyer" nötr (0) olabilir
            label = 1 if pnl > 0 else 0 
            
        labels.append(label)
        
    return pd.Series(labels, index=df_logs.index)

```

#### 3. Özellik Seti (Feature Set) Haritası

Modelin eğitimi için kullanılacak input listesi (X) artık netleşti. `TRADE_CLOSED` logundaki şu alanları modeline besle:

| Grup | Alan Adı (JSON Key) | Neden Önemli? |
| --- | --- | --- |
| **Sinyal** | `side` | Long/Short ayrımı (Model yönü bilmeli) |
| **Kalite** | `quality_score` | Botun kendi güveni |
| **Kalite** | `ml_price_score_normalized` | ML sinyal gücü |
| **Kalite** | `volume_strength_at_entry` | Hacim onayı var mı? |
| **Volatilite** | `vol_atr_bps` | Piyasa ne kadar hareketli? |
| **Volatilite** | `ml_uncertainty` | Model ne kadar kararsız? |
| **Risk** | `rr_required` | Hedeflenen Risk/Ödül oranı |
| **Risk** | `entry_slippage_bps` | Kayma (Maliyet) |
| **Market** | `rsi_at_entry` | Aşırı alım/satım durumu |
| **Market** | `regime_at_entry` | Trend/Range durumu |

#### 4. Validasyon: Purged K-Fold (Zorunlu)

Analizdeki **"Purged/Embargo CV"** uyarısı hayati. Finansal veride zaman serisi "sızması" (leakage) olur.
Standart `train_test_split` yerine, **`sklearn.model_selection.TimeSeriesSplit`**'in gelişmiş versiyonunu kullanmalısın.

```python
from sklearn.model_selection import TimeSeriesSplit

# Basit Embargo Mantığı:
# Test seti ile Train seti arasında boşluk bırak
gap = 50 # İşlem sayısı kadar boşluk
tscv = TimeSeriesSplit(n_splits=5, gap=gap) 

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Modeli eğit...

```

### Sonuç: Eylem Planı (Checklist)

1. **[ ] Log Zenginleştirme:** Botun kodunu aç, `TRADE_CLOSED` olayına eksik alanları (`rsi`, `regime`, `tp_ratio`) ekle. Deploy et.
2. **[ ] Veri Toplama:** Botu çalıştır. En az 50-100 işlem (trade) olana kadar bekle. (Mean Reversion stratejisi için bu birkaç gün sürebilir).
3. **[ ] Parser Script:** Log dosyasını okuyup yukarıdaki Feature Tablosuna çeviren bir Python scripti yaz.
4. **[ ] İlk Eğitim (Local):** Scripti çalıştır, `XGBoost` ile basit bir model eğit. Feature Importance'a bak (Hangi özellik en önemli?).
5. **[ ] Gatekeeper Servisi:** Modeli Docker içine koy, botun önüne bağla.

Ek öneriler:

Azure ML ile model eğitimi ve deployment süreçlerini otomatikleştirmek için Azure DevOps veya GitHub Actions CI/CD entegrasyonu kullanabilirsiniz.
Modeli ONNX olarak export edip FastAPI servisine entegre etmek, inference performansını artırır.
Azure VM yerine Azure Container Apps veya Azure Kubernetes Service (AKS) gibi daha yönetilebilir platformlar da uzun vadede değerlendirilebilir.
Detaylı Azure ML entegrasyonu ve FastAPI örnekleri için:

Azure Machine Learning ile model eğitimi
FastAPI on Azure Functions
Azure ML Python SDK

Azure tarafında bunu “bot çalışınca otomatik devreye giren Gatekeeper” şeklinde kurmanın en pratik yolu: Gatekeeper’ı aynı VM’de, botla aynı Docker Compose stack’i içinde bir sidecar servis olarak ayağa kaldırmak ve botun sinyal anında bu servise POST /evaluate ile danışması.

Aşağıdaki akış, agentplan5.md planınızla birebir uyumlu ve Logic App ile zamanlanmış çalıştırmaya da temiz oturur.

1) VM + Docker Compose mimarisi (önerilen)

risk-gatekeeper ayrı bir container (FastAPI + ONNX Runtime)
Bot container’ı http://risk-gatekeeper:8000/evaluate endpoint’ine istek atar (aynı docker network içinde)
Fail-safe: Gatekeeper timeout/çökme durumunda bot bloklanmaz (trade açmaz veya default kurala döner)
2) Logic App neyi tetikleyecek?
Logic App’in yaptığı “başlat” aksiyonunu şu hale getirin:

VM üzerinde docker compose up komutu iki servisi birden başlatsın:
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --remove-orphans
İsterseniz job bitince durdurma için:
docker compose -f ... down (veya sadece botu durdur)
Böylece “bot ne zaman çalışıyorsa Gatekeeper da o zaman çalışır” kuralını otomatik sağlamış olursunuz.

3) docker-compose.override.yml (kritik parçalar)
Oluşturacağınız dosyada (planınızdaki 4. teslimat): docker-compose.override.yml

risk-gatekeeper servisini ekleyin
Bot servisine şu env’leri verin:
GATEKEEPER_URL=http://risk-gatekeeper:8000/evaluate
GATEKEEPER_TIMEOUT_MS=250 (örnek; küçük tutun)
GATEKEEPER_FAIL_SAFE_MODE=deny (önerim: “deny” = trade açma)
depends_on + healthcheck kullanın (Gatekeeper hazır olmadan bot başlamasın)
4) Bot tarafındaki entegrasyon (fail-safe şart)
Bot sinyal üretirken:

POST /evaluate çağrısı kısa timeout ile yapılmalı
Hata/timeout olursa:
“deny” modunda: trade_permission=False (en güvenlisi)
“fallback” modunda: mevcut kural tabanlı filtre ile devam (daha agresif)
Bu çağrı için en önemli 3 nokta:

Timeout küçük (örn 150–400ms)
Retries minimum (0–1)
Exceptions kesin yakalanıp “safe mode”a düşülmeli
5) Model artefact’ı (ONNX) VM’de nasıl yönetilecek?
En basit/sağlam yöntem:

VM host’ta kalıcı bir dizin: örn. /mnt/bearish/models/risk-gatekeeper/model.onnx
Compose ile Gatekeeper container’ına volume mount:
container içinde /app/models/model.onnx
Azure ML eğitimi bittiğinde yeni ONNX’i bu dizine indirip sadece Gatekeeper container’ını restart edin
Alternatif (daha enterprise): artefact’ı Blob Storage’a koyup container startup’ında indir (Managed Identity/SAS ile). Ama ilk MVP için host-volume en hızlısı.

6) Log zenginleştirme ve veri toplama
Planınızdaki “TRADE_CLOSED log zenginleştirme” (tp_ratio, final_stop_ratio, floor_selected, rsi_at_entry, regime_at_entry) kesinlikle ilk adım olmalı; aksi halde modeliniz “gerçek hedef” bilgisinden kopuk kalır.

Gatekeeper ayrıca kendi karar logunu da yazsın (örn GATEKEEPER_DECISION): probability, threshold, trade_permission, trade_id/correlation_id
7) AI Toolkit Agent Inspector bu resimde nereye oturuyor?

AI Toolkit + Inspector ile eklediğimiz dev agent (agent_entrypoint.py) prod gatekeeper’ın yerine geçmek için değil.
Ama şu iş için çok faydalı: “feature JSON geldi → karar döndü → bot doğru davrandı mı?” akışını daha hızlı debug etmek.
Önerilen uygulama sırası (Azure’da sorunsuz ilerlemek için)

(A) Log patch → 50–100 trade veri topla
(B) Local train_model.py ile ilk modeli çıkar → ONNX export
(C) Gatekeeper FastAPI container + compose override → VM’de çalıştır
(D) Bot entegrasyonu + fail-safe → Logic App start/stop komutlarını compose’a bağla
(E) Sonra Azure ML pipeline ile eğitimi otomatikleştir