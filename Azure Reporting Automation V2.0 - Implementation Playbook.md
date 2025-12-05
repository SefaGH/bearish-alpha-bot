# 📘 Azure Reporting Automation V2.0 - Implementation Playbook

**Tarih:** 4 Aralık 2025  
**Durum:** 🚧 Implementation Phase (V1 Legacy -\> V2 Migration)  
**Hedef:** File-Based Raporlamadan Data-Driven (Log Analytics) Raporlamaya Geçiş  
**Doküman Tipi:** Teknik Uygulama ve Operasyon Kılavuzu

-----

## 🎯 Yönetici Özeti

Bu çalışma, mevcut metin tabanlı (`.log` dosyası regex analizi) raporlama sistemini, **Azure Monitor Log Analytics** tabanlı, sorgulanabilir, görselleştirilebilir ve hataya dayanıklı (robust) bir yapıya dönüştürür.

### V2.0 Temel Değişiklikler

  * **Veri Kaynağı:** Blob Storage (`.log` dosyaları) yerine **Log Analytics Workspace** (`BearishEvents_CL` tablosu).
  * **Tetikleme:** Asenkron Event Grid yerine, Logic App tarafından yönetilen **Senkron HTTP Çağrısı** (`run_report`).
  * **Analiz:** Regex yerine **KQL (Kusto Query Language)** ile kesin matematiksel analiz.
  * **Çıktı:** Basit `.txt` yerine, JSON veri yapısı ve (gelecekte) PDF/HTML görsel raporlar.
  * **Güvenlik & Dayanıklılık:** "Lazy Import" mimarisi ile sessiz çökmelerin (Silent Failure) önlenmesi ve veri yokluğunda (Day 0) sistemin çökmemesi (Graceful Handling).

-----

## 🏗️ Yeni Mimari (V2.0)

graph TD
    User[Tetikleyici: Zamanlayıcı/Manuel] -->|1. POST| LogicApp[Logic App: bearish-bot-orchestrator]
    
    subgraph "Orchestration Layer"
    LogicApp -->|2. Start Job| Runbook[Automation: Start-BearishBot-Enhanced]
    Runbook -->|3. Run Bot| VM[VM: BearishAlphaBot-VM-01]
    LogicApp -->|4. Wait for Completion| LogicApp
    end
    
    subgraph "Data Ingestion Layer"
    VM -->|5. Ingest Logs (Json)| AzureMonitor[Azure Monitor Agent / API]
    AzureMonitor -->|6. Store| LogAnalytics[(Log Analytics: bearish-logs)]
    LogAnalytics -->|Table: BearishEvents_CL| LogAnalytics
    end

    subgraph "Reporting Layer (V2)"
    LogicApp -->|7. HTTP POST (run_id)| FuncApp[Function: run_report]
    FuncApp -->|8. KQL Query| LogAnalytics
    LogAnalytics -->|9. Data Rows| FuncApp
    FuncApp -->|10. Generate PDF/JSON| FuncApp
    FuncApp -->|11. 200 OK + Report Body| LogicApp
    end

    LogicApp -->|12. Send Email (Rich Content)| SendGrid[SendGrid Email]

-----

## 🛠️ Faz 1: Altyapı ve Veri Hazırlığı

Bu aşamada botun veriyi Azure Monitor'e göndermesi ve Function App'in bu veriyi okuyabilmesi sağlanır.

### 1.1. Log Analytics Workspace Doğrulama

  * **Hedef:** Verilerin akacağı havuzun hazır olması.
  * **Komut:**
    ```powershell
    az monitor log-analytics workspace show --resource-group tradebot-ops --workspace-name bearish-logs --query customerId
    ```
  * **Beklenen:** Workspace ID (örn: `a6edc783-de73-4675-bafa-f2b1886c46bb`) dönmelidir.

### 1.2. Function App Ayarları

  * **Hedef:** Function App'in hangi workspace'i sorgulayacağını bilmesi.
  * **İşlem:** `LOG_ANALYTICS_WORKSPACE_ID` ayarının eklenmesi.
  * **Komut:**
    ```powershell
    az functionapp config appsettings set --name bearish-reporting-func-v2 --resource-group tradebot-ops --settings "LOG_ANALYTICS_WORKSPACE_ID=a6edc783-de73-4675-bafa-f2b1886c46bb"
    ```

### 1.3. Rol Atamaları (RBAC)

  * **Hedef:** Function App'in (`e221b20c...`) Log Analytics'ten veri okuyabilmesi.
  * **Rol:** `Log Analytics Reader`
  * **Komut:**
    ```powershell
    $funcPrincipalId = "e221b20c-0975-469c-b7ab-b5b282e2bb57"
    $workspaceId = "/subscriptions/74ab10ba-c96d-449e-97cb-ee4f9c0de714/resourceGroups/tradebot-ops/providers/Microsoft.OperationalInsights/workspaces/bearish-logs"

    az role assignment create --assignee $funcPrincipalId --role "Log Analytics Reader" --scope $workspaceId
    ```

-----

## 💻 Faz 2: Raporlama Fonksiyonu (`run_report`) Geliştirmesi

Bu bölüm, önceki "Sessiz 500" hatalarını önleyen "Lazy Import" mimarisini ve "Missing Table" (Eksik Tablo) yönetimini içerir.

### 2.1. Dosya Yapısı ve `requirements.txt`

Dosyaların şu şekilde yapılandırıldığından emin olun:

  * `function_app.py`: Sadece "Wrapper" (Kabuk) kodunu içerir. Hata yakalar.
  * `function_app_runtime.py`: Asıl iş mantığını (`pandas`, `playwright`, `azure-monitor-query`) içerir.
  * `requirements.txt`:
    ```text
    azure-functions
    azure-identity
    azure-monitor-query
    pandas
    playwright
    reportlab
    ```

### 2.2. Kod Implementasyonu: `function_app.py` (The Shield)

Bu dosya **asla** ağır kütüphaneleri (playwright vb.) en tepede import etmemelidir.

```python
import azure.functions as func
import logging
import traceback
import json

app = func.FunctionApp()

@app.route(route="run_report", auth_level=func.AuthLevel.FUNCTION)
def run_report(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Wrapper: run_report triggered.')
    try:
        # LAZY IMPORT: Kritik kütüphaneler burada yüklenir
        import function_app_runtime as runtime
        return runtime.run_report_logic(req)
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"CRITICAL FAILURE: {error_trace}")
        return func.HttpResponse(
            body=json.dumps({"status": "fatal_error", "traceback": error_trace}),
            status_code=500,
            mimetype="application/json"
        )
```

### 2.3. Kod Implementasyonu: `function_app_runtime.py` (The Logic)

Bu dosya `BearishEvents_CL` tablosu yoksa çökmemeli, boş liste dönmelidir.

  * **Kritik Düzeltme 1 (Timespan):** `query_workspace` çağrısında `timespan=timedelta(days=1)` parametresi **keyword argument** olarak verilmelidir.
  * **Kritik Düzeltme 2 (Graceful Handling):** `SemanticError` (Tablo yok) yakalanmalı ve `[]` (boş liste) dönülmelidir.

<!-- end list -->

```python
# (Özet Mantık)
try:
    result = client.query_workspace(
        workspace_id=WORKSPACE_ID,
        query=kql_query,
        timespan=timedelta(days=1) # Zorunlu keyword argüman
    )
except HttpResponseError as e:
    if "SemanticError" in str(e) and "BearishEvents_CL" in str(e):
        logging.warning("Tablo henüz yok (Day 0). Boş veri dönülüyor.")
        return [] # Çökme yok, boş rapor var.
    raise e
```

### 2.4. Deployment ve Remote Build

Linux ortamında C++ bağımlılıklarının derlenmesi için `remote build` şarttır.

  * **Komut:**
    ```bash
    func azure functionapp publish bearish-reporting-func-v2 --python --build remote --force
    ```

-----

## 🔄 Faz 3: Logic App Orkestrasyonu Güncellemesi

Eski sistemdeki "Dosya Yükle -\> Event Grid Bekle" yapısı kaldırılıp, "Fonksiyonu Çağır -\> Sonucu Al" yapısına geçilir.

### 3.1. Kaldırılacak/Pasifize Edilecek Adımlar

  * **Upload\_Raw\_Logs:** Bu adım kalabilir (yedekleme için), ancak artık raporu tetiklemeyecek.
  * **Event Grid Trigger:** Logic App içinde yoktur ama mimariden çıkarılabilir.

### 3.2. Eklenecek Adım: HTTP Call to `run_report`

Logic App Designer içinde `Upload_Raw_Logs` adımından sonraya ekleyin:

  * **Action:** HTTP
  * **Method:** POST
  * **URI:** `https://bearish-reporting-func-v2.azurewebsites.net/api/run_report?code=<FUNCTION_KEY>`
      * *Not: Function Key, Azure Portal -\> App Keys bölümünden alınır.*
  * **Headers:** `Content-Type: application/json`
  * **Body:**
    ```json
    {
      "run_id": "@{triggerBody()?['run_id']}",
      "mode": "production"
    }
    ```

### 3.3. E-Posta Adımının Güncellenmesi

E-posta gövdesine artık fonksiyonun çıktısı basılabilir.

  * **Action:** Send Email (SendGrid)
  * **Body:**
    ```text
    Bot Çalışması Tamamlandı.

    Rapor Özeti:
    @{body('HTTP_-_run_report')}
    ```

-----

## ✅ Faz 4: Doğrulama ve Test Prosedürleri (Verification)

Sistemi devreye almadan önce aşağıdaki testler sırasıyla yapılmalıdır.

### Test 1: Smoke Test (Tablo Yokken)

Bot henüz çalışmadıysa veya tablo silindiyse fonksiyon **200 OK** dönmelidir.

  * **Komut:**
    ```powershell
    Invoke-WebRequest -Method Post -Uri "https://bearish-reporting-func-v2.../api/run_report?code=..." -Body '{"run_id":"test"}'
    ```
  * **Başarı Kriteri:** HTTP 200 ve Body: `[]` veya `{"message": "No data found"}`. **ASLA 500 OLMAMALI.**

### Test 2: Ingestion Testi (Veri Girişi)

Log Analytics'e manuel sahte veri göndererek tablonun oluşmasını tetikleyin (Opsiyonel ama önerilir).

  * *Bu işlem için Data Collection Rule veya basit bir Python script kullanılabilir.*

### Test 3: Full End-to-End Test

Logic App'i manuel tetikleyin.

1.  Bot çalışır.
2.  Loglar Azure Monitor'e akar.
3.  Logic App `run_report`'u çağırır.
4.  Fonksiyon KQL ile veriyi çeker, hesaplar.
5.  E-posta kutunuza JSON veya HTML formatında detaylı rapor düşer.

-----

## 🐛 Sorun Giderme (Troubleshooting Guide)

Semptom,Olası Neden,Çözüm
HTTP 500 (Silent),import playwright hatası (lib eksik).,function_app.py içindeki Lazy Import yapısını kontrol et. Hata JSON dönmeli.
HTTP 500 (JSON),Missing keyword argument 'timespan'.,function_app_runtime.py içinde query_workspace çağrısına timespan=... ekle.
HTTP 500 (Semantic),Failed to resolve table.,Graceful Handling kodu çalışmıyor. try-except HttpResponseError bloğunu kontrol et.
HTTP 404,Function Indexing hatası.,"Kodda syntax hatası var. Portal -> ""Diagnose and Solve"" -> ""Function Load Errors""a bak."
Veri Gelmiyor,Bot logları Azure'a atamıyor.,VM üzerindeki Azure Monitor Agent (AMA) veya Data Collection Rule (DCR) ayarlarını kontrol et.

-----

## 🏁 Sonuç

Bu playbook uygulandığında, **Bearish Alpha Bot** raporlama sistemi;

1.  **Sessiz hatalardan arındırılmış**,
2.  **Veri tabanı gücünü kullanan**,
3.  **Başlangıç anında (Day 0) çökme yapmayan**,
4.  **Logic App ile tam senkronize çalışan**
    modern bir yapıya kavuşacaktır.