# 🚀 Bearish Alpha Bot - Email Notification System Migration Playbook

## 📊 Executive Summary

**Tarih:** 5 Aralık 2024  
**Proje:** Email Bildirim Sistemini Child'dan Parent Logic App'e Taşıma  
**Durum:** %60 Tamamlandı  
**Kritik Bulgu:** SendGrid API key'i yanlış girilmişti → Düzeltildi ✅

---

## 🎯 Ne Yapıyoruz?

### Problem
`bearish-report-orchestrator` (Child Logic App) SendGrid email gönderirken **401 Unauthorized** hatası alıyordu.

### Kök Neden
1. ❌ **Yanlış SendGrid API Key** kullanılıyordu
2. ⚠️ **Child Workflow'da Key Vault Reference** sorunlu olabilir (Microsoft docs önermiyor)

### Çözüm
✅ **Email gönderme işlemini Parent Logic App'e taşıyoruz**  
✅ **Doğru SendGrid API Key bulundu ve test edildi**  
✅ **SendGrid Managed Connector kullanılıyor** (HTTP action yerine)

---

## 🏗️ Mevcut Mimari (Şu An)

```
┌─────────────────────────────────────────────────────────────────┐
│ bearish-bot-orchestrator (PARENT)                              │
│ Resource Group: TradeBot                                        │
│ Subscription: 74ab10ba-c96d-449e-97cb-ee4f9c0de714             │
├─────────────────────────────────────────────────────────────────┤
│ Trigger: Recurrence (Her gün 08:00 UTC)                        │
│ ↓                                                                │
│ Action: Start_BearishAlphaBot_VM                                │
│ ↓                                                                │
│ Action: Upload_Raw_Logs_Backup (loguploader function)          │
│ ↓                                                                │
│ Action: Trigger_Report_Orchestrator_V2                          │
│         (bearish-report-orchestrator'ı çağırıyor)               │
│ ↓                                                                │
│ ❌ Email gönderme YOK (buraya eklenecek)                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ bearish-report-orchestrator (CHILD)                            │
│ Resource Group: tradebot-ops                                    │
│ Subscription: 74ab10ba-c96d-449e-97cb-ee4f9c0de714             │
├─────────────────────────────────────────────────────────────────┤
│ Trigger: Manual/HTTP (Parent'tan çağrılıyor)                   │
│ ↓                                                                │
│ Action: HTTP_-_run_report (Function çağırıyor)                 │
│         URL: https://bearish-reporting-func-v2.                 │
│              azurewebsites.net/api/run_report                   │
│ ↓                                                                │
│ Action: Parse_JSON (Function response'u parse eder)            │
│ ↓                                                                │
│ Condition: events var mı?                                       │
│ ├─ TRUE: Send_Report_Email_SendGrid ❌ (401 hatası)           │
│ └─ FALSE: Send_No_Data_Email_SendGrid ❌ (401 hatası)         │
│                                                                  │
│ ⚠️ Bu email action'ları kaldırılacak                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Hedef Mimari (Tamamlandığında)

```
┌─────────────────────────────────────────────────────────────────┐
│ bearish-bot-orchestrator (PARENT)                              │
│ Resource Group: TradeBot                                        │
├─────────────────────────────────────────────────────────────────┤
│ Trigger: Recurrence (Her gün 08:00 UTC)                        │
│ ↓                                                                │
│ Action: Start_BearishAlphaBot_VM                                │
│ ↓                                                                │
│ Action: Upload_Raw_Logs_Backup                                  │
│ ↓                                                                │
│ Action: Call_Report_Generator                                   │
│         (bearish-report-orchestrator'ı çağırır)                 │
│         Type: Logic Apps - Choose a workflow                    │
│         Workflow: bearish-report-orchestrator                   │
│ ↓                                                                │
│ Condition: Check_Report_Status                                  │
│ │                                                                │
│ ├─ IF statusCode = 200 (Rapor var)                             │
│ │  ↓                                                             │
│ │  Action: Send_Success_Email                                   │
│ │          Type: SendGrid - E-posta gönder (V4)                │
│ │          From: noreply@bearishbot.com                         │
│ │          To: sefaasar@hotmail.com                             │
│ │          Subject: Bearish Bot - Günlük Rapor Hazır           │
│ │          Body: Rapor linki (dinamik)                          │
│ │                                                                │
│ └─ ELSE                                                          │
│    ↓                                                             │
│    Condition: Check_If_No_Data                                  │
│    │                                                             │
│    ├─ IF statusCode = 204 (İşlem yok)                          │
│    │  ↓                                                          │
│    │  Action: Send_No_Data_Email                                │
│    │          Type: SendGrid - E-posta gönder (V4)             │
│    │          From: noreply@bearishbot.com                      │
│    │          To: sefaasar@hotmail.com                          │
│    │          Subject: Bearish Bot - İşlem Yok                  │
│    │          Body: Sistem mesajı (dinamik)                     │
│    │                                                             │
│    └─ ELSE (Hata durumu)                                        │
│       ↓                                                          │
│       Action: Send_Error_Email (Opsiyonel)                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ bearish-report-orchestrator (CHILD)                            │
│ Resource Group: tradebot-ops                                    │
├─────────────────────────────────────────────────────────────────┤
│ Trigger: Manual/HTTP                                            │
│ ↓                                                                │
│ Action: HTTP_-_run_report                                       │
│         URL: https://bearish-reporting-func-v2.                 │
│              azurewebsites.net/api/run_report                   │
│ ↓                                                                │
│ Action: Parse_JSON                                              │
│ ↓                                                                │
│ Condition: events var mı?                                       │
│ ├─ TRUE: Response (200) + report_url                           │
│ └─ FALSE: Response (204) + message                             │
│                                                                  │
│ ✅ Email action'ları kaldırıldı (Parent'ta)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Kalan İşlemler (Step-by-Step)

### ✅ Tamamlanan İşlemler
1. ✅ Sorun tespiti (401 Unauthorized)
2. ✅ Doğru SendGrid API Key bulundu
3. ✅ SendGrid connection parent'ta test edildi

### 🔄 Devam Eden İşlemler

#### **ADIM 1: Parent Logic App'e Child Workflow Çağrısı Ekle**

**Portal:**
1. Azure Portal → Resource Groups → **TradeBot**
2. **bearish-bot-orchestrator** → Logic app designer
3. Son action'dan sonra (**Upload_Raw_Logs_Backup**) → **+ New step**
4. Ara: `logic apps`
5. Seç: **"Choose a Logic Apps workflow"**
6. Form doldur:
   - **Subscription:** `74ab10ba-c96d-449e-97cb-ee4f9c0de714`
   - **Resource Group:** `tradebot-ops`
   - **Workflow:** `bearish-report-orchestrator`
7. Action adını değiştir: **`Call_Report_Generator`**
8. **Save**

**Doğrulama:**
- Designer'da `Call_Report_Generator` action'ı görünmeli
- Üzerine tıkla → Ayarlar: `bearish-report-orchestrator` seçili olmalı

---

#### **ADIM 2: Condition (Rapor Durumu Kontrolü) Ekle**

**Portal:**
1. `Call_Report_Generator` action'ından sonra → **+ New step**
2. Ara: `condition`
3. Seç: **Condition** (Control)
4. Action adını değiştir: **`Check_Report_Status`**

**Condition Ayarları:**
- Sol taraf: **Expression** kullan
  ```
  outputs('Call_Report_Generator')['statusCode']
  ```
- Operator: `is equal to`
- Sağ taraf: `200`

5. **Save**

**Doğrulama:**
- Condition iki branch oluşturmalı: **True** ve **False**

---

#### **ADIM 3: Success Email (200) Ekle**

**Portal - True Branch:**
1. **True** branch'ine tıkla → **Add an action**
2. Ara: `sendgrid`
3. Seç: **E-posta gönder (V4)**
4. Connection zaten var (önceki testte oluşturuldu): **sendgrid-connection**
5. Form doldur:

**Form Değerleri:**
```yaml
Kimden: noreply@bearishbot.com
Kime: sefaasar@hotmail.com
Konu: Bearish Bot - Günlük Rapor Hazır
E-posta gövdesi:
  Merhaba,
  
  Bot çalışması başarıyla tamamlandı ve rapor oluşturuldu.
  
  Rapor Linki: @{body('Call_Report_Generator')?['report_url']}
  
  Run ID: @{body('Call_Report_Generator')?['run_id']}
  
  Detaylı analiz için linke tıklayınız.
```

**Dinamik Content Ekleme:**
- "Rapor Linki:" yazdıktan sonra **`@{`** yaz
- Otomatik tamamlama gelecek: `body('Call_Report_Generator')?['report_url']` seç
- Veya direkt yazabilirsin: `@{body('Call_Report_Generator')?['report_url']}`

6. Action adını değiştir: **`Send_Success_Email`**
7. **Save**

**Doğrulama:**
- Body'de `@{body('Call_Report_Generator')?['report_url']}` görünmeli (mavi renkte)

---

#### **ADIM 4: No Data Condition Ekle**

**Portal - False Branch:**
1. **False** branch'ine tıkla → **Add an action**
2. Ara: `condition`
3. Seç: **Condition**
4. Action adını değiştir: **`Check_If_No_Data`**

**Condition Ayarları:**
- Sol taraf: **Expression**
  ```
  outputs('Call_Report_Generator')['statusCode']
  ```
- Operator: `is equal to`
- Sağ taraf: `204`

5. **Save**

---

#### **ADIM 5: No Data Email (204) Ekle**

**Portal - Check_If_No_Data → True Branch:**
1. **True** branch'ine tıkla → **Add an action**
2. Ara: `sendgrid`
3. Seç: **E-posta gönder (V4)**
4. Connection: **sendgrid-connection** (aynı)

**Form Değerleri:**
```yaml
Kimden: noreply@bearishbot.com
Kime: sefaasar@hotmail.com
Konu: Bearish Bot - İşlem Yok
E-posta gövdesi:
  Merhaba,
  
  Bot taraması tamamlandı ancak raporlanacak işlem verisi bulunamadı.
  
  Sistem Mesajı: @{body('Call_Report_Generator')?['message']}
  
  Run ID: @{body('Call_Report_Generator')?['run_id']}
  
  Tarih: @{utcNow()}
```

6. Action adını değiştir: **`Send_No_Data_Email`**
7. **Save**

**Doğrulama:**
- Body'de dinamik content'ler mavi renkte görünmeli

---

#### **ADIM 6: Child Logic App'ten Email Action'larını Kaldır**

**Portal:**
1. Azure Portal → Resource Groups → **tradebot-ops**
2. **bearish-report-orchestrator** → Logic app designer
3. **Code view** sekmesine geç
4. `Send_Report_Email_SendGrid` ve `Send_No_Data_Email_SendGrid` action'larını bul
5. Bu action'ları **SİL** (JSON'dan tamamen kaldır)
6. **Save**

**Alternatif - Designer'dan:**
1. Her email action'ına tıkla
2. Sağ üst **⋮** (üç nokta) → **Delete**
3. **Save**

**Doğrulama:**
- Child workflow'da sadece `HTTP_-_run_report`, `Parse_JSON`, `Condition` kalmalı
- Email action'ları tamamen gitmiş olmalı

---

#### **ADIM 7: Child'ın Response Action'larını Düzenle**

**Portal - Child Workflow:**
1. **Code view** → `Condition` action'ını bul
2. **True** branch'te Response action'ı olmalı:

```json
{
  "Response_Success": {
    "type": "Response",
    "inputs": {
      "statusCode": 200,
      "body": {
        "status": "success",
        "run_id": "@{body('Parse_JSON')?['run_id']}",
        "report_url": "@{body('Parse_JSON')?['report_url']}",
        "events_count": "@{body('Parse_JSON')?['events_count']}"
      }
    },
    "runAfter": {
      "Parse_JSON": ["Succeeded"]
    }
  }
}
```

3. **False** branch'te Response action'ı:

```json
{
  "Response_No_Data": {
    "type": "Response",
    "inputs": {
      "statusCode": 204,
      "body": {
        "status": "no_data",
        "run_id": "@{body('Parse_JSON')?['run_id']}",
        "message": "@{body('Parse_JSON')?['message']}"
      }
    },
    "runAfter": {
      "Parse_JSON": ["Succeeded"]
    }
  }
}
```

4. **Save**

**Doğrulama:**
- Condition → True: Response (200)
- Condition → False: Response (204)
- Email action'ları yok

---

## 🧪 Test Prosedürleri

### **TEST 1: Parent Logic App Manuel Test**

**Amaç:** Email gönderiminin çalışıp çalışmadığını test et

**Adımlar:**
1. Portal → **bearish-bot-orchestrator** → **Overview**
2. **Run Trigger** butonu → **Run**
3. **Runs history**'den en son run'ı aç
4. Her action'ı kontrol et:
   - ✅ `Start_BearishAlphaBot_VM`: Succeeded
   - ✅ `Upload_Raw_Logs_Backup`: Succeeded
   - ✅ `Call_Report_Generator`: Succeeded (statusCode: 200 veya 204)
   - ✅ `Check_Report_Status`: Succeeded
   - ✅ `Send_Success_Email` veya `Send_No_Data_Email`: Succeeded

**Başarı Kriterleri:**
- ✅ Email `sefaasar@hotmail.com`'a ulaştı
- ✅ Email içeriği doğru (rapor linki veya "işlem yok" mesajı)
- ✅ Hiçbir action failed olmadı

**Hata Durumunda:**
- `Send_Success_Email` veya `Send_No_Data_Email` action'ına tıkla
- **Outputs** sekmesine bak
- **Error details** varsa kopyala

---

### **TEST 2: Child Logic App Sadece Response Döndüğünü Kontrol Et**

**Amaç:** Child'ın email göndermediğini, sadece response döndüğünü doğrula

**Adımlar:**
1. Portal → **bearish-report-orchestrator** → **Overview**
2. Son run'ı aç (Parent test sırasında çağrılmış olmalı)
3. Action'ları kontrol et:
   - ✅ `HTTP_-_run_report`: Succeeded
   - ✅ `Parse_JSON`: Succeeded
   - ✅ `Condition`: Succeeded
   - ✅ `Response_Success` veya `Response_No_Data`: Succeeded
   - ❌ **Email action'ları görünmemeli**

**Başarı Kriterleri:**
- ✅ Response (200 veya 204) döndü
- ✅ Parent bu response'u aldı
- ✅ Email action'ları yok

---

### **TEST 3: End-to-End Otomatik Test (Scheduled Trigger)**

**Amaç:** Ertesi gün 08:00'de otomatik çalışmayı test et

**Adımlar:**
1. Ertesi gün saat 08:00 UTC'yi bekle
2. Saat 08:05'te Portal → **bearish-bot-orchestrator** → **Runs history**
3. En son run'ı kontrol et
4. Email geldimi kontrol et (`sefaasar@hotmail.com`)

**Başarı Kriterleri:**
- ✅ Logic App otomatik tetiklendi (Trigger: Recurrence)
- ✅ VM başladı
- ✅ Log upload oldu
- ✅ Rapor oluştu
- ✅ Email gönderildi

---

## 🚨 Hata Senaryoları ve Çözümleri

### **Senaryo 1: SendGrid 401 Unauthorized**

**Semptomlar:**
```json
{
  "statusCode": 401,
  "body": {
    "errors": [
      {
        "message": "The provided authorization grant is invalid, expired, or revoked"
      }
    ]
  }
}
```

**Kök Neden:**
- SendGrid API key geçersiz veya süresi dolmuş

**Çözüm:**
1. SendGrid Portal → Settings → API Keys
2. Mevcut key'i **Revoke** et
3. **Create API Key** → **Full Access** seç
4. Key'i kopyala
5. Portal → **bearish-bot-orchestrator** → **API Connections**
6. **sendgrid-connection** → **Edit API connection**
7. **API Key** field'ine yeni key'i yapıştır
8. **Save**
9. Logic App'i tekrar test et

---

### **Senaryo 2: Dynamic Content Çözülmüyor**

**Semptomlar:**
- Email body'de `@{body('Call_Report_Generator')?['report_url']}` string olarak görünüyor

**Kök Neden:**
- Expression yanlış yazılmış veya action adı yanlış

**Çözüm:**
1. Designer'da email action'ına tıkla
2. Body field'ini temizle
3. Şunu yaz (tam olarak):
   ```
   Rapor Linki: @{body('Call_Report_Generator')?['report_url']}
   ```
4. `@{` yazdığında otomatik tamamlama gelirse seç
5. **Save**
6. Test et

---

### **Senaryo 3: Parent Child'ı Çağıramıyor**

**Semptomlar:**
```
The workflow 'bearish-report-orchestrator' could not be found
```

**Kök Neden:**
- Yanlış resource group veya subscription seçilmiş

**Çözüm:**
1. `Call_Report_Generator` action'ını sil
2. Yeniden ekle:
   - **Subscription:** `74ab10ba-c96d-449e-97cb-ee4f9c0de714`
   - **Resource Group:** `tradebot-ops`
   - **Workflow:** `bearish-report-orchestrator`
3. **Save**

---

### **Senaryo 4: Email Gönderilmiyor Ama Hata Yok**

**Semptomlar:**
- Logic App başarılı (Succeeded)
- Email gelmiyor

**Kök Neden:**
- SendGrid'de sender email doğrulanmamış
- Spam filtresi

**Çözüm:**
1. SendGrid Portal → Settings → Sender Authentication
2. **Verify Single Sender** → `noreply@bearishbot.com` doğrula
3. `sefaasar@hotmail.com` spam klasörünü kontrol et
4. Logic App'i tekrar test et

---

## 📊 Monitoring ve Diagnostics

### **Azure Portal - Logic App Monitoring**

**Runs History Kontrolü:**
1. Portal → **bearish-bot-orchestrator** → **Overview**
2. **Runs history** grafiğine bak:
   - Yeşil: Succeeded
   - Kırmızı: Failed
   - Turuncu: Cancelled
3. Her run'a tıklayarak detaylı flow görebilirsin

**Metrics:**
1. Portal → **bearish-bot-orchestrator** → **Metrics**
2. Metric seç:
   - **Runs Started**: Toplam çalıştırma sayısı
   - **Runs Succeeded**: Başarılı çalıştırmalar
   - **Runs Failed**: Başarısız çalıştırmalar
   - **Run Latency**: Ortalama süre
3. Time range: Last 7 days

**Alerts (Opsiyonel):**
1. **Alerts** → **Create alert rule**
2. Condition: `Runs Failed greater than 0`
3. Action: Email notification (`sefaasar@hotmail.com`)
4. **Create**

---

### **Azure Log Analytics - Workflow Logs**

**Query - Son 24 Saatin Run'ları:**
```kusto
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.LOGIC"
| where resource_workflowName_s in ("bearish-bot-orchestrator", "bearish-report-orchestrator")
| where TimeGenerated > ago(24h)
| project TimeGenerated, resource_workflowName_s, status_s, clientTrackingId_g
| order by TimeGenerated desc
```

**Query - Failed Run'ları Bul:**
```kusto
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.LOGIC"
| where resource_workflowName_s == "bearish-bot-orchestrator"
| where status_s == "Failed"
| where TimeGenerated > ago(7d)
| project TimeGenerated, resource_actionName_s, code_s, error_message_s
```

**Query - SendGrid Email Success Rate:**
```kusto
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.LOGIC"
| where resource_actionName_s in ("Send_Success_Email", "Send_No_Data_Email")
| where TimeGenerated > ago(7d)
| summarize 
    Total = count(),
    Succeeded = countif(status_s == "Succeeded"),
    Failed = countif(status_s == "Failed")
| extend SuccessRate = (Succeeded * 100.0) / Total
```

---

## 📝 Kritik Bilgiler ve Referanslar

### **Azure Resources**

| Resource | Type | Resource Group | Region | Subscription |
|----------|------|----------------|--------|--------------|
| bearish-bot-orchestrator | Logic App | TradeBot | East US | 74ab10ba-... |
| bearish-report-orchestrator | Logic App | tradebot-ops | East US | 74ab10ba-... |
| bearish-reporting-func-v2 | Function App | tradebot-ops | East US | 74ab10ba-... |
| BearishAlphaBot-VM-01 | Virtual Machine | TradeBot | East US | 74ab10ba-... |
| bearish-kv | Key Vault | TradeBot | East US | 74ab10ba-... |

### **SendGrid**

| Parameter | Value |
|-----------|-------|
| API Endpoint | `https://api.sendgrid.com/v3/mail/send` |
| API Key Location | Key Vault: `bearish-kv` → Secret: `sendgrid-api-key` |
| Verified Sender | `noreply@bearishbot.com` (veya `sefaasar@gmail.com`) |
| Recipient | `sefaasar@hotmail.com` |
| Connection Name | `sendgrid-connection` (Parent Logic App'te) |

### **Function Endpoints**

| Function | URL | Method | Auth |
|----------|-----|--------|------|
| run_report | `https://bearish-reporting-func-v2.azurewebsites.net/api/run_report` | POST | Function Key |
| loguploader | `https://bearish-reporting-func-v2.azurewebsites.net/api/loguploader` | GET | Function Key |

### **Schedules**

| Workflow | Trigger | Schedule |
|----------|---------|----------|
| bearish-bot-orchestrator | Recurrence | Every day at 08:00 UTC |
| bearish-report-orchestrator | Manual/HTTP | Called by parent |

---

## ✅ Tamamlanma Checklist

### **Implementation Phase**
- [ ] **ADIM 1:** Parent'a `Call_Report_Generator` action eklendi
- [ ] **ADIM 2:** `Check_Report_Status` condition eklendi
- [ ] **ADIM 3:** `Send_Success_Email` (200) eklendi
- [ ] **ADIM 4:** `Check_If_No_Data` condition eklendi
- [ ] **ADIM 5:** `Send_No_Data_Email` (204) eklendi
- [ ] **ADIM 6:** Child'dan email action'ları kaldırıldı
- [ ] **ADIM 7:** Child'ın response action'ları düzenlendi

### **Testing Phase**
- [ ] **TEST 1:** Parent manuel test (email geldi mi?)
- [ ] **TEST 2:** Child sadece response döndürüyor mu?
- [ ] **TEST 3:** Scheduled trigger test (ertesi gün 08:00)

### **Monitoring Setup**
- [ ] Runs history kontrol edildi (son 7 gün)
- [ ] Metrics dashboard oluşturuldu
- [ ] Alert rule eklendi (failed runs için)
- [ ] Log Analytics query'leri kaydedildi

### **Documentation**
- [ ] Bu playbook README.md'ye eklendi
- [ ] Diagram'lar güncellendi
- [ ] Runbook oluşturuldu (troubleshooting için)

---

## 🎓 Troubleshooting Decision Tree

```
Email gelmedi mi?
├─ Logic App failed mı?
│  ├─ EVET → Hangi action failed?
│  │  ├─ Call_Report_Generator → Child workflow loglarına bak
│  │  ├─ Send_Success_Email → SendGrid connection kontrol et
│  │  └─ Send_No_Data_Email → SendGrid connection kontrol et
│  └─ HAYIR → Email action succeeded ama email yok
│     ├─ SendGrid Portal → Activity'ye bak (gönderildi mi?)
│     ├─ Spam klasörünü kontrol et
│     └─ Sender email doğrulanmış mı kontrol et
│
├─ Child workflow 401 veriyor mu?
│  └─ EVET → SendGrid API key'i güncelle (Senaryo 1)
│
├─ Dynamic content çözülmüyor mu?
│  └─ EVET → Expression syntax'ını kontrol et (Senaryo 2)
│
└─ Parent child'ı çağıramıyor mu?
   └─ EVET → Resource group/subscription ayarlarını kontrol et (Senaryo 3)
```

---

## 📞 İletişim ve Destek

**Proje Sahibi:** Sefa Asar  
**Email:** sefaasar@hotmail.com  
**Repository:** c:\Users\sefaa\bearish-alpha-bot  

**Azure Subscription:** 74ab10ba-c96d-449e-97cb-ee4f9c0de714  
**Tenant:** (Microsoft Entra ID)

**Kritik Durumda:**
1. Azure Portal → Support → **New support request**
2. Issue type: **Technical**
3. Service: **Logic Apps**
4. Problem type: **Connectivity**

---

## 🎯 Son Notlar

### **Güvenlik**
- ✅ SendGrid API Key, Key Vault'ta saklanıyor
- ✅ Logic App, Managed Identity kullanıyor
- ✅ Function Key'ler rotate ediliyor

### **Maliyet**
- **Logic App:** ~$0.10/day (100 workflow executions)
- **SendGrid:** Free tier (100 emails/day)
- **Total:** ~$3/month

### **Bakım**
- **Haftalık:** Runs history kontrol et, failed run'ları incele
- **Aylık:** SendGrid activity raporunu incele
- **Yıllık:** API key'leri rotate et

---

**Playbook Version:** 1.0  
**Son Güncelleme:** 5 Aralık 2024  
**Durum:** ✅ Production Ready (Implementation bekliyor)