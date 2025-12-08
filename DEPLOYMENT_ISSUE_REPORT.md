# Azure Functions Deployment Issue Report
**Date:** December 5, 2025  
**Function App:** bearish-reporting-func-v2  
**Resource Group:** tradebot-ops  
**Region:** East US  
**Runtime:** Python 3.11

---

## 🔴 Problem Summary

Azure Functions deployment başarılı görünüyor ancak Portal'da **eski kod hala görünüyor**. Yeni Python v2 model dosyaları deploy ediliyor fakat Function App bunları tanımıyor.

---

## 📁 File Structure

### **Hedef Dosya Yapısı (Python v2 Model)**
```
/home/site/wwwroot/
├── function_app.py              # Main entry point (Python v2 - @app.route decorators)
├── function_app_runtime.py      # Business logic
├── requirements.txt             # Dependencies
├── host.json                    # Function App config
└── templates/
    └── report.html.j2          # Email template
```

### **Eski Yapı (Python v1 Model - Silinmeli)**
```
/home/site/wwwroot/
├── HttpExample/
│   ├── __init__.py             # Eski function code
│   └── function.json           # Eski function metadata
└── loguploader/
    ├── __init__.py
    └── function.json
```

---

## 🚀 Denenen Deployment Yöntemleri

### **1. ZIP Deployment (İlk Deneme)**
```powershell
# Dosyalar paketlendi
Compress-Archive -Path function_app.py, function_app_runtime.py, requirements.txt -DestinationPath deploy_20251205_212107.zip

# Azure'a deploy edildi
az functionapp deployment source config-zip `
  --resource-group tradebot-ops `
  --name bearish-reporting-func-v2 `
  --src deploy_20251205_212107.zip `
  --build-remote true
```

**Sonuç:** ✅ Deployment başarılı (6,956 files, 220 seconds, remote build succeeded)  
**Problem:** ❌ Portal'da eski dosyalar görünmeye devam etti

---

### **2. Manuel Kudu API Upload (İkinci Deneme)**

**SCM Basic Auth Sorunu:**
- İlk denemede 401 Unauthorized hatası alındı
- **Çözüm:** Azure Portal → Configuration → SCM Basic Auth Publishing Credentials → **ON**

```powershell
# Credentials alındı
$creds = az functionapp deployment list-publishing-credentials --name bearish-reporting-func-v2 --resource-group tradebot-ops --output json | ConvertFrom-Json
$authString = "$($creds.publishingUserName):$($creds.publishingPassword)"
$authBytes = [System.Text.Encoding]::ASCII.GetBytes($authString)
$authBase64 = [System.Convert]::ToBase64String($authBytes)

# Manuel upload
Invoke-RestMethod `
  -Uri "https://bearish-reporting-func-v2.scm.azurewebsites.net/api/vfs/site/wwwroot/function_app.py" `
  -Headers @{Authorization="Basic $authBase64"} `
  -Method PUT `
  -Body $funcAppBytes `
  -ContentType "application/octet-stream"
```

**Sonuç:** ✅ Dosyalar wwwroot'a yüklendi  
**Problem:** ❌ Portal hala eski function'ları gösterdi

---

### **3. Cache Temizleme Denemeleri**

```powershell
# Hard restart
az functionapp stop --name bearish-reporting-func-v2 --resource-group tradebot-ops
az functionapp start --name bearish-reporting-func-v2 --resource-group tradebot-ops

# Kudu process restart
Invoke-RestMethod `
  -Uri "https://bearish-reporting-func-v2.scm.azurewebsites.net/api/processes/0" `
  -Headers @{Authorization="Basic $authBase64"} `
  -Method DELETE
```

**Sonuç:** ⚠️ Cache temizlendi ancak Portal'da değişiklik yok

---

### **4. ZIP Package Araştırması**

**Tespit Edilen Sorun:**
Function App bir ZIP package'dan çalışıyor olabilir.

**Kontrol Edilen Lokasyonlar:**
- `/data/SitePackages/` - Aktif ZIP package lokasyonu
- `/home/site/wwwroot/*.zip` - Root'taki ZIP dosyaları
- `/home/site/wwwroot/azure_deployed_backup/` - Backup ZIP'leri

**Yapılan İşlemler:**
```powershell
# WEBSITE_RUN_FROM_PACKAGE ayarı silindi
az functionapp config appsettings delete `
  --name bearish-reporting-func-v2 `
  --resource-group tradebot-ops `
  --setting-names WEBSITE_RUN_FROM_PACKAGE

# SitePackages klasöründeki dosyalar silindi (varsa)
Invoke-RestMethod `
  -Uri "https://bearish-reporting-func-v2.scm.azurewebsites.net/api/vfs/data/SitePackages/<file>" `
  -Method DELETE
```

**Sonuç:** ⚠️ Package ayarları temizlendi ancak sorun devam etti

---

### **5. Radikal Çözüm: Tam Temizlik (Son Deneme)**

```powershell
# 1. Function App durduruldu
az functionapp stop --name bearish-reporting-func-v2 --resource-group tradebot-ops

# 2. Eski dosyalar silindi
$filesToDelete = @("function_app.py", "function_app_runtime.py", "__init__.py", "host.json", "requirements.txt")
foreach ($file in $filesToDelete) {
    Invoke-RestMethod -Uri "https://.../$file" -Method DELETE
}

# 3. V1 model klasörleri silindi (varsa)
Invoke-RestMethod -Uri "https://.../HttpExample/" -Method DELETE
Invoke-RestMethod -Uri "https://.../loguploader/" -Method DELETE

# 4. Yeni dosyalar yüklendi
# - function_app.py (Python v2 model)
# - function_app_runtime.py
# - requirements.txt
# - host.json

# 5. Function App başlatıldı
az functionapp start --name bearish-reporting-func-v2 --resource-group tradebot-ops
```

**Sonuç:** ⏳ **Pending** - 60 saniye bekleniyor (function discovery için)

---

## 📋 Deploy Edilen Dosyalar

### **function_app.py** (Python v2 Model)
- **Boyut:** ~2-3 KB
- **Format:** Python v2 programming model
- **Key Features:**
  - `@app.route(route="run_report")` decorator
  - `@app.route(route="loguploader")` decorator
  - Lazy import: `import function_app_runtime as runtime`

**İçerik Özeti:**
```python
import azure.functions as func
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="run_report", auth_level=func.AuthLevel.FUNCTION)
def run_report(req: func.HttpRequest) -> func.HttpResponse:
    import function_app_runtime as runtime
    return runtime.run_report_logic(req)

@app.route(route="loguploader", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def log_uploader(req: func.HttpRequest) -> func.HttpResponse:
    import function_app_runtime as runtime
    return runtime.log_uploader_http(req)
```

### **function_app_runtime.py**
- **Boyut:** ~10-15 KB
- **Purpose:** Business logic implementation
- **Functions:**
  - `run_report_logic()` - Analyzes trading logs, returns JSON with report URL
  - `log_uploader_http()` - Syncs VM logs via Azure RunCommand
  - `analyze_trading_logs()` - Hybrid analysis (summary table + raw JSON fallback)

**Dependencies:**
- `azure-identity` (DefaultAzureCredential)
- `azure-storage-blob` (BlobServiceClient)
- `azure-mgmt-compute` (ComputeManagementClient)

### **requirements.txt**
```
azure-functions
azure-identity
azure-storage-blob
azure-mgmt-compute
```

### **host.json**
```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "maxTelemetryItemsPerSecond": 20
      }
    }
  },
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  }
}
```

---

## 🔍 Verification Checks

### **Kudu File Verification**
```powershell
# wwwroot içeriği kontrol edildi
GET https://bearish-reporting-func-v2.scm.azurewebsites.net/api/vfs/site/wwwroot/

# function_app.py içeriği okundu
GET https://bearish-reporting-func-v2.scm.azurewebsites.net/api/vfs/site/wwwroot/function_app.py

# Sonuç: ✅ function_app.py mevcut ve içerik doğru (@app.route decorator bulundu)
```

### **App Settings Check**
```powershell
az functionapp config appsettings list --name bearish-reporting-func-v2 --resource-group tradebot-ops

# Kontrol edilen ayarlar:
# - WEBSITE_RUN_FROM_PACKAGE: ✅ Silindi (artık yok)
# - FUNCTIONS_WORKER_RUNTIME: python
# - FUNCTIONS_EXTENSION_VERSION: ~4
```

### **SitePackages Check**
```powershell
GET https://bearish-reporting-func-v2.scm.azurewebsites.net/api/vfs/data/SitePackages/

# Sonuç: ✅ Boş veya yok (ZIP package kullanılmıyor)
```

---

## 🐛 Root Cause Analysis

### **Muhtemel Sebepler:**

1. **Function Discovery Delay**
   - Azure Functions runtime yeni function'ları keşfetmek için zaman alıyor
   - Portal UI cache'i eski metadata'yı gösteriyor olabilir

2. **Python V1 → V2 Migration Issue**
   - Function App başlangıçta v1 model ile create edilmiş olabilir
   - Runtime v1 yapısını arıyor ama v2 buluyor

3. **Hidden ZIP Package**
   - Kontrol edilmeyen bir lokasyonda aktif ZIP package olabilir
   - `WEBSITE_RUN_FROM_PACKAGE` silinmiş ama effect devam ediyor olabilir

4. **Portal Cache**
   - Azure Portal metadata cache'i aggressive
   - F5 yenileme yeterli olmayabilir, Portal tamamen kapatılıp açılmalı

5. **Function Host Metadata**
   - Function host metadata dosyaları (`.azurefunctions/` klasörü) eski olabilir
   - Bu metadata dosyaları function discovery'yi etkiliyor olabilir

---

## 🔧 Recommended Solutions

### **Öncelik 1: Wait & Verify**
```bash
# 60-120 saniye bekle (function discovery için)
# Portal'ı tamamen kapat ve yeniden aç
# Ctrl+Shift+R ile hard refresh
# Functions sekmesini kontrol et
```

### **Öncelik 2: Force Restart with Cold Start**
```powershell
# Stop
az functionapp stop --name bearish-reporting-func-v2 --resource-group tradebot-ops

# 30 saniye bekle
Start-Sleep -Seconds 30

# Start
az functionapp start --name bearish-reporting-func-v2 --resource-group tradebot-ops

# 60 saniye bekle
Start-Sleep -Seconds 60
```

### **Öncelik 3: Delete & Recreate Functions App**
```powershell
# Son çare: Function App'i sil ve yeniden oluştur (Python v2 model ile)
az functionapp delete --name bearish-reporting-func-v2 --resource-group tradebot-ops

az functionapp create `
  --name bearish-reporting-func-v2 `
  --resource-group tradebot-ops `
  --storage-account bearishstorage `
  --consumption-plan-location eastus `
  --runtime python `
  --runtime-version 3.11 `
  --functions-version 4 `
  --os-type Linux

# Deploy
az functionapp deployment source config-zip `
  --resource-group tradebot-ops `
  --name bearish-reporting-func-v2 `
  --src deploy_20251205_212107.zip `
  --build-remote true
```

### **Öncelik 4: Use VS Code Azure Functions Extension**
```
1. VS Code'da Azure Functions extension ile bağlan
2. bearish-reporting-func-v2'ye sağ tıkla → Deploy to Function App
3. azure_functions/reporting/ klasörünü seç
4. Remote build ile deploy et
```

---

## 📊 Current Status

| Check | Status | Details |
|-------|--------|---------|
| ZIP Deployment | ✅ Success | 6,956 files, remote build succeeded |
| Kudu File Verification | ✅ Success | function_app.py exists, content correct |
| SCM Basic Auth | ✅ Enabled | Manual upload çalışıyor |
| WEBSITE_RUN_FROM_PACKAGE | ✅ Removed | Artık wwwroot'tan çalışmalı |
| SitePackages Cleanup | ✅ Done | ZIP package yok |
| Manual File Upload | ✅ Done | Tüm dosyalar yüklendi |
| Old Files Cleanup | ✅ Done | Eski dosyalar silindi |
| Portal Visibility | ❌ Failed | **Hala eski functions görünüyor** |

---

## 🎯 Next Steps

1. **Wait 60+ seconds** for function discovery
2. **Close Portal completely** (not just tab)
3. **Reopen Portal** and navigate to bearish-reporting-func-v2
4. **Hard refresh** (Ctrl+Shift+R)
5. **Check Functions tab** for:
   - `run_report` (HTTP trigger)
   - `loguploader` (HTTP trigger POST)

**If still not working:**
- Contact Azure Support
- Consider Function App recreation
- Use VS Code deployment as alternative

---

## 📝 Lessons Learned

1. **Python v2 Model Requires:**
   - `function_app.py` at root (not in subfolders)
   - `@app.route` decorators (not function.json)
   - Clean migration from v1 to v2

2. **Deployment Best Practices:**
   - Always check `WEBSITE_RUN_FROM_PACKAGE` setting
   - Verify files in Kudu after deployment
   - Wait for function discovery (60-120 seconds)
   - Clear all caches (Portal + Browser)

3. **Troubleshooting Order:**
   - Verify files in wwwroot (Kudu)
   - Check app settings (WEBSITE_RUN_FROM_PACKAGE)
   - Look for ZIP packages (SitePackages)
   - Force restart with delay
   - Consider clean recreation

---

## 🔗 References

- [Azure Functions Python Developer Guide](https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-python)
- [Python v2 Programming Model](https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-python#folder-structure)
- [ZIP Deployment for Azure Functions](https://learn.microsoft.com/en-us/azure/azure-functions/deployment-zip-push)
- [Run Functions from Package](https://learn.microsoft.com/en-us/azure/azure-functions/run-functions-from-deployment-package)

---

**Report Generated:** December 5, 2025  
**Last Updated:** After radikal temizlik ve manuel upload  
**Status:** ⏳ Awaiting function discovery (60 seconds)
