# Deploy Playbook — bearish-reporting-func-v2 (Azure Functions, Python 3.11)

## 1) On Hazirlik
- Araçlar: Azure CLI (login), PowerShell (pwsh), Python 3.11, `az` extension `azure-functions`.
- Subscription: `az account set --subscription "<SUB_ID>"` ile dogru aboneliği sec.
- Kaynak: App name `bearish-reporting-func-v2`, RG `tradebot-ops`, bolge West Europe.
- Ayarlar kritik: `SCM_DO_BUILD_DURING_DEPLOYMENT=true`, `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true`, `FUNCTIONS_WORKER_RUNTIME=python`, `AzureWebJobsStorage`, `bearishstorage_STORAGE`, `PYTHON_ISOLATE_WORKER_DEPENDENCIES=1`. `WEBSITE_RUN_FROM_PACKAGE` **olmamali**.

## 2) Paket Hazirlama (Zip)
- Dizin: `azure_functions/reporting/`.
- Paket içeriği: `function_app.py`, `function_app_runtime.py`, `host.json`, `requirements.txt` (sablon/template dizinleri HARIÇ).
- Komut (pwsh, repo kokunde):
  ```pwsh
  $src = "azure_functions/reporting"
  $zip = "deploy_manual_$(Get-Date -Format yyyyMMdd).zip"
  Compress-Archive -Path `
    "$src/function_app.py", `
    "$src/function_app_runtime.py", `
    "$src/host.json", `
    "$src/requirements.txt" `
    -DestinationPath "$src/$zip" -Force
  ```
- Dogrulama: `Test-Path azure_functions/reporting/<zip>`.

## 3) (Opsiyonel) Run-from-package Temizligi
- Yalniz onceki run-from-package artigi varsa:
  ```pwsh
  az functionapp stop   --name bearish-reporting-func-v2 --resource-group tradebot-ops
  az rest --method delete --uri "https://bearish-reporting-func-v2.scm.azurewebsites.net/api/vfs/data/SitePackages" --headers @{ "If-Match"="*" }
  az functionapp start  --name bearish-reporting-func-v2 --resource-group tradebot-ops
  ```
- App settings’te `WEBSITE_RUN_FROM_PACKAGE` olmadigini kontrol et.

## 4) Deploy (Remote Build Zip Deploy)
- Komut:
  ```pwsh
  $zip = "C:\Users\sefaa\bearish-alpha-bot\azure_functions\reporting\deploy_manual_YYYYMMDD.zip"
  az functionapp deployment source config-zip `
    --name bearish-reporting-func-v2 `
    --resource-group tradebot-ops `
    --src $zip `
    --build-remote true
  ```
- Cikti: `status: 4` ve "Deployment successful" beklenir.

## 5) Trigger Sync
- CLI ile:
  ```pwsh
  az rest --method post --uri "https://bearish-reporting-func-v2.azurewebsites.net/admin/host/synctriggers"
  ```
- Gerekirse master key ile:
  ```pwsh
  $base = "https://bearish-reporting-func-v2.azurewebsites.net"
  $master = "<master-key>"
  Invoke-RestMethod -Uri "$base/admin/host/synctriggers?code=$master" -Method POST
  ```

## 6) Smoke Test
- `run_report` (Function key):
  ```pwsh
  $key = "<run_report_function_key>"
  Invoke-RestMethod -Method Post `
    -Uri "https://bearish-reporting-func-v2.azurewebsites.net/api/run_report?code=$key" `
    -ContentType "application/json" `
    -Body '{}' `
    -TimeoutSec 120
  ```
  Beklenen: 200 veya 202, JSON yanit `status` alanli.
- `loguploader` (Function key):
  ```pwsh
  $key = "<loguploader_function_key>"
  Invoke-RestMethod -Method Post `
    -Uri "https://bearish-reporting-func-v2.azurewebsites.net/api/loguploader?code=$key" `
    -ContentType "application/json" `
    -Body '{}' `
    -TimeoutSec 180
  ```
  Beklenen: `status=success`, blob adi doner.
- Storage kontrolu:
  ```pwsh
  az storage blob list --account-name bearishstorage --container-name raw-logs --num-results 5 --output table
  ```
- Kudu deploy log (detay):
  ```pwsh
  $deployId = "<deployment-id-from-config-zip-output>"
  $creds = az functionapp deployment list-publishing-credentials --name bearish-reporting-func-v2 --resource-group tradebot-ops --output json | ConvertFrom-Json
  $credential = New-Object pscredential($creds.publishingUserName, (ConvertTo-SecureString $creds.publishingPassword -AsPlainText -Force))
  Invoke-RestMethod -Authentication Basic -Credential $credential -Uri "https://bearish-reporting-func-v2.scm.azurewebsites.net/api/deployments/$deployId/log"
  ```

## 7) Roll-back / Retry
- Ayni zip yeniden gonderilebilir (idempotent).
- Geri donmek icin onceki zip’i `--src` ile yeniden deploy et.
- Sorun: App restart + synctriggers; Kudu `LogFiles/Application` incele; App Insights trace sorgula.

## 8) Bakim Notlari
- Python 3.11 disina cikma.
- `requirements.txt` degisirse yeni zip + remote build sart.
- `templates` klasorunu pakete ekleme.
- `PYTHON_ISOLATE_WORKER_DEPENDENCIES=1` korunmali.
- Loguploader icin Managed Identity / RunCommand izinlerinin calistigini dogrula.
- Tek bolge: West Europe; log/ADX ayni bolgede.

## 9) Hizli Kontrol Listesi
1) Zip hazir mi (dogru dosyalar, templates haric)?
2) App settings: `SCM_DO_BUILD_DURING_DEPLOYMENT=true`, `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true`, `WEBSITE_RUN_FROM_PACKAGE` yok.
3) `config-zip --build-remote true` status 4.
4) `synctriggers` success.
5) `run_report` ve `loguploader` smoke test OK.
6) `raw-logs` container’da yeni blob gorundu.
7) Gerekirse Kudu log’lari temiz.
