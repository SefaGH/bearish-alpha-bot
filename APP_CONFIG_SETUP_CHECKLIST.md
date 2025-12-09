# App Configuration Setup - Complete Checklist

**Status:** Build in progress (appconfig-rest-api-v2)  
**Date:** December 9, 2025

---

## ✅ Complete Setup Chain

### 1️⃣ Docker Image (`appconfig-rest-api-v2`)
- [x] `curl` installed in Dockerfile
- [x] Python 3.11 base image
- [x] TA-Lib, CCXT, PyTorch, Pandas
- [x] PYTHONPATH set correctly
- [ ] **[BUILDING]** ACR build in progress (freed space, restarted build)

**Build Command:**
```bash
az acr build --registry bearishalphabot --image "bearish-bot:appconfig-rest-api-v2" . -f Dockerfile
```

**Status:** ~13min expected build time

---

### 2️⃣ Environment File (`bearish-bot.env`)
- [x] Local copy updated with AZURE_APPCONFIG vars
- [ ] **[TODO]** VM file `/home/azureuser/bearish-bot.env` needs update

**Required Additions to VM File:**
```bash
# === AZURE APP CONFIGURATION ===
AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io
AZURE_APPCONFIG_LABEL=production
```

**Current Status:** Runbook will ensure these are added before container start

---

### 3️⃣ Python Wrapper Script (`vm_run_session.py`)
- [x] Env file validation added
  ```python
  if not env_path.exists():
      print(f"❌ ERROR: Environment file not found: {env_file}")
      return 1
  ```
- [x] Returns error code 1 if env file missing
- [x] Validates before docker run command

---

### 4️⃣ Container Entry Point (`vm_boot.py`)
- [x] Environment variable validation added
  ```python
  required_env_vars = ['BINGX_KEY', 'BINGX_SECRET', 'CAPITAL_USDT', 'TRADING_MODE', 'EXCHANGES']
  optional_env_vars = ['AZURE_APPCONFIG_ENDPOINT', 'AZURE_APPCONFIG_LABEL']
  ```
- [x] Returns error code 1 if required vars missing
- [x] Warns if App Config vars missing
- [x] Logs all checks before continuing

---

### 5️⃣ Initialization Scripts
- [x] `azure_boot.py` → Setup PYTHONPATH, directories, ML environment
- [x] `scripts/setup_gemma_artifacts.sh` → Copy ML models
- [x] `scripts/setup_ml_model_links.sh` → Link timeframe models
- [x] Both scripts run with error handling

---

### 6️⃣ Configuration Loader (`LiveTradingConfiguration`)
- [x] `_load_from_app_config()` method implemented
  - Uses REST API (not SDK) to work around IMDS API version issues
  - Calls IMDS with correct API version: `2017-12-01`
  - Queries App Config: `GET {endpoint}/kv?key=BearishAlphaBot/*&label=production`
  - Parses JSON response and extracts 64 settings
- [x] Error handling enhanced
  - Detects missing `curl` command
  - Logs IMDS response details
  - Logs App Config REST API response details
  - Full exception tracing

---

### 7️⃣ Runbook (`Start-BearishBot-Enhanced.ps1`)
- [x] Docker cleanup (prune unused)
- [x] Stop/remove existing container
- [x] Update TRADING_DURATION in env file
- [x] **[NEW]** Ensure AZURE_APPCONFIG_ENDPOINT in env file
- [x] **[NEW]** Ensure AZURE_APPCONFIG_LABEL in env file
- [x] Call `vm_run_session.py` with image tag
- [x] Health check after 10 seconds
- [x] Return status to Logic App

---

## 🔄 Complete Flow Diagram

```
RUNBOOK (PowerShell)
  ↓
  ├─ Clean Docker system
  ├─ Remove old container
  ├─ Update env file:
  │  ├─ TRADING_DURATION
  │  ├─ AZURE_APPCONFIG_ENDPOINT ← ensure set
  │  └─ AZURE_APPCONFIG_LABEL ← ensure set
  └─ Call vm_run_session.py
       ↓
vm_run_session.py (Python wrapper)
  ├─ Validate: env file exists ✅
  └─ docker run --env-file /home/azureuser/bearish-bot.env \
      --image bearishalphabot.azurecr.io/bearish-bot:appconfig-rest-api-v2
       ↓
CONTAINER STARTS (Python 3.11 slim + curl)
  ↓
vm_boot.py (Entry point)
  ├─ Validate: Required env vars present ✅
  │  ├─ BINGX_KEY ✓
  │  ├─ BINGX_SECRET ✓
  │  ├─ CAPITAL_USDT ✓
  │  ├─ TRADING_MODE ✓
  │  └─ EXCHANGES ✓
  ├─ Validate: Optional App Config vars ⚠️
  │  ├─ AZURE_APPCONFIG_ENDPOINT (from env file)
  │  └─ AZURE_APPCONFIG_LABEL (from env file)
  └─ Setup:
     ├─ setup_environment() → PYTHONPATH
     ├─ ensure_directories() → Create dirs
     ├─ setup_default_manifest() → GEMMA-2.0.0
     └─ setup_ml_environment():
        ├─ Set GEMMA env vars
        ├─ Run setup_gemma_artifacts.sh
        └─ Run setup_ml_model_links.sh
         ↓
live_trading_launcher.py
  ↓
LiveTradingConfiguration.load()
  ├─ Load YAML config
  └─ Load from App Configuration (REST API):
     ├─ os.getenv('AZURE_APPCONFIG_ENDPOINT') → get from env
     ├─ os.getenv('AZURE_APPCONFIG_LABEL') → get from env
     ├─ curl IMDS token endpoint (api-version=2017-12-01)
     ├─ curl App Config REST API
     ├─ Parse JSON → extract 64 settings
     └─ Merge: AppConfig > ENV > YAML
  ↓
Bot starts with merged configuration
```

---

## 📊 Environment Variables Flow

### Loaded by Docker (`--env-file`)
```
BINGX_KEY
BINGX_SECRET
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
CAPITAL_USDT
TRADING_MODE=paper
DEBUG_MODE=false
TRADING_DURATION=600
...
AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io  ← FROM ENV FILE
AZURE_APPCONFIG_LABEL=production                                 ← FROM ENV FILE
```

### Used by Configuration System
```
LiveTradingConfiguration.load():
  ├─ os.getenv('AZURE_APPCONFIG_ENDPOINT') → REST API endpoint
  └─ os.getenv('AZURE_APPCONFIG_LABEL') → Label filter
       ↓
  Query: https://appcs-bearish-bot.azconfig.io/kv?key=BearishAlphaBot/*&label=production
       ↓
  Expected Response: 64 settings
  ├─ Strategy parameters (STRATEGY_*)
  ├─ Trading parameters (CAPITAL_*, RISK_*, etc.)
  ├─ WebSocket parameters (WS_*)
  └─ ... (all 64 settings)
```

---

## 🚀 Critical Success Indicators (CSI)

### Build Success
```bash
# Expected in ~13 minutes:
az acr repository show-tags --name bearishalphabot --repository bearish-bot
→ appconfig-rest-api-v2 [PRESENT]
```

### Container Startup Success
```log
vm_boot.py:
✅ All required environment variables present
✅ Azure App Configuration environment variables configured
✅ PYTHONPATH configured: /app:/app/src:/app/scripts
✅ Required directories and placeholder files created
✅ GEMMA-2.0.0 manifest created for Azure
✅ GEMMA environment variables set
🔧 Running scripts/setup_gemma_artifacts.sh...
✅ scripts/setup_gemma_artifacts.sh completed successfully
🔧 Running scripts/setup_ml_model_links.sh...
✅ scripts/setup_ml_model_links.sh completed successfully
```

### Config Loading Success
```log
live_trading_config.py:
📡 Loading configuration from Azure App Configuration (via REST API)...
   Endpoint: https://appcs-bearish-bot.azconfig.io
   Label: production
✅ Loaded 64 settings from App Configuration
   Converted to nested structure
```

### Trading Bot Success
```log
bearish-alpha-bot:
✅ BingX credentials found
✅ Telegram notifications enabled
[1/8] Loading Environment Configuration...
[2/8] Initializing BingX Exchange Connection...
[5/8] Initializing Trading Strategies...
✓ 2 strategies ready for trading
...
```

---

## 📋 Pre-Deployment Checklist

- [x] Dockerfile has `curl` installed
- [x] bearish-bot.env has AZURE_APPCONFIG vars
- [x] vm_run_session.py validates env file
- [x] vm_boot.py validates required env vars
- [x] live_trading_config.py has REST API implementation
- [x] Error handling improved in config loader
- [x] Runbook ensures env vars before container start
- [ ] **[IN PROGRESS]** Build appconfig-rest-api-v2 image
- [ ] **[NEXT]** Deploy image to VM and test

---

## 🔗 File References

| File | Changes | Commit |
|------|---------|--------|
| `Dockerfile` | Added `curl` to apt-get | fix: Add curl |
| `bearish-bot.env` | Added AZURE_APPCONFIG* vars | feat: Add env vars |
| `scripts/vm_run_session.py` | Added env file validation | feat: Add validation |
| `vm_boot.py` | Added env var validation | feat: Add validation |
| `src/config/live_trading_config.py` | Enhanced error handling, REST API impl | (already in code) |
| `azure_automation/Start-BearishBot-Enhanced.ps1` | Enhanced with App Config env var setup | feat: Enhance runbook |

---

## ⏰ Timeline

| Time | Action | Status |
|------|--------|--------|
| T-0:00 | Identify curl missing in container | ✅ Done |
| T+0:30 | Delete old images from ACR (freed space) | ✅ Done |
| T+0:35 | Restart build (appconfig-rest-api-v2) | 🔄 Building |
| T+13min | **[EXPECTED]** Build completes | ⏳ Waiting |
| T+13min | Deploy image to VM | 📋 TODO |
| T+14min | Run container with new image | 📋 TODO |
| T+14:30 | Verify App Config loading in logs | 📋 TODO |

---

## 🛠️ Debugging Commands

### Check Build Status
```bash
az acr repository show-tags --name bearishalphabot --repository bearish-bot
```

### Check VM Env File
```bash
ssh azureuser@20.73.171.66 "cat /home/azureuser/bearish-bot.env | grep AZURE_APPCONFIG"
```

### Check Container Logs
```bash
docker logs bearish-bot 2>&1 | head -50
docker logs bearish-bot 2>&1 | grep -i "app config\|curl\|⊘"
```

### Verify Curl in Image
```bash
docker run --rm bearishalphabot.azurecr.io/bearish-bot:appconfig-rest-api-v2 which curl
```

### Test IMDS (from inside container)
```bash
curl -s -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2017-12-01&resource=https://appconfig.azure.com" | jq '.access_token'
```

### Test App Config API (from inside container)
```bash
TOKEN=$(curl -s -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2017-12-01&resource=https://appconfig.azure.com" | jq -r '.access_token')
curl -s -H "Authorization: Bearer $TOKEN" "https://appcs-bearish-bot.azconfig.io/kv?key=BearishAlphaBot/*&label=production&api-version=1.0" | jq '.items | length'
```

---

## 🎯 Next Steps

1. **Wait for build completion** (~13 minutes from T+0:35)
2. **Verify image exists:**
   ```bash
   az acr repository show-tags --name bearishalphabot --repository bearish-bot | grep appconfig-rest-api-v2
   ```
3. **Deploy to VM using Runbook or Logic App**
4. **Check logs for success indicators**
5. **Verify 64 settings loaded from App Configuration**

---

## 💡 Key Points

- **Curl is now in container** ✅ Can call REST APIs
- **Env vars are pre-validated** ✅ vm_boot.py catches missing vars early
- **App Config endpoint known** ✅ From env file
- **REST API uses correct API version** ✅ 2017-12-01 (not 2017-09-01)
- **Fallback is graceful** ✅ If App Config fails, falls back to YAML
- **Error messages are detailed** ✅ Will see exactly what failed

