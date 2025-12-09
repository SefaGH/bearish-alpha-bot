# Azure App Configuration System Guide

## 🎯 Overview

The Bearish Alpha Bot uses **Azure App Configuration** for centralized, cloud-based configuration management. This enables dynamic configuration updates without rebuilding Docker images or modifying environment files.

---

## 📊 Configuration Priority Hierarchy

Configuration values are loaded and merged in the following order (last wins):

```
┌─────────────────────────────────────────┐
│  1. config.example.yaml (Base/Defaults) │  ← Lowest Priority
├─────────────────────────────────────────┤
│  2. Environment Variables (Runtime)     │  ← Middle Priority
├─────────────────────────────────────────┤
│  3. Azure App Configuration (Cloud)     │  ← HIGHEST Priority ✅
└─────────────────────────────────────────┘
```

**Example:**
```yaml
# config.example.yaml
capital_usdt: 1000

# bearish-bot.env
CAPITAL_USDT=500

# Azure App Configuration
CAPITAL_USDT=2000  ← This value WINS (final value = 2000)
```

---

## 🏗️ Infrastructure Setup

### Azure Resources

| Resource | Value |
|----------|-------|
| **App Configuration Store** | `appcs-bearish-bot` |
| **Endpoint** | `https://appcs-bearish-bot.azconfig.io` |
| **Label** | `production` |
| **Authentication** | Managed Identity (System-Assigned) |
| **VM Identity ID** | `a85de1e4-29f5-4fb2-a7f5-c91a11adfa11` |
| **RBAC Role** | App Configuration Data Reader |

### Current Settings Count
- **Base Settings**: 26 (trading parameters, risk config, ML settings)
- **Strategy Settings**: 38 (2 strategies × 19 parameters each)
- **Total**: 64 settings

---

## 🔐 Authentication Flow

The system uses **REST API** with **Managed Identity** instead of SDK to work around IMDS API version compatibility issues:

```
1. Container requests token from IMDS
   ↓ (HTTP 169.254.169.254 with api-version=2017-12-01)
2. IMDS returns JWT Bearer token
   ↓
3. REST API call to App Configuration with Bearer token
   ↓ (curl to https://appcs-bearish-bot.azconfig.io/kv)
4. Parse JSON response and extract settings
   ↓
5. Convert flat keys to nested structure
   ↓
6. Deep merge with YAML and ENV vars
```

**Code Location**: `src/config/live_trading_config.py` → `_load_from_app_config()` (Lines 705-790)

---

## 📝 Configuration Settings

### Base Settings (26)

```bash
# Trading Configuration
TRADING_MODE=paper
ML_ENABLED=true
EXCHANGES=bingx
TRADING_SYMBOLS=BTC/USDT:USDT

# Capital & Risk
CAPITAL_USDT=100
PER_TRADE_RISK_PCT=0.01
DAILY_MAX_TRADES=8
MAX_POSITION_SIZE=0.1
MAX_PORTFOLIO_RISK=0.05
MAX_DRAWDOWN=0.1

# Duplicate Prevention
DUPLICATE_PREVENTION_THRESHOLD=0.0005
DUPLICATE_PREVENTION_COOLDOWN=20
PRICE_DELTA_BYPASS_ENABLED=true
PRICE_DELTA_BYPASS_THRESHOLD=0.0015

# WebSocket
WS_MAX_STREAMS_BINGX=10

# ML/Gemma
GEMMA_ENABLED=true
ML_ACTIVE_BUNDLE=artifacts/gemma/final
ML_FEAT_VOL_WINDOWS=5,10,20,50
ML_FEAT_MOM_WINDOWS=5,10,20,50

# Symbol-Specific RSI Thresholds
RSI_THRESHOLD_BTC=50
RSI_THRESHOLD_ETH=50
RSI_THRESHOLD_SOL=50
```

### Strategy Settings (38 total = 2 strategies × 19 params)

**adaptive_ob (Adaptive Order Book Strategy)**
```bash
STRATEGY_ADAPTIVE_OB_ENABLED=true
STRATEGY_ADAPTIVE_OB_MIN_CONFIDENCE=0.65
STRATEGY_ADAPTIVE_OB_TAKE_PROFIT_PCT=0.015
STRATEGY_ADAPTIVE_OB_STOP_LOSS_PCT=0.02
STRATEGY_ADAPTIVE_OB_RISK_REWARD_RATIO=2.0
STRATEGY_ADAPTIVE_OB_MIN_SIGNAL_SCORE=60
STRATEGY_ADAPTIVE_OB_ENABLED_DYNAMIC_RR=true
STRATEGY_ADAPTIVE_OB_BASE_RR=2.0
STRATEGY_ADAPTIVE_OB_MIN_RR=1.2
STRATEGY_ADAPTIVE_OB_MAX_RR=3.0
STRATEGY_ADAPTIVE_OB_WEIGHT_REGIME=0.3
STRATEGY_ADAPTIVE_OB_WEIGHT_VOLATILITY=0.25
STRATEGY_ADAPTIVE_OB_WEIGHT_MOMENTUM=0.25
STRATEGY_ADAPTIVE_OB_WEIGHT_PPO_CONFIDENCE=0.2
STRATEGY_ADAPTIVE_OB_ENABLED_REGIME_SOFT_WEIGHT=true
STRATEGY_ADAPTIVE_OB_REGIME_HARD_REJECT_THRESHOLD=0.3
STRATEGY_ADAPTIVE_OB_REGIME_FULL_WEIGHT_THRESHOLD=0.6
STRATEGY_ADAPTIVE_OB_SYMBOLS_BTC_RSI_THRESHOLD=50
STRATEGY_ADAPTIVE_OB_SYMBOLS_ETH_RSI_THRESHOLD=50
```

**adaptive_str (Adaptive STR Strategy)**
```bash
STRATEGY_ADAPTIVE_STR_ENABLED=true
STRATEGY_ADAPTIVE_STR_MIN_CONFIDENCE=0.65
STRATEGY_ADAPTIVE_STR_TAKE_PROFIT_PCT=0.015
STRATEGY_ADAPTIVE_STR_STOP_LOSS_PCT=0.02
STRATEGY_ADAPTIVE_STR_RISK_REWARD_RATIO=2.0
STRATEGY_ADAPTIVE_STR_MIN_SIGNAL_SCORE=60
STRATEGY_ADAPTIVE_STR_ENABLED_DYNAMIC_RR=true
STRATEGY_ADAPTIVE_STR_BASE_RR=2.0
STRATEGY_ADAPTIVE_STR_MIN_RR=1.2
STRATEGY_ADAPTIVE_STR_MAX_RR=3.0
STRATEGY_ADAPTIVE_STR_WEIGHT_REGIME=0.3
STRATEGY_ADAPTIVE_STR_WEIGHT_VOLATILITY=0.25
STRATEGY_ADAPTIVE_STR_WEIGHT_MOMENTUM=0.25
STRATEGY_ADAPTIVE_STR_WEIGHT_PPO_CONFIDENCE=0.2
STRATEGY_ADAPTIVE_STR_ENABLED_REGIME_SOFT_WEIGHT=true
STRATEGY_ADAPTIVE_STR_REGIME_HARD_REJECT_THRESHOLD=0.3
STRATEGY_ADAPTIVE_STR_REGIME_FULL_WEIGHT_THRESHOLD=0.6
STRATEGY_ADAPTIVE_STR_SYMBOLS_BTC_RSI_THRESHOLD=50
STRATEGY_ADAPTIVE_STR_SYMBOLS_ETH_RSI_THRESHOLD=50
```

---

## 🔧 Environment Variables (Required)

These must be set in `/home/azureuser/bearish-bot.env` on the VM:

```bash
# Azure App Configuration (Required for cloud config)
AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io
AZURE_APPCONFIG_LABEL=production

# Exchange Credentials (Not in App Config for security)
BINGX_KEY=your_key_here
BINGX_SECRET=your_secret_here

# Telegram Notifications (Optional, not in App Config)
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Python Environment
PYTHONUNBUFFERED=1
PYTHONPATH=/app:/app/src:/app/scripts
```

---

## 📦 Docker Image

### Current Image
```bash
bearishalphabot.azurecr.io/bearish-bot:appconfig-rest-api-v2
```

### Key Features
- ✅ **curl installed** (for REST API calls)
- ✅ **Enhanced error handling** (curl detection, detailed logging)
- ✅ **Correct IMDS API version** (2017-12-01)
- ✅ **REST API bypass** (avoids SDK IMDS issues)

### Build Command
```powershell
az acr build --registry bearishalphabot `
  --image "bearish-bot:appconfig-rest-api-v2" `
  . -f Dockerfile
```

---

## 🚀 Deployment Workflow

### 1. VM Environment Setup
```bash
# Ensure App Config env vars in bearish-bot.env
echo "AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io" >> bearish-bot.env
echo "AZURE_APPCONFIG_LABEL=production" >> bearish-bot.env
```

### 2. Container Startup
```bash
cd /home/azureuser
sudo python3 vm_run_session.py \
  --image bearishalphabot.azurecr.io/bearish-bot:appconfig-rest-api-v2 \
  --name bearish-bot
```

### 3. Verification
```bash
# Check container running
docker ps --filter "name=bearish-bot"

# Check App Config load success
docker logs bearish-bot 2>&1 | grep -i "App Config"
```

**Expected Log Output:**
```
✅ Azure App Configuration environment variables configured
📡 Loading configuration from Azure App Configuration (via REST API)...
✅ Loaded 64 settings from App Configuration
```

---

## 🎯 Adding New Settings

### Step 1: Add to Azure App Configuration
```bash
# Via Azure CLI
az appconfig kv set \
  --name appcs-bearish-bot \
  --key "NEW_PARAMETER" \
  --value "your_value" \
  --label production \
  --yes

# Via Azure Portal
# Navigate to App Configuration → Configuration Explorer → + Create
```

### Step 2: Restart Container (No rebuild needed!)
```bash
# On VM
sudo docker stop bearish-bot
sudo docker rm bearish-bot
cd /home/azureuser
sudo python3 vm_run_session.py \
  --image bearishalphabot.azurecr.io/bearish-bot:appconfig-rest-api-v2 \
  --name bearish-bot
```

### Step 3: Verify New Setting Loaded
```bash
docker logs bearish-bot 2>&1 | grep "NEW_PARAMETER"
```

---

## 🔍 Troubleshooting

### Issue: "App Configuration env vars missing"
```bash
# Check env file
cat /home/azureuser/bearish-bot.env | grep AZURE_APPCONFIG

# Add if missing
echo "AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io" | sudo tee -a /home/azureuser/bearish-bot.env
echo "AZURE_APPCONFIG_LABEL=production" | sudo tee -a /home/azureuser/bearish-bot.env
```

### Issue: "curl: command not found"
```bash
# Check if curl installed in container
docker exec bearish-bot which curl

# If missing, rebuild image with curl (already fixed in appconfig-rest-api-v2)
```

### Issue: "Failed to load from Azure App Configuration"
```bash
# Check Managed Identity assigned
az vm identity show --resource-group TradeBot --name BearishAlphaBot-VM-01

# Check RBAC role assignment
az role assignment list \
  --assignee a85de1e4-29f5-4fb2-a7f5-c91a11adfa11 \
  --scope "/subscriptions/YOUR_SUB_ID/resourceGroups/TradeBot/providers/Microsoft.AppConfiguration/configurationStores/appcs-bearish-bot"

# Test IMDS endpoint from VM
curl -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2017-12-01&resource=https://appconfig.azure.com"
```

### Issue: Settings not taking effect
```bash
# Check configuration merge order
docker logs bearish-bot 2>&1 | grep -A 20 "FINAL CONFIGURATION SUMMARY"

# Verify App Config loaded AFTER ENV vars
docker logs bearish-bot 2>&1 | grep -B 5 -A 5 "Loaded 64 settings from App Configuration"
```

---

## 📚 Code References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Config Loader** | `src/config/live_trading_config.py` | 143-175 | Orchestrates merge order |
| **App Config Loader** | `src/config/live_trading_config.py` | 705-790 | REST API implementation |
| **ENV Validation** | `vm_boot.py` | 50-80 | Checks required env vars |
| **Wrapper Script** | `scripts/vm_run_session.py` | 1-150 | Docker run wrapper |
| **Runbook** | `azure_automation/Start-BearishBot-Enhanced.ps1` | 1-200 | Azure automation |

---

## 🎓 Best Practices

### ✅ DO
- Store non-sensitive config in App Configuration (trading params, ML settings, strategy config)
- Use ENV vars for secrets (API keys, tokens)
- Use labels (`production`, `staging`) to separate environments
- Test new settings in `staging` label before applying to `production`
- Monitor App Configuration logs in Azure Portal

### ❌ DON'T
- Store API keys/secrets in App Configuration (use ENV vars or Key Vault)
- Modify YAML file for runtime config changes (use App Config instead)
- Rebuild Docker images for config changes (just update App Config and restart)
- Remove Managed Identity from VM (breaks App Config access)

---

## 📊 Monitoring

### Log Queries (Azure Log Analytics)
```kusto
// Check App Configuration load success
ContainerLog
| where ContainerName == "bearish-bot"
| where LogEntry contains "App Configuration"
| order by TimeGenerated desc
| take 50

// Verify settings count
ContainerLog
| where ContainerName == "bearish-bot"
| where LogEntry contains "Loaded 64 settings"
| order by TimeGenerated desc
| take 10
```

### Health Check
```bash
# On VM
docker logs bearish-bot 2>&1 | head -n 100 | grep -E "(App Config|YAML|ENV|settings)"
```

---

## 🔄 Migration History

| Version | Image Tag | Date | Changes |
|---------|-----------|------|---------|
| **v1** | `appconfig-fix-v1` | Dec 8, 2025 | Initial SDK-based implementation (failed due to IMDS API version) |
| **v2** | `appconfig-rest-api-v1` | Dec 9, 2025 | REST API implementation (failed due to missing curl) |
| **v3** | `appconfig-rest-api-v2` | Dec 9, 2025 | **PRODUCTION** - REST API + curl + enhanced error handling ✅ |

---

## 📞 Support

For issues or questions:
1. Check logs: `docker logs bearish-bot 2>&1 | grep -i "App Config"`
2. Verify env vars: `cat /home/azureuser/bearish-bot.env | grep AZURE_APPCONFIG`
3. Test IMDS: `curl -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2017-12-01&resource=https://appconfig.azure.com"`
4. Review troubleshooting section above

---

**Last Updated**: December 9, 2025  
**Status**: ✅ Production Ready  
**Image**: `bearishalphabot.azurecr.io/bearish-bot:appconfig-rest-api-v2`
