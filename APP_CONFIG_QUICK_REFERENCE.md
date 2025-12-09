# Azure App Configuration - Quick Reference Guide

## ⚡ Quick Start

### What Was Done
✅ All 29 settings migrated from `bearish-bot.env` to Azure cloud:
- **26 settings** → App Configuration (`appcs-bearish-bot`)
- **3 secrets** → Key Vault (`bearish-kv`)

### Configuration Loaded From (Priority)
1. **Azure App Configuration** ← NEW (cloud-based)
2. Environment Variables (legacy)
3. config.example.yaml (defaults)

---

## 🛠️ For Local Development

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment for local testing (optional)
# If you want to test App Config locally, get the endpoint:
export AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io
```

### Testing Config Loading
```bash
# Python
python -c "from src.config import LiveTradingConfiguration; config = LiveTradingConfiguration.load(); print('✓ Config loaded')"

# Or run the bot as normal
python scripts/live_trading_launcher.py
```

---

## 🚀 For Azure Deployment

### Environment Variables Required
Set these on Azure VM or in Docker run command:

```bash
# Required
AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io

# Optional (defaults to 'production')
AZURE_APPCONFIG_LABEL=production
```

### Docker Run Command
```bash
docker run \
  -e AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io \
  -e AZURE_APPCONFIG_LABEL=production \
  bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-13
```

### Remove Old Config
```bash
# Once confident, remove file-based config:
rm /home/azureuser/bearish-bot.env
```

---

## 📊 Settings Location

### Find a Setting in App Configuration
```bash
# List all BearishAlphaBot settings
az appconfig kv list --name appcs-bearish-bot \
  --key-filter "BearishAlphaBot/*"

# Get a specific setting
az appconfig kv show --name appcs-bearish-bot \
  --key "BearishAlphaBot/TRADING_MODE" \
  --label "production"
```

### Find a Secret in Key Vault
```bash
# List all secrets
az keyvault secret list --vault-name bearish-kv

# Get a specific secret
az keyvault secret show --vault-name bearish-kv \
  --name "bingx-key"
```

---

## 🔧 Modify Settings

### Via Azure Portal
1. Go to **App Configuration** → `appcs-bearish-bot`
2. Click **Configuration explorer**
3. Select **BearishAlphaBot/SETTING_NAME**
4. Edit value
5. ✅ Changes take effect next time app loads config

### Via Azure CLI
```bash
# Update a setting
az appconfig kv set \
  --name appcs-bearish-bot \
  --key "BearishAlphaBot/TRADING_MODE" \
  --value "live" \
  --label "production"

# Delete a setting
az appconfig kv delete \
  --name appcs-bearish-bot \
  --key "BearishAlphaBot/TRADING_MODE" \
  --label "production"
```

### Via Python Script
```python
from azure.appconfiguration import AzureAppConfigurationClient, ConfigurationSetting
from azure.identity import DefaultAzureCredential

endpoint = "https://appcs-bearish-bot.azconfig.io"
client = AzureAppConfigurationClient(base_url=endpoint, credential=DefaultAzureCredential())

# Set
setting = ConfigurationSetting(
    key="BearishAlphaBot/TRADING_MODE",
    value="live",
    label="production"
)
client.set_configuration_setting(setting)

# Get
setting = client.get_configuration_setting(
    key="BearishAlphaBot/TRADING_MODE",
    label="production"
)
print(setting.value)
```

---

## 🌍 Environment Labels

Use different labels for different environments:

### Create Settings for Multiple Environments
```bash
# Production
az appconfig kv set --name appcs-bearish-bot \
  --key "BearishAlphaBot/CAPITAL_USDT" --value "10000" \
  --label "production"

# Staging
az appconfig kv set --name appcs-bearish-bot \
  --key "BearishAlphaBot/CAPITAL_USDT" --value "1000" \
  --label "staging"

# Development
az appconfig kv set --name appcs-bearish-bot \
  --key "BearishAlphaBot/CAPITAL_USDT" --value "100" \
  --label "development"
```

### Switch Environment at Runtime
```bash
# In Docker
docker run \
  -e AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io \
  -e AZURE_APPCONFIG_LABEL=staging \  # ← Change this
  bearish-bot:vm-vmboot-13
```

---

## 🔐 Manage Secrets

### Add a New Secret
```bash
# 1. Create secret in Key Vault
az keyvault secret set \
  --vault-name bearish-kv \
  --name "my-new-secret" \
  --value "secret-value"

# 2. Create reference in App Configuration
SECRET_URI=$(az keyvault secret show \
  --vault-name bearish-kv \
  --name "my-new-secret" \
  --query id -o tsv)

az appconfig kv set \
  --name appcs-bearish-bot \
  --key "BearishAlphaBot/MY_NEW_SECRET" \
  --value "@Microsoft.KeyVault(SecretUri=$SECRET_URI)" \
  --content-type "application/vnd.microsoft.appconfig.keyvaultref+json;charset=utf-8"
```

### Rotate a Secret
```bash
# Key Vault handles versioning automatically
az keyvault secret set \
  --vault-name bearish-kv \
  --name "bingx-key" \
  --value "new-api-key-value"
# ✅ App Configuration automatically uses new version
```

---

## 📋 Migration Script

### Re-run Migration (Safe - Idempotent)
```bash
python scripts/migrate_config_to_appconfig.py \
  --env-file bearish-bot.env \
  --app-config-name appcs-bearish-bot \
  --app-config-rg TradeBot \
  --keyvault-name bearish-kv \
  --keyvault-rg tradebot-ops
```

### Dry-run Preview
```bash
python scripts/migrate_config_to_appconfig.py \
  --env-file bearish-bot.env \
  --app-config-name appcs-bearish-bot \
  --app-config-rg TradeBot \
  --keyvault-name bearish-kv \
  --keyvault-rg tradebot-ops \
  --dry-run
```

---

## 📊 Settings List (29 Total)

### Trading Config (5)
- TRADING_MODE → App Config
- TRADING_DURATION → App Config
- EXCHANGES → App Config
- TRADING_SYMBOLS → App Config
- BINGX_REST_DEBUG → App Config

### Capital & Risk (4)
- CAPITAL_USDT → App Config
- PER_TRADE_RISK_PCT → App Config
- DAILY_MAX_TRADES → App Config
- DUPLICATE_PREVENTION_COOLDOWN → App Config

### Thresholds (5)
- DUPLICATE_PREVENTION_THRESHOLD → App Config
- PRICE_DELTA_BYPASS_ENABLED → App Config
- PRICE_DELTA_BYPASS_THRESHOLD → App Config
- RSI_THRESHOLD_BTC → App Config
- RSI_THRESHOLD_ETH → App Config
- RSI_THRESHOLD_SOL → App Config

### ML & Features (4)
- GEMMA_ENABLED → App Config
- ML_ACTIVE_BUNDLE → App Config
- ML_FEAT_VOL_WINDOWS → App Config
- ML_FEAT_MOM_WINDOWS → App Config

### Infrastructure (4)
- WS_MAX_STREAMS_BINGX → App Config
- PYTHONUNBUFFERED → App Config
- PYTHONPATH → App Config
- LOG_LEVEL → App Config

### Monitoring (2)
- DEBUG_MODE → App Config
- ML_ENABLED → App Config
- TELEGRAM_CHAT_ID → App Config

### Secrets (3) 🔐
- BINGX_KEY → Key Vault
- BINGX_SECRET → Key Vault
- TELEGRAM_BOT_TOKEN → Key Vault

---

## 🐛 Troubleshooting

### Config Not Loading from App Configuration
```
Check logs for: "Loaded X settings from App Configuration"

If missing:
1. Set AZURE_APPCONFIG_ENDPOINT env var
2. Verify Managed Identity has "App Configuration Data Reader" role
3. Check network connectivity to Azure
4. Review Azure SDK logs

System gracefully falls back to ENV vars (no crash)
```

### Secret Not Resolving
```
Check logs for: "@Microsoft.KeyVault(...) resolved successfully"

If missing:
1. Verify Key Vault reference format correct
2. Check Managed Identity has "Key Vault Secrets User" role
3. Verify secret exists in Key Vault
4. Check secret names are lowercase with hyphens
```

### Performance
- Config loaded once at startup (Singleton pattern)
- All settings cached in memory
- No per-request API calls
- Minimal overhead vs env vars

---

## ✅ Health Check

### Verify App Configuration is Working
```python
import logging
from src.config import LiveTradingConfiguration

logging.basicConfig(level=logging.INFO)
config = LiveTradingConfiguration.load()

# Look for these log lines:
# ✅ "📡 Loading configuration from Azure App Configuration..."
# ✅ "Loaded 26 settings from App Configuration"
# ✅ Shows endpoint and label used

print(f"Trading mode: {config.get('trading', {}).get('mode')}")
```

---

## 📞 Support

**Issue**: Can't modify settings
- **Solution**: Use Azure Portal or Azure CLI with proper authentication

**Issue**: Changes not taking effect
- **Solution**: Container must reload config (requires restart)

**Issue**: Missing settings
- **Solution**: Check `AZURE_APPCONFIG_LABEL` matches where settings were created

**Issue**: Slow startup
- **Solution**: Check network connectivity to Azure, consider enabling caching

---

**Last Updated**: December 8, 2025  
**Migration Status**: ✅ Complete  
**Next Step**: Deploy vm-vmboot-13 image
