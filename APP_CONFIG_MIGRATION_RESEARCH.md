# Azure App Configuration Migration - Comprehensive Research Guide

**Date**: December 8, 2025  
**Status**: Analysis Complete - Ready for Implementation  
**Focus**: Migration Strategy, Technical Patterns, and Best Practices

---

## 1. Current State vs Target State

### Current Architecture (bearish-bot.env)
```
Config Loading Flow:
┌─────────────────────────────────────────┐
│ src/config/live_trading_config.py       │
│ (LiveTradingConfiguration - Singleton) │
└────────────────────┬────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   bearish-bot.env        config.example.yaml
   (27 overrides)         (637 lines - defaults)
         │                       │
         └───────────┬───────────┘
                     ▼
         Environment Variables
         (Runtime values)
```

**Priority Order (Current):**
1. ENV Variables (bearish-bot.env via docker --env-file)
2. config.example.yaml defaults
3. Hardcoded Python defaults

### Target Architecture (Azure App Configuration)
```
Config Loading Flow:
┌─────────────────────────────────────────┐
│ src/config/live_trading_config.py       │
│ (Enhanced with AppConfig support)      │
└────────────────────┬────────────────────┘
                     │
         ┌───────────┴────────────────────┬─────────────┐
         ▼                                ▼             ▼
   Azure App Config         Key Vault        config.example.yaml
   (27 settings)           (3 secrets)       (637 lines - defaults)
         │                      │                   │
         └──────────────┬───────┴───────────┬────────┘
                        ▼
            Cloud-based Override Layer
```

**Priority Order (Target):**
1. Azure App Configuration (cloud-based overrides)
2. Azure Key Vault (sensitive values via references)
3. config.example.yaml defaults
4. Hardcoded Python defaults

---

## 2. Migration Strategy Analysis

### Option A: File-Based Bulk Import + Manual Setup
**Approach**: Export bearish-bot.env → Import to App Config via Azure Portal/CLI

**Pros:**
- ✅ Simple, straightforward process
- ✅ No script dependencies
- ✅ Good for one-time migration

**Cons:**
- ❌ Manual effort for 27 settings
- ❌ No version control for import
- ❌ Secrets exposed in process
- ❌ Not replicable across environments

**Timeline**: 30-45 minutes

---

### Option B: Python Migration Script (RECOMMENDED ✓)
**Approach**: Create `scripts/migrate_config_to_appconfig.py` that:
1. Reads bearish-bot.env
2. Identifies sensitive values (credentials)
3. Creates Key Vault secrets
4. Imports non-sensitive to App Config
5. Creates Key Vault references

**Pros:**
- ✅ Fully automated, reproducible
- ✅ Secure secret handling
- ✅ Version-controlled process
- ✅ Can be re-run safely (idempotent)
- ✅ Creates audit trail
- ✅ Works for multiple environments

**Cons:**
- 🟡 Requires script development (~2 hours)
- 🟡 Needs proper error handling

**Timeline**: 2-3 hours (including testing)

**Recommended**: YES - Worth the investment for maintainability

---

### Option C: Hybrid Approach
**Approach**: 
- Phase 1: Manual import via Portal for quick validation
- Phase 2: Build Python script for future migrations

**Use Case**: When you want to validate App Config quickly before investing in automation

**Timeline**: Phase 1: 45 min, Phase 2: 2-3 hours later

---

## 3. Technical Implementation Details

### 3.1 Python SDK Options

#### Option 1: `azure-appconfiguration` (Client Library)
**Best for**: Direct CRUD operations on individual settings

```python
from azure.appconfiguration import AzureAppConfigurationClient, ConfigurationSetting
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
endpoint = os.getenv('AZURE_APPCONFIG_ENDPOINT')  # e.g., https://bearish-app-config.azconfig.io
client = AzureAppConfigurationClient(base_url=endpoint, credential=credential)

# Set individual setting
setting = ConfigurationSetting(
    key='BearishAlphaBot/TRADING_MODE',
    value='paper',
    label='production',
    content_type='text/plain'
)
client.add_configuration_setting(setting)

# Get individual setting
retrieved = client.get_configuration_setting(key='BearishAlphaBot/TRADING_MODE', label='production')
print(retrieved.value)  # 'paper'

# List all settings
settings = client.list_configuration_settings(key_filter='BearishAlphaBot/*')
for setting in settings:
    print(f"{setting.key}={setting.value}")
```

**Use Case**: Migration scripts, CRUD operations  
**Advantage**: Full control over each setting, supports labels, read-only locks  
**Limitation**: Must loop for bulk operations

---

#### Option 2: `azure-appconfiguration-provider` (Data Provider)
**Best for**: Loading all settings at once into Python dict-like object

```python
from azure.appconfiguration.provider import load, SettingSelector
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
endpoint = os.getenv('AZURE_APPCONFIG_ENDPOINT')

# Load all settings with prefix trimming
config = load(
    endpoint=endpoint,
    credential=credential,
    trim_prefixes={'BearishAlphaBot/'}  # Removes prefix from keys
)

# Access like dictionary
print(config['TRADING_MODE'])  # Value from BearishAlphaBot/TRADING_MODE

# Or access raw with prefix
print(config.get('BearishAlphaBot/TRADING_MODE'))

# Supports SettingSelector for filtering
selects = {SettingSelector(key_filter='BearishAlphaBot/*', label_filter='production')}
config = load(endpoint=endpoint, credential=credential, selects=selects)
```

**Use Case**: Application runtime config loading  
**Advantage**: Dictionary-like interface, prefix trimming, automatic env var interpolation  
**Limitation**: All-or-nothing load, less control for individual operations

---

### 3.2 Key Vault Integration Pattern

**For Sensitive Data**: Use Key Vault references in App Configuration

```yaml
# In App Configuration, reference Key Vault secrets:
BearishAlphaBot/BINGX_KEY: @Microsoft.KeyVault(SecretUri=https://bearish-keyvault.vault.azure.net/secrets/bingx-key/version123)
BearishAlphaBot/BINGX_SECRET: @Microsoft.KeyVault(SecretUri=https://bearish-keyvault.vault.azure.net/secrets/bingx-secret/version456)
BearishAlphaBot/TELEGRAM_BOT_TOKEN: @Microsoft.KeyVault(SecretUri=https://bearish-keyvault.vault.azure.net/secrets/telegram-token/version789)
```

**Benefits:**
- ✅ Secrets never stored in App Config
- ✅ Centralized secret management in Key Vault
- ✅ Audit trail for secret access
- ✅ Automatic secret rotation support
- ✅ Access control via RBAC

**Resolution Pattern** (automatic by SDK):
```python
# When you call client.get_configuration_setting(), the SDK:
# 1. Sees @Microsoft.KeyVault(...) pattern
# 2. Automatically fetches from Key Vault
# 3. Returns resolved secret value
# 4. Application code sees actual secret (transparent)
```

---

## 4. Detailed Migration Plan (Option B - Recommended)

### Phase 1: Pre-Migration Setup (15 minutes)

**Tasks:**
1. Create Azure Key Vault (if not exists)
2. Create Azure App Configuration store
3. Configure Managed Identity with permissions
4. Prepare environment variables

**Azure CLI Commands:**
```bash
# 1. Create Key Vault (skip if exists)
az keyvault create \
  --resource-group TradeBot \
  --name bearish-keyvault \
  --location eastus

# 2. Create App Configuration store
az appconfig create \
  --resource-group TradeBot \
  --name bearish-app-config \
  --location eastus \
  --sku free

# 3. Create Managed Identity for VM
az identity create \
  --resource-group TradeBot \
  --name bearish-bot-msi

# 4. Grant permissions to Key Vault
az role assignment create \
  --role "Key Vault Secrets User" \
  --assignee /subscriptions/{sub-id}/resourcegroups/TradeBot/providers/Microsoft.ManagedIdentity/userAssignedIdentities/bearish-bot-msi \
  --scope /subscriptions/{sub-id}/resourcegroups/TradeBot/providers/Microsoft.KeyVault/vaults/bearish-keyvault

# 5. Grant permissions to App Config
az role assignment create \
  --role "App Configuration Data Reader" \
  --assignee /subscriptions/{sub-id}/resourcegroups/TradeBot/providers/Microsoft.ManagedIdentity/userAssignedIdentities/bearish-bot-msi \
  --scope /subscriptions/{sub-id}/resourcegroups/TradeBot/providers/Microsoft.AppConfiguration/configurationStores/bearish-app-config
```

---

### Phase 2: Create Migration Script (1-1.5 hours)

**File**: `scripts/migrate_config_to_appconfig.py`

**Responsibilities:**
1. Parse bearish-bot.env
2. Categorize as sensitive/non-sensitive
3. Create Key Vault secrets
4. Create App Config references

**Key Features:**
- Idempotent (safe to run multiple times)
- Dry-run mode (preview without applying)
- Color-coded output
- Error recovery

**Structure:**
```python
class ConfigMigrator:
    def __init__(self, env_file, app_config_name, keyvault_name, prefix='BearishAlphaBot/'):
        self.env_data = self.load_env(env_file)
        self.appconfig_client = self.get_appconfig_client(app_config_name)
        self.keyvault_client = self.get_keyvault_client(keyvault_name)
        self.prefix = prefix
    
    def categorize_settings(self):
        """Separate sensitive from non-sensitive settings"""
        # Sensitive: BINGX_KEY, BINGX_SECRET, TELEGRAM_BOT_TOKEN
        # Non-sensitive: Everything else
    
    def create_keyvault_secrets(self, dry_run=False):
        """Create secrets in Key Vault for sensitive values"""
        # For each sensitive setting, create secret
        # Return URIs for reference
    
    def create_appconfig_references(self, secret_uris, dry_run=False):
        """Create App Config entries with Key Vault references"""
        # For each sensitive setting, create reference: @Microsoft.KeyVault(...)
        # For non-sensitive, create direct value
    
    def set_labels(self, label='production'):
        """Apply labels for environment separation"""
        # Example: production, staging, development
    
    def migrate(self, dry_run=False):
        """Execute full migration"""
        secrets = self.create_keyvault_secrets(dry_run)
        self.create_appconfig_references(secrets, dry_run)
        self.set_labels()
        return migration_summary
```

---

### Phase 3: Execute Migration (10 minutes)

**Command:**
```bash
# Dry-run first (no changes)
python scripts/migrate_config_to_appconfig.py \
  --env-file /home/azureuser/bearish-bot.env \
  --app-config-name bearish-app-config \
  --keyvault-name bearish-keyvault \
  --dry-run

# Actual execution
python scripts/migrate_config_to_appconfig.py \
  --env-file /home/azureuser/bearish-bot.env \
  --app-config-name bearish-app-config \
  --keyvault-name bearish-keyvault
```

---

### Phase 4: Update Application Code (1-2 hours)

**File**: `src/config/live_trading_config.py`

**Changes Required:**

1. **Add new method `_load_from_app_config()`**:
```python
def _load_from_app_config(self) -> dict:
    """Load configuration from Azure App Configuration"""
    endpoint = os.getenv('AZURE_APPCONFIG_ENDPOINT')
    if not endpoint:
        return {}
    
    from azure.appconfiguration.provider import load, SettingSelector
    from azure.identity import DefaultAzureCredential
    
    credential = DefaultAzureCredential()
    
    # Load with prefix trimming
    config = load(
        endpoint=endpoint,
        credential=credential,
        trim_prefixes={'BearishAlphaBot/'},
        selectors={SettingSelector(key_filter='BearishAlphaBot/*')}
    )
    
    return dict(config)
```

2. **Update priority in `_load_and_merge_configs()`**:
```python
# Current (OLD)
env_vars = self._get_env_overrides()
yaml_config = self._load_yaml_and_map_env_vars()
config = self._deep_merge(yaml_config, env_vars)

# New (UPDATED)
env_vars = self._get_env_overrides()
app_config = self._load_from_app_config()  # NEW
yaml_config = self._load_yaml_and_map_env_vars()

# Priority: App Config > ENV Vars > YAML
config = self._deep_merge(yaml_config, env_vars)
config = self._deep_merge(config, app_config)  # Override with App Config
```

---

### Phase 5: Build & Deploy (30-45 minutes)

**Steps:**
1. Update `requirements.txt`:
```
azure-appconfiguration-provider>=2.3.1
azure-identity>=1.15.0
```

2. Update Docker run command:
```bash
# OLD
docker run --env-file /home/azureuser/bearish-bot.env bearish-bot:vm-vmboot-12

# NEW
docker run \
  -e AZURE_APPCONFIG_ENDPOINT=https://bearish-app-config.azconfig.io \
  bearish-bot:vm-vmboot-13
```

3. Build new image:
```bash
docker build -t bearish-bot:vm-vmboot-13 .
docker push bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-13
```

---

## 5. Sensitive Data Handling

### Settings to Migrate to Key Vault

From `bearish-bot.env` (27 settings):

| Setting | Category | Migration | Notes |
|---------|----------|-----------|-------|
| BINGX_KEY | 🔒 Sensitive | Key Vault Secret | API credential |
| BINGX_SECRET | 🔒 Sensitive | Key Vault Secret | API credential |
| TELEGRAM_BOT_TOKEN | 🔒 Sensitive | Key Vault Secret | Bot token |
| TRADING_MODE | 📋 Config | Direct Value | paper/live |
| DEBUG_MODE | 📋 Config | Direct Value | true/false |
| ML_ENABLED | 📋 Config | Direct Value | true/false |
| EXCHANGES | 📋 Config | Direct Value | bingx |
| TRADING_DURATION | 📋 Config | Direct Value | seconds |
| TELEGRAM_CHAT_ID | 📋 Config | Direct Value | numeric ID |
| CAPITAL_USDT | 📋 Config | Direct Value | numeric |
| PER_TRADE_RISK_PCT | 📋 Config | Direct Value | decimal |
| DAILY_MAX_TRADES | 📋 Config | Direct Value | integer |
| DUPLICATE_PREVENTION_THRESHOLD | 📋 Config | Direct Value | decimal |
| DUPLICATE_PREVENTION_COOLDOWN | 📋 Config | Direct Value | seconds |
| TRADING_SYMBOLS | 📋 Config | Direct Value | BTC/USDT:USDT |
| RSI_THRESHOLD_* | 📋 Config | Direct Value | integer thresholds |
| GEMMA_ENABLED | 📋 Config | Direct Value | true/false |
| ML_ACTIVE_BUNDLE | 📋 Config | Direct Value | path |
| ML_FEAT_VOL_WINDOWS | 📋 Config | Direct Value | window sizes |
| ML_FEAT_MOM_WINDOWS | 📋 Config | Direct Value | window sizes |
| WS_MAX_STREAMS_BINGX | 📋 Config | Direct Value | integer |
| PRICE_DELTA_BYPASS_ENABLED | 📋 Config | Direct Value | true/false |
| PRICE_DELTA_BYPASS_THRESHOLD | 📋 Config | Direct Value | decimal |
| PYTHONUNBUFFERED | 📋 Config | Direct Value | 1 |
| LOG_LEVEL | 📋 Config | Direct Value | INFO/DEBUG |

**Total Secrets**: 3  
**Total Config Values**: 24

---

## 6. Labels Strategy (Environment Separation)

**Recommended Label Structure**:

```
BearishAlphaBot/TRADING_MODE
├─ label: production    # Live trading settings
├─ label: staging       # UAT settings
└─ label: development   # Dev settings

BearishAlphaBot/CAPITAL_USDT
├─ label: production    → 10000  (live)
├─ label: staging       → 1000   (uat)
└─ label: development   → 100    (dev)
```

**Benefits:**
- ✅ Single App Config store for all environments
- ✅ Easily switch between prod/staging/dev
- ✅ Unified audit trail
- ✅ Easy to compare settings across environments
- ✅ Reduce human error

**Implementation in Code**:
```python
def load_config_for_environment(self, env='production'):
    config = load(
        endpoint=endpoint,
        credential=credential,
        trim_prefixes={'BearishAlphaBot/'},
        selectors={SettingSelector(
            key_filter='BearishAlphaBot/*',
            label_filter=env  # 'production', 'staging', 'development'
        )}
    )
    return dict(config)
```

---

## 7. Best Practices from Microsoft Documentation

### 7.1 Authentication & Authorization
✅ **Use Managed Identity** (DefaultAzureCredential)
- No credentials to manage
- Automatic token refresh
- RBAC-based access control

❌ **Avoid**: Connection strings in code
❌ **Avoid**: Hardcoded credentials

### 7.2 Performance Optimization
✅ **Load config once at startup** (Singleton pattern - already doing this)
✅ **Cache settings in memory**
✅ **Use labels efficiently** (filter by label to reduce payload)

❌ **Avoid**: Calling get_configuration_setting() in hot paths
❌ **Avoid**: Loading all settings if only need few

### 7.3 Resilience & Reliability
✅ **Implement retry logic** with exponential backoff
✅ **Have local fallback config** (config.example.yaml)
✅ **Monitor and log config load failures**
✅ **Test graceful degradation**

❌ **Avoid**: Failing fast on config load errors
❌ **Avoid**: No fallback mechanism

### 7.4 Security Best Practices
✅ **Separate secrets into Key Vault**
✅ **Use references** (@Microsoft.KeyVault())
✅ **Implement RBAC** for both App Config and Key Vault
✅ **Enable audit logging** for sensitive operations
✅ **Rotate secrets regularly**

❌ **Avoid**: Storing secrets in App Configuration directly
❌ **Avoid**: Granting excessive permissions

### 7.5 Bulk Operations
✅ **For bulk import**: Create migration script (Option B)
✅ **For bulk export**: Use Azure CLI or Python SDK
✅ **Version control** the migration logic

❌ **Avoid**: Manual entry of all 27 settings
❌ **Avoid**: One-off scripts without error handling

---

## 8. Validation Checklist

### Pre-Migration
- [ ] Azure Key Vault created and accessible
- [ ] Azure App Configuration store created
- [ ] Managed Identity configured with proper RBAC
- [ ] bearish-bot.env backed up and validated
- [ ] Migration script tested in dry-run mode

### During Migration
- [ ] All 3 secrets created in Key Vault
- [ ] All 24 config values in App Configuration
- [ ] Labels applied correctly
- [ ] Sensitive values NOT visible in App Config
- [ ] Key Vault references properly formatted

### Post-Migration
- [ ] Application code updated and tested
- [ ] ENV variable not passed to Docker container
- [ ] AZURE_APPCONFIG_ENDPOINT passed correctly
- [ ] Config loaded successfully at startup
- [ ] All 27 settings accessible via new path
- [ ] bearish-bot.env no longer used/referenced
- [ ] New image tagged vm-vmboot-13

---

## 9. Rollback Plan

If issues occur:

### Quick Rollback (< 5 minutes)
```bash
# Keep old container running
docker run --env-file /home/azureuser/bearish-bot.env bearish-bot:vm-vmboot-12
```

### Full Rollback (< 30 minutes)
1. Stop vm-vmboot-13 container
2. Revert to vm-vmboot-12
3. Investigate failures
4. Fix issues
5. Test locally
6. Re-attempt migration

### Safe Points
- ✓ Before Phase 1: No changes made
- ✓ After Phase 1: Azure resources created, no config migrated
- ✓ After Phase 2: Script ready, bearish-bot.env still used
- ✓ After Phase 3: Config migrated, app code not updated yet
- ✓ After Phase 4: App code updated, can test locally
- ✓ After Phase 5: Production deployment

---

## 10. Code Samples from Microsoft Learn

### Loading with Provider Library (Recommended for Application)
```python
from azure.appconfiguration.provider import load, SettingSelector
from azure.identity import DefaultAzureCredential
import os

endpoint = os.environ.get('AZURE_APPCONFIG_ENDPOINT')
credential = DefaultAzureCredential()

# Load all BearishAlphaBot settings for production
config = load(
    endpoint=endpoint,
    credential=credential,
    trim_prefixes={'BearishAlphaBot/'},
    selectors={
        SettingSelector(
            key_filter='BearishAlphaBot/*',
            label_filter='production'
        )
    }
)

# Access settings like a dictionary
trading_mode = config.get('TRADING_MODE')
capital = config.get('CAPITAL_USDT')
```

### Direct CRUD Operations (Recommended for Migration Script)
```python
from azure.appconfiguration import AzureAppConfigurationClient, ConfigurationSetting
from azure.identity import DefaultAzureCredential
import os

endpoint = os.environ.get('AZURE_APPCONFIG_ENDPOINT')
credential = DefaultAzureCredential()
client = AzureAppConfigurationClient(base_url=endpoint, credential=credential)

# Add a setting
setting = ConfigurationSetting(
    key='BearishAlphaBot/TRADING_MODE',
    value='paper',
    label='production',
    content_type='text/plain',
    tags={'environment': 'production', 'service': 'bearish-bot'}
)
client.set_configuration_setting(setting)

# Get all settings with filter
settings = client.list_configuration_settings(
    key_filter='BearishAlphaBot/*',
    label_filter='production'
)
for setting in settings:
    print(f"{setting.key}={setting.value} (label={setting.label})")
```

---

## 11. Summary & Recommendations

### ✅ Recommended Approach: Option B (Python Migration Script)
**Timeline**: 4-5 hours total  
**Effort**: Medium (requires script development)  
**Benefits**: Fully automated, reproducible, secure, maintainable

### Phase Breakdown:
1. **Pre-Migration Setup** (15 min) - Create Azure resources
2. **Migration Script** (1.5-2 hours) - Build automation
3. **Execute Migration** (10 min) - Run migration with validation
4. **Update Application** (1-2 hours) - Enhance LiveTradingConfiguration
5. **Build & Deploy** (30-45 min) - New Docker image

### Key Success Factors:
- ✓ Automated migration (no manual entry of 27 settings)
- ✓ Proper secret handling (Key Vault, not App Config)
- ✓ Comprehensive error handling
- ✓ Dry-run validation before actual migration
- ✓ Proper labels for environment separation
- ✓ Full RBAC configuration
- ✓ Rollback plan in place

### Expected Outcomes:
- ✅ Centralized cloud-based configuration
- ✅ Improved security (secrets in Key Vault)
- ✅ Easy environment management (labels)
- ✅ Audit trail for all config changes
- ✅ Scalable for future environments/services
- ✅ Version-controlled migration process

---

## 12. References & Documentation

### Official Microsoft Learn
- [Azure App Configuration Overview](https://learn.microsoft.com/en-us/azure/azure-app-configuration/overview)
- [App Configuration Python SDK](https://learn.microsoft.com/en-us/azure/azure-app-configuration/quickstart-python)
- [App Configuration Provider for Python](https://learn.microsoft.com/en-us/azure/azure-app-configuration/quickstart-python-provider)
- [Key Vault References](https://learn.microsoft.com/en-us/azure/azure-app-configuration/use-key-vault-references-python)
- [Azure SDK Best Practices](https://learn.microsoft.com/en-us/azure/sdk/python/)

### Key NuGet/PyPI Packages
- `azure-appconfiguration>=1.4.0` - Client library for CRUD
- `azure-appconfiguration-provider>=2.3.1` - Provider for application loading
- `azure-identity>=1.15.0` - DefaultAzureCredential
- `azure-keyvault-secrets>=4.4.0` - Key Vault secrets client

### Code Patterns Used
- Singleton pattern (already in LiveTradingConfiguration)
- Deep merge for config layering (already implemented)
- Managed Identity for authentication (recommended for production)
- Labels for environment separation (Azure best practice)
- Key Vault references for secrets (Azure security best practice)

---

**Status**: Ready for Phase 1 Implementation ✓  
**Next Step**: Run Pre-Migration Setup (Phase 1 - 15 minutes)
