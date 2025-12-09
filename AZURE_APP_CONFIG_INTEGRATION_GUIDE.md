# Azure App Configuration Integration Guide

## 📋 Özet

Azure App Configuration oluşturduktan sonra, mevcut mimarinizde ortam değişkenlerini **3 dosyadan birinden** yürütebilirsiniz:

| Dosya | Konum | Kullanım Durumu | Öncelik |
|-------|------|-----------------|---------|
| **`src/config/live_trading_config.py`** | Container: `/app/src/config/` | ⭐ **ÖNERİLEN** - Dinamik config yükleme | 1️⃣ BİRİNCİ |
| `config/config.example.yaml` | Container: `/app/config/` | Fallback değerler | 2️⃣ İKİNCİ |
| Azure Functions | `/azure_functions/reporting/` | Reporting logic için env vars | Bağımsız |

---

## 🎯 Mevcut Mimariye App Configuration Nasıl Entegre Edilir?

### 1️⃣ **Seçenek A: Live Trading Config'i Güncelle (ÖNERİLEN)**

**Dosya:** `src/config/live_trading_config.py`

Mevcut yapı:
```python
# Priority sırası:
# 1. Environment Variables (Ortam değişkenleri)
# 2. config/config.example.yaml (YAML dosyası)
# 3. Hardcoded defaults (Sabit değerler)
```

**Yeni yapı - Azure App Configuration desteği:**

```python
"""
Güncellenmiş Live Trading Configuration - App Configuration desteği ile
"""

import os
from typing import Optional, Dict, Any
from azure.identity import DefaultAzureCredential
from azure.appconfiguration.provider import load as app_config_load
from azure.appconfiguration.provider import SettingSelector

class LiveTradingConfiguration:
    """Güncellenmiş config loader - Azure App Config desteğiyle"""
    
    CONFIG_FILE_PATH = 'config/config.example.yaml'
    
    @classmethod
    def load(cls, log_summary: bool = True, *, 
             config_path: Optional[str] = None,
             use_app_config: bool = True,
             force_reload: bool = False) -> Dict[str, Any]:
        """
        Güncellenmiş load metodu - Azure App Configuration desteği
        
        Priority (Yüksekten düşüğe):
        1. Azure App Configuration (yeni) - use_app_config=True
        2. Environment Variables - os.getenv('KEY')
        3. config/config.example.yaml - YAML dosyası
        4. Hardcoded defaults - Python kodunda yazılı
        """
        
        config = {}
        
        # 1. ADIM: Azure App Configuration'dan oku (varsa)
        if use_app_config:
            config = cls._load_from_app_config()
            if config:
                logger.info("✅ Loaded config from Azure App Configuration")
        
        # 2. ADIM: YAML dosyasından ve ENV'den oku
        yaml_config = cls._load_yaml_and_merge_with_env(config_path)
        
        # 3. ADIM: Deep merge - App Config > ENV > YAML
        merged = cls._deep_merge(yaml_config, config)
        
        if log_summary:
            cls._log_config_summary(merged)
        
        return merged
    
    @staticmethod
    def _load_from_app_config() -> Dict[str, Any]:
        """
        Azure App Configuration'dan config yükle.
        
        Gereklilikler:
        - AZURE_APPCONFIG_ENDPOINT env var ayarlanmış
        - Managed Identity veya Connection String ile auth
        """
        app_config_endpoint = os.getenv('AZURE_APPCONFIG_ENDPOINT')
        
        if not app_config_endpoint:
            logger.debug("AZURE_APPCONFIG_ENDPOINT not set, skipping App Config")
            return {}
        
        try:
            # Seçenek A: Managed Identity (recommended)
            credential = DefaultAzureCredential()
            
            # Seçenek B: Connection String (fallback)
            # connection_string = os.getenv('AZURE_APPCONFIG_CONNECTION_STRING')
            # config = app_config_load(connection_string=connection_string)
            
            # Load config - "BearishAlphaBot/" prefix'i ile filtreleme
            selectors = [
                SettingSelector(key_filter="BearishAlphaBot/*", label_filter="\0")  # No label
            ]
            
            config_dict = app_config_load(
                endpoint=app_config_endpoint,
                credential=credential,
                selectors=selectors,
                trim_prefixes=["BearishAlphaBot/"]  # Prefix'i kaldır
            )
            
            logger.info(f"✅ Loaded {len(config_dict)} settings from App Config")
            return dict(config_dict)
            
        except Exception as e:
            logger.warning(f"Failed to load from App Config: {e}. Falling back to YAML.")
            return {}
```

### 2️⃣ **Seçenek B: vm_boot.py'ye App Config Desteği Ekle**

**Dosya:** `vm_boot.py`

```python
"""
Güncellenmiş vm_boot.py - Azure App Configuration desteği
"""

import os
from azure.appconfiguration.provider import load as app_config_load

def setup_app_configuration():
    """
    Azure App Configuration'dan env variables yükle.
    Mevcut bearish-bot.env dosyasını override etmez, tamamlar.
    """
    app_config_endpoint = os.getenv('AZURE_APPCONFIG_ENDPOINT')
    
    if not app_config_endpoint:
        log.info("AZURE_APPCONFIG_ENDPOINT not set, using bearish-bot.env")
        return
    
    try:
        log.info("Loading config from Azure App Configuration...")
        
        from azure.identity import DefaultAzureCredential
        from azure.appconfiguration.provider import SettingSelector
        
        credential = DefaultAzureCredential()
        
        # BearishAlphaBot/ altındaki tüm settings'i yükle
        selectors = [
            SettingSelector(key_filter="BearishAlphaBot/*", label_filter="\0")
        ]
        
        config = app_config_load(
            endpoint=app_config_endpoint,
            credential=credential,
            selectors=selectors,
            trim_prefixes=["BearishAlphaBot/"]
        )
        
        # Yüklenen config'i ortam değişkenlerine aktar
        for key, value in config.items():
            # Sadece belirlenen variables'ları set et
            env_key = _config_key_to_env_var(key)
            if env_key and not os.getenv(env_key):  # Mevcut ENV değerleri override etme
                os.environ[env_key] = str(value)
                log.info(f"Set {env_key} from App Config")
        
        log.info(f"✅ Loaded {len(config)} settings from App Config")
        
    except Exception as e:
        log.warning(f"Failed to load App Config: {e}")
        log.info("Continuing with bearish-bot.env variables...")

def _config_key_to_env_var(key: str) -> str:
    """
    App Config key'i ENV variable adına dönüştür.
    
    Örnek: "trading_mode" -> "TRADING_MODE"
    """
    return key.upper().replace('/', '_')

def main() -> int:
    log.info("========================================")
    log.info("Bearish Alpha Bot - VM Boot (v2 - App Config)")
    log.info("========================================")
    
    setup_environment()
    ensure_directories()
    setup_default_manifest()
    setup_ml_environment()
    
    # YENİ: App Configuration desteği
    setup_app_configuration()
    
    # Mevcut kod devam et...
    mode_args = build_mode_args()
    cmd = [sys.executable, "scripts/live_trading_launcher.py", *mode_args]
    
    ...
```

### 3️⃣ **Seçenek C: Azure Functions'ta App Config Kullan**

**Dosya:** `azure_functions/reporting/function_app_runtime.py`

```python
"""
Güncellenmiş Azure Functions - App Configuration desteği
"""

import os
from azure.identity import DefaultAzureCredential
from azure.appconfiguration.provider import load as app_config_load

# Startup'ta App Config'i yükle
def load_app_config():
    """Azure App Configuration'dan settings yükle"""
    
    app_config_endpoint = os.getenv('AZURE_APPCONFIG_ENDPOINT')
    
    if app_config_endpoint:
        try:
            credential = DefaultAzureCredential()
            
            config = app_config_load(
                endpoint=app_config_endpoint,
                credential=credential,
                feature_flags_enabled=True  # Feature flags desteği
            )
            
            # Önemli settings'i ortam değişkenlerine aktar
            os.environ.setdefault('TRADING_MODE', 
                                 str(config.get('trading_mode', 'paper')))
            os.environ.setdefault('DEBUG_MODE', 
                                 str(config.get('debug_mode', 'false')))
            
            return config
        except Exception as e:
            LOGGER.warning(f"Failed to load App Config: {e}")
    
    return {}

# Function startup
app_config = load_app_config()
```

---

## 🔧 Azure App Configuration Kurulumu

### Adım 1: App Configuration Store Oluştur

```bash
# Azure Portal'da veya CLI ile
az appconfig create \
  --name bearish-app-config \
  --resource-group BearishAlphaBot-RG \
  --location eastus \
  --sku free
```

### Adım 2: Key-Value'ları Ekle

```bash
# Trading Mode
az appconfig kv set \
  --name bearish-app-config \
  --key "BearishAlphaBot/trading_mode" \
  --value "paper" \
  --label "production"

# Debug Mode
az appconfig kv set \
  --name bearish-app-config \
  --key "BearishAlphaBot/debug_mode" \
  --value "false" \
  --label "production"

# Leverage Default
az appconfig kv set \
  --name bearish-app-config \
  --key "BearishAlphaBot/leverage_default" \
  --value "5" \
  --label "production"

# Trading Symbols (JSON)
az appconfig kv set \
  --name bearish-app-config \
  --key "BearishAlphaBot/trading_symbols" \
  --value '["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]' \
  --content-type "application/json" \
  --label "production"
```

### Adım 3: Managed Identity Permissions Ekle

```bash
# Azure VM Managed Identity'ye App Config Data Reader role ver
MANAGED_IDENTITY_ID=$(az vm identity show \
  --resource-group BearishAlphaBot-RG \
  --name BearishAlphaBot-VM-01 \
  --query principalId -o tsv)

az role assignment create \
  --assignee $MANAGED_IDENTITY_ID \
  --role "App Configuration Data Reader" \
  --scope "/subscriptions/{subscription-id}/resourceGroups/BearishAlphaBot-RG/providers/Microsoft.AppConfiguration/configurationStores/bearish-app-config"
```

### Adım 4: Docker'a Endpoint'i Geç

```bash
# Logic App trigger body'sine ekle
{
  "imageTag": "vm-vmboot-12",
  "durationMinutes": 60,
  "appConfigEndpoint": "https://bearish-app-config.azconfig.io"
}

# Runbook'ta ortam değişkeni olarak ayarla
docker run \
  -e AZURE_APPCONFIG_ENDPOINT="https://bearish-app-config.azconfig.io" \
  --env-file /home/azureuser/bearish-bot.env \
  bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-12
```

---

## 📊 Priority Sırası (Güncellenmiş)

```
1. Azure App Configuration (varsa)     ← EN YÜKSEKTadır
   ↓
2. Ortam Değişkenleri (bearish-bot.env)
   ↓
3. config/config.example.yaml (YAML)
   ↓
4. Hardcoded Python Defaults           ← EN DÜŞÜĞÜ
```

---

## 🔄 Entegrasyon Akışı

### Seçenek A: Config Loader Güncellemesi (Önerilen)

```
Docker Container Starts
    ↓
vm_boot.py:
    1. setup_environment()
    2. ensure_directories()
    3. setup_default_manifest()
    4. setup_ml_environment()
    ↓
scripts/live_trading_launcher.py
    ↓
LiveTradingConfiguration.load():
    1. Check AZURE_APPCONFIG_ENDPOINT
    2. Load from App Configuration (yeni)
    3. Load from YAML + ENV
    4. Deep merge: AppConfig > ENV > YAML
    5. Return final config
    ↓
ProductionCoordinator:
    Uses merged config
```

### Seçenek B: vm_boot.py Güncellemesi

```
Docker Container Starts
    ↓
vm_boot.py:
    1. setup_environment()
    2. ensure_directories()
    3. setup_app_configuration() [YENİ]
       → Load App Config
       → Set OS environ
    4. setup_default_manifest()
    5. setup_ml_environment()
    ↓
bearish-bot.env (bearish-bot.env + App Config merged)
    ↓
scripts/live_trading_launcher.py
    ↓
LiveTradingConfiguration.load()
    (Mevcut logic devam ediyor - App Config değerleri ENV'de)
```

---

## 💡 Avantajlar

### Mevcut Sistem (Seçenek D: Devam Et)
- ✅ Basit, tested
- ❌ File-based management
- ❌ Version control riski (secrets)

### App Configuration (Önerilen)
- ✅ **Centralized management** (Azure Portal'da)
- ✅ **Feature flags** desteği
- ✅ **Revision history** (audit trail)
- ✅ **Geo-replication** (high availability)
- ✅ **Real-time notifications** (config change)
- ✅ **RBAC** (access control)
- ✅ **Key Vault integration** (secrets)
- ✅ **Dynamic refresh** (app restart olmadan)
- ✅ **No version control secrets**

---

## 🚀 Implementasyon Aşamaları

### Faz 1: Setup
1. Azure App Configuration store oluştur
2. Key-value'ları import et
3. Managed Identity permissions konfigüre et
4. Docker image'da AZURE_APPCONFIG_ENDPOINT ayarla

### Faz 2: Integration
1. `src/config/live_trading_config.py` güncelle
   - OR `vm_boot.py` güncelle
2. Requirement'lara `azure-appconfiguration-provider` ekle
3. Test et (dev label'ı ile)

### Faz 3: Production Rollout
1. Production label'ı ile settings'ler ekle
2. Blue-green deployment yap
3. vm-vmboot-13 image'i build et
4. Logic App trigger'ı güncelle

### Faz 4: Monitoring
1. App Config revision history izle
2. Feature flags dashboards'u kur
3. Config change notifications al

---

## 📝 Required Packages

```txt
# requirements.txt'e ekle
azure-appconfiguration-provider>=2.3.1
azure-identity>=1.15.0
```

---

## 🔐 Best Practices

1. **Managed Identity Kullan** ✅
   - Connection string değil
   - Secrets rotasyonu otomatik

2. **Labels ile Environments Ayırt Et** ✅
   - `production`, `staging`, `dev`
   - Per-label configuration

3. **Sensitive Data için Key Vault** ✅
   ```
   BearishAlphaBot/bingx_api_key = @Microsoft.KeyVault(SecretUri=https://bearish-kv.vault.azure.net/secrets/bingx-key/)
   ```

4. **Feature Flags Kullan** ✅
   ```python
   config = app_config_load(..., feature_flags_enabled=True)
   if config['feature_management']['feature_flags']['NewStrategy']['enabled']:
       # Yeni strateji kodu
   ```

5. **Dynamic Refresh** ✅
   ```python
   config = app_config_load(..., refresh_interval=30)
   # 30 saniyede bir App Config'i kontrol et
   ```

---

## 🔗 İlgili Dosyalar

- `src/config/live_trading_config.py` - Config loader
- `vm_boot.py` - Container entry point
- `scripts/live_trading_launcher.py` - Launcher
- `config/config.example.yaml` - Default values
- `requirements.txt` - Dependencies

---

## 📚 Microsoft Kaynakları

- [Azure App Configuration Overview](https://learn.microsoft.com/en-us/azure/azure-app-configuration/overview)
- [Python Configuration Provider](https://learn.microsoft.com/en-us/azure/azure-app-configuration/reference-python-provider)
- [Dynamic Configuration](https://learn.microsoft.com/en-us/azure/azure-app-configuration/enable-dynamic-configuration-python)
- [Feature Flags](https://learn.microsoft.com/en-us/azure/azure-app-configuration/quickstart-feature-flag-python)
