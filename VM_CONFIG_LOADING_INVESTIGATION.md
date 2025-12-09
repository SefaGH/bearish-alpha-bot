# vm-vmboot-12 İmajında Config Ayarlarının Okunması

## 📋 Özet (Executive Summary)

**vm-vmboot-12** imajında konfigürasyon ayarları aşağıdaki dosyadan ve öncelikle okunmaktadır:

| Dosya | Konum | Amaç | Varsayılan Yol |
|-------|------|------|-----------------|
| **`config/config.example.yaml`** | Container: `/app/config/config.example.yaml` | Trading parametreleri, semboller, risk limitleri | ✅ **ANA KAYNAK** |
| `config/config.debug.yaml` | Container: `/app/config/config.debug.yaml` | Test optimizasyonu için alternatif config | İsteğe bağlı |
| Ortam Değişkenleri (ENV) | Docker runtime | Config dosyasını geçersiz kılmak için | İsteğe bağlı |

---

## 🔄 Konfigürasyon Yükleme Sıraması (Priority Order)

### Yüksekten Düşüğe Öncelik:

```
1. ORTAM DEĞİŞKENLERİ (Environment Variables) - En Yüksek Öncelik
   ↓ (Eğer tanımlanmışsa bu değerleri kullan)
2. config/config.example.yaml (YAML Dosyası)
   ↓ (Eğer ortam değişkeni yoksa)
3. Hardcoded Python Defaults (Kodda yazılı sabit değerler)
   ↓ (En son çare)
```

### Örnek:
```yaml
execution:
  enable_live: true              # Override with: ENABLE_LIVE_TRADING
  order_type: market             # Override with: ORDER_TYPE
```

Eğer `ENABLE_LIVE_TRADING` ortam değişkeni ayarlanmışsa, YAML dosyasındaki `execution.enable_live` değeri geçersiz kılınır.

---

## 📂 Dosya Yapısı

### 1. **Ana Config Dosyası: `config/config.example.yaml`**

**Konum:** 
- Host: `c:\Users\sefaa\bearish-alpha-bot\config\config.example.yaml`
- Container: `/app/config/config.example.yaml` 

**Boyut:** 637 satır  
**Format:** YAML

**İçerik Örneği:**
```yaml
execution:
  enable_live: true              # Override with: ENABLE_LIVE_TRADING
  order_type: market             # Override with: ORDER_TYPE
  fee_pct: 0.0006                # Override with: FEE_PCT
  leverage:
    default: 5                   # Override with: LEVERAGE_DEFAULT

risk_management:
  max_position_size: 0.1         # Override with: MAX_POSITION_SIZE
  stop_loss_pct: 0.02            # Override with: STOP_LOSS_PCT
```

---

## 🔧 Config Yükleme Mekanizması

### Dosya Hiyerarşisi:

```
Docker Image: bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-12
    ↓
Dockerfile CMD: ["python", "vm_boot.py"]
    ↓
vm_boot.py:
    1. setup_environment() - PYTHONPATH ayarla
    2. ensure_directories() - logs/, data/, artifacts/ oluştur
    3. subprocess.call(['python', 'scripts/live_trading_launcher.py', ...])
    ↓
scripts/live_trading_launcher.py:
    from config.live_trading_config import LiveTradingConfiguration
    config = LiveTradingConfiguration.load()
    ↓
src/config/live_trading_config.py (Ana Config Loader):
    _load_and_merge_configs():
        1. Load YAML: config/config.example.yaml
        2. Parse env var mappings from comments
        3. Get overrides from environment variables
        4. Deep merge YAML + ENV overrides
        5. Return final config dict
```

---

## 🏗️ Config Yükleme Detayları

### Sınıf: `LiveTradingConfiguration`

**Dosya:** `src/config/live_trading_config.py`  
**Ana Metotlar:**

#### 1. `load()` - Ana Giriş Noktası
```python
@classmethod
def load(cls, log_summary: bool = True, *, 
         config_path: Optional[str] = None, 
         force_reload: bool = False) -> Dict[str, Any]:
    """
    Ana giriş noktası. Config dosyasını yükler ve birleştirir.
    
    Singleton pattern kullanır - 1 kere yüklenir, sonra cache'den okunur.
    """
    # 1. config_path'ı çöz (custom, ENV 'CONFIG_PATH', veya default)
    resolved_path = cls._resolve_config_path(config_path)
    
    # 2. Cache kontrol et (env değişmedi mi?)
    current_signature = cls._build_signature_from_keys(resolved_path, _config_env_keys)
    if current_signature == _config_signature:
        return _config_instance  # Cache'den dön
    
    # 3. Yeni instance oluştur ve yükle
    instance = cls(resolved_path)
    config = instance._load_and_merge_configs()
    return config
```

#### 2. `_load_and_merge_configs()` - Yükleme & Birleştirme
```python
def _load_and_merge_configs(self) -> Dict[str, Any]:
    # 1. YAML dosyasını yükle ve ortam değişkeni mapping'lerini parse et
    yaml_config, env_map = self._load_yaml_and_map_env_vars()
    
    # 2. YAML değerlerini normalize et (örn: "BTC/USDT" → ["BTC/USDT"])
    yaml_config = self._normalize_yaml_values(yaml_config)
    
    # 3. Ortam değişkenlerinden override'ları al
    env_overrides = self._get_env_overrides(env_map, yaml_config)
    
    # 4. YAML + ENV'yi derin merge et
    merged = self._deep_merge(yaml_config, env_overrides)
    
    # 5. Ek normalizasyonlar ve default'ları uygula
    self._apply_universe_defaults(merged)
    self._normalize_risk_config(merged)
    
    return merged
```

#### 3. `_load_yaml_and_map_env_vars()` - YAML Parsing
```python
def _load_yaml_and_map_env_vars(self) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    YAML dosyasını satır satır okur:
    - Config dictionary'sini parse et
    - Commentler'deki ortam değişkeni mapping'lerini çıkart
    
    Örnek comment:
        enable_live: true    # Override with: ENABLE_LIVE_TRADING
        ↓
    env_map['ENABLE_LIVE_TRADING'] = ['execution', 'enable_live']
    """
    
    # Dosyayı satır satır oku
    with open(self.config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # YAML yapısını trace et (indentation takibi)
    path_stack = []
    env_map = {}
    
    for line in lines:
        # ENV mapping pattern'i bul: "# Override with: ENV_VAR"
        match = self.ENV_OVERRIDE_PATTERN.search(line)
        if match:
            env_var = match.group(1)
            current_path = [p[1] for p in path_stack]
            env_map[env_var] = current_path
    
    # YAML'ı parse et
    with open(self.config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    return yaml_config or {}, env_map
```

#### 4. `_get_env_overrides()` - Ortam Değişkeni Override'ları
```python
def _get_env_overrides(self, env_map: Dict[str, List[str]], 
                       base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ortam değişkenlerinden override'lar oluştur.
    Otomatik tip dönüşümü yapar (string → int/float/bool).
    
    Örnek:
        ENV: LEVERAGE_DEFAULT=10
        ↓
        config.execution.leverage.default = 10  (int olarak)
    """
    
    overrides = {}
    
    for env_var, path in env_map.items():
        env_value_str = os.getenv(env_var)
        
        if env_value_str is None:
            continue  # Tanımlanmamış ENV'ler atla
        
        # YAML dosyasındaki default değerin tipine göre dönüştür
        base_value = self._get_nested_value(base_config, path)
        base_type = type(base_value)
        
        # Tip dönüşümü yap: "10" → 10, "true" → True, vb.
        casted_value = self._cast_value(env_value_str, base_type)
        
        # Override dictionary'sine ekle
        self._set_nested_value(overrides, path, casted_value)
    
    return overrides
```

#### 5. `_deep_merge()` - Birleştirme
```python
def _deep_merge(self, base: Dict[str, Any], 
                overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    YAML config'i ortam değişkeni override'larıyla birleştirir.
    
    Örnek:
        base = {execution: {leverage: {default: 5}}}
        overrides = {execution: {leverage: {default: 10}}}
        ↓
        result = {execution: {leverage: {default: 10}}}
    """
    # Recursive merge işlemi
    ...
    return merged_config
```

---

## 🔑 Ortam Değişkenleri (Environment Variables)

### Config Dosyasında Tanımlı Ortam Değişkenleri:

**Execution (Çalıştırma):**
- `ENABLE_LIVE_TRADING` → `execution.enable_live`
- `ORDER_TYPE` → `execution.order_type`
- `TIME_IN_FORCE` → `execution.time_in_force`
- `FEE_PCT` → `execution.fee_pct`
- `LEVERAGE_DEFAULT` → `execution.leverage.default`

**Risk Management (Risk Yönetimi):**
- `MAX_POSITION_SIZE` → `risk_management.max_position_size`
- `STOP_LOSS_PCT` → `risk_management.stop_loss_pct`
- `MAX_CONCURRENT_POSITIONS` → `risk_management.max_concurrent_positions`

**Trading Symbols (İşlem Sembolleri):**
- `TRADING_SYMBOLS` → `universe.trading_symbols`
- `RSI_THRESHOLD_BTC` → `indicators.rsi_thresholds['BTC/USDT:USDT']`
- `RSI_THRESHOLD_ETH` → `indicators.rsi_thresholds['ETH/USDT:USDT']`

**ML (Machine Learning):**
- `ML_ENABLED` → `ml.enabled`
- `GEMMA_ENABLED` → `ml.gemma.enabled`
- `MODEL_PATH` → `ml.model_params.model_path`

### Docker Ortam Değişkenleri (bearish-bot.env):

```bash
# bearish-bot.env (VM host: /home/azureuser/)
TRADING_MODE=paper          # paper|live
DEBUG_MODE=false            # true|false
TRADING_DURATION=3600       # saniye cinsinden
EXCHANGES=bingx             # exchange adları
ENABLE_LIVE_TRADING=false   # override için
LEVERAGE_DEFAULT=5          # override için
```

---

## 🚀 Docker İmajında Config Yükleme Akışı

```
┌─────────────────────────────────────────────────────────────┐
│ Logic App / Azure Automation Runbook                        │
│ Parameters: imageTag="vm-vmboot-12", durationMinutes=60    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ VM Script: Start-BearishBot-Fixed.ps1                       │
│ → Calls: python scripts/vm_run_session.py                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Python: scripts/vm_run_session.py                           │
│ → Passes: --env-file /home/azureuser/bearish-bot.env       │
│ → Docker: docker run --env-file bearish-bot.env IMAGE      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Docker Container Starts                                     │
│ CMD: ["python", "vm_boot.py"]                              │
│ ENV: (from bearish-bot.env file)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ vm_boot.py Execution                                        │
│ 1. Import azure_boot                                        │
│ 2. setup_environment() → PYTHONPATH = "/app:/app/src:..."  │
│ 3. ensure_directories() → logs/, data/, artifacts/          │
│ 4. setup_default_manifest() → GEMMA manifest                │
│ 5. setup_ml_environment() → ML env vars                     │
│ 6. Build mode args from env vars                            │
│    - TRADING_MODE != 'live' → --paper                       │
│    - DEBUG_MODE='true' → --debug                            │
│    - TRADING_DURATION=3600 → --duration 3600                │
│ 7. Execute: subprocess.call(['python',                      │
│    'scripts/live_trading_launcher.py', '--paper', ...])     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ scripts/live_trading_launcher.py                            │
│ from config.live_trading_config import LiveTradingConfiguration
│ config = LiveTradingConfiguration.load()                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ src/config/live_trading_config.py                           │
│ 1. _resolve_config_path()                                   │
│    → CONFIG_PATH env var veya default                       │
│    → 'config/config.example.yaml'                           │
│                                                             │
│ 2. _load_and_merge_configs()                                │
│    a. Load YAML: config/config.example.yaml                 │
│    b. Parse comments for ENV mappings                       │
│    c. Read environment variables                            │
│    d. Deep merge: yaml_config + env_overrides               │
│    e. Apply defaults & normalization                        │
│                                                             │
│ 3. Return: Final merged config dict                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ live_trading_launcher.py continues...                       │
│ → ProductionCoordinator instantiate with config             │
│ → WebSocket connections, ML models, trading...              │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 Config Dosya Seçimi

### Varsayılan Dosya: `config/config.example.yaml`

**Nasıl Değiştirilir:**

#### 1. Ortam Değişkeni ile:
```bash
export CONFIG_PATH="config/config.debug.yaml"
python scripts/live_trading_launcher.py
```

#### 2. Docker Komutu ile:
```bash
docker run -e CONFIG_PATH=config/config.debug.yaml \
  bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-12
```

#### 3. Logic App Trigger'ında (Future):
```json
{
  "imageTag": "vm-vmboot-12",
  "durationMinutes": 10,
  "configPath": "config/config.debug.yaml"
}
```

---

## 📊 Singleton Cache Mekanizması

`LiveTradingConfiguration` singleton pattern kullanır:

```python
_config_instance = None          # Cache'de config dict
_config_signature = None         # Config dosyası + ENV values hash'ı
_config_env_keys = ()           # Ortam değişkenleri liste'si
_config_path_cache = None       # Config dosya path'ı

# Load çağrısı:
config = LiveTradingConfiguration.load()

# Arkaplanda:
# 1. Current signature hesapla (config file + ENV values)
# 2. Eğer signature cache'deki signature ile eşleşirse → cache'den dön
# 3. Değişmişse → yeniden yükle ve cache'i güncelle
```

**Avantajları:**
- ✅ Config 1 kere yüklenir, sonra cache'den okunur
- ✅ ENV değişirse cache invalidate olur
- ✅ Performance: Her istekte dosya okunmaz

---

## 🔍 Debug Bilgileri

### Logging:
```
✅ YAML config loaded. Found 42 environment variable mappings.
🔧 Applying overrides from environment variables...
📊 Configuration summary:
   - execution.enable_live: false
   - execution.leverage.default: 5
   - universe.trading_symbols: ['BTC/USDT:USDT', 'ETH/USDT:USDT']
```

### Cache Durumu:
```
Returning cached configuration instance (env unchanged).
OR
Configuration signature changed. Reloading config from disk.
```

---

## 📝 Özet Tablo

| Seçenek | Dosya | Konum | Yükleme Sırası | Kullanım |
|---------|-------|-------|-----------------|---------|
| **Birincil (Default)** | `config/config.example.yaml` | `/app/config/` | 2️⃣ İkinci | ✅ İş config'i |
| **Alternatif** | `config/config.debug.yaml` | `/app/config/` | 2️⃣ İkinci | 🧪 Test için |
| **Custom** | `$CONFIG_PATH` | Özel lokasyon | 2️⃣ İkinci | 🔧 Override için |
| **Ortam Var.** | Tüm ENV başlayanlar | Runtime | 1️⃣ **BİRİNCİ** | 🌍 En yüksek öncelik |
| **Hardcoded** | Python kodunda | src/ | 3️⃣ Üçüncü | 📌 Fallback |

---

## ✅ Sonuç

**vm-vmboot-12** imajında config ayarları şu şekilde okunmaktadır:

1. **Ana Kaynak:** `/app/config/config.example.yaml` (YAML formatı)
2. **Yükleme Mekanizması:** `src/config/live_trading_config.py` (Singleton + Deep Merge)
3. **Öncelik Sırası:** ENV Variables > YAML File > Hardcoded Defaults
4. **Cache:** Singleton pattern ile 1 kere yükleme, sonra cache'den
5. **Override:** Ortam değişkenleriyle kolayca override edilebilir

Config dosyasında `# Override with: ENV_VAR` comment'i ile hangi ortam değişkenleriyle override edileceği belirtilmiştir.
