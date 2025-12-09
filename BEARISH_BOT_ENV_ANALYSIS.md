# bearish-bot.env - Azure VM Mevcut Konfigürasyonu

## 📋 Dosya Konumu
**Azure VM**: `/home/azureuser/bearish-bot.env`

---

## 📊 Mevcut Ortam Değişkenleri (Tam Liste)

### 🎮 Trading Konfigürasyonu
```
TRADING_MODE=paper              # paper mode aktif (live değil)
DEBUG_MODE=false                # debug logging kapalı
ML_ENABLED=true                 # Machine Learning aktif
EXCHANGES=bingx                 # Tek exchange: BingX
TRADING_DURATION=7200           # 2 saat trading dönemi
BINGX_REST_DEBUG=1              # BingX REST debug aktif
```

### 🔑 Exchange Credentials (API Keys)
```
BINGX_KEY=2cPdB7GaD3dRdvHoPe3rCN2rCcmCixlujWq6vhYD7gprATEWDMkSsB0e11aoMc4lW3xGuidO2XtiN6aCEYH4w
BINGX_SECRET=R0WJPPl85RlUdSVkeLuOY94PFuNG2MHiduN3EKYYwFImblzpyT6jjPGzGIEKfgPP2wHzcgBS8NotDgjlvoFKg
```

### 📱 Telegram Notifications
```
TELEGRAM_BOT_TOKEN=8430411522:AAEBNktJplfrY4a8b4RSQpGBi4PtjLwXAUw
TELEGRAM_CHAT_ID=1359128753
```

### 💰 Trading Parametreleri
```
CAPITAL_USDT=100                            # Başlangıç sermayesi: 100 USDT
PER_TRADE_RISK_PCT=0.01                     # Ticaret başına risk: %1
DAILY_MAX_TRADES=8                          # Günlük maksimum işlem: 8
DUPLICATE_PREVENTION_THRESHOLD=0.0005       # İşlem kopya önleme eşiği
DUPLICATE_PREVENTION_COOLDOWN=20            # Cooldown: 20 saniye
```

### 📈 Semboller ve RSI Eşikleri
```
TRADING_SYMBOLS=BTC/USDT:USDT               # Sadece Bitcoin işlemi
RSI_THRESHOLD_BTC=50                        # BTC RSI eşiği: 50
RSI_THRESHOLD_ETH=50                        # ETH RSI eşiği: 50 (kullanılmıyor)
RSI_THRESHOLD_SOL=50                        # SOL RSI eşiği: 50 (kullanılmıyor)
```

### 🤖 Machine Learning (GEMMA)
```
GEMMA_ENABLED=true                          # GEMMA ML modeli aktif
ML_ACTIVE_BUNDLE=artifacts/gemma/final      # ML model bundle yolu
ML_FEAT_VOL_WINDOWS=5,10,20,50               # Volume feature windows
ML_FEAT_MOM_WINDOWS=5,10,20,50               # Momentum feature windows
```

### 🌐 WebSocket Limitler
```
WS_MAX_STREAMS_BINGX=10                     # BingX WebSocket max streams
```

### 💲 Fiyat Delta Bypass
```
PRICE_DELTA_BYPASS_ENABLED=true              # Fiyat delta bypass açık
PRICE_DELTA_BYPASS_THRESHOLD=0.0015          # Threshold: 0.15%
```

### 🐍 Python Ortamı
```
PYTHONUNBUFFERED=1                          # Unbuffered output
PYTHONPATH=/home/site/wwwroot:/home/site/wwwroot/src:/home/site/wwwroot/scripts
```

### 📝 Logging
```
LOG_LEVEL=INFO                              # Log seviyesi: INFO
```

### 💾 Ek Ayarlar
```
# PORT: Azure App Service'ten otomatik ayarlanır
# SCM_DO_BUILD_DURING_DEPLOYMENT: (commented out)
# KEYVAULT_NAME: (commented out)
```

---

## 🔍 Analiz

### Mevcut Yapı İle İlgili Gözlemler:

1. **API Keys Açık Text'te** ⚠️
   - BINGX_KEY ve BINGX_SECRET dosyada açık text
   - Azure App Configuration'a taşındığında Key Vault ile şifrelenebilir

2. **Paper Mode Aktif** 
   - Gerçek ticaret yapılmıyor
   - Test/development için uygun

3. **Tek Symbol** 
   - Sadece BTC/USDT:USDT işlemi yapılıyor
   - ETH ve SOL RSI eşikleri tanımlanmış ama kullanılmıyor

4. **ML Aktif** 
   - GEMMA modeli çalışıyor
   - Feature windows: 5, 10, 20, 50

5. **Telegram Notifications Var**
   - Trading events'leri bildirim alabilecek durumda

---

## 📐 Azure App Configuration Mapping

Dosya Azure App Configuration'a aktarılmışsa, şu şekilde organize edilebilir:

```
Azure App Configuration Store: "bearish-app-config"

BearishAlphaBot/
├── trading_mode                    = "paper"
├── debug_mode                      = "false"
├── ml_enabled                      = "true"
├── exchanges                       = "bingx"
├── trading_duration                = "7200"
├── bingx_rest_debug                = "1"
├── bingx_key                       = "@Microsoft.KeyVault(..." ← Key Vault'tan
├── bingx_secret                    = "@Microsoft.KeyVault(..." ← Key Vault'tan
├── telegram_bot_token              = "@Microsoft.KeyVault(...)"
├── telegram_chat_id                = "1359128753"
├── capital_usdt                    = "100"
├── per_trade_risk_pct              = "0.01"
├── daily_max_trades                = "8"
├── duplicate_prevention_threshold  = "0.0005"
├── duplicate_prevention_cooldown   = "20"
├── trading_symbols                 = "BTC/USDT:USDT"
├── rsi_threshold_btc               = "50"
├── rsi_threshold_eth               = "50"
├── rsi_threshold_sol               = "50"
├── gemma_enabled                   = "true"
├── ml_active_bundle                = "artifacts/gemma/final"
├── ml_feat_vol_windows             = "5,10,20,50"
├── ml_feat_mom_windows             = "5,10,20,50"
├── ws_max_streams_bingx            = "10"
├── price_delta_bypass_enabled      = "true"
├── price_delta_bypass_threshold    = "0.0015"
└── log_level                       = "INFO"
```

---

## 🔄 Migration Planı

### Adım 1: Key Vault'a Secrets Taşı
```bash
az keyvault secret set \
  --vault-name bearish-kv \
  --name "bingx-key" \
  --value "2cPdB7GaD3dRdvHoPe3rCN2rCcmCixlujWq6vhYD7gprATEWDMkSsB0e11aoMc4lW3xGuidO2XtiN6aCEYH4w"

az keyvault secret set \
  --vault-name bearish-kv \
  --name "bingx-secret" \
  --value "R0WJPPl85RlUdSVkeLuOY94PFuNG2MHiduN3EKYYwFImblzpyT6jjPGzGIEKfgPP2wHzcgBS8NotDgjlvoFKg"

az keyvault secret set \
  --vault-name bearish-kv \
  --name "telegram-bot-token" \
  --value "8430411522:AAEBNktJplfrY4a8b4RSQpGBi4PtjLwXAUw"
```

### Adım 2: App Configuration'a Migrate
```bash
# Non-secret settings
az appconfig kv set --name bearish-app-config \
  --key "BearishAlphaBot/trading_mode" \
  --value "paper"

az appconfig kv set --name bearish-app-config \
  --key "BearishAlphaBot/debug_mode" \
  --value "false"

# Secret settings (Key Vault references)
az appconfig kv set --name bearish-app-config \
  --key "BearishAlphaBot/bingx_key" \
  --value "@Microsoft.KeyVault(SecretUri=https://bearish-kv.vault.azure.net/secrets/bingx-key/)"
```

### Adım 3: vm_boot.py Güncellemesi
```python
# Eski (bearish-bot.env)
# docker run --env-file /home/azureuser/bearish-bot.env ...

# Yeni (App Configuration)
# docker run -e AZURE_APPCONFIG_ENDPOINT="https://bearish-app-config.azconfig.io" ...
```

---

## 📊 Özet

| Kategori | Değer |
|----------|-------|
| **Aktif Mod** | Paper Trading |
| **Exchange** | BingX Türev (Futures) |
| **Symbol** | BTC/USDT:USDT |
| **Başlangıç Sermayesi** | 100 USDT |
| **Risk/Trade** | %1 |
| **Max Trades/Gün** | 8 |
| **Trading Duration** | 2 saat (7200s) |
| **ML Status** | Aktif (GEMMA) |
| **Notifications** | Telegram (aktif) |
| **Total Settings** | 27 key-value pair |
| **Sensitive Items** | 3 (API keys + Bot token) |

---

## ✅ Sonraki Adımlar

1. ✅ **Key Vault'a Secrets Taşı** - API keys güvenli sakla
2. ✅ **Azure App Configuration Oluştur** - Centralized config
3. ✅ **LiveTradingConfiguration Güncelle** - App Config'den oku
4. ✅ **vm_boot.py Sadeleştir** - bearish-bot.env logic'ini kaldır
5. ✅ **Yeni Image Build** - vm-vmboot-13
6. ✅ **Test & Deploy** - Production'a geç
