# Extended Migration Script - Strategy Variables Implementation

**Date**: December 8, 2025  
**Script**: `scripts/migrate_config_to_appconfig_v2.py`  
**Status**: ✓ READY FOR PRODUCTION

---

## 📋 Overview

Expanded migration script with full support for strategy configuration variables from `config.example.yaml`. Enables dynamic configuration of trading strategies without Docker rebuilds.

---

## 🎯 Features

### 1. Three-Phase Strategy Variable System

#### Phase 1: Critical Feature Flags (8 variables)
```
STRATEGY_OB_ENABLED              - Enable/disable Oversold Bounce strategy
STRATEGY_STR_ENABLED             - Enable/disable Short the Rip strategy
SIGNAL_BYPASS_ENABLED            - Enable extreme RSI signal bypass
SIGNAL_BYPASS_RSI_OVERSOLD       - Oversold RSI threshold (12)
SIGNAL_BYPASS_RSI_OVERBOUGHT     - Overbought RSI threshold (88)
ML_REGIME_MIN_CONFIDENCE         - Regime detection confidence (0.6)
ML_RL_PPO_ENABLED                - Enable PPO agent
ML_RL_TRAINING_MODE              - Training vs inference mode
```

**Use Case**: Quick strategy enable/disable without redeploy  
**Impact**: VERY_HIGH  
**Recommended**: Add immediately

---

#### Phase 2: Operational Parameters (13 variables)
```
ADAPTIVE_STRATEGIES_ENABLED      - Adaptive strategy adjustments
ADAPTIVE_MONITORING_ENABLED      - Performance monitoring
ADAPTIVE_MIN_VOLATILITY          - Volatility threshold (0.02)
ADAPTIVE_MAX_POS_MULT            - Position multiplier max (2.0)
ADAPTIVE_MIN_POS_MULT            - Position multiplier min (0.5)
ML_LSTM_HIDDEN                   - LSTM hidden size (64) [CRITICAL]
ML_LSTM_LAYERS                   - LSTM layers (2) [CRITICAL]
ML_LSTM_DROPOUT                  - LSTM dropout (0.6) [CRITICAL]
ML_RL_HOLD_CONFIDENCE_THRESHOLD  - RL confidence threshold (0.60)
OB_RSI_MAX                       - Oversold Bounce RSI (45)
STR_RSI_MIN                      - Short the Rip RSI (55)
OB_MIN_RR_RATIO                  - Oversold Bounce R/R (1.5)
STR_MIN_RR_RATIO                 - Short the Rip R/R (1.5)
```

**Use Case**: Fine-tune trade quality  
**Impact**: HIGH  
**Warning**: ML_LSTM_* parameters must match trained models

---

#### Phase 3: Technical Parameters (17 variables)
```
RSI_BASE_OB                      - Adaptive RSI base (32)
RSI_RANGE_OB                     - Adaptive RSI range (8)
RSI_BASE_STR                     - Adaptive RSI base (68)
RSI_RANGE_STR                    - Adaptive RSI range (8)
ML_FEATURE_SIZE                  - Feature dimension (42) [CRITICAL]
ML_FEAT_RSI_PERIOD              - RSI period (14)
ML_FEAT_ATR_PERIOD              - ATR period (14)
ML_FEAT_MACD_FAST               - MACD fast (12)
ML_FEAT_MACD_SLOW               - MACD slow (26)
ML_FEAT_BB_PERIOD               - Bollinger Bands (20)
ML_RL_EPSILON_INFERENCE         - Exploration rate (0.01)
ML_RL_LEARNING_RATE             - Learning rate (0.00003)
ML_RL_GAMMA                     - Discount factor (0.95)
OB_TP_ATR_MULT                  - TP multiplier (2.5)
OB_SL_ATR_MULT                  - SL multiplier (1.2)
STR_TP_ATR_MULT                 - TP multiplier (3.0)
STR_SL_ATR_MULT                 - SL multiplier (1.5)
```

**Use Case**: Advanced optimization  
**Impact**: MEDIUM  
**Recommended**: For power users only

---

## 📊 Total Variables by Phase

| Phase | Count | Total | Cumulative |
|-------|-------|-------|------------|
| Original (env file) | 26 | 26 | 26 |
| Phase 1 (Critical) | 8 | 34 | 34 |
| Phase 2 (Operational) | 13 | 47 | 47 |
| Phase 3 (Technical) | 17 | 64 | 64 |

---

## 🚀 Usage

### Basic Dry-Run (Preview Changes)
```powershell
python scripts/migrate_config_to_appconfig_v2.py `
  --env-file "bearish-bot.env" `
  --app-config-name "appcs-bearish-bot" `
  --app-config-rg "TradeBot" `
  --keyvault-name "bearish-kv" `
  --keyvault-rg "tradebot-ops" `
  --phase 1 `
  --dry-run
```

### Phase 1: Critical Feature Flags (Recommended)
```powershell
python scripts/migrate_config_to_appconfig_v2.py `
  --env-file "bearish-bot.env" `
  --app-config-name "appcs-bearish-bot" `
  --app-config-rg "TradeBot" `
  --keyvault-name "bearish-kv" `
  --keyvault-rg "tradebot-ops" `
  --phase 1 `
  --label production
```

### Phase 2: Operational Parameters (Next Step)
```powershell
python scripts/migrate_config_to_appconfig_v2.py `
  --env-file "bearish-bot.env" `
  --app-config-name "appcs-bearish-bot" `
  --app-config-rg "TradeBot" `
  --keyvault-name "bearish-kv" `
  --keyvault-rg "tradebot-ops" `
  --phase 2 `
  --label production
```

### Phase 3: All Technical Parameters (Optional)
```powershell
python scripts/migrate_config_to_appconfig_v2.py `
  --env-file "bearish-bot.env" `
  --app-config-name "appcs-bearish-bot" `
  --app-config-rg "TradeBot" `
  --keyvault-name "bearish-kv" `
  --keyvault-rg "tradebot-ops" `
  --phase 3 `
  --label production
```

---

## ⚙️ Command Line Options

```
--env-file              Path to bearish-bot.env (required)
--app-config-name       App Configuration store name (required)
--app-config-rg         App Configuration resource group (required)
--keyvault-name         Key Vault name (required)
--keyvault-rg           Key Vault resource group (required)
--prefix                Key prefix (default: BearishAlphaBot/)
--label                 Settings label (default: production)
--phase                 Strategy variables phase (0=none, 1=critical, 2=operational, 3=all)
--dry-run               Preview without making changes
```

---

## 📈 Test Results

### Phase 1 Test (Dry-Run)
```
[OK] Loaded 29 settings from bearish-bot.env
[OK] Loaded 8 strategy variables (Phase 1)

Config Settings to Create:
  - 26 from env file
  - 8 strategy variables
  Total: 34 new settings
```

### Phase 2 Test (Dry-Run)
```
[OK] Loaded 29 settings from bearish-bot.env
[OK] Loaded 21 strategy variables (Phase 2)
  - 8 from Phase 1
  - 13 from Phase 2
  
Total: 47 new settings
```

### Phase 3 Test (Dry-Run)
```
[OK] Loaded 29 settings from bearish-bot.env
[OK] Loaded 38 strategy variables (Phase 3)
  - 8 from Phase 1
  - 13 from Phase 2
  - 17 from Phase 3
  
Total: 64 new settings
```

---

## ⚠️ Critical Parameters

These parameters MUST MATCH trained models. Changing requires retraining:

```
ML_LSTM_HIDDEN       (64)  - Model input dimension
ML_LSTM_LAYERS       (2)   - Model architecture
ML_LSTM_DROPOUT      (0.6) - Model architecture
ML_FEATURE_SIZE      (42)  - Feature vector size
```

When changing these, follow this process:
1. Update parameters in App Configuration
2. Retrain models with new parameters
3. Update ML_ACTIVE_BUNDLE to new model path
4. Test thoroughly in staging before production

---

## 🔄 Implementation Roadmap

### Week 1: Phase 1 Deployment (Immediate)
- [ ] Deploy Phase 1 script (critical flags)
- [ ] Test strategy enable/disable in production
- [ ] Monitor trading behavior for anomalies
- [ ] Document any side effects

### Week 2: Phase 2 Deployment (Next)
- [ ] Deploy Phase 2 script (operational parameters)
- [ ] Create tuning guidelines document
- [ ] Train team on parameter adjustment
- [ ] Set up monitoring for parameter impact

### Week 3: Phase 3 Deployment (Optional)
- [ ] Deploy Phase 3 for advanced users
- [ ] Create model retraining workflow
- [ ] Document parameter-model compatibility
- [ ] Set up validation checks

---

## 📝 Usage Examples

### Example 1: Disable Short the Rip Strategy
```powershell
# After Phase 1 deployed, change in Azure Portal:
STRATEGY_STR_ENABLED = false

# Or use Azure CLI:
az appconfig kv set `
  --name appcs-bearish-bot `
  --key "BearishAlphaBot/STRATEGY_STR_ENABLED" `
  --value "false" `
  --label "production"

# Restart application - changes take effect immediately
```

### Example 2: Adjust Risk/Reward Ratios
```powershell
# After Phase 2 deployed:
OB_MIN_RR_RATIO = 2.0    # Require 2x R/R instead of 1.5x
STR_MIN_RR_RATIO = 2.5   # Require 2.5x R/R instead of 1.5x

# Application automatically loads new values on next read
```

### Example 3: Tune RSI Thresholds
```powershell
# After Phase 2 deployed:
OB_RSI_MAX = 42          # More conservative entry
STR_RSI_MIN = 60         # More conservative entry

# Takes effect on next market data poll
```

---

## 🛡️ Safety Features

1. **Dry-Run Mode**: Preview all changes before execution
2. **Idempotent**: Safe to run multiple times
3. **Skip Existing**: Won't overwrite already-migrated settings
4. **Error Recovery**: Detailed logging of any failures
5. **Label Support**: Separate labels for staging/production
6. **Validation**: Critical parameters marked with [WARN]

---

## 📊 Configuration Structure

App Configuration keys follow this pattern:
```
BearishAlphaBot/STRATEGY_OB_ENABLED
BearishAlphaBot/ML_REGIME_MIN_CONFIDENCE
BearishAlphaBot/ADAPTIVE_STRATEGIES_ENABLED
```

Access in code:
```python
from src.config.live_trading_config import LiveTradingConfiguration

config = LiveTradingConfiguration.load()
ob_enabled = config.get('strategy.oversold_bounce.enabled', True)
regime_confidence = config.get('ml.regime_prediction.min_confidence', 0.6)
```

---

## 🔍 Verification

### Check if Settings Exist
```powershell
az appconfig kv list `
  --name appcs-bearish-bot `
  --label "production" `
  --query "[?starts_with(key, 'BearishAlphaBot/STRATEGY')]"
```

### Count Migrated Settings
```powershell
az appconfig kv list `
  --name appcs-bearish-bot `
  --label "production" `
  --query "length(@)"
```

### View Specific Setting
```powershell
az appconfig kv show `
  --name appcs-bearish-bot `
  --key "BearishAlphaBot/ML_REGIME_MIN_CONFIDENCE" `
  --label "production"
```

---

## 🎓 Next Steps

1. **Review**: Examine `STRATEGY_VARIABLES_ANALYSIS.md` for detailed parameter descriptions
2. **Deploy Phase 1**: Use script to add critical feature flags
3. **Test**: Verify settings load correctly in application
4. **Monitor**: Track impact of parameter changes
5. **Plan Phase 2**: Schedule operational parameters deployment
6. **Document**: Create team runbook for parameter adjustments

---

## 📚 Related Files

- **Analysis**: `STRATEGY_VARIABLES_ANALYSIS.md` (82 variables analyzed)
- **Config Loader**: `src/config/live_trading_config.py` (updated for App Config)
- **Defaults**: `config/config.example.yaml` (637 lines)
- **Migration Original**: `scripts/migrate_config_to_appconfig.py` (original version)
- **Migration Extended**: `scripts/migrate_config_to_appconfig_v2.py` (this script)

---

## ✅ Checklist Before First Deployment

- [ ] Review all Phase 1 parameters (8 variables)
- [ ] Backup current bearish-bot.env
- [ ] Run script with --dry-run to preview changes
- [ ] Review migration summary for errors
- [ ] Deploy script to production
- [ ] Verify settings appear in Azure Portal
- [ ] Restart application to load new configuration
- [ ] Monitor logs for any configuration errors
- [ ] Test strategy enable/disable functionality
- [ ] Document any parameter impacts observed

---

**Status**: Ready for Phase 1 Deployment  
**Maintainer**: Bearish Alpha Bot Team  
**Last Updated**: December 8, 2025
