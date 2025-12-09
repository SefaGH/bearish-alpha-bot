# Strategy Variables Configuration - Quick Reference

## 📊 At a Glance

| Category | Variables | Status | When to Use |
|----------|-----------|--------|-------------|
| Phase 1: Critical Flags | 8 | Ready NOW | Week 1 |
| Phase 2: Operational | 13 | Ready NOW | Week 2 |
| Phase 3: Technical | 17 | Ready NOW | Week 3+ |
| **Total** | **38** | **All Ready** | Progressive |

---

## 🎯 Phase 1: Critical Feature Flags (IMMEDIATE)

Deploy this week. Enable/disable major strategies without redeploy.

```
STRATEGY_OB_ENABLED              true/false  - Oversold Bounce strategy
STRATEGY_STR_ENABLED             true/false  - Short the Rip strategy
SIGNAL_BYPASS_ENABLED            true/false  - Extreme RSI bypass
SIGNAL_BYPASS_RSI_OVERSOLD       12-50       - Oversold threshold
SIGNAL_BYPASS_RSI_OVERBOUGHT     50-88       - Overbought threshold
ML_REGIME_MIN_CONFIDENCE         0.3-0.9     - Regime confidence
ML_RL_PPO_ENABLED                true/false  - PPO agent
ML_RL_TRAINING_MODE              true/false  - Training mode
```

**Command**:
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

**Impact**: VERY HIGH - Directly enables/disables trading strategies  
**Testing**: Low risk - just toggles existing functionality  
**Rollback**: Simple - toggle back to previous value

---

## 📈 Phase 2: Operational Parameters (NEXT)

Deploy after Phase 1 stabilizes. Fine-tune strategy quality.

```
ADAPTIVE_STRATEGIES_ENABLED      true/false  - Adaptive adjustments
ADAPTIVE_MONITORING_ENABLED      true/false  - Performance tracking
ADAPTIVE_MIN_VOLATILITY          0.01-0.05   - Volatility threshold
ADAPTIVE_MAX_POS_MULT            1.5-3.0     - Position sizing max
ADAPTIVE_MIN_POS_MULT            0.3-1.0     - Position sizing min
ML_LSTM_HIDDEN                   32-128      - Model size [CRITICAL]
ML_LSTM_LAYERS                   1-3         - Model depth [CRITICAL]
ML_LSTM_DROPOUT                  0.3-0.8     - Regularization [CRITICAL]
ML_RL_HOLD_CONFIDENCE_THRESHOLD  0.4-0.8     - RL confidence
OB_RSI_MAX                       30-50       - Entry threshold
STR_RSI_MIN                      50-70       - Entry threshold
OB_MIN_RR_RATIO                  1.0-3.0     - Risk/reward
STR_MIN_RR_RATIO                 1.0-3.0     - Risk/reward
```

**Important**: ML_LSTM_* parameters must match trained models

**Command**:
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

**Impact**: HIGH - Affects trade quality and execution  
**Testing**: Moderate - test with small capital first  
**Rollback**: Moderate - adjust values back to original

---

## ⚙️ Phase 3: Technical Parameters (OPTIONAL)

Deploy for advanced optimization. Requires careful tuning.

```
RSI_BASE_OB                      20-40       - Oversold Bounce RSI
RSI_RANGE_OB                     5-15        - Oversold Bounce range
RSI_BASE_STR                     60-80       - Short the Rip RSI
RSI_RANGE_STR                    5-15        - Short the Rip range
ML_FEATURE_SIZE                  32-64       - Features [CRITICAL]
ML_FEAT_RSI_PERIOD               7-21        - RSI period
ML_FEAT_ATR_PERIOD               7-21        - ATR period
ML_FEAT_MACD_FAST                8-14        - MACD fast
ML_FEAT_MACD_SLOW                20-30       - MACD slow
ML_FEAT_BB_PERIOD                15-30       - Bollinger Bands
ML_RL_EPSILON_INFERENCE          0.0-0.1     - Exploration
ML_RL_LEARNING_RATE              0.00001-0.0001 - Learning
ML_RL_GAMMA                      0.9-0.99    - Discount
OB_TP_ATR_MULT                   1.5-4.0     - TP multiplier
OB_SL_ATR_MULT                   0.8-2.0     - SL multiplier
STR_TP_ATR_MULT                  1.5-4.0     - TP multiplier
STR_SL_ATR_MULT                  0.8-2.0     - SL multiplier
```

**Command**:
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

**Impact**: MEDIUM - Fine-tuning performance  
**Testing**: High - test extensively before changing  
**Rollback**: Complex - may need model retraining

---

## 🚀 Recommended Timeline

### Week 1: Phase 1 Deployment
```
Monday:    Review Phase 1 parameters
Tuesday:   Test with --dry-run
Wednesday: Deploy Phase 1
Thursday:  Monitor for anomalies
Friday:    Document observations
```

### Week 2: Phase 2 Preparation & Deployment
```
Monday:    Review Phase 2 parameters
Tuesday:   Plan model compatibility
Wednesday: Test with --dry-run
Thursday:  Deploy Phase 2
Friday:    Monitor impact
```

### Week 3+: Phase 3 (Optional)
```
Monitor results from Phase 1 & 2
Plan optimization strategy
Deploy Phase 3 if needed
Establish tuning workflow
```

---

## 📋 Change Management

### Change Request Format
```
Parameter: STRATEGY_OB_ENABLED
Current Value: true
New Value: false
Reason: Testing strategy without OB
Risk Level: LOW
Rollback Plan: Change back to true
```

### Safe Parameter Changes
These can be changed anytime (low risk):
- Feature flags (ENABLED, TRAINING_MODE)
- Entry thresholds (RSI_MAX, RSI_MIN)
- Confidence levels (CONFIDENCE_THRESHOLD)
- Multipliers (ATR_MULT, position multipliers)

### Dangerous Changes
These require planning (high risk):
- LSTM parameters (requires model retraining)
- Feature size (requires model retraining)
- Learning rate (affects training results)

---

## 🔍 Verification Commands

### Check Settings Exist
```powershell
az appconfig kv list `
  --name appcs-bearish-bot `
  --label "production" | Select-String "STRATEGY"
```

### View Single Setting
```powershell
az appconfig kv show `
  --name appcs-bearish-bot `
  --key "BearishAlphaBot/STRATEGY_OB_ENABLED" `
  --label "production"
```

### Count All Settings
```powershell
az appconfig kv list `
  --name appcs-bearish-bot `
  --label "production" | Measure-Object
```

---

## 💡 Pro Tips

1. **Test Before Changing**: Use --dry-run to see what will happen
2. **Change One at a Time**: Don't change multiple parameters together
3. **Monitor Impact**: Watch logs before/after parameter change
4. **Keep Documentation**: Record why each parameter was changed
5. **Label Your Settings**: Use production/staging labels
6. **Backup Values**: Write down original values before changing

---

## ❌ What NOT to Change Without Planning

1. **ML_LSTM_HIDDEN** - Requires model retraining
2. **ML_LSTM_LAYERS** - Requires model retraining
3. **ML_LSTM_DROPOUT** - Requires model retraining
4. **ML_FEATURE_SIZE** - Requires model retraining
5. **ML_RL_LEARNING_RATE** - Affects training, requires retraining
6. **Model architecture** - Any model structure changes need retraining

**If you need to change these**:
1. Plan model retraining workflow
2. Test new parameters with training data
3. Validate model performance
4. Update ML_ACTIVE_BUNDLE path
5. Deploy new model before changing config

---

## 📞 Support

### Questions About Parameters?
See `STRATEGY_VARIABLES_ANALYSIS.md` for detailed descriptions

### Need to Change Multiple Parameters?
Use `EXTENDED_MIGRATION_SCRIPT_README.md` for automation

### What's Currently Deployed?
Check Phase 1 variables in `scripts/migrate_config_to_appconfig_v2.py`

### Issues After Deployment?
1. Check logs for config loading errors
2. Verify setting value in Azure Portal
3. Compare with config.example.yaml defaults
4. Rollback to previous value if needed

---

**Last Updated**: December 8, 2025  
**Ready for Deployment**: Phase 1, 2, 3 all validated
