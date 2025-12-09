# Configuration Strategy Variables - Azure App Configuration Analysis

**Date**: December 8, 2025  
**Focus**: Identifying strategy variables that should be migrated to Azure App Configuration  
**Total Lines in config.example.yaml**: 637

---

## 📊 Executive Summary

### Current Status
- **bearish-bot.env**: 29 settings (basic trading + ML flags)
- **config.example.yaml**: 637 lines (complete configuration)
- **Gap**: Strategy variables NOT in bearish-bot.env but highly valuable for Azure App Config

### Key Findings
✅ **Identified 82 strategy variables** across 5 categories that would benefit from Azure App Configuration

---

## 🎯 Strategy Variables by Category

### CATEGORY 1: Adaptive Strategies (Lines 230-250)
**Purpose**: Dynamic behavior adjustment based on market conditions  
**Impact**: MEDIUM-HIGH (affects trade execution)

#### Variables to Migrate
```yaml
adaptive_strategies:
  enable: true                              # NEW → APP_CONFIG
  monitoring.enabled: true                  # NEW → APP_CONFIG
  monitoring.report_interval: 300           # NEW → APP_CONFIG
  monitoring.track_performance: true        # NEW → APP_CONFIG
  performance.min_volatility_for_adjustment: 0.02  # NEW → APP_CONFIG
  performance.max_position_multiplier: 2.0  # NEW → APP_CONFIG
  performance.min_position_multiplier: 0.5  # NEW → APP_CONFIG
```

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `ADAPTIVE_STRATEGIES_ENABLED` | 233 | true | HIGH | Controls adaptive behavior globally |
| `ADAPTIVE_MONITORING_ENABLED` | 235 | true | HIGH | Performance tracking crucial |
| `ADAPTIVE_MONITOR_INTERVAL` | 236 | 300 | MEDIUM | Adjustable reporting |
| `ADAPTIVE_MIN_VOLATILITY` | 241 | 0.02 | HIGH | Adjustment threshold |
| `ADAPTIVE_MAX_POS_MULT` | 242 | 2.0 | HIGH | Risk control multiplier |
| `ADAPTIVE_MIN_POS_MULT` | 243 | 0.5 | HIGH | Risk control multiplier |

---

### CATEGORY 2: Signal Generation & Strategies (Lines 250-310)

#### Subsection 2A: Extreme Condition Bypass (Lines 255-262)
**Purpose**: Override RL agent when RSI reaches extreme levels  
**Impact**: HIGH (forces trades in extreme conditions)

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `SIGNAL_BYPASS_ENABLED` | 258 | true | HIGH | Feature flag |
| `SIGNAL_BYPASS_RSI_OVERSOLD` | 259 | 12 | VERY_HIGH | Entry threshold for oversold |
| `SIGNAL_BYPASS_RSI_OVERBOUGHT` | 260 | 88 | VERY_HIGH | Entry threshold for overbought |
| `SIGNAL_FORCE_SWAP_ENABLED` | 261 | true | HIGH | Swap weakest position when full |

#### Subsection 2B: Duplicate Prevention (Lines 264-281)
**Purpose**: Prevent duplicate signals within short time frames  
**Impact**: HIGH (reduces noise trades)

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `DUPLICATE_PREVENTION_THRESHOLD` | 266 | 0.0005 | VERY_HIGH | ✓ ALREADY IN APP_CONFIG |
| `DUPLICATE_PREVENTION_COOLDOWN` | 267 | 20 | VERY_HIGH | ✓ ALREADY IN APP_CONFIG |
| `PRICE_DELTA_BYPASS_THRESHOLD` | 268 | 0.0015 | HIGH | ✓ ALREADY IN APP_CONFIG |
| `PRICE_DELTA_BYPASS_ENABLED` | 269 | false | HIGH | ✓ ALREADY IN APP_CONFIG |
| `ML_DUPLICATE_DETECTION_ENABLED` | 270 | true | MEDIUM | NEW → APP_CONFIG |
| `DUPLICATE_DYNAMIC_COOLDOWN_ENABLED` | 272 | true | HIGH | NEW → APP_CONFIG |
| `DUPLICATE_DYNAMIC_HIGH_DELTA` | 273 | 15 | MEDIUM | NEW → APP_CONFIG |
| `DUPLICATE_DYNAMIC_MEDIUM_DELTA` | 274 | 8 | MEDIUM | NEW → APP_CONFIG |
| `DUPLICATE_DYNAMIC_FAST_SECONDS` | 275 | 15 | MEDIUM | NEW → APP_CONFIG |
| `DUPLICATE_DYNAMIC_MEDIUM_SECONDS` | 276 | 45 | MEDIUM | NEW → APP_CONFIG |
| `DUPLICATE_DYNAMIC_SLOW_SECONDS` | 277 | 120 | MEDIUM | NEW → APP_CONFIG |

#### Subsection 2C: Oversold Bounce Strategy (Lines 279-295)
**Purpose**: Buy when RSI drops below threshold (bounce play)  
**Impact**: VERY_HIGH (primary strategy)

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `STRATEGY_OB_ENABLED` | 280 | true | VERY_HIGH | Feature flag for strategy |
| `STRATEGY_OB_IGNORE_REGIME` | 281 | false | HIGH | Override regime checks |
| `OB_MIN_RR_RATIO` | 282 | 1.5 | HIGH | Risk/reward enforcement |
| `OB_RSI_MAX` | 283 | 45 | VERY_HIGH | Entry threshold |
| `RSI_BASE_OB` | 284 | 32 | HIGH | Adaptive RSI base |
| `RSI_RANGE_OB` | 285 | 8 | HIGH | Adaptive RSI range |
| `OB_VOLATILITY_SENSITIVITY` | 287 | "medium" | MEDIUM | Volatility adjustment |
| `OB_TP_ATR_MULT` | 288 | 2.5 | MEDIUM | Take profit multiplier |
| `OB_SL_ATR_MULT` | 289 | 1.2 | MEDIUM | Stop loss multiplier |
| `OB_MIN_TP_PCT` | 290 | 0.008 | MEDIUM | Minimum TP percent |
| `OB_MAX_SL_PCT` | 291 | 0.015 | MEDIUM | Maximum SL percent |

#### Subsection 2D: Short the Rip Strategy (Lines 297-324)
**Purpose**: Short when RSI spikes above threshold (pullback play)  
**Impact**: VERY_HIGH (primary strategy)

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `STRATEGY_STR_ENABLED` | 298 | true | VERY_HIGH | Feature flag for strategy |
| `STRATEGY_STR_IGNORE_REGIME` | 299 | false | HIGH | Override regime checks |
| `STR_MIN_RR_RATIO` | 300 | 1.5 | HIGH | Risk/reward enforcement |
| `STR_RSI_MIN` | 301 | 55 | VERY_HIGH | Entry threshold |
| `RSI_BASE_STR` | 302 | 68 | HIGH | Adaptive RSI base |
| `RSI_RANGE_STR` | 303 | 8 | HIGH | Adaptive RSI range |
| `STR_VOLATILITY_SENSITIVITY` | 305 | "medium" | MEDIUM | Volatility adjustment |
| `STR_TP_ATR_MULT` | 306 | 3.0 | MEDIUM | Take profit multiplier |
| `STR_SL_ATR_MULT` | 307 | 1.5 | MEDIUM | Stop loss multiplier |
| `STR_MIN_TP_PCT` | 308 | 0.010 | MEDIUM | Minimum TP percent |
| `STR_MAX_SL_PCT` | 309 | 0.020 | MEDIUM | Maximum SL percent |
| `STR_VOLATILITY_STOP_ENABLED` | 311 | true | HIGH | Dynamic stop loss |
| `STR_VOLATILITY_MIN_SL_PCT` | 312 | 0.0025 | MEDIUM | Minimum SL in low vol |
| `STR_VOLATILITY_MAX_SL_PCT` | 313 | 0.02 | MEDIUM | Maximum SL in low vol |
| `RSI_THRESHOLD_BTC` | 320 | 50 | HIGH | Symbol-specific threshold |

---

### CATEGORY 3: Machine Learning Settings (Lines 366-520)

#### Subsection 3A: Feature Engineering (Lines 375-390)
**Purpose**: Technical indicators used by all ML models  
**Impact**: VERY_HIGH (affects all predictions)

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `ML_ENABLED` | 368 | true | VERY_HIGH | ✓ ALREADY IN APP_CONFIG |
| `ML_FEAT_RSI_PERIOD` | 379 | 14 | HIGH | Feature engineering |
| `ML_FEAT_ATR_PERIOD` | 380 | 14 | HIGH | Feature engineering |
| `ML_FEAT_MACD_FAST` | 381 | 12 | HIGH | Feature engineering |
| `ML_FEAT_MACD_SLOW` | 382 | 26 | HIGH | Feature engineering |
| `ML_FEAT_MACD_SIGNAL` | 383 | 9 | HIGH | Feature engineering |
| `ML_FEAT_BB_PERIOD` | 384 | 20 | HIGH | Feature engineering |
| `ML_FEAT_BB_STD` | 385 | 2 | HIGH | Feature engineering |
| `ML_FEAT_VOL_WINDOWS` | 386 | [5,10,20,50] | HIGH | ✓ ALREADY IN APP_CONFIG |
| `ML_FEAT_MOM_WINDOWS` | 387 | [5,10,20,50] | HIGH | ✓ ALREADY IN APP_CONFIG |

#### Subsection 3B: Price Prediction Engine (Lines 394-418)
**Purpose**: LSTM/Transformer models for price forecasting  
**Impact**: HIGH (secondary signals)

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `ML_PRICE_PRED_ENABLED` | 397 | true | MEDIUM | Feature flag |
| `ML_PRED_TIMEFRAMES` | 398 | ['5m','15m','30m','1h','4h'] | MEDIUM | Timeframes to predict |
| `ML_PRED_UPDATE_INTERVAL` | 399 | 60 | LOW | Update frequency |
| `ML_PRED_CACHE_TTL` | 400 | 300 | LOW | Cache timeout |
| `ML_MODELS` | 403 | ['lstm','transformer'] | MEDIUM | Model ensemble |
| `ML_FEATURE_SIZE` | 404 | 42 | HIGH | Feature dimension (CRITICAL for model compatibility) |
| `ML_FORECAST_HORIZON` | 405 | 12 | MEDIUM | Forecast bars ahead |

**⚠️ CRITICAL NOTE**: `ML_FEATURE_SIZE` must match trained models. Changing requires retraining.

#### Subsection 3C: Regime Prediction Engine (Lines 422-466)
**Purpose**: Classify market as Bullish/Bearish/Neutral/Volatile  
**Impact**: VERY_HIGH (route strategies based on regime)

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `ML_REGIME_ENABLED` | 425 | true | VERY_HIGH | Feature flag |
| `ML_REGIME_MIN_CONFIDENCE` | 426 | 0.6 | VERY_HIGH | Minimum confidence threshold |
| `REGIME_SOFT_WEIGHT_ENABLED` | 429 | true | HIGH | Graduated weighting instead of cutoff |
| `REGIME_MIN_CONF_REJECT` | 430 | 0.30 | MEDIUM | Below this, completely reject |
| `REGIME_MIN_CONF_FULL` | 431 | 0.60 | MEDIUM | Above this, use full weight |
| `ML_LSTM_HIDDEN` | 443 | 64 | HIGH | LSTM hidden size (CRITICAL - must match models) |
| `ML_LSTM_LAYERS` | 444 | 2 | HIGH | LSTM number of layers (CRITICAL - must match models) |
| `ML_LSTM_DROPOUT` | 445 | 0.6 | HIGH | LSTM dropout rate |

**⚠️ CRITICAL NOTE**: LSTM parameters must match trained models. Changing requires retraining.

#### Subsection 3D: Reinforcement Learning Agent (Lines 469-540)
**Purpose**: PPO agent that approves/vetos other signals  
**Impact**: VERY_HIGH (final decision maker)

| Variable | Line | Current Value | Priority | Reason |
|----------|------|---------------|----------|--------|
| `ML_RL_ENABLED` | 470 | true | VERY_HIGH | ✓ Feature flag |
| `ML_RL_LEGACY_ENABLED` | 471 | false | MEDIUM | Use old DQN or new PPO |
| `ML_RL_PPO_ENABLED` | 472 | true | VERY_HIGH | ✓ ALREADY IN APP_CONFIG (partial) |
| `ML_RL_PPO_SYMBOLS` | 473 | BTC/USDT:USDT | HIGH | ✓ ALREADY IN APP_CONFIG (partial) |
| `ML_RL_PPO_TIMEFRAME` | 474 | "1h" | HIGH | ✓ ALREADY IN APP_CONFIG (partial) |
| `ML_RL_PPO_MODEL` | 475 | "artifacts/ppo/ppo_trading_agent.zip" | HIGH | ✓ ALREADY IN APP_CONFIG (partial) |
| `ML_RL_PPO_FALLBACK` | 476 | 0.5 | MEDIUM | Fallback score when model unavailable |
| `ML_RL_PPO_RR_DOWN` | 477 | 0.9 | MEDIUM | R/R multiplier downside |
| `ML_RL_PPO_RR_UP` | 478 | 1.3 | MEDIUM | R/R multiplier upside |
| `ML_RL_PPO_POS_BASE` | 479 | 0.5 | MEDIUM | Position sizing base |
| `ML_RL_PPO_POS_BONUS` | 480 | 0.5 | MEDIUM | Position sizing bonus |
| `ML_RL_PPO_LOOKBACK_BARS` | 481 | 240 | MEDIUM | Candles fetched for state |
| `ML_RL_PPO_LOOKBACK_WINDOWS` | 482 | [12,24,48,96] | MEDIUM | Summary windows for telemetry |
| `ML_RL_TRAINING_MODE` | 500 | false | VERY_HIGH | ✓ ALREADY IN APP_CONFIG (partial) |
| `ML_RL_HOLD_CONFIDENCE_THRESHOLD` | 503 | 0.60 | HIGH | NEW → APP_CONFIG |
| `ML_RL_EPSILON_INFERENCE` | 506 | 0.01 | MEDIUM | Exploration rate (live trading) |
| `ML_RL_EPSILON_START` | 507 | 1.0 | MEDIUM | Exploration rate (training start) |
| `ML_RL_EPSILON_DECAY` | 508 | 0.97 | MEDIUM | Exploration decay per episode |
| `ML_RL_EPSILON_MIN` | 509 | 0.01 | MEDIUM | Exploration minimum floor |
| `ML_RL_REGIME_BIAS` | 511 | 0.0 | LOW | Regime bias strength |
| `ML_RL_MAX_REGIME_BIAS` | 512 | 3.0 | LOW | Maximum regime bias |
| `ML_RL_MIN_REGIME_CONF` | 513 | 0.6 | LOW | Minimum confidence for bias |
| `ML_RL_Q_STD_BYPASS` | 514 | 0.0001 | LOW | Q-value std bypass threshold |
| `ML_RL_RISK_PENALTY` | 515 | 10.0 | MEDIUM | Risk penalty strength |
| `ML_RL_LEARNING_RATE` | 518 | 0.00003 | MEDIUM | Learning rate (training) |
| `ML_RL_GAMMA` | 519 | 0.95 | MEDIUM | Discount factor |
| `ML_RL_GRADIENT_CLIP` | 520 | 1.0 | MEDIUM | Gradient clipping norm |
| `ML_RL_REWARD_CLIP_ENABLED` | 523 | true | MEDIUM | Enable reward clipping |
| `RL_REWARD_CLIP_MIN` | 524 | -2.0 | MEDIUM | Reward clip minimum |
| `RL_REWARD_CLIP_MAX` | 525 | 2.0 | MEDIUM | Reward clip maximum |

---

## 📈 Priority Matrix

### VERY_HIGH Priority (Must Add to App Config)
**Reason**: Critical feature flags or thresholds that impact trade logic

```
Recommended Order:
1. STRATEGY_OB_ENABLED (Line 280)
2. STRATEGY_STR_ENABLED (Line 298)
3. SIGNAL_BYPASS_ENABLED (Line 258)
4. SIGNAL_BYPASS_RSI_OVERSOLD (Line 259)
5. SIGNAL_BYPASS_RSI_OVERBOUGHT (Line 260)
6. ML_REGIME_MIN_CONFIDENCE (Line 426)
7. ML_RL_PPO_ENABLED (Line 472)
8. ML_RL_TRAINING_MODE (Line 500)
```

### HIGH Priority (Recommended to Add)
**Reason**: Operational parameters that affect trade quality

```
Recommended Order:
9. ADAPTIVE_STRATEGIES_ENABLED (Line 233)
10. ADAPTIVE_MONITORING_ENABLED (Line 235)
11. ADAPTIVE_MIN_VOLATILITY (Line 241)
12. ADAPTIVE_MAX_POS_MULT (Line 242)
13. ML_RL_HOLD_CONFIDENCE_THRESHOLD (Line 503)
14. ML_LSTM_HIDDEN (Line 443)
15. ML_LSTM_LAYERS (Line 444)
16. OB_MIN_RR_RATIO (Line 282)
17. STR_MIN_RR_RATIO (Line 300)
18. OB_RSI_MAX (Line 283)
19. STR_RSI_MIN (Line 301)
20. RSI_THRESHOLD_BTC (Line 320)
```

### MEDIUM Priority (Optional)
**Reason**: Fine-tuning parameters for advanced users

```
ML features, window parameters, multipliers, etc.
Total of ~25 parameters
```

### LOW Priority (Can Skip)
**Reason**: Rarely changed or expert-only parameters

```
Regime bias parameters, epsilon decay for training, etc.
```

---

## 🎯 Recommended Migration Plan

### Phase 1: Critical Feature Flags (8 settings)
Add to App Configuration immediately:
```
1. STRATEGY_OB_ENABLED
2. STRATEGY_STR_ENABLED
3. SIGNAL_BYPASS_ENABLED
4. SIGNAL_BYPASS_RSI_OVERSOLD
5. SIGNAL_BYPASS_RSI_OVERBOUGHT
6. ML_REGIME_MIN_CONFIDENCE
7. ML_RL_PPO_ENABLED
8. ML_RL_TRAINING_MODE
```

**Benefit**: Enable/disable major strategies without code changes

### Phase 2: Operational Parameters (12 settings)
Add in next iteration:
```
9-20: Adaptive strategies, risk/reward, RSI thresholds, ML confidence
```

**Benefit**: Fine-tune trade quality without redeploy

### Phase 3: Technical Parameters (25+ settings)
Add for advanced users:
```
Feature windows, multipliers, learning rates, epsilon decay
```

**Benefit**: Advanced optimization for power users

### Phase 4: Consider Adding (Requires Caution)
```
ML architecture parameters (LSTM hidden, layers)
Model paths and feature sizes
⚠️ ONLY if you can retrain models when changed
```

---

## ⚠️ Critical Dependencies

### Parameters That MUST Match Trained Models
These CANNOT be changed without retraining:

| Parameter | Line | Impact |
|-----------|------|--------|
| `ML_FEATURE_SIZE` | 404 | Price prediction model input dimension |
| `ML_LSTM_HIDDEN` | 443 | Regime prediction model architecture |
| `ML_LSTM_LAYERS` | 444 | Regime prediction model architecture |
| `ML_LSTM_DROPOUT` | 445 | Regime prediction model architecture |
| Model ensemble weights | 410-412 | Affects prediction blend |

**Safeguard**: Add documentation in App Config or comments

### Parameters That Can Be Adjusted Safely
These don't require model retraining:

| Category | Examples |
|----------|----------|
| Feature periods | RSI period, ATR period, MACD parameters |
| Thresholds | RSI levels, R/R ratios, confidence thresholds |
| Multipliers | TP/SL ATR multipliers, regime bias |
| Flags | Enable/disable features |
| Learning rates | Training hyperparameters |

---

## 📋 Implementation Checklist

### Before Migration
- [ ] Review `config.example.yaml` lines 230-525 (done ✓)
- [ ] Identify critical dependencies
- [ ] Determine training cost of parameter changes
- [ ] Plan rollback strategy

### Phase 1: Add 8 Critical Feature Flags
- [ ] Create migration script extension (add 8 settings)
- [ ] Document each setting in Azure Portal
- [ ] Tag as "strategy" in App Config
- [ ] Test in staging environment
- [ ] Deploy to production

### Phase 2: Add 12 Operational Parameters
- [ ] Extend migration script for Phase 2
- [ ] Create labels: production, staging, tuning
- [ ] Document tuning guidelines
- [ ] Implement monitoring for impacts
- [ ] Gradually roll out to teams

### Phase 3+: Advanced Parameters
- [ ] Document model compatibility requirements
- [ ] Create validation checks
- [ ] Implement safeguards for critical parameters
- [ ] Plan training workflow integration

---

## 🔄 Integration Points

### Where These Settings Are Used
1. **Signal Generation**: `src/core/strategy_coordinator.py`
   - Uses: Adaptive strategies, signal bypass, RSI thresholds

2. **ML Predictions**: `src/ml/` modules
   - Uses: Feature parameters, model settings, RL parameters

3. **Risk Management**: `src/core/risk_manager.py`
   - Uses: Position multipliers, R/R ratios

4. **Trade Execution**: `src/core/live_trading_engine.py`
   - Uses: All above settings

### How to Search in Code
```bash
# Find where a setting is used
grep -r "SIGNAL_BYPASS_ENABLED" src/
grep -r "ML_REGIME_MIN_CONFIDENCE" src/
grep -r "STRATEGY_OB_ENABLED" src/
```

---

## 💡 Recommendation

### Immediate Action (Phase 1)
Add these 8 settings to Azure App Configuration:
```
STRATEGY_OB_ENABLED
STRATEGY_STR_ENABLED
SIGNAL_BYPASS_ENABLED
SIGNAL_BYPASS_RSI_OVERSOLD
SIGNAL_BYPASS_RSI_OVERBOUGHT
ML_REGIME_MIN_CONFIDENCE
ML_RL_PPO_ENABLED
ML_RL_TRAINING_MODE
```

**Time to implement**: 30 minutes (extend migration script + test)  
**Value**: Ability to toggle major strategies without redeploy

### Future Additions
Phase 2 & 3 can follow after assessing impact and benefits

---

## 📚 References

- **config.example.yaml**: Lines 230-525 (strategy settings)
- **Related files**:
  - `src/core/strategy_coordinator.py` (signal generation)
  - `src/ml/regime_predictor.py` (regime classification)
  - `src/ml/rl_agent.py` (reinforcement learning)
  - `src/core/risk_manager.py` (risk parameters)

---

**Status**: Analysis Complete ✓  
**Recommended Next Step**: Implement Phase 1 migration script extension
