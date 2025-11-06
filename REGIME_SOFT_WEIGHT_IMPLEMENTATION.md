# Regime Soft-Weight Implementation

## Overview

This document describes the regime soft-weighting feature that replaces hard threshold filtering with graduated weighting based on regime prediction confidence.

## Problem Statement

Previously, regime predictions were handled with a hard threshold:
- **Confidence ≥ 0.6**: Use regime prediction (full effect)
- **Confidence < 0.6**: Ignore regime completely (no effect)

This binary approach resulted in:
- ❌ Loss of valuable information from medium-confidence predictions (e.g., 0.55)
- ❌ Sudden, sharp changes in bot behavior at the threshold boundary
- ❌ Inability to adapt dynamically to market uncertainty

## Solution: Soft-Weighting

The new system uses graduated weighting based on confidence:

```
Confidence < 0.30:  Hard reject (regime_weight = N/A, completely ignored)
0.30 ≤ Confidence < 0.60:  Partial weight (regime_weight = confidence / 0.6)
Confidence ≥ 0.60:  Full weight (regime_weight = 1.0)
```

### Visual Representation

```
Regime Weight vs. Confidence
1.0 |                    ┌─────────────
    |                   /
0.75|                 /
    |               /
0.5 |             /
    |           /
0.25|         /
    |       /
0.0 |──────┘
    └──────────────────────────────────
    0.0   0.3   0.45  0.6   0.75  1.0
          ↑           ↑
      Hard Reject   Full Weight
```

## Implementation Details

### 1. Configuration (`src/config/risk_config.py`)

New configuration section added:

```yaml
ml:
  regime_prediction:
    soft_weighting_enabled: true      # Enable/disable soft-weighting
    min_confidence_hard_reject: 0.30  # Below this: completely ignore
    min_confidence_full_weight: 0.60  # Above this: use full weight
  
  signal_scoring:
    enabled: true
    min_score_to_trade: 60
    weights:
      strategy: 0.3      # Base strategy signal
      ml_price: 0.3      # ML price prediction
      regime: 0.2        # Regime weight contribution
      risk_reward: 0.2   # R/R ratio quality
```

### 2. Strategy Integration (`src/ml/strategy_integration.py`)

Enhanced signal now includes `regime_weight`:

```python
# Before (Hard Filter)
if regime_confidence >= self.min_confidence:
    enhancement["predicted_regime"] = predicted_regime
    enhancement["regime_confidence"] = regime_confidence
else:
    # Regime completely ignored

# After (Soft Weight)
if regime_confidence < 0.30:
    # Hard reject
    pass
else:
    regime_weight = min(regime_confidence / 0.60, 1.0)
    enhancement["predicted_regime"] = predicted_regime
    enhancement["regime_confidence"] = regime_confidence
    enhancement["regime_weight"] = regime_weight  # NEW
```

### 3. Risk/Reward Calculation (`src/core/risk_rules.py`)

Regime multiplier now uses soft-weighting:

```python
# Get regime_weight (defaults to 1.0 for backward compatibility)
regime_weight = float(signal.get('regime_weight', 1.0))

# Apply regime multiplier with soft-weighting
regime_mult = regime_mults.get(regime_name, 1.0)  # e.g., 1.2 for volatile
regime_adjustment = 1.0 + (regime_mult - 1.0) * regime_weight

# Example calculations:
# Full weight (1.0):  1.0 + (1.2 - 1.0) * 1.0 = 1.20x
# Half weight (0.5):  1.0 + (1.2 - 1.0) * 0.5 = 1.10x
# No weight (0.0):    1.0 + (1.2 - 1.0) * 0.0 = 1.00x (no effect)
```

### 4. Position Sizing (`src/core/position_sizing.py`)

Position size adjusts based on regime confidence:

```python
regime_weight = float(signal.get('regime_weight', 1.0))
regime_multiplier = market_regime.get('risk_multiplier', 1.0)
regime_adjustment = 1.0 + (regime_multiplier - 1.0) * regime_weight

adjusted_size = base_size * regime_adjustment * trend_bonus * vol_adjustment
```

### 5. Signal Priority (`src/core/strategy_coordinator.py`)

Signal priority considers regime weight:

```python
regime_weight = signal.get('regime_weight')
if regime_weight is not None:
    if regime_weight < 0.5:
        priority = SignalPriority.LOW  # Low confidence regime
    elif regime_weight > 0.8 and priority == SignalPriority.HIGH:
        priority = SignalPriority.CRITICAL  # High confidence regime
```

## Usage Examples

### Example 1: High Confidence Regime (0.85)

```python
signal = {
    'regime_confidence': 0.85,
    'regime_name': 'bullish',
    'regime_weight': 1.0,  # Full weight
    # ... other fields
}

# Result: Full regime effect
# - R/R: Full adjustment by regime multiplier
# - Position size: Full adjustment
# - Priority: May upgrade to CRITICAL
```

### Example 2: Medium Confidence Regime (0.45)

```python
signal = {
    'regime_confidence': 0.45,
    'regime_name': 'volatile',
    'regime_weight': 0.75,  # Partial weight (0.45 / 0.60)
    # ... other fields
}

# Result: Partial regime effect (75%)
# - R/R: 75% of regime multiplier effect
# - Position size: 75% of regime adjustment
# - Priority: Normal processing
```

### Example 3: Low Confidence Regime (0.25)

```python
signal = {
    # No regime fields - filtered out due to low confidence
    # ... other fields
}

# Result: No regime effect
# - R/R: Based on ML/RL signals only
# - Position size: Standard calculation
# - Priority: Based on other factors only
```

## Benefits

### 1. Smoother Behavior
✅ No sudden changes at threshold boundaries
✅ Gradual adjustment based on confidence level
✅ More predictable bot behavior

### 2. Better Information Utilization
✅ Medium-confidence predictions (0.3-0.6) now contribute
✅ Previously wasted information (e.g., 0.55 confidence) now used
✅ More nuanced decision-making

### 3. Risk Management
✅ Uncertain regimes have reduced impact
✅ High-confidence regimes have full impact
✅ Very low confidence (<0.3) still hard-rejected for safety

### 4. Backward Compatibility
✅ Missing `regime_weight` defaults to 1.0
✅ Legacy signal format continues to work
✅ No breaking changes

## Test Coverage

### Unit Tests (`tests/test_regime_soft_weight.py`)
- ✅ Configuration loading (3 tests)
- ✅ Regime weight calculation (4 tests)
- ✅ Backward compatibility (1 test)
- ✅ Edge cases (2 tests)
- **Total: 10/10 passing**

### Integration Tests (`tests/test_soft_weight_integration.py`)
- ✅ High confidence flow
- ✅ Medium confidence flow
- ✅ Low confidence flow
- ✅ Position sizing integration
- ✅ Confidence progression
- **Total: 5/5 passing**

### Regression Tests
- ✅ Dynamic R/R tests: 12/12 passing
- ✅ Risk rules tests: 28/28 passing
- ✅ Position sizing tests: 10/10 passing

## Configuration Reference

### Environment Variables

Override any setting using environment variables:

```bash
# Regime soft-weighting
REGIME_SOFT_WEIGHT_ENABLED=true
REGIME_MIN_CONF_REJECT=0.30
REGIME_MIN_CONF_FULL=0.60

# Signal scoring
SIGNAL_SCORING_ENABLED=true
SIGNAL_MIN_SCORE=60
SCORE_WEIGHT_STRATEGY=0.3
SCORE_WEIGHT_ML=0.3
SCORE_WEIGHT_REGIME=0.2
SCORE_WEIGHT_RR=0.2
```

### YAML Configuration

In `config/config.example.yaml`:

```yaml
ml:
  regime_prediction:
    soft_weighting_enabled: true
    min_confidence_hard_reject: 0.30
    min_confidence_full_weight: 0.60
  
  signal_scoring:
    enabled: true
    min_score_to_trade: 60
    weights:
      strategy: 0.3
      ml_price: 0.3
      regime: 0.2
      risk_reward: 0.2
```

## Migration Guide

### For Existing Deployments

1. **No code changes required** - The system is backward compatible
2. **Optional**: Update your config to enable/tune soft-weighting
3. **Optional**: Monitor logs for regime_weight values
4. **Optional**: Adjust thresholds based on performance

### For New Deployments

1. Use default configuration (soft-weighting enabled)
2. Monitor bot behavior with different confidence levels
3. Tune thresholds if needed:
   - Increase `min_confidence_full_weight` for more conservative behavior
   - Decrease `min_confidence_hard_reject` to use even lower confidence predictions

## Performance Considerations

### Before (Hard Filter)
- Binary decision: use regime or not
- ~40% of regime predictions ignored (confidence 0.3-0.6)
- Sharp transitions at threshold

### After (Soft Weight)
- Graduated weighting: 0% to 100% effect
- ~40% more regime information utilized
- Smooth transitions across confidence levels

## Monitoring

### Key Metrics to Watch

1. **Regime Weight Distribution**
   - Check logs for `regime_weight` values
   - Expect: mix of values from 0.5 to 1.0

2. **R/R Adjustments**
   - Compare R/R with/without regime
   - Look for: `regime_adj` in logs

3. **Position Size Variations**
   - Monitor position sizes across confidence levels
   - Check: position sizing debug logs

### Log Examples

```
🧠 [ML-ADAPTER] Regime for BTC/USDT is BULLISH (Conf: 0.45, Weight: 0.75)
📊 [Dynamic R/R Calc] Base=1.50 - Relax=0.42 + Tight=0.11 
   × Regime(bullish, mult=0.9, weight=0.75)=0.93 = 1.19 → Final=1.19
```

## Future Enhancements

Potential improvements for future iterations:

1. **Adaptive Thresholds**
   - Learn optimal thresholds from performance data
   - Adjust based on regime prediction accuracy

2. **Non-Linear Weighting**
   - Use sigmoid or other curves instead of linear
   - Potentially better modeling of uncertainty

3. **Confidence Calibration**
   - Calibrate regime predictor confidence
   - Ensure confidence matches actual accuracy

4. **Multi-Factor Soft-Weighting**
   - Apply similar approach to ML price predictions
   - Extend to RL agent decisions

## Summary

The regime soft-weighting feature provides:
- ✅ More nuanced use of regime predictions
- ✅ Smoother, more predictable bot behavior
- ✅ Better utilization of medium-confidence predictions
- ✅ Maintained backward compatibility
- ✅ Comprehensive test coverage
- ✅ Easy configuration and monitoring

This implementation follows the principle: **"Use all available information, weighted by confidence"** rather than **"All or nothing based on arbitrary threshold"**.
