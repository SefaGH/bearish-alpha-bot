# GEMMA Feature Engineering - Phase 2

This directory contains feature lists and metadata for the GEMMA (87-feature) integration with Bearish Alpha Bot.

## 📁 Directory Structure

```
features/gemma/
├── selected/
│   ├── gemma_full_87.json          # Complete 87-feature list
│   ├── gemma_price_selected_82.json    # 82 features for price prediction
│   └── gemma_regime_selected_82.json   # 82 features for regime prediction
└── metadata/
    └── feature_metadata.json       # Statistics and exclusion details
```

## 🎯 Feature Breakdown

### Full Set (87 features)

1. **Price-based features (30)**
   - Simple Moving Averages (SMA) for periods: 5, 10, 15, 20, 30
   - Exponential Moving Averages (EMA) for periods: 5, 10, 15, 20, 30
   - Relative Strength Index (RSI) for periods: 5, 10, 15, 20, 30
   - Stochastic Oscillator (K & D) for periods: 5, 10, 15, 20, 30
   - Williams %R for periods: 5, 10, 15, 20, 30

2. **Volume-based features (15)**
   - Volume SMA for periods: 5, 10, 15
   - Volume ratio for periods: 5, 10, 15
   - On-Balance Volume (OBV) for periods: 5, 10, 15
   - Money Flow Index (MFI) for periods: 5, 10, 15
   - Volume Weighted Average Price (VWAP) for periods: 5, 10, 15

3. **Volatility features (20)**
   - Bollinger Bands (upper, middle, lower, width, position) for periods: 10, 20
   - Average True Range (ATR) for periods: 10, 20
   - Volatility (standard deviation) for periods: 10, 20
   - Keltner Channels (upper, lower) for periods: 10, 20
   - Donchian Channels for periods: 10, 20

4. **Trend features (12)**
   - MACD (line, signal, histogram)
   - Average Directional Index (ADX) at period 14
   - Directional Indicators (+DI, -DI) at period 14
   - Commodity Channel Index (CCI) at period 20
   - Rate of Change (ROC) at period 10
   - Momentum at period 10
   - TRIX at period 15
   - Detrended Price Oscillator (DPO) at period 20
   - Vortex Indicator (positive) at period 14

5. **Market structure features (10)**
   - Support/Resistance distance
   - Pivot points (pivot, R1, S1)
   - Fibonacci levels (38.2%, 50%, 61.8%)
   - Trend strength
   - Market phase

### Selected Set (82 features)

The production feature set excludes 5 features from the full 87:
- `donchian_10`
- `donchian_20`
- `trix_15`
- `dpo_20`
- `vortex_pos_14`

These were excluded based on feature selection analysis to optimize model performance.

## 🔧 Usage

### Generating Feature Lists

```bash
# Run the feature generator script
python scripts/generate_gemma_features.py
```

This will create/update all JSON files and the numpy mask.

### Using in Feature Engineering

The `FeatureEngineeringPipeline` automatically uses GEMMA features when enabled in config:

```python
from src.ml.feature_engineering import FeatureEngineeringPipeline

config = {
    'ml': {
        'gemma': {
            'enabled': True  # Enable 87-feature GEMMA mode
        }
    }
}

pipeline = FeatureEngineeringPipeline(config)
features = pipeline.extract_features(price_data)  # Returns 87 features
```

### Loading Feature Lists

```python
import json

# Load full feature list
with open('features/gemma/selected/gemma_full_87.json', 'r') as f:
    full_config = json.load(f)
    features = full_config['features']  # List of 87 feature names

# Load selected features for price model
with open('features/gemma/selected/gemma_price_selected_82.json', 'r') as f:
    price_config = json.load(f)
    features = price_config['features']  # List of 82 feature names
```

## 📊 Validation

Run the validation script to verify the setup:

```bash
python scripts/validate_gemma_phase2.py
```

This checks:
- ✅ All JSON files exist
- ✅ Feature counts are correct (87 full, 82 selected)
- ✅ Metadata is valid
- ✅ No duplicate features
- ✅ Excluded features are properly handled

## 🔄 Backward Compatibility

The implementation maintains full backward compatibility with the existing 42-feature system:
- When `gemma.enabled = false` (default), uses legacy feature extraction
- When `gemma.enabled = true`, uses new 87-feature extraction
- No changes required to existing code unless GEMMA is explicitly enabled

## 📝 Version

- **Version:** GEMMA-1.0.0
- **Phase:** 2 (Feature Engineering Integration)
- **Repository:** github.com/SefaGH/bearish-alpha-bot

## 🚀 Next Steps (Phase 3+)

1. Integrate GEMMA features with ML training pipeline
2. Create GEMMA-specific model architectures
3. Performance testing and optimization
4. Production deployment with feature selection
