# GEMMA Dynamic Feature Adaptation System

## Overview

The GEMMA Dynamic Feature Adaptation System enables the bearish-alpha-bot to automatically adjust to different model bundles with varying feature counts without requiring code changes. This system eliminates hardcoded feature dimensions and enables seamless model upgrades.

## Key Features

### 1. Manifest-Driven Configuration
All ML components load their configuration from a central manifest file:
- **Feature Count**: Dynamic, loaded from manifest
- **Feature Names**: Ordered list ensuring consistency
- **Model Paths**: Relative to bundle directory
- **RL State Size**: Configurable per bundle

### 2. Multi-Mode Feature Extraction
```python
# Extract price prediction features
features_price = pipeline.extract_features(df, mode='price')

# Extract regime prediction features  
features_regime = pipeline.extract_features(df, mode='regime')

# Extract all features
features_all = pipeline.extract_features(df, mode='all')
```

### 3. Shadow Mode Deployment
GEMMA adapter supports safe deployment through shadow mode:
- **Shadow Mode**: Predictions logged but not used (validation)
- **Active Mode**: Predictions actively used in trading decisions

### 4. Dimension Validation
All components validate model dimensions before loading:
- Scaler dimension checking
- Model input size verification
- Graceful handling of mismatches

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Config (config.yaml)                     │
│  models:                                                     │
│    active_bundle: "artifacts/legacy"                        │
│    gemma:                                                    │
│      shadow_mode: true                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            ManifestManager (Singleton)                       │
│  - Load manifest.json from bundle                           │
│  - Cache feature mappings                                    │
│  - Provide feature lists by mode                            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Feature    │  │   Regime     │  │  RL Agent    │
│ Engineering  │  │  Predictor   │  │              │
│              │  │              │  │              │
│ - 42 or 87   │  │ - Dynamic    │  │ - Dynamic    │
│   features   │  │   input size │  │   state size │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Bundle Structure

```
artifacts/
├── legacy/                    # Legacy 42-feature bundle
│   ├── manifest.json         # Manifest file
│   └── scaler.pkl           # Optional: model files
└── gemma/                    # GEMMA 87-feature bundle
    ├── manifest.json
    ├── scaler.pkl
    └── model.pt
```

### Manifest Format

```json
{
  "version": "1.0-legacy",
  "mode": "legacy",
  "feature_count": 42,
  "feature_names_ordered": ["feature_0", "feature_1", ...],
  "selected_features_price": [0, 1, 2, ...],
  "selected_features_regime": [0, 1, 2, ...],
  "rl_state_size": 42,
  "regime_scaler_path": "scaler.pkl",
  "regime_model_path": "lstm_regime.pth",
  "rl_model_path": "../../data/models/rl_agent_final.pth",
  "metadata": {
    "system": "legacy",
    "description": "Legacy 42-feature system"
  }
}
```

## Configuration

### Basic Configuration

```yaml
# config.yaml
models:
  # Active model bundle
  active_bundle: "artifacts/legacy"
  
  # Fallback bundle if active bundle fails
  fallback_bundle: "artifacts/legacy"
  
  # Deployment settings
  deployment:
    validation_mode: "strict"      # strict | lenient
    allow_missing_features: false
    canary_percentage: 0           # 0-100
  
  # GEMMA configuration
  gemma:
    use_manifest: true
    shadow_mode: true              # Start with shadow mode
```

### Switching Bundles

To switch from legacy (42 features) to GEMMA (87 features):

```yaml
models:
  active_bundle: "artifacts/gemma"  # ← Just change this!
  gemma:
    shadow_mode: false              # ← Activate when ready
```

## Usage

### 1. Initialize Components

```python
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.regime_predictor import MLRegimePredictor
from src.ml.reinforcement_learning import TradingRLAgent

# Config with bundle path
config = {
    'models': {'active_bundle': 'artifacts/legacy'}
}

# Components automatically load from manifest
feature_pipeline = FeatureEngineeringPipeline(config)
regime_predictor = MLRegimePredictor(feature_pipeline, config)
rl_agent = TradingRLAgent(config=config)  # state_size loaded from manifest
```

### 2. Extract Features

```python
# For price prediction
price_features = feature_pipeline.extract_features(df, mode='price')

# For regime prediction
regime_features = feature_pipeline.extract_features(df, mode='regime')
```

### 3. GEMMA Integration

```python
# GEMMA automatically uses manifest configuration
# In shadow mode: predictions logged but not used
# In active mode: predictions used for trading
```

## Health Checks

### Run Health Check

```bash
# Check legacy bundle
python scripts/gemma_manifest_health_check_standalone.py --bundle artifacts/legacy

# Check GEMMA bundle
python scripts/gemma_manifest_health_check_standalone.py --bundle artifacts/gemma

# Output as JSON
python scripts/gemma_manifest_health_check_standalone.py --json
```

### Health Check Results

```
======================================================================
🏥 GEMMA Manifest Health Check
======================================================================
✅ Manifest Existence: Manifest found and valid JSON
✅ Manifest Structure: All required fields present
✅ Feature Count Consistency: Feature count consistent: 42 features
✅ Selected Features Validity: All selected features valid
⚠️  Model Files Existence: Found 1 files, missing 3 optional files

Summary: 4/5 passed, 1 warning, 0 failed
Status: ✅ PASSED
```

## Migration Guide

### Phase 1: Legacy Mode (Current State)
- Bundle: `artifacts/legacy`
- Features: 42
- GEMMA: Shadow mode only

### Phase 2: GEMMA Validation
```yaml
models:
  active_bundle: "artifacts/gemma"  # Switch to GEMMA bundle
  gemma:
    shadow_mode: true                # Validate predictions
```

**Actions:**
1. Create GEMMA bundle with 87 features
2. Train models on 87 features
3. Deploy with shadow_mode=true
4. Monitor shadow predictions vs legacy

### Phase 3: Canary Deployment
```yaml
models:
  active_bundle: "artifacts/gemma"
  gemma:
    shadow_mode: false
    canary_percentage: 10            # Start with 10%
```

**Actions:**
1. Activate GEMMA for 10% of traffic
2. Monitor performance metrics
3. Gradually increase to 25%, 50%, 100%

### Phase 4: Full GEMMA
```yaml
models:
  active_bundle: "artifacts/gemma"
  gemma:
    shadow_mode: false
    canary_percentage: 100           # Full deployment
```

## Testing

### Run Integration Tests

```bash
pytest tests/test_gemma_dynamic_integration.py -v
```

### Test Coverage

- ✅ Manifest loading and validation
- ✅ Dynamic feature extraction
- ✅ Dimension handling
- ✅ Legacy compatibility
- ✅ Error handling
- ✅ Multi-mode feature selection

## Troubleshooting

### Issue: Feature Count Mismatch

**Symptom:**
```
ValueError: X has 87 features, but StandardScaler is expecting 42 features
```

**Solution:**
1. Check manifest feature_count matches your data
2. Verify bundle path is correct
3. Check scaler was trained with same feature count

### Issue: Model Won't Load

**Symptom:**
```
WARNING: RL model dimension mismatch
```

**Solution:**
1. Verify model was trained with manifest feature count
2. Check manifest rl_state_size matches model
3. Use lenient validation_mode during migration

### Issue: Missing Features

**Symptom:**
```
KeyError: Feature selection failed
```

**Solution:**
1. Verify feature_names_ordered in manifest
2. Check selected_features indices are valid
3. Ensure feature extraction produces expected features

## Best Practices

### 1. Always Use Health Checks
Run health check after creating/modifying manifests:
```bash
python scripts/gemma_manifest_health_check_standalone.py
```

### 2. Start with Shadow Mode
Always deploy new models in shadow mode first:
```yaml
gemma:
  shadow_mode: true  # Validate before activating
```

### 3. Validate Feature Consistency
Ensure training and inference use same features:
```python
# Training
features = pipeline.extract_features(df, mode='price')
# Inference (same mode!)
features = pipeline.extract_features(df, mode='price')
```

### 4. Use Canary Deployments
Roll out new models gradually:
```yaml
deployment:
  canary_percentage: 10  # Start small, increase gradually
```

## API Reference

### ManifestManager

```python
from src.ml.manifest_manager import ManifestManager

mgr = ManifestManager()  # Singleton
manifest = mgr.load_manifest('artifacts/legacy')
price_features = mgr.get_selected_features('price')
regime_features = mgr.get_selected_features('regime')
```

### FeatureEngineeringPipeline

```python
from src.ml.feature_engineering import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline(config)
features = pipeline.extract_features(df, mode='price')
# mode: 'price' | 'regime' | 'all'
```

### Dynamic Model Loading

All components support dynamic configuration:
- `MLRegimePredictor`: Loads input_size from manifest
- `TradingRLAgent`: Loads state_size from manifest
- `GemmaTorchScriptAdapter`: Loads feature_count from manifest

## Files Modified

### Core Components
- `src/ml/feature_engineering.py` - Dynamic feature extraction
- `src/ml/regime_predictor.py` - Manifest-driven loading
- `src/ml/reinforcement_learning.py` - Optional state_size
- `src/core/production_coordinator.py` - Manifest initialization
- `src/core/strategy_coordinator.py` - GEMMA shadow mode

### Configuration
- `config/config.example.yaml` - Bundle management config

### Tools
- `scripts/gemma_manifest_health_check_standalone.py` - Health validator
- `tests/test_gemma_dynamic_integration.py` - Integration tests

### Artifacts
- `artifacts/legacy/manifest.json` - Legacy bundle manifest

## Support

For issues or questions:
1. Check health check output
2. Review logs for dimension mismatches
3. Verify manifest structure
4. Test with standalone health check tool

## Version History

- **v1.0** (2025-11-15): Initial implementation
  - Manifest-driven configuration
  - Dynamic feature extraction
  - Shadow mode support
  - Health check tools
  - Legacy compatibility
