# GEMMA Workflow Verification Guide

## Overview

This guide explains how to use the `verify_gemma_workflow.py` script to validate the GEMMA tuning workflow locally before executing it on GitHub Actions.

## Prerequisites

### 1. Python 3.11 Environment

**CRITICAL:** This project requires Python 3.11. Other versions are not supported.

```bash
# Check Python version
python --version  # Should show: Python 3.11.x

# If using pyenv
pyenv install 3.11
pyenv local 3.11

# If using conda
conda create -n bearish-bot python=3.11
conda activate bearish-bot
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python3.11 -m venv venv311
source venv311/bin/activate  # On Windows: venv311\Scripts\activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration Files

Ensure you have:
- `.env` file (can use minimal configuration for testing)
- `config.yaml` file (copy from `config/config.example.yaml`)

## Usage

### Quick Start (Dry Run)

Execute a quick validation with only 3 trials (~30 seconds):

```bash
cd /path/to/bearish-alpha-bot
source venv311/bin/activate
python scripts/verify_gemma_workflow.py
```

This will:
1. Prepare training data (87 features)
2. Run feature selection (~82 features)
3. Re-prepare data with selected features
4. Run GEMMA tuning with 3 trials
5. Analyze results
6. Validate all artifacts

### Full Validation

Execute a complete validation with 30 trials (~15-20 minutes):

```bash
python scripts/verify_gemma_workflow.py --full
```

### Custom Number of Trials

```bash
python scripts/verify_gemma_workflow.py --full --trials 50
```

## What the Script Does

### Step 1: Data Preparation

**Command executed:**
```bash
python scripts/prepare_training_data.py --symbol BTC/USDT
```

**Expected output:**
- Creates `data/cache/BTC-USDT_training_data.npz`
- ~6,830 samples from 5 timeframes (15m, 30m, 1h, 4h, 1d)
- 87 features initially extracted
- Balanced label distribution

**Validation checks:**
- ✅ File created: `data/cache/BTC-USDT_training_data.npz`
- ✅ Feature count: 85-90 (expected ~87)
- ✅ Sample count: > 5000

### Step 2: Feature Selection

**Commands executed:**
```bash
# Analyze features
python scripts/analyze_features.py --analyze

# Select features
python scripts/analyze_features.py --select \
  --variance-threshold 0.00005 \
  --correlation-threshold 0.005
```

**Expected output:**
- Creates `data/cache/feature_selection_mask.npy`
- Selects ~82-83 features (5 low-quality features filtered)
- Analysis reports saved

**Validation checks:**
- ✅ File created: `data/cache/feature_selection_mask.npy`
- ✅ Selected features: 80-85 (expected ~82)

### Step 3: Data Re-preparation

**Command executed:**
```bash
python scripts/prepare_training_data.py --symbol BTC/USDT
```

**Expected output:**
- Updates `data/cache/BTC-USDT_training_data.npz` with selected features
- Data now has 82 features instead of 87

**Validation checks:**
- ✅ File updated: `data/cache/BTC-USDT_training_data.npz`
- ✅ Feature count matches selection: 80-85

### Step 4: GEMMA Tuning

**Command executed:**
```bash
python scripts/tune_gemma_model_standalone.py \
  --model gemma \
  --symbol BTC/USDT \
  --trials 3 \
  --cv-splits 3
```

**Expected output:**
- Optuna optimization runs 3 trials (or more if --full)
- Each trial tests different hyperparameters
- Best parameters saved to JSON
- Creates `logs/tuning_results/gemma_tuning_*.json`

**Validation checks:**
- ✅ Optuna completes all trials
- ✅ No errors during training
- ✅ Results file created

### Step 5: Results Analysis

**Expected output:**
- Reads latest `gemma_tuning_*.json` file
- Extracts key metrics:
  - `balanced_cv_score`: CV accuracy (balanced)
  - `balanced_holdout_score`: Holdout accuracy (balanced)
  - `gap`: Generalization gap
  - `best_params`: Optimal hyperparameters

**Validation checks:**
- ✅ Balanced holdout score ≥ 40%
- ✅ Generalization gap < 10%
- ✅ Input size matches feature count

### Step 6: Artifact Validation

**Expected artifacts:**
- `data/cache/BTC-USDT_training_data.npz` - Training data
- `data/cache/feature_selection_mask.npy` - Feature mask
- `data/cache/scaler_production.joblib` - Production scaler
- `logs/tuning_results/gemma_tuning_*.json` - Results

**Validation checks:**
- ✅ All files exist
- ✅ Files have non-zero size
- ✅ Can be loaded without errors

## Output Files

### Verification Reports

The script generates two types of reports:

1. **Text Report** (`logs/gemma_workflow_verification_*.txt`)
   - Human-readable summary
   - Step-by-step results
   - Success/failure status
   - Recommendations

2. **JSON Report** (`logs/gemma_workflow_verification_*.json`)
   - Machine-readable results
   - Detailed metrics
   - Timing information
   - Complete validation data

### Report Location

```
logs/
├── gemma_workflow_verification_20251112_194205.txt
├── gemma_workflow_verification_20251112_194205.json
└── tuning_results/
    └── gemma_tuning_20251112_194204.json
```

## Understanding Results

### Success Criteria

The verification passes if ALL of these are true:

1. ✅ **All 6 steps complete without errors**
2. ✅ **Balanced holdout score ≥ 40%**
   - Shows model performs better than random (33.3% for 3 classes)
3. ✅ **Generalization gap < 10%**
   - Shows model isn't overfitting or underfitting
4. ✅ **All artifacts created and valid**

### Example: Successful Output

```
======================================================================
Phase 3.5: GEMMA Workflow Verification Report
======================================================================

**Completion Date:** 2025-11-12 19:42:05
**Executor:** @github-copilot (automated verification)
**Mode:** DRY RUN (3 trials)

### ✅ Overall Status: SUCCESS

---

### Step-by-Step Verification

| Adım | Beklenen Sonuç | Gerçekleşen Durum |
| :--- | :--- | :--- |
| **Step 1** | 87 özellikli .npz dosyası oluşturuldu. | ✅ Başarılı, **87** özellik |
| **Step 2b** | feature_mask.npy oluşturuldu (~82-83 özellik). | ✅ Başarılı, **82** özellik seçildi |
| **Step 3** | Nihai .npz dosyası seçilmiş özelliklerle oluşturuldu. | ✅ Başarılı, **82** özellik |
| **Step 4** | tune_gemma_model_standalone.py çalıştı ve Optuna tamamlandı. | ✅ Başarılı |
| **Step 5** | .json okundu ve metrikler raporlandı. | ✅ Başarılı, balanced_holdout_score: **81.78%** |
| **Step 6** | gemma-tuning-results ve production-scaler yüklendi. | ✅ Başarılı |
```

### Key Metrics Explained

#### Balanced CV Score (79.84%)
- Cross-validation accuracy using balanced accuracy metric
- Accounts for class imbalance (Neutral class is dominant)
- Higher is better (> 50% is good, > 70% is excellent)

#### Balanced Holdout Score (81.78%)
- Hold-out test set accuracy using balanced accuracy
- More reliable than CV score for production prediction
- **Primary metric for production readiness**
- Requirement: ≥ 40% (random = 33.3%)

#### Generalization Gap (-1.94%)
- Difference between CV and holdout scores
- Negative gap = model generalizes better on unseen data
- Positive gap < 10% = acceptable generalization
- **Shows no overfitting**

## Troubleshooting

### Issue: "Training data file not created"

**Cause:** Network error fetching market data or feature extraction failed

**Solution:**
```bash
# Try running manually
python scripts/prepare_training_data.py --symbol BTC/USDT

# Check logs for specific error
```

### Issue: "Feature selection mask not found"

**Cause:** Feature analysis step failed

**Solution:**
```bash
# Run feature analysis manually
python scripts/analyze_features.py --analyze
python scripts/analyze_features.py --select \
  --variance-threshold 0.00005 \
  --correlation-threshold 0.005
```

### Issue: "Optuna trials failing"

**Cause:** Insufficient data or memory issues

**Solution:**
```bash
# Check available memory
free -h

# Try with fewer CV splits
python scripts/tune_gemma_model_standalone.py \
  --model gemma \
  --symbol BTC/USDT \
  --trials 3 \
  --cv-splits 2  # Reduced from 3
```

### Issue: "Python version mismatch"

**Cause:** Wrong Python version active

**Solution:**
```bash
# Ensure Python 3.11 is active
python --version  # Must show 3.11.x

# If not, activate correct environment
source venv311/bin/activate
```

## Advanced Usage

### Custom Thresholds

Modify feature selection thresholds:

```python
# Edit verify_gemma_workflow.py, step2_feature_selection method
cmd_select = [
    sys.executable,
    'scripts/analyze_features.py',
    '--select',
    '--variance-threshold', '0.0001',      # Stricter
    '--correlation-threshold', '0.01'      # Stricter
]
```

### Different Symbols

Test with different trading pairs:

```python
# Edit verify_gemma_workflow.py, step1_prepare_data method
cmd = [
    sys.executable,
    'scripts/prepare_training_data.py',
    '--symbol', 'ETH/USDT'  # Change symbol
]
```

### Timeout Adjustments

Increase timeouts for slower systems:

```python
# Edit verify_gemma_workflow.py, step4_gemma_tuning method
timeout = 3600 if not self.dry_run else 1200  # 60 min full, 20 min dry run
```

## Integration with GitHub Actions

After successful local verification:

1. **Commit changes** (if any modifications made)
2. **Push to repository**
3. **Navigate to GitHub Actions**
4. **Select workflow:** `💎 GEMMA - Full Hyperparameter Tuning`
5. **Click "Run workflow"**
6. **Set parameters:**
   - Symbol: `BTC/USDT`
   - Trials: `30`
   - CV Splits: `5`
7. **Monitor execution** (~45-60 minutes)
8. **Download artifacts** when complete

## Best Practices

### Before Running on GitHub Actions

1. ✅ Run local dry run first (`python scripts/verify_gemma_workflow.py`)
2. ✅ Verify all 6 steps complete successfully
3. ✅ Check that metrics are reasonable
4. ✅ Ensure all artifacts are generated

### After Successful Local Verification

1. ✅ Run full workflow on GitHub Actions with 30 trials
2. ✅ Monitor for any errors or warnings
3. ✅ Download and review artifacts
4. ✅ Validate production readiness (balanced_holdout_score ≥ 45%)

### Regular Maintenance

1. 🔄 Run verification after major code changes
2. 🔄 Test with different market conditions
3. 🔄 Update thresholds based on production feedback
4. 🔄 Archive successful verification reports

## FAQ

### Q: Why use a dry run first?

**A:** Dry runs (3 trials) complete in ~30 seconds and verify the infrastructure works. Full runs (30 trials) take ~15-20 minutes. Always validate infrastructure first.

### Q: What if balanced holdout score is low (< 40%)?

**A:** This indicates the model isn't learning well. Try:
- Increasing trials (50-100)
- Adjusting feature selection thresholds
- Checking data quality
- Reviewing feature engineering

### Q: Can I run this on Windows?

**A:** Yes, but use `venv311\Scripts\activate` instead of `source venv311/bin/activate`

### Q: How much disk space is needed?

**A:** Approximately 500 MB for:
- Dependencies: ~400 MB
- Training data: ~5 MB
- Models and results: ~50 MB
- Logs: ~10 MB

### Q: Can I run multiple verifications in parallel?

**A:** Not recommended. Each run modifies the same cache files. Run sequentially.

## Support

For issues or questions:
1. Check this guide first
2. Review error logs in `logs/`
3. Check verification report JSON for details
4. Open an issue on GitHub with:
   - Verification report
   - Error logs
   - Python version
   - Operating system

---

**Last Updated:** 2025-11-12  
**Script Version:** 1.0  
**Compatible with:** Python 3.11.x  
**Workflow Version:** full-gemma-tuning.yml (Phase 3.5)
