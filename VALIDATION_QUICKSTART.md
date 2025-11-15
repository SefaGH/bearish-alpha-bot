# GEMMA Architecture Validation - Quick Start Guide

## 📋 Overview

This guide provides quick instructions for running the GEMMA manifest-driven architecture validation tests.

## 🔧 Prerequisites

1. **Python 3.11** (REQUIRED)
   ```bash
   python --version  # Must show Python 3.11.x
   ```

2. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running Tests

### Run All Validation Tests
```bash
# Set ML_ENABLED environment variable (required)
export ML_ENABLED=true

# Run comprehensive validation suite
python tests/validation/run_all_validation_tests.py
```

### Run Individual Test Suites

```bash
# Legacy system validation
python tests/validation/test_legacy_system.py

# ManifestManager tests
python tests/validation/test_manifest_manager.py

# Feature engineering tests
python tests/validation/test_feature_engineering.py

# Component compatibility tests
python tests/validation/test_component_compatibility.py

# Performance benchmarks
python tests/validation/test_performance.py
```

### Run Health Check
```bash
python scripts/gemma_manifest_health_check.py --bundle artifacts/legacy
```

## 📊 Expected Results

### All Tests Should Pass
```
============================================================
GEMMA INTEGRATION TEST SUMMARY
============================================================
Total Tests: 4
Passed: 4
Failed: 0
Total Time: ~1.0s

✅ SYSTEM READY FOR PRODUCTION
```

### Health Check Should Pass with Warnings
```
Total Checks:  7
✅ Passed:      5
⚠️  Warnings:    2
❌ Failed:      0

⚠️  Health check PASSED with warnings
```

**Note:** Warnings about missing model files are expected in test environment.

## 📁 Test Reports

Test reports are automatically saved to `test_reports/` directory:
```
test_reports/
  └── gemma_validation_YYYYMMDD_HHMMSS.json
```

## ✅ Validation Checklist

- [x] Python 3.11 installed and active
- [x] All dependencies installed
- [x] ML_ENABLED=true set
- [x] Legacy manifest exists (artifacts/legacy/manifest.json)
- [x] All validation tests pass
- [x] Health check passes (warnings acceptable)

## 🐛 Troubleshooting

### Issue: "MLRegimePredictor requires ML_ENABLED=true"
**Solution:** Set environment variable before running tests:
```bash
export ML_ENABLED=true
```

### Issue: "Module not found" errors
**Solution:** Install all dependencies:
```bash
pip install -r requirements.txt
pip install psutil  # For performance tests
```

### Issue: Wrong Python version
**Solution:** Use Python 3.11:
```bash
python3.11 -m venv venv311
source venv311/bin/activate
```

## 📖 Test Descriptions

### Task 1: Legacy System Validation
Verifies that the legacy 42-feature system is properly configured with valid manifest.

### Task 2: ManifestManager Functionality
Tests singleton pattern, thread safety, and feature name mapping.

### Task 3: Feature Engineering Dynamic Loading
Validates that feature extraction uses manifest-driven dimensions.

### Task 4: Component Dimension Compatibility
Ensures all ML components (regime predictor, RL agent, etc.) correctly load dimensions from manifest.

## 🎯 Success Criteria

✅ **All validation tests must pass**  
✅ **Health checks must pass (warnings acceptable)**  
✅ **No dimension mismatch errors**  
✅ **Thread safety verified**  
✅ **Performance within acceptable limits**

## 📚 Additional Documentation

- Full validation report: `GEMMA_VALIDATION_REPORT.md`
- Manifest structure: `artifacts/legacy/manifest.json`
- Health check script: `scripts/gemma_manifest_health_check.py`

## 🔗 Related Files

- `src/ml/manifest_manager.py` - ManifestManager implementation
- `src/ml/feature_engineering.py` - Feature extraction with manifest support
- `src/ml/regime_predictor.py` - Regime predictor with dynamic dimensions
- `src/ml/reinforcement_learning.py` - RL agent with manifest integration

## 💡 Best Practices

1. **Always run tests with ML_ENABLED=true**
2. **Run full validation suite before production deployment**
3. **Review test reports for any warnings**
4. **Monitor feature extraction performance in production**
5. **Keep manifest files in sync with model artifacts**

## 🆘 Support

For issues or questions:
1. Check `GEMMA_VALIDATION_REPORT.md` for detailed findings
2. Review individual test output for specific errors
3. Verify Python 3.11 and all dependencies are installed
4. Ensure `artifacts/legacy/manifest.json` exists and is valid

---

**Last Updated:** 2025-11-15  
**Version:** 1.0
