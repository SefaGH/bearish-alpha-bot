# Paper Trading Test - Quick Reference Guide

## 🚀 Quick Start (TL;DR)

### Prerequisites Check
```bash
python --version  # Must be 3.11.x
ls artifacts/legacy/manifest.json  # Must exist
```

### Run Short Test (5 minutes)
```bash
./scripts/run_short_test.sh
```

### Run Extended Test (1 hour)
```bash
./scripts/run_extended_test.sh
```

### Check Deployment Readiness
```bash
python scripts/production_readiness_decision.py
```

---

## 📋 Manual Testing (Step-by-Step)

### Step 1: Pre-Launch Verification (Required)
```bash
python scripts/verify_system_ready.py
```

**Expected Output:**
```
✅ python_version
✅ manifest_exists
✅ manifest_valid
✅ config_exists
✅ exchange_configured
✅ ml_components_ready
✅ disk_space_available
✅ memory_available

✅ SYSTEM READY FOR PAPER TRADING
```

**If Fails:**
- Python not 3.11? → Install Python 3.11: `pyenv install 3.11 && pyenv local 3.11`
- Manifest missing? → Run: `python scripts/create_legacy_manifest.py`
- ML components fail? → Reinstall: `pip install -r requirements.txt`

---

### Step 2: Launch Paper Trading Bot

#### Environment Setup
```bash
export ML_ENABLED=true
export GEMMA_ENABLED=false
export PAPER_TRADING=true
export DEBUG_MODE=true
export LOG_LEVEL=INFO
```

#### Start Bot (5 minutes)
```bash
nohup python scripts/live_trading_launcher.py \
    --paper \
    --debug \
    --duration 300 \
    --symbols "BTC/USDT,ETH/USDT" \
    > paper_trading_test_300s.log 2>&1 &

echo $! > paper_trading_bot.pid
```

#### Monitor Progress
```bash
# In another terminal
tail -f paper_trading_test_300s.log

# Or run monitoring script
python scripts/monitor_paper_trading.py paper_trading_test_300s.log 300
```

---

### Step 3: Analyze Results

#### Health Check
```bash
python scripts/paper_trading_health.py paper_trading_test_300s.log 300
```

**Look for:**
- ✅ Status: HEALTHY
- ✅ Feature Counts: [42]
- ✅ Dimension Errors: 0
- ✅ Errors per Hour: < 1.0

#### ML Component Verification
```bash
python scripts/verify_ml_live.py paper_trading_test_300s.log
```

**Look for:**
- ✅ Correct Extractions (42): > 0
- ✅ Wrong Extractions: 0
- ✅ Regime Predictor Errors: 0
- ✅ RL Agent Wrong State: 0

---

### Step 4: Make Deployment Decision
```bash
python scripts/production_readiness_decision.py
```

**Possible Decisions:**
- ✅ **GO** - Deploy to production
- ⚠️ **GO_WITH_CAUTION** - Deploy with close monitoring
- ⚠️ **NO_GO_FIX_WARNINGS** - Fix warnings first
- ❌ **NO_GO** - Critical issues, do not deploy

---

## 🔍 Common Issues & Solutions

### Issue: Bot doesn't start
```bash
# Check logs
tail -100 paper_trading_test_300s.log

# Common causes:
# 1. Python version wrong → Use Python 3.11
# 2. Dependencies missing → pip install -r requirements.txt
# 3. Config missing → Copy config.example.yaml to config.yaml
```

### Issue: Dimension mismatch errors
```bash
# Check manifest
cat artifacts/legacy/manifest.json | grep feature_count
# Should show: "feature_count": 42

# Verify feature extraction
grep "Extracted.*features" paper_trading_test_300s.log
# Should show: "Extracted 42 features"

# If mismatch persists:
# 1. Recreate manifest: python scripts/create_legacy_manifest.py
# 2. Check ML models are trained with 42 features
# 3. Review feature engineering pipeline
```

### Issue: High memory usage
```bash
# Check memory usage in report
cat paper_trading_report_*.json | grep -A5 "performance"

# If > 2GB:
# 1. Reduce number of symbols
# 2. Decrease batch sizes in ML models
# 3. Enable garbage collection
```

### Issue: High error rate
```bash
# Find errors
grep ERROR paper_trading_test_300s.log | head -20

# Common patterns:
# - Connection errors → Check internet/exchange
# - ML errors → Review model loading
# - Memory errors → Reduce workload
```

---

## 📊 Understanding Reports

### paper_trading_report_*.json
```json
{
  "summary": {
    "total_errors": 0,        // Should be 0
    "total_signals": 12,      // Trading signals generated
    "total_trades": 3,        // Trades executed
    "ml_predictions": 45      // ML predictions made
  },
  "performance": {
    "avg_memory_mb": 1234.5,  // Should be < 2000
    "avg_cpu_percent": 25.0,  // Should be < 50
  },
  "status": "PASS"            // PASS or FAIL
}
```

### paper_health_*.json
```json
{
  "status": "HEALTHY",        // HEALTHY, WARNING, or CRITICAL
  "error_count": 0,           // Total errors
  "dimension_errors": 0,      // MUST be 0
  "feature_counts": [42],     // Should only contain 42
  "consistent_features": true,// MUST be true
  "errors_per_hour": 0.0     // Should be < 1.0
}
```

### deployment_decision_*.json
```json
{
  "decision": "GO",           // GO, GO_WITH_CAUTION, NO_GO, NO_GO_FIX_WARNINGS
  "overall_score": 100.0,     // Out of 100
  "critical_passed": true,    // MUST be true to deploy
  "all_passed": true          // Ideal state
}
```

---

## 🎯 Success Criteria Checklist

### Critical (Must Pass)
- [ ] Python 3.11.x installed
- [ ] Legacy manifest exists with 42 features
- [ ] ML components load without errors
- [ ] No dimension mismatch errors in any test
- [ ] Bot runs for full test duration

### Performance (Should Pass)
- [ ] Average memory < 2GB
- [ ] Average CPU < 50%
- [ ] Error rate < 1 per hour
- [ ] Feature extraction consistently returns 42

### ML Components (Should Pass)
- [ ] Feature engineering works correctly
- [ ] Regime predictor makes predictions
- [ ] RL agent uses correct state size (42)
- [ ] Price predictor generates confidence scores

---

## 💡 Pro Tips

### Faster Testing
```bash
# Use shorter duration for quick checks
python scripts/live_trading_launcher.py --paper --duration 60 --symbols "BTC/USDT"
```

### Continuous Monitoring
```bash
# Run in tmux/screen session for long tests
tmux new -s paper_trading
./scripts/run_extended_test.sh
# Detach: Ctrl+B, D
# Re-attach: tmux attach -t paper_trading
```

### Automated Analysis
```bash
# Chain all analysis commands
python scripts/paper_trading_health.py paper_trading_1hour.log 3600 && \
python scripts/verify_ml_live.py paper_trading_1hour.log && \
python scripts/production_readiness_decision.py
```

### Debug Mode
```bash
# Enable detailed debug logging
export DEBUG_STRATEGY_LOGGING=true
export LOG_LEVEL=DEBUG
```

---

## 📞 Getting Help

1. **Check logs**: `tail -100 paper_trading_test_300s.log`
2. **Review reports**: `cat paper_health_*.json | jq .`
3. **Verify system**: `python scripts/verify_system_ready.py`
4. **Read full docs**: `scripts/README_PAPER_TRADING_TESTS.md`

---

## 🔗 Related Documentation

- [Paper Trading Tests README](./README_PAPER_TRADING_TESTS.md) - Full documentation
- [Live Trading Launcher](./README_LIVE_TRADING_LAUNCHER.md) - Launcher documentation
- [Repository README](../README.md) - Main project documentation

---

## ⏱️ Time Estimates

| Task | Duration |
|------|----------|
| System verification | 1 minute |
| Short test (5 min) | 5-6 minutes |
| Extended test (1 hour) | 60-65 minutes |
| Analysis | 2-3 minutes |
| Total (with extended test) | ~70 minutes |

---

## 🎓 Example Session

```bash
# 1. Verify system
$ python scripts/verify_system_ready.py
✅ SYSTEM READY FOR PAPER TRADING

# 2. Run short test
$ ./scripts/run_short_test.sh
✅ Short test completed successfully!

# 3. Run extended test (if short test passed)
$ ./scripts/run_extended_test.sh
✅ Extended test completed successfully!

# 4. Make deployment decision
$ python scripts/production_readiness_decision.py
✅ RECOMMENDATION: READY FOR PRODUCTION DEPLOYMENT
Overall Score: 100.0/100
```

**Result: Ready to deploy! 🚀**
