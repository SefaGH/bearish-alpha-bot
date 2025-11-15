# Paper Trading Test & Deployment Readiness Validation

This directory contains scripts for comprehensive paper trading tests and deployment readiness validation for the Bearish Alpha Bot.

## 📋 Overview

The testing framework validates production readiness by:
- Running paper trading tests with the legacy system (42 features)
- Monitoring ML components in live market conditions
- Analyzing system health and performance
- Making data-driven deployment decisions

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11** (Required - see repository custom instructions)
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure legacy manifest exists: `artifacts/legacy/manifest.json`

### Running Tests

#### Option 1: Automated Test Scripts (Recommended)

**Short Test (5 minutes):**
```bash
chmod +x scripts/run_short_test.sh
./scripts/run_short_test.sh
```

**Extended Test (1 hour):**
```bash
chmod +x scripts/run_extended_test.sh
./scripts/run_extended_test.sh
```

#### Option 2: Manual Step-by-Step

**Step 1: Pre-Launch Verification**
```bash
python scripts/verify_system_ready.py
```

**Step 2: Short Test (5 minutes)**
```bash
# Setup environment
export ML_ENABLED=true
export GEMMA_ENABLED=false
export PAPER_TRADING=true
export DEBUG_MODE=true
export LOG_LEVEL=INFO

# Run bot
nohup python scripts/live_trading_launcher.py \
    --paper \
    --debug \
    --duration 300 \
    --symbols "BTC/USDT,ETH/USDT" \
    > paper_trading_test_300s.log 2>&1 &

# Monitor (in another terminal)
python scripts/monitor_paper_trading.py paper_trading_test_300s.log 300
```

**Step 3: Extended Test (1 hour)**
```bash
# Run bot for 1 hour
nohup python scripts/live_trading_launcher.py \
    --paper \
    --debug \
    --duration 3600 \
    --symbols "BTC/USDT,ETH/USDT,SOL/USDT" \
    > paper_trading_1hour.log 2>&1 &

BOT_PID=$!

# Monitor in background
python scripts/monitor_paper_trading.py paper_trading_1hour.log 3600 &
MONITOR_PID=$!

# Wait for completion
wait $BOT_PID
wait $MONITOR_PID
```

**Step 4: Health Analysis**
```bash
python scripts/paper_trading_health.py paper_trading_1hour.log 3600
```

**Step 5: ML Component Verification**
```bash
python scripts/verify_ml_live.py paper_trading_1hour.log
```

**Step 6: Deployment Decision**
```bash
python scripts/production_readiness_decision.py
```

## 📁 Scripts

### Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `verify_system_ready.py` | Pre-launch system verification | `python scripts/verify_system_ready.py` |
| `monitor_paper_trading.py` | Real-time monitoring during tests | `python scripts/monitor_paper_trading.py <log> <duration>` |
| `paper_trading_health.py` | Health analysis of test results | `python scripts/paper_trading_health.py <log> <duration>` |
| `verify_ml_live.py` | ML component verification | `python scripts/verify_ml_live.py <log>` |
| `production_readiness_decision.py` | Final deployment decision | `python scripts/production_readiness_decision.py` |

### Convenience Scripts

| Script | Purpose |
|--------|---------|
| `test_environment.sh` | Setup environment variables |
| `run_short_test.sh` | Automated 5-minute test |
| `run_extended_test.sh` | Automated 1-hour test |

## 📊 Test Criteria

### Critical Criteria (Must Pass)
- ✅ Python 3.11 installed
- ✅ Legacy manifest valid (42 features)
- ✅ ML components load without errors
- ✅ No dimension mismatch errors
- ✅ Bot runs without crashes

### Performance Criteria
- ✅ Memory usage < 2GB average
- ✅ CPU usage < 50% average
- ✅ Error rate < 1 per hour
- ✅ Feature extraction consistent (42 features)

### ML Component Criteria
- ✅ Feature extraction always returns 42 features
- ✅ Regime predictor makes predictions without errors
- ✅ RL agent uses state_size=42
- ✅ Price predictor generates confidence scores
- ✅ Strategy coordinator applies ML enhancements

## 📈 Reports Generated

After running tests, the following reports are created:

| Report File | Description |
|-------------|-------------|
| `paper_trading_report_*.json` | Performance metrics and resource usage |
| `paper_health_*.json` | Health analysis with error counts and rates |
| `deployment_decision_*.json` | Final deployment readiness decision |
| `ml_components_verified.flag` | Flag file indicating ML verification passed |

## 🎯 Expected Outcomes

### Successful Test Results

**Short Test (5 minutes):**
- Bot starts without errors ✅
- Connects to exchange (paper mode) ✅
- ML components initialize ✅
- Feature extraction works (42 features) ✅
- No critical errors ✅

**Extended Test (1 hour):**
- Runs for full hour without crashes ✅
- No dimension mismatch errors ✅
- Memory usage stable (<2GB) ✅
- CPU usage reasonable (<50% avg) ✅
- Error rate <1 per hour ✅
- Consistent feature count (42) ✅

**ML Components:**
- Feature extraction: 42 features consistently ✅
- Regime predictor: Predictions without errors ✅
- RL agent: state_size=42 ✅
- Price predictor: Confidence scores generated ✅
- Strategy coordinator: ML enhancements applied ✅

## 🔍 Troubleshooting

### Common Issues

**Issue: Python version mismatch**
```
Solution: Install Python 3.11
$ pyenv install 3.11
$ pyenv local 3.11
```

**Issue: Manifest not found**
```
Solution: Ensure artifacts/legacy/manifest.json exists
$ ls -la artifacts/legacy/manifest.json
```

**Issue: ML components fail to load**
```
Solution: Check dependencies and re-install
$ pip install -r requirements.txt
```

**Issue: Dimension mismatch errors**
```
Solution: This is a critical issue. Check:
- Manifest feature_count is 42
- FeatureEngineeringPipeline returns 42 features
- Models are trained with 42 features
```

## 📖 Deployment Decision Logic

The `production_readiness_decision.py` script evaluates all criteria and makes one of four decisions:

| Decision | Score | Meaning |
|----------|-------|---------|
| **GO** | 100/100 | All criteria met - deploy to production |
| **GO_WITH_CAUTION** | 80-99/100 | Critical checks passed - monitor closely |
| **NO_GO_FIX_WARNINGS** | 60-79/100 | Fix warnings before deployment |
| **NO_GO** | <60/100 | Critical issues - do not deploy |

## 🚦 Next Steps After Testing

1. **Review all reports** in the current directory
2. **Run deployment decision**: `python scripts/production_readiness_decision.py`
3. **If GO or GO_WITH_CAUTION**:
   - Prepare production environment
   - Update deployment documentation
   - Plan rollback strategy
   - Deploy to production
4. **If NO_GO**:
   - Review errors in log files
   - Fix critical issues
   - Re-run tests

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLED` | `true` | Enable ML components |
| `GEMMA_ENABLED` | `false` | Use GEMMA (set false for legacy) |
| `PAPER_TRADING` | `true` | Enable paper trading mode |
| `DEBUG_MODE` | `true` | Enable debug logging |
| `LOG_LEVEL` | `INFO` | Logging level |

### Test Duration

Modify duration in scripts:
- Short test: 300 seconds (5 minutes)
- Extended test: 3600 seconds (1 hour)
- Custom: Use `--duration <seconds>` flag

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review log files for detailed errors
3. Refer to main project README.md
4. Open an issue on GitHub

## ⚠️ Important Notes

1. **Python 3.11 Only**: This project requires Python 3.11.x (not 3.12+)
2. **Legacy System**: Tests use legacy 42-feature system
3. **Paper Trading**: All tests run in paper trading mode (no real money)
4. **Duration**: Extended tests take 1 hour - plan accordingly
5. **Resources**: Monitor system resources during tests

## 📝 Example Output

**Successful Test:**
```
🚀 Starting Short Paper Trading Test (5 minutes)
==================================================

Step 1: Verifying system readiness...
✅ SYSTEM READY FOR PAPER TRADING

Step 2: Launching paper trading bot (5 minutes)...
Bot started with PID: 12345

Step 3: Monitoring bot (5 minutes)...
[30s] Interim Report:
  Errors: 0
  Warnings: 2
  Signals: 5
  Trades: 2
  ML Predictions: 12

✅ No critical errors in 5-minute test

==================================================
✅ Short test completed successfully!
```

## 🎓 Learning Resources

- [Live Trading Launcher Documentation](./README_LIVE_TRADING_LAUNCHER.md)
- [Feature Engineering Guide](../docs/feature_engineering.md)
- [ML Integration Guide](../docs/ml_integration.md)
- [Deployment Guide](../docs/deployment.md)
