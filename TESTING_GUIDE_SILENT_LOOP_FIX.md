# Testing Guide: Silent Loop Fix

## Overview

This guide explains how to verify the fix for the silent trading loop issue in both local and GitHub Actions environments.

## Prerequisites

- **Python Version**: 3.11.x (Required - see `.python-version`)
- **Dependencies**: Install with `pip install -r requirements.txt`
- **Exchange Credentials**: BingX API keys (optional for dry-run tests)

## Quick Verification

### 1. Run Unit Tests (Local)

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run the test suite
pytest tests/test_silent_loop_fix.py -v

# Expected output:
# test_fetch_ohlcv_with_timeout PASSED
# test_process_symbol_timeout_handling PASSED
# test_watchdog_task_runs PASSED
# test_process_trading_loop_completes PASSED
# test_circuit_breaker_timeout PASSED
# test_run_production_loop_debug_off PASSED
# test_run_production_loop_with_slow_fetch PASSED
# test_engine_sync_delay_sufficient PASSED
```

### 2. Integration Test (Local - Dry Run)

```bash
# Test without debug flag (previously failing)
python scripts/live_trading_launcher.py --paper --duration 60 --dry-run

# Expected logs (emojis may render differently across environments):
# ✓ WebSocket manager initialized
# ✓ Trading engine started (state = RUNNING)
# 🐕 [WATCHDOG-1] Heartbeat - is_running=True
# 🔁 [ITERATION 1] Processing 3 symbols...
# [PROCESSING] Symbol 1/3: BTC/USDT:USDT
# [PROCESSING] BTC/USDT:USDT completed in 2.5s
# ✅ [PROCESSING] Completed processing loop in 7.5s
# 🐕 [WATCHDOG-2] Heartbeat - is_running=True
# Note: Emojis (🐕✅🔁) may not render in some log viewers/terminals
```

### 3. Integration Test (Local - Full Run)

```bash
# Test with actual trading loop (requires credentials)
export BINGX_KEY="your_key_here"
export BINGX_SECRET="your_secret_here"

python scripts/live_trading_launcher.py --paper --duration 60

# Monitor for:
# ✅ Watchdog heartbeats every 10s
# ✅ Loop iterations every 30s
# ✅ Symbol processing with timing
# ✅ Clean exit after 60s
```

## GitHub Actions Verification

### 1. Trigger Workflow Manually

Navigate to: `.github/workflows/live_trading_launcher.yml`

**Configuration**:
- Mode: `paper`
- Duration: `300` (5 minutes)
- Debug Mode: `false` (test non-debug mode first)
- Dry Run: `false`

### 2. Monitor Logs

Key indicators of success:

#### Startup Phase (First 10 seconds)
```
✓ Config loaded: 3 symbols
✓ BingX client optimized for 3 symbols only
✓ Risk configuration loaded
✓ WebSocket manager initialized
✓ Trading engine started (state = RUNNING)
🐕 [WATCHDOG] Watchdog task started - will log every 10s
```

#### Runtime Phase (Every 10 seconds)
```
🐕 [WATCHDOG-1] Heartbeat - is_running=True
   Active symbols: 3
   Processed symbols: 0
   Engine state: running

🔁 [ITERATION 1] Processing 3 symbols...
📋 [PROCESSING] Starting processing loop for 3 symbols
[PROCESSING] Symbol 1/3: BTC/USDT:USDT
[DATA-FETCH] Fetching market data for BTC/USDT:USDT
[PROCESSING] BTC/USDT:USDT completed in 2.5s
...
✅ [PROCESSING] Completed processing loop in 7.5s
   Processed: 3/3 symbols
   Signals: 0 | Errors: 0

🐕 [WATCHDOG-2] Heartbeat - is_running=True
   Active symbols: 3
   Processed symbols: 3
```

#### Shutdown Phase
```
⏱️ Duration 300s reached - stopping
Initiating graceful shutdown
✅ Production trading loop stopped
✓ CLEANUP COMPLETED SUCCESSFULLY
```

### 3. Expected Behavior

**Success Criteria**:
- ✅ **Watchdog heartbeats**: Appear every 10 seconds throughout run
- ✅ **Loop iterations**: Appear every 30 seconds (loop_interval)
- ✅ **Symbol processing**: Shows progress with timing for each symbol
- ✅ **No silent stalls**: Continuous logging from start to finish
- ✅ **Timeout handling**: Any slow fetches timeout after 15-20s with warning
- ✅ **Clean exit**: Stops after specified duration with code 0

**Failure Indicators** (if fix doesn't work):
- ❌ Watchdog stops logging after startup
- ❌ No loop iteration logs after "STARTING PRODUCTION TRADING LOOP"
- ❌ Long gaps (>30s) between logs
- ❌ Process hangs indefinitely
- ❌ No timeout warnings despite slow network

## Debugging Failed Tests

### Issue: Tests Hang

**Symptom**: Test suite hangs on specific test
**Cause**: Async operation not properly mocked or timed out
**Fix**: Check for missing `await` or `AsyncMock` in test setup

```python
# Bad - will hang
mock_method = Mock(return_value=asyncio.Future())

# Good - returns immediately
mock_method = AsyncMock(return_value={'success': True})
```

### Issue: "asyncio.TimeoutError not raised"

**Symptom**: `test_fetch_ohlcv_with_timeout` fails
**Cause**: Timeout not applied or mocking incorrect
**Check**: Verify `asyncio.wait_for()` is actually called in production code

### Issue: "Watchdog didn't log"

**Symptom**: `test_watchdog_task_runs` fails
**Cause**: Task cancelled before first log
**Fix**: Increase sleep duration in test or reduce watchdog interval

## Performance Benchmarks

### Before Fix
- **Loop Start**: ~5 seconds
- **Silent Stall**: Indefinite (never completes)
- **Timeout**: None (blocks forever)

### After Fix
- **Loop Start**: ~6 seconds (includes 1.0s sync delay)
- **Per-Symbol**: 2-5 seconds typical (with timeout protection)
- **Per-Iteration**: 7-15 seconds typical (3 symbols × 2-5s)
- **Max Per-Symbol**: 60 seconds worst-case (3 timeframes × 20s timeout each)
- **Max Per-Iteration**: ~180 seconds worst-case (3 symbols × 60s max)

### Expected Timings (GitHub Actions)

```
Startup:           0-10s     (initialization)
First Watchdog:    10s       (after startup)
First Iteration:   10-15s    (starts after startup, completes within this time)
Iteration Period:  30s       (loop_interval between iterations)
Shutdown:          <5s       (cleanup)

Note: "First Iteration: 10-15s" means it starts immediately after startup 
      completes and takes 10-15s to process all symbols and return.
```

## Common Issues and Solutions

### Issue: Python 3.12 Instead of 3.11

**Error**: `aiohttp==3.8.6 fails to install on Python 3.12 due to incompatible internal API changes (PyLongObject.ob_digit)`
**Solution**: Install Python 3.11 (required version)

```bash
# Using pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Using asdf
asdf install python 3.11.9
asdf local python 3.11.9

# Verify
python --version  # Should show Python 3.11.x
```

### Issue: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'pytest'`
**Solution**: Install test dependencies

```bash
pip install pytest pytest-asyncio pytest-mock
```

### Issue: Credentials Not Found

**Error**: `❌ Missing required environment variables: ['BINGX_KEY', 'BINGX_SECRET']`
**Solution**: Use dry-run mode or set credentials

```bash
# Option 1: Dry run (no credentials needed)
python scripts/live_trading_launcher.py --paper --duration 60 --dry-run

# Option 2: Set credentials
export BINGX_KEY="your_key"
export BINGX_SECRET="your_secret"
```

## Regression Testing

After any future changes to `production_coordinator.py`, run:

```bash
# Quick regression test
pytest tests/test_silent_loop_fix.py -v

# Full integration test
python scripts/live_trading_launcher.py --paper --duration 60 --dry-run

# Verify watchdog logs appear (log files follow pattern: live_trading_YYYYMMDD_HHMMSS.log)
grep "WATCHDOG" live_trading_*.log 2>/dev/null | head -10 || echo "No log files found matching live_trading_*.log"
```

## Test Reports

### Unit Test Coverage

- ✅ Timeout handling (REST API)
- ✅ Timeout handling (symbol processing)
- ✅ Watchdog task execution
- ✅ Processing loop completion
- ✅ Circuit breaker timeout
- ✅ Debug mode OFF operation
- ✅ Slow fetch recovery
- ✅ Engine synchronization

### Integration Test Coverage

- ✅ Full startup sequence
- ✅ Loop iteration processing
- ✅ Symbol data fetching
- ✅ Timeout error recovery
- ✅ Graceful shutdown
- ✅ Log file generation

## Success Metrics

### Quantitative
- ✅ 100% of unit tests pass
- ✅ 0 test timeouts
- ✅ Loop completes within expected time
- ✅ All timeouts trigger within ±2s of configured value

### Qualitative
- ✅ Logs are continuous (no gaps >30s)
- ✅ Error messages are clear and actionable
- ✅ System recovers from slow network
- ✅ Behavior identical in debug and non-debug modes

## References

- **Root Cause Analysis**: `ROOT_CAUSE_ANALYSIS_SILENT_LOOP.md`
- **Test Suite**: `tests/test_silent_loop_fix.py`
- **Code Changes**: `src/core/production_coordinator.py`
- **Workflow**: `.github/workflows/live_trading_launcher.yml`

## Support

If tests fail or loop still stalls:
1. Review `ROOT_CAUSE_ANALYSIS_SILENT_LOOP.md` for detailed explanation
2. Check GitHub Actions logs for specific error messages
3. Verify Python version is 3.11.x
4. Ensure all dependencies are installed correctly
5. Run tests with `-v -s` for verbose output with print statements
