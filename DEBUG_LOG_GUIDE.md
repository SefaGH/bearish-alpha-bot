# Debug Log Guide - Silent Loop Issue Investigation

## Overview

This guide explains how to interpret the debug logs added to diagnose the silent trading loop issue.

## Quick Start

1. Run the GitHub Actions workflow with `--paper --duration 300` flags
2. Search logs for `🔍 [DEBUG]` markers to find diagnostic messages
3. Follow the flow below to identify where execution stops

## Expected Log Flow

### 1. Launcher Initialization
```
🚀 STARTING PRODUCTION LOOP
🔍 [LAUNCHER-DEBUG] About to call coordinator.run_production_loop()
🔍 [LAUNCHER-DEBUG] coordinator type: <class 'core.production_coordinator.ProductionCoordinator'>
🔍 [LAUNCHER-DEBUG] coordinator.is_running: False
🔍 [LAUNCHER-DEBUG] coordinator.is_initialized: True
🔍 [LAUNCHER-DEBUG] Parameters: mode=paper, duration=300, continuous=False
🔍 [LAUNCHER-DEBUG] Calling await coordinator.run_production_loop()...
```

### 2. Coordinator Method Entry
```
🔍 [DEBUG] run_production_loop() method ENTERED
🔍 [DEBUG] Parameters: mode=paper, duration=300, continuous=False
🔍 [DEBUG] self.is_initialized = True
🔍 [DEBUG] Inside try block
🔍 [DEBUG] Passed initialization check
======================================================================
STARTING PRODUCTION TRADING LOOP
======================================================================
```

### 3. Trading Engine Validation
```
🔍 [DEBUG] Checking trading engine...
🔍 [DEBUG] trading_engine exists, state=running
✅ Trading engine already running (state=running)
```

### 4. is_running State Management
```
🔍 [DEBUG] Current is_running = False
⚠️ is_running was False, setting to True
🔍 [DEBUG] is_running now = True
```

### 5. Queue Monitoring Setup
```
🔍 [DEBUG] Creating queue monitoring task...
🔍 [DEBUG] Queue monitoring task created
Queue monitoring task started
📊 [QUEUE-MONITOR] StrategyCoordinator: 0 signals | LiveTradingEngine: 0 signals
```

### 6. Loop Initialization
```
🔍 [DEBUG] About to print production loop info...
🚀 Production trading loop active
   Mode: paper
   Duration: 300s
   Continuous Mode: DISABLED
   Active Symbols: 3
🔍 [DEBUG] Checking active_symbols: ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
🔍 [DEBUG] active_symbols check passed
🔍 [DEBUG] Initializing loop variables...
🔍 [DEBUG] Loop variables initialized: start_time=..., loop_iteration=0
```

### 7. Loop Start
```
======================================================================
🔄 STARTING TRADING LOOP ITERATIONS
======================================================================
   Loop interval: 30s
   Symbols to process: 3
   Will run for: 300s
======================================================================

🔍 [DEBUG] About to enter while loop. is_running=True
```

### 8. First Iteration
```
🔍 [DEBUG] INSIDE WHILE LOOP - Iteration starting
🔍 [DEBUG] Loop iteration: 1, is_running: True
🔁 [ITERATION 1] Processing 3 symbols...
🔍 [DEBUG] Duration check: elapsed=0.1s, duration=300s
🔍 [DEBUG] Duration check passed - continuing loop
🔍 [DEBUG] About to call _process_trading_loop()...
📋 [PROCESSING] Starting processing loop for 3 symbols
[PROCESSING] Symbol: BTC/USDT:USDT
[DATA-FETCH] Fetching market data for BTC/USDT:USDT
...
🔍 [DEBUG] _process_trading_loop() completed
```

### 9. Loop Iterations Continue
```
🔁 [ITERATION 2] Processing 3 symbols...
🔁 [ITERATION 3] Processing 3 symbols...
...
🔁 [ITERATION 10] Processing 3 symbols...
```

### 10. Duration Expiry
```
🔍 [DEBUG] Duration check: elapsed=300.2s, duration=300s
⏱️ Duration 300s reached - stopping (elapsed: 300.2s)
```

### 11. Loop Exit & Cleanup
```
Shutting down production trading loop...
✅ Trading engine stopped
✅ Production coordinator stopped
✅ WebSocket streams stopped
```

## Diagnostic Scenarios

### Scenario A: No Coordinator Entry Logs
**Missing**: All logs after "Calling await coordinator.run_production_loop()..."
**Cause**: Method not being called or immediate exception
**Action**: Check if coordinator object is valid

### Scenario B: No Loop Initialization
**Missing**: Logs after "STARTING PRODUCTION TRADING LOOP"
**Cause**: Exception in initialization checks
**Action**: Review trading engine state, is_initialized flag

### Scenario C: No Loop Entry
**Missing**: "INSIDE WHILE LOOP" message
**Cause**: is_running is False or loop condition not met
**Action**: Check is_running state in previous logs

### Scenario D: Immediate Loop Exit
**Present**: Loop entry but no iterations
**Cause**: Duration check failing on first iteration
**Action**: Check elapsed time in duration check logs

### Scenario E: Loop Hangs
**Present**: Loop enters, iteration starts, but stops mid-iteration
**Cause**: Await call hanging (circuit breaker, process_symbol)
**Action**: Find last log before silence to identify hanging await

### Scenario F: No Symbol Processing
**Present**: Iteration logs but no "[PROCESSING] Symbol:" messages
**Cause**: _process_trading_loop() not called or returning immediately
**Action**: Check if "About to call _process_trading_loop()" appears

## How to Use This Guide

1. **Collect Logs**: Download GitHub Actions workflow logs
2. **Search for Markers**: Use search/grep for `🔍 [DEBUG]` 
3. **Find Last Log**: Identify the last debug log that appears
4. **Match Scenario**: Compare with scenarios above
5. **Identify Issue**: The missing logs indicate where execution stops

## Expected vs. Actual

| Expected Behavior | Debug Log Marker | Issue if Missing |
|-------------------|------------------|------------------|
| Method called | `run_production_loop() method ENTERED` | Coordinator not invoked |
| Initialization passed | `Passed initialization check` | System not initialized |
| Engine running | `trading_engine exists, state=running` | Engine not started |
| is_running set | `is_running now = True` | State not updated |
| Loop entry | `INSIDE WHILE LOOP` | Loop condition false |
| Iteration start | `[ITERATION N]` | Loop exiting early |
| Duration check | `Duration check: elapsed=...` | Check logic issue |
| Processing call | `About to call _process_trading_loop()` | Method not reached |
| Symbol processing | `[PROCESSING] Symbol:` | No symbols processed |

## Common Issues & Solutions

### Issue: Buffered Logs
**Symptom**: Long delay before logs appear
**Solution**: PYTHONUNBUFFERED=1 is now set in workflow
**Verification**: Logs should appear in real-time

### Issue: Loop Exits Immediately  
**Symptom**: Duration check shows elapsed >= duration on first iteration
**Solution**: Check startup time, ensure it's not consuming entire duration
**Debug**: Look for "Duration check: elapsed=..." on first iteration

### Issue: Hanging Await
**Symptom**: Logs stop mid-iteration at a specific point
**Solution**: Identify the await call causing the hang
**Debug**: The log just before silence indicates the blocking call

### Issue: is_running False
**Symptom**: "About to enter while loop. is_running=False"
**Solution**: Something is setting is_running=False after it was set to True
**Debug**: Search for any is_running state changes between logs

## Contact & Support

If logs show unexpected behavior not covered in this guide:
1. Export full logs
2. Note the last debug marker that appeared
3. Identify any error messages
4. Report in GitHub issue with:
   - Last successful debug marker
   - First missing debug marker
   - Any error messages
   - Elapsed time when issue occurred
