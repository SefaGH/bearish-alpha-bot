# Debug Logging Guide

## Overview

This guide explains how to use the enhanced debug logging features for troubleshooting signal generation and strategy control issues in the Bearish Alpha Bot.

## Problem Addressed

When using `live_trading_launcher.py`, strategy control logs were not visible in the output, making it difficult to debug why signals were or weren't being generated.

## Solution: Enhanced Debug Logging

We've added comprehensive debug logging throughout the signal generation pipeline with structured tags for easy filtering and analysis.

## Logging Tags Reference

### Universe Building Tags
- `[UNIVERSE]` - Universe building and symbol filtering
  - Shows which symbols are evaluated
  - Displays market characteristics (active, quote, swap, spot, linear)
  - Indicates acceptance/rejection with ✅ emoji

### Signal Scanning Tags
- `[PROCESSING]` - Symbol currently being processed
- `[DATA]` - Market data fetching results
- `[INDICATORS]` - Technical indicator calculations
- `[STRATEGY-CHECK]` - Strategy evaluation in progress
- `[SIGNAL]` - Successful signal generation with ✅ emoji
- `[NO-SIGNAL]` - No signal generated (with reason)

## How to Enable Debug Logging

### Method 1: Command Line Flag (Recommended)

```bash
python scripts/live_trading_launcher.py --debug --paper
```

This enables comprehensive debug logging throughout the system.

### Method 2: Environment Variable

```bash
export LOG_LEVEL=DEBUG
python scripts/live_trading_launcher.py --paper
```

### Method 3: Test Config

Use the debug configuration file:

```bash
cp config/config.debug.yaml config/config.yaml
python scripts/live_trading_launcher.py --paper
```

## Example Output

### Universe Building

```
[UNIVERSE] Building universe with: min_qv=50000, top_n=20, only_linear=True
[UNIVERSE] Processing exchange: bingx
[UNIVERSE] bingx: loaded 2528 markets
[UNIVERSE] BTC/USDT:USDT: active=True, quote=USDT, swap=True, spot=False, linear=True
[UNIVERSE] ✅ BTC/USDT:USDT accepted as linear USDT perpetual
[UNIVERSE] bingx: 156 candidates after filtering
[UNIVERSE] bingx: volume filter removed 23 symbols
[UNIVERSE] bingx: selected 20 symbols: ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...]
[UNIVERSE] Built universe: {'bingx': ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...]}
```

### Signal Scanning (Signal Generated)

```
[PROCESSING] Symbol: BTC/USDT:USDT
[DATA] BTC/USDT:USDT: 30m=200 bars, last_close=67845.50
[INDICATORS] BTC/USDT:USDT: RSI=28.3, ATR=234.5678
[STRATEGY-CHECK] adaptive_ob for BTC/USDT:USDT
✅ [SIGNAL] BTC/USDT:USDT: {'side': 'buy', 'reason': 'RSI oversold', 'entry': 67845.50}
   Strategy: adaptive_ob
   Side: BUY
   Reason: RSI oversold
```

### Signal Scanning (No Signal)

```
[PROCESSING] Symbol: ETH/USDT:USDT
[DATA] ETH/USDT:USDT: 30m=200 bars, last_close=3421.80
[INDICATORS] ETH/USDT:USDT: RSI=55.3, ATR=45.2341
[STRATEGY-CHECK] adaptive_ob for ETH/USDT:USDT
[NO-SIGNAL] ETH/USDT:USDT (adaptive_ob): RSI=55.3
```

## Test Configuration

For testing signal generation, use `config/config.debug.yaml` which has:

### Relaxed Universe Settings
- `min_quote_volume_usdt: 50000` (lower threshold = more symbols)
- `max_symbols_per_exchange: 20` (more coverage)
- `only_linear: true` (BingX perpetuals)

### Relaxed Signal Thresholds
- **OversoldBounce**: `rsi_max: 50` (triggers more easily)
- **ShortTheRip**: `rsi_min: 50` (triggers more easily)
- `ignore_regime: true` (disables regime filtering for testing)

## Testing Workflow

### 1. Manual Signal Generation Test

```bash
# Set up environment
export EXCHANGES=bingx
export BINGX_KEY=your_key
export BINGX_SECRET=your_secret

# Run manual test
python scripts/test_signal_generation.py
```

**Expected Output:**
```
Testing symbol: BTC/USDT:USDT
Fetched 100 candles
Last RSI: 45.2
✅ OVERSOLD signal possible! (RSI 45.2 <= 50)
```

### 2. Debug Demo Script

Run the demonstration script to see logging in action:

```bash
python scripts/demo_debug_logging.py
```

This shows:
- Universe symbol filtering with logs
- Expected log format for strategy checks
- Usage examples

### 3. Live Trading Launcher with Debug

```bash
# Paper mode with debug logging
export MODE=paper
export EXCHANGES=bingx
export EXECUTION_EXCHANGE=bingx
export LOG_LEVEL=DEBUG

python scripts/live_trading_launcher.py --debug
```

## Troubleshooting Common Issues

### Issue 1: Universe is Empty

**Symptoms:**
```
[UNIVERSE] bingx: no eligible symbols after filtering; skipping.
Universe is empty.
```

**Solutions:**
- Lower `min_quote_volume_usdt` to 0 or 50000
- Increase `max_symbols_per_exchange`
- Set `prefer_perps: false` to include spot pairs

**Check:**
```bash
# Check what was filtered out
grep "\[UNIVERSE\]" logfile.log | grep -v "✅"
```

### Issue 2: Strategies Never Called

**Symptoms:**
- See `[PROCESSING]` and `[DATA]` logs
- No `[STRATEGY-CHECK]` logs

**Solutions:**
- Verify strategy is enabled in config: `enable: true`
- Check if strategies are registered:
  ```bash
  grep "strategies registered" logfile.log
  ```

**Check:**
```python
ob_enabled = cfg.get('signals', {}).get('oversold_bounce', {}).get('enable', True)
print(f"OversoldBounce enabled: {ob_enabled}")
```

### Issue 3: RSI Values Never in Range

**Symptoms:**
- See `[NO-SIGNAL]` with RSI values always too high/low
- Example: `[NO-SIGNAL] BTC/USDT:USDT (adaptive_ob): RSI=65.3`

**Solutions:**
- Use relaxed thresholds for testing:
  - OversoldBounce: `rsi_max: 60`
  - ShortTheRip: `rsi_min: 40`
- Or wait for market conditions to change

**Check:**
```bash
# See all RSI values
grep "\[INDICATORS\]" logfile.log | grep RSI
```

## Log Filtering Examples

### Show Only Signals
```bash
grep "\[SIGNAL\]" logfile.log
```

### Show Strategy Checks
```bash
grep "\[STRATEGY-CHECK\]" logfile.log
```

### Show Universe Building
```bash
grep "\[UNIVERSE\]" logfile.log
```

### Show Why Symbols Rejected
```bash
grep "\[UNIVERSE\]" logfile.log | grep -v "✅"
```

### Show All Processing Steps for One Symbol
```bash
grep "BTC/USDT:USDT" logfile.log | grep -E "\[(PROCESSING|DATA|INDICATORS|STRATEGY-CHECK|SIGNAL|NO-SIGNAL)\]"
```

## Running Tests

Verify debug logging functionality:

```bash
# Run debug logging tests
python -m pytest tests/test_debug_logging.py -v -s

# Run all tests
python -m pytest tests/ -v
```

## Additional Resources

- **Demo Script**: `scripts/demo_debug_logging.py`
- **Test Config**: `config/config.debug.yaml`
- **Test Script**: `scripts/test_signal_generation.py`
- **Test Suite**: `tests/test_debug_logging.py`

## Best Practices

1. **Always use `--debug` flag** when troubleshooting signal issues
2. **Save logs to file** for analysis: `python script.py 2>&1 | tee debug.log`
3. **Use grep to filter** specific log tags you're interested in
4. **Start with relaxed thresholds** in test config, tighten in production
5. **Check RSI values** in `[INDICATORS]` logs to understand market conditions
6. **Verify universe** is not empty before expecting signals

## Summary

The enhanced debug logging provides complete visibility into:
- ✅ Which symbols are being evaluated
- ✅ Why symbols are accepted/rejected
- ✅ What market data is fetched
- ✅ What indicator values are calculated
- ✅ Which strategies are being evaluated
- ✅ Why signals are or aren't generated

This makes it easy to diagnose and fix signal generation issues.
