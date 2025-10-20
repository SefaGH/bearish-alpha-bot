# Issue #131: Debug Multi-Symbol Trading Implementation

**Status:** ✅ COMPLETE  
**Date:** 2025-10-20  
**Branch:** `copilot/debug-multi-symbol-trading`

---

## 📋 Problem Statement

Bot only trades BTC/USDT and does not generate signals or open positions for ETH/USDT or SOL/USDT pairs during live paper trading sessions.

**Impact:**
- No diversification: missed opportunities in ETH and SOL markets
- Portfolio risk not balanced
- Strategy performance metrics incomplete

---

## 🎯 Solution Overview

Implemented comprehensive debug logging and symbol-specific RSI threshold configuration to enable signal generation for all symbols.

---

## 🔧 Implementation Details

### 1. Debug Logging

Added detailed per-symbol logging to both adaptive strategies (`adaptive_str.py` and `adaptive_ob.py`):

```python
[STR-DEBUG] ETH/USDT:USDT
  📌 Using symbol-specific RSI threshold: 50.00
  RSI: 52.80 (threshold: 50.00)
  ✅ RSI check passed: 52.80 >= 50.00
  EMA Align: ✅ (21=2257.70, 50=2280.50, 200=2303.30)
  Volume: 1000.00
  ATR: 45.6100
  ✅ Signal: SELL (RSI 52.8 >= 50.0, regime=neutral)
  Entry: $2280.50, Target: $2143.67, Stop: $2326.11, R/R: 3.00
```

**Logged Information:**
- Symbol name
- RSI value vs. threshold (with symbol-specific override indicator)
- EMA alignment status with actual values
- Volume checks
- ATR calculations
- Final signal decision with clear reasoning

### 2. Symbol-Specific Configuration

Added support for per-symbol RSI threshold overrides in `config.example.yaml`:

```yaml
signals:
  short_the_rip:
    # Default parameters
    adaptive_rsi_base: 55
    adaptive_rsi_range: 10
    
    # Symbol-specific RSI threshold overrides
    symbols:
      "BTC/USDT:USDT":
        rsi_threshold: 55  # BTC: More selective (higher threshold for shorts)
      "ETH/USDT:USDT":
        rsi_threshold: 50  # ETH: More sensitive (lower threshold)
      "SOL/USDT:USDT":
        rsi_threshold: 50  # SOL: More sensitive (lower threshold)
```

**How It Works:**
1. Strategy checks if symbol has specific threshold configured
2. If yes, uses symbol-specific threshold
3. If no, falls back to `adaptive_rsi_base` with regime adjustments
4. Logs which threshold is being applied for transparency

### 3. Code Changes

**Files Modified:**

1. **`src/strategies/adaptive_str.py`**
   - Added `get_symbol_specific_threshold(symbol)` method
   - Updated `signal()` to accept `symbol` parameter
   - Added comprehensive debug logging throughout
   - Changed from `logger.debug()` to `logger.info()` for visibility

2. **`src/strategies/adaptive_ob.py`**
   - Same changes as above for consistency
   - Both long and short strategies now have identical logging

3. **`src/main.py`**
   - Updated strategy calls to pass symbol parameter:
     ```python
     out_sig = adaptive_str.signal(df_30i, df_1hi, symbol=sym)
     out_sig = adaptive_ob.signal(df_30i, df_1hi, symbol=sym)
     ```

4. **`config/config.example.yaml`**
   - Added `symbols` section under both strategies
   - Documented configuration format
   - Set recommended thresholds (BTC: 55, ETH/SOL: 50)

5. **`README.md`**
   - Added complete "Symbol-Specific Configuration" section
   - Tuning guidelines by asset type
   - Debug mode explanation
   - Troubleshooting steps

### 4. Testing

**Test Files Created:**

1. **`tests/test_utils.py`**
   - Shared utilities for test data generation
   - Constants for price ratios and EMA alignment
   - Reusable test dataframe creation

2. **`tests/test_symbol_specific_thresholds.py`**
   - Unit tests for symbol-specific threshold method
   - Tests for default fallback behavior
   - Integration tests for signal generation
   - All tests passing ✅

3. **`tests/demo_debug_multi_symbol.py`**
   - Interactive demonstration of debug logging
   - Shows different scenarios (RSI variations, EMA failures)
   - Educational tool for understanding the system

4. **`tests/simulate_multi_symbol_scan.py`**
   - Simulates real market scan across BTC/ETH/SOL
   - Shows before/after comparison
   - Demonstrates problem resolution

**Test Results:**
```
✅ get_symbol_specific_threshold method tests passed
✅ Default threshold fallback tests passed
✅ Symbol-specific signal generation tests passed
✅ Multi-symbol scan simulation passed
✅ Debug logging demonstration passed
```

---

## 📊 Results

### Before Implementation

**Configuration:**
- All symbols used RSI threshold 55
- No visibility into filtering decisions

**Outcome:**
- ❌ BTC with RSI 53: No signal (correct, 53 < 55)
- ❌ ETH with RSI 53: No signal (incorrect, filtered out)
- ❌ SOL with RSI 53: No signal (incorrect, filtered out)
- **Result:** Only BTC trades, no diversification

### After Implementation

**Configuration:**
- BTC: RSI threshold 55
- ETH/SOL: RSI threshold 50
- Full debug logging enabled

**Outcome:**
- ❌ BTC with RSI 53: No signal (correct, 53 < 55)
- ✅ ETH with RSI 53: Signal generated (correct, 53 >= 50)
- ✅ SOL with RSI 53: Signal generated (correct, 53 >= 50)
- **Result:** Multi-symbol trading with diversification ✅

---

## 🎓 Tuning Guidelines

### RSI Threshold Recommendations by Asset Type

| Asset Type | Short Strategy | Long Strategy | Reasoning |
|------------|----------------|---------------|-----------|
| **Large Cap** (BTC) | 55-60 | 40-45 | More selective, wait for stronger signals |
| **Mid Cap** (ETH) | 50-55 | 35-40 | Balanced approach |
| **Small Cap** (SOL, MATIC) | 45-50 | 30-35 | More sensitive, catch earlier moves |

### How to Adjust

1. **Too few signals?** Lower the threshold (for shorts) or raise it (for longs)
2. **Too many signals?** Raise the threshold (for shorts) or lower it (for longs)
3. **Symbol-specific behavior?** Use symbol overrides in config
4. **Debug issues?** Check debug logs to see exact RSI values and filtering

---

## ✅ Acceptance Criteria

All criteria met:

- [x] Debug logs show all relevant filter checks per symbol
- [x] ETH/SOL signals generated in test environment
- [x] Symbol-specific config documented in README
- [x] At least 1 position can be opened for ETH and SOL (when market conditions are right)
- [x] Code reviewed and security scanned (no issues found)

---

## 🔄 Testing Instructions

### Run Unit Tests

```bash
cd /home/runner/work/bearish-alpha-bot/bearish-alpha-bot
python3 tests/test_symbol_specific_thresholds.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```

### View Debug Logging Demo

```bash
python3 tests/demo_debug_multi_symbol.py
```

Shows debug output for various scenarios.

### Run Market Scan Simulation

```bash
python3 tests/simulate_multi_symbol_scan.py
```

Shows realistic market scan with current prices.

### Live Paper Trading Test (15 minutes)

```bash
python scripts/live_trading_launcher.py --paper --duration 900
```

Expected behavior:
- Debug logs appear for all symbols (BTC, ETH, SOL)
- ETH/SOL generate signals when RSI >= 50
- BTC generates signals when RSI >= 55
- All filters clearly logged

---

## 📝 Configuration Example

Add to your `config/config.example.yaml`:

```yaml
signals:
  short_the_rip:
    enable: true
    adaptive_rsi_base: 55
    
    # Symbol-specific overrides
    symbols:
      "BTC/USDT:USDT":
        rsi_threshold: 55
      "ETH/USDT:USDT":
        rsi_threshold: 50
      "SOL/USDT:USDT":
        rsi_threshold: 50
      # Add more symbols as needed
```

---

## 🔍 Troubleshooting

### Issue: No signals for a specific symbol

1. Check debug logs:
   ```
   [STR-DEBUG] SYMBOL/USDT:USDT
     RSI: X.X (threshold: Y.Y)
   ```

2. If RSI is below threshold:
   - Lower threshold in config
   - Verify market conditions are appropriate

3. If EMA alignment fails:
   - Check if bearish alignment is too strict
   - Consider adjusting regime requirements

### Issue: Too many signals for a symbol

1. Raise the RSI threshold
2. Add additional filters (volume, ATR)
3. Increase EMA strictness

---

## 🚀 Production Readiness

**Status:** ✅ Ready for production

- ✅ All tests passing
- ✅ Code reviewed (minor nitpicks addressed)
- ✅ Security scan clean (0 vulnerabilities)
- ✅ Documentation complete
- ✅ Backward compatible (fallback to default behavior)

**Deployment Steps:**

1. Merge this PR to main branch
2. Update production config with symbol-specific thresholds
3. Run 15-minute paper test to verify
4. Monitor debug logs for first hour
5. Adjust thresholds based on observed RSI values

---

## 📚 Related Documentation

- [README.md](README.md) - Symbol-Specific Configuration section
- [config/config.example.yaml](config/config.example.yaml) - Configuration examples
- [tests/test_utils.py](tests/test_utils.py) - Test utilities
- [DEBUG_LOGGING_GUIDE.md](DEBUG_LOGGING_GUIDE.md) - General debug logging guide

---

## 🙏 Credits

**Issue:** SefaGH/bearish-alpha-bot#131  
**Implementation:** GitHub Copilot  
**Testing:** Automated + Manual verification  
**Review:** Code review + Security scan passed

---

**End of Implementation Report**
