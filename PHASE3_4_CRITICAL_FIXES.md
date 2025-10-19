# Phase 3.4 Live Trading Engine - Critical Fixes

## Overview

This document describes the critical fixes implemented for Phase 3.4 Live Trading Engine to resolve three major issues that were preventing the system from working properly.

## Issues Fixed

### Issue 1: Missing --live Mode in main.py ✅

**Problem**: No `--live` argument existed in main.py, preventing users from starting the bot with `python src/main.py --live`

**Solution**: 
- Added argparse support with `--live` and `--paper` flags
- Implemented `main_live_trading()` async function that:
  - Initializes ProductionCoordinator
  - Loads exchange clients from environment variables
  - Sets up portfolio configuration
  - Runs production loop with proper mode (paper/live)
  - Handles TRADING_MODE and TRADING_DURATION environment variables

**Usage**:
```bash
# Start in paper trading mode
python src/main.py --live --paper

# Start in live trading mode (requires TRADING_MODE=live)
export TRADING_MODE=live
python src/main.py --live

# With custom duration (in seconds)
export TRADING_DURATION=3600
python src/main.py --live --paper
```

### Issue 2: Complex Config Loading with Silent Failures ✅

**Problem**: Config loading in live_trading_engine.py had complex try-except blocks that failed silently, leading to wrong symbols, empty lists, and incorrect risk parameters.

**Solution**: Implemented clear 3-step priority system with proper validation and error messages:

1. **YAML Config** (highest priority)
   - Loads from CONFIG_PATH environment variable or default path
   - Validates that fixed_symbols is a list and not empty
   - Logs success with symbol count

2. **Environment Variables** (fallback)
   - Uses TRADING_SYMBOLS environment variable (comma-separated)
   - Creates config from parsed symbols
   - Logs fallback to ENV

3. **Hard-coded Defaults** (last resort)
   - BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT
   - Logs warning about using defaults

**Usage**:
```bash
# Method 1: Using YAML config (recommended)
export CONFIG_PATH=config/config.example.yaml
python src/main.py --live

# Method 2: Using environment variables
export TRADING_SYMBOLS="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT"
python src/main.py --live

# Method 3: Use defaults (automatic fallback)
python src/main.py --live
```

**Error Handling**:
- All failures are logged as ERROR (not warning)
- Clear messages indicate which step failed and why
- No silent failures - always provides symbols from some source

### Issue 3: WebSocket Initialization Race Conditions ✅

**Problem**: WebSocket setup in production_coordinator.py had race conditions, no validation, and could exceed stream limits causing connection failures.

**Solution**: Implemented proper validation flow with per-exchange limits:

1. **Validate Prerequisites**
   - Checks that exchange_clients exist
   - Gets symbols with fallback to defaults
   - Returns False if prerequisites fail

2. **Initialize Manager**
   - Creates WebSocketManager with error handling
   - Returns False if initialization fails

3. **Setup Streams with Limits**
   - Respects per-exchange stream limits:
     - BingX: 10 streams
     - Binance: 20 streams
     - KuCoin Futures: 15 streams
     - Default: 10 streams
   - Only subscribes up to limit per exchange
   - Counts successful streams
   - Returns False if zero streams started

4. **Return Status**
   - Returns True if any streams started successfully
   - Returns False otherwise

**Per-Exchange Stream Limits**:
```python
stream_limits = {
    'bingx': 10,
    'binance': 20,
    'kucoinfutures': 15,
}
```

**Configuration Override**:
You can override limits in your YAML config:
```yaml
websocket:
  max_streams_per_exchange:
    bingx: 5
    binance: 30
    custom_exchange: 25
```

## Files Modified

### 1. src/main.py
- Added argparse support for command-line arguments
- Added `main_live_trading()` async function
- Added `run_with_pipeline()` stub
- Modified main block to handle --live and --paper flags

### 2. src/core/live_trading_engine.py
- Replaced lines 119-157 with new config loading logic
- Implemented 3-step priority system (YAML > ENV > defaults)
- Added comprehensive validation and error logging
- No breaking changes to existing APIs

### 3. src/core/production_coordinator.py
- Replaced lines 112-159 with new WebSocket setup
- Changed `_setup_websocket_connections()` to return bool
- Added `_get_stream_limit(exchange_name)` helper method
- Implemented proper validation and error handling
- Added stream counting and status reporting

## Environment Variables

### Required for Live Trading
- `EXCHANGES`: Comma-separated list of exchanges (e.g., "bingx,binance,kucoinfutures")
- Exchange-specific credentials (see multi-exchange documentation)

### Optional Configuration
- `CONFIG_PATH`: Path to YAML config file (default: "config/config.example.yaml")
- `TRADING_SYMBOLS`: Comma-separated trading symbols (fallback if YAML fails)
- `TRADING_MODE`: "paper" or "live" (default: "paper")
- `TRADING_DURATION`: Duration in seconds (0 or unset = unlimited)
- `EQUITY_USD`: Portfolio equity in USD (default: 100)

## Testing

### Validation Script
A comprehensive validation script is included at `scripts/validate_phase3_4_fixes.py`:
```bash
python scripts/validate_phase3_4_fixes.py
```

This verifies:
- ✅ Fix 1: All 9 validation checks for --live mode support
- ✅ Fix 2: All 9 validation checks for config loading
- ✅ Fix 3: All 12 validation checks for WebSocket initialization

### Unit Tests
Focused unit tests are available at `tests/test_phase3_4_critical_fixes.py`:
```bash
pytest tests/test_phase3_4_critical_fixes.py -v
```

## Success Criteria ✅

All success criteria from the problem statement have been met:

- ✅ `python src/main.py --live --paper` command works
- ✅ Config loads reliably with clear error messages
- ✅ WebSocket connections establish properly or fail with clear errors
- ✅ System can run in paper trading mode
- ✅ Backwards compatible with Phase 4
- ✅ No breaking changes to existing APIs
- ✅ Comprehensive logging (INFO for success, ERROR for failures)
- ✅ Detailed error messages for debugging

## Backwards Compatibility

All changes are backwards compatible:
- Existing code that doesn't use --live mode continues to work
- LiveTradingEngine API remains unchanged
- ProductionCoordinator API remains unchanged (only internal method modified)
- All existing tests pass
- Configuration files continue to work as before

## Security Considerations

- No secrets are hardcoded
- Environment variables used for sensitive data
- Proper error handling prevents information leakage
- Input validation prevents injection attacks
- No new security vulnerabilities introduced

## Logging Improvements

All three fixes include enhanced logging:
- **INFO level**: Successful operations, configuration loaded, streams started
- **ERROR level**: Failures, validation errors, connection issues
- **WARNING level**: Fallbacks, limit adjustments, deprecated usage

## Example Usage

### Paper Trading Mode (Safest)
```bash
# Set up environment
export EXCHANGES="kucoinfutures"
export KUCOIN_FUTURES_API_KEY="your-key"
export KUCOIN_FUTURES_SECRET="your-secret"
export KUCOIN_FUTURES_PASSPHRASE="your-passphrase"
export EQUITY_USD=100
export CONFIG_PATH=config/config.example.yaml

# Start in paper mode
python src/main.py --live --paper
```

### Live Trading Mode (Real Money)
```bash
# Same setup as above, but set live mode
export TRADING_MODE=live

# Start in live mode
python src/main.py --live
```

### With Custom Duration
```bash
# Run for 1 hour (3600 seconds)
export TRADING_DURATION=3600
python src/main.py --live --paper
```

## Troubleshooting

### Issue: "No exchange clients configured"
**Solution**: Set the EXCHANGES environment variable
```bash
export EXCHANGES="bingx,kucoinfutures"
```

### Issue: "Config file not found"
**Solution**: Set CONFIG_PATH or use TRADING_SYMBOLS
```bash
export CONFIG_PATH=config/config.example.yaml
# OR
export TRADING_SYMBOLS="BTC/USDT:USDT,ETH/USDT:USDT"
```

### Issue: "No WebSocket streams were started"
**Solution**: Check exchange credentials and symbol configuration
```bash
# Verify symbols are valid for your exchange
export TRADING_SYMBOLS="BTC/USDT:USDT,ETH/USDT:USDT"
```

### Issue: "Stream limit exceeded"
**Solution**: Reduce number of symbols or timeframes in config
```yaml
universe:
  fixed_symbols:
    - BTC/USDT:USDT
    - ETH/USDT:USDT
    # Limit to 2-3 symbols per exchange

websocket:
  stream_timeframes:
    - 1m  # Use fewer timeframes
```

## Next Steps

After deploying these fixes:
1. Test with paper trading mode first
2. Verify WebSocket connections are stable
3. Monitor logs for any errors
4. Gradually increase symbol count if needed
5. Only switch to live mode after thorough testing

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify environment variables are set correctly
3. Run validation script: `python scripts/validate_phase3_4_fixes.py`
4. Run unit tests: `pytest tests/test_phase3_4_critical_fixes.py -v`
