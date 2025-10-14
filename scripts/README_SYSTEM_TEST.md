# System Test Guide

## Overview

`system_test.py` is a comprehensive test suite that verifies all components of the Bearish Alpha Bot are functioning correctly across all implemented phases.

## What It Tests

### Phase 1: Multi-Exchange Integration (6 tests)
- ✅ KuCoin and BingX API initialization
- ✅ Server time synchronization
- ✅ Contract discovery
- ✅ MultiExchangeManager
- ✅ OHLCV data fetching
- ✅ WebSocket infrastructure

### Phase 2: Market Intelligence Engine (5 tests)
- ✅ Market regime analysis
- ✅ Base strategies (OversoldBounce, ShortTheRip)
- ✅ Adaptive strategies (Phase 2 enhancements)
- ✅ Performance monitoring
- ✅ VST intelligence system

### Phase 3: Portfolio & Risk Management (5 tests)
- ✅ Risk manager (position sizing, risk assessment)
- ✅ Portfolio manager (strategy registration, allocation)
- ✅ Strategy coordinator (signal processing, coordination)
- ✅ Position management
- ✅ Live trading engine

### Phase 4: AI Enhancement System (5 tests)
- ✅ ML regime prediction (Phase 4.1)
- ✅ Adaptive learning with RL (Phase 4.2)
- ✅ Strategy optimization (Phase 4.3)
- ✅ Advanced price prediction (Phase 4 Final)
- ✅ Feature engineering pipeline

### Integration Tests (3 tests)
- ✅ Phase 1 + Phase 2 integration
- ✅ Phase 1 + Phase 3 integration
- ✅ Complete pipeline integration

## Running the Tests

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Set API credentials (for full testing)
export KUCOIN_API_KEY="your_key"
export KUCOIN_API_SECRET="your_secret"
export KUCOIN_API_PASSPHRASE="your_passphrase"
export BINGX_API_KEY="your_key"
export BINGX_API_SECRET="your_secret"
```

### Run Tests

```bash
# Run all system tests
python3 scripts/system_test.py

# Or make it executable and run directly
chmod +x scripts/system_test.py
./scripts/system_test.py
```

### Expected Output

```
======================================================================
BEARISH ALPHA BOT - COMPREHENSIVE SYSTEM TEST
======================================================================
Started at: 2025-10-14T00:09:43.045167+00:00

======================================================================
Phase 1: Exchange Connection Tests
======================================================================
Testing multi-exchange infrastructure...
✓ PASS: KuCoin initialization
✓ PASS: BingX initialization
✓ PASS: Server time synchronization
✓ PASS: Contract discovery
✓ PASS: MultiExchangeManager

[... more tests ...]

======================================================================
SYSTEM TEST SUMMARY
======================================================================
Test Groups: 6/6 passed
Total Time: 9.13s

✓ PASS: Exchange Connections
✓ PASS: Data Pipeline
✓ PASS: Strategy System
✓ PASS: Portfolio Manager
✓ PASS: ML Models
✓ PASS: System Integration

======================================================================
Individual Tests: 27/27 passed

✅ ALL SYSTEM TESTS PASSED!

System Status: OPERATIONAL
- Phase 1: Multi-Exchange Integration ✓
- Phase 2: Market Intelligence Engine ✓
- Phase 3: Portfolio Management ✓
- Phase 4: AI Enhancement System ✓
======================================================================
```

## Test Details

### Test Structure

Each test group is organized by phase and includes:

1. **Component Import Tests**: Verify all modules can be imported
2. **Initialization Tests**: Verify components can be instantiated
3. **Functionality Tests**: Verify core methods work correctly
4. **Integration Tests**: Verify components work together

### Exit Codes

- `0`: All tests passed ✅
- `1`: Some tests failed ⚠️

### Test Duration

Expected runtime: **8-10 seconds**

This includes:
- API calls to exchange servers
- OHLCV data fetching
- ML model loading
- Component initialization

## Troubleshooting

### Common Issues

#### Missing Dependencies

```
ModuleNotFoundError: No module named 'ccxt'
```

**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

#### Network Issues

```
✗ FAIL: Contract discovery
  Network restricted or timeout
```

**Solution**: Check network connectivity or firewall settings

#### API Rate Limits

```
✗ FAIL: OHLCV data fetching
  Rate limit exceeded
```

**Solution**: Wait a few seconds and retry, or use API credentials

### Getting Help

If tests fail unexpectedly:

1. Check the error messages in the output
2. Review the individual test details
3. Run specific test files in `tests/` directory
4. Check component logs for more details

## Integration with CI/CD

This test can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run System Tests
  run: |
    pip install -r requirements.txt
    python3 scripts/system_test.py
```

## Related Tests

- `tests/smoke_test.py` - Quick smoke tests without API calls
- `tests/test_market_intelligence.py` - Phase 2 detailed tests
- `tests/test_portfolio_management.py` - Phase 3 detailed tests
- `tests/test_ml_regime_prediction.py` - Phase 4.1 detailed tests
- `tests/test_price_prediction.py` - Phase 4 Final detailed tests

## Maintenance

When adding new components:

1. Add new test methods to appropriate phase section
2. Update test count in summary
3. Update this README with new test descriptions
4. Ensure tests follow existing patterns

## Version History

- **v1.0** (2025-10-14): Initial comprehensive system test
  - 27 tests covering all 4 phases
  - 6 test groups
  - ~9 second execution time
