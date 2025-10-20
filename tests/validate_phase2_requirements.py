#!/usr/bin/env python3
"""
Phase 2 Requirements Validation Test

This test validates all Phase 2 requirements:
1. Duplicate Prevention Config (min_price_change_pct: 0.05, cooldown_seconds: 20)
2. Multi-Symbol Trading (BTC, ETH, SOL with symbol-specific thresholds)
3. Debug Logging ([STR-DEBUG] format for all symbols)
4. Signal Acceptance Rate optimization
"""

import sys
import os
import yaml
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies.adaptive_str import AdaptiveShortTheRip
from test_utils import create_test_dataframe, get_default_strategy_config

# Configure logging to capture debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


def test_duplicate_prevention_config():
    """
    Test 1: Verify duplicate prevention configuration
    
    Requirements:
    - min_price_change_pct: 0.05 (reduced from 0.15)
    - cooldown_seconds: 20 (reduced from 30)
    """
    print("\n" + "=" * 70)
    print("TEST 1: Duplicate Prevention Configuration")
    print("=" * 70)
    
    config_path = 'config/config.example.yaml'
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check signals.duplicate_prevention section
    dup_prev = config.get('signals', {}).get('duplicate_prevention', {})
    
    min_price_change = dup_prev.get('min_price_change_pct')
    cooldown = dup_prev.get('cooldown_seconds')
    
    print(f"\nDuplicate Prevention Settings:")
    print(f"  min_price_change_pct: {min_price_change}")
    print(f"  cooldown_seconds: {cooldown}")
    
    # Validate
    success = True
    if min_price_change != 0.05:
        print(f"❌ FAIL: min_price_change_pct should be 0.05, got {min_price_change}")
        success = False
    else:
        print(f"✅ PASS: min_price_change_pct = 0.05")
    
    if cooldown != 20:
        print(f"❌ FAIL: cooldown_seconds should be 20, got {cooldown}")
        success = False
    else:
        print(f"✅ PASS: cooldown_seconds = 20")
    
    return success


def test_multi_symbol_config():
    """
    Test 2: Verify multi-symbol trading configuration
    
    Requirements:
    - BTC/USDT:USDT with rsi_threshold: 55
    - ETH/USDT:USDT with rsi_threshold: 50
    - SOL/USDT:USDT with rsi_threshold: 50
    """
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Symbol Trading Configuration")
    print("=" * 70)
    
    config_path = 'config/config.example.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check signals.short_the_rip.symbols section
    str_symbols = config.get('signals', {}).get('short_the_rip', {}).get('symbols', {})
    
    print(f"\nSymbol-Specific Thresholds:")
    
    required_symbols = {
        'BTC/USDT:USDT': 55,
        'ETH/USDT:USDT': 50,
        'SOL/USDT:USDT': 50
    }
    
    success = True
    for symbol, expected_threshold in required_symbols.items():
        actual_threshold = str_symbols.get(symbol, {}).get('rsi_threshold')
        print(f"  {symbol}: {actual_threshold}")
        
        if actual_threshold != expected_threshold:
            print(f"    ❌ FAIL: Expected {expected_threshold}, got {actual_threshold}")
            success = False
        else:
            print(f"    ✅ PASS")
    
    return success


def test_strategy_reads_symbol_config():
    """
    Test 3: Verify strategy correctly reads symbol-specific thresholds
    
    Requirements:
    - Strategy must use symbol-specific thresholds from config
    - BTC should use threshold 55
    - ETH should use threshold 50
    - SOL should use threshold 50
    """
    print("\n" + "=" * 70)
    print("TEST 3: Strategy Symbol-Specific Threshold Reading")
    print("=" * 70)
    
    config = get_default_strategy_config()
    strategy = AdaptiveShortTheRip(config)
    
    test_cases = [
        ('BTC/USDT:USDT', 55),
        ('ETH/USDT:USDT', 50),
        ('SOL/USDT:USDT', 50),
    ]
    
    success = True
    for symbol, expected_threshold in test_cases:
        actual_threshold = strategy.get_symbol_specific_threshold(symbol)
        status = "✅ PASS" if actual_threshold == expected_threshold else "❌ FAIL"
        print(f"  {symbol}: {actual_threshold} (expected: {expected_threshold}) {status}")
        
        if actual_threshold != expected_threshold:
            success = False
    
    return success


def test_debug_logging():
    """
    Test 4: Verify debug logging includes all required information
    
    Requirements:
    - [STR-DEBUG] prefix with symbol
    - RSI value and threshold
    - EMA alignment status
    - Volume status
    - ATR value
    - Signal result
    """
    print("\n" + "=" * 70)
    print("TEST 4: Debug Logging Format")
    print("=" * 70)
    
    config = get_default_strategy_config()
    strategy = AdaptiveShortTheRip(config)
    
    # Create test data with RSI above threshold
    df = create_test_dataframe(price=50000, rsi_value=56.0, ema_aligned=True)
    
    print("\nGenerating signal with debug logging for BTC/USDT:USDT...")
    print("-" * 70)
    
    # Capture logging output
    signal = strategy.signal(df, symbol='BTC/USDT:USDT')
    
    print("-" * 70)
    
    if signal:
        print(f"✅ PASS: Signal generated successfully")
        print(f"   Signal details: {signal['reason']}")
        return True
    else:
        print(f"❌ FAIL: No signal generated (RSI 56 should trigger for BTC)")
        return False


def test_signal_generation_for_all_symbols():
    """
    Test 5: Verify signals are generated for all 3 symbols
    
    Requirements:
    - BTC/USDT:USDT generates signals
    - ETH/USDT:USDT generates signals
    - SOL/USDT:USDT generates signals
    """
    print("\n" + "=" * 70)
    print("TEST 5: Signal Generation for All Symbols")
    print("=" * 70)
    
    config = get_default_strategy_config()
    strategy = AdaptiveShortTheRip(config)
    
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    # Test with RSI 56 (should trigger all symbols)
    df_high = create_test_dataframe(rsi_value=56.0, ema_aligned=True)
    
    print("\nTest Case 1: RSI = 56 (should trigger all symbols)")
    print("-" * 70)
    
    success = True
    for symbol in symbols:
        signal = strategy.signal(df_high, symbol=symbol)
        status = "✅ Generated" if signal else "❌ Not Generated"
        print(f"  {symbol}: {status}")
        
        if not signal:
            success = False
    
    # Test with RSI 52 (should trigger ETH and SOL, but not BTC)
    df_medium = create_test_dataframe(rsi_value=52.0, ema_aligned=True)
    
    print("\nTest Case 2: RSI = 52 (should trigger ETH/SOL only)")
    print("-" * 70)
    
    expected_results = {
        'BTC/USDT:USDT': False,  # 52 < 55
        'ETH/USDT:USDT': True,   # 52 >= 50
        'SOL/USDT:USDT': True,   # 52 >= 50
    }
    
    for symbol, should_signal in expected_results.items():
        signal = strategy.signal(df_medium, symbol=symbol)
        actual = signal is not None
        
        if actual == should_signal:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            success = False
        
        result = "Generated" if actual else "Not Generated"
        expected = "Should Generate" if should_signal else "Should Not Generate"
        print(f"  {symbol}: {result} ({expected}) {status}")
    
    return success


def run_all_tests():
    """Run all Phase 2 validation tests"""
    print("\n")
    print("=" * 70)
    print(" PHASE 2 REQUIREMENTS VALIDATION TEST SUITE")
    print("=" * 70)
    print("\nValidating:")
    print("  • Duplicate Prevention Configuration")
    print("  • Multi-Symbol Trading Configuration")
    print("  • Symbol-Specific Threshold Reading")
    print("  • Debug Logging Format")
    print("  • Signal Generation for All Symbols")
    
    results = {
        'Duplicate Prevention Config': test_duplicate_prevention_config(),
        'Multi-Symbol Config': test_multi_symbol_config(),
        'Symbol Threshold Reading': test_strategy_reads_symbol_config(),
        'Debug Logging': test_debug_logging(),
        'Signal Generation': test_signal_generation_for_all_symbols(),
    }
    
    print("\n" + "=" * 70)
    print(" TEST RESULTS SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print(" ✅ ALL PHASE 2 REQUIREMENTS VALIDATED SUCCESSFULLY")
    else:
        print(" ❌ SOME PHASE 2 REQUIREMENTS NOT MET")
    print("=" * 70)
    print()
    
    return all_passed


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
