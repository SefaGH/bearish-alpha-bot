#!/usr/bin/env python3
"""
Test duplicate prevention with Phase 2 optimized settings.

Validates:
- Cooldown period of 20 seconds (reduced from 30)
- Price delta threshold of 0.05% (reduced from 0.15%)
- Multi-symbol independent tracking
"""

import sys
import os
import time
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.strategy_coordinator import StrategyCoordinator


def create_mock_dependencies():
    """Create mock portfolio manager and risk manager"""
    portfolio_manager = MagicMock()
    risk_manager = MagicMock()
    
    # Configure portfolio manager with Phase 2 settings
    portfolio_manager.cfg = {
        'signals': {
            'duplicate_prevention': {
                'enabled': True,
                'min_price_change_pct': 0.05,  # Phase 2: reduced from 0.15
                'cooldown_seconds': 20          # Phase 2: reduced from 30
            }
        }
    }
    
    return portfolio_manager, risk_manager


def test_cooldown_duration():
    """Test that cooldown is 20 seconds as per Phase 2"""
    print("\n" + "=" * 70)
    print("TEST 1: Cooldown Duration (20 seconds)")
    print("=" * 70)
    
    pm, rm = create_mock_dependencies()
    coordinator = StrategyCoordinator(pm, rm)
    
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50000.0
    }
    
    # First signal should pass
    is_valid_1, reason_1 = coordinator.validate_duplicate(signal, 'adaptive_str')
    print(f"\nFirst signal: {is_valid_1} - {reason_1}")
    assert is_valid_1, "First signal should be accepted"
    
    # Immediate second signal should be rejected (within cooldown)
    is_valid_2, reason_2 = coordinator.validate_duplicate(signal, 'adaptive_str')
    print(f"Immediate second signal: {is_valid_2} - {reason_2}")
    assert not is_valid_2, "Second signal should be rejected (within cooldown)"
    assert "cooldown" in reason_2.lower(), "Rejection reason should mention cooldown"
    
    # Wait 10 seconds - should still be rejected
    print("\nWaiting 10 seconds...")
    time.sleep(10)
    is_valid_3, reason_3 = coordinator.validate_duplicate(signal, 'adaptive_str')
    print(f"After 10s: {is_valid_3} - {reason_3}")
    assert not is_valid_3, "Signal should still be rejected after 10s (< 20s cooldown)"
    
    # Wait another 11 seconds (total 21s) - should now pass
    print("Waiting another 11 seconds (total 21s)...")
    time.sleep(11)
    is_valid_4, reason_4 = coordinator.validate_duplicate(signal, 'adaptive_str')
    print(f"After 21s: {is_valid_4} - {reason_4}")
    assert is_valid_4, "Signal should be accepted after 21s (> 20s cooldown)"
    
    print("\n✅ PASS: Cooldown period is 20 seconds as expected")
    return True


def test_price_delta_bypass():
    """Test that 0.05% price movement bypasses cooldown"""
    print("\n" + "=" * 70)
    print("TEST 2: Price Delta Bypass (0.05%)")
    print("=" * 70)
    
    pm, rm = create_mock_dependencies()
    coordinator = StrategyCoordinator(pm, rm)
    
    # First signal at $50,000
    signal_1 = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50000.0
    }
    
    is_valid_1, reason_1 = coordinator.validate_duplicate(signal_1, 'adaptive_str')
    print(f"\nFirst signal at $50,000: {is_valid_1} - {reason_1}")
    assert is_valid_1, "First signal should be accepted"
    
    # Second signal at $50,010 (0.02% change) - should be rejected
    signal_2 = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50010.0  # 0.02% increase
    }
    
    is_valid_2, reason_2 = coordinator.validate_duplicate(signal_2, 'adaptive_str')
    price_change_2 = (50010 - 50000) / 50000 * 100
    print(f"\nSecond signal at $50,010 ({price_change_2:.2f}% change): {is_valid_2} - {reason_2}")
    assert not is_valid_2, f"Signal with {price_change_2:.2f}% change should be rejected (< 0.05%)"
    
    # Third signal at $50,025 (0.05% change) - should bypass cooldown
    signal_3 = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50025.0  # 0.05% increase
    }
    
    is_valid_3, reason_3 = coordinator.validate_duplicate(signal_3, 'adaptive_str')
    price_change_3 = (50025 - 50000) / 50000 * 100
    print(f"\nThird signal at $50,025 ({price_change_3:.2f}% change): {is_valid_3} - {reason_3}")
    assert is_valid_3, f"Signal with {price_change_3:.2f}% change should bypass cooldown (>= 0.05%)"
    assert "bypass" in reason_3.lower(), "Reason should mention bypass"
    
    # Fourth signal at $50,060 (0.10% change from last) - should also bypass
    signal_4 = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50060.0  # ~0.07% increase from 50025
    }
    
    is_valid_4, reason_4 = coordinator.validate_duplicate(signal_4, 'adaptive_str')
    price_change_4 = (50060 - 50025) / 50025 * 100
    print(f"\nFourth signal at $50,060 ({price_change_4:.2f}% change from last): {is_valid_4} - {reason_4}")
    assert is_valid_4, "Signal with significant price change should bypass cooldown"
    
    print("\n✅ PASS: Price delta bypass works at 0.05% threshold")
    return True


def test_multi_symbol_independence():
    """Test that different symbols have independent cooldowns"""
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Symbol Independence")
    print("=" * 70)
    
    pm, rm = create_mock_dependencies()
    coordinator = StrategyCoordinator(pm, rm)
    
    # Signal for BTC
    signal_btc = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50000.0
    }
    
    is_valid_btc_1, reason_btc_1 = coordinator.validate_duplicate(signal_btc, 'adaptive_str')
    print(f"\nBTC signal 1: {is_valid_btc_1} - {reason_btc_1}")
    assert is_valid_btc_1, "First BTC signal should be accepted"
    
    # Immediate second BTC signal should be rejected
    is_valid_btc_2, reason_btc_2 = coordinator.validate_duplicate(signal_btc, 'adaptive_str')
    print(f"BTC signal 2 (immediate): {is_valid_btc_2} - {reason_btc_2}")
    assert not is_valid_btc_2, "Second BTC signal should be rejected"
    
    # Signal for ETH should be accepted (different symbol)
    signal_eth = {
        'symbol': 'ETH/USDT:USDT',
        'side': 'sell',
        'entry': 3500.0
    }
    
    is_valid_eth_1, reason_eth_1 = coordinator.validate_duplicate(signal_eth, 'adaptive_str')
    print(f"\nETH signal 1: {is_valid_eth_1} - {reason_eth_1}")
    assert is_valid_eth_1, "ETH signal should be accepted (different symbol from BTC)"
    
    # Signal for SOL should be accepted (different symbol)
    signal_sol = {
        'symbol': 'SOL/USDT:USDT',
        'side': 'sell',
        'entry': 100.0
    }
    
    is_valid_sol_1, reason_sol_1 = coordinator.validate_duplicate(signal_sol, 'adaptive_str')
    print(f"\nSOL signal 1: {is_valid_sol_1} - {reason_sol_1}")
    assert is_valid_sol_1, "SOL signal should be accepted (different symbol from BTC/ETH)"
    
    # Immediate second ETH signal should be rejected (same symbol)
    is_valid_eth_2, reason_eth_2 = coordinator.validate_duplicate(signal_eth, 'adaptive_str')
    print(f"ETH signal 2 (immediate): {is_valid_eth_2} - {reason_eth_2}")
    assert not is_valid_eth_2, "Second ETH signal should be rejected"
    
    print("\n✅ PASS: Symbols have independent duplicate prevention")
    return True


def test_strategy_independence():
    """Test that different strategies for same symbol have independent cooldowns"""
    print("\n" + "=" * 70)
    print("TEST 4: Strategy Independence")
    print("=" * 70)
    
    pm, rm = create_mock_dependencies()
    coordinator = StrategyCoordinator(pm, rm)
    
    signal_btc = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50000.0
    }
    
    # Signal from adaptive_str
    is_valid_str, reason_str = coordinator.validate_duplicate(signal_btc, 'adaptive_str')
    print(f"\nBTC from adaptive_str: {is_valid_str} - {reason_str}")
    assert is_valid_str, "First adaptive_str signal should be accepted"
    
    # Immediate signal from same strategy should be rejected
    is_valid_str_2, reason_str_2 = coordinator.validate_duplicate(signal_btc, 'adaptive_str')
    print(f"BTC from adaptive_str (immediate): {is_valid_str_2} - {reason_str_2}")
    assert not is_valid_str_2, "Second adaptive_str signal should be rejected"
    
    # Signal from different strategy should be accepted
    is_valid_ob, reason_ob = coordinator.validate_duplicate(signal_btc, 'adaptive_ob')
    print(f"\nBTC from adaptive_ob: {is_valid_ob} - {reason_ob}")
    assert is_valid_ob, "Signal from adaptive_ob should be accepted (different strategy)"
    
    # Immediate second signal from adaptive_ob should be rejected
    is_valid_ob_2, reason_ob_2 = coordinator.validate_duplicate(signal_btc, 'adaptive_ob')
    print(f"BTC from adaptive_ob (immediate): {is_valid_ob_2} - {reason_ob_2}")
    assert not is_valid_ob_2, "Second adaptive_ob signal should be rejected"
    
    print("\n✅ PASS: Strategies have independent duplicate prevention")
    return True


def run_all_tests():
    """Run all duplicate prevention tests"""
    print("\n")
    print("=" * 70)
    print(" PHASE 2 DUPLICATE PREVENTION TEST SUITE")
    print("=" * 70)
    print("\nValidating Phase 2 optimized settings:")
    print("  • Cooldown: 20 seconds (reduced from 30)")
    print("  • Price delta: 0.05% (reduced from 0.15%)")
    print("  • Multi-symbol independence")
    print("  • Strategy independence")
    
    results = {
        'Cooldown Duration (20s)': test_cooldown_duration(),
        'Price Delta Bypass (0.05%)': test_price_delta_bypass(),
        'Multi-Symbol Independence': test_multi_symbol_independence(),
        'Strategy Independence': test_strategy_independence(),
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
        print(" ✅ ALL DUPLICATE PREVENTION TESTS PASSED")
        print("\n Phase 2 optimized settings validated:")
        print("   • 20-second cooldown working correctly")
        print("   • 0.05% price delta bypass functioning")
        print("   • Symbols independently tracked")
        print("   • Strategies independently tracked")
    else:
        print(" ❌ SOME TESTS FAILED")
    print("=" * 70)
    print()
    
    return all_passed


if __name__ == "__main__":
    try:
        print("\n⏱️  Note: This test includes time.sleep() calls and will take ~22 seconds to run")
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
