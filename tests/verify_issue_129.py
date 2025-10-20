#!/usr/bin/env python3
"""
Verification script for Issue #129 - Duplicate Prevention Optimization.
Demonstrates that the new config is properly loaded and applied.
"""

import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import Mock
from core.strategy_coordinator import StrategyCoordinator


def load_config():
    """Load config from config.example.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.example.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("=" * 70)
    print("Issue #129: Duplicate Prevention Optimization Verification")
    print("=" * 70)
    
    # Load config
    config = load_config()
    
    # Check new config section exists
    print("\n1. Checking config.example.yaml...")
    if 'signals' in config and 'duplicate_prevention' in config['signals']:
        dp_config = config['signals']['duplicate_prevention']
        print("   ✅ signals.duplicate_prevention section found")
        print(f"   ✅ min_price_change_pct: {dp_config.get('min_price_change_pct')}%")
        print(f"   ✅ cooldown_seconds: {dp_config.get('cooldown_seconds')}s")
        
        # Verify values match requirements
        assert dp_config.get('min_price_change_pct') == 0.05, "Expected min_price_change_pct = 0.05"
        assert dp_config.get('cooldown_seconds') == 20, "Expected cooldown_seconds = 20"
        print("   ✅ Values match Issue #129 requirements")
    else:
        print("   ❌ signals.duplicate_prevention section NOT found")
        return 1
    
    # Test coordinator reads config correctly
    print("\n2. Testing StrategyCoordinator configuration...")
    portfolio_manager = Mock()
    portfolio_manager.cfg = config
    risk_manager = Mock()
    
    coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
    
    # Test with signals to verify config is used
    signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
    is_valid, reason = coordinator.validate_duplicate(signal1, 'test_strategy')
    
    if is_valid:
        print("   ✅ First signal accepted (expected)")
    else:
        print(f"   ❌ First signal rejected (unexpected): {reason}")
        return 1
    
    # Test with signal that has 0.06% price change (should be accepted with new config)
    signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50030, 'side': 'long'}
    is_valid, reason = coordinator.validate_duplicate(signal2, 'test_strategy')
    
    if is_valid and 'bypass' in reason.lower():
        print(f"   ✅ Signal with 0.06% price change accepted via bypass (expected)")
    elif is_valid:
        print(f"   ✅ Signal accepted: {reason}")
    else:
        print(f"   ❌ Signal with 0.06% price change rejected (unexpected): {reason}")
        return 1
    
    # Test with signal that has 0.03% price change from last accepted (should be rejected)
    signal3 = {'symbol': 'BTC/USDT:USDT', 'entry': 50045, 'side': 'long'}  # 0.03% from signal2
    is_valid, reason = coordinator.validate_duplicate(signal3, 'test_strategy')
    
    if not is_valid and 'cooldown' in reason.lower():
        print(f"   ✅ Signal with small price change rejected (expected)")
    elif is_valid:
        print(f"   ❌ Signal with small price change accepted (unexpected): {reason}")
        return 1
    
    # Check statistics
    stats = coordinator.get_duplicate_prevention_stats()
    print(f"\n3. Duplicate Prevention Statistics:")
    print(f"   Total signals: {stats['total_signals_processed']}")
    print(f"   Cooldown bypasses: {stats['cooldown_bypasses']}")
    print(f"   Rejected by price delta: {stats['rejected_by_price_delta']}")
    
    print("\n" + "=" * 70)
    print("✅ All verifications passed!")
    print("=" * 70)
    print("\nSummary:")
    print("- Config updated: min_price_change_pct = 0.05% (was 0.15%)")
    print("- Config updated: cooldown_seconds = 20s (was 30s)")
    print("- Strategy coordinator correctly reads new config")
    print("- Signal acceptance logic working as expected")
    print("- Expected outcome: 70%+ signal acceptance rate in low-volatility markets")
    print("\nNext steps:")
    print("- Deploy config to production")
    print("- Monitor signal acceptance rate in live trading")
    print("- Verify no spam trades occur")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
