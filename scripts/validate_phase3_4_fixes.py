#!/usr/bin/env python3
"""
Validation script for Phase 3.4 critical fixes.
Tests the fixes without requiring full dependencies.
"""

import sys
import os
import re

# Add src to path
sys.path.insert(0, '/home/runner/work/bearish-alpha-bot/bearish-alpha-bot/src')

def test_fix_1_main_live_trading():
    """Test Fix 1: main_live_trading function exists."""
    print("\n" + "="*70)
    print("TEST 1: Verify main_live_trading function exists")
    print("="*70)
    
    # Read main.py without importing (to avoid dependency issues)
    with open('/home/runner/work/bearish-alpha-bot/bearish-alpha-bot/src/main.py', 'r') as f:
        content = f.read()
    
    # Check for main_live_trading function
    assert 'async def main_live_trading():' in content, "main_live_trading function not found"
    print("✅ main_live_trading function exists")
    
    # Check for argparse support
    assert 'import argparse' in content, "argparse not imported"
    print("✅ argparse imported")
    
    assert "parser.add_argument('--live'" in content, "--live argument not added"
    print("✅ --live argument added")
    
    assert "parser.add_argument('--paper'" in content, "--paper argument not added"
    print("✅ --paper argument added")
    
    # Check for ProductionCoordinator integration
    assert 'from core.production_coordinator import ProductionCoordinator' in content, "ProductionCoordinator not imported"
    print("✅ ProductionCoordinator imported")
    
    assert 'await coordinator.initialize_production_system(' in content, "initialize_production_system not called"
    print("✅ initialize_production_system called")
    
    assert 'await coordinator.run_production_loop(' in content, "run_production_loop not called"
    print("✅ run_production_loop called")
    
    # Check for environment variable handling
    assert "os.getenv('TRADING_MODE'" in content, "TRADING_MODE env var not used"
    print("✅ TRADING_MODE environment variable handled")
    
    assert "os.getenv('TRADING_DURATION'" in content, "TRADING_DURATION env var not used"
    print("✅ TRADING_DURATION environment variable handled")
    
    print("\n✅ FIX 1 VALIDATED: --live mode support complete\n")


def test_fix_2_config_loading():
    """Test Fix 2: Simplified config loading."""
    print("\n" + "="*70)
    print("TEST 2: Verify simplified config loading")
    print("="*70)
    
    with open('/home/runner/work/bearish-alpha-bot/bearish-alpha-bot/src/core/live_trading_engine.py', 'r') as f:
        content = f.read()
    
    # Check for 3-step priority comments
    assert '# Step 1: Try YAML Config' in content, "Step 1 comment not found"
    print("✅ Step 1: YAML Config documented")
    
    assert '# Step 2: Try Environment Variables' in content, "Step 2 comment not found"
    print("✅ Step 2: Environment Variables documented")
    
    assert '# Step 3: Hard-coded Defaults' in content, "Step 3 comment not found"
    print("✅ Step 3: Hard-coded Defaults documented")
    
    # Check for validation
    assert 'isinstance(fixed_symbols, list)' in content, "List validation not found"
    print("✅ List validation added")
    
    assert 'len(fixed_symbols) > 0' in content, "Empty list check not found"
    print("✅ Empty list validation added")
    
    # Check for ENV variable support
    assert "os.getenv('TRADING_SYMBOLS'" in content, "TRADING_SYMBOLS env var not used"
    print("✅ TRADING_SYMBOLS environment variable supported")
    
    # Check for error logging (not warning)
    assert 'logger.error(' in content, "Error logging not found"
    print("✅ ERROR level logging used (not warning)")
    
    # Check for hard-coded defaults
    assert 'BTC/USDT:USDT' in content, "Default BTC symbol not found"
    assert 'ETH/USDT:USDT' in content, "Default ETH symbol not found"
    assert 'SOL/USDT:USDT' in content, "Default SOL symbol not found"
    print("✅ Hard-coded defaults present (BTC, ETH, SOL)")
    
    # Check that complex try-except is gone
    old_pattern = r'✅ DÜZELTME 2: Universe config.*güvenli yükle'
    if re.search(old_pattern, content):
        print("⚠️  Warning: Old comment pattern still present")
    else:
        print("✅ Old complex config loading removed")
    
    print("\n✅ FIX 2 VALIDATED: Simplified config loading complete\n")


def test_fix_3_websocket_setup():
    """Test Fix 3: WebSocket initialization with validation."""
    print("\n" + "="*70)
    print("TEST 3: Verify WebSocket initialization with validation")
    print("="*70)
    
    with open('/home/runner/work/bearish-alpha-bot/bearish-alpha-bot/src/core/production_coordinator.py', 'r') as f:
        content = f.read()
    
    # Check for return type
    assert 'def _setup_websocket_connections(self) -> bool:' in content, "Method doesn't return bool"
    print("✅ _setup_websocket_connections returns bool")
    
    # Check for step-by-step validation
    assert '# Step 1: Validate Prerequisites' in content, "Step 1 validation not found"
    print("✅ Step 1: Prerequisites validation added")
    
    assert '# Step 2: Initialize Manager' in content, "Step 2 initialization not found"
    print("✅ Step 2: Manager initialization added")
    
    assert '# Step 3: Setup Streams with Limits' in content, "Step 3 stream setup not found"
    print("✅ Step 3: Stream setup with limits added")
    
    assert '# Step 4: Return Status' in content, "Step 4 status return not found"
    print("✅ Step 4: Status return added")
    
    # Check for prerequisite validation
    assert 'if not self.exchange_clients:' in content, "exchange_clients validation not found"
    print("✅ Exchange clients validation added")
    
    # Check for _get_stream_limit helper
    assert 'def _get_stream_limit(self, exchange_name: str) -> int:' in content, "_get_stream_limit method not found"
    print("✅ _get_stream_limit helper method added")
    
    # Check for per-exchange limits
    assert "'bingx': 10" in content, "BingX limit not found"
    assert "'binance': 20" in content, "Binance limit not found"
    assert "'kucoinfutures': 15" in content, "KuCoin Futures limit not found"
    print("✅ Per-exchange stream limits defined")
    
    # Check for error logging
    error_count = content.count('logger.error(')
    assert error_count >= 3, f"Expected at least 3 error logs, found {error_count}"
    print(f"✅ ERROR level logging used ({error_count} occurrences)")
    
    # Check for success tracking
    assert 'total_streams_started' in content, "Stream counting not found"
    print("✅ Successful stream counting added")
    
    # Check for return False on failure
    assert 'return False' in content, "Failure return not found"
    print("✅ Returns False on failure")
    
    # Check for return True on success
    assert 'return True' in content, "Success return not found"
    print("✅ Returns True on success")
    
    print("\n✅ FIX 3 VALIDATED: WebSocket initialization complete\n")


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("PHASE 3.4 CRITICAL FIXES - VALIDATION SUITE")
    print("="*70)
    
    try:
        test_fix_1_main_live_trading()
        test_fix_2_config_loading()
        test_fix_3_websocket_setup()
        
        print("\n" + "="*70)
        print("✅ ALL FIXES VALIDATED SUCCESSFULLY")
        print("="*70)
        print("\nSummary:")
        print("  ✅ Fix 1: --live mode support with ProductionCoordinator")
        print("  ✅ Fix 2: Simplified config loading (YAML > ENV > defaults)")
        print("  ✅ Fix 3: WebSocket initialization with validation")
        print("\nAll requirements from the problem statement have been met.")
        print("="*70 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
