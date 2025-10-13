#!/usr/bin/env python3
"""
Test symbol validation for CcxtClient.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient


class MockExchange:
    """Mock exchange for testing symbol validation."""
    def __init__(self):
        self._markets = {
            'BTC/USDT:USDT': {'id': 'BTCUSDTM', 'symbol': 'BTC/USDT:USDT'},  # KuCoin futures format
            'ETH/USDT:USDT': {'id': 'ETHUSDTM', 'symbol': 'ETH/USDT:USDT'},
            'SOL/USDT:USDT': {'id': 'SOLUSDTM', 'symbol': 'SOL/USDT:USDT'},
        }
    
    def load_markets(self):
        return self._markets


def test_validate_exact_match():
    """Test that exact symbol match is returned."""
    print("Testing exact symbol match...")
    
    try:
        client = CcxtClient('kucoinfutures')
        # Mock the markets method
        client.ex = MockExchange()
        
        # Test with exact match
        result = client.validate_and_get_symbol('BTC/USDT:USDT')
        
        if result == 'BTC/USDT:USDT':
            print(f"  ✓ Exact match returned: {result}")
            return True
        else:
            print(f"  ✗ Expected BTC/USDT:USDT but got: {result}")
            return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validate_fallback():
    """Test that fallback mechanism works for BTC variants."""
    print("Testing BTC variant fallback...")
    
    try:
        client = CcxtClient('kucoinfutures')
        # Mock the markets method
        client.ex = MockExchange()
        
        # Request BTC/USDT which doesn't exist, should fallback to BTC/USDT:USDT
        result = client.validate_and_get_symbol('BTC/USDT')
        
        if result == 'BTC/USDT:USDT':
            print(f"  ✓ Fallback worked: BTC/USDT → {result}")
            return True
        else:
            print(f"  ✗ Expected BTC/USDT:USDT but got: {result}")
            return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validate_invalid_symbol():
    """Test that invalid symbols raise SystemExit."""
    print("Testing invalid symbol handling...")
    
    try:
        client = CcxtClient('kucoinfutures')
        # Mock the markets method
        client.ex = MockExchange()
        
        # Try an invalid symbol that doesn't exist and has no fallback
        try:
            result = client.validate_and_get_symbol('INVALID/PAIR')
            print(f"  ✗ Should have raised SystemExit but got: {result}")
            return False
        except SystemExit as e:
            print(f"  ✓ Correctly raised SystemExit for invalid symbol")
            return True
    except Exception as e:
        print(f"  ✗ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all symbol validation tests."""
    print("=" * 60)
    print("CcxtClient Symbol Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_validate_exact_match,
        test_validate_fallback,
        test_validate_invalid_symbol,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All symbol validation tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
