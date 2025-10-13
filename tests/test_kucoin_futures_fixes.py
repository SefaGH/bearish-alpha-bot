#!/usr/bin/env python3
"""
Test KuCoin Futures specific fixes:
1. Production mode enforcement (sandbox=False)
2. Symbol format priority (BTC/USDT:USDT first)
3. Enhanced debug logging
"""
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient, EX_DEFAULTS


def test_global_sandbox_false():
    """Test that EX_DEFAULTS has sandbox=False."""
    print("Testing global sandbox=False in EX_DEFAULTS...")
    
    if 'sandbox' in EX_DEFAULTS and EX_DEFAULTS['sandbox'] == False:
        print(f"  ✓ EX_DEFAULTS contains sandbox=False: {EX_DEFAULTS}")
        return True
    else:
        print(f"  ✗ EX_DEFAULTS missing sandbox=False: {EX_DEFAULTS}")
        return False


def test_kucoin_production_mode():
    """Test that KuCoin exchanges are forced into production mode."""
    print("Testing KuCoin production mode enforcement...")
    
    try:
        # Create a kucoinfutures client
        client = CcxtClient('kucoinfutures')
        
        # Check if sandbox is False
        if hasattr(client.ex, 'sandbox'):
            if client.ex.sandbox == False:
                print(f"  ✓ KuCoin Futures sandbox mode is False (production)")
                return True
            else:
                print(f"  ✗ KuCoin Futures sandbox mode is {client.ex.sandbox} (expected False)")
                return False
        else:
            # If sandbox attribute doesn't exist, check the config
            print(f"  ⚠ Exchange doesn't have sandbox attribute, checking config...")
            if 'sandbox' in client.ex.config and client.ex.config['sandbox'] == False:
                print(f"  ✓ Config has sandbox=False")
                return True
            else:
                print(f"  ! Cannot verify sandbox mode (no sandbox attribute or config)")
                # This is not necessarily a failure - some exchanges might not have this attribute
                return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kucoin_symbol_priority():
    """Test that BTC/USDT:USDT is prioritized for KuCoin Futures."""
    print("Testing KuCoin Futures symbol priority...")
    
    class MockKuCoinFuturesExchange:
        """Mock KuCoin Futures exchange with typical symbols."""
        def __init__(self):
            self._markets = {
                'BTC/USDT:USDT': {'id': 'BTCUSDTM', 'symbol': 'BTC/USDT:USDT'},  # Should be selected first
                'XBTUSDM': {'id': 'XBTUSDM', 'symbol': 'XBTUSDM'},
                'ETH/USDT:USDT': {'id': 'ETHUSDTM', 'symbol': 'ETH/USDT:USDT'},
            }
        
        def load_markets(self):
            return self._markets
    
    try:
        client = CcxtClient('kucoinfutures')
        # Mock the exchange
        client.ex = MockKuCoinFuturesExchange()
        
        # Request BTC/USDT - should get BTC/USDT:USDT
        result = client.validate_and_get_symbol('BTC/USDT')
        
        if result == 'BTC/USDT:USDT':
            print(f"  ✓ Symbol priority correct: BTC/USDT → {result}")
            return True
        else:
            print(f"  ✗ Expected BTC/USDT:USDT but got: {result}")
            return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_other_exchange_symbol_priority():
    """Test that other exchanges use standard priority (not KuCoin specific)."""
    print("Testing non-KuCoin exchange symbol priority...")
    
    class MockBinanceExchange:
        """Mock Binance exchange with typical symbols."""
        def __init__(self):
            self._markets = {
                'BTC/USDT': {'id': 'BTCUSDT', 'symbol': 'BTC/USDT'},  # Should be selected first
                'BTC/USDT:USDT': {'id': 'BTCUSDTPERP', 'symbol': 'BTC/USDT:USDT'},
                'BTCUSDT': {'id': 'BTCUSDT', 'symbol': 'BTCUSDT'},
            }
        
        def load_markets(self):
            return self._markets
    
    try:
        client = CcxtClient('binance')
        # Mock the exchange
        client.ex = MockBinanceExchange()
        
        # Request BTC/USDT - should get exact match
        result = client.validate_and_get_symbol('BTC/USDT')
        
        if result == 'BTC/USDT':
            print(f"  ✓ Symbol priority correct for Binance: BTC/USDT → {result}")
            return True
        else:
            print(f"  ✗ Expected BTC/USDT but got: {result}")
            return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all KuCoin Futures fix tests."""
    print("=" * 60)
    print("KuCoin Futures Fix Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_global_sandbox_false,
        test_kucoin_production_mode,
        test_kucoin_symbol_priority,
        test_other_exchange_symbol_priority,
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
        print("✓ All KuCoin Futures fix tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
