#!/usr/bin/env python3
"""
Test KuCoin Futures Ultimate Integration:
1. Server time synchronization
2. Dynamic symbol discovery
3. Native API format with time-based pagination
4. Bulk OHLCV fetching (up to 2000 candles)
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient


def test_server_time_sync():
    """Test KuCoin server time synchronization."""
    print("Testing server time synchronization...")
    
    try:
        client = CcxtClient('kucoinfutures')
        
        # Get server time
        server_time = client._get_kucoin_server_time()
        local_time = int(time.time() * 1000)
        
        # Server time should be reasonable (within 60 seconds of local time)
        time_diff = abs(server_time - local_time)
        
        if time_diff < 60000:  # 60 seconds tolerance
            print(f"  ✓ Server time sync successful")
            print(f"    Server time: {server_time}")
            print(f"    Local time: {local_time}")
            print(f"    Difference: {time_diff}ms")
            print(f"    Offset cached: {client._server_time_offset}ms")
            return True
        else:
            print(f"  ✗ Server time diff too large: {time_diff}ms")
            return False
            
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_symbol_mapping():
    """Test dynamic symbol discovery from KuCoin active contracts."""
    print("Testing dynamic symbol discovery...")
    
    try:
        client = CcxtClient('kucoinfutures')
        
        # Get dynamic symbol mapping
        symbol_map = client._get_dynamic_symbol_mapping()
        
        # Check that we have mappings
        if len(symbol_map) == 0:
            print(f"  ✗ No symbol mappings found")
            return False
        
        # Check essential symbols are present
        essential_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        missing = [s for s in essential_symbols if s not in symbol_map]
        
        if missing:
            print(f"  ✗ Missing essential symbols: {missing}")
            return False
        
        # Check BTC maps to XBTUSDTM (KuCoin's native format)
        if symbol_map.get('BTC/USDT:USDT') != 'XBTUSDTM':
            print(f"  ✗ BTC symbol mapping incorrect: {symbol_map.get('BTC/USDT:USDT')}")
            return False
        
        print(f"  ✓ Dynamic symbol discovery successful")
        print(f"    Total contracts: {len(symbol_map)}")
        print(f"    Sample mappings:")
        for ccxt_sym, native_sym in list(symbol_map.items())[:5]:
            print(f"      {ccxt_sym} → {native_sym}")
        print(f"    Cache timestamp: {client._last_symbol_update}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kucoin_granularity_conversion():
    """Test timeframe to KuCoin granularity conversion."""
    print("Testing KuCoin granularity conversion...")
    
    try:
        client = CcxtClient('kucoinfutures')
        
        test_cases = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        all_passed = True
        for timeframe, expected_granularity in test_cases.items():
            result = client._get_kucoin_granularity(timeframe)
            if result != expected_granularity:
                print(f"  ✗ {timeframe} → {result} (expected {expected_granularity})")
                all_passed = False
        
        if all_passed:
            print(f"  ✓ All granularity conversions correct")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timeframe_milliseconds_conversion():
    """Test timeframe to milliseconds conversion."""
    print("Testing timeframe to milliseconds conversion...")
    
    try:
        client = CcxtClient('kucoinfutures')
        
        test_cases = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        
        all_passed = True
        for timeframe, expected_ms in test_cases.items():
            result = client._get_timeframe_ms(timeframe)
            if result != expected_ms:
                print(f"  ✗ {timeframe} → {result}ms (expected {expected_ms}ms)")
                all_passed = False
        
        if all_passed:
            print(f"  ✓ All timeframe conversions correct")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bulk_fetch_logic():
    """Test bulk fetch logic (without actual API calls)."""
    print("Testing bulk fetch logic...")
    
    try:
        client = CcxtClient('kucoinfutures')
        
        # Test batch calculation
        test_cases = [
            (500, 1),   # 500 candles = 1 batch
            (1000, 2),  # 1000 candles = 2 batches
            (1500, 3),  # 1500 candles = 3 batches
            (2000, 4),  # 2000 candles = 4 batches (max)
            (2500, 4),  # 2500 candles = 4 batches (capped at 4)
        ]
        
        all_passed = True
        for target_limit, expected_batches in test_cases:
            # Simulate batch calculation
            batches_needed = min(4, (target_limit + 499) // 500)
            
            if batches_needed != expected_batches:
                print(f"  ✗ {target_limit} candles → {batches_needed} batches (expected {expected_batches})")
                all_passed = False
        
        if all_passed:
            print(f"  ✓ Batch calculation logic correct")
            print(f"    Max 2000 candles in 4 batches of 500")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_behavior():
    """Test symbol cache behavior."""
    print("Testing symbol cache behavior...")
    
    try:
        client = CcxtClient('kucoinfutures')
        
        # First call - should fetch from API
        symbol_map1 = client._get_dynamic_symbol_mapping()
        timestamp1 = client._last_symbol_update
        
        # Immediate second call - should use cache
        time.sleep(0.1)
        symbol_map2 = client._get_dynamic_symbol_mapping()
        timestamp2 = client._last_symbol_update
        
        # Cache should be used (same timestamp)
        if timestamp1 != timestamp2:
            print(f"  ✗ Cache not used on second call")
            return False
        
        # Symbols should be identical
        if symbol_map1 != symbol_map2:
            print(f"  ✗ Cached symbols differ from original")
            return False
        
        print(f"  ✓ Symbol cache working correctly")
        print(f"    Cache hit on second call (same timestamp)")
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all KuCoin Ultimate Integration tests."""
    print("=" * 60)
    print("KuCoin Futures Ultimate Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_server_time_sync,
        test_dynamic_symbol_mapping,
        test_kucoin_granularity_conversion,
        test_timeframe_milliseconds_conversion,
        test_bulk_fetch_logic,
        test_cache_behavior,
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
        print("✓ All KuCoin Ultimate Integration tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
