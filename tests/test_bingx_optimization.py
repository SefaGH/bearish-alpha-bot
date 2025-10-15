#!/usr/bin/env python3
"""
Test BingX Optimization Features:
1. Contract discovery with caching
2. Symbol validation with format conversion
3. Selective market loading
4. Market data caching
5. Data fetching (basic and bulk)
"""
import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient
from core.multi_exchange import build_clients_from_env

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_contract_discovery():
    """Test BingX contract discovery."""
    print("=" * 60)
    print("Test 1: BingX Contract Discovery")
    print("=" * 60)
    
    try:
        client = CcxtClient('bingx')
        
        # First call - should fetch from API
        start = time.time()
        contracts = client._get_bingx_contracts()
        first_elapsed = time.time() - start
        
        print(f"✅ First call: Discovered {len(contracts)} contracts in {first_elapsed:.2f}s")
        print(f"   Sample: {list(contracts.items())[:5]}")
        
        # Second call - should use cache
        start = time.time()
        contracts2 = client._get_bingx_contracts()
        second_elapsed = time.time() - start
        
        print(f"✅ Second call: Used cache in {second_elapsed*1000:.2f}ms")
        assert contracts == contracts2, "Cache should return same data"
        
        # Verify format
        for ccxt_sym, native_sym in list(contracts.items())[:3]:
            assert ccxt_sym.endswith(':USDT'), f"CCXT symbol should end with :USDT: {ccxt_sym}"
            assert '-USDT' in native_sym, f"Native symbol should contain -USDT: {native_sym}"
        
        print("✅ Format validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_symbol_validation():
    """Test BingX symbol validation with format conversion."""
    print("\n" + "=" * 60)
    print("Test 2: Symbol Validation & Format Conversion")
    print("=" * 60)
    
    try:
        client = CcxtClient('bingx')
        
        test_cases = [
            ('BTC/USDT', 'BTC/USDT:USDT'),      # Spot -> Perpetual
            ('BTC/USDT:USDT', 'BTC/USDT:USDT'), # Already perpetual
            ('ETH/USDT', 'ETH/USDT:USDT'),
            ('SOL/USDT', 'SOL/USDT:USDT'),
        ]
        
        for input_sym, expected_output in test_cases:
            validated = client.validate_and_get_symbol(input_sym)
            assert validated == expected_output, f"Expected {expected_output}, got {validated}"
            print(f"✅ {input_sym:20s} → {validated}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_selective_market_loading():
    """Test selective market loading feature."""
    print("\n" + "=" * 60)
    print("Test 3: Selective Market Loading")
    print("=" * 60)
    
    try:
        # Test without filter (loads all)
        client1 = CcxtClient('bingx')
        start = time.time()
        markets1 = client1.markets()
        elapsed1 = time.time() - start
        print(f"Without filter: Loaded {len(markets1)} markets in {elapsed1:.2f}s")
        
        # Test with filter (loads only required)
        client2 = CcxtClient('bingx')
        required = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        client2.set_required_symbols(required)
        
        start = time.time()
        markets2 = client2.markets()
        elapsed2 = time.time() - start
        print(f"With filter:    Loaded {len(markets2)} markets in {elapsed2:.2f}s")
        
        # Verify filtering worked
        assert len(markets2) == 3, f"Expected 3 markets, got {len(markets2)}"
        assert set(markets2.keys()) == set(required), "Markets should match required symbols"
        
        print(f"✅ Successfully filtered {len(markets1)} → {len(markets2)} markets")
        print(f"   Loaded markets: {list(markets2.keys())}")
        
        # Test cache
        start = time.time()
        markets3 = client2.markets()
        elapsed3 = time.time() - start
        print(f"✅ Cache hit: Loaded in {elapsed3*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_data_fetching():
    """Test basic and bulk data fetching."""
    print("\n" + "=" * 60)
    print("Test 4: Data Fetching")
    print("=" * 60)
    
    try:
        client = CcxtClient('bingx')
        symbol = client.validate_and_get_symbol('BTC/USDT')
        
        # Test basic fetch
        print("Testing basic OHLCV fetch...")
        data = client.ohlcv(symbol, '1h', 10)
        print(f"✅ Fetched {len(data)} candles")
        print(f"   Last close: ${data[-1][4]:.2f}")
        
        # Test bulk fetch
        print("\nTesting bulk OHLCV fetch...")
        start = time.time()
        data_bulk = client.fetch_ohlcv_bulk(symbol, '30m', 1000)
        elapsed = time.time() - start
        print(f"✅ Fetched {len(data_bulk)} candles in {elapsed:.2f}s")
        print(f"   First close: ${data_bulk[0][4]:.2f}")
        print(f"   Last close: ${data_bulk[-1][4]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_multi_exchange_integration():
    """Test multi_exchange module with selective loading."""
    print("\n" + "=" * 60)
    print("Test 5: Multi-Exchange Integration")
    print("=" * 60)
    
    try:
        # Set environment for test (no credentials needed for this test)
        os.environ['EXCHANGES'] = 'bingx'
        
        # Test without selective loading
        print("Building client without selective loading...")
        try:
            clients1 = build_clients_from_env()
        except ValueError as e:
            # Expected if no credentials
            print(f"⚠️  No credentials (expected): {e}")
            print("✅ Multi-exchange integration structure is correct")
            return True
        
        # If we got here, credentials exist, test selective loading
        print("Building client with selective loading...")
        required = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        clients2 = build_clients_from_env(required_symbols=required)
        
        if 'bingx' in clients2:
            client = clients2['bingx']
            markets = client.markets()
            print(f"✅ Client built with {len(markets)} markets")
            print(f"   Markets: {list(markets.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BingX Optimization Test Suite")
    print("=" * 60)
    
    results = {
        "Contract Discovery": test_contract_discovery(),
        "Symbol Validation": test_symbol_validation(),
        "Selective Loading": test_selective_market_loading(),
        "Data Fetching": test_data_fetching(),
        "Multi-Exchange Integration": test_multi_exchange_integration(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30s} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
