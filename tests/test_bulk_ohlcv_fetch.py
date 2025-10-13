#!/usr/bin/env python3
"""
Tests for bulk OHLCV fetching functionality.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO)

from core.ccxt_client import CcxtClient


class MockExchange:
    """Mock exchange for testing."""
    def __init__(self):
        self.sandbox = False
        self.urls = {'api': 'https://mock.api.example.com'}
        self.call_count = 0
        
    def load_markets(self):
        return {
            'BTC/USDT:USDT': {'id': 'BTCUSDT', 'symbol': 'BTC/USDT:USDT'},
        }
    
    def fetch_ohlcv(self, symbol, timeframe='30m', limit=100):
        """Return mock OHLCV data."""
        import time
        self.call_count += 1
        now = int(time.time() * 1000)
        data = []
        for i in range(min(limit, 200)):  # Simulate 200 candle limit
            timestamp = now - (limit - i) * 30 * 60 * 1000  # 30m candles
            base_price = 50000 + i * 10
            data.append([
                timestamp,
                base_price,      # open
                base_price + 100, # high
                base_price - 100, # low
                base_price + 50,  # close
                1000.0           # volume
            ])
        return data


class MockExchangeWithError:
    """Mock exchange that fails on certain calls."""
    def __init__(self, fail_on_batch=None):
        self.sandbox = False
        self.urls = {'api': 'https://mock.api.example.com'}
        self.call_count = 0
        self.fail_on_batch = fail_on_batch
        
    def load_markets(self):
        return {
            'BTC/USDT:USDT': {'id': 'BTCUSDT', 'symbol': 'BTC/USDT:USDT'},
        }
    
    def fetch_ohlcv(self, symbol, timeframe='30m', limit=100):
        """Return mock OHLCV data or raise error."""
        import time
        self.call_count += 1
        
        # Fail on specific batch if configured
        # Note: Each batch will retry 3 times, so fail_on_batch=1 means fail calls 1,2,3
        if self.fail_on_batch is not None:
            batch_num = ((self.call_count - 1) // 3) + 1  # Calculate which batch we're in
            if batch_num == self.fail_on_batch:
                raise RuntimeError("Mock API error")
        
        now = int(time.time() * 1000)
        data = []
        for i in range(min(limit, 200)):  # Simulate 200 candle limit
            timestamp = now - (limit - i) * 30 * 60 * 1000
            base_price = 50000 + i * 10
            data.append([
                timestamp,
                base_price,
                base_price + 100,
                base_price - 100,
                base_price + 50,
                1000.0
            ])
        return data


def test_bulk_fetch_single_batch():
    """Test bulk fetch with target_limit <= 200 (single API call)."""
    print("\n=== Test: Single batch (<=200) ===")
    
    # Create mock client
    client = CcxtClient.__new__(CcxtClient)
    client.name = 'mock'
    client.ex = MockExchange()
    
    # Test with limit=100 (should use single call)
    result = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', target_limit=100)
    
    assert len(result) == 100, f"Expected 100 candles, got {len(result)}"
    assert client.ex.call_count == 1, f"Expected 1 API call, got {client.ex.call_count}"
    print(f"✓ Single batch test passed: {len(result)} candles fetched")
    return True


def test_bulk_fetch_multiple_batches():
    """Test bulk fetch with target_limit > 200 (multiple API calls)."""
    print("\n=== Test: Multiple batches (>200) ===")
    
    # Create mock client
    client = CcxtClient.__new__(CcxtClient)
    client.name = 'mock'
    client.ex = MockExchange()
    
    # Test with limit=500 (should use 3 batches: 200 + 200 + 100)
    result = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', target_limit=500)
    
    # Each batch makes 1 successful call (mock doesn't fail)
    expected_calls = 3  # 3 batches * 1 call each
    assert len(result) == 500, f"Expected 500 candles, got {len(result)}"
    assert client.ex.call_count == expected_calls, f"Expected {expected_calls} API calls, got {client.ex.call_count}"
    print(f"✓ Multiple batch test passed: {len(result)} candles fetched with {client.ex.call_count} API calls")
    return True


def test_bulk_fetch_max_limit():
    """Test bulk fetch respects max 5 batches (1000 candles)."""
    print("\n=== Test: Max limit (1000 candles) ===")
    
    # Create mock client
    client = CcxtClient.__new__(CcxtClient)
    client.name = 'mock'
    client.ex = MockExchange()
    
    # Test with limit=1000 (should use 5 batches: 200 * 5)
    result = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', target_limit=1000)
    
    # 5 batches * 1 call each
    expected_calls = 5
    assert len(result) == 1000, f"Expected 1000 candles, got {len(result)}"
    assert client.ex.call_count == expected_calls, f"Expected {expected_calls} API calls, got {client.ex.call_count}"
    print(f"✓ Max limit test passed: {len(result)} candles fetched with {client.ex.call_count} API calls")
    return True


def test_bulk_fetch_exceeds_max():
    """Test bulk fetch caps at 1000 candles even if more requested."""
    print("\n=== Test: Request exceeds max (>1000) ===")
    
    # Create mock client
    client = CcxtClient.__new__(CcxtClient)
    client.name = 'mock'
    client.ex = MockExchange()
    
    # Test with limit=1500 (should cap at 1000: 5 batches max)
    result = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', target_limit=1500)
    
    assert len(result) == 1000, f"Expected 1000 candles (capped), got {len(result)}"
    print(f"✓ Exceeds max test passed: requested 1500, got {len(result)} (properly capped)")
    return True


def test_bulk_fetch_error_on_first_batch():
    """Test bulk fetch raises error if first batch fails."""
    print("\n=== Test: Error on first batch ===")
    
    # Create mock client that fails on first call
    client = CcxtClient.__new__(CcxtClient)
    client.name = 'mock'
    client.ex = MockExchangeWithError(fail_on_batch=1)  # Fails on 1st call
    
    # Should raise error since first batch fails
    try:
        result = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', target_limit=500)
        assert False, "Expected exception to be raised"
    except RuntimeError as e:
        assert "Mock API error" in str(e), f"Expected 'Mock API error', got {e}"
        print(f"✓ Error on first batch test passed: properly raised exception")
        return True


def test_bulk_fetch_error_on_later_batch():
    """Test bulk fetch uses partial data if later batch returns empty."""
    print("\n=== Test: Empty data on later batch (partial data) ===")
    
    # Create a special mock that returns empty on 2nd batch
    class MockEmptyOnSecond:
        def __init__(self):
            self.sandbox = False
            self.urls = {'api': 'https://mock.api.example.com'}
            self.call_count = 0
            self.batch_count = 0
        def load_markets(self):
            return {'BTC/USDT:USDT': {'id': 'BTCUSDT', 'symbol': 'BTC/USDT:USDT'}}
        def fetch_ohlcv(self, symbol, timeframe='30m', limit=100):
            import time
            self.call_count += 1
            # Track batches by detecting when limit changes or when we successfully return data
            # For simplicity, we'll return empty after the first successful batch
            if self.batch_count >= 1:  # Return empty for 2nd batch and beyond
                return []
            now = int(time.time() * 1000)
            data = []
            for i in range(min(limit, 200)):
                timestamp = now - (limit - i) * 30 * 60 * 1000
                base_price = 50000 + i * 10
                data.append([timestamp, base_price, base_price + 100, base_price - 100, base_price + 50, 1000.0])
            self.batch_count += 1  # Increment after successful return
            return data
    
    client = CcxtClient.__new__(CcxtClient)
    client.name = 'mock'
    client.ex = MockEmptyOnSecond()
    
    # Should return partial data from first successful batch
    result = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', target_limit=500)
    
    # Should have first batch only (200 candles)
    assert len(result) == 200, f"Expected 200 candles (first batch), got {len(result)}"
    print(f"✓ Empty data on later batch test passed: returned {len(result)} candles from successful batch")
    return True


def test_bulk_fetch_backward_compatible():
    """Test that bulk fetch is backward compatible with single batch requests."""
    print("\n=== Test: Backward compatibility ===")
    
    # Create mock client
    client = CcxtClient.__new__(CcxtClient)
    client.name = 'mock'
    client.ex = MockExchange()
    
    # Test with various limits <= 200
    for limit in [50, 100, 150, 200]:
        client.ex.call_count = 0
        result = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', target_limit=limit)
        assert len(result) == limit, f"Expected {limit} candles, got {len(result)}"
    
    print(f"✓ Backward compatibility test passed: all single-batch requests work correctly")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("BULK OHLCV FETCH TESTS")
    print("=" * 60)
    
    tests = [
        test_bulk_fetch_single_batch,
        test_bulk_fetch_multiple_batches,
        test_bulk_fetch_max_limit,
        test_bulk_fetch_exceeds_max,
        test_bulk_fetch_error_on_first_batch,
        test_bulk_fetch_error_on_later_batch,
        test_bulk_fetch_backward_compatible,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed.append(test.__name__)
    
    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed!")
        sys.exit(0)
