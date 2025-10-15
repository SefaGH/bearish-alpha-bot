#!/usr/bin/env python3
"""
Test Market Data Pipeline Core Foundation.

Tests for Phase 2.2 market data pipeline implementation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.market_data_pipeline import MarketDataPipeline
from core.ccxt_client import CcxtClient
import pandas as pd


class MockCcxtClient:
    """Mock CcxtClient for testing."""
    
    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.fetch_count = 0
    
    def validate_and_get_symbol(self, symbol: str) -> str:
        """Mock symbol validation."""
        return symbol
    
    def ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        """Mock OHLCV fetch."""
        self.fetch_count += 1
        
        if self.should_fail:
            raise Exception(f"Mock error from {self.name}")
        
        # Generate mock OHLCV data
        import time
        current_time = int(time.time() * 1000)
        
        data = []
        for i in range(min(limit, 100)):
            timestamp = current_time - (i * 30 * 60 * 1000)  # 30min intervals
            data.append([
                timestamp,
                100.0 + i,  # open
                101.0 + i,  # high
                99.0 + i,   # low
                100.5 + i,  # close
                1000.0      # volume
            ])
        
        return list(reversed(data))
    
    def fetch_ohlcv_bulk(self, symbol: str, timeframe: str, target_limit: int):
        """Mock bulk OHLCV fetch."""
        return self.ohlcv(symbol, timeframe, target_limit)


def test_pipeline_initialization():
    """Test MarketDataPipeline initialization."""
    print("\n" + "="*60)
    print("TEST: Pipeline Initialization")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
            'mock2': MockCcxtClient('mock2')
        }
        
        pipeline = MarketDataPipeline(exchanges)
        
        assert pipeline.exchanges == exchanges, "Exchanges not stored correctly"
        assert pipeline.total_requests == 0, "Initial requests should be 0"
        assert pipeline.failed_requests == 0, "Initial failures should be 0"
        assert pipeline.is_running == False, "Pipeline should not be running initially"
        
        print("✅ PASS: Pipeline initialized correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_start_feeds():
    """Test starting data feeds."""
    print("\n" + "="*60)
    print("TEST: Start Data Feeds")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        
        # Start feeds for multiple symbols and timeframes
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        timeframes = ['30m', '1h']
        
        results = pipeline.start_feeds(symbols, timeframes)
        
        assert results['successful_fetches'] > 0, "Should have successful fetches"
        assert 'mock1' in results['exchanges_used'], "Mock exchange should be used"
        assert pipeline.is_running == True, "Pipeline should be running after start"
        
        print(f"✅ PASS: Started feeds successfully")
        print(f"   - Successful fetches: {results['successful_fetches']}")
        print(f"   - Failed fetches: {results['failed_fetches']}")
        print(f"   - Exchanges used: {results['exchanges_used']}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_latest_ohlcv():
    """Test getting latest OHLCV data."""
    print("\n" + "="*60)
    print("TEST: Get Latest OHLCV")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        
        # Start feeds first
        pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
        
        # Get data
        df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
        
        assert df is not None, "Should return DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        assert 'open' in df.columns, "Should have 'open' column"
        assert 'close' in df.columns, "Should have 'close' column"
        assert 'rsi' in df.columns, "Should have 'rsi' indicator"
        assert 'ema21' in df.columns, "Should have 'ema21' indicator"
        
        print(f"✅ PASS: Retrieved OHLCV data successfully")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_buffer_limits():
    """Test circular buffer memory management."""
    print("\n" + "="*60)
    print("TEST: Buffer Limits")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        
        # Verify buffer limits are configured
        assert '30m' in pipeline.BUFFER_LIMITS, "Should have 30m buffer limit"
        assert '1h' in pipeline.BUFFER_LIMITS, "Should have 1h buffer limit"
        assert '4h' in pipeline.BUFFER_LIMITS, "Should have 4h buffer limit"
        assert '1d' in pipeline.BUFFER_LIMITS, "Should have 1d buffer limit"
        
        assert pipeline.BUFFER_LIMITS['30m'] == 1000, "30m buffer should be 1000"
        assert pipeline.BUFFER_LIMITS['1h'] == 500, "1h buffer should be 500"
        assert pipeline.BUFFER_LIMITS['4h'] == 200, "4h buffer should be 200"
        assert pipeline.BUFFER_LIMITS['1d'] == 100, "1d buffer should be 100"
        
        print("✅ PASS: Buffer limits configured correctly")
        print(f"   - Limits: {pipeline.BUFFER_LIMITS}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_check():
    """Test health check functionality."""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        
        # Initial health check
        health = pipeline.health_check()
        
        assert 'status' in health, "Should have status"
        assert 'uptime_seconds' in health, "Should have uptime"
        assert 'total_requests' in health, "Should have total_requests"
        assert 'failed_requests' in health, "Should have failed_requests"
        assert 'error_rate' in health, "Should have error_rate"
        assert 'active_streams' in health, "Should have active_streams"
        
        assert health['status'] == 'healthy', "Initial status should be healthy"
        assert health['total_requests'] == 0, "Initial requests should be 0"
        
        print("✅ PASS: Health check working correctly")
        print(f"   - Status: {health['status']}")
        print(f"   - Error rate: {health['error_rate']}%")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_status():
    """Test detailed pipeline status."""
    print("\n" + "="*60)
    print("TEST: Pipeline Status")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
            'mock2': MockCcxtClient('mock2'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        
        # Start some feeds
        pipeline.start_feeds(['BTC/USDT:USDT'], ['30m', '1h'])
        
        # Get status
        status = pipeline.get_pipeline_status()
        
        assert 'exchanges' in status, "Should have exchanges breakdown"
        assert 'memory_estimate_mb' in status, "Should have memory estimate"
        assert 'data_freshness' in status, "Should have freshness metrics"
        assert 'buffer_limits' in status, "Should have buffer limits"
        
        assert status['active_streams'] > 0, "Should have active streams"
        
        print("✅ PASS: Pipeline status working correctly")
        print(f"   - Active streams: {status['active_streams']}")
        print(f"   - Memory estimate: {status['memory_estimate_mb']} MB")
        print(f"   - Exchanges: {list(status['exchanges'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling with retry logic."""
    print("\n" + "="*60)
    print("TEST: Error Handling and Retry Logic")
    print("="*60)
    
    try:
        # First exchange fails, second succeeds
        exchanges = {
            'mock_fail': MockCcxtClient('mock_fail', should_fail=True),
            'mock_success': MockCcxtClient('mock_success', should_fail=False),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        
        # This should try mock_fail (and fail), then try mock_success (and succeed)
        results = pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
        
        assert pipeline.failed_requests > 0, "Should have some failed requests"
        assert results['successful_fetches'] > 0, "Should eventually succeed with fallback"
        
        # Check that data was stored from successful exchange
        df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
        assert df is not None, "Should have data from successful exchange"
        
        print("✅ PASS: Error handling working correctly")
        print(f"   - Failed requests: {pipeline.failed_requests}")
        print(f"   - Successful fetches: {results['successful_fetches']}")
        print(f"   - Fallback worked: data retrieved from backup exchange")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_best_data_source():
    """Test best data source selection."""
    print("\n" + "="*60)
    print("TEST: Best Data Source Selection")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
            'mock2': MockCcxtClient('mock2'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        
        # Both exchanges should have data
        pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
        
        # Get data without specifying exchange (should pick best)
        df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
        
        assert df is not None, "Should return data from best source"
        assert not df.empty, "Data should not be empty"
        
        # Get data from specific exchange
        df_specific = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='mock1')
        
        assert df_specific is not None, "Should return data from specific exchange"
        
        print("✅ PASS: Best data source selection working")
        print(f"   - Auto-selected best source: {len(df)} rows")
        print(f"   - Specific exchange fetch: {len(df_specific)} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shutdown():
    """Test pipeline shutdown."""
    print("\n" + "="*60)
    print("TEST: Pipeline Shutdown")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
        
        assert pipeline.is_running == True, "Should be running"
        
        pipeline.shutdown()
        
        assert pipeline.is_running == False, "Should not be running after shutdown"
        
        print("✅ PASS: Pipeline shutdown correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all market data pipeline tests."""
    print("\n" + "="*70)
    print("MARKET DATA PIPELINE TEST SUITE")
    print("Phase 2.2: Core Foundation")
    print("="*70)
    
    tests = [
        test_pipeline_initialization,
        test_start_feeds,
        test_get_latest_ohlcv,
        test_buffer_limits,
        test_health_check,
        test_pipeline_status,
        test_error_handling,
        test_best_data_source,
        test_shutdown,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed}/{total} PASSED")
    print("="*70)
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
