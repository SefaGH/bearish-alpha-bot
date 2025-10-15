#!/usr/bin/env python3
"""
Test Market Data Pipeline Core Foundation.

Tests for Phase 2.2 market data pipeline implementation.
Uses pytest and unittest.mock for comprehensive testing.

Tests cover:
- Pipeline initialization with various configurations
- Data feed management and retrieval
- Health monitoring and status tracking
- Error handling and resilience
- Buffer management and memory optimization
- Integration with CcxtClient
"""

import sys
import os
from unittest.mock import Mock, patch
import pytest
import pandas as pd
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.market_data_pipeline import MarketDataPipeline
from core.ccxt_client import CcxtClient


# Fixed timestamp for deterministic tests (recent enough to pass freshness checks)
FIXED_TIMESTAMP = int(time.time() * 1000) - (5 * 60 * 1000)  # 5 minutes ago


@pytest.fixture
def current_timestamp():
    """Provide a fixed timestamp for deterministic tests."""
    return FIXED_TIMESTAMP


@pytest.fixture
def generate_ohlcv_data():
    """Factory fixture to generate consistent OHLCV test data."""
    def _generate(count=100, interval_minutes=30, base_price=50000):
        data = []
        for i in range(count):
            timestamp = FIXED_TIMESTAMP - (i * interval_minutes * 60 * 1000)
            data.append([
                timestamp,
                base_price + i * 10,      # open
                base_price + i * 10 + 100,  # high
                base_price + i * 10 - 100,  # low
                base_price + i * 10 + 50,   # close
                1000 + i                   # volume
            ])
        return list(reversed(data))
    return _generate


@pytest.fixture
def mock_ccxt_client(generate_ohlcv_data):
    """Create a mock CcxtClient for testing."""
    client = Mock(spec=CcxtClient)
    client.name = 'test_exchange'
    client.validate_and_get_symbol.return_value = 'BTC/USDT:USDT'
    
    # Use fixture to generate consistent test data
    sample_data = generate_ohlcv_data()
    
    client.ohlcv.return_value = sample_data
    client.fetch_ohlcv_bulk.return_value = sample_data
    
    return client


@pytest.fixture
def mock_failing_client():
    """Create a mock CcxtClient that fails."""
    client = Mock(spec=CcxtClient)
    client.name = 'failing_exchange'
    client.validate_and_get_symbol.side_effect = Exception("Mock error")
    client.ohlcv.side_effect = Exception("Mock error")
    
    return client


@pytest.fixture
def pipeline_with_mock(mock_ccxt_client):
    """Create a pipeline with mock clients."""
    exchanges = {
        'mock1': mock_ccxt_client,
    }
    return MarketDataPipeline(exchanges)


@pytest.fixture
def pipeline_multi_exchange(mock_ccxt_client, generate_ohlcv_data):
    """Create a pipeline with multiple mock exchanges."""
    # Create second mock client
    client2 = Mock(spec=CcxtClient)
    client2.name = 'test_exchange_2'
    client2.validate_and_get_symbol.return_value = 'BTC/USDT:USDT'
    
    # Different count to differentiate
    sample_data = generate_ohlcv_data(count=90)
    
    client2.ohlcv.return_value = sample_data
    client2.fetch_ohlcv_bulk.return_value = sample_data
    
    exchanges = {
        'mock1': mock_ccxt_client,
        'mock2': client2,
    }
    return MarketDataPipeline(exchanges)


def test_pipeline_initialization(mock_ccxt_client):
    """Test MarketDataPipeline initialization with different configurations."""
    # Test basic initialization
    exchanges = {'mock1': mock_ccxt_client}
    pipeline = MarketDataPipeline(exchanges)
    
    assert pipeline.exchanges == exchanges
    assert pipeline.total_requests == 0
    assert pipeline.failed_requests == 0
    assert pipeline.is_running == False
    
    # Test initialization with config
    config = {'indicators': {'rsi_period': 14}}
    pipeline_with_config = MarketDataPipeline(exchanges, config=config)
    assert pipeline_with_config.config == config
    
    # Test initialization with multiple exchanges
    client2 = Mock(spec=CcxtClient)
    client2.name = 'exchange2'
    multi_exchanges = {'mock1': mock_ccxt_client, 'mock2': client2}
    pipeline_multi = MarketDataPipeline(multi_exchanges)
    assert len(pipeline_multi.exchanges) == 2


def test_start_feeds(pipeline_with_mock, mock_ccxt_client):
    """Test starting data feeds with mock CcxtClient responses."""
    # Test single symbol and timeframe
    results = pipeline_with_mock.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    assert results['successful_fetches'] > 0
    assert 'mock1' in results['exchanges_used']
    assert pipeline_with_mock.is_running == True
    assert mock_ccxt_client.ohlcv.called or mock_ccxt_client.fetch_ohlcv_bulk.called
    
    # Test multiple symbols and timeframes
    results_multi = pipeline_with_mock.start_feeds(
        ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
        ['30m', '1h']
    )
    
    assert results_multi['successful_fetches'] >= 4
    assert results_multi['failed_fetches'] == 0


def test_get_latest_ohlcv_specific_exchange(pipeline_with_mock):
    """Test getting latest OHLCV data from specific exchange."""
    pipeline_with_mock.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    # Get data from specific exchange
    df = pipeline_with_mock.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='mock1')
    
    assert df is not None
    assert not df.empty
    assert 'open' in df.columns
    assert 'close' in df.columns
    assert 'rsi' in df.columns
    assert 'ema21' in df.columns
    assert len(df) > 0


def test_get_latest_ohlcv_auto_selection(pipeline_multi_exchange):
    """Test getting latest OHLCV data with auto-selection of best source."""
    pipeline_multi_exchange.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    # Get data without specifying exchange (auto-selection)
    df = pipeline_multi_exchange.get_latest_ohlcv('BTC/USDT:USDT', '30m')
    
    assert df is not None
    assert not df.empty
    assert 'open' in df.columns
    assert 'close' in df.columns


def test_buffer_limits(pipeline_with_mock):
    """Test circular buffer memory management and enforcement."""
    # Verify buffer limits are configured
    assert '30m' in pipeline_with_mock.BUFFER_LIMITS
    assert '1h' in pipeline_with_mock.BUFFER_LIMITS
    assert '4h' in pipeline_with_mock.BUFFER_LIMITS
    assert '1d' in pipeline_with_mock.BUFFER_LIMITS
    
    assert pipeline_with_mock.BUFFER_LIMITS['30m'] == 1000
    assert pipeline_with_mock.BUFFER_LIMITS['1h'] == 500
    assert pipeline_with_mock.BUFFER_LIMITS['4h'] == 200
    assert pipeline_with_mock.BUFFER_LIMITS['1d'] == 100
    
    # Test buffer limit enforcement
    pipeline_with_mock.start_feeds(['BTC/USDT:USDT'], ['1h'])
    df = pipeline_with_mock.get_latest_ohlcv('BTC/USDT:USDT', '1h')
    
    # Verify data doesn't exceed buffer limit
    if df is not None:
        assert len(df) <= pipeline_with_mock.BUFFER_LIMITS['1h']


def test_health_check_accuracy(pipeline_with_mock, mock_failing_client):
    """Test health check accuracy with various data states."""
    # Test initial healthy state
    health = pipeline_with_mock.health_check()
    
    assert 'status' in health
    assert 'uptime_seconds' in health
    assert 'total_requests' in health
    assert 'failed_requests' in health
    assert 'error_rate' in health
    assert 'active_streams' in health
    
    assert health['status'] == 'healthy'
    assert health['total_requests'] == 0
    assert health['error_rate'] == 0
    
    # Test after successful fetches
    pipeline_with_mock.start_feeds(['BTC/USDT:USDT'], ['30m'])
    health_after = pipeline_with_mock.health_check()
    assert health_after['status'] == 'healthy'
    assert health_after['total_requests'] > 0
    
    # Test degraded state with some failures
    pipeline_with_failures = MarketDataPipeline({
        'failing': mock_failing_client,
        'mock1': pipeline_with_mock.exchanges['mock1']
    })
    pipeline_with_failures.start_feeds(['BTC/USDT:USDT'], ['30m'])
    health_degraded = pipeline_with_failures.health_check()
    # Should have some failed requests from the failing client
    assert health_degraded['failed_requests'] > 0


def test_pipeline_status_detailed_metrics(pipeline_multi_exchange):
    """Test get_pipeline_status() detailed metrics validation."""
    # Start feeds to generate data
    pipeline_multi_exchange.start_feeds(['BTC/USDT:USDT'], ['30m', '1h'])
    
    # Get detailed status
    status = pipeline_multi_exchange.get_pipeline_status()
    
    # Validate all required fields
    assert 'exchanges' in status
    assert 'memory_estimate_mb' in status
    assert 'data_freshness' in status
    assert 'buffer_limits' in status
    assert 'active_streams' in status
    assert 'total_requests' in status
    assert 'failed_requests' in status
    
    # Validate exchanges breakdown
    assert isinstance(status['exchanges'], dict)
    for exchange_name, exchange_data in status['exchanges'].items():
        assert 'streams' in exchange_data
        assert 'symbols' in exchange_data
    
    # Validate data freshness
    assert 'fresh' in status['data_freshness']
    assert 'stale' in status['data_freshness']
    assert 'expired' in status['data_freshness']
    
    # Validate memory estimate is reasonable
    assert status['memory_estimate_mb'] >= 0
    assert status['memory_estimate_mb'] < 1000  # Should be reasonable for test data
    
    # Validate active streams
    assert status['active_streams'] > 0


def test_error_handling_resilience(mock_ccxt_client, mock_failing_client):
    """Test error handling resilience with retry logic and fallback."""
    # First exchange fails, second succeeds
    exchanges = {
        'failing': mock_failing_client,
        'success': mock_ccxt_client,
    }
    
    pipeline = MarketDataPipeline(exchanges)
    
    # This should try failing (and fail with retries), then try success (and succeed)
    results = pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    assert pipeline.failed_requests > 0  # Should have failed attempts
    assert results['successful_fetches'] > 0  # Should eventually succeed with fallback
    
    # Check that data was stored from successful exchange
    df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
    assert df is not None
    assert not df.empty


def test_store_data_circular_buffer(pipeline_with_mock, mock_ccxt_client, generate_ohlcv_data):
    """Test _store_data() circular buffer management."""
    # Create a DataFrame larger than buffer limit
    large_data = generate_ohlcv_data(count=2000, interval_minutes=60, base_price=50000)
    
    # Mock the client to return large dataset
    mock_ccxt_client.ohlcv.return_value = large_data
    
    # Start feeds for 1h timeframe (buffer limit = 500)
    pipeline_with_mock.start_feeds(['BTC/USDT:USDT'], ['1h'])
    
    # Get stored data
    df = pipeline_with_mock.get_latest_ohlcv('BTC/USDT:USDT', '1h')
    
    # Verify data respects buffer limit
    assert len(df) <= pipeline_with_mock.BUFFER_LIMITS['1h']


def test_get_best_data_source_algorithm(pipeline_multi_exchange):
    """Test _get_best_data_source() selection algorithm."""
    # Start feeds to populate data (fallback means only first successful exchange stores data)
    pipeline_multi_exchange.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    # Test auto-selection of best source
    df = pipeline_multi_exchange.get_latest_ohlcv('BTC/USDT:USDT', '30m')
    
    assert df is not None
    assert not df.empty
    
    # Test specific exchange selection (only mock1 has data due to fallback)
    df_mock1 = pipeline_multi_exchange.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='mock1')
    
    assert df_mock1 is not None
    assert not df_mock1.empty
    
    # mock2 should not have data due to fallback mechanism
    df_mock2 = pipeline_multi_exchange.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='mock2')
    assert df_mock2 is None  # No data stored in mock2 due to fallback


def test_graceful_shutdown(pipeline_with_mock):
    """Test graceful shutdown() behavior."""
    pipeline_with_mock.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    assert pipeline_with_mock.is_running == True
    
    # Test shutdown
    pipeline_with_mock.shutdown()
    
    assert pipeline_with_mock.is_running == False
    
    # Test idempotent shutdown (should not raise error)
    pipeline_with_mock.shutdown()
    assert pipeline_with_mock.is_running == False


def test_integration_pipeline_ccxt_compatibility(mock_ccxt_client):
    """Test Pipeline + existing CcxtClient compatibility."""
    # Verify mock client has all expected methods
    assert hasattr(mock_ccxt_client, 'name')
    assert hasattr(mock_ccxt_client, 'validate_and_get_symbol')
    assert hasattr(mock_ccxt_client, 'ohlcv')
    
    # Test pipeline initialization with mock client
    exchanges = {'test': mock_ccxt_client}
    pipeline = MarketDataPipeline(exchanges)
    
    # Test that pipeline can work with the client
    results = pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
    assert results['successful_fetches'] > 0
    
    # Verify client methods were called correctly
    assert mock_ccxt_client.validate_and_get_symbol.called
    assert mock_ccxt_client.ohlcv.called or mock_ccxt_client.fetch_ohlcv_bulk.called


def test_real_exchange_symbol_validation():
    """Test real exchange symbol validation (with mocks)."""
    client = Mock(spec=CcxtClient)
    client.name = 'binance'
    
    # Mock successful validation
    client.validate_and_get_symbol.return_value = 'BTC/USDT:USDT'
    
    result = client.validate_and_get_symbol('BTC/USDT:USDT')
    assert result == 'BTC/USDT:USDT'
    
    # Mock validation failure
    client.validate_and_get_symbol.side_effect = ValueError("Invalid symbol")
    
    with pytest.raises(ValueError):
        client.validate_and_get_symbol('INVALID')


def test_memory_usage_estimation(pipeline_multi_exchange):
    """Test memory usage estimation accuracy."""
    pipeline_multi_exchange.start_feeds(['BTC/USDT:USDT', 'ETH/USDT:USDT'], ['30m', '1h'])
    
    status = pipeline_multi_exchange.get_pipeline_status()
    
    # Verify memory estimate exists and is reasonable
    assert 'memory_estimate_mb' in status
    assert status['memory_estimate_mb'] > 0
    assert status['memory_estimate_mb'] < 100  # Should be reasonable for test data
    
    # Verify exchange tracking exists
    assert 'exchanges' in status
    for exchange_name, exchange_data in status['exchanges'].items():
        assert 'streams' in exchange_data
        assert 'symbols' in exchange_data
        assert exchange_data['streams'] >= 0
        assert exchange_data['symbols'] >= 0


def test_error_handling_all_exchanges_fail(mock_failing_client):
    """Test error handling when all exchanges fail."""
    exchanges = {
        'failing1': mock_failing_client,
    }
    
    pipeline = MarketDataPipeline(exchanges)
    
    # Should handle gracefully when all exchanges fail
    results = pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    assert results['failed_fetches'] > 0
    assert results['successful_fetches'] == 0
    assert len(results['errors']) > 0


def test_buffer_limit_enforcement_multiple_timeframes(pipeline_with_mock):
    """Test buffer limit enforcement across multiple timeframes."""
    pipeline_with_mock.start_feeds(['BTC/USDT:USDT'], ['30m', '1h', '4h'])
    
    # Check each timeframe respects its buffer limit
    for timeframe in ['30m', '1h', '4h']:
        df = pipeline_with_mock.get_latest_ohlcv('BTC/USDT:USDT', timeframe)
        if df is not None and not df.empty:
            assert len(df) <= pipeline_with_mock.BUFFER_LIMITS[timeframe]


@pytest.mark.asyncio
async def test_start_feeds_async(pipeline_with_mock, mock_ccxt_client):
    """Test async version of start_feeds."""
    # Test single symbol and timeframe
    results = await pipeline_with_mock.start_feeds_async(['BTC/USDT:USDT'], ['30m'])
    
    assert results['successful_fetches'] > 0
    assert 'mock1' in results['exchanges_used']
    assert pipeline_with_mock.is_running == True
    assert mock_ccxt_client.ohlcv.called or mock_ccxt_client.fetch_ohlcv_bulk.called
    
    # Test multiple symbols and timeframes
    results_multi = await pipeline_with_mock.start_feeds_async(
        ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
        ['30m', '1h']
    )
    
    assert results_multi['successful_fetches'] >= 4
    assert results_multi['failed_fetches'] == 0


def test_get_health_status(pipeline_with_mock):
    """Test get_health_status method."""
    pipeline_with_mock.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    health = pipeline_with_mock.get_health_status()
    
    # Validate required fields
    assert 'overall_status' in health
    assert 'uptime_seconds' in health
    assert 'active_feeds' in health
    assert 'error_rate' in health
    assert 'memory_mb' in health
    
    assert health['overall_status'] == 'healthy'
    assert health['active_feeds'] > 0
    assert health['error_rate'] == 0


# Legacy main function for backward compatibility
def main():
    """Run tests using pytest."""
    import pytest
    return pytest.main([__file__, '-v'])


if __name__ == '__main__':
    sys.exit(main())
