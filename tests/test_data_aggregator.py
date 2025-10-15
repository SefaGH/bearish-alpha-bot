#!/usr/bin/env python3
"""
Test Data Aggregator for Cross-Exchange Normalization.

Tests for Phase 2.2 data aggregator implementation.
Uses pytest and unittest.mock for comprehensive testing.

Tests cover:
- Data aggregator initialization
- Exchange-specific data normalization
- Multi-exchange aggregation with quality scoring
- Best data source selection
- Consensus data generation
- Data validation and cleaning
- Integration with MarketDataPipeline
"""

import sys
import os
from unittest.mock import Mock
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_aggregator import DataAggregator
from core.market_data_pipeline import MarketDataPipeline
from core.ccxt_client import CcxtClient
from core.indicators import add_indicators


# Fixed timestamp for deterministic tests (recent enough to pass freshness checks)
FIXED_TIMESTAMP = int(time.time() * 1000) - (5 * 60 * 1000)  # 5 minutes ago


@pytest.fixture
def generate_ohlcv_data():
    """Factory fixture to generate consistent OHLCV test data."""
    def _generate(count=100, interval_minutes=30, base_price=50000, with_issues=False):
        data = []
        for i in range(count):
            timestamp = FIXED_TIMESTAMP - (i * interval_minutes * 60 * 1000)
            
            if with_issues and i % 10 == 0:
                # Invalid OHLC relationship
                high = base_price - 200
                low = base_price - 100
            else:
                high = base_price + 100
                low = base_price - 100
            
            volume = 0 if (with_issues and i % 15 == 0) else (1000 + i)
            
            data.append([
                timestamp,
                base_price,
                high,
                low,
                base_price + 50,
                volume
            ])
            base_price += 10
        return list(reversed(data))
    return _generate


@pytest.fixture
def mock_ccxt_client(generate_ohlcv_data):
    """Create a mock CcxtClient for testing with clean data."""
    client = Mock(spec=CcxtClient)
    client.name = 'binance'
    client.validate_and_get_symbol.return_value = 'BTC/USDT:USDT'
    
    # Use fixture to generate consistent test data
    sample_data = generate_ohlcv_data()
    
    client.ohlcv.return_value = sample_data
    client.fetch_ohlcv_bulk.return_value = sample_data
    
    return client


@pytest.fixture
def mock_ccxt_client_with_issues(generate_ohlcv_data):
    """Create a mock CcxtClient with data quality issues."""
    client = Mock(spec=CcxtClient)
    client.name = 'bitget'
    client.validate_and_get_symbol.return_value = 'BTC/USDT:USDT'
    
    # Use fixture to generate test data with issues
    sample_data = generate_ohlcv_data(count=50, with_issues=True)
    
    client.ohlcv.return_value = sample_data
    client.fetch_ohlcv_bulk.return_value = sample_data
    
    return client


@pytest.fixture
def pipeline_with_aggregator(mock_ccxt_client):
    """Create a pipeline with aggregator."""
    exchanges = {'binance': mock_ccxt_client}
    pipeline = MarketDataPipeline(exchanges)
    aggregator = DataAggregator(pipeline)
    return pipeline, aggregator


@pytest.fixture
def pipeline_multi_exchange(mock_ccxt_client, mock_ccxt_client_with_issues):
    """Create a pipeline with multiple exchanges."""
    exchanges = {
        'binance': mock_ccxt_client,
        'bitget': mock_ccxt_client_with_issues
    }
    pipeline = MarketDataPipeline(exchanges)
    return pipeline


def test_initialization(pipeline_with_aggregator):
    """Test DataAggregator initialization."""
    pipeline, aggregator = pipeline_with_aggregator
    
    assert aggregator.pipeline == pipeline
    assert 'min_candles' in aggregator.quality_thresholds
    assert 'max_gap_ratio' in aggregator.quality_thresholds
    assert 'freshness_minutes' in aggregator.quality_thresholds
    
    assert aggregator.quality_thresholds['min_candles'] == 50
    assert aggregator.quality_thresholds['max_gap_ratio'] == 0.05
    assert aggregator.quality_thresholds['freshness_minutes'] == 60


def test_normalize_ohlcv_data_for_different_exchanges(pipeline_with_aggregator, generate_ohlcv_data):
    """Test exchange-specific data normalization for different exchanges."""
    pipeline, aggregator = pipeline_with_aggregator
    
    # Use fixture to generate consistent test data
    raw_data = generate_ohlcv_data(count=10)
    
    # Test normalization for different exchanges
    exchanges_to_test = ['bingx', 'kucoinfutures', 'binance', 'bitget', 'ascendex']
    for exchange in exchanges_to_test:
        df = aggregator.normalize_ohlcv_data(raw_data, exchange)
        
        assert not df.empty
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex)


def test_aggregate_multi_exchange_quality_scoring(pipeline_with_aggregator):
    """Test cross-exchange data aggregation with quality scoring."""
    pipeline, aggregator = pipeline_with_aggregator
    
    # Start feeds to populate data
    pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    # Aggregate data
    result = aggregator.aggregate_multi_exchange('BTC/USDT:USDT', '30m')
    
    assert 'sources' in result
    assert 'best_exchange' in result
    assert 'total_sources' in result
    
    assert result['total_sources'] > 0
    assert result['best_exchange'] is not None
    
    # Verify source data structure
    for exchange_name, source_data in result['sources'].items():
        assert 'data' in source_data
        assert 'quality_score' in source_data
        assert 'candle_count' in source_data
        assert 'freshness' in source_data
        
        assert 0.0 <= source_data['quality_score'] <= 1.0


def test_get_best_data_source_reliability_ranking(pipeline_with_aggregator):
    """Test get_best_data_source() reliability ranking."""
    pipeline, aggregator = pipeline_with_aggregator
    
    # Start feeds
    pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
    
    # Get best source
    best = aggregator.get_best_data_source('BTC/USDT:USDT', '30m')
    
    assert best is not None
    assert best in pipeline.exchanges.keys()


def test_get_consensus_data_weighted_average(pipeline_multi_exchange, generate_ohlcv_data):
    """Test get_consensus_data() weighted average calculation."""
    aggregator = DataAggregator(pipeline_multi_exchange)
    
    # Use fixture to generate synchronized data
    shared_ohlcv = generate_ohlcv_data()
    
    # Store same timestamps for both exchanges with slight variations
    for exchange_name in ['binance', 'bitget']:
        ohlcv_data = []
        for candle in shared_ohlcv:
            variation = 10 if exchange_name == 'binance' else 0
            ohlcv_data.append([
                candle[0],  # Same timestamp
                candle[1] + variation,
                candle[2] + variation,
                candle[3] + variation,
                candle[4] + variation,
                candle[5]
            ])
        
        df = pipeline_multi_exchange._ohlcv_to_dataframe(ohlcv_data)
        df = add_indicators(df, None)
        pipeline_multi_exchange._store_data(exchange_name, 'BTC/USDT:USDT', '30m', df)
    
    # Generate consensus
    consensus = aggregator.get_consensus_data('BTC/USDT:USDT', '30m', min_sources=2)
    
    assert consensus is not None
    assert not consensus.empty
    assert 'open' in consensus.columns
    assert 'close' in consensus.columns
    
    # Test with insufficient sources
    consensus_fail = aggregator.get_consensus_data('BTC/USDT:USDT', '30m', min_sources=10)
    assert consensus_fail is None


def test_validate_and_clean_ohlc_integrity(pipeline_with_aggregator):
    """Test _validate_and_clean() OHLC integrity checks."""
    pipeline, aggregator = pipeline_with_aggregator
    
    # Create test data with issues
    timestamps = pd.date_range(start='2024-01-01', periods=20, freq='30min')
    df_test = pd.DataFrame({
        'open': np.random.uniform(90, 110, 20),
        'high': np.random.uniform(95, 115, 20),
        'low': np.random.uniform(85, 105, 20),
        'close': np.random.uniform(90, 110, 20),
        'volume': np.random.uniform(500, 1500, 20)
    }, index=timestamps)
    
    # Add invalid OHLC relationship
    df_test.loc[timestamps[5], 'high'] = 50
    df_test.loc[timestamps[5], 'low'] = 100
    
    # Add outlier
    df_test.loc[timestamps[10], 'close'] = 10000
    
    # Add negative volume
    df_test.loc[timestamps[15], 'volume'] = -100
    
    # Clean data
    df_clean = aggregator._validate_and_clean(df_test)
    
    assert len(df_clean) < len(df_test)
    
    # Verify OHLC integrity
    valid_ohlc = (
        (df_clean['high'] >= df_clean['low']) &
        (df_clean['high'] >= df_clean['open']) &
        (df_clean['high'] >= df_clean['close']) &
        (df_clean['low'] <= df_clean['open']) &
        (df_clean['low'] <= df_clean['close'])
    )
    assert valid_ohlc.all()
    
    # Verify no negative volume
    assert (df_clean['volume'] >= 0).all()


def test_calculate_quality_score_accuracy(pipeline_multi_exchange):
    """Test _calculate_quality_score() scoring accuracy."""
    aggregator = DataAggregator(pipeline_multi_exchange)
    
    # Manually fetch and store data from both exchanges
    for exchange_name in ['binance', 'bitget']:
        client = pipeline_multi_exchange.exchanges[exchange_name]
        ohlcv_data = client.ohlcv('BTC/USDT:USDT', '30m', 100)
        df = pipeline_multi_exchange._ohlcv_to_dataframe(ohlcv_data)
        df = add_indicators(df, None)
        pipeline_multi_exchange._store_data(exchange_name, 'BTC/USDT:USDT', '30m', df)
    
    # Get data and calculate scores
    df_binance = pipeline_multi_exchange.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='binance')
    df_bitget = pipeline_multi_exchange.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='bitget')
    
    assert df_binance is not None
    assert df_bitget is not None
    
    score_binance = aggregator._calculate_quality_score(df_binance, 'binance')
    score_bitget = aggregator._calculate_quality_score(df_bitget, 'bitget')
    
    assert 0.0 <= score_binance <= 1.0
    assert 0.0 <= score_bitget <= 1.0
    
    # binance should have better score (more candles, no issues)
    assert score_binance > score_bitget


def test_empty_data_handling(pipeline_with_aggregator):
    """Test handling of empty data."""
    pipeline, aggregator = pipeline_with_aggregator
    
    # Test normalize with empty data
    df_empty = aggregator.normalize_ohlcv_data([], 'bingx')
    assert df_empty.empty
    
    # Test quality score with empty DataFrame
    score = aggregator._calculate_quality_score(pd.DataFrame(), 'binance')
    assert score == 0.0
    
    # Test validate_and_clean with empty DataFrame
    df_clean = aggregator._validate_and_clean(pd.DataFrame())
    assert df_clean.empty


def test_integration_with_pipeline(pipeline_multi_exchange):
    """Test integration with MarketDataPipeline."""
    aggregator = DataAggregator(pipeline_multi_exchange)
    
    # Start feeds with multiple symbols and timeframes
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    timeframes = ['30m', '1h']
    
    pipeline_multi_exchange.start_feeds(symbols, timeframes)
    
    # Test aggregation for each symbol/timeframe combination
    for symbol in symbols:
        for timeframe in timeframes:
            result = aggregator.aggregate_multi_exchange(symbol, timeframe)
            
            assert result['total_sources'] > 0
            assert result['best_exchange'] is not None


def test_multi_timeframe_support(pipeline_with_aggregator):
    """Test support for multiple timeframes."""
    pipeline, aggregator = pipeline_with_aggregator
    
    # Test all required timeframes
    timeframes = ['30m', '1h', '4h', '1d']
    symbol = 'BTC/USDT:USDT'
    
    for timeframe in timeframes:
        pipeline.start_feeds([symbol], [timeframe])
        
        best = aggregator.get_best_data_source(symbol, timeframe)
        assert best is not None


def test_exchange_specific_normalization_methods(pipeline_with_aggregator, generate_ohlcv_data):
    """Test exchange-specific normalization methods."""
    pipeline, aggregator = pipeline_with_aggregator
    
    # Use fixture to generate test data
    raw_data_standard = generate_ohlcv_data(count=5, base_price=100)
    
    df_standard = aggregator.normalize_ohlcv_data(raw_data_standard, 'binance')
    assert not df_standard.empty
    assert len(df_standard) == 5
    
    # Test case 2: With string timestamps (if supported)
    # Exchange-specific behavior can be tested here
    for exchange in ['bingx', 'binance', 'bitget']:
        df = aggregator.normalize_ohlcv_data(raw_data_standard, exchange)
        assert 'timestamp' in str(df.index.name).lower() or isinstance(df.index, pd.DatetimeIndex)


# Legacy main function for backward compatibility
def main():
    """Run tests using pytest."""
    import pytest
    return pytest.main([__file__, '-v'])


if __name__ == '__main__':
    sys.exit(main())
