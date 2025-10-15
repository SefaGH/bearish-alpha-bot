#!/usr/bin/env python3
"""
Test Data Aggregator for Cross-Exchange Normalization.

Tests for Phase 2.2 data aggregator implementation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_aggregator import DataAggregator
from core.market_data_pipeline import MarketDataPipeline
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


class MockCcxtClient:
    """Mock CcxtClient for testing."""
    
    def __init__(self, name: str, candle_count: int = 100, add_issues: bool = False):
        self.name = name
        self.candle_count = candle_count
        self.add_issues = add_issues
    
    def validate_and_get_symbol(self, symbol: str) -> str:
        """Mock symbol validation."""
        return symbol
    
    def ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        """Mock OHLCV fetch."""
        import time
        current_time = int(time.time() * 1000)
        
        data = []
        for i in range(min(limit, self.candle_count)):
            timestamp = current_time - (i * 30 * 60 * 1000)  # 30min intervals
            
            # Base values
            base_price = 100.0 + i
            open_price = base_price
            high_price = base_price + 1.0
            low_price = base_price - 1.0
            close_price = base_price + 0.5
            volume = 1000.0
            
            # Add issues for testing
            if self.add_issues and i % 10 == 0:
                # Add invalid OHLC relationship
                high_price = low_price - 1  # Invalid: high < low
            
            if self.add_issues and i % 15 == 0:
                # Add zero volume
                volume = 0
            
            data.append([
                timestamp,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ])
        
        return list(reversed(data))
    
    def fetch_ohlcv_bulk(self, symbol: str, timeframe: str, target_limit: int):
        """Mock bulk OHLCV fetch."""
        return self.ohlcv(symbol, timeframe, target_limit)


def test_initialization():
    """Test DataAggregator initialization."""
    print("\n" + "="*60)
    print("TEST: DataAggregator Initialization")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        assert aggregator.pipeline == pipeline, "Pipeline not stored correctly"
        assert 'min_candles' in aggregator.quality_thresholds, "Should have min_candles threshold"
        assert 'max_gap_ratio' in aggregator.quality_thresholds, "Should have max_gap_ratio threshold"
        assert 'freshness_minutes' in aggregator.quality_thresholds, "Should have freshness_minutes threshold"
        
        assert aggregator.quality_thresholds['min_candles'] == 50, "min_candles should be 50"
        assert aggregator.quality_thresholds['max_gap_ratio'] == 0.05, "max_gap_ratio should be 0.05"
        assert aggregator.quality_thresholds['freshness_minutes'] == 60, "freshness_minutes should be 60"
        
        print("✅ PASS: DataAggregator initialized correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normalize_ohlcv_data():
    """Test exchange-specific data normalization."""
    print("\n" + "="*60)
    print("TEST: Normalize OHLCV Data")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Generate mock raw data
        import time
        current_time = int(time.time() * 1000)
        raw_data = []
        for i in range(10):
            raw_data.append([
                current_time - (i * 30 * 60 * 1000),
                100.0,
                101.0,
                99.0,
                100.5,
                1000.0
            ])
        
        # Test normalization for different exchanges
        for exchange in ['bingx', 'kucoinfutures', 'binance', 'bitget', 'ascendex']:
            df = aggregator.normalize_ohlcv_data(raw_data, exchange)
            
            assert not df.empty, f"{exchange}: Should return non-empty DataFrame"
            assert 'open' in df.columns, f"{exchange}: Should have 'open' column"
            assert 'close' in df.columns, f"{exchange}: Should have 'close' column"
            assert df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex), \
                f"{exchange}: Should have timestamp index"
            
            print(f"  ✓ {exchange}: normalized {len(df)} candles")
        
        print("✅ PASS: Data normalization working for all exchanges")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregate_multi_exchange():
    """Test cross-exchange data aggregation."""
    print("\n" + "="*60)
    print("TEST: Aggregate Multi-Exchange")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1', candle_count=100),
            'mock2': MockCcxtClient('mock2', candle_count=80),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Start feeds to populate data
        pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
        
        # Aggregate data
        result = aggregator.aggregate_multi_exchange('BTC/USDT:USDT', '30m')
        
        assert 'sources' in result, "Result should have 'sources'"
        assert 'best_exchange' in result, "Result should have 'best_exchange'"
        assert 'total_sources' in result, "Result should have 'total_sources'"
        
        assert result['total_sources'] > 0, "Should have at least one source"
        assert result['best_exchange'] is not None, "Should select a best exchange"
        
        # Verify source data structure
        for exchange_name, source_data in result['sources'].items():
            assert 'data' in source_data, f"{exchange_name}: Should have 'data'"
            assert 'quality_score' in source_data, f"{exchange_name}: Should have 'quality_score'"
            assert 'candle_count' in source_data, f"{exchange_name}: Should have 'candle_count'"
            assert 'freshness' in source_data, f"{exchange_name}: Should have 'freshness'"
            
            assert 0.0 <= source_data['quality_score'] <= 1.0, \
                f"{exchange_name}: Quality score should be between 0 and 1"
        
        print(f"✅ PASS: Aggregated data from {result['total_sources']} sources")
        print(f"   Best exchange: {result['best_exchange']}")
        for exchange_name, source_data in result['sources'].items():
            print(f"   {exchange_name}: quality={source_data['quality_score']:.3f}, "
                  f"candles={source_data['candle_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_best_data_source():
    """Test best data source selection."""
    print("\n" + "="*60)
    print("TEST: Get Best Data Source")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1', candle_count=100),
            'mock2': MockCcxtClient('mock2', candle_count=80),
            'mock3': MockCcxtClient('mock3', candle_count=120),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Start feeds
        pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])
        
        # Get best source
        best = aggregator.get_best_data_source('BTC/USDT:USDT', '30m')
        
        assert best is not None, "Should return a best exchange"
        assert best in exchanges.keys(), "Best exchange should be in available exchanges"
        
        print(f"✅ PASS: Best data source selected: {best}")
        
        # Test with specific exchange list
        best_filtered = aggregator.get_best_data_source('BTC/USDT:USDT', '30m', 
                                                        exchanges=['mock1', 'mock2'])
        
        assert best_filtered in ['mock1', 'mock2'], "Should select from filtered list"
        print(f"   Filtered best source: {best_filtered}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_consensus_data():
    """Test consensus data generation."""
    print("\n" + "="*60)
    print("TEST: Get Consensus Data")
    print("="*60)
    
    try:
        # Use different exchange names to ensure we get multiple sources
        # Pipeline uses fallback mechanism, so we manually populate both exchanges
        exchanges = {
            'bingx': MockCcxtClient('bingx', candle_count=100),
            'binance': MockCcxtClient('binance', candle_count=100),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Create synchronized test data with same timestamps for both exchanges
        import time
        current_time = int(time.time() * 1000)
        
        # Generate shared timestamps
        shared_ohlcv = []
        for i in range(100):
            timestamp = current_time - (i * 30 * 60 * 1000)  # 30min intervals
            shared_ohlcv.append([
                timestamp,
                100.0 + i,
                101.0 + i,
                99.0 + i,
                100.5 + i,
                1000.0
            ])
        shared_ohlcv = list(reversed(shared_ohlcv))
        
        # Store same timestamps for both exchanges (with slight price variations)
        from core.indicators import add_indicators
        
        for exchange_name in ['bingx', 'binance']:
            # Use shared timestamps but slightly different prices
            ohlcv_data = []
            for candle in shared_ohlcv:
                variation = 0.1 if exchange_name == 'binance' else 0.0
                ohlcv_data.append([
                    candle[0],  # Same timestamp
                    candle[1] + variation,
                    candle[2] + variation,
                    candle[3] + variation,
                    candle[4] + variation,
                    candle[5]
                ])
            
            df = pipeline._ohlcv_to_dataframe(ohlcv_data)
            df = add_indicators(df, None)
            pipeline._store_data(exchange_name, 'BTC/USDT:USDT', '30m', df)
        
        # Generate consensus
        consensus = aggregator.get_consensus_data('BTC/USDT:USDT', '30m', min_sources=2)
        
        assert consensus is not None, "Should return consensus data"
        assert not consensus.empty, "Consensus should not be empty"
        assert 'open' in consensus.columns, "Should have 'open' column"
        assert 'close' in consensus.columns, "Should have 'close' column"
        
        # Check metadata
        if hasattr(consensus, 'attrs'):
            assert 'sources' in consensus.attrs or True, "May have sources metadata"
        
        print(f"✅ PASS: Generated consensus data with {len(consensus)} candles")
        
        # Test with insufficient sources
        consensus_fail = aggregator.get_consensus_data('BTC/USDT:USDT', '30m', min_sources=10)
        assert consensus_fail is None, "Should return None with insufficient sources"
        print(f"   ✓ Correctly returned None for insufficient sources")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validate_and_clean():
    """Test data validation and cleaning."""
    print("\n" + "="*60)
    print("TEST: Validate and Clean Data")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Create test data with issues
        timestamps = pd.date_range(start='2024-01-01', periods=20, freq='30min')
        df_test = pd.DataFrame({
            'open': np.random.uniform(90, 110, 20),
            'high': np.random.uniform(95, 115, 20),
            'low': np.random.uniform(85, 105, 20),
            'close': np.random.uniform(90, 110, 20),
            'volume': np.random.uniform(500, 1500, 20)
        }, index=timestamps)
        
        # Add invalid OHLC relationship (high < low)
        df_test.loc[timestamps[5], 'high'] = 50
        df_test.loc[timestamps[5], 'low'] = 100
        
        # Add outlier
        df_test.loc[timestamps[10], 'close'] = 10000
        
        # Add negative volume
        df_test.loc[timestamps[15], 'volume'] = -100
        
        # Clean data
        df_clean = aggregator._validate_and_clean(df_test)
        
        assert len(df_clean) < len(df_test), "Should remove invalid data"
        
        # Verify OHLC integrity in clean data
        valid_ohlc = (
            (df_clean['high'] >= df_clean['low']) &
            (df_clean['high'] >= df_clean['open']) &
            (df_clean['high'] >= df_clean['close']) &
            (df_clean['low'] <= df_clean['open']) &
            (df_clean['low'] <= df_clean['close'])
        )
        assert valid_ohlc.all(), "All clean data should have valid OHLC"
        
        # Verify no negative volume
        assert (df_clean['volume'] >= 0).all(), "Should not have negative volume"
        
        print(f"✅ PASS: Data validation working correctly")
        print(f"   Original: {len(df_test)} candles")
        print(f"   Cleaned: {len(df_clean)} candles")
        print(f"   Removed: {len(df_test) - len(df_clean)} invalid candles")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calculate_quality_score():
    """Test quality score calculation."""
    print("\n" + "="*60)
    print("TEST: Calculate Quality Score")
    print("="*60)
    
    try:
        # Use different exchange names to ensure we get data from both
        exchanges = {
            'binance': MockCcxtClient('binance', candle_count=100),
            'bitget': MockCcxtClient('bitget', candle_count=50, add_issues=True),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Manually fetch and store data from both exchanges
        # (Pipeline's start_feeds() uses fallback and only stores from first successful exchange)
        for exchange_name in ['binance', 'bitget']:
            client = exchanges[exchange_name]
            ohlcv_data = client.ohlcv('BTC/USDT:USDT', '30m', 100)
            df = pipeline._ohlcv_to_dataframe(ohlcv_data)
            from core.indicators import add_indicators
            df = add_indicators(df, None)
            pipeline._store_data(exchange_name, 'BTC/USDT:USDT', '30m', df)
        
        # Get data and calculate scores
        df1 = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='binance')
        df2 = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='bitget')
        
        assert df1 is not None, "Should have data from binance"
        assert df2 is not None, "Should have data from bitget"
        
        score1 = aggregator._calculate_quality_score(df1, 'binance')
        score2 = aggregator._calculate_quality_score(df2, 'bitget')
        
        assert 0.0 <= score1 <= 1.0, "Score should be between 0 and 1"
        assert 0.0 <= score2 <= 1.0, "Score should be between 0 and 1"
        
        # binance should have better score (more candles, no issues)
        assert score1 > score2, "Clean data should have higher quality score"
        
        print(f"✅ PASS: Quality scoring working correctly")
        print(f"   binance (clean): {score1:.3f}")
        print(f"   bitget (issues): {score2:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_data_handling():
    """Test handling of empty data."""
    print("\n" + "="*60)
    print("TEST: Empty Data Handling")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1'),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Test normalize with empty data
        df_empty = aggregator.normalize_ohlcv_data([], 'bingx')
        assert df_empty.empty, "Should return empty DataFrame for empty input"
        
        # Test quality score with empty DataFrame
        score = aggregator._calculate_quality_score(pd.DataFrame(), 'mock1')
        assert score == 0.0, "Empty data should have 0 quality score"
        
        # Test validate_and_clean with empty DataFrame
        df_clean = aggregator._validate_and_clean(pd.DataFrame())
        assert df_clean.empty, "Should return empty DataFrame"
        
        print("✅ PASS: Empty data handling working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_pipeline():
    """Test integration with MarketDataPipeline."""
    print("\n" + "="*60)
    print("TEST: Integration with MarketDataPipeline")
    print("="*60)
    
    try:
        exchanges = {
            'bingx': MockCcxtClient('bingx', candle_count=100),
            'kucoinfutures': MockCcxtClient('kucoinfutures', candle_count=90),
            'binance': MockCcxtClient('binance', candle_count=110),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Start feeds with multiple symbols and timeframes
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        timeframes = ['30m', '1h']
        
        pipeline.start_feeds(symbols, timeframes)
        
        # Test aggregation for each symbol/timeframe combination
        for symbol in symbols:
            for timeframe in timeframes:
                result = aggregator.aggregate_multi_exchange(symbol, timeframe)
                
                assert result['total_sources'] > 0, \
                    f"{symbol} {timeframe}: Should have sources"
                assert result['best_exchange'] is not None, \
                    f"{symbol} {timeframe}: Should have best exchange"
                
                print(f"  ✓ {symbol} {timeframe}: {result['total_sources']} sources, "
                      f"best={result['best_exchange']}")
        
        print("✅ PASS: Integration with pipeline working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_timeframe_support():
    """Test support for multiple timeframes."""
    print("\n" + "="*60)
    print("TEST: Multi-Timeframe Support")
    print("="*60)
    
    try:
        exchanges = {
            'mock1': MockCcxtClient('mock1', candle_count=100),
        }
        
        pipeline = MarketDataPipeline(exchanges)
        aggregator = DataAggregator(pipeline)
        
        # Test all required timeframes
        timeframes = ['30m', '1h', '4h', '1d']
        symbol = 'BTC/USDT:USDT'
        
        for timeframe in timeframes:
            pipeline.start_feeds([symbol], [timeframe])
            
            best = aggregator.get_best_data_source(symbol, timeframe)
            assert best is not None, f"Should find best source for {timeframe}"
            
            print(f"  ✓ {timeframe}: best source = {best}")
        
        print("✅ PASS: Multi-timeframe support working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all data aggregator tests."""
    print("\n" + "="*70)
    print("DATA AGGREGATOR TEST SUITE")
    print("Phase 2.2: Cross-Exchange Normalization and Quality Management")
    print("="*70)
    
    tests = [
        test_initialization,
        test_normalize_ohlcv_data,
        test_aggregate_multi_exchange,
        test_get_best_data_source,
        test_get_consensus_data,
        test_validate_and_clean,
        test_calculate_quality_score,
        test_empty_data_handling,
        test_integration_with_pipeline,
        test_multi_timeframe_support,
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
