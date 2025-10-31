#!/usr/bin/env python3
"""
Test for Issue: [CRITICAL BUG] Configuration Loading and Data Integration Errors

This test verifies the fixes for:
1. Configuration loading: ML timeframes should load from YAML when env var not set
2. Data integration: prime_buffer and get_latest_ohlcv should use same data structure
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.live_trading_config import LiveTradingConfiguration
from core.stream_data_collector import StreamDataCollector


class TestConfigurationLoading:
    """Test configuration loading fixes."""
    
    def test_ml_timeframes_from_yaml_when_env_not_set(self):
        """
        Test that ML timeframes load from YAML when ML_TIMEFRAMES env var is not set.
        
        This addresses the bug where empty list from env_config was overwriting
        YAML values during deep_merge.
        """
        # Ensure ML_TIMEFRAMES env var is not set
        if 'ML_TIMEFRAMES' in os.environ:
            del os.environ['ML_TIMEFRAMES']
        
        # Load configuration
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Get ML timeframes
        ml_config = config.get('ml', {})
        timeframes = ml_config.get('timeframes', [])
        
        # Assert timeframes were loaded from YAML
        assert timeframes is not None, "ML timeframes should not be None"
        assert len(timeframes) > 0, "ML timeframes should not be empty when loaded from YAML"
        assert '5m' in timeframes or '1h' in timeframes, "ML timeframes should contain expected values from YAML"
        
        print(f"✓ ML timeframes loaded from YAML: {timeframes}")
    
    def test_ml_timeframes_from_env_when_set(self, monkeypatch):
        """
        Test that ML timeframes load from environment variable when set.
        
        This ensures env var takes priority over YAML.
        
        Note: Using monkeypatch for proper test isolation.
        """
        # Set ML_TIMEFRAMES env var using monkeypatch for proper cleanup
        test_timeframes = '1m,5m,15m'
        monkeypatch.setenv('ML_TIMEFRAMES', test_timeframes)
        
        # Load configuration
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Get ML timeframes
        ml_config = config.get('ml', {})
        timeframes = ml_config.get('timeframes', [])
        
        # Assert timeframes were loaded from env var
        expected = ['1m', '5m', '15m']
        assert timeframes == expected, f"Expected {expected}, got {timeframes}"
        
        print(f"✓ ML timeframes loaded from env var: {timeframes}")


class TestDataIntegration:
    """Test data integration fixes."""
    
    def test_prime_buffer_and_read_consistency(self):
        """
        Test that data primed with prime_buffer_with_dataframe can be read with get_latest_ohlcv.
        
        This addresses the bug where data was written but couldn't be read back.
        """
        # Create collector
        collector = StreamDataCollector(buffer_size=300)
        
        # Create test data
        timestamps = pd.date_range(end=datetime.now(), periods=250, freq='1min')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 250),
            'high': np.random.uniform(45000, 46000, 250),
            'low': np.random.uniform(39000, 40000, 250),
            'close': np.random.uniform(40000, 45000, 250),
            'volume': np.random.uniform(100, 1000, 250)
        }, index=timestamps)
        
        # Prime buffer
        exchange = 'bingx'
        symbol = 'BTC/USDT:USDT'
        timeframe = '1m'
        
        collector.prime_buffer_with_dataframe(exchange, symbol, timeframe, df)
        
        # Read data back
        ohlcv_list = collector.get_latest_ohlcv(exchange, symbol, timeframe, limit=250)
        
        # Assert data was read successfully
        assert ohlcv_list is not None, "get_latest_ohlcv should not return None"
        assert len(ohlcv_list) == 250, f"Expected 250 candles, got {len(ohlcv_list)}"
        
        # Verify data structure
        first_candle = ohlcv_list[0]
        assert len(first_candle) == 6, f"Each candle should have 6 fields, got {len(first_candle)}"
        
        print(f"✓ Successfully primed and read 250 candles for {symbol}")
    
    def test_buffer_key_consistency(self):
        """
        Test that _get_buffer_key generates consistent keys.
        
        This ensures both prime_buffer and get_latest_ohlcv use same keys.
        """
        collector = StreamDataCollector(buffer_size=300)
        
        # Test key generation
        symbol = 'BTC/USDT:USDT'
        timeframe = '1m'
        
        key = collector._get_buffer_key(symbol, timeframe)
        expected_key = 'BTC/USDT:USDT_1m'
        
        assert key == expected_key, f"Expected key '{expected_key}', got '{key}'"
        
        print(f"✓ Buffer key generated correctly: {key}")
    
    def test_multiple_symbols(self):
        """
        Test that multiple symbols can be primed and read correctly.
        
        This simulates the real-world scenario with multiple trading pairs.
        """
        collector = StreamDataCollector(buffer_size=300)
        
        # Create test data
        timestamps = pd.date_range(end=datetime.now(), periods=250, freq='1min')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 250),
            'high': np.random.uniform(45000, 46000, 250),
            'low': np.random.uniform(39000, 40000, 250),
            'close': np.random.uniform(40000, 45000, 250),
            'volume': np.random.uniform(100, 1000, 250)
        }, index=timestamps)
        
        # Test with multiple symbols
        exchange = 'bingx'
        timeframe = '1m'
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        
        for symbol in symbols:
            # Prime buffer
            collector.prime_buffer_with_dataframe(exchange, symbol, timeframe, df)
            
            # Read data back
            ohlcv_list = collector.get_latest_ohlcv(exchange, symbol, timeframe, limit=250)
            
            # Assert data was read successfully
            assert ohlcv_list is not None, f"get_latest_ohlcv returned None for {symbol}"
            assert len(ohlcv_list) == 250, f"Expected 250 candles for {symbol}, got {len(ohlcv_list)}"
            
            print(f"✓ {symbol}: Successfully primed and read 250 candles")


if __name__ == '__main__':
    # Run tests
    print("="*70)
    print("Running Configuration and Data Integration Tests")
    print("="*70)
    
    # Test configuration
    config_tests = TestConfigurationLoading()
    config_tests.test_ml_timeframes_from_yaml_when_env_not_set()
    config_tests.test_ml_timeframes_from_env_when_set()
    
    # Test data integration
    data_tests = TestDataIntegration()
    data_tests.test_prime_buffer_and_read_consistency()
    data_tests.test_buffer_key_consistency()
    data_tests.test_multiple_symbols()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
