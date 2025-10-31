#!/usr/bin/env python3
"""
Test for Symbol Key Normalization Fix

This test verifies that the StreamDataCollector can handle symbol format inconsistencies
by automatically normalizing symbols to include settlement currency (:USDT).

This addresses the issue where:
- Data written with 'BTC/USDT:USDT' couldn't be read with 'BTC/USDT'
- Data written with 'BTC/USDT' couldn't be read with 'BTC/USDT:USDT'
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.stream_data_collector import StreamDataCollector


class TestSymbolNormalization:
    """Test symbol normalization in StreamDataCollector."""
    
    @pytest.fixture
    def collector(self):
        """Create a StreamDataCollector instance for testing."""
        return StreamDataCollector(buffer_size=300)
    
    @pytest.fixture
    def test_data(self):
        """Create test OHLCV data."""
        timestamps = pd.date_range(end=datetime.now(), periods=250, freq='1min')
        return pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 250),
            'high': np.random.uniform(45000, 46000, 250),
            'low': np.random.uniform(39000, 40000, 250),
            'close': np.random.uniform(40000, 45000, 250),
            'volume': np.random.uniform(100, 1000, 250)
        }, index=timestamps)
    
    def test_write_with_suffix_read_without_suffix(self, collector, test_data):
        """
        Test that data written with :USDT can be read without :USDT.
        
        This is the main bug scenario from the issue.
        """
        # Prime buffer with full format
        collector.prime_buffer_with_dataframe('bingx', 'BTC/USDT:USDT', '1m', test_data)
        
        # Read with shortened format (what might happen in some code paths)
        ohlcv = collector.get_latest_ohlcv('bingx', 'BTC/USDT', '1m', limit=250)
        
        assert ohlcv is not None, "Should be able to read data"
        assert len(ohlcv) == 250, f"Expected 250 candles, got {len(ohlcv)}"
    
    def test_write_without_suffix_read_with_suffix(self, collector, test_data):
        """
        Test that data written without :USDT can be read with :USDT.
        
        This is the reverse scenario.
        """
        # Prime buffer with shortened format
        collector.prime_buffer_with_dataframe('bingx', 'ETH/USDT', '1m', test_data)
        
        # Read with full format
        ohlcv = collector.get_latest_ohlcv('bingx', 'ETH/USDT:USDT', '1m', limit=250)
        
        assert ohlcv is not None, "Should be able to read data"
        assert len(ohlcv) == 250, f"Expected 250 candles, got {len(ohlcv)}"
    
    def test_both_without_suffix(self, collector, test_data):
        """Test that both write and read without :USDT works."""
        collector.prime_buffer_with_dataframe('bingx', 'SOL/USDT', '1m', test_data)
        ohlcv = collector.get_latest_ohlcv('bingx', 'SOL/USDT', '1m', limit=250)
        
        assert ohlcv is not None, "Should be able to read data"
        assert len(ohlcv) == 250, f"Expected 250 candles, got {len(ohlcv)}"
    
    def test_both_with_suffix(self, collector, test_data):
        """Test that both write and read with :USDT works."""
        collector.prime_buffer_with_dataframe('bingx', 'AVAX/USDT:USDT', '1m', test_data)
        ohlcv = collector.get_latest_ohlcv('bingx', 'AVAX/USDT:USDT', '1m', limit=250)
        
        assert ohlcv is not None, "Should be able to read data"
        assert len(ohlcv) == 250, f"Expected 250 candles, got {len(ohlcv)}"
    
    def test_keys_are_normalized_in_buffer(self, collector, test_data):
        """
        Test that all keys in buffer are normalized to the same format.
        
        This ensures consistent key format regardless of input.
        """
        # Write with different formats
        collector.prime_buffer_with_dataframe('bingx', 'BTC/USDT', '1m', test_data)
        collector.prime_buffer_with_dataframe('bingx', 'ETH/USDT:USDT', '1m', test_data)
        collector.prime_buffer_with_dataframe('bingx', 'SOL/USDT', '5m', test_data)
        
        # Check that all keys are normalized
        keys = list(collector.ohlcv_data['bingx'].keys())
        
        # All keys should have settlement currency
        expected_keys = ['BTC/USDT:USDT_1m', 'ETH/USDT:USDT_1m', 'SOL/USDT:USDT_5m']
        assert set(keys) == set(expected_keys), f"Expected {expected_keys}, got {keys}"
    
    def test_non_usdt_pairs_unchanged(self, collector, test_data):
        """Test that non-USDT pairs are not modified."""
        # If we ever support other quote currencies
        collector.prime_buffer_with_dataframe('binance', 'BTC/EUR', '1m', test_data)
        
        keys = list(collector.ohlcv_data['binance'].keys())
        assert 'BTC/EUR_1m' in keys, "Non-USDT pairs should remain unchanged"
    
    def test_indicator_validator_scenario(self, collector, test_data):
        """
        Test the exact scenario from the issue:
        - MarketDataPipeline primes with normalized symbols
        - IndicatorValidator reads with possibly unnormalized symbols
        """
        # Simulate what MarketDataPipeline does (priming with normalized symbols)
        symbols_from_config = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        for symbol in symbols_from_config:
            # Use the collector's normalization (no duplication)
            collector.prime_buffer_with_dataframe('bingx', symbol, '1m', test_data)
        
        # Simulate what IndicatorValidator might do (possibly with unnormalized symbols)
        # In some scenarios, symbols might lose their :USDT suffix
        validator_symbols = ['BTC/USDT', 'ETH/USDT']  # Without :USDT
        
        for symbol in validator_symbols:
            ohlcv = collector.get_latest_ohlcv('bingx', symbol, '1m', limit=250)
            assert ohlcv is not None, f"Validator should find data for {symbol}"
            assert len(ohlcv) == 250, f"Validator should get 250 candles for {symbol}, got {len(ohlcv)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
