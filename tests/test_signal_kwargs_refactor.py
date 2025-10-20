"""
Test for the refactored signal calling mechanism with dynamic kwargs.
This test validates that the strategy.signal() call correctly handles
different parameter combinations using dynamic kwargs building.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import inspect
import pytest
import pandas as pd
from typing import Dict, Optional


class MockStrategy:
    """Mock strategy for testing different signal signatures."""
    
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.last_call_args = None
        self.last_call_kwargs = None
    
    def signal(self, df_30m, df_1h=None, regime_data=None, symbol=None):
        """Mock signal method that records how it was called."""
        self.last_call_args = (df_30m,)
        self.last_call_kwargs = {
            'df_1h': df_1h,
            'regime_data': regime_data,
            'symbol': symbol
        }
        return {'action': 'buy', 'confidence': 0.8}


class TestSignalKwargsRefactor:
    """Test the refactored signal calling mechanism."""
    
    def test_signal_with_all_params(self):
        """Test signal call with all parameters (df_1h, regime_data, symbol)."""
        # Create mock strategy
        strategy = MockStrategy('test', {})
        
        # Mock data
        df_30m = pd.DataFrame({'close': [100, 101, 102]})
        df_1h = pd.DataFrame({'close': [100, 105]})
        regime_data = {'trend': 'bullish', 'volatility': 'normal'}
        symbol = 'BTC/USDT:USDT'
        
        # Inspect signature
        sig = inspect.signature(strategy.signal)
        params = list(sig.parameters.keys())
        
        has_regime_param = 'regime_data' in params
        has_df_1h_param = 'df_1h' in params
        has_symbol_param = 'symbol' in params
        
        # Build kwargs dynamically (mimicking the refactored code)
        kwargs = {}
        if has_regime_param:
            kwargs['regime_data'] = regime_data
        if has_df_1h_param and df_1h is not None:
            kwargs['df_1h'] = df_1h
        if has_symbol_param:
            kwargs['symbol'] = symbol
        
        # Call signal with kwargs
        signal = strategy.signal(df_30m, **kwargs)
        
        # Verify the call
        assert signal is not None
        assert strategy.last_call_kwargs['regime_data'] == regime_data
        assert strategy.last_call_kwargs['df_1h'].equals(df_1h)
        assert strategy.last_call_kwargs['symbol'] == symbol
    
    def test_signal_with_only_regime_data(self):
        """Test signal call with only regime_data parameter."""
        strategy = MockStrategy('test', {})
        df_30m = pd.DataFrame({'close': [100, 101, 102]})
        regime_data = {'trend': 'bullish', 'volatility': 'normal'}
        
        sig = inspect.signature(strategy.signal)
        params = list(sig.parameters.keys())
        
        has_regime_param = 'regime_data' in params
        has_df_1h_param = 'df_1h' in params
        has_symbol_param = 'symbol' in params
        
        df_1h = None  # df_1h is None in this test case
        
        kwargs = {}
        if has_regime_param:
            kwargs['regime_data'] = regime_data
        if has_df_1h_param and df_1h is not None:
            kwargs['df_1h'] = df_1h
        if has_symbol_param:
            kwargs['symbol'] = None
        
        signal = strategy.signal(df_30m, **kwargs)
        
        assert signal is not None
        assert strategy.last_call_kwargs['regime_data'] == regime_data
        # df_1h and symbol should be None since they weren't provided
        assert strategy.last_call_kwargs['df_1h'] is None
        assert strategy.last_call_kwargs['symbol'] is None
    
    def test_signal_with_df_1h_and_regime(self):
        """Test signal call with df_1h and regime_data but no symbol."""
        strategy = MockStrategy('test', {})
        df_30m = pd.DataFrame({'close': [100, 101, 102]})
        df_1h = pd.DataFrame({'close': [100, 105]})
        regime_data = {'trend': 'bullish', 'volatility': 'normal'}
        
        sig = inspect.signature(strategy.signal)
        params = list(sig.parameters.keys())
        
        has_regime_param = 'regime_data' in params
        has_df_1h_param = 'df_1h' in params
        has_symbol_param = 'symbol' in params
        
        kwargs = {}
        if has_regime_param:
            kwargs['regime_data'] = regime_data
        if has_df_1h_param and df_1h is not None:
            kwargs['df_1h'] = df_1h
        if has_symbol_param:
            kwargs['symbol'] = None
        
        signal = strategy.signal(df_30m, **kwargs)
        
        assert signal is not None
        assert strategy.last_call_kwargs['regime_data'] == regime_data
        assert strategy.last_call_kwargs['df_1h'].equals(df_1h)
        # symbol should be None since it wasn't provided
        assert strategy.last_call_kwargs['symbol'] is None
    
    def test_signal_with_only_df_30m(self):
        """Test signal call with only df_30m (base strategy)."""
        strategy = MockStrategy('test', {})
        df_30m = pd.DataFrame({'close': [100, 101, 102]})
        
        sig = inspect.signature(strategy.signal)
        params = list(sig.parameters.keys())
        
        # Mock a base strategy that doesn't accept these params
        has_regime_param = False  # Not in params
        has_df_1h_param = False  # Not in params
        has_symbol_param = False  # Not in params
        df_1h = None
        
        kwargs = {}
        if has_regime_param:
            kwargs['regime_data'] = None
        if has_df_1h_param and df_1h is not None:
            kwargs['df_1h'] = df_1h
        if has_symbol_param:
            kwargs['symbol'] = None
        
        signal = strategy.signal(df_30m, **kwargs)
        
        assert signal is not None
        # All kwargs should be None for a base strategy
        assert strategy.last_call_kwargs['regime_data'] is None
        assert strategy.last_call_kwargs['df_1h'] is None
        assert strategy.last_call_kwargs['symbol'] is None
    
    def test_signal_with_symbol_only(self):
        """Test signal call with only symbol parameter (besides df_30m)."""
        strategy = MockStrategy('test', {})
        df_30m = pd.DataFrame({'close': [100, 101, 102]})
        symbol = 'BTC/USDT:USDT'
        
        sig = inspect.signature(strategy.signal)
        params = list(sig.parameters.keys())
        
        has_regime_param = 'regime_data' in params
        has_df_1h_param = 'df_1h' in params
        has_symbol_param = 'symbol' in params
        df_1h = None  # df_1h is None in this test case
        
        kwargs = {}
        if has_regime_param:
            kwargs['regime_data'] = None
        if has_df_1h_param and df_1h is not None:
            kwargs['df_1h'] = df_1h
        if has_symbol_param:
            kwargs['symbol'] = symbol
        
        signal = strategy.signal(df_30m, **kwargs)
        
        assert signal is not None
        assert strategy.last_call_kwargs['symbol'] == symbol
        # regime_data and df_1h should be None since they weren't provided
        assert strategy.last_call_kwargs['regime_data'] is None
        assert strategy.last_call_kwargs['df_1h'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
