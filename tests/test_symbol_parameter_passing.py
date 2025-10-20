"""
Test that symbol parameter is correctly passed to adaptive strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np


class TestSymbolParameterPassing(unittest.TestCase):
    """Test that symbol parameter is passed to adaptive strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.df_30m = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'rsi': [65.0, 70.0, 75.0],
            'atr': [1.0, 1.1, 1.2],
            'ema21': [99.0, 100.0, 101.0],
            'ema50': [98.0, 99.0, 100.0],
            'ema200': [97.0, 98.0, 99.0]
        })
        self.df_1h = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'rsi': [60.0, 65.0, 70.0],
            'atr': [1.0, 1.1, 1.2]
        })
        self.regime_data = {
            'trend': 'neutral',
            'momentum': 'sideways',
            'volatility': 'normal',
            'micro_trend_strength': 0.5
        }
        self.test_symbol = 'BTC/USDT:USDT'
    
    def test_adaptive_str_accepts_symbol_parameter(self):
        """Verify AdaptiveShortTheRip accepts symbol parameter."""
        from strategies.adaptive_str import AdaptiveShortTheRip
        
        cfg = {
            'rsi_min': 70,
            'tp_pct': 0.012,
            'sl_atr_mult': 1.0
        }
        strategy = AdaptiveShortTheRip(cfg)
        
        # Should not raise an error when symbol is passed
        try:
            result = strategy.signal(
                self.df_30m, 
                self.df_1h, 
                regime_data=self.regime_data,
                symbol=self.test_symbol
            )
            # Test passes if no exception is raised
            self.assertTrue(True, "symbol parameter accepted")
        except TypeError as e:
            self.fail(f"AdaptiveShortTheRip.signal should accept symbol parameter: {e}")
    
    def test_adaptive_ob_accepts_symbol_parameter(self):
        """Verify AdaptiveOversoldBounce accepts symbol parameter."""
        from strategies.adaptive_ob import AdaptiveOversoldBounce
        
        cfg = {
            'rsi_max': 30,
            'tp_pct': 0.015,
            'sl_atr_mult': 1.0
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        # Create data with low RSI for oversold condition
        df_oversold = self.df_30m.copy()
        df_oversold['rsi'] = [25.0, 28.0, 29.0]
        
        # Should not raise an error when symbol is passed
        try:
            result = strategy.signal(
                df_oversold, 
                self.df_1h, 
                regime_data=self.regime_data,
                symbol=self.test_symbol
            )
            # Test passes if no exception is raised
            self.assertTrue(True, "symbol parameter accepted")
        except TypeError as e:
            self.fail(f"AdaptiveOversoldBounce.signal should accept symbol parameter: {e}")
    
    def test_production_coordinator_passes_symbol(self):
        """Verify that production_coordinator.py passes symbol to adaptive strategies."""
        coord_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'production_coordinator.py')
        with open(coord_path, 'r') as f:
            content = f.read()
        
        # Check that symbol parameter is passed in adaptive strategy calls
        # For AdaptiveOversoldBounce
        self.assertIn('symbol=symbol', content,
                     "symbol parameter should be passed in strategy calls")
        
        # Check specifically for AdaptiveOversoldBounce call
        self.assertIn('ob.signal(df_30m, df_1h, regime_data=metadata.get(\'regime\'), symbol=symbol)', content,
                     "AdaptiveOversoldBounce should receive symbol parameter")
        
        # Check specifically for AdaptiveShortTheRip call
        self.assertIn('strp.signal(df_30m, df_1h, regime_data=metadata.get(\'regime\'), symbol=symbol)', content,
                     "AdaptiveShortTheRip should receive symbol parameter")
        
        # Check for registered strategies with regime_data support
        # This pattern checks the dynamic strategy calling code
        adaptive_pattern_found = False
        if 'regime_data=metadata.get(\'regime\'), symbol=symbol' in content:
            adaptive_pattern_found = True
        
        self.assertTrue(adaptive_pattern_found,
                       "Symbol should be passed when calling adaptive strategies with regime_data")
    
    def test_live_trading_engine_passes_symbol(self):
        """Verify that live_trading_engine.py passes symbol to strategies."""
        engine_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'live_trading_engine.py')
        with open(engine_path, 'r') as f:
            content = f.read()
        
        # Check that symbol parameter check is present
        self.assertIn('has_symbol_param', content,
                     "live_trading_engine should check for symbol parameter")
        
        # Check that symbol is passed when has_symbol_param is True
        self.assertIn('symbol=symbol', content,
                     "symbol should be passed to strategies when they support it")
    
    def test_symbol_specific_threshold_works(self):
        """Verify that symbol-specific threshold logic is triggered."""
        from strategies.adaptive_str import AdaptiveShortTheRip
        
        # Config with symbol-specific threshold
        cfg = {
            'rsi_min': 70,
            'tp_pct': 0.012,
            'sl_atr_mult': 1.0,
            'symbols': {
                'ETH/USDT:USDT': {
                    'rsi_threshold': 65.0
                }
            }
        }
        strategy = AdaptiveShortTheRip(cfg)
        
        # Test that get_symbol_specific_threshold returns correct value
        threshold = strategy.get_symbol_specific_threshold('ETH/USDT:USDT')
        self.assertEqual(threshold, 65.0, "Should return symbol-specific threshold")
        
        # Test that None is returned for unknown symbol
        threshold = strategy.get_symbol_specific_threshold('BTC/USDT:USDT')
        self.assertIsNone(threshold, "Should return None for unknown symbol")
        
        # Test that None is returned when symbol is None
        threshold = strategy.get_symbol_specific_threshold(None)
        self.assertIsNone(threshold, "Should return None when symbol is None")


if __name__ == '__main__':
    unittest.main()
