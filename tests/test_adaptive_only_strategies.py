"""
Test for verifying that only adaptive strategies are used in production coordinator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from unittest.mock import MagicMock, patch
import yaml


class TestAdaptiveOnlyStrategies(unittest.TestCase):
    """Test that production coordinator uses only adaptive strategies."""
    
    def test_imports_use_adaptive_strategies(self):
        """Verify that production_coordinator imports adaptive strategies."""
        # Read the production coordinator file
        coord_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'production_coordinator.py')
        with open(coord_path, 'r') as f:
            content = f.read()
        
        # Check for adaptive imports
        self.assertIn('from strategies.adaptive_ob import AdaptiveOversoldBounce', content,
                     "AdaptiveOversoldBounce should be imported")
        self.assertIn('from strategies.adaptive_str import AdaptiveShortTheRip', content,
                     "AdaptiveShortTheRip should be imported")
        
        # Check that old imports are NOT present
        self.assertNotIn('from strategies.oversold_bounce import OversoldBounce', content,
                        "Old OversoldBounce import should be removed")
        self.assertNotIn('from strategies.short_the_rip import ShortTheRip', content,
                        "Old ShortTheRip import should be removed")
    
    def test_strategy_instantiation_uses_adaptive(self):
        """Verify that adaptive strategies are instantiated in production coordinator."""
        coord_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'production_coordinator.py')
        with open(coord_path, 'r') as f:
            content = f.read()
        
        # Check for adaptive strategy instantiation
        self.assertIn('AdaptiveOversoldBounce(', content,
                     "AdaptiveOversoldBounce should be instantiated")
        self.assertIn('AdaptiveShortTheRip(', content,
                     "AdaptiveShortTheRip should be instantiated")
        
        # Check that MarketRegimeAnalyzer is used
        self.assertIn('MarketRegimeAnalyzer()', content,
                     "MarketRegimeAnalyzer should be instantiated")
    
    def test_config_has_adaptive_parameters(self):
        """Verify that config file has adaptive parameters."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.example.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        signals = config.get('signals', {})
        
        # Check oversold_bounce config
        ob = signals.get('oversold_bounce', {})
        self.assertEqual(ob.get('enable'), True, "oversold_bounce should be enabled")
        self.assertEqual(ob.get('adaptive_rsi_base'), 30, "adaptive_rsi_base should be 30")
        self.assertEqual(ob.get('adaptive_rsi_range'), 15, "adaptive_rsi_range should be 15")
        self.assertEqual(ob.get('ignore_regime'), False, "ignore_regime should be False for production")
        
        # Check short_the_rip config
        str_cfg = signals.get('short_the_rip', {})
        self.assertEqual(str_cfg.get('enable'), True, "short_the_rip should be enabled")
        self.assertEqual(str_cfg.get('adaptive_rsi_base'), 70, "adaptive_rsi_base should be 70")
        self.assertEqual(str_cfg.get('adaptive_rsi_range'), 15, "adaptive_rsi_range should be 15")
        self.assertEqual(str_cfg.get('ignore_regime'), False, "ignore_regime should be False for production")
    
    def test_adaptive_str_safe_data_handling(self):
        """Verify that AdaptiveShortTheRip handles missing data safely."""
        from strategies.adaptive_str import AdaptiveShortTheRip
        import pandas as pd
        
        cfg = {
            'rsi_min': 70,
            'tp_pct': 0.012,
            'sl_atr_mult': 1.0
        }
        strategy = AdaptiveShortTheRip(cfg)
        
        # Test empty dataframe
        result = strategy.signal(pd.DataFrame(), None)
        self.assertIsNone(result, "Empty dataframe should return None")
        
        # Test None dataframe
        result = strategy.signal(None, None)
        self.assertIsNone(result, "None dataframe should return None")
        
        # Test dataframe without required columns
        df_incomplete = pd.DataFrame({'close': [100.0, 101.0]})
        result = strategy.signal(df_incomplete, None)
        self.assertIsNone(result, "Incomplete dataframe should return None")
    
    def test_adaptive_ob_safe_data_handling(self):
        """Verify that AdaptiveOversoldBounce handles missing data safely."""
        from strategies.adaptive_ob import AdaptiveOversoldBounce
        import pandas as pd
        
        cfg = {
            'rsi_max': 30,
            'tp_pct': 0.015,
            'sl_atr_mult': 1.0
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        # Test empty dataframe
        result = strategy.signal(pd.DataFrame())
        self.assertIsNone(result, "Empty dataframe should return None")


if __name__ == '__main__':
    unittest.main()
