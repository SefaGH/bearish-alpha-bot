#!/usr/bin/env python3
"""
Test suite for live trading launcher initialization fixes.

Verifies that all critical components can be initialized with proper parameters.
"""

import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestInitializationFixes(unittest.TestCase):
    """Test that initialization errors are fixed."""
    
    def test_strategy_optimizer_initialization(self):
        """Test StrategyOptimizer can be initialized with config."""
        from ml.strategy_optimizer import StrategyOptimizer
        from config.optimization_config import OptimizationConfiguration
        
        # Create config as specified in the fix
        config = OptimizationConfiguration.get_default_config()
        
        # Should not raise TypeError
        optimizer = StrategyOptimizer(config)
        
        # Verify initialization
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.config)
        self.assertEqual(optimizer.optimization_history, [])
    
    def test_adaptive_oversold_bounce_initialization(self):
        """Test AdaptiveOversoldBounce can be initialized with cfg and regime_analyzer."""
        from strategies.adaptive_ob import AdaptiveOversoldBounce
        from core.market_regime import MarketRegimeAnalyzer
        
        # Create regime analyzer
        regime_analyzer = MarketRegimeAnalyzer()
        
        # Create config as specified in the fix
        adaptive_ob_config = {
            'rsi_max': 30,
            'tp_pct': 0.015,
            'sl_atr_mult': 1.0
        }
        
        # Should not raise TypeError
        strategy = AdaptiveOversoldBounce(adaptive_ob_config, regime_analyzer)
        
        # Verify initialization
        self.assertIsNotNone(strategy)
        self.assertIsNotNone(strategy.regime_analyzer)
        self.assertEqual(strategy.base_cfg, adaptive_ob_config)
    
    def test_adaptive_short_the_rip_initialization(self):
        """Test AdaptiveShortTheRip can be initialized with cfg and regime_analyzer."""
        from strategies.adaptive_str import AdaptiveShortTheRip
        from core.market_regime import MarketRegimeAnalyzer
        
        # Create regime analyzer
        regime_analyzer = MarketRegimeAnalyzer()
        
        # Create config as specified in the fix
        adaptive_str_config = {
            'rsi_min': 70,
            'tp_pct': 0.012,
            'sl_atr_mult': 1.0
        }
        
        # Should not raise TypeError
        strategy = AdaptiveShortTheRip(adaptive_str_config, regime_analyzer)
        
        # Verify initialization
        self.assertIsNotNone(strategy)
        self.assertIsNotNone(strategy.regime_analyzer)
        self.assertEqual(strategy.base_cfg, adaptive_str_config)
    
    def test_market_regime_analyzer_initialization(self):
        """Test MarketRegimeAnalyzer can be initialized without parameters."""
        from core.market_regime import MarketRegimeAnalyzer
        
        # Should not raise any errors
        analyzer = MarketRegimeAnalyzer()
        
        # Verify initialization
        self.assertIsNotNone(analyzer)
        self.assertIn('trend', analyzer.current_regime)
        self.assertIn('volatility', analyzer.current_regime)
        self.assertIn('momentum', analyzer.current_regime)
    
    def test_optimization_configuration_factory(self):
        """Test OptimizationConfiguration factory methods."""
        from config.optimization_config import OptimizationConfiguration
        
        # Test get_default_config
        config = OptimizationConfiguration.get_default_config()
        self.assertIsNotNone(config)
        
        # Verify it has the expected attributes
        self.assertIsNotNone(config.genetic_algorithm)
        self.assertIsNotNone(config.neural_architecture_search)
        self.assertIsNotNone(config.multi_objective)
    
    def test_all_components_together(self):
        """Integration test: Initialize all components as in the launcher."""
        from ml.strategy_optimizer import StrategyOptimizer
        from config.optimization_config import OptimizationConfiguration
        from strategies.adaptive_ob import AdaptiveOversoldBounce
        from strategies.adaptive_str import AdaptiveShortTheRip
        from core.market_regime import MarketRegimeAnalyzer
        
        # Initialize StrategyOptimizer
        config = OptimizationConfiguration.get_default_config()
        optimizer = StrategyOptimizer(config)
        self.assertIsNotNone(optimizer)
        
        # Initialize regime analyzer
        regime_analyzer = MarketRegimeAnalyzer()
        self.assertIsNotNone(regime_analyzer)
        
        # Initialize strategies
        adaptive_ob_config = {
            'rsi_max': 30,
            'tp_pct': 0.015,
            'sl_atr_mult': 1.0
        }
        
        adaptive_str_config = {
            'rsi_min': 70,
            'tp_pct': 0.012,
            'sl_atr_mult': 1.0
        }
        
        ob_strategy = AdaptiveOversoldBounce(adaptive_ob_config, regime_analyzer)
        str_strategy = AdaptiveShortTheRip(adaptive_str_config, regime_analyzer)
        
        self.assertIsNotNone(ob_strategy)
        self.assertIsNotNone(str_strategy)
        
        # Verify both strategies use the same regime analyzer
        self.assertIs(ob_strategy.regime_analyzer, regime_analyzer)
        self.assertIs(str_strategy.regime_analyzer, regime_analyzer)


if __name__ == '__main__':
    unittest.main()
