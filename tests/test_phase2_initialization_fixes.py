#!/usr/bin/env python3
"""
Test suite for Phase 2 initialization fixes.

Verifies that:
1. AdvancedPricePredictionEngine can be initialized with required parameters
2. Strategy registration result checking uses correct dictionary key
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.price_predictor import (
    AdvancedPricePredictionEngine,
    MultiTimeframePricePredictor,
    EnsemblePricePredictor
)


class TestPhase2InitializationFixes(unittest.TestCase):
    """Test that Phase 2 initialization errors are fixed."""
    
    def test_ensemble_price_predictor_initialization(self):
        """Test EnsemblePricePredictor can be initialized with models dict."""
        models = {'lstm': None, 'transformer': None}
        predictor = EnsemblePricePredictor(models)
        
        self.assertIsNotNone(predictor)
        self.assertEqual(predictor.models, models)
    
    def test_multi_timeframe_predictor_initialization(self):
        """Test MultiTimeframePricePredictor can be initialized with ensemble models."""
        models = {
            '5m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '15m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '1h': EnsemblePricePredictor({'lstm': None, 'transformer': None})
        }
        predictor = MultiTimeframePricePredictor(models)
        
        self.assertIsNotNone(predictor)
        self.assertEqual(len(predictor.models), 3)
        self.assertIn('5m', predictor.models)
        self.assertIn('15m', predictor.models)
        self.assertIn('1h', predictor.models)
    
    def test_advanced_price_prediction_engine_initialization(self):
        """Test AdvancedPricePredictionEngine can be initialized with multi_timeframe_predictor."""
        # Create multi-timeframe predictor as in the fix
        models = {
            '5m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '15m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '1h': EnsemblePricePredictor({'lstm': None, 'transformer': None})
        }
        multi_timeframe_predictor = MultiTimeframePricePredictor(models)
        
        # Should not raise TypeError about missing parameter
        engine = AdvancedPricePredictionEngine(multi_timeframe_predictor)
        
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.predictor)
        self.assertEqual(engine.predictor, multi_timeframe_predictor)
    
    def test_advanced_price_prediction_engine_requires_parameter(self):
        """Test that AdvancedPricePredictionEngine raises error without required parameter."""
        with self.assertRaises(TypeError) as cm:
            engine = AdvancedPricePredictionEngine()
        
        self.assertIn('multi_timeframe_predictor', str(cm.exception))
    
    def test_strategy_registration_result_format(self):
        """Test that strategy registration result uses 'status' key."""
        from core.portfolio_manager import PortfolioManager
        from core.risk_manager import RiskManager
        from core.performance_monitor import RealTimePerformanceMonitor
        
        # Initialize required components
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Create a mock strategy
        mock_strategy = Mock()
        
        # Register strategy
        result = portfolio_manager.register_strategy(
            strategy_name='test_strategy',
            strategy_instance=mock_strategy,
            initial_allocation=0.25
        )
        
        # Verify result has 'status' key, not 'success' key
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'success')
        self.assertNotIn('success', result)
    
    def test_production_coordinator_register_strategy_result(self):
        """Test that ProductionCoordinator.register_strategy returns correct format."""
        from core.production_coordinator import ProductionCoordinator
        
        coordinator = ProductionCoordinator()
        
        # Test without initialization (should return with success=False)
        mock_strategy = Mock()
        result = coordinator.register_strategy(
            strategy_name='test_strategy',
            strategy_instance=mock_strategy,
            initial_allocation=0.25
        )
        
        # Result should have 'success' key from ProductionCoordinator
        # (it uses different format than portfolio_manager)
        self.assertIn('success', result)
        self.assertEqual(result['success'], False)
        self.assertIn('reason', result)
    
    def test_complete_initialization_workflow(self):
        """Integration test: Complete initialization workflow as in launcher."""
        # Step 1: Initialize price prediction components
        models = {
            '5m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '15m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '1h': EnsemblePricePredictor({'lstm': None, 'transformer': None})
        }
        multi_timeframe_predictor = MultiTimeframePricePredictor(models)
        price_engine = AdvancedPricePredictionEngine(multi_timeframe_predictor)
        
        self.assertIsNotNone(price_engine)
        self.assertIsNotNone(price_engine.predictor)
        
        # Step 2: Test strategy registration result checking
        from core.portfolio_manager import PortfolioManager
        from core.risk_manager import RiskManager
        from core.performance_monitor import RealTimePerformanceMonitor
        
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        mock_strategy = Mock()
        result = portfolio_manager.register_strategy(
            strategy_name='test_strategy',
            strategy_instance=mock_strategy,
            initial_allocation=0.25
        )
        
        # Verify the correct way to check success using 'status' key
        self.assertTrue(result.get('status') == 'success')
        
        # Verify the old way (using 'success' key) would fail
        with self.assertRaises(KeyError):
            _ = result['success']


if __name__ == '__main__':
    unittest.main()
