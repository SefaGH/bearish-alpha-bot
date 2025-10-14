#!/usr/bin/env python3
"""
Integration test that simulates the live trading launcher initialization flow.

This test verifies that the exact code patterns from the launcher work correctly
after applying the Phase 2 fixes.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


async def test_launcher_initialization_flow():
    """Test the complete initialization flow as it appears in the launcher."""
    
    print("\n" + "="*70)
    print("SIMULATING LIVE TRADING LAUNCHER INITIALIZATION FLOW")
    print("="*70)
    
    try:
        # Simulate the imports from live_trading_launcher.py
        print("\n[Step 1] Testing imports from live_trading_launcher.py...")
        from ml.price_predictor import (
            AdvancedPricePredictionEngine, 
            MultiTimeframePricePredictor,
            EnsemblePricePredictor
        )
        from ml.regime_predictor import MLRegimePredictor
        from ml.strategy_optimizer import StrategyOptimizer
        from config.optimization_config import OptimizationConfiguration
        print("  ✓ All imports successful")
        
        # Simulate Phase 4.4: Price Prediction initialization
        print("\n[Step 2] Phase 4.4: Initializing price prediction engine...")
        print("  - Creating ensemble models for multiple timeframes...")
        
        # This is the exact code from the fixed launcher
        models = {
            '5m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '15m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '1h': EnsemblePricePredictor({'lstm': None, 'transformer': None})
        }
        print(f"  ✓ Created {len(models)} ensemble models")
        
        multi_timeframe_predictor = MultiTimeframePricePredictor(models)
        print("  ✓ MultiTimeframePricePredictor initialized")
        
        # Correct initialization with required parameter (the fix!)
        price_engine = AdvancedPricePredictionEngine(multi_timeframe_predictor)
        print("  ✓ AdvancedPricePredictionEngine initialized")
        
        # Simulate Phase 4.3: Strategy Optimizer initialization
        print("\n[Step 3] Phase 4.3: Initializing strategy optimizer...")
        config = OptimizationConfiguration.get_default_config()
        strategy_optimizer = StrategyOptimizer(config)
        print("  ✓ Strategy Optimizer initialized")
        
        # Simulate Phase 4.1: ML Regime Predictor initialization
        print("\n[Step 4] Phase 4.1: Initializing regime predictor...")
        regime_predictor = MLRegimePredictor()
        print("  ✓ ML Regime Predictor initialized")
        
        # Simulate strategy registration
        print("\n[Step 5] Simulating strategy registration...")
        from core.portfolio_manager import PortfolioManager
        from core.risk_manager import RiskManager
        from core.performance_monitor import RealTimePerformanceMonitor
        
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register test strategies
        strategies = {
            'adaptive_ob': Mock(),
            'adaptive_str': Mock()
        }
        
        allocation_per_strategy = 1.0 / len(strategies)
        print(f"  - Registering {len(strategies)} strategies with {allocation_per_strategy:.1%} allocation each")
        
        for strategy_name, strategy_instance in strategies.items():
            result = portfolio_manager.register_strategy(
                strategy_name=strategy_name,
                strategy_instance=strategy_instance,
                initial_allocation=allocation_per_strategy
            )
            
            # This is the exact fixed code from the launcher
            if result.get('status') == 'success':
                print(f"  ✓ {strategy_name}: {allocation_per_strategy:.1%} allocation")
            else:
                print(f"  ⚠ Failed to register {strategy_name}: {result.get('reason')}")
                return False
        
        print("\n" + "="*70)
        print("✅ LAUNCHER INITIALIZATION FLOW COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nAll critical components initialized without errors:")
        print("  ✓ ML Price Prediction Engine (with required parameters)")
        print("  ✓ Strategy Optimizer")
        print("  ✓ ML Regime Predictor")
        print("  ✓ Strategy Registration (with correct result checking)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INITIALIZATION FLOW FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_scenarios():
    """Test that the old broken code patterns would fail."""
    
    print("\n" + "="*70)
    print("TESTING ERROR SCENARIOS (OLD BROKEN CODE)")
    print("="*70)
    
    # Test 1: Old broken AdvancedPricePredictionEngine initialization
    print("\n[Error Scenario 1] Old code: AdvancedPricePredictionEngine()")
    from ml.price_predictor import AdvancedPricePredictionEngine
    
    try:
        engine = AdvancedPricePredictionEngine()
        print("  ✗ UNEXPECTED: Old code didn't fail!")
        return False
    except TypeError as e:
        if 'multi_timeframe_predictor' in str(e):
            print(f"  ✓ Expected TypeError: {e}")
        else:
            print(f"  ✗ Wrong error: {e}")
            return False
    
    # Test 2: Old broken strategy registration result checking
    print("\n[Error Scenario 2] Old code: result['success']")
    from core.portfolio_manager import PortfolioManager
    from core.risk_manager import RiskManager
    from core.performance_monitor import RealTimePerformanceMonitor
    from unittest.mock import Mock
    
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    
    result = portfolio_manager.register_strategy(
        strategy_name='test',
        strategy_instance=Mock(),
        initial_allocation=0.5
    )
    
    try:
        success = result['success']  # This should fail
        print("  ✗ UNEXPECTED: Old code didn't fail!")
        return False
    except KeyError as e:
        if 'success' in str(e):
            print(f"  ✓ Expected KeyError: {e}")
        else:
            print(f"  ✗ Wrong error: {e}")
            return False
    
    print("\n✅ Both old broken patterns correctly identified as errors")
    return True


async def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("PHASE 2 INITIALIZATION FIXES - INTEGRATION TEST")
    print("="*70)
    
    # Test the complete initialization flow
    flow_success = await test_launcher_initialization_flow()
    
    # Test error scenarios
    error_success = await test_error_scenarios()
    
    # Final summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    if flow_success and error_success:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        print("\nThe fixes successfully resolve both critical errors:")
        print("  1. AdvancedPricePredictionEngine now receives required parameter")
        print("  2. Strategy registration uses correct 'status' key")
        print("\nThe launcher should now complete all 8 initialization phases.")
        return 0
    else:
        print("\n❌ SOME INTEGRATION TESTS FAILED!")
        print(f"  Initialization Flow: {'PASS' if flow_success else 'FAIL'}")
        print(f"  Error Scenarios: {'PASS' if error_success else 'FAIL'}")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
