#!/usr/bin/env python3
"""
Focused validation script for Phase 2 initialization fixes.

Tests the exact initialization steps that were failing:
1. AdvancedPricePredictionEngine initialization
2. Strategy registration result checking
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.price_predictor import (
    AdvancedPricePredictionEngine, 
    MultiTimeframePricePredictor,
    EnsemblePricePredictor
)

def test_fix_1_price_engine_initialization():
    """Test Fix 1: AdvancedPricePredictionEngine initialization with required parameter."""
    print("\n" + "="*70)
    print("Testing Fix 1: AdvancedPricePredictionEngine initialization")
    print("="*70)
    
    try:
        # This is how it was BEFORE the fix (should fail):
        print("\n❌ Testing OLD broken code: AdvancedPricePredictionEngine()")
        try:
            engine = AdvancedPricePredictionEngine()
            print("  UNEXPECTED: Old code didn't fail!")
        except TypeError as e:
            print(f"  ✓ Expected error: {e}")
        
        # This is how it is AFTER the fix (should succeed):
        print("\n✅ Testing NEW fixed code:")
        print("  - Creating EnsemblePricePredictor instances...")
        models = {
            '5m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '15m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
            '1h': EnsemblePricePredictor({'lstm': None, 'transformer': None})
        }
        print(f"  ✓ Created {len(models)} ensemble models")
        
        print("  - Creating MultiTimeframePricePredictor...")
        multi_timeframe_predictor = MultiTimeframePricePredictor(models)
        print("  ✓ MultiTimeframePricePredictor initialized")
        
        print("  - Creating AdvancedPricePredictionEngine with required parameter...")
        engine = AdvancedPricePredictionEngine(multi_timeframe_predictor)
        print("  ✓ AdvancedPricePredictionEngine initialized successfully!")
        
        print("\n✅ Fix 1 VERIFIED: Price engine initialization works correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Fix 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fix_2_strategy_registration_result():
    """Test Fix 2: Strategy registration result checking uses correct key."""
    print("\n" + "="*70)
    print("Testing Fix 2: Strategy registration result checking")
    print("="*70)
    
    try:
        from core.portfolio_manager import PortfolioManager
        from core.risk_manager import RiskManager
        from core.performance_monitor import RealTimePerformanceMonitor
        from unittest.mock import Mock
        
        # Initialize required components
        print("\n  - Initializing portfolio manager...")
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        print("  ✓ Portfolio manager initialized")
        
        # Register a mock strategy
        print("  - Registering test strategy...")
        mock_strategy = Mock()
        result = portfolio_manager.register_strategy(
            strategy_name='test_strategy',
            strategy_instance=mock_strategy,
            initial_allocation=0.25
        )
        print(f"  ✓ Strategy registered, result keys: {list(result.keys())}")
        
        # This is how it was BEFORE the fix (should fail):
        print("\n❌ Testing OLD broken code: result['success']")
        try:
            success = result['success']
            print(f"  UNEXPECTED: Old code didn't fail! Got: {success}")
        except KeyError as e:
            print(f"  ✓ Expected KeyError: {e}")
        
        # This is how it is AFTER the fix (should succeed):
        print("\n✅ Testing NEW fixed code: result.get('status') == 'success'")
        if result.get('status') == 'success':
            print("  ✓ Strategy registration check works correctly!")
        else:
            print(f"  ✗ Unexpected status: {result.get('status')}")
            return False
        
        print("\n✅ Fix 2 VERIFIED: Strategy registration result checking works correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Fix 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("PHASE 2 INITIALIZATION FIXES VALIDATION")
    print("="*70)
    print("\nValidating critical fixes from problem statement:")
    print("  1. AdvancedPricePredictionEngine missing parameter")
    print("  2. Strategy registration KeyError on 'success'")
    
    results = []
    
    # Test Fix 1
    results.append(test_fix_1_price_engine_initialization())
    
    # Test Fix 2
    results.append(test_fix_2_strategy_registration_result())
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if all(results):
        print("\n✅ ALL FIXES VERIFIED SUCCESSFULLY!")
        print("\nBoth critical initialization errors have been resolved:")
        print("  ✓ Fix 1: AdvancedPricePredictionEngine initialization")
        print("  ✓ Fix 2: Strategy registration result checking")
        print("\nThe bot should now pass initialization without errors.")
        return 0
    else:
        print("\n❌ SOME FIXES FAILED!")
        for i, result in enumerate(results, 1):
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  Fix {i}: {status}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
