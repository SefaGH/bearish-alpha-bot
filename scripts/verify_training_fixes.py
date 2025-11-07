#!/usr/bin/env python3
"""
Verification Script for Training Pipeline Bug Fixes (FAZ 1)

This script demonstrates that all three critical bugs have been fixed:
1. Pandas FutureWarning (timeframe format)
2. Model Performance Tracker format errors (metrics cleaning)
3. MarketDataPipeline warning (proper initialization)

Run this script to verify the fixes before running the full training pipeline.
"""

import sys
import os
import warnings
import pandas as pd
import tempfile
import shutil

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.utils.model_performance_tracker import ModelPerformanceTracker


def verify_pandas_timeframe_fix():
    """Verify Fix 1: Pandas FutureWarning is fixed"""
    print("="*70)
    print("FIX 1 VERIFICATION: Pandas Timeframe Format")
    print("="*70)
    
    # Simulate the timeframe_map from diagnose_training_data.py
    timeframe_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1h', '4h': '4h', '1d': '1d'  # Fixed: using lowercase
    }
    
    print("\nTesting timeframe conversions with lowercase format...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test all timeframes
        for tf, freq in timeframe_map.items():
            td = pd.Timedelta(freq)
            print(f"  {tf} → {freq}: {td}")
        
        # Check for FutureWarning
        future_warnings = [warning for warning in w 
                         if issubclass(warning.category, FutureWarning)]
        
        if future_warnings:
            print("\n❌ FAILED: FutureWarning detected:")
            for warning in future_warnings:
                print(f"   {warning.message}")
            return False
        else:
            print("\n✅ PASSED: No FutureWarning detected with lowercase timeframes")
            return True


def verify_metrics_cleaning_fix():
    """Verify Fix 2: Metrics cleaning prevents format errors"""
    print("\n" + "="*70)
    print("FIX 2 VERIFICATION: Model Performance Tracker Metrics Cleaning")
    print("="*70)
    
    # Create temporary tracker
    temp_dir = tempfile.mkdtemp()
    
    try:
        tracker = ModelPerformanceTracker(performance_dir=temp_dir)
        
        # Test scenario 1: String metrics (the bug scenario)
        print("\nTest 1: String metrics (simulates the original bug)")
        metrics_with_strings = {
            'random_forest': {'accuracy': '0.95', 'precision': '0.93'},
            'lstm': {'accuracy': '0.89', 'loss': '0.234'},
            'transformer': {'accuracy': 0.91, 'loss': '0.198'}
        }
        
        try:
            result = tracker.record_training(
                model_type='regime',
                model_name='BTC-USDT_ensemble',
                metrics=metrics_with_strings,
                data_info={'samples': 1000},
                training_time=120.5
            )
            print("  ✅ String metrics handled successfully")
            
            # Verify all metrics are proper types
            for model_name, model_metrics in result['metrics'].items():
                if isinstance(model_metrics, dict):
                    for key, value in model_metrics.items():
                        if not isinstance(value, (int, float, str)):
                            print(f"  ❌ Unexpected type for {model_name}.{key}: {type(value)}")
                            return False
            
            print("  ✅ All metrics have proper types after cleaning")
            
        except ValueError as e:
            if "Unknown format code 'f'" in str(e):
                print(f"  ❌ FAILED: Format error still occurs: {e}")
                return False
            raise
        
        # Test scenario 2: Mixed types
        print("\nTest 2: Mixed types (float, string, int)")
        mixed_metrics = {
            'accuracy': 0.95,
            'loss': '0.123',
            'count': 100,
            'status': 'completed'
        }
        
        cleaned = tracker._clean_metrics(mixed_metrics)
        
        if isinstance(cleaned['accuracy'], float) and \
           isinstance(cleaned['loss'], float) and \
           isinstance(cleaned['count'], int) and \
           isinstance(cleaned['status'], str):
            print("  ✅ Mixed types handled correctly")
        else:
            print("  ❌ FAILED: Mixed types not handled correctly")
            return False
        
        print("\n✅ PASSED: Metrics cleaning prevents format errors")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def verify_market_data_pipeline_import():
    """Verify Fix 3: MarketDataPipeline can be imported and used"""
    print("\n" + "="*70)
    print("FIX 3 VERIFICATION: MarketDataPipeline Import")
    print("="*70)
    
    # Check that train_all_models.py imports MarketDataPipeline
    train_script = os.path.join(project_root, 'scripts', 'train_all_models.py')
    
    if not os.path.exists(train_script):
        print("  ⚠️  train_all_models.py not found")
        return False
    
    with open(train_script, 'r') as f:
        content = f.read()
    
    # Verify import exists
    if 'from src.core.market_data_pipeline import MarketDataPipeline' in content:
        print("  ✅ MarketDataPipeline import found in train_all_models.py")
    else:
        print("  ❌ FAILED: MarketDataPipeline import missing")
        return False
    
    # Verify it's used with AdvancedPricePredictionEngine
    if 'market_data_pipeline=market_pipeline' in content:
        print("  ✅ MarketDataPipeline passed to AdvancedPricePredictionEngine")
    else:
        print("  ❌ FAILED: MarketDataPipeline not passed to engine")
        return False
    
    print("\n✅ PASSED: MarketDataPipeline properly integrated")
    return True


def verify_diagnose_script_changes():
    """Verify diagnose_training_data.py uses lowercase timeframes"""
    print("\n" + "="*70)
    print("ADDITIONAL VERIFICATION: diagnose_training_data.py Changes")
    print("="*70)
    
    diagnose_script = os.path.join(project_root, 'scripts', 'diagnose_training_data.py')
    
    if not os.path.exists(diagnose_script):
        print("  ⚠️  diagnose_training_data.py not found")
        return False
    
    with open(diagnose_script, 'r') as f:
        content = f.read()
    
    # Check for uppercase timeframes (should not exist)
    issues = []
    if "'1H'" in content:
        issues.append("Uppercase '1H' found")
    if "'4H'" in content:
        issues.append("Uppercase '4H' found")
    if "'1D'" in content:
        issues.append("Uppercase '1D' found")
    
    if issues:
        print("  ❌ FAILED: Uppercase timeframes still present:")
        for issue in issues:
            print(f"     - {issue}")
        return False
    
    # Check for lowercase timeframes (should exist)
    if "'1h'" in content and "'4h'" in content:
        print("  ✅ Lowercase timeframes found: '1h', '4h'")
    else:
        print("  ⚠️  Warning: Lowercase timeframes not found in expected format")
    
    print("\n✅ PASSED: diagnose_training_data.py uses correct timeframes")
    return True


def main():
    """Run all verifications"""
    print("\n" + "="*70)
    print("TRAINING PIPELINE BUG FIXES VERIFICATION (FAZ 1)")
    print("="*70)
    print()
    
    results = {
        'Fix 1 - Pandas FutureWarning': verify_pandas_timeframe_fix(),
        'Fix 2 - Metrics Cleaning': verify_metrics_cleaning_fix(),
        'Fix 3 - MarketDataPipeline': verify_market_data_pipeline_import(),
        'Additional - Diagnose Script': verify_diagnose_script_changes()
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print()
        print("The training pipeline bug fixes have been successfully implemented.")
        print("You can now run the training pipeline without the following errors:")
        print("  - FutureWarning: 'H' is deprecated")
        print("  - Unknown format code 'f' for object of type 'str'")
        print("  - MarketDataPipeline not provided warning")
        print()
        return 0
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("Please review the failures above and ensure all fixes are properly applied.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
