#!/usr/bin/env python3
"""
Demonstration script for FeatureEngineeringPipeline array parsing fix.

This script shows that the bug is fixed and the pipeline can now handle
various input formats for volatility_windows and momentum_windows.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.feature_engineering import FeatureEngineeringPipeline

def test_format(name, config):
    """Test a specific format and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    try:
        pipeline = FeatureEngineeringPipeline(config)
        print(f"✅ SUCCESS: Pipeline initialized")
        print(f"   - volatility_features: {type(pipeline.volatility_features).__name__}")
        print(f"   - momentum_features: {type(pipeline.momentum_features).__name__}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def main():
    """Run demonstration of all supported formats."""
    print("\n" + "="*60)
    print("FeatureEngineeringPipeline Array Parsing Fix Demonstration")
    print("="*60)
    
    test_cases = [
        ("Plain CSV format", {
            'volatility_windows': '5,10,20,50',
            'momentum_windows': '5,10,20,50'
        }),
        ("Bracket format (GitHub Actions issue)", {
            'volatility_windows': '[5,10,20,50]',
            'momentum_windows': '[5,10,20,50]'
        }),
        ("Single-quoted format", {
            'volatility_windows': "['5','10','20','50']",
            'momentum_windows': "['5','10','20','50']"
        }),
        ("Double-quoted format", {
            'volatility_windows': '["5","10","20","50"]',
            'momentum_windows': '["5","10","20","50"]'
        }),
        ("Bracket with spaces", {
            'volatility_windows': '[5, 10, 20, 50]',
            'momentum_windows': '[5, 10, 20, 50]'
        }),
        ("Extra spaces format", {
            'volatility_windows': ' [ 5 , 10 , 20 , 50 ] ',
            'momentum_windows': ' [ 5 , 10 , 20 , 50 ] '
        }),
        ("List input (already parsed)", {
            'volatility_windows': [5, 10, 20, 50],
            'momentum_windows': [5, 10, 20, 50]
        }),
        ("Empty config (use defaults)", {}),
    ]
    
    results = []
    for name, config in test_cases:
        success = test_format(name, config)
        results.append((name, success))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All formats are supported!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
