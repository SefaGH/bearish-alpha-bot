#!/usr/bin/env python3
"""
Quick validation script for Feature Analysis Tool

This script demonstrates the complete workflow of the feature analyzer
and validates that all outputs are generated correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.analyze_features import FeatureAnalyzer


def validate_feature_analyzer():
    """Validate the feature analyzer with test data."""
    print("="*70)
    print("FEATURE ANALYZER VALIDATION")
    print("="*70)
    
    # Check if test data exists
    data_path = "data/cache/BTC-USDT_training_data.npz"
    if not Path(data_path).exists():
        print(f"❌ Test data not found at {data_path}")
        print("   Run: python scripts/create_test_training_data.py")
        return False
    
    print(f"✅ Test data found: {data_path}\n")
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer(
        data_path=data_path,
        variance_threshold=0.01,
        correlation_threshold=0.05
    )
    
    # Load data
    print("Step 1: Loading data...")
    if not analyzer.load_data():
        print("❌ Failed to load data")
        return False
    print(f"✅ Loaded {analyzer.n_samples} samples, {analyzer.n_features} features\n")
    
    # Analyze variance
    print("Step 2: Analyzing variance...")
    var_result = analyzer.analyze_variance()
    print(f"✅ Found {var_result['low_variance_count']} low-variance features\n")
    
    # Analyze correlations
    print("Step 3: Analyzing correlations...")
    corr_result = analyzer.analyze_correlations()
    print(f"✅ Found {corr_result['weak_count']} weak features, {corr_result['strong_count']} strong features\n")
    
    # Select features
    print("Step 4: Selecting features...")
    select_result = analyzer.select_features()
    print(f"✅ Selected {select_result['selected_count']}/{analyzer.n_features} features\n")
    
    # Save feature mask
    print("Step 5: Saving feature mask...")
    if not analyzer.save_feature_mask():
        print("❌ Failed to save feature mask")
        return False
    print("✅ Saved mask and metadata\n")
    
    # Generate report
    print("Step 6: Generating report...")
    if not analyzer.generate_report():
        print("❌ Failed to generate report")
        return False
    print("✅ Generated report\n")
    
    # Verify outputs
    print("Step 7: Verifying outputs...")
    outputs = [
        ("Feature mask", "data/cache/feature_selection_mask.npy"),
        ("Metadata", "data/cache/feature_selection_metadata.json"),
        ("Report", "logs/feature_analysis_report.md")
    ]
    
    all_exist = True
    for name, path in outputs:
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} NOT FOUND")
            all_exist = False
    
    print("\n" + "="*70)
    if all_exist:
        print("✅ VALIDATION PASSED - All outputs generated successfully!")
        print("="*70)
        return True
    else:
        print("❌ VALIDATION FAILED - Some outputs are missing")
        print("="*70)
        return False


if __name__ == '__main__':
    success = validate_feature_analyzer()
    sys.exit(0 if success else 1)
