#!/usr/bin/env python3
"""
Test Feature Engineering Dynamic Loading
Tests: Legacy feature extraction, GEMMA feature extraction, consistency
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.manifest_manager import ManifestManager


def test_feature_extraction_legacy():
    """Test feature extraction with legacy manifest"""
    print("\n🧪 Testing Legacy Feature Extraction...")
    
    try:
        # Configure for legacy
        config = {
            'models': {'active_bundle': 'artifacts/legacy'},
            'gemma_enabled': False
        }
        
        # Initialize
        fe = FeatureEngineeringPipeline(config)
        
        # Check expected feature count
        if fe.expected_feature_count != 42:
            print(f"❌ Expected 42 features, got {fe.expected_feature_count}")
            return False
        
        # Create sample data
        df = pd.DataFrame({
            'open': np.random.randn(100) * 100 + 50000,
            'high': np.random.randn(100) * 100 + 50100,
            'low': np.random.randn(100) * 100 + 49900,
            'close': np.random.randn(100) * 100 + 50000,
            'volume': np.random.randn(100) * 1000 + 10000
        })
        
        # Ensure positive values
        df = df.abs() + 1000
        
        # Extract features
        features = fe.extract_features(df, mode='price')
        
        # Validate
        if features.shape[1] != 42:
            print(f"❌ Feature extraction returned {features.shape[1]} features, expected 42")
            return False
        
        print(f"✅ Legacy extraction: {features.shape[1]} features")
        return True
        
    except Exception as e:
        print(f"❌ Legacy feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction_gemma():
    """Test feature extraction with GEMMA manifest (if available)"""
    print("\n🧪 Testing GEMMA Feature Extraction...")
    
    try:
        # Check if GEMMA bundle exists
        gemma_bundles = list(Path('artifacts').glob('gemma_run_*'))
        
        if not gemma_bundles:
            print("⏭️ No GEMMA bundles found, skipping")
            return True
        
        latest_bundle = max(gemma_bundles, key=lambda p: p.stat().st_mtime)
        
        # Configure for GEMMA
        config = {
            'models': {'active_bundle': str(latest_bundle)},
            'gemma_enabled': True
        }
        
        # Initialize
        fe = FeatureEngineeringPipeline(config)
        
        # Load manifest to get expected count
        mgr = ManifestManager()
        manifest = mgr.load_manifest(str(latest_bundle))
        expected_count = manifest['feature_count']
        
        # Create sample data
        df = pd.DataFrame({
            'open': np.random.randn(100) * 100 + 50000,
            'high': np.random.randn(100) * 100 + 50100,
            'low': np.random.randn(100) * 100 + 49900,
            'close': np.random.randn(100) * 100 + 50000,
            'volume': np.random.randn(100) * 1000 + 10000
        })
        
        # Ensure positive values
        df = df.abs() + 1000
        
        # Extract features
        features = fe.extract_features(df, mode='price')
        
        # Validate
        if features.shape[1] != expected_count:
            print(f"❌ GEMMA extraction returned {features.shape[1]} features, expected {expected_count}")
            return False
        
        print(f"✅ GEMMA extraction: {features.shape[1]} features")
        return True
        
    except Exception as e:
        print(f"❌ GEMMA feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_consistency():
    """Test feature extraction consistency"""
    print("\n🧪 Testing Feature Consistency...")
    
    try:
        config = {'models': {'active_bundle': 'artifacts/legacy'}}
        fe = FeatureEngineeringPipeline(config)
        
        # Create sample data
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [1000.0, 1100.0, 1200.0]
        })
        
        # Need more data points for technical indicators
        # Repeat and add noise
        df_large = pd.concat([df] * 50, ignore_index=True)
        df_large['close'] = df_large['close'] + np.random.randn(len(df_large)) * 0.1
        
        # Extract multiple times
        features1 = fe.extract_features(df_large, mode='price')
        features2 = fe.extract_features(df_large, mode='price')
        
        # Should be identical for same input
        if not features1.equals(features2):
            print("❌ Inconsistent feature extraction")
            return False
        
        print("✅ Feature consistency verified")
        return True
        
    except Exception as e:
        print(f"❌ Feature consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all feature engineering tests"""
    print("="*60)
    print("🧪 Testing Feature Engineering...")
    print("="*60)
    
    results = {
        'legacy_extraction': test_feature_extraction_legacy(),
        'gemma_extraction': test_feature_extraction_gemma(),
        'feature_consistency': test_feature_consistency()
    }
    
    print("\n" + "="*60)
    print("📊 Test Results:")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All Feature Engineering tests PASSED")
    else:
        print("\n❌ Some Feature Engineering tests FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
