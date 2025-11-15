#!/usr/bin/env python3
"""
Test Legacy System Validation
Tests: Legacy manifest validation, 42 features configuration
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_legacy_manifest_exists():
    """Test 1.1: Verify legacy manifest exists and is valid"""
    print("\n🧪 Test 1.1: Legacy Manifest Validation...")
    
    manifest_path = Path('artifacts/legacy/manifest.json')
    if not manifest_path.exists():
        print(f"❌ Legacy manifest not found at {manifest_path}")
        return False
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        print('📋 Legacy Manifest:')
        print(f'  Version: {manifest.get("version")}')
        print(f'  Feature Count: {manifest.get("feature_count")}')
        print(f'  Mode: {manifest.get("mode")}')
        
        # Validate feature count
        if manifest.get('feature_count') != 42:
            print(f'❌ Expected 42 features, got {manifest.get("feature_count")}')
            return False
        
        # Validate mode
        if manifest.get('mode') != 'legacy':
            print(f'❌ Expected legacy mode, got {manifest.get("mode")}')
            return False
        
        # Validate feature names
        feature_names = manifest.get('feature_names_ordered', [])
        if len(feature_names) != 42:
            print(f'❌ Expected 42 feature names, got {len(feature_names)}')
            return False
        
        print('✅ Legacy manifest validation PASSED')
        return True
        
    except Exception as e:
        print(f'❌ Error loading manifest: {e}')
        return False


def test_legacy_manifest_structure():
    """Test manifest has all required fields"""
    print("\n🧪 Testing Legacy Manifest Structure...")
    
    manifest_path = Path('artifacts/legacy/manifest.json')
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Check required fields
        required_fields = [
            'version', 'feature_count', 'feature_names_ordered',
            'mode', 'selected_features_price', 'selected_features_regime'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in manifest:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        
        # Validate selected features
        selected_price = manifest['selected_features_price']
        selected_regime = manifest['selected_features_regime']
        
        if len(selected_price) != 42:
            print(f"❌ Expected 42 selected price features, got {len(selected_price)}")
            return False
        
        if len(selected_regime) != 42:
            print(f"❌ Expected 42 selected regime features, got {len(selected_regime)}")
            return False
        
        print("✅ Legacy manifest structure validation PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error validating manifest structure: {e}")
        return False


def run_all_tests():
    """Run all legacy system tests"""
    print("="*60)
    print("🧪 Testing Legacy System Configuration...")
    print("="*60)
    
    results = {
        'manifest_exists': test_legacy_manifest_exists(),
        'manifest_structure': test_legacy_manifest_structure()
    }
    
    print("\n" + "="*60)
    print("📊 Test Results:")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All Legacy System tests PASSED")
    else:
        print("\n❌ Some Legacy System tests FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
