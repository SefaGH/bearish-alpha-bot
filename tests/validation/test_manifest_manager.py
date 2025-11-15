#!/usr/bin/env python3
"""
Test ManifestManager functionality
Tests: Singleton pattern, thread safety, feature mapping
"""
import threading
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.manifest_manager import ManifestManager


def test_singleton_pattern():
    """Test singleton pattern works correctly"""
    print("\n🧪 Testing Singleton Pattern...")
    
    mgr1 = ManifestManager()
    mgr2 = ManifestManager()
    
    assert mgr1 is mgr2, "❌ Singleton pattern failed"
    print("✅ Singleton pattern test PASSED")
    return True


def test_thread_safety():
    """Test thread-safe manifest loading"""
    print("\n🧪 Testing Thread Safety...")
    
    results = []
    errors = []
    
    def load_manifest_thread(thread_id):
        try:
            mgr = ManifestManager()
            manifest = mgr.load_manifest('artifacts/legacy')
            results.append({
                'thread': thread_id,
                'feature_count': manifest['feature_count']
            })
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    # Create multiple threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=load_manifest_thread, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Validate results
    if errors:
        print(f"❌ Thread errors: {errors}")
        return False
    
    if len(results) != 10:
        print(f"❌ Expected 10 results, got {len(results)}")
        return False
    
    # All should have same feature count
    feature_counts = [r['feature_count'] for r in results]
    if len(set(feature_counts)) != 1:
        print(f"❌ Inconsistent feature counts: {set(feature_counts)}")
        return False
    
    print(f"✅ Thread safety test PASSED (10 threads, all got {feature_counts[0]} features)")
    return True


def test_feature_name_mapping():
    """Test feature name to index mapping"""
    print("\n🧪 Testing Feature Name Mapping...")
    
    mgr = ManifestManager()
    manifest = mgr.load_manifest('artifacts/legacy')
    
    # Test feature name retrieval
    selected_features = mgr.get_selected_features('price')
    if len(selected_features) == 0:
        print("❌ No selected features found")
        return False
    
    print(f"✅ Feature mapping test PASSED ({len(selected_features)} features)")
    return True


def run_all_tests():
    """Run all ManifestManager tests"""
    print("="*60)
    print("🧪 Testing ManifestManager...")
    print("="*60)
    
    results = {
        'singleton_pattern': test_singleton_pattern(),
        'thread_safety': test_thread_safety(),
        'feature_name_mapping': test_feature_name_mapping()
    }
    
    print("\n" + "="*60)
    print("📊 Test Results:")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All ManifestManager tests PASSED")
    else:
        print("\n❌ Some ManifestManager tests FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
