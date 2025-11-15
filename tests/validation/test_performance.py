#!/usr/bin/env python3
"""
Performance Benchmark Tests for GEMMA Architecture
Tests: Feature extraction speed, memory usage, inference time
"""
import time
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

# Enable ML for testing
os.environ['ML_ENABLED'] = 'true'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def benchmark_feature_extraction():
    """Benchmark feature extraction performance"""
    print("\n⏱️  Benchmarking Feature Extraction...")
    
    try:
        from src.ml.feature_engineering import FeatureEngineeringPipeline
        
        config = {'models': {'active_bundle': 'artifacts/legacy'}}
        fe = FeatureEngineeringPipeline(config)
        
        # Create test data
        df = pd.DataFrame({
            'open': np.random.randn(1000) * 100 + 50000,
            'high': np.random.randn(1000) * 100 + 50100,
            'low': np.random.randn(1000) * 100 + 49900,
            'close': np.random.randn(1000) * 100 + 50000,
            'volume': np.random.randn(1000) * 1000 + 10000
        })
        
        # Ensure positive values
        df = df.abs() + 1000
        
        # Warm up
        _ = fe.extract_features(df)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            features = fe.extract_features(df)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Std Dev: {std_time:.2f} ms")
        print(f"  Min: {np.min(times):.2f} ms")
        print(f"  Max: {np.max(times):.2f} ms")
        
        # Check if meets target
        target = 50  # ms
        meets_target = avg_time < target
        if meets_target:
            print(f"  ✅ Meets target (<{target} ms)")
        else:
            print(f"  ⚠️  Exceeds target (>{target} ms)")
        
        return {
            'passed': meets_target,
            'avg_time': avg_time,
            'target': target
        }
        
    except Exception as e:
        print(f"  ❌ Feature extraction benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


def benchmark_memory_usage():
    """Benchmark memory usage"""
    print("\n💾 Benchmarking Memory Usage...")
    
    try:
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load all components
        from src.ml.manifest_manager import ManifestManager
        from src.ml.feature_engineering import FeatureEngineeringPipeline
        from src.ml.regime_predictor import MLRegimePredictor
        from src.ml.reinforcement_learning import TradingRLAgent
        
        config = {'models': {'active_bundle': 'artifacts/legacy'}}
        
        mgr = ManifestManager()
        fe = FeatureEngineeringPipeline(config)
        
        # Regime predictor requires feature pipeline
        regime_config = {'active_bundle': 'artifacts/legacy'}
        rp = MLRegimePredictor(fe, regime_config)
        
        # RL agent
        agent = TradingRLAgent(config={'active_bundle': 'artifacts/legacy'})
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Used: {memory_used:.2f} MB")
        
        # Check if meets target
        target = 2048  # MB
        meets_target = final_memory < target
        if meets_target:
            print(f"  ✅ Meets target (<{target} MB)")
        else:
            print(f"  ⚠️  Exceeds target (>{target} MB)")
        
        return {
            'passed': meets_target,
            'final_memory': final_memory,
            'target': target
        }
        
    except Exception as e:
        print(f"  ❌ Memory usage benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


def benchmark_gemma_inference():
    """Benchmark GEMMA inference if available"""
    print("\n⏱️  Benchmarking GEMMA Inference...")
    
    if not Path('data/models/gemma/final/gemma_price.pt').exists():
        print("  ⏭️  GEMMA models not found, skipping")
        return {'passed': True, 'skipped': True}
    
    try:
        from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
        
        config = {
            'feature_count': 82,
            'model_path': 'data/models/gemma/final/gemma_price.pt',
            'scaler_path': 'data/models/gemma/final/gemma_price_scaler.joblib',
            'shadow_mode': True
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Warm up
        dummy_features = {f'feature_{i}': np.random.randn() for i in range(82)}
        _ = adapter.predict(dummy_features)
        
        # Benchmark
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = adapter.predict(dummy_features)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        avg_time = np.mean(times)
        
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Std Dev: {np.std(times):.2f} ms")
        
        # Check if meets target
        target = 100  # ms
        meets_target = avg_time < target
        if meets_target:
            print(f"  ✅ Meets target (<{target} ms)")
        else:
            print(f"  ⚠️  Exceeds target (>{target} ms)")
        
        return {
            'passed': meets_target,
            'avg_time': avg_time,
            'target': target
        }
        
    except Exception as e:
        print(f"  ❌ GEMMA benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


def run_all_benchmarks():
    """Run all performance benchmarks"""
    print("="*60)
    print("🧪 Running Performance Benchmarks...")
    print("="*60)
    
    results = {
        'feature_extraction': benchmark_feature_extraction(),
        'memory_usage': benchmark_memory_usage(),
        'gemma_inference': benchmark_gemma_inference()
    }
    
    print("\n" + "="*60)
    print("📊 Performance Summary:")
    print("="*60)
    
    for benchmark, result in results.items():
        if result.get('skipped'):
            print(f"  ⏭️  {benchmark}: Skipped")
        elif result['passed']:
            print(f"  ✅ {benchmark}: Passed")
        else:
            print(f"  ❌ {benchmark}: Failed")
    
    # Overall pass if critical benchmarks pass
    critical_passed = (
        results['feature_extraction']['passed'] and 
        results['memory_usage']['passed']
    )
    
    if critical_passed:
        print("\n✅ All critical performance benchmarks PASSED")
    else:
        print("\n❌ Some critical performance benchmarks FAILED")
    
    return critical_passed


if __name__ == "__main__":
    success = run_all_benchmarks()
    sys.exit(0 if success else 1)
