#!/usr/bin/env python3
"""
Test Component Dimension Compatibility
Tests: Regime predictor, RL agent, Price predictor, GEMMA adapter
"""
import numpy as np
import sys
import os
from pathlib import Path

# Enable ML for testing
os.environ['ML_ENABLED'] = 'true'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_regime_predictor():
    """Test regime predictor with dynamic dimensions"""
    print("\n🧪 Testing Regime Predictor...")
    
    try:
        from src.ml.regime_predictor import MLRegimePredictor
        from src.ml.feature_engineering import FeatureEngineeringPipeline
        
        config = {'models': {'active_bundle': 'artifacts/legacy'}}
        
        # Create feature pipeline first
        fe = FeatureEngineeringPipeline(config)
        
        # Create regime predictor with feature pipeline and config
        regime_config = {'active_bundle': 'artifacts/legacy'}
        rp = MLRegimePredictor(fe, regime_config)
        
        if rp.expected_features != 42:
            print(f"❌ Regime predictor expects {rp.expected_features}, should be 42")
            return False
        
        print(f"✅ Regime predictor initialized for {rp.expected_features} features")
        return True
        
    except Exception as e:
        print(f"❌ Regime predictor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rl_agent():
    """Test RL agent with dynamic state size"""
    print("\n🧪 Testing RL Agent...")
    
    try:
        from src.ml.reinforcement_learning import TradingRLAgent
        
        config = {'active_bundle': 'artifacts/legacy'}
        
        agent = TradingRLAgent(config=config)
        
        # Test with correct state size
        state = np.random.randn(agent.state_size)
        action = agent.act(state, training=False)
        
        if action not in [0, 1, 2]:
            print(f"❌ Invalid action: {action}")
            return False
        
        print(f"✅ RL Agent working with state_size={agent.state_size}")
        return True
        
    except Exception as e:
        print(f"❌ RL Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_price_predictor():
    """Test price predictor with dynamic dimensions"""
    print("\n🧪 Testing Price Predictor...")
    
    try:
        from src.ml.price_predictor import AdvancedPricePredictionEngine
        from src.ml.feature_engineering import FeatureEngineeringPipeline
        
        config = {'models': {'active_bundle': 'artifacts/legacy'}}
        
        # Create feature pipeline first
        fe = FeatureEngineeringPipeline(config)
        
        # Mock market data pipeline (can be None for initialization test)
        market_data_pipeline = None
        
        # Create price predictor config
        predictor_config = {'active_bundle': 'artifacts/legacy'}
        
        predictor = AdvancedPricePredictionEngine(market_data_pipeline, fe, predictor_config)
        
        # Use feature pipeline's feature count
        feature_count = fe.expected_feature_count
        print(f"✅ Price predictor initialized for {feature_count} features")
        return True
        
    except Exception as e:
        print(f"❌ Price predictor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemma_adapter():
    """Test GEMMA adapter if available"""
    print("\n🧪 Testing GEMMA Adapter...")
    
    # Check if GEMMA models exist
    gemma_model = Path('data/models/gemma/final/gemma_price.pt')
    if not gemma_model.exists():
        print("⏭️ GEMMA models not found, skipping")
        return True
    
    try:
        from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
        
        config = {
            'feature_count': 82,  # Or from manifest
            'model_path': str(gemma_model),
            'scaler_path': 'data/models/gemma/final/gemma_price_scaler.joblib',
            'shadow_mode': True
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Test prediction
        dummy_features = {f'feature_{i}': np.random.randn() for i in range(82)}
        result = adapter.predict(dummy_features)
        
        if 'price_confidence' not in result:
            print("❌ Missing price_confidence in result")
            return False
        
        print(f"✅ GEMMA adapter working in shadow mode")
        return True
        
    except Exception as e:
        print(f"❌ GEMMA adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all component compatibility tests"""
    print("="*60)
    print("🧪 Testing Component Compatibility...")
    print("="*60)
    
    results = {
        'regime_predictor': test_regime_predictor(),
        'rl_agent': test_rl_agent(),
        'price_predictor': test_price_predictor(),
        'gemma_adapter': test_gemma_adapter()
    }
    
    print("\n" + "="*60)
    print("📊 Component Test Results:")
    print("="*60)
    for component, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {component}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All component tests PASSED")
    else:
        print("\n❌ Some component tests FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
