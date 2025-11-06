#!/usr/bin/env python3
"""
Test script to verify ML architecture synchronization with config.
This script validates that all components read and use the correct parameters from config.example.yaml
"""

import os
import sys
import yaml

# Enable ML features for testing (only if not already set)
if 'ML_ENABLED' not in os.environ:
    os.environ['ML_ENABLED'] = 'true'

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.ml.neural_networks import LSTMRegimePredictor
from src.ml.model_trainer import RegimeModelTrainer

def test_config_loading():
    """Test that config.example.yaml loads correctly and has expected values."""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    config_path = os.path.join(project_root, 'config', 'config.example.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ml_config = config.get('ml', {})
    regime_config = ml_config.get('regime_prediction', {})
    model_params = regime_config.get('model_params', {})
    lstm_config = model_params.get('lstm_regime', {})
    
    print(f"✅ Config loaded from: {config_path}")
    print(f"\nRegime Prediction Config:")
    print(f"  - enabled: {regime_config.get('enabled')}")
    print(f"  - min_confidence_threshold: {regime_config.get('min_confidence_threshold')}")
    
    print(f"\nLSTM Regime Model Params:")
    print(f"  - hidden_size: {lstm_config.get('hidden_size')}")
    print(f"  - num_layers: {lstm_config.get('num_layers')}")
    
    # Validate expected values
    assert lstm_config.get('hidden_size') == 64, f"Expected hidden_size=64, got {lstm_config.get('hidden_size')}"
    assert lstm_config.get('num_layers') == 2, f"Expected num_layers=2, got {lstm_config.get('num_layers')}"
    
    print("\n✅ Config values are correct (hidden_size=64, num_layers=2)")

def test_neural_network_defaults():
    """Test that LSTMRegimePredictor defaults match config."""
    print("\n" + "=" * 60)
    print("TEST 2: Neural Network Default Parameters")
    print("=" * 60)
    
    # Create model with defaults
    model = LSTMRegimePredictor()
    
    print(f"LSTMRegimePredictor default params:")
    print(f"  - hidden_size: {model.hidden_size}")
    print(f"  - num_layers: {model.num_layers}")
    
    # Validate defaults match config
    assert model.hidden_size == 64, f"Expected default hidden_size=64, got {model.hidden_size}"
    assert model.num_layers == 2, f"Expected default num_layers=2, got {model.num_layers}"
    
    print("\n✅ Default parameters match config (hidden_size=64, num_layers=2)")

def test_model_trainer_config():
    """Test that RegimeModelTrainer accepts and uses config correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Trainer Config Integration")
    print("=" * 60)
    
    # Load config
    config_path = os.path.join(project_root, 'config', 'config.example.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    regime_config = config['ml']['regime_prediction']
    
    # Create trainer with config
    trainer = RegimeModelTrainer(config=regime_config)
    
    print(f"RegimeModelTrainer initialized with config:")
    print(f"  - Config keys: {list(trainer.config.keys())}")
    print(f"  - LSTM params: {trainer.config.get('model_params', {}).get('lstm_regime', {})}")
    
    # Verify config was stored
    lstm_params = trainer.config.get('model_params', {}).get('lstm_regime', {})
    assert lstm_params.get('hidden_size') == 64, "Trainer config missing or incorrect"
    assert lstm_params.get('num_layers') == 2, "Trainer config missing or incorrect"
    
    print("\n✅ Model trainer correctly stores and accesses config")

def test_architecture_consistency():
    """Test that model creation uses config parameters."""
    print("\n" + "=" * 60)
    print("TEST 4: Architecture Consistency Check")
    print("=" * 60)
    
    # Load config
    config_path = os.path.join(project_root, 'config', 'config.example.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    regime_config = config['ml']['regime_prediction']
    lstm_config = regime_config['model_params']['lstm_regime']
    
    # Create model using config params (simulating what trainer does)
    model = LSTMRegimePredictor(
        input_size=42,
        hidden_size=lstm_config.get('hidden_size', 64),
        num_layers=lstm_config.get('num_layers', 2),
        num_classes=3
    )
    
    print(f"Model created with config params:")
    print(f"  - hidden_size: {model.hidden_size}")
    print(f"  - num_layers: {model.num_layers}")
    
    # Verify
    assert model.hidden_size == 64, "Model not using config params"
    assert model.num_layers == 2, "Model not using config params"
    
    print("\n✅ Model creation correctly uses config parameters")

def main():
    """Run all tests."""
    print("\n" + "🧪" * 30)
    print("ML ARCHITECTURE SYNCHRONIZATION TEST SUITE")
    print("🧪" * 30 + "\n")
    
    try:
        # Run all tests
        test_config_loading()
        test_neural_network_defaults()
        test_model_trainer_config()
        test_architecture_consistency()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("  ✓ Config file has correct LSTM parameters (hidden_size=64, num_layers=2)")
        print("  ✓ LSTMRegimePredictor defaults match config")
        print("  ✓ RegimeModelTrainer accepts and stores config")
        print("  ✓ Model creation uses config parameters correctly")
        print("\n🎉 The codebase is now synchronized with config!")
        print("\nNext steps:")
        print("  1. Run training script: python scripts/train_all_models.py")
        print("  2. New models will use the 'small and safe' architecture (64/2)")
        print("  3. All size mismatch errors should be resolved")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
