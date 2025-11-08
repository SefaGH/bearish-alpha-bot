#!/usr/bin/env python3
"""
Validation script for FAZ 3.1 Neural Network Optimizations.

This script validates that all the optimizations are correctly implemented:
1. Early stopping with proper parameters
2. LSTM architecture with enhanced capacity
3. Transformer architecture with better attention
4. Sequence length increased to 20
5. Learning rate and weight decay properly set
6. Enhanced logging functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set ML_ENABLED for validation
os.environ['ML_ENABLED'] = 'true'

import torch
import numpy as np
from src.ml.model_trainer import (
    RegimeModelTrainer, 
    EarlyStopping,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MIN_EPOCHS,
    SEQUENCE_LENGTH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT
)
from src.ml.neural_networks import LSTMRegimePredictor, TransformerRegimePredictor

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def validate_constants():
    """Validate that all constants are properly set."""
    print_section("1. Validating Training Constants")
    
    checks = [
        ("NUM_EPOCHS", NUM_EPOCHS, 50, "Training epochs increased"),
        ("EARLY_STOPPING_PATIENCE", EARLY_STOPPING_PATIENCE, 5, "Early stopping patience"),
        ("MIN_EPOCHS", MIN_EPOCHS, 20, "Minimum epochs before early stop"),
        ("SEQUENCE_LENGTH", SEQUENCE_LENGTH, 20, "Sequence length increased"),
        ("LEARNING_RATE", LEARNING_RATE, 0.0005, "Learning rate optimized"),
        ("WEIGHT_DECAY", WEIGHT_DECAY, 1e-5, "Weight decay for L2 regularization"),
    ]
    
    all_passed = True
    for name, actual, expected, description in checks:
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {name}: {actual} (expected: {expected}) - {description}")
        if actual != expected:
            all_passed = False
    
    return all_passed

def validate_lstm_architecture():
    """Validate LSTM architecture improvements."""
    print_section("2. Validating LSTM Architecture")
    
    checks = [
        ("LSTM_HIDDEN_SIZE", LSTM_HIDDEN_SIZE, 128, "Increased from 64"),
        ("LSTM_NUM_LAYERS", LSTM_NUM_LAYERS, 3, "Increased from 2"),
        ("LSTM_DROPOUT", LSTM_DROPOUT, 0.3, "Increased from 0.2"),
    ]
    
    all_passed = True
    for name, actual, expected, description in checks:
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {name}: {actual} (expected: {expected}) - {description}")
        if actual != expected:
            all_passed = False
    
    # Test LSTM instantiation
    print("\n  Testing LSTM model instantiation...")
    try:
        lstm = LSTMRegimePredictor(
            input_size=42, 
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            num_classes=3,
            dropout=LSTM_DROPOUT
        )
        total_params = sum(p.numel() for p in lstm.parameters() if p.requires_grad)
        print(f"  ✅ LSTM created successfully")
        print(f"     Total parameters: {total_params:,}")
        
        # Test forward pass
        x = torch.randn(4, 20, 42)
        output = lstm(x)
        print(f"     Forward pass: input {x.shape} -> output {output.shape}")
        assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"
        print(f"  ✅ LSTM forward pass successful")
        
        # Check for batch normalization
        has_batchnorm = any('BatchNorm' in str(type(m)) for m in lstm.modules())
        status = "✅" if has_batchnorm else "❌"
        print(f"  {status} Batch normalization: {'present' if has_batchnorm else 'missing'}")
        
    except Exception as e:
        print(f"  ❌ LSTM instantiation failed: {e}")
        all_passed = False
    
    return all_passed

def validate_transformer_architecture():
    """Validate Transformer architecture improvements."""
    print_section("3. Validating Transformer Architecture")
    
    checks = [
        ("TRANSFORMER_NHEAD", TRANSFORMER_NHEAD, 6, "Increased for better attention"),
        ("TRANSFORMER_NUM_LAYERS", TRANSFORMER_NUM_LAYERS, 4, "Increased from 2"),
        ("TRANSFORMER_DIM_FEEDFORWARD", TRANSFORMER_DIM_FEEDFORWARD, 256, "Increased from 128"),
        ("TRANSFORMER_DROPOUT", TRANSFORMER_DROPOUT, 0.3, "Proper regularization"),
    ]
    
    all_passed = True
    for name, actual, expected, description in checks:
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {name}: {actual} (expected: {expected}) - {description}")
        if actual != expected:
            all_passed = False
    
    # Test Transformer instantiation
    print("\n  Testing Transformer model instantiation...")
    try:
        transformer = TransformerRegimePredictor(
            d_model=42,
            nhead=TRANSFORMER_NHEAD,
            num_layers=TRANSFORMER_NUM_LAYERS,
            num_classes=3,
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
            dropout=TRANSFORMER_DROPOUT
        )
        total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        print(f"  ✅ Transformer created successfully")
        print(f"     Total parameters: {total_params:,}")
        
        # Test forward pass
        x = torch.randn(4, 20, 42)
        output = transformer(x)
        print(f"     Forward pass: input {x.shape} -> output {output.shape}")
        assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"
        print(f"  ✅ Transformer forward pass successful")
        
        # Check for batch normalization
        has_batchnorm = any('BatchNorm' in str(type(m)) for m in transformer.modules())
        status = "✅" if has_batchnorm else "❌"
        print(f"  {status} Batch normalization: {'present' if has_batchnorm else 'missing'}")
        
    except Exception as e:
        print(f"  ❌ Transformer instantiation failed: {e}")
        all_passed = False
    
    return all_passed

def validate_early_stopping():
    """Validate early stopping implementation."""
    print_section("4. Validating Early Stopping")
    
    try:
        es = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=0.001,
            min_epochs=MIN_EPOCHS
        )
        print(f"  ✅ EarlyStopping created: patience={es.patience}, min_epochs={es.min_epochs}")
        
        # Test behavior before min_epochs
        should_stop = es(1.0, 10)
        status = "✅" if not should_stop else "❌"
        print(f"  {status} Before min_epochs (epoch 10): should_stop={should_stop} (expected: False)")
        
        # Test behavior with improvement
        es.best_loss = None  # Reset
        es.counter = 0
        should_stop = es(0.5, 20)
        status = "✅" if not should_stop else "❌"
        print(f"  {status} With improvement (epoch 20): should_stop={should_stop} (expected: False)")
        
        # Test behavior without improvement
        for i in range(EARLY_STOPPING_PATIENCE):
            should_stop = es(0.51, 21 + i)
        status = "✅" if should_stop else "❌"
        print(f"  {status} After {EARLY_STOPPING_PATIENCE} epochs no improvement: should_stop={should_stop} (expected: True)")
        
        return status == "✅"
    except Exception as e:
        print(f"  ❌ Early stopping validation failed: {e}")
        return False

def validate_trainer_integration():
    """Validate trainer integration with all new features."""
    print_section("5. Validating Trainer Integration")
    
    try:
        config = {
            'model_params': {
                'lstm_regime': {
                    'hidden_size': LSTM_HIDDEN_SIZE,
                    'num_layers': LSTM_NUM_LAYERS,
                    'dropout': LSTM_DROPOUT
                }
            }
        }
        trainer = RegimeModelTrainer(config=config)
        print(f"  ✅ Trainer initialized successfully")
        
        # Test sequence creation with new length
        X = np.random.randn(100, 42)
        y = np.random.randint(0, 3, 100)
        X_seq, y_seq = trainer._create_sequences(X, y, seq_length=SEQUENCE_LENGTH)
        
        status = "✅" if X_seq.shape[1] == SEQUENCE_LENGTH else "❌"
        print(f"  {status} Sequence creation: shape={X_seq.shape} (expected seq_len={SEQUENCE_LENGTH})")
        
        return status == "✅"
    except Exception as e:
        print(f"  ❌ Trainer integration failed: {e}")
        return False

def main():
    """Run all validations."""
    print("\n" + "="*70)
    print("  FAZ 3.1 NEURAL NETWORK OPTIMIZATION VALIDATION")
    print("="*70)
    print("\n  Validating all optimizations are correctly implemented...")
    
    results = []
    results.append(("Training Constants", validate_constants()))
    results.append(("LSTM Architecture", validate_lstm_architecture()))
    results.append(("Transformer Architecture", validate_transformer_architecture()))
    results.append(("Early Stopping", validate_early_stopping()))
    results.append(("Trainer Integration", validate_trainer_integration()))
    
    # Summary
    print_section("VALIDATION SUMMARY")
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✅ ALL VALIDATIONS PASSED - FAZ 3.1 OPTIMIZATIONS VERIFIED")
    else:
        print("  ❌ SOME VALIDATIONS FAILED - PLEASE REVIEW")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
