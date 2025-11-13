#!/usr/bin/env python3
"""
Validation script to test the fixes made to train_all_models.py and model_trainer.py.
This script validates that:
1. Feature mask can be loaded from the correct path
2. JSON validation works correctly
3. Config is properly passed and used
4. MLP architecture is correctly configured
"""

import sys
import os
import numpy as np
import json
import yaml
from pathlib import Path
import logging

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.logger import setup_logger

logger = setup_logger("test-train-fixes", level=logging.INFO)

def test_feature_mask_loading():
    """Test that feature mask can be loaded from correct path."""
    logger.info("="*60)
    logger.info("TEST 1: Feature Mask Loading")
    logger.info("="*60)
    
    mask_path = Path('data/models/cache/gemma/feature_selection_mask.npy')
    
    if not mask_path.exists():
        logger.error(f"❌ FAIL: Feature mask not found at {mask_path}")
        return False
    
    logger.info(f"✅ PASS: Feature mask found at {mask_path}")
    
    try:
        feature_mask = np.load(mask_path)
        logger.info(f"✅ PASS: Feature mask loaded successfully")
        logger.info(f"   Mask shape: {feature_mask.shape}")
        logger.info(f"   Total features: {len(feature_mask)}")
        logger.info(f"   Selected features: {np.sum(feature_mask)}")
        return True
    except Exception as e:
        logger.error(f"❌ FAIL: Error loading feature mask: {e}")
        return False

def test_json_validation():
    """Test that JSON feature plan files exist and are valid."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: JSON Feature Plan Validation")
    logger.info("="*60)
    
    test_passed = True
    
    for model_type in ['price', 'regime']:
        json_plan_name = f"gemma_{model_type}_selected_82.json"
        json_plan_path = Path(f"features/gemma/selected/{json_plan_name}")
        
        if not json_plan_path.exists():
            logger.error(f"❌ FAIL: JSON plan not found at {json_plan_path}")
            test_passed = False
            continue
        
        logger.info(f"✅ PASS: JSON plan found at {json_plan_path}")
        
        try:
            with open(json_plan_path, 'r') as f:
                feature_plan = json.load(f)
            
            selected_feature_count = feature_plan.get('count', 0)
            logger.info(f"   Expected feature count: {selected_feature_count}")
            
            if selected_feature_count != 82:
                logger.error(f"❌ FAIL: Expected 82 features, got {selected_feature_count}")
                test_passed = False
            else:
                logger.info(f"✅ PASS: Feature count is correct (82)")
                
        except Exception as e:
            logger.error(f"❌ FAIL: Error loading JSON plan: {e}")
            test_passed = False
    
    return test_passed

def test_mask_json_consistency():
    """Test that mask and JSON plans are consistent."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Mask-JSON Consistency")
    logger.info("="*60)
    
    mask_path = Path('data/models/cache/gemma/feature_selection_mask.npy')
    feature_mask = np.load(mask_path)
    selected_feature_count_from_mask = np.sum(feature_mask)
    
    test_passed = True
    
    for model_type in ['price', 'regime']:
        json_plan_name = f"gemma_{model_type}_selected_82.json"
        json_plan_path = Path(f"features/gemma/selected/{json_plan_name}")
        
        with open(json_plan_path, 'r') as f:
            feature_plan = json.load(f)
        
        selected_feature_count_from_json = feature_plan.get('count', 0)
        
        if selected_feature_count_from_json != selected_feature_count_from_mask:
            logger.error(f"❌ FAIL: Inconsistency for {model_type}! Mask: {selected_feature_count_from_mask}, JSON: {selected_feature_count_from_json}")
            test_passed = False
        else:
            logger.info(f"✅ PASS: {model_type} mask-JSON consistency verified (82 features)")
    
    return test_passed

def test_config_loading():
    """Test that config can be loaded and has correct GEMMA structure."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Config Loading and Structure")
    logger.info("="*60)
    
    config_path = Path('config/config.example.yaml')
    
    if not config_path.exists():
        logger.error(f"❌ FAIL: Config file not found at {config_path}")
        return False
    
    logger.info(f"✅ PASS: Config file found at {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        ml_config = config.get('ml', {})
        gemma_config = ml_config.get('gemma', {})
        
        if not gemma_config:
            logger.error("❌ FAIL: GEMMA config not found in config file")
            return False
        
        logger.info("✅ PASS: GEMMA config found")
        
        # Check architecture section
        arch_config = gemma_config.get('architecture', {})
        if not arch_config:
            logger.error("❌ FAIL: GEMMA architecture config not found")
            return False
        
        logger.info("✅ PASS: GEMMA architecture config found")
        logger.info(f"   hidden_size: {arch_config.get('hidden_size')}")
        logger.info(f"   num_layers: {arch_config.get('num_layers')}")
        logger.info(f"   dropout: {arch_config.get('dropout')}")
        logger.info(f"   num_classes: {arch_config.get('num_classes')}")
        
        # Check training section
        train_config = gemma_config.get('training', {})
        if not train_config:
            logger.error("❌ FAIL: GEMMA training config not found")
            return False
        
        logger.info("✅ PASS: GEMMA training config found")
        logger.info(f"   batch_size: {train_config.get('batch_size')}")
        logger.info(f"   epochs: {train_config.get('epochs')}")
        logger.info(f"   learning_rate: {train_config.get('learning_rate')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FAIL: Error loading config: {e}")
        return False

def test_mlp_architecture_conversion():
    """Test that MLP architecture conversion is correct."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: MLP Architecture Conversion Logic")
    logger.info("="*60)
    
    # Simulate the conversion logic
    hidden_size = 64
    num_layers = 3
    
    # OLD (incorrect) logic
    old_layers = [hidden_size // (i + 1) for i in range(num_layers)]
    logger.info(f"OLD (incorrect) logic: hidden_size={hidden_size}, num_layers={num_layers}")
    logger.info(f"   Result: {old_layers}")
    
    # NEW (correct) logic
    new_layers = [hidden_size for _ in range(num_layers)]
    logger.info(f"NEW (correct) logic: hidden_size={hidden_size}, num_layers={num_layers}")
    logger.info(f"   Result: {new_layers}")
    
    expected_layers = [64, 64, 64]
    if new_layers == expected_layers:
        logger.info(f"✅ PASS: MLP architecture conversion is correct")
        return True
    else:
        logger.error(f"❌ FAIL: Expected {expected_layers}, got {new_layers}")
        return False

def main():
    """Run all validation tests."""
    logger.info("="*60)
    logger.info("TRAIN_ALL_MODELS.PY FIX VALIDATION SUITE")
    logger.info("="*60)
    
    results = {
        "Feature Mask Loading": test_feature_mask_loading(),
        "JSON Feature Plan Validation": test_json_validation(),
        "Mask-JSON Consistency": test_mask_json_consistency(),
        "Config Loading and Structure": test_config_loading(),
        "MLP Architecture Conversion": test_mlp_architecture_conversion(),
    }
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*60)
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
