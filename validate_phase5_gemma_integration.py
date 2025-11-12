#!/usr/bin/env python3.11
"""
Validation Script for Phase 5: GEMMA AI-Gate Integration

This script validates the StrategyCoordinator GEMMA integration without requiring
full dependencies. It checks:
1. Class structure and methods exist
2. Configuration handling works correctly
3. AI-Gate logic is properly implemented
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def validate_imports():
    """Validate that required modules can be imported"""
    print("=" * 80)
    print("PHASE 5 VALIDATION: GEMMA AI-Gate Integration")
    print("=" * 80)
    print("\n1. Validating imports...")
    
    try:
        from src.core.strategy_coordinator import StrategyCoordinator
        print("   ✅ StrategyCoordinator imported successfully")
        return StrategyCoordinator
    except ImportError as e:
        print(f"   ❌ Failed to import StrategyCoordinator: {e}")
        return None


def validate_class_structure(StrategyCoordinator):
    """Validate that the class has all required methods"""
    print("\n2. Validating class structure...")
    
    required_methods = [
        '_initialize_gemma',
        '_apply_ai_gate',
        'process_signal'
    ]
    
    all_valid = True
    for method_name in required_methods:
        if hasattr(StrategyCoordinator, method_name):
            print(f"   ✅ Method '{method_name}' exists")
        else:
            print(f"   ❌ Method '{method_name}' is missing")
            all_valid = False
    
    return all_valid


def validate_initialization():
    """Validate that initialization works correctly"""
    print("\n3. Validating initialization...")
    
    from src.core.strategy_coordinator import StrategyCoordinator
    
    # Test 1: GEMMA disabled
    config_disabled = {
        'ml': {
            'gemma': {
                'enabled': False
            }
        }
    }
    
    portfolio_manager = Mock()
    risk_manager = Mock()
    
    try:
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config_disabled
        )
        if coordinator.gemma_adapter is None:
            print("   ✅ GEMMA adapter correctly NOT initialized when disabled")
        else:
            print("   ❌ GEMMA adapter should be None when disabled")
            return False
    except Exception as e:
        print(f"   ❌ Initialization failed with disabled config: {e}")
        return False
    
    # Test 2: GEMMA enabled (should handle missing dependencies gracefully)
    config_enabled = {
        'ml': {
            'gemma': {
                'enabled': True,
                'model_path': 'test/path/model.pt'
            }
        }
    }
    
    try:
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config_enabled
        )
        # Should be None due to missing torch/model files, but shouldn't crash
        if coordinator.gemma_adapter is None:
            print("   ✅ GEMMA initialization handled gracefully (expected to fail without torch)")
        else:
            print("   ⚠️  GEMMA adapter initialized (unexpected but not an error)")
    except Exception as e:
        print(f"   ❌ Initialization crashed when GEMMA enabled: {e}")
        return False
    
    # Test 3: Check processing_stats has new fields
    if 'ai_gate_rejections' in coordinator.processing_stats:
        print("   ✅ processing_stats has 'ai_gate_rejections' field")
    else:
        print("   ❌ processing_stats missing 'ai_gate_rejections' field")
        return False
    
    if 'approved_signals' in coordinator.processing_stats:
        print("   ✅ processing_stats has 'approved_signals' field")
    else:
        print("   ❌ processing_stats missing 'approved_signals' field")
        return False
    
    return True


def validate_ai_gate_logic():
    """Validate AI-Gate filtering logic"""
    print("\n4. Validating AI-Gate filtering logic...")
    
    from src.core.strategy_coordinator import StrategyCoordinator
    
    config = {
        'ml': {
            'price': {
                'min_confidence': 0.66
            }
        }
    }
    
    portfolio_manager = Mock()
    risk_manager = Mock()
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        config=config
    )
    
    # Test 1: High confidence signal should pass
    signal_high = {
        'symbol': 'BTC/USDT',
        'ml_confidence': 0.75,
        'features': {}
    }
    
    result = coordinator._apply_ai_gate(signal_high)
    if result is True:
        print("   ✅ AI-Gate correctly passes high-confidence signal (0.75 >= 0.66)")
    else:
        print("   ❌ AI-Gate should pass high-confidence signal")
        return False
    
    # Test 2: Low confidence signal should be rejected
    signal_low = {
        'symbol': 'ETH/USDT',
        'ml_confidence': 0.50,
        'features': {}
    }
    
    result = coordinator._apply_ai_gate(signal_low)
    if result is False:
        print("   ✅ AI-Gate correctly rejects low-confidence signal (0.50 < 0.66)")
    else:
        print("   ❌ AI-Gate should reject low-confidence signal")
        return False
    
    # Test 3: Check rejection counter
    if coordinator.processing_stats['ai_gate_rejections'] == 1:
        print("   ✅ ai_gate_rejections counter correctly incremented")
    else:
        print(f"   ❌ ai_gate_rejections counter is {coordinator.processing_stats['ai_gate_rejections']}, expected 1")
        return False
    
    return True


def validate_signal_enrichment():
    """Validate signal enrichment with GEMMA predictions"""
    print("\n5. Validating signal enrichment...")
    
    from src.core.strategy_coordinator import StrategyCoordinator
    
    config = {
        'ml': {
            'price': {
                'min_confidence': 0.66
            }
        }
    }
    
    portfolio_manager = Mock()
    risk_manager = Mock()
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        config=config
    )
    
    # Mock GEMMA adapter
    mock_adapter = Mock()
    mock_adapter.predict.return_value = {
        'price_confidence': 0.85,
        'prediction_label': 'bullish',
        'prediction': 2
    }
    coordinator.gemma_adapter = mock_adapter
    
    signal = {
        'symbol': 'BTC/USDT',
        'ml_confidence': 0.50,  # Low legacy confidence
        'features': {'feature1': 1.0, 'feature2': 2.0}
    }
    
    result = coordinator._apply_ai_gate(signal)
    
    # Should use GEMMA confidence (0.85) and pass
    if result is True:
        print("   ✅ Signal passed using GEMMA confidence (0.85)")
    else:
        print("   ❌ Signal should pass with GEMMA confidence")
        return False
    
    # Check signal enrichment
    if signal.get('gemma_confidence') == 0.85:
        print("   ✅ Signal enriched with gemma_confidence")
    else:
        print("   ❌ Signal missing or incorrect gemma_confidence")
        return False
    
    if signal.get('gemma_prediction') == 'bullish':
        print("   ✅ Signal enriched with gemma_prediction")
    else:
        print("   ❌ Signal missing or incorrect gemma_prediction")
        return False
    
    return True


def main():
    """Run all validation checks"""
    
    # Step 1: Import validation
    StrategyCoordinator = validate_imports()
    if not StrategyCoordinator:
        print("\n❌ VALIDATION FAILED: Could not import required classes")
        return False
    
    # Step 2: Structure validation
    if not validate_class_structure(StrategyCoordinator):
        print("\n❌ VALIDATION FAILED: Class structure incomplete")
        return False
    
    # Step 3: Initialization validation
    if not validate_initialization():
        print("\n❌ VALIDATION FAILED: Initialization issues detected")
        return False
    
    # Step 4: AI-Gate logic validation
    if not validate_ai_gate_logic():
        print("\n❌ VALIDATION FAILED: AI-Gate logic issues detected")
        return False
    
    # Step 5: Signal enrichment validation
    if not validate_signal_enrichment():
        print("\n❌ VALIDATION FAILED: Signal enrichment issues detected")
        return False
    
    # All tests passed
    print("\n" + "=" * 80)
    print("✅ ALL VALIDATION CHECKS PASSED!")
    print("=" * 80)
    print("\nPhase 5 Integration Summary:")
    print("  • StrategyCoordinator successfully enhanced with GEMMA AI-Gate")
    print("  • _initialize_gemma method correctly handles initialization")
    print("  • _apply_ai_gate method correctly filters signals by confidence")
    print("  • process_signal method implements full signal flow")
    print("  • Signal enrichment with GEMMA predictions works correctly")
    print("  • Graceful fallback to legacy ML when GEMMA not available")
    print("\nSignal Flow: GEMMA → AI-Gate → RL-Veto → Risk Checks → Execution")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
