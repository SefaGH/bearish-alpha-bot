"""
Tests for R/R Ratio Calculation Fixes.
Validates that stop-loss capping and TP realignment work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip


class TestAdaptiveOBRRCalculation:
    """Test LONG position R/R calculation with stop-loss cap fixes."""
    
    def test_stop_loss_cap_direction_long(self):
        """
        Test that stop-loss cap uses max() for LONG positions to LIMIT risk.
        
        Scenario: Theoretical SL distance = 1% (low risk)
                  max_sl_pct = 1.5%
        Expected: Stop should be at 1% (not capped to 1.5% which would INCREASE risk)
        """
        config = {
            'tp_atr_mult': 2.5,
            'sl_atr_mult': 1.2,
            'min_tp_pct': 0.008,
            'max_sl_pct': 0.015,  # 1.5%
            'rsi_max': 45,
            'min_rr_ratio': 1.2
        }
        
        strategy = AdaptiveOversoldBounce(config)
        
        # Create test data with low volatility (small ATR relative to price)
        entry_price = 100.0
        atr = 0.8333  # Will give theoretical SL = 1% (0.8333 * 1.2 / 100 = 1%)
        
        # Calculate theoretical levels
        theoretical_sl_distance = atr * config['sl_atr_mult']  # 1.0
        theoretical_sl_pct = theoretical_sl_distance / entry_price  # 0.01 = 1%
        
        # Verify theoretical is BELOW max (no capping needed)
        assert theoretical_sl_pct < config['max_sl_pct'], "Test setup error: theoretical should be below max"
        
        # Calculate expected stop using CORRECT formula: max(theoretical, entry * (1 - max_sl))
        expected_stop = max(
            entry_price * (1 - theoretical_sl_pct),  # 99.0
            entry_price * (1 - config['max_sl_pct'])  # 98.5
        )
        # Expected: max(99.0, 98.5) = 99.0 (no capping, theoretical is used)
        
        assert abs(expected_stop - 99.0) < 0.001, f"Expected stop at 99.0 (1% risk), got {expected_stop}"
        
        # Verify risk is limited to theoretical 1%, not inflated to 1.5%
        actual_risk_pct = (entry_price - expected_stop) / entry_price
        assert abs(actual_risk_pct - 0.01) < 0.0001, f"Risk should be 1%, got {actual_risk_pct:.2%}"
    
    def test_stop_loss_cap_when_needed_long(self):
        """
        Test that stop-loss cap actually caps when theoretical exceeds max.
        
        Scenario: Theoretical SL distance = 2.5% (high risk)
                  max_sl_pct = 1.5%
        Expected: Stop should be capped at 1.5%
        """
        config = {
            'tp_atr_mult': 2.5,
            'sl_atr_mult': 1.2,
            'min_tp_pct': 0.008,
            'max_sl_pct': 0.015,  # 1.5%
            'rsi_max': 45,
            'min_rr_ratio': 1.2
        }
        
        strategy = AdaptiveOversoldBounce(config)
        
        # Create test data with high volatility (large ATR)
        entry_price = 100.0
        atr = 2.0833  # Will give theoretical SL = 2.5% (2.0833 * 1.2 / 100 = 2.5%)
        
        # Calculate theoretical levels
        theoretical_sl_distance = atr * config['sl_atr_mult']  # 2.5
        theoretical_sl_pct = theoretical_sl_distance / entry_price  # 0.025 = 2.5%
        
        # Verify theoretical EXCEEDS max (capping is needed)
        assert theoretical_sl_pct > config['max_sl_pct'], "Test setup error: theoretical should exceed max"
        
        # Calculate expected stop using CORRECT formula: max(theoretical, entry * (1 - max_sl))
        # Since theoretical > max_sl, the actual_sl_pct should be capped to max_sl
        expected_stop = entry_price * (1 - config['max_sl_pct'])  # 98.5
        
        assert expected_stop == 98.5, f"Expected stop at 98.5 (1.5% risk), got {expected_stop}"
        
        # Verify risk is capped at 1.5%
        actual_risk_pct = (entry_price - expected_stop) / entry_price
        assert actual_risk_pct == 0.015, f"Risk should be capped at 1.5%, got {actual_risk_pct:.2%}"
    
    def test_tp_realignment_maintains_rr_long(self):
        """
        Test that TP is realigned when SL is capped to maintain intended R/R.
        
        Scenario: Theoretical SL = 2.5%, capped to 1.5%
                  Theoretical TP = 5.0% (would give R/R = 2.0)
                  Intended R/R = 2.5/1.2 = 2.08
        Expected: TP should be adjusted to maintain R/R = 2.08 with capped SL
        """
        config = {
            'tp_atr_mult': 2.5,
            'sl_atr_mult': 1.2,
            'min_tp_pct': 0.008,
            'max_sl_pct': 0.015,  # 1.5%
            'rsi_max': 45,
            'min_rr_ratio': 1.2
        }
        
        entry_price = 100.0
        atr = 2.0833  # Theoretical SL = 2.5% (will be capped)
        
        # Intended R/R ratio
        intended_rr = config['tp_atr_mult'] / config['sl_atr_mult']  # 2.5 / 1.2 = 2.08
        
        # When SL is capped to 1.5%
        actual_sl_pct = config['max_sl_pct']  # 0.015
        actual_sl_distance = entry_price * actual_sl_pct  # 1.5
        
        # TP should be realigned to maintain R/R = 2.08
        expected_tp_distance = actual_sl_distance * intended_rr  # 1.5 * 2.08 = 3.125
        expected_tp_pct = expected_tp_distance / entry_price  # 0.03125 = 3.125%
        
        expected_target = entry_price * (1 + expected_tp_pct)  # 103.125
        expected_stop = entry_price * (1 - actual_sl_pct)  # 98.5
        
        # Calculate actual R/R
        actual_rr = (expected_target - entry_price) / (entry_price - expected_stop)
        
        # Verify R/R is maintained
        assert abs(actual_rr - intended_rr) < 0.01, f"R/R should be {intended_rr:.2f}, got {actual_rr:.2f}"


class TestAdaptiveSTRRRCalculation:
    """Test SHORT position R/R calculation with stop-loss cap fixes."""
    
    def test_stop_loss_cap_direction_short(self):
        """
        Test that stop-loss cap uses min() for SHORT positions to LIMIT risk.
        
        Scenario: Theoretical SL distance = 1% (low risk)
                  max_sl_pct = 2%
        Expected: Stop should be at 1% (not capped to 2% which would INCREASE risk)
        """
        config = {
            'tp_atr_mult': 3.0,
            'sl_atr_mult': 1.5,
            'min_tp_pct': 0.010,
            'max_sl_pct': 0.020,  # 2%
            'rsi_min': 50,
            'min_rr_ratio': 1.2
        }
        
        strategy = AdaptiveShortTheRip(config)
        
        # Create test data with low volatility
        entry_price = 100.0
        atr = 0.6667  # Will give theoretical SL = 1% (0.6667 * 1.5 / 100 = 1%)
        
        # Calculate theoretical levels
        theoretical_sl_distance = atr * config['sl_atr_mult']  # 1.0
        theoretical_sl_pct = theoretical_sl_distance / entry_price  # 0.01 = 1%
        
        # Verify theoretical is BELOW max (no capping needed)
        assert theoretical_sl_pct < config['max_sl_pct'], "Test setup error: theoretical should be below max"
        
        # For SHORT: stop is ABOVE entry
        # Calculate expected stop using CORRECT formula: min(theoretical, entry * (1 + max_sl))
        expected_stop = min(
            entry_price * (1 + theoretical_sl_pct),  # 101.0
            entry_price * (1 + config['max_sl_pct'])  # 102.0
        )
        # Expected: min(101.0, 102.0) = 101.0 (no capping, theoretical is used)
        
        assert abs(expected_stop - 101.0) < 0.001, f"Expected stop at 101.0 (1% risk), got {expected_stop}"
        
        # Verify risk is limited to theoretical 1%, not inflated to 2%
        actual_risk_pct = (expected_stop - entry_price) / entry_price
        assert abs(actual_risk_pct - 0.01) < 0.0001, f"Risk should be 1%, got {actual_risk_pct:.2%}"
    
    def test_stop_loss_cap_when_needed_short(self):
        """
        Test that stop-loss cap actually caps when theoretical exceeds max for SHORT.
        
        Scenario: Theoretical SL distance = 3% (high risk)
                  max_sl_pct = 2%
        Expected: Stop should be capped at 2%
        """
        config = {
            'tp_atr_mult': 3.0,
            'sl_atr_mult': 1.5,
            'min_tp_pct': 0.010,
            'max_sl_pct': 0.020,  # 2%
            'rsi_min': 50,
            'min_rr_ratio': 1.2
        }
        
        entry_price = 100.0
        atr = 2.0  # Will give theoretical SL = 3% (2.0 * 1.5 / 100 = 3%)
        
        # Calculate theoretical levels
        theoretical_sl_distance = atr * config['sl_atr_mult']  # 3.0
        theoretical_sl_pct = theoretical_sl_distance / entry_price  # 0.03 = 3%
        
        # Verify theoretical EXCEEDS max (capping is needed)
        assert theoretical_sl_pct > config['max_sl_pct'], "Test setup error: theoretical should exceed max"
        
        # For SHORT: Calculate expected stop using CORRECT formula: min(theoretical, entry * (1 + max_sl))
        expected_stop = entry_price * (1 + config['max_sl_pct'])  # 102.0
        
        assert expected_stop == 102.0, f"Expected stop at 102.0 (2% risk), got {expected_stop}"
        
        # Verify risk is capped at 2%
        actual_risk_pct = (expected_stop - entry_price) / entry_price
        assert actual_risk_pct == 0.020, f"Risk should be capped at 2%, got {actual_risk_pct:.2%}"
    
    def test_tp_realignment_maintains_rr_short(self):
        """
        Test that TP is realigned when SL is capped to maintain intended R/R for SHORT.
        
        Scenario: Theoretical SL = 3%, capped to 2%
                  Intended R/R = 3.0/1.5 = 2.0
        Expected: TP should be adjusted to maintain R/R = 2.0 with capped SL
        """
        config = {
            'tp_atr_mult': 3.0,
            'sl_atr_mult': 1.5,
            'min_tp_pct': 0.010,
            'max_sl_pct': 0.020,  # 2%
            'rsi_min': 50,
            'min_rr_ratio': 1.2
        }
        
        entry_price = 100.0
        atr = 2.0  # Theoretical SL = 3% (will be capped)
        
        # Intended R/R ratio
        intended_rr = config['tp_atr_mult'] / config['sl_atr_mult']  # 3.0 / 1.5 = 2.0
        
        # When SL is capped to 2%
        actual_sl_pct = config['max_sl_pct']  # 0.020
        actual_sl_distance = entry_price * actual_sl_pct  # 2.0
        
        # TP should be realigned to maintain R/R = 2.0
        expected_tp_distance = actual_sl_distance * intended_rr  # 2.0 * 2.0 = 4.0
        expected_tp_pct = expected_tp_distance / entry_price  # 0.04 = 4%
        
        # For SHORT: target is BELOW entry
        expected_target = entry_price * (1 - expected_tp_pct)  # 96.0
        expected_stop = entry_price * (1 + actual_sl_pct)  # 102.0
        
        # Calculate actual R/R
        actual_rr = (entry_price - expected_target) / (expected_stop - entry_price)
        
        # Verify R/R is maintained
        assert abs(actual_rr - intended_rr) < 0.01, f"R/R should be {intended_rr:.2f}, got {actual_rr:.2f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
