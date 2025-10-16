"""
Test adaptive strategy RSI threshold adjustments.
Validates that the new gentler threshold logic works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip


class TestAdaptiveOBThresholds:
    """Test AdaptiveOversoldBounce threshold logic."""
    
    def test_base_threshold_from_config(self):
        """Test that base threshold is read from config correctly."""
        cfg = {
            'adaptive_rsi_base': 45,
            'adaptive_rsi_range': 10,
            'tp_pct': 0.006,
            'sl_atr_mult': 1.5
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        # Neutral regime should return base threshold
        regime = {'trend': 'neutral', 'momentum': 'sideways', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        assert threshold == 45, f"Expected 45 for neutral regime, got {threshold}"
    
    def test_bullish_strong_adjustment(self):
        """Test bullish strong momentum adjustment."""
        cfg = {
            'adaptive_rsi_base': 45,
            'adaptive_rsi_range': 10,
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        regime = {'trend': 'bullish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        # Should be base - 5 = 40
        assert threshold == 40, f"Expected 40 for bullish strong, got {threshold}"
    
    def test_bullish_weak_adjustment(self):
        """Test bullish weak momentum adjustment."""
        cfg = {
            'adaptive_rsi_base': 45,
            'adaptive_rsi_range': 10,
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        regime = {'trend': 'bullish', 'momentum': 'weak', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        # Should be around base - 3 = 42
        assert 41 <= threshold <= 43, f"Expected ~42 for bullish weak, got {threshold}"
    
    def test_bearish_strong_adjustment(self):
        """Test bearish strong momentum adjustment."""
        cfg = {
            'adaptive_rsi_base': 45,
            'adaptive_rsi_range': 10,
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        regime = {'trend': 'bearish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        # Should be base + 5 = 50
        assert threshold == 50, f"Expected 50 for bearish strong, got {threshold}"
    
    def test_bearish_weak_adjustment(self):
        """Test bearish weak momentum adjustment."""
        cfg = {
            'adaptive_rsi_base': 45,
            'adaptive_rsi_range': 10,
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        regime = {'trend': 'bearish', 'momentum': 'weak', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        # Should be around base + 3 = 48
        assert 47 <= threshold <= 49, f"Expected ~48 for bearish weak, got {threshold}"
    
    def test_threshold_clamping_lower_bound(self):
        """Test that threshold never goes below 30."""
        cfg = {
            'adaptive_rsi_base': 35,  # Lower base
            'adaptive_rsi_range': 20,  # Large range
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        # Even in extreme bullish, should not go below 30
        regime = {'trend': 'bullish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        assert threshold >= 30, f"Threshold {threshold} went below minimum of 30"
    
    def test_threshold_clamping_upper_bound(self):
        """Test that threshold never goes above 50."""
        cfg = {
            'adaptive_rsi_base': 48,  # Higher base
            'adaptive_rsi_range': 20,  # Large range
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        # Even in extreme bearish, should not go above 50
        regime = {'trend': 'bearish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        assert threshold <= 50, f"Threshold {threshold} went above maximum of 50"
    
    def test_fallback_to_rsi_max(self):
        """Test fallback when adaptive_rsi_base is not in config."""
        cfg = {
            'rsi_max': 45,  # No adaptive_rsi_base
            'tp_pct': 0.006,
        }
        strategy = AdaptiveOversoldBounce(cfg)
        
        regime = {'trend': 'neutral', 'momentum': 'sideways', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        assert threshold == 45, f"Expected fallback to rsi_max (45), got {threshold}"


class TestAdaptiveSTRThresholds:
    """Test AdaptiveShortTheRip threshold logic."""
    
    def test_base_threshold_from_config(self):
        """Test that base threshold is read from config correctly."""
        cfg = {
            'adaptive_rsi_base': 50,
            'adaptive_rsi_range': 10,
            'tp_pct': 0.006,
            'sl_atr_mult': 1.5
        }
        strategy = AdaptiveShortTheRip(cfg)
        
        # Neutral regime should return base threshold
        regime = {'trend': 'neutral', 'momentum': 'sideways', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        assert threshold == 50, f"Expected 50 for neutral regime, got {threshold}"
    
    def test_bearish_strong_adjustment(self):
        """Test bearish strong momentum adjustment (more aggressive shorting)."""
        cfg = {
            'adaptive_rsi_base': 50,
            'adaptive_rsi_range': 10,
        }
        strategy = AdaptiveShortTheRip(cfg)
        
        regime = {'trend': 'bearish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        # For shorts in bearish: lower threshold = more aggressive
        # Should be base - 5 = 45 (but clamped to min 50 for shorts)
        assert threshold == 50, f"Expected 50 (clamped) for bearish strong, got {threshold}"
    
    def test_bullish_strong_adjustment(self):
        """Test bullish strong momentum adjustment (more selective shorting)."""
        cfg = {
            'adaptive_rsi_base': 60,
            'adaptive_rsi_range': 10,
        }
        strategy = AdaptiveShortTheRip(cfg)
        
        regime = {'trend': 'bullish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        # For shorts in bullish: higher threshold = more selective
        # Should be base + 5 = 65
        assert threshold == 65, f"Expected 65 for bullish strong, got {threshold}"
    
    def test_threshold_clamping_for_shorts(self):
        """Test that short thresholds stay in 50-70 range."""
        cfg = {
            'adaptive_rsi_base': 55,
            'adaptive_rsi_range': 10,
        }
        strategy = AdaptiveShortTheRip(cfg)
        
        # Test various regimes
        regimes = [
            {'trend': 'bearish', 'momentum': 'strong'},
            {'trend': 'bearish', 'momentum': 'weak'},
            {'trend': 'bullish', 'momentum': 'strong'},
            {'trend': 'bullish', 'momentum': 'weak'},
            {'trend': 'neutral', 'momentum': 'sideways'},
        ]
        
        for regime in regimes:
            threshold = strategy.get_adaptive_rsi_threshold(regime)
            assert 50 <= threshold <= 70, f"Threshold {threshold} outside 50-70 range for regime {regime}"
    
    def test_fallback_to_rsi_min(self):
        """Test fallback when adaptive_rsi_base is not in config."""
        cfg = {
            'rsi_min': 50,  # No adaptive_rsi_base
            'tp_pct': 0.006,
        }
        strategy = AdaptiveShortTheRip(cfg)
        
        regime = {'trend': 'neutral', 'momentum': 'sideways', 'volatility': 'normal'}
        threshold = strategy.get_adaptive_rsi_threshold(regime)
        
        assert threshold == 50, f"Expected fallback to rsi_min (50), got {threshold}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
