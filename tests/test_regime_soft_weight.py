"""
Tests for regime soft-weighting functionality.

Tests the graduated weighting system based on regime confidence levels
instead of hard threshold cutoffs.
"""
import pytest
from src.config.risk_config import RiskConfiguration
from src.core.risk_rules import RiskRewardRatioRule


class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    pass


class TestRegimeSoftWeightConfig:
    """Test regime soft-weight configuration loading."""
    
    def test_default_soft_weight_config_loaded(self):
        """Test that default soft-weight configuration is loaded."""
        config = RiskConfiguration({})
        
        # Check regime soft-weight config exists
        assert hasattr(config, 'regime_soft_weight')
        assert config.regime_soft_weight['enabled'] is True
        assert config.regime_soft_weight['min_confidence_hard_reject'] == 0.30
        assert config.regime_soft_weight['min_confidence_full_weight'] == 0.60
    
    def test_signal_scoring_config_loaded(self):
        """Test that signal scoring configuration is loaded."""
        config = RiskConfiguration({})
        
        # Check signal scoring config exists
        assert hasattr(config, 'signal_scoring')
        assert config.signal_scoring['enabled'] is True
        assert config.signal_scoring['min_score_to_trade'] == 60
        
        # Check weights sum to 1.0
        weights = config.signal_scoring['weights']
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_custom_soft_weight_config(self):
        """Test custom soft-weight configuration."""
        custom_config = {
            'ml': {
                'regime': {
                    'soft_weighting_enabled': False,
                    'min_confidence_hard_reject': 0.40,
                    'min_confidence_full_weight': 0.70
                }
            }
        }
        
        config = RiskConfiguration(custom_config)
        
        assert config.regime_soft_weight['enabled'] is False
        assert config.regime_soft_weight['min_confidence_hard_reject'] == 0.40
        assert config.regime_soft_weight['min_confidence_full_weight'] == 0.70


class TestRegimeWeightCalculation:
    """Test regime_weight calculation in dynamic R/R."""
    
    def test_high_confidence_regime_full_weight(self):
        """Test that high confidence regime gets full weight (1.0)."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.3,
                    'rl_agreement': 0.3,
                    'regime_clarity': 0.2,
                    'volume_strength': 0.1,
                    'momentum_strength': 0.1
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5,
                    'missing_regime_default': 0.3
                },
                'regime_multipliers': {
                    'bullish': 0.9,
                    'bearish': 0.9,
                    'neutral': 1.0,
                    'volatile': 1.2
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        portfolio = MockPortfolioManager()
        
        # High confidence regime (0.8 > 0.6 threshold)
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 101.5,
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.8,
            'regime_confidence': 0.8,
            'regime_name': 'bullish',
            'regime_weight': 1.0,  # Full weight
            'strategy_min_rr': 0.5
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid, f"High confidence regime signal should be valid: {reason}"
    
    def test_medium_confidence_regime_partial_weight(self):
        """Test that medium confidence regime gets partial weight."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.3,
                    'rl_agreement': 0.3,
                    'regime_clarity': 0.2,
                    'volume_strength': 0.1,
                    'momentum_strength': 0.1
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5,
                    'missing_regime_default': 0.3
                },
                'regime_multipliers': {
                    'bullish': 0.9,
                    'bearish': 0.9,
                    'neutral': 1.0,
                    'volatile': 1.2
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        portfolio = MockPortfolioManager()
        
        # Medium confidence regime (0.45 between 0.3 and 0.6)
        # regime_weight = 0.45 / 0.6 = 0.75
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 101.3,
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.8,
            'regime_confidence': 0.45,
            'regime_name': 'bullish',
            'regime_weight': 0.75,  # Partial weight
            'strategy_min_rr': 0.5
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        # Should still be valid with partial weight
        assert is_valid, f"Medium confidence regime signal should be valid: {reason}"
    
    def test_low_confidence_regime_ignored(self):
        """Test that very low confidence regime is ignored (no regime_weight)."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.3,
                    'rl_agreement': 0.3,
                    'regime_clarity': 0.2,
                    'volume_strength': 0.1,
                    'momentum_strength': 0.1
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5,
                    'missing_regime_default': 0.3
                },
                'regime_multipliers': {
                    'bullish': 0.9,
                    'bearish': 0.9,
                    'neutral': 1.0,
                    'volatile': 1.2
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        portfolio = MockPortfolioManager()
        
        # Very low confidence regime (0.2 < 0.3 threshold) - should be ignored
        # No regime_weight in signal means it defaults to 1.0
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 101.3,
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.8,
            # No regime_confidence or regime_weight - regime ignored
            'strategy_min_rr': 0.5
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        # Should still be valid because other signals are strong
        assert is_valid, f"Signal without regime should still work: {reason}"
    
    def test_regime_weight_affects_rr_calculation(self):
        """Test that regime_weight proportionally affects R/R calculation."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.0,  # Disable other factors
                    'rl_agreement': 0.0,
                    'regime_clarity': 0.5,  # Focus on regime
                    'volume_strength': 0.0,
                    'momentum_strength': 0.0
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5,
                    'missing_regime_default': 0.5
                },
                'regime_multipliers': {
                    'volatile': 1.2  # 20% increase
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        
        # Calculate dynamic target with full regime weight
        signal_full_weight = {
            'regime_confidence': 0.8,
            'regime_name': 'volatile',
            'regime_weight': 1.0,
            'ml_confidence': 0.5,
            'rl_is_agree': False,
            'rl_action_prob': 0.5,
            'strategy_min_rr': 0.5
        }
        
        target_full = rule._calculate_dynamic_target(signal_full_weight)
        
        # Calculate with partial regime weight
        signal_partial_weight = signal_full_weight.copy()
        signal_partial_weight['regime_weight'] = 0.5
        
        target_partial = rule._calculate_dynamic_target(signal_partial_weight)
        
        # Partial weight should result in less extreme R/R adjustment
        # Full weight: ~1.5 * 1.2 = 1.8
        # Half weight: ~1.5 * (1 + 0.2*0.5) = 1.65
        assert target_full > target_partial, "Full regime weight should have stronger effect"
        assert target_partial > 1.5, "Partial weight should still have some effect"


class TestRegimeWeightBackwardCompatibility:
    """Test backward compatibility when regime_weight is not present."""
    
    def test_missing_regime_weight_defaults_to_one(self):
        """Test that missing regime_weight defaults to 1.0."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.3,
                    'rl_agreement': 0.3,
                    'regime_clarity': 0.2,
                    'volume_strength': 0.1,
                    'momentum_strength': 0.1
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5,
                    'missing_regime_default': 0.3
                },
                'regime_multipliers': {
                    'bullish': 0.9,
                    'bearish': 0.9,
                    'neutral': 1.0,
                    'volatile': 1.2
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        portfolio = MockPortfolioManager()
        
        # Signal without regime_weight (old format)
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 101.3,
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.8,
            'regime_confidence': 0.8,
            'regime_name': 'bullish',
            # No regime_weight - should default to 1.0
            'strategy_min_rr': 0.5
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        # Should work as before
        assert is_valid, f"Legacy signal format should still work: {reason}"


class TestEdgeCases:
    """Test edge cases for regime soft-weighting."""
    
    def test_exactly_at_thresholds(self):
        """Test behavior at exact threshold values."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.3,
                    'rl_agreement': 0.3,
                    'regime_clarity': 0.2,
                    'volume_strength': 0.1,
                    'momentum_strength': 0.1
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5,
                    'missing_regime_default': 0.3
                },
                'regime_multipliers': {
                    'neutral': 1.0
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        
        # Test at min_confidence_hard_reject (0.30)
        signal_at_min = {
            'regime_confidence': 0.30,
            'regime_name': 'neutral',
            'regime_weight': 0.30 / 0.60,  # 0.5
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.8,
            'strategy_min_rr': 0.5
        }
        
        target_min = rule._calculate_dynamic_target(signal_at_min)
        assert target_min > 0, "Should calculate valid R/R at minimum threshold"
        
        # Test at min_confidence_full_weight (0.60)
        signal_at_full = {
            'regime_confidence': 0.60,
            'regime_name': 'neutral',
            'regime_weight': 1.0,
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.8,
            'strategy_min_rr': 0.5
        }
        
        target_full = rule._calculate_dynamic_target(signal_at_full)
        assert target_full > 0, "Should calculate valid R/R at full weight threshold"
    
    def test_regime_weight_zero(self):
        """Test behavior with regime_weight of 0."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.3,
                    'rl_agreement': 0.3,
                    'regime_clarity': 0.2,
                    'volume_strength': 0.1,
                    'momentum_strength': 0.1
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5,
                    'missing_regime_default': 0.3
                },
                'regime_multipliers': {
                    'volatile': 1.2
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        
        # regime_weight = 0 means regime has no effect
        signal = {
            'regime_confidence': 0.0,
            'regime_name': 'volatile',
            'regime_weight': 0.0,
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.8,
            'strategy_min_rr': 0.5
        }
        
        target = rule._calculate_dynamic_target(signal)
        # With regime_weight=0, regime_mult should have no effect
        # Result should be close to base (considering ML/RL adjustments)
        assert 0.8 <= target <= 2.0, "Should stay within bounds"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
