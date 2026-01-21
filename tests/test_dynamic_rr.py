"""
Tests for Dynamic Risk/Reward Management System.
Tests the intelligent R/R adjustment based on ML/RL confidence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock

from core.risk_rules import RiskRewardRatioRule
from config.risk_config import RiskConfiguration


class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    
    def __init__(self, equity=10000):
        self.equity = equity
    
    def get_current_equity(self):
        return self.equity


class TestDynamicRRConfiguration:
    """Test dynamic R/R configuration loading."""
    
    def test_default_config_loaded(self):
        """Test that default dynamic R/R config is loaded correctly."""
        config = RiskConfiguration({
            'equity_usd': 10000,
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.3,
                    'rl_agreement': 0.3,
                    'regime_clarity': 0.2
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5
                },
                'regime_multipliers': {
                    'bullish': 0.9,
                    'bearish': 0.9,
                    'neutral': 1.0,
                    'volatile': 1.2
                }
            }
        })
        
        assert hasattr(config, 'rr_dynamic')
        assert config.rr_dynamic['enabled'] is True
        assert config.rr_dynamic['base_target_rr'] == 1.5
        assert config.rr_dynamic['lower_bound_rr'] == 0.8
        assert config.rr_dynamic['upper_bound_rr'] == 2.0
    
    def test_env_override_works(self):
        """Legacy behavior: RiskConfiguration no longer reads env vars directly."""
        os.environ['RR_BASE_TARGET'] = '2.0'
        try:
            config = RiskConfiguration({
                'equity_usd': 10000,
                'rr_dynamic': {
                    'base_target_rr': 1.5
                }
            })
            assert config.rr_dynamic['base_target_rr'] == 1.5
        finally:
            del os.environ['RR_BASE_TARGET']


class TestDynamicRRCalculation:
    """Test dynamic R/R threshold calculation."""
    
    def test_high_confidence_relaxes_rr(self):
        """High ML/RL confidence should reduce R/R requirement."""
        config = RiskConfiguration({
            'equity_usd': 10000,
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
        
        # High confidence signal with R/R = 0.85
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,      # Risk = 1
            'target': 100.85, # Reward = 0.85, R/R = 0.85
            'ml_confidence': 0.9,
            'rl_is_agree': True,
            'rl_action_prob': 0.95,
            'regime_confidence': 0.8,
            'regime_name': 'bullish',
            'strategy_min_rr': 0.5
        }
        
        target_rr = rule._calculate_dynamic_target(signal)
        assert target_rr == pytest.approx(0.8865, rel=1e-3)

        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "0.89" in reason  # Reason should surface the rounded dynamic target
    
    def test_low_confidence_tightens_rr(self):
        """Low confidence should increase R/R requirement."""
        config = RiskConfiguration({
            'equity_usd': 10000,
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
        
        # Low confidence signal with R/R = 1.2
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,      # Risk = 1
            'target': 101.2,  # Reward = 1.2, R/R = 1.2
            'ml_confidence': 0.3,
            'rl_is_agree': False,
            'rl_action_prob': 0.4,
            'regime_confidence': 0.4,
            'regime_name': 'volatile',
            'strategy_min_rr': 0.5
        }
        
        target_rr = rule._calculate_dynamic_target(signal)
        assert target_rr == pytest.approx(1.836, rel=1e-3)

        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "1.84" in reason

    def test_ppo_rr_multiplier_increases_requirement(self):
        """PPO RR multiplier should tighten the required R/R threshold."""
        config = RiskConfiguration({
            'equity_usd': 10000,
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.0,
                'lower_bound_rr': 0.5,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.0,
                    'rl_agreement': 0.0,
                    'regime_clarity': 0.0,
                    'volume_strength': 0.0,
                    'momentum_strength': 0.0
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

        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 101,
            'ml_confidence': 0.5,
            'rl_is_agree': True,
            'rl_action_prob': 0.5,
            'regime_confidence': 0.6,
            'regime_name': 'neutral',
            'strategy_min_rr': 0.5,
            'ppo_rr_multiplier': 1.3
        }

        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "1.30" in reason
    
    def test_respects_bounds(self):
        """Dynamic target should stay within configured bounds."""
        config = RiskConfiguration({
            'equity_usd': 10000,
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 1.0,  # Extreme weight for testing
                    'rl_agreement': 0.0,
                    'regime_clarity': 0.0,
                    'volume_strength': 0.0,
                    'momentum_strength': 0.0
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
        portfolio = MockPortfolioManager()
        
        # Extreme high confidence - should hit lower bound
        signal_high = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 100.75,  # R/R = 0.75
            'ml_confidence': 1.0,  # Perfect confidence
            'regime_name': 'neutral',
            'strategy_min_rr': 0.5
        }
        
        target_rr = rule._calculate_dynamic_target(signal_high)
        # With ml_weight = 1.0 and ml_conf = 1.0:
        # Relaxation = 1.0, Dynamic = 1.5 - 1.0 = 0.5
        # But lower_bound is 0.8, so final should be 0.8
        assert target_rr >= 0.8
        assert target_rr <= 2.0
    
    def test_respects_strategy_minimum(self):
        """Dynamic R/R should never go below strategy's minimum."""
        config = RiskConfiguration({
            'equity_usd': 10000,
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
        
        # High confidence but high strategy minimum
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 101.8,
            'ml_confidence': 0.9,
            'rl_is_agree': True,
            'rl_action_prob': 0.9,
            'regime_confidence': 0.8,
            'regime_name': 'neutral',
            'strategy_min_rr': 1.5  # Strategy requires 1.5 minimum
        }
        
        target_rr = rule._calculate_dynamic_target(signal)
        assert target_rr >= 1.5  # Should respect strategy minimum

    def test_v2_includes_volume_and_momentum(self):
        """RR model v2 should include volume_strength and momentum_strength contributions."""
        base_rr_cfg = {
            'enabled': True,
            'base_target_rr': 2.0,
            'lower_bound_rr': 0.0,
            'upper_bound_rr': 10.0,
            'weights': {
                'ml_confidence': 0.0,
                'rl_agreement': 0.0,
                'regime_clarity': 0.0,
                'volume_strength': 0.1,
                'momentum_strength': 0.1,
            },
            'fallback': {
                'missing_ml_default': 0.5,
                'missing_rl_default': 0.5,
                'missing_regime_default': 0.3,
            },
            'regime_multipliers': {
                'neutral': 1.0,
            },
        }

        cfg_v1 = RiskConfiguration({'equity_usd': 10000, 'rr_dynamic': {**base_rr_cfg, 'model_version': 'v1'}})
        cfg_v2 = RiskConfiguration({'equity_usd': 10000, 'rr_dynamic': {**base_rr_cfg, 'model_version': 'v2'}})
        rule_v1 = RiskRewardRatioRule(config=cfg_v1)
        rule_v2 = RiskRewardRatioRule(config=cfg_v2)

        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 102,
            'regime_name': 'neutral',
            'regime_confidence': 1.0,
            'strategy_min_rr': 0.0,
            'volume_strength': 1.0,
            'momentum_strength': 1.0,
        }

        assert rule_v1._calculate_dynamic_target(signal) == pytest.approx(2.0, rel=1e-6)
        assert rule_v2._calculate_dynamic_target(signal) == pytest.approx(1.8, rel=1e-6)


class TestDynamicRRValidation:
    """Test R/R validation with dynamic thresholds."""
    
    def test_valid_rr_passes(self):
        """Signal with R/R above dynamic threshold should pass."""
        config = RiskConfiguration({
            'equity_usd': 10000,
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
        portfolio = MockPortfolioManager()
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 102,  # R/R = 2.0
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.7,
            'regime_confidence': 0.7,
            'regime_name': 'neutral',
            'strategy_min_rr': 0.5
        }
        
        target_rr = rule._calculate_dynamic_target(signal)
        assert target_rr == pytest.approx(1.14, rel=1e-2)

        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True
        assert "1.14" in reason
    
    def test_disabled_dynamic_uses_static(self):
        """When dynamic R/R is disabled, should use static threshold."""
        config = RiskConfiguration({
            'equity_usd': 10000,
            'rr_dynamic': {
                'enabled': False,
                'base_target_rr': 1.5
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        portfolio = MockPortfolioManager()
        
        # R/R = 1.0, which is below static 1.5
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 99,
            'target': 101,
            'ml_confidence': 0.9,  # High confidence, but dynamic is disabled
            'strategy_min_rr': 0.5
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False  # Should fail with static 1.5 threshold
    
    def test_missing_price_data_fails(self):
        """Signal with missing price data should fail validation."""
        config = RiskConfiguration({'equity_usd': 10000, 'rr_dynamic': {'enabled': True}})
        rule = RiskRewardRatioRule(config=config)
        portfolio = MockPortfolioManager()
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            # Missing stop and target
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert 'Invalid price levels' in reason or 'Missing' in reason
    
    def test_zero_risk_fails(self):
        """Signal with zero risk (stop = entry) should fail."""
        config = RiskConfiguration({'equity_usd': 10000, 'rr_dynamic': {'enabled': True}})
        rule = RiskRewardRatioRule(config=config)
        portfolio = MockPortfolioManager()
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100,
            'stop': 100,  # Same as entry
            'target': 102
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert 'Zero risk' in reason


class TestRegimeMultipliers:
    """Test regime-specific multipliers."""
    
    def test_bullish_regime_reduces_rr(self):
        """Bullish regime should have multiplier < 1.0."""
        config = RiskConfiguration({
            'equity_usd': 10000,
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.0,
                    'rl_agreement': 0.0,
                    'regime_clarity': 0.0,
                    'volume_strength': 0.0,
                    'momentum_strength': 0.0
                },
                'regime_multipliers': {
                    'bullish': 0.9,
                    'neutral': 1.0
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        
        signal_bullish = {
            'symbol': 'BTC/USDT',
            'regime_name': 'bullish',
            'strategy_min_rr': 0.5
        }
        
        signal_neutral = {
            'symbol': 'BTC/USDT',
            'regime_name': 'neutral',
            'strategy_min_rr': 0.5
        }
        
        target_bullish = rule._calculate_dynamic_target(signal_bullish)
        target_neutral = rule._calculate_dynamic_target(signal_neutral)
        
        assert target_bullish < target_neutral
    
    def test_volatile_regime_increases_rr(self):
        """Volatile regime should have multiplier > 1.0."""
        config = RiskConfiguration({
            'equity_usd': 10000,
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.0,
                    'rl_agreement': 0.0,
                    'regime_clarity': 0.0,
                    'volume_strength': 0.0,
                    'momentum_strength': 0.0
                },
                'regime_multipliers': {
                    'volatile': 1.2,
                    'neutral': 1.0
                }
            }
        })
        
        rule = RiskRewardRatioRule(config=config)
        
        signal_volatile = {
            'symbol': 'BTC/USDT',
            'regime_name': 'volatile',
            'strategy_min_rr': 0.5
        }
        
        signal_neutral = {
            'symbol': 'BTC/USDT',
            'regime_name': 'neutral',
            'strategy_min_rr': 0.5
        }
        
        target_volatile = rule._calculate_dynamic_target(signal_volatile)
        target_neutral = rule._calculate_dynamic_target(signal_neutral)
        
        assert target_volatile > target_neutral


class TestStrategyOverrides:
    """Validate per-strategy override behavior for dynamic RR."""

    def _build_base_config(self):
        return RiskConfiguration({
            'equity_usd': 10000,
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.0,
                    'rl_agreement': 0.0,
                    'regime_clarity': 0.0,
                    'volume_strength': 0.0,
                    'momentum_strength': 0.0
                },
                'regime_multipliers': {
                    'bullish': 0.9,
                    'neutral': 1.0
                },
                'strategy_overrides': {
                    'scalper': {
                        'base_target_rr': 1.1,
                        'regime_multipliers': {
                            'bullish': 0.8
                        },
                        'lower_bound_rr': 0.7
                    }
                }
            }
        })

    def test_strategy_override_applies_case_insensitive(self):
        config = self._build_base_config()
        rule = RiskRewardRatioRule(config=config)

        base_signal = {
            'symbol': 'BTC/USDT',
            'regime_name': 'bullish',
            'strategy_min_rr': 0.5
        }

        override_signal = {
            **base_signal,
            'strategy_name': 'Scalper'  # Different casing than config key
        }

        base_target = rule._calculate_dynamic_target(base_signal)
        override_target = rule._calculate_dynamic_target(override_signal)

        assert base_target == pytest.approx(1.35, rel=1e-2)
        assert override_target == pytest.approx(0.88, rel=1e-2)

    def test_strategy_override_can_change_weights(self):
        config = RiskConfiguration({
            'equity_usd': 10000,
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.1,
                    'rl_agreement': 0.0,
                    'regime_clarity': 0.0,
                    'volume_strength': 0.0,
                    'momentum_strength': 0.0
                },
                'regime_multipliers': {
                    'neutral': 1.0
                },
                'strategy_overrides': {
                    'mean_reversion': {
                        'weights': {
                            'ml_confidence': 0.5
                        }
                    }
                }
            }
        })

        rule = RiskRewardRatioRule(config=config)

        base_signal = {
            'symbol': 'BTC/USDT',
            'strategy_min_rr': 0.5,
            'ml_confidence': 1.0,
            'regime_name': 'neutral'
        }

        override_signal = {
            **base_signal,
            'strategy_name': 'mean_reversion'
        }

        base_target = rule._calculate_dynamic_target(base_signal)
        override_target = rule._calculate_dynamic_target(override_signal)

        # Higher ML weight should reduce the required RR further
        assert override_target < base_target
        assert base_target == pytest.approx(1.4, rel=1e-2)
        assert override_target == pytest.approx(1.0, rel=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
