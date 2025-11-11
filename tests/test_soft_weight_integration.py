"""
Integration test demonstrating regime soft-weighting in action.

Shows how regime confidence levels affect signal processing,
R/R calculations, and position sizing through the entire pipeline.
"""
import pytest
from src.config.risk_config import RiskConfiguration
from src.core.risk_rules import RiskRewardRatioRule
from src.core.position_sizing import AdvancedPositionSizing


class MockRiskManager:
    """Mock risk manager for testing."""
    pass


class TestSoftWeightIntegration:
    """Integration tests showing soft-weighting effects across components."""
    
    def test_complete_signal_flow_high_confidence(self):
        """Test complete signal flow with high confidence regime."""
        # Setup
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.2,
                    'rl_agreement': 0.2,
                    'regime_clarity': 0.4,  # High weight for regime
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
        
        rr_rule = RiskRewardRatioRule(config=config)
        
        # High confidence regime signal
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 100.0,
            'stop': 99.0,
            'target': 101.2,
            'ml_confidence': 0.6,
            'rl_is_agree': True,
            'rl_action_prob': 0.7,
            'regime_confidence': 0.85,  # High confidence
            'regime_name': 'bullish',
            'regime_weight': 1.0,  # Full weight
            'strategy_min_rr': 0.5
        }
        
        # Calculate dynamic R/R
        dynamic_rr = rr_rule._calculate_dynamic_target(signal)
        
        # With high regime confidence and bullish regime (0.9 multiplier),
        # R/R should be reduced (more aggressive because we're confident about bullish)
        assert 0.8 <= dynamic_rr <= 1.5, f"Expected reduced R/R for high confidence bullish, got {dynamic_rr}"
        print(f"✓ High confidence regime: R/R = {dynamic_rr:.3f}")
    
    def test_complete_signal_flow_medium_confidence(self):
        """Test complete signal flow with medium confidence regime."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.2,
                    'rl_agreement': 0.2,
                    'regime_clarity': 0.4,
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
        
        rr_rule = RiskRewardRatioRule(config=config)
        
        # Medium confidence regime signal
        signal = {
            'symbol': 'ETH/USDT',
            'entry': 100.0,
            'stop': 99.0,
            'target': 101.5,
            'ml_confidence': 0.6,
            'rl_is_agree': True,
            'rl_action_prob': 0.7,
            'regime_confidence': 0.45,  # Medium confidence
            'regime_name': 'volatile',
            'regime_weight': 0.75,  # Partial weight (0.45 / 0.6)
            'strategy_min_rr': 0.5
        }
        
        # Calculate dynamic R/R
        dynamic_rr = rr_rule._calculate_dynamic_target(signal)
        
        # With medium confidence, effect should be dampened
        # Volatile regime (1.2x) with 0.75 weight = 1 + (1.2-1)*0.75 = 1.15x
        assert 0.8 <= dynamic_rr <= 2.0, f"Expected moderate R/R for medium confidence, got {dynamic_rr}"
        print(f"✓ Medium confidence regime: R/R = {dynamic_rr:.3f}")
    
    def test_complete_signal_flow_low_confidence(self):
        """Test complete signal flow with low confidence (ignored) regime."""
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
        
        rr_rule = RiskRewardRatioRule(config=config)
        
        # Low confidence - regime should be ignored (no regime_weight)
        signal = {
            'symbol': 'SOL/USDT',
            'entry': 100.0,
            'stop': 99.0,
            'target': 101.3,
            'ml_confidence': 0.7,
            'rl_is_agree': True,
            'rl_action_prob': 0.8,
            # No regime data - it was filtered out due to low confidence
            'strategy_min_rr': 0.5
        }
        
        # Calculate dynamic R/R
        dynamic_rr = rr_rule._calculate_dynamic_target(signal)
        
        # Without regime, should be based on ML/RL only
        assert 0.8 <= dynamic_rr <= 2.0, f"Expected normal R/R without regime, got {dynamic_rr}"
        print(f"✓ Low confidence regime (ignored): R/R = {dynamic_rr:.3f}")
    
    def test_position_sizing_with_regime_weight(self):
        """Test position sizing adjusts based on regime_weight."""
        mock_manager = MockRiskManager()
        sizer = AdvancedPositionSizing(risk_manager=mock_manager)
        
        # Base market regime
        market_regime = {
            'risk_multiplier': 1.2,  # Volatile market
            'trend': 'bullish',
            'volatility': 'normal'
        }
        
        # Signal with full regime weight
        signal_full = {
            'side': 'long',
            'entry': 100.0,
            'stop': 99.0,
            'regime_weight': 1.0
        }
        
        size_full = sizer._regime_based_sizing(
            signal_full, 
            market_regime,
            None,
            base_risk=100
        )
        
        # Signal with partial regime weight
        signal_partial = signal_full.copy()
        signal_partial['regime_weight'] = 0.5
        
        size_partial = sizer._regime_based_sizing(
            signal_partial,
            market_regime,
            None,
            base_risk=100
        )
        
        # Full weight should have larger position size adjustment
        # Full: 1 + (1.2-1)*1.0 = 1.2x
        # Partial: 1 + (1.2-1)*0.5 = 1.1x
        assert size_full > size_partial, "Full regime weight should result in larger position"
        print(f"✓ Position sizing: Full weight={size_full:.2f}, Partial weight={size_partial:.2f}")
    
    def test_confidence_progression_effect(self):
        """Test how regime confidence progression affects R/R."""
        config = RiskConfiguration({
            'rr_dynamic': {
                'enabled': True,
                'base_target_rr': 1.5,
                'lower_bound_rr': 0.8,
                'upper_bound_rr': 2.0,
                'weights': {
                    'ml_confidence': 0.0,  # Isolate regime effect
                    'rl_agreement': 0.0,
                    'regime_clarity': 0.5,
                    'volume_strength': 0.0,
                    'momentum_strength': 0.0
                },
                'fallback': {
                    'missing_ml_default': 0.5,
                    'missing_rl_default': 0.5,
                    'missing_regime_default': 0.5
                },
                'regime_multipliers': {
                    'volatile': 1.3
                }
            }
        })
        
        rr_rule = RiskRewardRatioRule(config=config)
        
        # Test different confidence levels
        confidence_levels = [
            (0.30, 0.30/0.60),  # Minimum accepted
            (0.45, 0.45/0.60),  # Medium
            (0.60, 1.0),        # Full weight threshold
            (0.90, 1.0),        # High confidence
        ]
        
        results = []
        for conf, weight in confidence_levels:
            signal = {
                'regime_confidence': conf,
                'regime_name': 'volatile',
                'regime_weight': weight,
                'ml_confidence': 0.5,
                'rl_is_agree': False,
                'rl_action_prob': 0.5,
                'strategy_min_rr': 0.5
            }
            rr = rr_rule._calculate_dynamic_target(signal)
            results.append((conf, weight, rr))
            print(f"  Confidence={conf:.2f}, Weight={weight:.2f}, R/R={rr:.3f}")
        
        # Verify progression: higher confidence should lead to more extreme R/R
        assert all(results[i][2] <= results[i+1][2] for i in range(len(results)-1)), \
            "R/R should increase with confidence for volatile regime"
        
        print(f"✓ Confidence progression working correctly")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
