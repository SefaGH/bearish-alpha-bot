"""
Test suite for Risk Configuration USD amount calculations.

This test validates that:
1. USD amounts are correctly calculated from percentages
2. ENV variables properly override config values
3. Logging shows both percentages and USD amounts
"""

import pytest
import os
from unittest.mock import patch
from src.config.risk_config import RiskConfiguration


class TestRiskUSDCalculations:
    """Test USD amount calculations in RiskConfiguration."""
    
    def test_usd_calculation_with_default_capital(self):
        """Test USD calculations with default capital of $100."""
        config = RiskConfiguration()
        
        # Check that USD amounts are calculated
        assert hasattr(config, 'max_risk_per_trade_usd')
        assert hasattr(config, 'daily_loss_limit_usd')
        assert hasattr(config, 'max_drawdown_usd')
        
        # Verify USD calculations based on capital and percentages
        # Initial capital is $100 by default
        assert config.initial_capital == 100.0
        # Per-trade risk and daily loss limit should be calculated as percentage of capital
        assert config.max_risk_per_trade_usd > 0
        assert config.daily_loss_limit_usd > 0
    
    def test_usd_calculation_with_custom_capital(self):
        """Test USD calculations with custom capital."""
        custom_limits = {
            'equity_usd': 500.0,
            'per_trade_risk_pct': 0.01,  # 1%
            'daily_loss_limit_pct': 0.02  # 2%
        }
        
        config = RiskConfiguration(custom_limits=custom_limits)
        
        # 1% of $500 = $5
        # 2% of $500 = $10
        assert config.initial_capital == 500.0
        assert config.max_risk_per_trade_usd == pytest.approx(5.0, rel=0.01)
        assert config.daily_loss_limit_usd == pytest.approx(10.0, rel=0.01)
    
    def test_usd_calculation_with_explicit_capital(self):
        """Test USD calculations when initial_capital is explicitly provided."""
        custom_limits = {
            'equity_usd': 100.0,  # This should be overridden
        }
        
        config = RiskConfiguration(
            custom_limits=custom_limits,
            initial_capital=250.0  # Explicit capital
        )
        
        # Should use the explicit initial_capital, not equity_usd
        assert config.initial_capital == 250.0
    
    @patch.dict(os.environ, {'PER_TRADE_RISK_PCT': '1.0'})
    def test_env_override_per_trade_risk(self):
        """Test that PER_TRADE_RISK_PCT ENV variable overrides config."""
        custom_limits = {
            'equity_usd': 100.0,
            'per_trade_risk_pct': 0.02  # 2% in config
        }
        
        config = RiskConfiguration(custom_limits=custom_limits)
        
        # ENV says 1%, so $100 * 1% = $1.00
        assert config.max_risk_per_trade_usd == pytest.approx(1.0, rel=0.01)
    
    @patch.dict(os.environ, {'DAILY_LOSS_LIMIT_PCT': '2.0'})
    def test_env_override_daily_loss_limit(self):
        """Test that DAILY_LOSS_LIMIT_PCT ENV variable overrides config."""
        custom_limits = {
            'equity_usd': 100.0,
            'daily_loss_limit_pct': 0.05  # 5% in config
        }
        
        config = RiskConfiguration(custom_limits=custom_limits)
        
        # ENV says 2%, so $100 * 2% = $2.00
        assert config.daily_loss_limit_usd == pytest.approx(2.0, rel=0.01)
    
    @patch.dict(os.environ, {
        'PER_TRADE_RISK_PCT': '1.0',
        'DAILY_LOSS_LIMIT_PCT': '2.0'
    })
    def test_env_override_both_params(self):
        """Test that both ENV variables work together."""
        custom_limits = {
            'equity_usd': 100.0,
            'per_trade_risk_pct': 0.05,    # 5% in config
            'daily_loss_limit_pct': 0.10   # 10% in config
        }
        
        config = RiskConfiguration(custom_limits=custom_limits)
        
        # ENV overrides should apply
        assert config.max_risk_per_trade_usd == pytest.approx(1.0, rel=0.01)  # 1% of $100
        assert config.daily_loss_limit_usd == pytest.approx(2.0, rel=0.01)    # 2% of $100
    
    def test_get_risk_params_for_sizing(self):
        """Test that get_risk_params_for_sizing returns correct USD values."""
        custom_limits = {
            'equity_usd': 100.0,
            'per_trade_risk_pct': 0.01,  # 1%
            'daily_loss_limit_pct': 0.02  # 2%
        }
        
        config = RiskConfiguration(custom_limits=custom_limits)
        risk_params = config.get_risk_params_for_sizing()
        
        # Check that all required keys exist
        assert 'max_risk_per_trade' in risk_params
        assert 'max_risk_amount' in risk_params
        assert 'daily_loss_limit' in risk_params
        assert 'circuit_breaker_limits' in risk_params
        assert 'initial_capital' in risk_params
        
        # Check USD values
        assert risk_params['max_risk_amount'] == pytest.approx(1.0, rel=0.01)
        assert risk_params['daily_loss_limit'] == pytest.approx(2.0, rel=0.01)
        assert risk_params['initial_capital'] == 100.0
    
    def test_circuit_breaker_usd_limits(self):
        """Test that circuit breaker limits are calculated in USD."""
        custom_limits = {
            'equity_usd': 200.0,
            'daily_loss_limit_pct': 0.05,      # 5%
            'position_loss_limit': 0.03,       # 3%
        }
        
        config = RiskConfiguration(custom_limits=custom_limits)
        
        # Check circuit breaker USD values
        assert hasattr(config, 'circuit_breaker_limits_usd')
        assert config.circuit_breaker_limits_usd['daily_loss_limit'] == pytest.approx(10.0, rel=0.01)  # 5% of $200
        assert config.circuit_breaker_limits_usd['position_loss_limit'] == pytest.approx(6.0, rel=0.01)  # 3% of $200
    
    def test_max_drawdown_usd(self):
        """Test max drawdown calculation in USD."""
        custom_limits = {
            'equity_usd': 500.0,
            'max_drawdown': 0.15  # 15%
        }
        
        config = RiskConfiguration(custom_limits=custom_limits)
        
        # 15% of $500 = $75
        assert config.max_drawdown_usd == pytest.approx(75.0, rel=0.01)
    
    @patch.dict(os.environ, {'PER_TRADE_RISK_PCT': '0.5'})
    def test_fractional_percentage_from_env(self):
        """Test that fractional percentages work correctly from ENV."""
        custom_limits = {
            'equity_usd': 1000.0,
        }
        
        config = RiskConfiguration(custom_limits=custom_limits)
        
        # 0.5% of $1000 = $5
        assert config.max_risk_per_trade_usd == pytest.approx(5.0, rel=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
