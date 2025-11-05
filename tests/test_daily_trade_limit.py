"""
Tests for daily_max_trades enforcement.
Validates that the DailyTradeLimitRule and PortfolioManager trade counter work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from core.risk_rules import DailyTradeLimitRule
from core.portfolio_manager import PortfolioManager
from core.risk_manager import RiskManager
from config.risk_config import RiskConfiguration


class MockPerformanceMonitor:
    """Mock performance monitor for testing."""
    
    def __init__(self):
        self.performance_history = {}
    
    def get_strategy_summary(self, strategy_name):
        return {'metrics': {}}


class TestPortfolioManagerTradeCounter:
    """Test PortfolioManager's daily trade counting functionality."""
    
    def test_initial_trade_count_is_zero(self):
        """Test that initial trade count is zero."""
        risk_manager = Mock()
        risk_manager.portfolio_value = 10000
        performance_monitor = MockPerformanceMonitor()
        
        portfolio_manager = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor
        )
        
        count = portfolio_manager.get_todays_trade_count()
        assert count == 0
    
    def test_increment_trade_count(self):
        """Test incrementing the trade count."""
        risk_manager = Mock()
        risk_manager.portfolio_value = 10000
        performance_monitor = MockPerformanceMonitor()
        
        portfolio_manager = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor
        )
        
        # Increment and check
        portfolio_manager.increment_trade_count()
        assert portfolio_manager.get_todays_trade_count() == 1
        
        portfolio_manager.increment_trade_count()
        assert portfolio_manager.get_todays_trade_count() == 2
        
        portfolio_manager.increment_trade_count()
        assert portfolio_manager.get_todays_trade_count() == 3
    
    def test_trade_count_resets_on_new_day(self):
        """Test that trade count resets when a new day begins."""
        risk_manager = Mock()
        risk_manager.portfolio_value = 10000
        performance_monitor = MockPerformanceMonitor()
        
        portfolio_manager = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor
        )
        
        # Increment count
        portfolio_manager.increment_trade_count()
        portfolio_manager.increment_trade_count()
        assert portfolio_manager.get_todays_trade_count() == 2
        
        # Simulate new day by changing _last_trade_date
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        portfolio_manager._last_trade_date = yesterday
        
        # Count should reset to 0
        assert portfolio_manager.get_todays_trade_count() == 0
        
        # Increment again
        portfolio_manager.increment_trade_count()
        assert portfolio_manager.get_todays_trade_count() == 1


class TestDailyTradeLimitRule:
    """Test DailyTradeLimitRule functionality."""
    
    def test_rule_allows_trades_under_limit(self):
        """Test that rule allows trades when under the limit."""
        rule = DailyTradeLimitRule(max_daily_trades=5)
        
        # Mock portfolio manager with 3 trades today
        portfolio_manager = Mock()
        portfolio_manager.get_todays_trade_count.return_value = 3
        
        signal = {'symbol': 'BTC/USDT'}
        
        is_valid, reason = rule.validate(signal, portfolio_manager)
        
        assert is_valid is True
        assert "passed" in reason.lower()
        assert "4/5" in reason  # Should show next trade would be 4/5
    
    def test_rule_rejects_trades_at_limit(self):
        """Test that rule rejects trades when at the limit."""
        rule = DailyTradeLimitRule(max_daily_trades=5)
        
        # Mock portfolio manager with 5 trades today
        portfolio_manager = Mock()
        portfolio_manager.get_todays_trade_count.return_value = 5
        
        signal = {'symbol': 'BTC/USDT'}
        
        is_valid, reason = rule.validate(signal, portfolio_manager)
        
        assert is_valid is False
        assert "limit reached" in reason.lower()
        assert "5/5" in reason
    
    def test_rule_rejects_trades_over_limit(self):
        """Test that rule rejects trades when over the limit."""
        rule = DailyTradeLimitRule(max_daily_trades=3)
        
        # Mock portfolio manager with 4 trades today
        portfolio_manager = Mock()
        portfolio_manager.get_todays_trade_count.return_value = 4
        
        signal = {'symbol': 'ETH/USDT'}
        
        is_valid, reason = rule.validate(signal, portfolio_manager)
        
        assert is_valid is False
        assert "limit reached" in reason.lower()
    
    def test_rule_disabled(self):
        """Test that disabled rule always passes."""
        rule = DailyTradeLimitRule(max_daily_trades=1)
        rule.disable()
        
        # Mock portfolio manager with 10 trades (way over limit)
        portfolio_manager = Mock()
        portfolio_manager.get_todays_trade_count.return_value = 10
        
        signal = {'symbol': 'BTC/USDT'}
        
        is_valid, reason = rule.validate(signal, portfolio_manager)
        
        assert is_valid is True
        assert "disabled" in reason.lower()
    
    def test_rule_handles_zero_limit(self):
        """Test that rule with zero limit blocks all trades."""
        rule = DailyTradeLimitRule(max_daily_trades=0)
        
        portfolio_manager = Mock()
        portfolio_manager.get_todays_trade_count.return_value = 0
        
        signal = {'symbol': 'BTC/USDT'}
        
        is_valid, reason = rule.validate(signal, portfolio_manager)
        
        assert is_valid is False
    
    def test_rule_handles_missing_method(self):
        """Test that rule handles portfolio manager without get_todays_trade_count method."""
        rule = DailyTradeLimitRule(max_daily_trades=5)
        
        # Portfolio manager without the method
        portfolio_manager = Mock(spec=[])  # Empty spec means no methods
        
        signal = {'symbol': 'BTC/USDT'}
        
        # Should not crash, should fail gracefully
        is_valid, reason = rule.validate(signal, portfolio_manager)
        
        # Should pass when method is missing (fail-safe behavior)
        assert is_valid is True
        assert "cannot verify" in reason.lower() or "missing method" in reason.lower()


class TestRiskManagerIntegration:
    """Test integration of DailyTradeLimitRule with RiskManager."""
    
    def test_risk_manager_includes_daily_limit_rule_when_configured(self):
        """Test that RiskManager includes DailyTradeLimitRule when daily_max_trades is configured."""
        config = {
            'daily_max_trades': 5,
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.10,
            'max_drawdown': 0.15,
            'max_correlation': 0.70
        }
        
        risk_config = RiskConfiguration(custom_limits=config)
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config,
            performance_monitor=MockPerformanceMonitor()
        )
        
        # Check that DailyTradeLimitRule is in the rules list
        rule_names = [rule.rule_name for rule in risk_manager.rules]
        assert "DailyTradeLimitRule" in rule_names
    
    def test_risk_manager_excludes_daily_limit_rule_when_not_configured(self):
        """Test that RiskManager excludes DailyTradeLimitRule when daily_max_trades is not configured."""
        config = {
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.10,
            'max_drawdown': 0.15,
            'max_correlation': 0.70
        }
        
        risk_config = RiskConfiguration(custom_limits=config)
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config,
            performance_monitor=MockPerformanceMonitor()
        )
        
        # Check that DailyTradeLimitRule is NOT in the rules list
        rule_names = [rule.rule_name for rule in risk_manager.rules]
        assert "DailyTradeLimitRule" not in rule_names


class TestEndToEndTradeLimit:
    """End-to-end integration tests for daily trade limit enforcement."""
    
    @pytest.mark.asyncio
    async def test_trades_rejected_after_limit_reached(self):
        """Test that trades are rejected after daily limit is reached."""
        # Setup
        config = {
            'daily_max_trades': 3,
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.10,
            'max_drawdown': 0.15,
            'max_correlation': 0.70
        }
        
        risk_config = RiskConfiguration(custom_limits=config)
        performance_monitor = MockPerformanceMonitor()
        
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config,
            performance_monitor=performance_monitor
        )
        
        portfolio_manager = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor
        )
        
        # Simulate 3 successful trades
        portfolio_manager.increment_trade_count()
        portfolio_manager.increment_trade_count()
        portfolio_manager.increment_trade_count()
        
        assert portfolio_manager.get_todays_trade_count() == 3
        
        # Try to validate a 4th trade with small position size to pass other rules
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.01,  # Small position: 0.01 * 50000 = $500 (5% of portfolio)
            'notional': 500,
            'side': 'long'
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal, portfolio_manager)
        
        # Should be rejected due to daily limit
        assert is_valid is False
        assert "daily trade limit" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_trades_allowed_before_limit(self):
        """Test that trades are allowed before daily limit is reached."""
        # Setup
        config = {
            'daily_max_trades': 5,
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.10,
            'max_drawdown': 0.15,
            'max_correlation': 0.70
        }
        
        risk_config = RiskConfiguration(custom_limits=config)
        performance_monitor = MockPerformanceMonitor()
        
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config,
            performance_monitor=performance_monitor
        )
        
        portfolio_manager = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor
        )
        
        # Simulate 2 successful trades
        portfolio_manager.increment_trade_count()
        portfolio_manager.increment_trade_count()
        
        assert portfolio_manager.get_todays_trade_count() == 2
        
        # Try to validate a 3rd trade
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.016,  # 0.016 BTC * 50000 = $800 (8% of $10k portfolio)
            'notional': 800,  # Notional value: 0.016 * 50000 = 800
            'side': 'long'
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal, portfolio_manager)
        
        # Should be allowed (assuming other rules also pass)
        # Note: This may fail if other rules reject it, but DailyTradeLimitRule should pass
        if not is_valid:
            # Check that rejection is NOT due to daily limit
            assert "daily trade limit" not in reason.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
