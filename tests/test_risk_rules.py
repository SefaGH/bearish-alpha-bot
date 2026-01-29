"""
Tests for Phase 3: Risk Rules Engine.
Tests individual risk rule classes and rules engine integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock

from core.risk_rules import (
    BaseRiskRule,
    CapitalLimitRule,
    PositionSizeRule,
    PortfolioHeatRule,
    MaxDrawdownRule,
    RiskRewardRatioRule,
    StrategyPerformanceRule,
    DailyTradeLimitRule,
)


class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    
    def __init__(self, equity=10000, exposure=0, drawdown=0.0, positions=None):
        self.equity = equity
        self.exposure = exposure
        self.drawdown = drawdown
        self.positions = positions or {}
    
    def get_current_equity(self):
        return self.equity
    
    def get_total_exposure(self):
        return self.exposure
    
    def get_current_drawdown(self):
        return self.drawdown
    
    def get_open_positions(self):
        return self.positions


class MockDailyLimitPortfolio:
    def __init__(
        self,
        *,
        trades_today: int,
        todays_pnl_usd: float = 0.0,
        pnl_since_start_usd: float = 0.0,
        todays_start_equity_usd: float = 1000.0,
        todays_drawdown_pct: float = 0.0,
        current_drawdown: float = 0.0,
    ):
        self._trades_today = trades_today
        self._todays_pnl_usd = todays_pnl_usd
        self._pnl_since_start_usd = pnl_since_start_usd
        self._todays_start_equity_usd = todays_start_equity_usd
        self._todays_drawdown_pct = todays_drawdown_pct
        self._current_drawdown = current_drawdown

    def get_todays_trade_count(self) -> int:
        return self._trades_today

    def get_todays_pnl_usd(self) -> float:
        return self._todays_pnl_usd

    def get_todays_start_equity_usd(self) -> float:
        return self._todays_start_equity_usd

    def get_todays_drawdown_pct(self) -> float:
        return self._todays_drawdown_pct

    def get_current_drawdown(self) -> float:
        return self._current_drawdown

    def get_pnl_since_start_usd(self) -> float:
        return self._pnl_since_start_usd


class TestBaseRiskRule:
    """Test base risk rule functionality."""
    
    def test_base_rule_abstract(self):
        """Test that BaseRiskRule is abstract."""
        with pytest.raises(TypeError):
            BaseRiskRule()
    
    def test_rule_enable_disable(self):
        """Test enabling and disabling rules."""
        rule = PositionSizeRule()
        
        assert rule.enabled is True
        
        rule.disable()
        assert rule.enabled is False
        
        rule.enable()
        assert rule.enabled is True
    
    def test_rule_name(self):
        """Test rule naming."""
        rule = PositionSizeRule()
        assert rule.rule_name == "PositionSizeRule"
        
        rule = PositionSizeRule(rule_name="CustomName")
        assert rule.rule_name == "CustomName"


class TestCapitalLimitRule:
    """Test capital limit validation rule."""
    
    def test_within_capital_limit(self):
        """Test position that fits within capital limit."""
        rule = CapitalLimitRule()
        portfolio = MockPortfolioManager(equity=10000, exposure=5000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.08,  # $4000 position
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True
    
    def test_exceeds_capital_limit(self):
        """Test position that exceeds capital limit."""
        rule = CapitalLimitRule()
        portfolio = MockPortfolioManager(equity=10000, exposure=8000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.05,  # $2500 position, total would be $10500
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "exceeds affordable" in reason.lower()
    
    def test_disabled_rule(self):
        """Test disabled rule always passes."""
        rule = CapitalLimitRule()
        rule.disable()
        
        portfolio = MockPortfolioManager(equity=10000, exposure=12000)  # Over limit
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.1,
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True


class TestPositionSizeRule:
    """Test position size validation rule."""
    
    def test_within_position_size_limit(self):
        """Test position within size limit."""
        rule = PositionSizeRule(max_position_size=0.10)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.02,  # $1000 position (10% of portfolio)
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True
    
    def test_exceeds_position_size_limit(self):
        """Test position exceeding size limit."""
        rule = PositionSizeRule(max_position_size=0.10)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.025,  # $1250 position (12.5% of portfolio)
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "exceeds max" in reason.lower()
    
    def test_custom_position_size_limit(self):
        """Test custom position size limit."""
        rule = PositionSizeRule(max_position_size=0.05)  # 5% limit
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.012,  # $600 position (6% - should fail at 5% limit)
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False


class TestPortfolioHeatRule:
    """Test portfolio heat validation rule."""
    
    def test_within_portfolio_heat_limit(self):
        """Test portfolio heat within limits."""
        rule = PortfolioHeatRule(max_portfolio_heat=0.10, max_portfolio_risk=0.06)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'position_size': 0.02,  # Risk: $20 (0.2% of portfolio)
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True
    
    def test_exceeds_individual_risk_limit(self):
        """Test position exceeding individual risk limit."""
        rule = PortfolioHeatRule(max_portfolio_heat=0.10, max_portfolio_risk=0.02)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 45000,  # $5000 risk distance
            'position_size': 0.1,  # Risk: $500 (5% of portfolio - exceeds 2% limit)
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "exceeds max" in reason.lower()
    
    def test_exceeds_portfolio_heat_limit(self):
        """Test portfolio heat exceeding limit."""
        rule = PortfolioHeatRule(max_portfolio_heat=0.10, max_portfolio_risk=0.06)
        
        # Existing positions with $800 total risk (8% heat)
        existing_positions = {
            'pos1': {'risk_amount': 400},
            'pos2': {'risk_amount': 400}
        }
        portfolio = MockPortfolioManager(equity=10000, positions=existing_positions)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'position_size': 0.03,  # Risk: $30, total heat would be 8.3%
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        # Should still pass as 8.3% < 10%
        assert is_valid is True
        
        # Now test with position that would exceed
        signal['position_size'] = 0.1  # Risk: $100, total heat would be 9%
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True  # Still under 10%
        
        # Add more existing risk
        existing_positions['pos3'] = {'risk_amount': 200}
        signal['position_size'] = 0.05  # Risk: $50, total heat would be 10.5%
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "portfolio heat" in reason.lower()
    
    def test_stop_loss_calculation_atr(self):
        """Test stop loss calculation from ATR."""
        rule = PortfolioHeatRule(max_portfolio_heat=0.10, max_portfolio_risk=0.06)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'atr': 500,
            'sl_atr_mult': 2.0,
            'position_size': 0.02,
            'side': 'long'
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        # Stop should be 50000 - (500 * 2) = 49000
        # Risk: 1000 * 0.02 = $20 (0.2% of portfolio)
        assert is_valid is True
    
    def test_stop_loss_calculation_percentage(self):
        """Test stop loss calculation from percentage."""
        rule = PortfolioHeatRule(max_portfolio_heat=0.10, max_portfolio_risk=0.06)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'sl_pct': 0.02,  # 2% stop
            'position_size': 0.02,
            'side': 'long'
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        # Stop should be 50000 * (1 - 0.02) = 49000
        # Risk: 1000 * 0.02 = $20 (0.2% of portfolio)
        assert is_valid is True


class TestMaxDrawdownRule:
    """Test max drawdown validation rule."""
    
    def test_within_drawdown_limit(self):
        """Test drawdown within limit."""
        rule = MaxDrawdownRule(max_drawdown=0.15)
        portfolio = MockPortfolioManager(equity=10000, drawdown=0.10)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True
    
    def test_exceeds_drawdown_limit(self):
        """Test drawdown exceeding limit."""
        rule = MaxDrawdownRule(max_drawdown=0.15)
        portfolio = MockPortfolioManager(equity=10000, drawdown=0.20)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "drawdown" in reason.lower()
    
    def test_at_drawdown_limit(self):
        """Test drawdown exactly at limit."""
        rule = MaxDrawdownRule(max_drawdown=0.15)
        portfolio = MockPortfolioManager(equity=10000, drawdown=0.15)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        # At limit should pass
        assert is_valid is True


class TestRiskRewardRatioRule:
    """Test risk/reward ratio validation rule."""
    
    def test_acceptable_risk_reward_ratio(self):
        """Test acceptable risk/reward ratio."""
        rule = RiskRewardRatioRule(min_risk_reward=1.5)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,  # R:R = 2000/1000 = 2.0
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True
    
    def test_unacceptable_risk_reward_ratio(self):
        """Test unacceptable risk/reward ratio."""
        rule = RiskRewardRatioRule(min_risk_reward=1.5)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 50500,  # R:R = 500/1000 = 0.5
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "risk/reward" in reason.lower()
    
    def test_custom_min_risk_reward(self):
        """Test custom minimum risk/reward ratio."""
        rule = RiskRewardRatioRule(min_risk_reward=2.0)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 51500,  # R:R = 1500/1000 = 1.5
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False  # 1.5 < 2.0
    
    def test_stop_loss_from_atr(self):
        """Test risk/reward calculation with ATR-based stop."""
        rule = RiskRewardRatioRule(min_risk_reward=1.5)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'atr': 500,
            'sl_atr_mult': 2.0,
            'target': 52000,
            'side': 'long'
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        # Stop: 50000 - (500 * 2) = 49000
        # R:R = (52000-50000) / (50000-49000) = 2000/1000 = 2.0
        assert is_valid is True


class TestStrategyPerformanceRule:
    """Test strategy performance validation rule."""
    
    def test_acceptable_win_rate(self):
        """Test strategy with acceptable win rate."""
        mock_monitor = Mock()
        mock_monitor.get_strategy_summary.return_value = {
            'metrics': {'win_rate': 0.55}
        }
        
        rule = StrategyPerformanceRule(min_win_rate=0.35, performance_monitor=mock_monitor)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'strategy': 'momentum',
            'entry': 50000,
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True
    
    def test_unacceptable_win_rate(self):
        """Test strategy with unacceptable win rate."""
        mock_monitor = Mock()
        mock_monitor.get_strategy_summary.return_value = {
            'metrics': {'win_rate': 0.25}
        }
        
        rule = StrategyPerformanceRule(min_win_rate=0.35, performance_monitor=mock_monitor)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'strategy': 'momentum',
            'entry': 50000,
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "win rate" in reason.lower()
    
    def test_no_performance_monitor(self):
        """Test rule behavior without performance monitor."""
        rule = StrategyPerformanceRule(min_win_rate=0.35, performance_monitor=None)
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'strategy': 'momentum',
            'entry': 50000,
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True  # Should pass when no monitor available


class TestRulesEngineIntegration:
    """Test rules engine integration scenarios."""
    
    def test_all_rules_pass(self):
        """Test scenario where all rules pass."""
        rules = [
            CapitalLimitRule(),
            PositionSizeRule(max_position_size=0.10),
            PortfolioHeatRule(max_portfolio_heat=0.10, max_portfolio_risk=0.06),
            MaxDrawdownRule(max_drawdown=0.15),
            RiskRewardRatioRule(min_risk_reward=1.5)
        ]
        
        portfolio = MockPortfolioManager(equity=10000, exposure=5000, drawdown=0.05)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.02,
        }
        
        for rule in rules:
            is_valid, reason = rule.validate(signal, portfolio)
            assert is_valid is True, f"{rule.rule_name} failed: {reason}"
    
    def test_first_rule_fails(self):
        """Test scenario where first rule fails."""
        rules = [
            CapitalLimitRule(),
            PositionSizeRule(max_position_size=0.10),
        ]
        
        portfolio = MockPortfolioManager(equity=10000, exposure=9500)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.02,  # Would exceed capital
        }
        
        # First rule should fail
        is_valid, reason = rules[0].validate(signal, portfolio)
        assert is_valid is False
    
    def test_middle_rule_fails(self):
        """Test scenario where middle rule fails."""
        rules = [
            CapitalLimitRule(),
            PositionSizeRule(max_position_size=0.05),  # Strict limit
            MaxDrawdownRule(max_drawdown=0.15),
        ]
        
        portfolio = MockPortfolioManager(equity=10000, exposure=5000, drawdown=0.05)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.015,  # 7.5% - exceeds 5% limit
        }
        
        # First rule passes
        is_valid, _ = rules[0].validate(signal, portfolio)
        assert is_valid is True
        
        # Second rule fails
        is_valid, reason = rules[1].validate(signal, portfolio)
        assert is_valid is False
    
    def test_disabled_rule_skipped(self):
        """Test that disabled rules are skipped."""
        rule = PositionSizeRule(max_position_size=0.05)
        rule.disable()
        
        portfolio = MockPortfolioManager(equity=10000)
        
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'position_size': 0.02,  # 10% - would normally fail
        }
        
        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True
        assert "disabled" in reason.lower()


class TestDailyTradeLimitRule:
    def test_allows_under_limit(self):
        rule = DailyTradeLimitRule(max_daily_trades=2)
        portfolio = MockDailyLimitPortfolio(trades_today=1)
        signal = {"symbol": "BTC/USDT"}

        is_valid, _ = rule.validate(signal, portfolio)
        assert is_valid is True

    def test_rejects_at_static_limit(self):
        rule = DailyTradeLimitRule(max_daily_trades=2)
        portfolio = MockDailyLimitPortfolio(trades_today=2)
        signal = {"symbol": "BTC/USDT"}

        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "daily trade limit" in reason.lower()

    def test_profit_unlock_allows_extra_trades(self):
        rule = DailyTradeLimitRule(
            max_daily_trades=2,
            dynamic_config={
                "profit_unlock": {
                    "enabled": True,
                    "pnl_source": "daily",
                    "min_pnl_usd": 0.0,
                    "extra_trades": 1,
                }
            },
        )
        portfolio = MockDailyLimitPortfolio(trades_today=2, todays_pnl_usd=10.0)
        signal = {"symbol": "BTC/USDT"}

        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True, reason

        portfolio_block = MockDailyLimitPortfolio(trades_today=3, todays_pnl_usd=10.0)
        is_valid, reason = rule.validate(signal, portfolio_block)
        assert is_valid is False
        assert "3/3" in reason

    def test_profit_unlock_since_start_source(self):
        rule = DailyTradeLimitRule(
            max_daily_trades=2,
            dynamic_config={
                "profit_unlock": {
                    "enabled": True,
                    "pnl_source": "since_start",
                    "min_pnl_usd": 0.0,
                    "extra_trades": 2,
                }
            },
        )
        portfolio = MockDailyLimitPortfolio(trades_today=3, pnl_since_start_usd=5.0)
        signal = {"symbol": "BTC/USDT"}

        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is True, reason

    def test_profit_unlock_blocked_by_daily_drawdown(self):
        rule = DailyTradeLimitRule(
            max_daily_trades=2,
            dynamic_config={
                "profit_unlock": {
                    "enabled": True,
                    "pnl_source": "daily",
                    "min_pnl_usd": 0.0,
                    "max_drawdown_pct": 0.03,
                    "drawdown_source": "daily",
                    "extra_trades": 2,
                }
            },
        )
        portfolio = MockDailyLimitPortfolio(trades_today=2, todays_pnl_usd=10.0, todays_drawdown_pct=0.05)
        signal = {"symbol": "BTC/USDT"}

        is_valid, reason = rule.validate(signal, portfolio)
        assert is_valid is False
        assert "2/2" in reason

    def test_profit_unlock_requires_min_pnl_pct(self):
        rule = DailyTradeLimitRule(
            max_daily_trades=2,
            dynamic_config={
                "profit_unlock": {
                    "enabled": True,
                    "pnl_source": "daily",
                    "min_pnl_usd": 0.0,
                    "min_pnl_pct": 0.02,  # +2%
                    "extra_trades": 2,
                }
            },
        )
        signal = {"symbol": "BTC/USDT"}

        portfolio_not_enough = MockDailyLimitPortfolio(trades_today=2, todays_pnl_usd=10.0, todays_start_equity_usd=1000.0)
        is_valid, reason = rule.validate(signal, portfolio_not_enough)
        assert is_valid is False
        assert "2/2" in reason

        portfolio_enough = MockDailyLimitPortfolio(trades_today=2, todays_pnl_usd=25.0, todays_start_equity_usd=1000.0)
        is_valid, reason = rule.validate(signal, portfolio_enough)
        assert is_valid is True, reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
