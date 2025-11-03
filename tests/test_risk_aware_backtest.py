"""
Tests for Risk-Aware Backtesting Module.
Tests the integration of risk management with backtesting.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np

from backtest.risk_aware_backtest import (
    RiskAwareBacktest,
    compare_risk_configurations,
    generate_backtest_report
)


class TestRiskAwareBacktest:
    """Test risk-aware backtesting functionality."""
    
    def test_initialization(self):
        """Test backtest initialization."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        
        assert backtest.initial_capital == 10000
        assert backtest.capital == 10000
        assert backtest.peak_capital == 10000
        assert len(backtest.positions) == 0
        assert len(backtest.closed_trades) == 0
    
    def test_default_risk_config(self):
        """Test default risk configuration."""
        backtest = RiskAwareBacktest()
        config = backtest.risk_config
        
        assert 'max_position_size' in config
        assert 'max_portfolio_risk' in config
        assert 'max_drawdown' in config
        assert config['max_position_size'] == 0.10
        assert config['max_portfolio_risk'] == 0.02
    
    def test_custom_risk_config(self):
        """Test custom risk configuration."""
        custom_config = {
            'max_position_size': 0.05,
            'max_portfolio_risk': 0.01,
            'max_drawdown': 0.10,
        }
        
        backtest = RiskAwareBacktest(risk_config=custom_config)
        
        assert backtest.risk_config['max_position_size'] == 0.05
        assert backtest.risk_config['max_portfolio_risk'] == 0.01
    
    def test_validate_signal_success(self):
        """Test successful signal validation."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        
        signal = {
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.02,  # $1000 position (10% of capital)
            'side': 'long'
        }
        
        is_valid, reason = backtest._validate_signal_with_risk_rules(signal)
        assert is_valid is True
    
    def test_validate_signal_position_size_exceeded(self):
        """Test signal validation failure - position size."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        
        signal = {
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.025,  # $1250 position (12.5% - exceeds 10% limit)
            'side': 'long'
        }
        
        is_valid, reason = backtest._validate_signal_with_risk_rules(signal)
        assert is_valid is False
        assert "position size" in reason.lower()
    
    def test_validate_signal_risk_exceeded(self):
        """Test signal validation failure - risk amount."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        
        signal = {
            'entry': 50000,
            'stop': 45000,  # $5000 risk distance
            'target': 52000,
            'position_size': 0.01,  # Risk: $50 (0.5% - under position size limit but risk is $500 total which is 5%)
            'side': 'long'
        }
        
        # Actually this would pass position size check but fail risk check
        # Let's make position smaller to pass size check
        signal['position_size'] = 0.01  # $500 position (5% of capital - under 10% limit)
        # Risk: 5000 * 0.01 = $50 which is 0.5% - this actually passes!
        
        # To make it fail risk, we need larger position
        signal['position_size'] = 0.01  # $500 position
        # But stop is very far, so risk is high
        # Actually with position_size = 0.01 BTC and risk distance $5000/BTC, risk is 0.01 * 5000 = $50
        # That's only 0.5%, which passes.
        
        # Let's increase position size to trigger risk limit
        signal['position_size'] = 0.006  # $300 position (3% - under limit)
        # Risk: 5000 * 0.006 = $30 (0.3% - still under!)
        
        # We need position size small enough to pass size check but risk high enough to fail
        # Risk = abs(entry - stop) * position_size = 5000 * position_size
        # Max risk = 10000 * 0.02 = $200
        # So position_size * 5000 > 200 means position_size > 0.04
        # But position_size * 50000 must be < 10000 * 0.10 = 1000 means position_size < 0.02
        
        signal['position_size'] = 0.015  # $750 position (7.5% - under 10% limit)
        # Risk: 5000 * 0.015 = $75 (0.75% - still under 2% limit!)
        
        # Let's use a wider stop
        signal['stop'] = 40000  # $10000 risk distance
        signal['position_size'] = 0.015  # $750 position
        # Risk: 10000 * 0.015 = $150 (1.5% - still under!)
        
        # Even wider
        signal['stop'] = 35000  # $15000 risk distance  
        signal['position_size'] = 0.015  # $750 position
        # Risk: 15000 * 0.015 = $225 (2.25% - exceeds 2% limit!)
        
        is_valid, reason = backtest._validate_signal_with_risk_rules(signal)
        assert is_valid is False
        assert "risk amount" in reason.lower()
    
    def test_validate_signal_drawdown_exceeded(self):
        """Test signal validation failure - drawdown limit."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        backtest.capital = 8000  # 20% drawdown
        
        signal = {
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.01,  # Smaller position to pass position size check relative to current capital
            'side': 'long'
        }
        
        # Position value = 0.01 * 50000 = $500
        # Max position = 8000 * 0.10 = $800, so this passes position size
        # Now it should fail on drawdown
        
        is_valid, reason = backtest._validate_signal_with_risk_rules(signal)
        assert is_valid is False
        assert "drawdown" in reason.lower()
    
    def test_validate_signal_rr_too_low(self):
        """Test signal validation failure - risk/reward ratio."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        
        signal = {
            'entry': 50000,
            'stop': 49000,
            'target': 50500,  # R:R = 500/1000 = 0.5 (below 1.5 minimum)
            'position_size': 0.02,
            'side': 'long'
        }
        
        is_valid, reason = backtest._validate_signal_with_risk_rules(signal)
        assert is_valid is False
        assert "risk/reward" in reason.lower()
    
    def test_run_backtest_basic(self):
        """Test basic backtest execution."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        
        signals = [
            {
                'symbol': 'BTC/USDT',
                'entry': 50000,
                'stop': 49000,
                'target': 52000,
                'position_size': 0.02,
                'side': 'long',
                'sl_pct': 0.02
            }
        ] * 5  # 5 similar signals
        
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': 50000,
            'high': 51000,
            'low': 49000,
            'close': 50500,
            'volume': 1000
        })
        
        results = backtest.run_backtest(signals, price_data)
        
        assert 'total_trades' in results
        assert 'win_rate' in results
        assert 'total_pnl' in results
        assert 'risk_analysis' in results
        assert results['initial_capital'] == 10000
    
    def test_run_backtest_risk_rejection(self):
        """Test backtest with risk-rejected signals."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        
        # Create signals that exceed position size limit
        signals = [
            {
                'symbol': 'BTC/USDT',
                'entry': 50000,
                'stop': 49000,
                'target': 52000,
                'position_size': 0.03,  # 15% - exceeds 10% limit
                'side': 'long',
                'sl_pct': 0.02
            }
        ] * 3
        
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': 50000,
            'high': 51000,
            'low': 49000,
            'close': 50500,
            'volume': 1000
        })
        
        results = backtest.run_backtest(signals, price_data)
        
        risk_analysis = results['risk_analysis']
        assert risk_analysis['signals_rejected'] == 3
        assert risk_analysis['signals_approved'] == 0
        assert risk_analysis['approval_rate'] == 0.0
    
    def test_calculate_results_no_trades(self):
        """Test results calculation with no trades."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        results = backtest._calculate_results()
        
        assert results['total_trades'] == 0
        assert results['win_rate'] == 0
        assert results['total_pnl'] == 0
    
    def test_calculate_results_with_trades(self):
        """Test results calculation with trades."""
        backtest = RiskAwareBacktest(initial_capital=10000)
        
        # Add some sample trades
        backtest.closed_trades = [
            {'pnl': 100, 'result': 'win', 'risk_amount': 50, 'reward_amount': 100},
            {'pnl': -50, 'result': 'loss', 'risk_amount': 50, 'reward_amount': 0},
            {'pnl': 150, 'result': 'win', 'risk_amount': 75, 'reward_amount': 150},
        ]
        
        results = backtest._calculate_results()
        
        assert results['total_trades'] == 3
        assert results['winning_trades'] == 2
        assert results['losing_trades'] == 1
        assert results['win_rate'] == pytest.approx(2/3)
        assert results['total_pnl'] == 200
        assert results['avg_win'] == pytest.approx(125)
        assert results['avg_loss'] == pytest.approx(-50)


class TestRiskConfigurationComparison:
    """Test risk configuration comparison functionality."""
    
    def test_compare_configurations(self):
        """Test comparing multiple risk configurations."""
        signals = [
            {
                'symbol': 'BTC/USDT',
                'entry': 50000,
                'stop': 49000,
                'target': 52000,
                'position_size': 0.02,
                'side': 'long',
                'sl_pct': 0.02
            }
        ] * 3
        
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': 50000,
            'high': 51000,
            'low': 49000,
            'close': 50500,
            'volume': 1000
        })
        
        risk_configs = [
            {'max_position_size': 0.10, 'max_portfolio_risk': 0.02, 'max_drawdown': 0.15, 'max_portfolio_heat': 0.10, 'min_risk_reward': 1.5},
            {'max_position_size': 0.05, 'max_portfolio_risk': 0.01, 'max_drawdown': 0.10, 'max_portfolio_heat': 0.08, 'min_risk_reward': 2.0},
        ]
        
        comparison = compare_risk_configurations(signals, price_data, risk_configs)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'config_id' in comparison.columns
        assert 'total_trades' in comparison.columns
        assert 'approval_rate' in comparison.columns


class TestReportGeneration:
    """Test backtest report generation."""
    
    def test_generate_report(self):
        """Test report generation."""
        results = {
            'initial_capital': 10000,
            'final_capital': 11000,
            'total_trades': 10,
            'winning_trades': 6,
            'losing_trades': 4,
            'win_rate': 0.6,
            'total_pnl': 1000,
            'total_return_pct': 10.0,
            'avg_win': 200,
            'avg_loss': -100,
            'profit_factor': 2.0,
            'max_drawdown': 0.05,
            'avg_risk_amount': 50,
            'avg_reward_amount': 100,
            'sharpe_ratio': 1.5,
            'risk_analysis': {
                'signals_processed': 15,
                'signals_approved': 10,
                'signals_rejected': 5,
                'approval_rate': 0.667,
                'rejection_reasons': {
                    'Position size exceeds limit': 3,
                    'Drawdown limit exceeded': 2
                },
                'risk_config': {
                    'max_position_size': 0.10,
                    'max_portfolio_risk': 0.02,
                    'max_drawdown': 0.15,
                    'max_portfolio_heat': 0.10,
                    'min_risk_reward': 1.5
                }
            }
        }
        
        report = generate_backtest_report(results)
        
        assert isinstance(report, str)
        assert 'Risk-Aware Backtest Report' in report
        assert 'Risk Configuration' in report
        assert 'Performance Metrics' in report
        assert 'Trade Statistics' in report
        assert 'Win Rate: 60.0%' in report
        assert 'Total Return: 10.00%' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
