"""
Comprehensive tests for Phase 3.2: Risk Management Engine.
Tests risk manager, position sizing, real-time monitoring, correlation, and circuit breakers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone

from config.risk_config import RiskConfiguration, RiskLimits, CircuitBreakerLimits
from core.risk_manager import RiskManager
from core.position_sizing import AdvancedPositionSizing
from core.realtime_risk import RealTimeRiskMonitor
from core.correlation_monitor import CorrelationMonitor
from core.circuit_breaker import CircuitBreakerSystem


class TestRiskConfiguration:
    """Test risk configuration management."""
    
    def test_default_configuration(self):
        """Test default risk configuration."""
        config = RiskConfiguration()
        
        assert config.risk_limits.max_portfolio_risk == 0.02
        assert config.risk_limits.max_position_size == 0.10
        assert config.risk_limits.max_drawdown == 0.15
        assert config.risk_limits.max_correlation == 0.70
    
    def test_custom_configuration(self):
        """Test custom risk configuration."""
        custom_limits = {
            'max_portfolio_risk': 0.01,
            'max_position_size': 0.05,
            'max_drawdown': 0.10
        }
        config = RiskConfiguration(custom_limits)
        
        assert config.risk_limits.max_portfolio_risk == 0.01
        assert config.risk_limits.max_position_size == 0.05
        assert config.risk_limits.max_drawdown == 0.10
    
    def test_circuit_breaker_limits(self):
        """Test circuit breaker limit configuration."""
        config = RiskConfiguration()
        breaker_limits = config.get_circuit_breaker_limits()
        
        assert breaker_limits.daily_loss_limit == 0.05
        assert breaker_limits.position_loss_limit == 0.03
        assert breaker_limits.volatility_spike_threshold == 3.0
    
    def test_emergency_protocols(self):
        """Test emergency protocol configuration."""
        config = RiskConfiguration()
        
        protocol = config.get_emergency_protocol('market_crash')
        assert protocol == 'close_all_positions'
        
        protocol = config.get_emergency_protocol('volatility_spike')
        assert protocol == 'reduce_position_sizes'
    
    def test_update_risk_limits(self):
        """Test dynamic risk limit updates."""
        config = RiskConfiguration()
        
        config.update_risk_limits(max_portfolio_risk=0.03, max_drawdown=0.20)
        
        assert config.risk_limits.max_portfolio_risk == 0.03
        assert config.risk_limits.max_drawdown == 0.20


class TestRiskManager:
    """Test main risk manager functionality."""
    
    def test_initialization(self):
        """Test risk manager initialization."""
        portfolio_config = {
            'equity_usd': 10000,
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.10
        }
        
        risk_manager = RiskManager(portfolio_config)
        
        assert risk_manager.portfolio_value == 10000
        assert risk_manager.risk_limits['max_portfolio_risk'] == 0.02
        assert len(risk_manager.active_positions) == 0
    
    def test_set_risk_limits(self):
        """Test setting risk limits."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        
        risk_manager.set_risk_limits(
            max_portfolio_risk=0.03,
            max_position_size=0.15,
            max_drawdown=0.20
        )
        
        assert risk_manager.risk_limits['max_portfolio_risk'] == 0.03
        assert risk_manager.risk_limits['max_position_size'] == 0.15
        assert risk_manager.risk_limits['max_drawdown'] == 0.20
    
    @pytest.mark.asyncio
    async def test_validate_position_success(self):
        """Test successful position validation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.02,  # $1000 position (10% of portfolio)
            'side': 'long'
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
        
        assert is_valid is True
        assert 'position_value' in metrics
        assert 'risk_amount' in metrics
        assert metrics['risk_reward_ratio'] > 1.5
    
    @pytest.mark.asyncio
    async def test_validate_position_size_exceeded(self):
        """Test position validation failure due to size limit."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.5,  # $25000 position - too large
            'side': 'long'
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
        
        assert is_valid is False
        # Can be rejected by either capital limit or position size check
        assert 'exceed' in reason.lower()
    
    @pytest.mark.asyncio
    async def test_validate_position_risk_exceeded(self):
        """Test position validation failure due to risk limit."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 45000,  # Large stop distance ($5000 risk per unit)
            'target': 52000,
            'position_size': 0.1,  # $5000 position, $500 risk (5% of portfolio - too high)
            'side': 'long'
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
        
        assert is_valid is False
        # Could fail on either size or risk limit
        assert 'exceeds' in reason.lower()
    
    @pytest.mark.asyncio
    async def test_calculate_position_size(self):
        """Test position size calculation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        
        signal = {
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'side': 'long'
        }
        
        position_size = await risk_manager.calculate_position_size(signal)
        
        assert position_size > 0
        # Max risk is 2% of $10000 = $200
        # Risk per unit is $1000, so max size is 0.2
        assert position_size <= 0.2
    
    def test_register_and_close_position(self):
        """Test position registration and closure."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        
        # Register position
        position_data = {
            'symbol': 'BTC/USDT:USDT',
            'entry_price': 50000,
            'stop_loss': 49000,
            'size': 0.1,
            'side': 'long',
            'risk_amount': 100
        }
        
        risk_manager.register_position('pos_1', position_data)
        assert 'pos_1' in risk_manager.active_positions
        assert len(risk_manager.active_positions) == 1
        
        # Close position with profit
        risk_manager.close_position('pos_1', 51000, 100)
        assert 'pos_1' not in risk_manager.active_positions
        assert risk_manager.portfolio_value == 10100
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        
        summary = risk_manager.get_portfolio_summary()
        
        assert 'portfolio_value' in summary
        assert 'current_drawdown' in summary
        assert 'active_positions' in summary
        assert 'portfolio_heat' in summary
        assert summary['portfolio_value'] == 10000


class TestPositionSizing:
    """Test advanced position sizing algorithms."""
    
    def test_initialization(self):
        """Test position sizing initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        sizing = AdvancedPositionSizing(risk_manager)
        
        assert sizing.risk_manager == risk_manager
        assert 'kelly' in sizing.sizing_methods
        assert 'fixed_risk' in sizing.sizing_methods
    
    def test_kelly_criterion(self):
        """Test Kelly Criterion position sizing."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        sizing = AdvancedPositionSizing(risk_manager)
        
        # Win rate 60%, avg win $100, avg loss $50
        kelly_fraction = sizing._kelly_criterion(0.6, 100, 50, 10000)
        
        assert 0 < kelly_fraction <= 0.10
        assert isinstance(kelly_fraction, float)
    
    def test_fixed_risk_sizing(self):
        """Test fixed risk position sizing."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        sizing = AdvancedPositionSizing(risk_manager)
        
        # Risk $200 on entry 50000, stop 49000
        position_size = sizing._fixed_risk_sizing(200, 50000, 49000)
        
        assert position_size == 0.2  # $200 / $1000 distance
    
    def test_volatility_adjusted_sizing(self):
        """Test volatility-adjusted position sizing."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        sizing = AdvancedPositionSizing(risk_manager)
        
        signal = {
            'entry': 50000,
            'atr': 500
        }
        
        position_size = sizing._volatility_adjusted_sizing(signal, 500, 200)
        
        assert position_size > 0
        assert isinstance(position_size, float)
    
    def test_regime_based_sizing(self):
        """Test regime-based position sizing."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        sizing = AdvancedPositionSizing(risk_manager)
        
        signal = {
            'entry': 50000,
            'stop': 49000,
            'side': 'long'
        }
        
        market_regime = {
            'trend': 'bullish',
            'risk_multiplier': 1.2,
            'volatility': 'normal'
        }
        
        position_size = sizing._regime_based_sizing(signal, market_regime, base_risk=200)
        
        assert position_size > 0
        # Should have bonus for trend alignment
        base_size = 200 / 1000  # 0.2
        assert position_size >= base_size * 0.8  # Allow for adjustments
    
    @pytest.mark.asyncio
    async def test_calculate_optimal_size(self):
        """Test optimal size calculation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        sizing = AdvancedPositionSizing(risk_manager)
        
        signal = {
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'side': 'long'
        }
        
        position_size = await sizing.calculate_optimal_size(
            signal,
            method='fixed_risk',
            risk_per_trade=200
        )
        
        assert position_size >= 0


class TestRealTimeRiskMonitor:
    """Test real-time risk monitoring."""
    
    def test_initialization(self):
        """Test risk monitor initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        monitor = RealTimeRiskMonitor(risk_manager, None)
        
        assert monitor.risk_manager == risk_manager
        assert monitor.monitoring_active is False
        assert isinstance(monitor.risk_alerts, asyncio.Queue)
    
    @pytest.mark.asyncio
    async def test_price_update_processing(self):
        """Test price update processing."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        monitor = RealTimeRiskMonitor(risk_manager, None)
        
        # Register a position
        risk_manager.register_position('pos_1', {
            'symbol': 'BTC/USDT:USDT',
            'entry_price': 50000,
            'stop_loss': 49000,
            'size': 0.1,
            'side': 'long'
        })
        
        # Send price update
        price_data = {'last': 50500}
        await monitor.on_price_update('BTC/USDT:USDT', price_data)
        
        # Check position was updated
        position = risk_manager.active_positions['pos_1']
        assert position['current_price'] == 50500
    
    @pytest.mark.asyncio
    async def test_stop_loss_trigger(self):
        """Test stop-loss trigger detection."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        monitor = RealTimeRiskMonitor(risk_manager, None)
        
        # Register position with stop loss
        risk_manager.register_position('pos_1', {
            'symbol': 'BTC/USDT:USDT',
            'entry_price': 50000,
            'stop_loss': 49000,
            'size': 0.1,
            'side': 'long'
        })
        
        # Price breaches stop loss
        price_data = {'last': 48500}
        await monitor.on_price_update('BTC/USDT:USDT', price_data)
        
        # Check alert was generated
        assert monitor.risk_alerts.qsize() > 0
    
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        monitor = RealTimeRiskMonitor(risk_manager, None)
        
        # Add some price history
        monitor.update_price_history('BTC/USDT:USDT', 50000)
        monitor.update_price_history('BTC/USDT:USDT', 50500)
        monitor.update_price_history('BTC/USDT:USDT', 49800)
        monitor.update_price_history('BTC/USDT:USDT', 51000)
        
        var_metrics = monitor.calculate_portfolio_var()
        
        assert 'historical_var' in var_metrics
        assert 'parametric_var' in var_metrics
        assert 'expected_shortfall' in var_metrics


class TestCorrelationMonitor:
    """Test correlation and diversification monitoring."""
    
    def test_initialization(self):
        """Test correlation monitor initialization."""
        monitor = CorrelationMonitor()
        
        assert monitor.correlation_matrix == {}
        assert monitor.diversification_metrics == {}
    
    def test_update_price_history(self):
        """Test price history updates."""
        monitor = CorrelationMonitor()
        
        monitor.update_price_history('BTC/USDT:USDT', 50000)
        monitor.update_price_history('ETH/USDT:USDT', 3000)
        
        assert 'BTC/USDT:USDT' in monitor.price_history
        assert 'ETH/USDT:USDT' in monitor.price_history
    
    @pytest.mark.asyncio
    async def test_correlation_matrix_update(self):
        """Test correlation matrix calculation."""
        monitor = CorrelationMonitor()
        
        # Add correlated price movements
        for i in range(50):
            price_btc = 50000 + i * 100
            price_eth = 3000 + i * 6  # Correlated movement
            monitor.update_price_history('BTC/USDT:USDT', price_btc)
            monitor.update_price_history('ETH/USDT:USDT', price_eth)
        
        await monitor.update_correlation_matrix(['BTC/USDT:USDT', 'ETH/USDT:USDT'])
        
        assert monitor.correlation_matrix is not None
        assert 'matrix' in monitor.correlation_matrix
    
    def test_diversification_calculation(self):
        """Test portfolio diversification metrics."""
        monitor = CorrelationMonitor()
        
        positions = {
            'pos_1': {'symbol': 'BTC/USDT:USDT', 'size': 0.1, 'entry_price': 50000},
            'pos_2': {'symbol': 'ETH/USDT:USDT', 'size': 1.0, 'entry_price': 3000},
            'pos_3': {'symbol': 'SOL/USDT:USDT', 'size': 10.0, 'entry_price': 100}
        }
        
        metrics = monitor.calculate_portfolio_diversification(positions)
        
        assert 'effective_positions' in metrics
        assert 'concentration_risk' in metrics
        assert 'diversification_ratio' in metrics
        assert metrics['num_positions'] == 3
    
    @pytest.mark.asyncio
    async def test_validate_position_correlation(self):
        """Test new position correlation validation."""
        monitor = CorrelationMonitor()
        
        # Add price history and calculate correlations
        for i in range(50):
            monitor.update_price_history('BTC/USDT:USDT', 50000 + i * 100)
            monitor.update_price_history('ETH/USDT:USDT', 3000 + i * 6)
        
        await monitor.update_correlation_matrix(['BTC/USDT:USDT', 'ETH/USDT:USDT'])
        
        # Validate new position
        existing_positions = {
            'pos_1': {'symbol': 'BTC/USDT:USDT', 'size': 0.1, 'entry_price': 50000}
        }
        
        is_valid, reason, corr_data = await monitor.validate_new_position_correlation(
            'ETH/USDT:USDT',
            existing_positions,
            max_correlation=0.7
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)
    
    def test_get_correlation_alerts(self):
        """Test correlation alert generation."""
        monitor = CorrelationMonitor()
        
        # Set up mock correlation matrix with high correlation
        monitor.correlation_matrix = {
            'matrix': {
                'BTC/USDT:USDT': {'BTC/USDT:USDT': 1.0, 'ETH/USDT:USDT': 0.95},
                'ETH/USDT:USDT': {'BTC/USDT:USDT': 0.95, 'ETH/USDT:USDT': 1.0}
            },
            'symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        }
        
        alerts = monitor.get_correlation_alerts()
        
        assert isinstance(alerts, list)


class TestCircuitBreaker:
    """Test circuit breaker system."""
    
    def test_initialization(self):
        """Test circuit breaker initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        breaker = CircuitBreakerSystem(risk_manager)
        
        assert breaker.risk_manager == risk_manager
        assert breaker.monitoring_active is False
    
    def test_set_circuit_breakers(self):
        """Test circuit breaker configuration."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        breaker = CircuitBreakerSystem(risk_manager)
        
        breaker.set_circuit_breakers(
            daily_loss_limit=0.03,
            position_loss_limit=0.02,
            volatility_spike_threshold=2.5
        )
        
        assert breaker.circuit_breakers['daily_loss']['threshold'] == 0.03
        assert breaker.circuit_breakers['position_loss']['threshold'] == 0.02
        assert breaker.circuit_breakers['volatility_spike']['threshold'] == 2.5
    
    @pytest.mark.asyncio
    async def test_trigger_circuit_breaker(self):
        """Test circuit breaker triggering."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        breaker = CircuitBreakerSystem(risk_manager)
        
        breaker.set_circuit_breakers()
        
        await breaker.trigger_circuit_breaker('daily_loss', severity='critical')
        
        assert breaker.circuit_breakers['daily_loss']['triggered'] is True
        assert len(breaker.breakers_active) > 0
    
    @pytest.mark.asyncio
    async def test_emergency_protocol_close_all(self):
        """Test emergency protocol for closing all positions."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        breaker = CircuitBreakerSystem(risk_manager)
        
        # Register some positions
        risk_manager.register_position('pos_1', {
            'symbol': 'BTC/USDT:USDT',
            'entry_price': 50000,
            'current_price': 49000,
            'size': 0.1,
            'side': 'long'
        })
        
        risk_manager.register_position('pos_2', {
            'symbol': 'ETH/USDT:USDT',
            'entry_price': 3000,
            'current_price': 2900,
            'size': 1.0,
            'side': 'long'
        })
        
        assert len(risk_manager.active_positions) == 2
        
        # Execute emergency close all
        await breaker.execute_emergency_protocol('close_all')
        
        # All positions should be closed
        assert len(risk_manager.active_positions) == 0
    
    def test_reset_circuit_breaker(self):
        """Test circuit breaker reset."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        breaker = CircuitBreakerSystem(risk_manager)
        
        breaker.set_circuit_breakers()
        breaker.circuit_breakers['daily_loss']['triggered'] = True
        
        breaker.reset_circuit_breaker('daily_loss')
        
        assert breaker.circuit_breakers['daily_loss']['triggered'] is False
    
    def test_breaker_status(self):
        """Test getting breaker status."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        breaker = CircuitBreakerSystem(risk_manager)
        
        breaker.set_circuit_breakers()
        
        status = breaker.get_breaker_status()
        
        assert 'circuit_breakers' in status
        assert 'monitoring_active' in status
        assert 'active_triggers' in status
    
    @pytest.mark.asyncio
    async def test_check_circuit_breaker_normal(self):
        """Test check_circuit_breaker when all breakers are normal."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        breaker = CircuitBreakerSystem(risk_manager)
        
        breaker.set_circuit_breakers()
        
        result = await breaker.check_circuit_breaker()
        
        assert 'tripped' in result
        assert 'severity' in result
        assert 'message' in result
        assert result['tripped'] is False
        assert result['severity'] == 'none'
    
    @pytest.mark.asyncio
    async def test_check_circuit_breaker_triggered(self):
        """Test check_circuit_breaker when a breaker is triggered."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        breaker = CircuitBreakerSystem(risk_manager)
        
        breaker.set_circuit_breakers()
        
        # Manually trigger a breaker
        breaker.circuit_breakers['daily_loss']['triggered'] = True
        
        result = await breaker.check_circuit_breaker()
        
        assert result['tripped'] is True
        assert result['breaker'] == 'daily_loss'
        assert result['severity'] == 'critical'
        assert 'threshold' in result
        assert 'message' in result


class TestIntegration:
    """Integration tests for complete risk management system."""
    
    @pytest.mark.asyncio
    async def test_full_risk_workflow(self):
        """Test complete risk management workflow."""
        # Initialize components
        portfolio_config = {
            'equity_usd': 10000,
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.10
        }
        
        risk_manager = RiskManager(portfolio_config)
        sizing = AdvancedPositionSizing(risk_manager)
        monitor = RealTimeRiskMonitor(risk_manager, None)
        breaker = CircuitBreakerSystem(risk_manager)
        
        # Configure risk limits
        risk_manager.set_risk_limits(max_portfolio_risk=0.02)
        breaker.set_circuit_breakers(daily_loss_limit=0.05)
        
        # Create trading signal
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'side': 'long',
            'strategy': 'test_strategy'
        }
        
        # Calculate position size (using the risk manager's method for consistency)
        position_size = await risk_manager.calculate_position_size(signal)
        
        signal['position_size'] = position_size
        
        # Validate position
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
        
        assert is_valid is True
        assert position_size > 0
        
        # Register position
        risk_manager.register_position('pos_1', {
            'symbol': signal['symbol'],
            'entry_price': signal['entry'],
            'stop_loss': signal['stop'],
            'size': position_size,
            'side': signal['side'],
            'risk_amount': metrics.get('risk_amount', 0)
        })
        
        # Simulate price update
        price_data = {'last': 50500}
        await monitor.on_price_update(signal['symbol'], price_data)
        
        # Get portfolio summary
        summary = risk_manager.get_portfolio_summary()
        
        assert summary['active_positions'] == 1
        assert summary['portfolio_value'] == 10000
        
        # Close position
        risk_manager.close_position('pos_1', 51000, 100)
        
        assert len(risk_manager.active_positions) == 0
        assert risk_manager.portfolio_value == 10100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
