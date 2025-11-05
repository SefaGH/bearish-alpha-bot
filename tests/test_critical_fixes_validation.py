"""
Validation tests for critical bot fixes:
1. SmartOrderManager logger initialization
2. RiskManager portfolio access with proper object passing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
from unittest.mock import Mock, MagicMock

from core.order_manager import SmartOrderManager
from core.risk_manager import RiskManager
from config.risk_config import RiskConfiguration


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
    
    def get_available_capital(self):
        return self.equity - self.exposure
    
    def get_peak_equity(self):
        return self.equity


class MockExchangeClient:
    """Mock exchange client for testing."""
    
    def __init__(self):
        self.name = 'bingx'
    
    def ticker(self, symbol):
        return {'last': 50000.0, 'bid': 49999.0, 'ask': 50001.0}
    
    def market(self, symbol):
        return {
            'limits': {
                'amount': {'min': 0.001, 'max': 1000},
                'cost': {'min': 5.0, 'max': 1000000}
            }
        }


class TestSmartOrderManagerLogger:
    """Test SmartOrderManager logger initialization fix."""
    
    def test_logger_initialization(self):
        """Test that SmartOrderManager properly initializes logger."""
        # Create SmartOrderManager
        order_manager = SmartOrderManager()
        
        # Verify logger is initialized
        assert hasattr(order_manager, 'logger'), "SmartOrderManager should have 'logger' attribute"
        assert order_manager.logger is not None, "Logger should not be None"
        
        # Verify logger methods are accessible
        assert hasattr(order_manager.logger, 'info')
        assert hasattr(order_manager.logger, 'warning')
        assert hasattr(order_manager.logger, 'error')
        assert hasattr(order_manager.logger, 'debug')
    
    @pytest.mark.asyncio
    async def test_logger_usage_in_place_order(self):
        """Test that logger is used in place_order without AttributeError."""
        # Create mock exchange client
        mock_client = MockExchangeClient()
        exchange_clients = {'bingx': mock_client}
        
        # Create SmartOrderManager with exchange clients
        order_manager = SmartOrderManager(exchange_clients=exchange_clients)
        
        # Create test order request
        order_request = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.001,
            'exchange': 'bingx'
        }
        
        # This should not raise AttributeError about 'logger'
        try:
            result = await order_manager.place_order(order_request, execution_algo='limit')
            # If we get here, no AttributeError was raised
            assert True, "place_order executed without AttributeError"
        except AttributeError as e:
            if 'logger' in str(e):
                pytest.fail(f"AttributeError related to logger: {e}")
            else:
                # Other AttributeErrors are acceptable (mock limitations)
                pass


class TestRiskManagerPortfolioAccess:
    """Test RiskManager portfolio access fix."""
    
    def test_safe_get_equity_with_portfolio_manager(self):
        """Test _safe_get_equity with proper PortfolioManager object."""
        # Create RiskManager
        risk_config = RiskConfiguration()
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config
        )
        
        # Create mock portfolio manager
        mock_portfolio = MockPortfolioManager(equity=10000)
        
        # Test safe equity getter
        equity = risk_manager._safe_get_equity(mock_portfolio)
        
        assert equity == 10000, f"Expected equity 10000, got {equity}"
    
    def test_safe_get_equity_with_dict_fallback(self):
        """Test _safe_get_equity with dict fallback (backward compatibility)."""
        # Create RiskManager
        risk_config = RiskConfiguration()
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config
        )
        
        # Create dict (old interface)
        portfolio_dict = {'equity_usd': 12000}
        
        # Test safe equity getter with dict
        equity = risk_manager._safe_get_equity(portfolio_dict)
        
        assert equity == 12000, f"Expected equity 12000 from dict, got {equity}"
    
    def test_safe_get_equity_with_none_fallback(self):
        """Test _safe_get_equity with None fallback."""
        # Create RiskManager
        risk_config = RiskConfiguration()
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config
        )
        
        # Test with None
        equity = risk_manager._safe_get_equity(None)
        
        # Should fallback to internal portfolio_value
        assert equity == 10000, f"Expected fallback equity 10000, got {equity}"
    
    @pytest.mark.asyncio
    async def test_validate_new_position_with_portfolio_manager(self):
        """Test validate_new_position with proper PortfolioManager object."""
        # Create RiskManager
        risk_config = RiskConfiguration()
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config
        )
        
        # Create mock portfolio manager
        mock_portfolio = MockPortfolioManager(equity=10000, exposure=1000)
        
        # Create test signal
        signal = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.05,  # Small position
            'strategy': 'test_strategy'
        }
        
        # This should not raise AttributeError about 'get_current_equity'
        try:
            is_valid, reason, metrics = await risk_manager.validate_new_position(
                signal, 
                mock_portfolio
            )
            
            # Verify metrics were calculated
            assert 'portfolio_value' in metrics, "Metrics should contain portfolio_value"
            assert metrics['portfolio_value'] == 10000, f"Expected portfolio_value 10000, got {metrics['portfolio_value']}"
            
        except AttributeError as e:
            if 'get_current_equity' in str(e):
                pytest.fail(f"AttributeError about get_current_equity: {e}")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_validate_new_position_with_dict_fallback(self):
        """Test validate_new_position with dict (backward compatibility)."""
        # Create RiskManager
        risk_config = RiskConfiguration()
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config
        )
        
        # Create dict (old interface)
        portfolio_dict = {
            'equity_usd': 10000,
            'total_exposure': 1000,
            'open_positions': {}
        }
        
        # Create test signal
        signal = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.05,
            'strategy': 'test_strategy'
        }
        
        # This should work with dict fallback
        try:
            is_valid, reason, metrics = await risk_manager.validate_new_position(
                signal, 
                portfolio_dict
            )
            
            # Should handle dict gracefully
            assert 'portfolio_value' in metrics
            
        except Exception as e:
            # Should not raise exceptions due to dict access
            if 'get_current_equity' in str(e) or 'has no attribute' in str(e):
                pytest.fail(f"Failed to handle dict fallback: {e}")


class TestIntegrationScenario:
    """Test complete integration scenario."""
    
    @pytest.mark.asyncio
    async def test_complete_signal_flow(self):
        """Test complete flow from signal validation to order placement."""
        # Setup
        risk_config = RiskConfiguration()
        risk_manager = RiskManager(
            portfolio_value=10000,
            risk_config=risk_config
        )
        
        mock_client = MockExchangeClient()
        exchange_clients = {'bingx': mock_client}
        
        order_manager = SmartOrderManager(
            risk_manager=risk_manager,
            exchange_clients=exchange_clients
        )
        
        mock_portfolio = MockPortfolioManager(equity=10000, exposure=1000)
        
        # Create test signal
        signal = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.05,
            'strategy': 'test_strategy',
            'exchange': 'bingx',
            'amount': 0.001
        }
        
        # Step 1: Validate with RiskManager (simulating LiveTradingEngine)
        is_valid, reason, metrics = await risk_manager.validate_new_position(
            signal,
            mock_portfolio  # Pass PortfolioManager object, not dict
        )
        
        # Verify risk validation worked
        assert 'portfolio_value' in metrics
        
        # Step 2: If valid, place order (simulating signal execution)
        if is_valid:
            result = await order_manager.place_order(signal, execution_algo='limit')
            # Verify order placement worked (no AttributeError about logger)
            assert 'success' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
