"""
Test critical fixes for Issues #104, #105, #106, #107.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock
from datetime import datetime, timezone


class TestIssue107PortfolioManagerExchangeClients:
    """Test Issue #107: PortfolioManager Missing exchange_clients Attribute"""
    
    def test_portfolio_manager_accepts_exchange_clients(self):
        """Test that PortfolioManager accepts exchange_clients parameter"""
        from src.core.portfolio_manager import PortfolioManager
        
        # Create mock dependencies
        risk_manager = Mock()
        risk_manager.portfolio_value = 1000.0
        performance_monitor = Mock()
        
        # Create exchange clients dict
        exchange_clients = {
            'bingx': Mock(),
            'binance': Mock()
        }
        
        # Initialize PortfolioManager with exchange_clients
        pm = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor,
            websocket_manager=None,
            exchange_clients=exchange_clients
        )
        
        # Verify exchange_clients is set
        assert hasattr(pm, 'exchange_clients')
        assert pm.exchange_clients == exchange_clients
        assert 'bingx' in pm.exchange_clients
        assert 'binance' in pm.exchange_clients
    
    def test_portfolio_manager_default_empty_exchange_clients(self):
        """Test that exchange_clients defaults to empty dict"""
        from src.core.portfolio_manager import PortfolioManager
        
        risk_manager = Mock()
        risk_manager.portfolio_value = 1000.0
        performance_monitor = Mock()
        
        pm = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor
        )
        
        assert hasattr(pm, 'exchange_clients')
        assert pm.exchange_clients == {}
    
    def test_portfolio_manager_has_cfg_attribute(self):
        """Test that PortfolioManager has cfg attribute"""
        from src.core.portfolio_manager import PortfolioManager
        
        risk_manager = Mock()
        risk_manager.portfolio_value = 1000.0
        performance_monitor = Mock()
        
        pm = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor
        )
        
        assert hasattr(pm, 'cfg')
        assert pm.cfg == {}
    
    def test_add_exchange_client_method(self):
        """Test add_exchange_client() method"""
        from src.core.portfolio_manager import PortfolioManager
        
        risk_manager = Mock()
        risk_manager.portfolio_value = 1000.0
        performance_monitor = Mock()
        
        pm = PortfolioManager(
            risk_manager=risk_manager,
            performance_monitor=performance_monitor
        )
        
        # Add exchange client
        mock_client = Mock()
        pm.add_exchange_client('bingx', mock_client)
        
        assert 'bingx' in pm.exchange_clients
        assert pm.exchange_clients['bingx'] == mock_client


class TestIssue104CooldownLogic:
    """Test Issue #104: Fix Cooldown Logic"""
    
    def test_cooldown_uses_combined_key(self):
        """Test that cooldown uses 'symbol:strategy' combined key"""
        from src.core.strategy_coordinator import StrategyCoordinator
        import time
        
        # Create mocks
        portfolio_manager = Mock()
        portfolio_manager.cfg = {
            'monitoring': {
                'duplicate_prevention': {
                    'enabled': True,
                    'same_symbol_cooldown': 1,  # Use 1 second for fast testing
                    'min_price_change': 0.002
                }
            }
        }
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # Test signal
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000}
        
        # First signal should pass
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True
        
        # Same symbol+strategy should be blocked (within cooldown)
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is False
        assert 'Signal cooldown' in reason or 'price movement' in reason.lower()
        
        # Different strategy on same symbol should pass
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50100}  # Different price to pass price check
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy2')
        assert is_valid is True
        
        # Wait for cooldown to expire, then same symbol+strategy should pass
        time.sleep(1.1)
        signal3 = {'symbol': 'BTC/USDT:USDT', 'entry': 50200}
        is_valid, reason = coordinator.validate_duplicate(signal3, 'strategy1')
        assert is_valid is True
    
    def test_cooldown_allows_different_symbols_same_strategy(self):
        """Test that cooldown allows different symbols with same strategy"""
        from src.core.strategy_coordinator import StrategyCoordinator
        
        portfolio_manager = Mock()
        portfolio_manager.cfg = {
            'monitoring': {
                'duplicate_prevention': {
                    'enabled': True,
                    'same_symbol_cooldown': 60,
                    'min_price_change': 0.002
                }
            }
        }
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # First signal: BTC with strategy1
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000}
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True
        
        # Second signal: ETH with strategy1 - should pass
        signal2 = {'symbol': 'ETH/USDT:USDT', 'entry': 3000}
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy1')
        assert is_valid is True
    
    def test_last_signal_time_structure(self):
        """Test that last_signal_time uses combined key structure"""
        from src.core.strategy_coordinator import StrategyCoordinator
        
        portfolio_manager = Mock()
        portfolio_manager.cfg = {
            'monitoring': {
                'duplicate_prevention': {
                    'enabled': True,
                    'same_symbol_cooldown': 60,
                    'min_price_change': 0.002
                }
            }
        }
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # Process a signal
        signal = {'symbol': 'BTC/USDT:USDT', 'entry': 50000}
        coordinator.validate_duplicate(signal, 'strategy1')
        
        # Check that last_signal_time has combined key
        expected_key = 'BTC/USDT:USDT:strategy1'
        assert expected_key in coordinator.last_signal_time


class TestIssue105PositionDashboard:
    """Test Issue #105: Position Dashboard & Test Improvements"""
    
    def test_print_position_dashboard_method_exists(self):
        """Test that _print_position_dashboard method exists"""
        from src.core.production_coordinator import ProductionCoordinator
        
        coordinator = ProductionCoordinator()
        assert hasattr(coordinator, '_print_position_dashboard')
        assert callable(coordinator._print_position_dashboard)
    
    def test_cooldown_reduced_to_60s(self):
        """Test that cooldown times are reduced to 60s"""
        from src.config.live_trading_config import LiveTradingConfiguration
        
        config = LiveTradingConfiguration.get_all_configs()
        monitoring = config.get('monitoring', {})
        duplicate_prevention = monitoring.get('duplicate_prevention', {})
        
        assert duplicate_prevention.get('same_symbol_cooldown') == 60
        assert duplicate_prevention.get('enabled') is True


class TestIssue106ConfigurationLogging:
    """Test Issue #106: Improve Configuration Logging"""
    
    def test_print_configuration_summary_exists(self):
        """Test that _print_configuration_summary method exists in source"""
        # Read the launcher file to check methods exist
        with open('scripts/live_trading_launcher.py', 'r') as f:
            content = f.read()
        
        assert 'def _print_configuration_summary(self):' in content
        assert 'CONFIGURATION SUMMARY' in content
        assert 'Capital:' in content
        assert 'Exchange:' in content
    
    def test_generate_post_session_analysis_exists(self):
        """Test that _generate_post_session_analysis method exists in source"""
        # Read the launcher file to check methods exist
        with open('scripts/live_trading_launcher.py', 'r') as f:
            content = f.read()
        
        assert 'def _generate_post_session_analysis(self' in content
        assert 'POST-SESSION ANALYSIS' in content
        assert 'Signals Generated:' in content
        assert 'Trades Executed:' in content


class TestPositionManagerSafetyCheck:
    """Test position_manager.py safety improvements"""
    
    @pytest.mark.asyncio
    async def test_get_current_price_safely_handles_missing_exchange_clients(self):
        """Test that _get_current_price_from_ws safely handles missing exchange_clients"""
        from src.core.position_manager import AdvancedPositionManager
        
        # Create mocks
        portfolio_manager = Mock()
        # Don't set exchange_clients attribute to test safety
        risk_manager = Mock()
        ws_manager = Mock()
        ws_manager.get_latest_ticker = Mock(return_value=None)
        
        pm = AdvancedPositionManager(portfolio_manager, risk_manager, ws_manager)
        
        # This should not raise AttributeError
        price = await pm._get_current_price_from_ws('BTC/USDT:USDT')
        
        # Price should be None since no exchange_clients or WS data
        assert price is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
