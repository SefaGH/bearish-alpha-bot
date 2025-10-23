"""
Comprehensive tests for Phase 3.4: Live Trading Engine.
Tests live trading engine, order manager, position manager, and production coordinator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
from datetime import datetime, timezone

from core.live_trading_engine import LiveTradingEngine, TradingMode, EngineState
from core.order_manager import SmartOrderManager, OrderStatus
from core.position_manager import AdvancedPositionManager, PositionStatus
from core.execution_analytics import ExecutionAnalytics
from core.production_coordinator import ProductionCoordinator
from core.portfolio_manager import PortfolioManager
from core.risk_manager import RiskManager
from core.performance_monitor import RealTimePerformanceMonitor
from core.websocket_manager import WebSocketManager


class MockExchangeClient:
    """Mock exchange client for testing."""
    
    def __init__(self, exchange_name):
        self.exchange_name = exchange_name
    
    def ticker(self, symbol):
        """Mock ticker data."""
        return {
            'symbol': symbol,
            'last': 50000.0,
            'bid': 49990.0,
            'ask': 50010.0,
            'close': 50000.0
        }
    
    def create_order(self, symbol, side, type_, amount, price=None):
        """Mock order creation."""
        return {
            'id': f'order_{int(datetime.now(timezone.utc).timestamp())}',
            'symbol': symbol,
            'side': side,
            'type': type_,
            'amount': amount,
            'price': price,
            'status': 'closed',
            'filled': amount,
            'average': price if price else 50000.0
        }


class TestOrderManager:
    """Test smart order manager."""
    
    def test_initialization(self):
        """Test order manager initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        order_manager = SmartOrderManager(risk_manager, exchange_clients)
        
        assert order_manager.risk_manager is not None
        assert len(order_manager.exchange_clients) == 1
        assert len(order_manager.execution_algorithms) == 4
    
    @pytest.mark.asyncio
    async def test_place_market_order(self):
        """Test market order placement."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        order_manager = SmartOrderManager(risk_manager, exchange_clients)
        
        order_request = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'amount': 0.01,
            'exchange': 'kucoinfutures'
        }
        
        result = await order_manager.place_order(order_request, execution_algo='market')
        
        assert result['success'] is True
        assert 'order_id' in result
        assert result['filled_amount'] == 0.01
        assert 'avg_price' in result
    
    @pytest.mark.asyncio
    async def test_place_limit_order(self):
        """Test limit order placement."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        order_manager = SmartOrderManager(risk_manager, exchange_clients)
        
        order_request = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'amount': 0.01,
            'exchange': 'kucoinfutures'
        }
        
        result = await order_manager.place_order(order_request, execution_algo='limit')
        
        assert result['success'] is True
        assert 'order_id' in result
        assert result['filled_amount'] == 0.01
    
    @pytest.mark.asyncio
    async def test_order_validation(self):
        """Test order validation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        order_manager = SmartOrderManager(risk_manager, exchange_clients)
        
        # Missing required fields
        invalid_order = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy'
            # Missing 'amount' and 'exchange'
        }
        
        result = await order_manager.place_order(invalid_order)
        
        assert result['success'] is False
        assert 'Missing required field' in result['reason']
    
    @pytest.mark.asyncio
    async def test_cancel_order(self):
        """Test order cancellation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        order_manager = SmartOrderManager(risk_manager, exchange_clients)
        
        # First place an order
        order_request = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'amount': 0.01,
            'exchange': 'kucoinfutures'
        }
        
        result = await order_manager.place_order(order_request)
        order_id = result['order_id']
        
        # Cancel the order
        cancel_result = await order_manager.cancel_order(order_id, 'kucoinfutures')
        
        assert cancel_result['success'] is True
        assert cancel_result['order_id'] == order_id
    
    def test_execution_statistics(self):
        """Test execution statistics."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        order_manager = SmartOrderManager(risk_manager, exchange_clients)
        
        stats = order_manager.get_execution_statistics()
        
        assert 'total_orders' in stats
        assert 'successful_orders' in stats
        assert 'failed_orders' in stats
        assert 'success_rate' in stats


class TestPositionManager:
    """Test position manager."""
    
    def test_initialization(self):
        """Test position manager initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        position_manager = AdvancedPositionManager(portfolio_manager, risk_manager)
        
        assert position_manager.portfolio_manager is not None
        assert position_manager.risk_manager is not None
        assert len(position_manager.positions) == 0
    
    @pytest.mark.asyncio
    async def test_open_position(self):
        """Test opening a position."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        position_manager = AdvancedPositionManager(portfolio_manager, risk_manager)
        
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': 'test_strategy',
            'exchange': 'kucoinfutures'
        }
        
        execution_result = {
            'success': True,
            'order_id': 'order_123',
            'filled_amount': 0.01,
            'avg_price': 50000.0
        }
        
        result = await position_manager.open_position(signal, execution_result)
        
        assert result['success'] is True
        assert 'position_id' in result
        assert len(position_manager.positions) == 1
    
    @pytest.mark.asyncio
    async def test_monitor_position_pnl(self):
        """Test position P&L monitoring."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        position_manager = AdvancedPositionManager(portfolio_manager, risk_manager)
        
        # Open a position
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0
        }
        
        execution_result = {
            'success': True,
            'filled_amount': 0.01,
            'avg_price': 50000.0
        }
        
        open_result = await position_manager.open_position(signal, execution_result)
        position_id = open_result['position_id']
        
        # Monitor P&L with price increase
        pnl_result = await position_manager.monitor_position_pnl(position_id, current_price=51000.0)
        
        assert pnl_result['success'] is True
        assert pnl_result['unrealized_pnl'] > 0  # Profit
        assert pnl_result['pnl_pct'] > 0
    
    @pytest.mark.asyncio
    async def test_close_position(self):
        """Test closing a position."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        position_manager = AdvancedPositionManager(portfolio_manager, risk_manager)
        
        # Open a position
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0
        }
        
        execution_result = {
            'success': True,
            'filled_amount': 0.01,
            'avg_price': 50000.0
        }
        
        open_result = await position_manager.open_position(signal, execution_result)
        position_id = open_result['position_id']
        
        # Close position with profit
        close_result = await position_manager.close_position(position_id, exit_price=51000.0)
        
        assert close_result['success'] is True
        assert close_result['realized_pnl'] > 0
        assert len(position_manager.positions) == 0
        assert len(position_manager.closed_positions) == 1
    
    def test_calculate_position_metrics(self):
        """Test position metrics calculation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        position_manager = AdvancedPositionManager(portfolio_manager, risk_manager)
        
        # Manually create a closed position for testing
        position = {
            'position_id': 'test_pos',
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'amount': 0.01,
            'status': PositionStatus.CLOSED.value,
            'realized_pnl': 10.0,
            'return_pct': 2.0,
            'max_adverse_excursion': -5.0,
            'max_favorable_excursion': 15.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0,
            'exit_reason': 'take_profit',
            'opened_at': datetime.now(timezone.utc),
            'closed_at': datetime.now(timezone.utc)
        }
        
        position_manager.closed_positions.append(position)
        
        metrics_result = position_manager.calculate_position_metrics('test_pos')
        
        assert metrics_result['success'] is True
        assert 'metrics' in metrics_result
        assert metrics_result['metrics']['realized_pnl'] == 10.0


class TestExecutionAnalytics:
    """Test execution analytics."""
    
    def test_initialization(self):
        """Test execution analytics initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        exchange_clients = {'kucoinfutures': MockExchangeClient('kucoinfutures')}
        
        order_manager = SmartOrderManager(risk_manager, exchange_clients)
        position_manager = AdvancedPositionManager(portfolio_manager, risk_manager)
        
        analytics = ExecutionAnalytics(order_manager, position_manager)
        
        assert analytics.order_manager is not None
        assert analytics.position_manager is not None
    
    def test_get_best_execution_algorithm(self):
        """Test execution algorithm recommendation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        exchange_clients = {'kucoinfutures': MockExchangeClient('kucoinfutures')}
        
        order_manager = SmartOrderManager(risk_manager, exchange_clients)
        position_manager = AdvancedPositionManager(portfolio_manager, risk_manager)
        analytics = ExecutionAnalytics(order_manager, position_manager)
        
        # Small order, high urgency -> market
        algo = analytics.get_best_execution_algorithm(order_size=5000, urgency='high')
        assert algo == 'market'
        
        # Large order, normal urgency -> twap
        algo = analytics.get_best_execution_algorithm(order_size=60000, urgency='normal')
        assert algo == 'twap'
        
        # Medium order, low urgency -> limit
        algo = analytics.get_best_execution_algorithm(order_size=15000, urgency='low')
        assert algo == 'limit'


class TestLiveTradingEngine:
    """Test live trading engine."""
    
    def test_initialization(self):
        """Test trading engine initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        exchange_clients = {'kucoinfutures': MockExchangeClient('kucoinfutures')}
        
        trading_engine = LiveTradingEngine(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            websocket_manager=None,
            exchange_clients=exchange_clients
        )
        
        assert trading_engine.state == EngineState.STOPPED
        assert trading_engine.mode == TradingMode.PAPER
        assert trading_engine.order_manager is not None
        assert trading_engine.position_manager is not None
    
    @pytest.mark.asyncio
    async def test_start_stop_engine(self):
        """Test starting and stopping the trading engine."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        exchange_clients = {'kucoinfutures': MockExchangeClient('kucoinfutures')}
        
        trading_engine = LiveTradingEngine(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            websocket_manager=None,
            exchange_clients=exchange_clients
        )
        
        # Start engine
        start_result = await trading_engine.start_live_trading(mode='paper')
        
        assert start_result['success'] is True
        assert trading_engine.state == EngineState.RUNNING
        
        # Stop engine
        stop_result = await trading_engine.stop_live_trading()

        assert stop_result['success'] is True
        assert trading_engine.state == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_start_live_trading_reports_running_state(self):
        """Ensure start_live_trading returns a running state immediately."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        exchange_clients = {'kucoinfutures': MockExchangeClient('kucoinfutures')}

        trading_engine = LiveTradingEngine(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            websocket_manager=None,
            exchange_clients=exchange_clients
        )

        start_result = await trading_engine.start_live_trading(mode='paper')

        assert start_result['success'] is True
        assert start_result['state'] == EngineState.RUNNING.value
        assert trading_engine.state == EngineState.RUNNING

        await trading_engine.stop_live_trading()

    @pytest.mark.asyncio
    async def test_execute_signal(self):
        """Test signal execution through trading engine."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        exchange_clients = {'kucoinfutures': MockExchangeClient('kucoinfutures')}
        
        trading_engine = LiveTradingEngine(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            websocket_manager=None,
            exchange_clients=exchange_clients
        )
        
        # Start engine
        await trading_engine.start_live_trading(mode='paper')
        
        # Create a signal
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': 'test_strategy',
            'exchange': 'kucoinfutures',
            'position_size': 0.01
        }
        
        # Execute signal
        result = await trading_engine.execute_signal(signal)
        
        assert result['success'] is True
        assert 'position_id' in result
        assert 'order_id' in result
        
        # Stop engine
        await trading_engine.stop_live_trading()


class TestProductionCoordinator:
    """Test production coordinator."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test production coordinator initialization."""
        coordinator = ProductionCoordinator()
        
        assert coordinator.is_running is False
        assert coordinator.is_initialized is False
        assert coordinator.trading_engine is None
    
    @pytest.mark.asyncio
    async def test_initialize_production_system(self):
        """Test production system initialization."""
        coordinator = ProductionCoordinator()
        
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        portfolio_config = {
            'equity_usd': 10000
        }
        
        result = await coordinator.initialize_production_system(
            exchange_clients=exchange_clients,
            portfolio_config=portfolio_config
        )
        
        assert result['success'] is True
        assert coordinator.is_initialized is True
        assert coordinator.trading_engine is not None
        assert coordinator.risk_manager is not None
        assert coordinator.portfolio_manager is not None


class TestWebSocketPerformanceLogging:
    """Test WebSocket performance logging functionality."""
    
    def test_websocket_stats_calculation(self):
        """Test WebSocket statistics calculation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            websocket_manager=None,
            exchange_clients=exchange_clients
        )
        
        # Simulate some WebSocket and REST fetches
        engine._record_ws_fetch(15.0, success=True)
        engine._record_ws_fetch(20.0, success=True)
        engine._record_ws_fetch(18.0, success=True)
        
        engine._record_rest_fetch(250.0, success=True)
        engine._record_rest_fetch(230.0, success=True)
        
        # Get stats
        stats = engine.get_websocket_stats()
        
        # Verify calculations
        assert stats['websocket_fetches'] == 3
        assert stats['rest_fetches'] == 2
        assert stats['websocket_usage_ratio'] == pytest.approx(60.0, abs=0.1)  # 3/(3+2) * 100 = 60%
        assert stats['avg_latency_ws'] == pytest.approx(17.67, abs=0.1)  # (15+20+18)/3
        assert stats['avg_latency_rest'] == pytest.approx(240.0, abs=0.1)  # (250+230)/2
        
        # Verify improvement calculation
        expected_improvement = ((240.0 - 17.67) / 240.0) * 100  # ~92.6%
        assert stats['latency_improvement_pct'] == pytest.approx(expected_improvement, abs=0.1)
    
    def test_log_websocket_performance(self, caplog):
        """Test WebSocket performance logging output."""
        import logging
        
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        exchange_clients = {
            'kucoinfutures': MockExchangeClient('kucoinfutures')
        }
        
        engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            websocket_manager=None,
            exchange_clients=exchange_clients
        )
        
        # Simulate some WebSocket and REST fetches
        engine._record_ws_fetch(18.3, success=True)
        engine._record_rest_fetch(234.7, success=True)
        
        # Log performance
        with caplog.at_level(logging.INFO):
            engine._log_websocket_performance()
        
        # Check log output contains expected elements
        log_text = caplog.text
        assert '[WS-PERFORMANCE]' in log_text
        assert 'Usage Ratio:' in log_text
        assert 'WS Latency:' in log_text
        assert 'REST Latency:' in log_text
        assert 'Improvement:' in log_text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
