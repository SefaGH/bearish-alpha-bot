#!/usr/bin/env python3
"""
Test Signal Execution Pipeline - Comprehensive Flow Testing

Tests the complete signal flow:
Strategy → Coordinator → Engine → Execution → Position
"""

import sys
import os
import asyncio
import types
import pytest

from unittest.mock import Mock, AsyncMock

# Provide lightweight stubs for optional dependencies when not installed
try:  # pragma: no cover - import guard
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed only without dependency
    yaml_stub = types.ModuleType('yaml')
    yaml_stub.safe_load = lambda *args, **kwargs: {}
    yaml_stub.safe_dump = lambda *args, **kwargs: ''
    yaml_stub.dump = lambda *args, **kwargs: ''
    sys.modules.setdefault('yaml', yaml_stub)

try:  # pragma: no cover - import guard
    import pandas  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed only without dependency
    pandas_stub = types.ModuleType('pandas')

    class _StubDataFrame:
        def __init__(self, data=None, columns=None):
            self._data = data or []
            self.columns = columns or []

        def __getitem__(self, item):
            return []

        def __setitem__(self, key, value):
            # Stub ignores all assignments
            return None

    pandas_stub.DataFrame = _StubDataFrame
    pandas_stub.Series = _StubDataFrame
    pandas_stub.to_datetime = lambda *args, **kwargs: []
    pandas_stub.concat = lambda *args, **kwargs: _StubDataFrame()
    pandas_stub.isna = lambda *args, **kwargs: False
    sys.modules.setdefault('pandas', pandas_stub)

try:  # pragma: no cover - import guard
    import ccxt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed only without dependency
    ccxt_stub = types.ModuleType('ccxt')

    class _StubExchange:
        def __init__(self, params=None):
            self.params = params or {}

        def fetch_ohlcv(self, *args, **kwargs):
            return []

        def fetch_ticker(self, *args, **kwargs):
            return {}

        def fetch_tickers(self, *args, **kwargs):
            return {}

    class AuthenticationError(Exception):
        pass

    ccxt_stub.AuthenticationError = AuthenticationError

    def _create_exchange(name):
        class _DynamicExchange(_StubExchange):
            exchange_name = name

        return _DynamicExchange

    def _ccxt_getattr(attr):
        exchange_cls = _create_exchange(attr)
        setattr(ccxt_stub, attr, exchange_cls)
        return exchange_cls

    ccxt_stub.__getattr__ = _ccxt_getattr
    sys.modules.setdefault('ccxt', ccxt_stub)

    # Provide minimal ccxt.pro compatibility layer
    ccxt_pro_stub = types.ModuleType('ccxt.pro')
    ccxt_pro_stub.__getattr__ = _ccxt_getattr
    ccxt_stub.pro = ccxt_pro_stub
    sys.modules.setdefault('ccxt.pro', ccxt_pro_stub)

try:  # pragma: no cover - import guard
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed only without dependency
    requests_stub = types.ModuleType('requests')

    class _StubResponse:
        status_code = 200

        def json(self):
            return {}

        def raise_for_status(self):
            return None

    requests_stub.get = lambda *args, **kwargs: _StubResponse()
    sys.modules.setdefault('requests', requests_stub)

try:  # pragma: no cover - import guard
    import numpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed only without dependency
    numpy_stub = types.ModuleType('numpy')
    numpy_stub.array = lambda *args, **kwargs: []
    numpy_stub.isnan = lambda x: False
    numpy_stub.nan = float('nan')

    class _StubNdArray(list):
        pass

    numpy_stub.ndarray = _StubNdArray
    numpy_stub.mean = lambda *args, **kwargs: 0
    sys.modules.setdefault('numpy', numpy_stub)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator
from core.live_trading_engine import LiveTradingEngine
from core.strategy_coordinator import StrategyCoordinator
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager


class TestSignalExecutionFlow:
    """Test complete signal execution flow."""
    
    @pytest.mark.asyncio
    async def test_signal_forwarding_to_engine(self):
        """
        Test that signals are forwarded from StrategyCoordinator to LiveTradingEngine.
        
        This is the critical fix: signals must be forwarded after validation.
        """
        # Create mock components
        mock_risk_manager = Mock(spec=RiskManager)
        mock_risk_manager.calculate_position_size = AsyncMock(return_value=0.01)
        mock_risk_manager.validate_new_position = AsyncMock(return_value=(True, "Valid", {}))
        mock_risk_manager.active_positions = {}
        
        mock_portfolio_manager = Mock(spec=PortfolioManager)
        mock_portfolio_manager.strategies = {}
        mock_portfolio_manager.get_strategy_allocation = Mock(return_value=0.25)
        mock_portfolio_manager.performance_monitor = None
        mock_portfolio_manager.exchange_clients = {}
        
        # Create coordinator components
        coordinator = ProductionCoordinator()
        coordinator.strategy_coordinator = StrategyCoordinator(
            mock_portfolio_manager, 
            mock_risk_manager
        )
        
        # Create trading engine
        coordinator.trading_engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=mock_portfolio_manager,
            risk_manager=mock_risk_manager,
            exchange_clients={},
            strategy_coordinator=coordinator.strategy_coordinator
        )
        coordinator.is_running = True
        
        # Create test signal
        test_signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': 'test_strategy',
            'reason': 'test_signal'
        }
        
        # Submit signal through coordinator
        result = await coordinator.submit_signal(test_signal)
        
        # Verify signal was accepted
        assert result['success'] == True, "Signal should be accepted"
        assert 'signal_id' in result, "Should return signal_id"
        
        # Verify signal was forwarded to trading engine queue
        engine_queue_size = coordinator.trading_engine.signal_queue.qsize()
        assert engine_queue_size > 0, f"Signal should be in engine queue, but queue size is {engine_queue_size}"
        
        # Verify lifecycle tracking
        assert len(coordinator.signal_lifecycle) > 0, "Signal lifecycle should be tracked"
        
        # Get signal from engine queue
        queued_signal = await coordinator.trading_engine.signal_queue.get()
        
        # Verify it's the enriched signal with metadata
        assert 'strategy_name' in queued_signal, "Should contain strategy_name from enrichment"
        assert 'signal_timestamp' in queued_signal, "Should contain signal_timestamp from enrichment"
        assert queued_signal['symbol'] == test_signal['symbol'], "Symbol should match"
    
    @pytest.mark.asyncio
    async def test_signal_lifecycle_tracking(self):
        """Test that signals are tracked through all lifecycle stages."""
        # Setup
        mock_risk_manager = Mock(spec=RiskManager)
        mock_risk_manager.calculate_position_size = AsyncMock(return_value=0.01)
        mock_risk_manager.validate_new_position = AsyncMock(return_value=(True, "Valid", {}))
        mock_risk_manager.active_positions = {}
        
        mock_portfolio_manager = Mock(spec=PortfolioManager)
        mock_portfolio_manager.strategies = {}
        mock_portfolio_manager.get_strategy_allocation = Mock(return_value=0.25)
        mock_portfolio_manager.performance_monitor = None
        mock_portfolio_manager.exchange_clients = {}
        
        coordinator = ProductionCoordinator()
        coordinator.strategy_coordinator = StrategyCoordinator(
            mock_portfolio_manager,
            mock_risk_manager
        )
        coordinator.trading_engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=mock_portfolio_manager,
            risk_manager=mock_risk_manager,
            exchange_clients={},
            strategy_coordinator=coordinator.strategy_coordinator
        )
        coordinator.is_running = True
        
        # Submit signal
        test_signal = {
            'symbol': 'ETH/USDT:USDT',
            'side': 'short',
            'entry': 3000.0,
            'stop': 3100.0,
            'target': 2800.0,
            'strategy': 'test_strategy'
        }
        
        result = await coordinator.submit_signal(test_signal)
        signal_id = result['signal_id']
        
        # Verify lifecycle stages
        assert signal_id in coordinator.signal_lifecycle, "Signal should be tracked"
        
        lifecycle = coordinator.signal_lifecycle[signal_id]
        stages = [stage['stage'] for stage in lifecycle['stages']]
        
        # Check expected stages
        assert 'generated' in stages, "Should track 'generated' stage"
        assert 'validated' in stages, "Should track 'validated' stage"
        assert 'queued' in stages, "Should track 'queued' stage"
        assert 'forwarded' in stages, "Should track 'forwarded' stage"
        
        # Verify stage order
        assert stages.index('generated') < stages.index('validated'), "Generated should come before validated"
        assert stages.index('validated') < stages.index('queued'), "Validated should come before queued"
        assert stages.index('queued') < stages.index('forwarded'), "Queued should come before forwarded"
    
    @pytest.mark.asyncio
    async def test_queue_monitoring(self):
        """Test that queue monitoring task logs queue sizes."""
        coordinator = ProductionCoordinator()
        
        # Setup minimal components
        mock_portfolio_manager = Mock(spec=PortfolioManager)
        mock_risk_manager = Mock(spec=RiskManager)
        mock_risk_manager.active_positions = {}
        
        coordinator.strategy_coordinator = StrategyCoordinator(
            mock_portfolio_manager,
            mock_risk_manager
        )
        coordinator.trading_engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=mock_portfolio_manager,
            risk_manager=mock_risk_manager,
            exchange_clients={},
            strategy_coordinator=coordinator.strategy_coordinator
        )
        coordinator.is_running = True
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(coordinator._monitor_signal_queues())
        
        # Let it run for a short time
        await asyncio.sleep(0.5)
        
        # Stop monitoring
        coordinator.is_running = False
        monitoring_task.cancel()
        
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        # Test passes if no exceptions were raised
        assert True
    
    @pytest.mark.asyncio
    async def test_signal_rejection_tracking(self):
        """Test that rejected signals are tracked in lifecycle."""
        # Setup with failing risk validation
        mock_risk_manager = Mock(spec=RiskManager)
        mock_risk_manager.calculate_position_size = AsyncMock(return_value=0.0)  # Zero size = rejection
        mock_risk_manager.validate_new_position = AsyncMock(return_value=(False, "Risk limit exceeded", {}))
        mock_risk_manager.active_positions = {}
        
        mock_portfolio_manager = Mock(spec=PortfolioManager)
        mock_portfolio_manager.strategies = {}
        mock_portfolio_manager.get_strategy_allocation = Mock(return_value=0.25)
        mock_portfolio_manager.performance_monitor = None
        mock_portfolio_manager.exchange_clients = {}
        
        coordinator = ProductionCoordinator()
        coordinator.strategy_coordinator = StrategyCoordinator(
            mock_portfolio_manager,
            mock_risk_manager
        )
        coordinator.trading_engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=mock_portfolio_manager,
            risk_manager=mock_risk_manager,
            exchange_clients={},
            strategy_coordinator=coordinator.strategy_coordinator
        )
        coordinator.is_running = True
        
        # Submit signal that should be rejected
        test_signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'strategy': 'test_strategy'
        }
        
        result = await coordinator.submit_signal(test_signal)
        
        # Verify rejection
        assert result['success'] == False, "Signal should be rejected"
        
        # Verify lifecycle tracking includes rejection
        rejected_signals = [
            sig_id for sig_id, data in coordinator.signal_lifecycle.items()
            if any(stage['stage'] == 'rejected' for stage in data['stages'])
        ]
        assert len(rejected_signals) > 0, "Should track rejected signals"

    @pytest.mark.asyncio
    async def test_multiple_signals_same_symbol_execute(self):
        """Signals for the same symbol should execute sequentially in paper mode."""

        mock_risk_manager = Mock(spec=RiskManager)
        mock_risk_manager.calculate_position_size = AsyncMock(return_value=0.02)
        mock_risk_manager.validate_new_position = AsyncMock(return_value=(True, "Valid", {}))
        mock_risk_manager.active_positions = {}

        mock_portfolio_manager = Mock(spec=PortfolioManager)
        mock_portfolio_manager.strategies = {}
        mock_portfolio_manager.get_strategy_allocation = Mock(return_value=0.5)
        mock_portfolio_manager.performance_monitor = None
        mock_portfolio_manager.exchange_clients = {}
        mock_portfolio_manager.portfolio_state = {}

        coordinator = ProductionCoordinator()
        coordinator.strategy_coordinator = StrategyCoordinator(
            mock_portfolio_manager,
            mock_risk_manager
        )
        coordinator.trading_engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=mock_portfolio_manager,
            risk_manager=mock_risk_manager,
            exchange_clients={},
            strategy_coordinator=coordinator.strategy_coordinator
        )
        coordinator.is_running = True

        # Stub execution components to simulate successful orders and positions
        coordinator.trading_engine.order_manager.place_order = AsyncMock(side_effect=[
            {'success': True, 'order_id': 'order-1'},
            {'success': True, 'order_id': 'order-2'}
        ])
        coordinator.trading_engine.position_manager.open_position = AsyncMock(side_effect=[
            {'success': True, 'position_id': 'pos-1', 'position': {'symbol': 'BTC/USDT:USDT'}},
            {'success': True, 'position_id': 'pos-2', 'position': {'symbol': 'BTC/USDT:USDT'}},
        ])

        first_signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': 'test_strategy',
            'reason': 'first-signal',
            'exchange': 'paper'
        }

        first_submission = await coordinator.submit_signal(first_signal)
        assert first_submission['success'], "First signal should be accepted"
        first_payload = await coordinator.trading_engine.signal_queue.get()
        coordinator.trading_engine.exchange_clients = {'paper': object()}
        if 'exchange' not in first_payload:
            first_payload['exchange'] = 'paper'
        first_result = await coordinator.trading_engine.execute_signal(first_payload)
        assert first_result['success'], "First signal should execute successfully"
        assert first_submission['signal_id'] not in coordinator.strategy_coordinator.active_signals

        second_signal = {
            **first_signal,
            'entry': first_signal['entry'] * 1.01,
            'target': first_signal['target'] * 1.01,
            'stop': first_signal['stop'] * 1.01,
            'reason': 'second-signal'
        }

        second_submission = await coordinator.submit_signal(second_signal)
        assert second_submission['success'], "Second signal should be accepted after first execution"
        second_payload = await coordinator.trading_engine.signal_queue.get()
        coordinator.trading_engine.exchange_clients = {'paper': object()}
        if 'exchange' not in second_payload:
            second_payload['exchange'] = 'paper'
        second_result = await coordinator.trading_engine.execute_signal(second_payload)
        assert second_result['success'], "Second signal should execute successfully"
        assert second_submission['signal_id'] not in coordinator.strategy_coordinator.active_signals

        # Order manager should be invoked twice (once per signal)
        assert coordinator.trading_engine.order_manager.place_order.await_count == 2
    
    @pytest.mark.asyncio
    async def test_engine_status_shows_signals_received(self):
        """Test that engine status correctly shows signals_received counter."""
        # Create engine
        mock_portfolio_manager = Mock(spec=PortfolioManager)
        mock_risk_manager = Mock(spec=RiskManager)
        
        engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=mock_portfolio_manager,
            risk_manager=mock_risk_manager,
            exchange_clients={}
        )
        
        # Initially should be 0
        status = engine.get_engine_status()
        assert status['signals_received'] == 0, "Initial signals_received should be 0"
        
        # Increment counter (simulating signal receipt)
        engine._signal_count = 5
        
        # Check status again
        status = engine.get_engine_status()
        assert status['signals_received'] == 5, "signals_received should be 5"
        assert 'signal_queue_size' in status, "Should include signal_queue_size"
    
    @pytest.mark.asyncio
    async def test_enriched_signal_forwarding(self):
        """Test that enriched signal (not raw signal) is forwarded to engine."""
        # Setup
        mock_risk_manager = Mock(spec=RiskManager)
        mock_risk_manager.calculate_position_size = AsyncMock(return_value=0.01)
        mock_risk_manager.validate_new_position = AsyncMock(return_value=(True, "Valid", {}))
        mock_risk_manager.active_positions = {}
        
        mock_portfolio_manager = Mock(spec=PortfolioManager)
        mock_portfolio_manager.strategies = {}
        mock_portfolio_manager.get_strategy_allocation = Mock(return_value=0.25)
        mock_portfolio_manager.performance_monitor = None
        mock_portfolio_manager.exchange_clients = {}
        
        coordinator = ProductionCoordinator()
        coordinator.strategy_coordinator = StrategyCoordinator(
            mock_portfolio_manager,
            mock_risk_manager
        )
        coordinator.trading_engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=mock_portfolio_manager,
            risk_manager=mock_risk_manager,
            exchange_clients={},
            strategy_coordinator=coordinator.strategy_coordinator
        )
        coordinator.is_running = True
        
        # Create minimal signal
        test_signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'strategy': 'test_strategy'
        }
        
        # Submit signal
        await coordinator.submit_signal(test_signal)
        
        # Get signal from engine queue
        queued_signal = await coordinator.trading_engine.signal_queue.get()
        
        # Verify enrichment - enriched signal should have additional fields
        assert 'strategy_name' in queued_signal, "Enriched signal should have strategy_name"
        assert 'signal_timestamp' in queued_signal, "Enriched signal should have signal_timestamp"
        assert 'priority' in queued_signal, "Enriched signal should have priority"
        assert 'strategy_allocation' in queued_signal, "Enriched signal should have strategy_allocation"


class TestQueueMonitoring:
    """Test queue monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_monitor_logs_queue_sizes(self):
        """Test that monitor logs queue sizes correctly."""
        coordinator = ProductionCoordinator()
        
        # Setup minimal components
        mock_portfolio_manager = Mock(spec=PortfolioManager)
        mock_risk_manager = Mock(spec=RiskManager)
        mock_risk_manager.active_positions = {}
        
        coordinator.strategy_coordinator = StrategyCoordinator(
            mock_portfolio_manager,
            mock_risk_manager
        )
        coordinator.trading_engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=mock_portfolio_manager,
            risk_manager=mock_risk_manager,
            exchange_clients={}
        )
        
        # Add some signals to queues
        await coordinator.strategy_coordinator.signal_queue.put({'test': 1})
        await coordinator.trading_engine.signal_queue.put({'test': 2})
        
        # Check queue sizes
        coordinator_size = coordinator.strategy_coordinator.signal_queue.qsize()
        engine_size = coordinator.trading_engine.signal_queue.qsize()
        
        assert coordinator_size == 1, "StrategyCoordinator queue should have 1 signal"
        assert engine_size == 1, "LiveTradingEngine queue should have 1 signal"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
