#!/usr/bin/env python3
"""
Test Signal Bridge - Queue Transfer from StrategyCoordinator to LiveTradingEngine

This test validates that the bridge properly transfers signals between queues.
"""

import sys
import os
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.live_trading_engine import LiveTradingEngine, EngineState
from core.strategy_coordinator import StrategyCoordinator


@pytest.fixture
def mock_portfolio_manager():
    """Create mock portfolio manager."""
    pm = Mock()
    pm.cfg = {}
    pm.portfolio_state = {'total_value': 10000.0, 'used_capital': 0.0}
    pm.get_available_capital = Mock(return_value=10000.0)
    return pm


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    rm = AsyncMock()
    rm.cfg = {
        'portfolio': {
            'max_position_size': 0.1,
            'capital_limit': 10000.0
        }
    }
    rm.calculate_position_size = AsyncMock(return_value=100.0)
    rm.validate_new_position = AsyncMock(return_value=(True, "Valid", {}))
    return rm


@pytest.fixture
def mock_exchange_clients():
    """Create mock exchange clients."""
    return {
        'bingx': Mock()
    }


@pytest.mark.asyncio
async def test_bridge_transfers_signals_between_queues():
    """Test that bridge transfers signals from StrategyCoordinator to LiveTradingEngine queue."""
    
    # Create simple mocks
    pm = Mock()
    pm.cfg = {}
    pm.portfolio_state = {}
    
    rm = AsyncMock()
    rm.calculate_position_size = AsyncMock(return_value=100.0)
    
    # Create coordinator and engine
    coordinator = StrategyCoordinator(portfolio_manager=pm, risk_manager=rm)
    
    engine = LiveTradingEngine(
        mode='paper',
        portfolio_manager=pm,
        risk_manager=rm,
        exchange_clients={'bingx': Mock()},
        strategy_coordinator=coordinator
    )
    
    # Mock engine methods to prevent actual execution
    engine._initialize_risk_management = AsyncMock(return_value={'success': True})
    engine._initialize_portfolio_management = AsyncMock(return_value={'success': True})
    engine._prefetch_historical_data = AsyncMock()
    engine._position_monitoring_loop = AsyncMock()
    engine._order_management_loop = AsyncMock()
    engine._performance_reporting_loop = AsyncMock()
    engine._signal_processing_loop = AsyncMock()  # Prevent actual signal processing
    
    # Manually add test signals to coordinator queue (bypass validation)
    test_signal_1 = {
        'signal_id': 'test_signal_1',
        'signal': {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0,
            'strategy': 'test_strategy'
        },
        'risk_assessment': {'acceptable': True},
        'routing': {'priority': 'high'}
    }
    
    test_signal_2 = {
        'signal_id': 'test_signal_2',
        'signal': {
            'symbol': 'ETH/USDT:USDT',
            'side': 'short',
            'entry': 3000.0,
            'stop_loss': 3100.0,
            'take_profit': 2800.0,
            'strategy': 'test_strategy'
        },
        'risk_assessment': {'acceptable': True},
        'routing': {'priority': 'medium'}
    }
    
    # Add signals to coordinator queue
    await coordinator.signal_queue.put(test_signal_1)
    await coordinator.signal_queue.put(test_signal_2)
    
    # Verify signals are in coordinator queue
    coordinator_queue_size = coordinator.signal_queue.qsize()
    assert coordinator_queue_size == 2, f"Coordinator queue should have 2 signals, has {coordinator_queue_size}"
    print(f"✓ Coordinator queue has {coordinator_queue_size} signals")
    
    # Verify engine queue is empty
    engine_queue_size_before = engine.signal_queue.qsize()
    assert engine_queue_size_before == 0, f"Engine queue should be empty, has {engine_queue_size_before}"
    print(f"✓ Engine queue is empty before bridge starts")
    
    # Start the engine (this starts the bridge)
    await engine.start_live_trading(mode='paper')
    assert engine.state == EngineState.RUNNING, "Engine should be in RUNNING state"
    print("✓ Engine started and bridge is running")
    
    # Wait for bridge to transfer signals
    print("  Waiting for bridge to transfer signals...")
    await asyncio.sleep(3)
    
    # Verify signals were transferred to engine queue
    engine_queue_size_after = engine.signal_queue.qsize()
    print(f"  Engine queue size after bridge: {engine_queue_size_after}")
    assert engine_queue_size_after >= 2, f"Engine queue should have at least 2 signals, has {engine_queue_size_after}"
    print(f"✓ Bridge transferred {engine_queue_size_after} signals to engine queue")
    
    # Verify coordinator queue was consumed
    coordinator_queue_size_after = coordinator.signal_queue.qsize()
    print(f"  Coordinator queue size after bridge: {coordinator_queue_size_after}")
    assert coordinator_queue_size_after == 0, f"Coordinator queue should be empty, has {coordinator_queue_size_after}"
    print("✓ Coordinator queue was consumed by bridge")
    
    # Verify signal metadata
    if not engine.signal_queue.empty():
        transferred_signal = await asyncio.wait_for(engine.signal_queue.get(), timeout=1.0)
        assert 'signal_id' in transferred_signal, "Signal should have signal_id"
        assert 'from_coordinator' in transferred_signal, "Signal should have from_coordinator flag"
        assert transferred_signal['from_coordinator'] is True, "from_coordinator should be True"
        assert 'bridge_timestamp' in transferred_signal, "Signal should have bridge_timestamp"
        print(f"✓ Transferred signal has proper metadata: {list(transferred_signal.keys())}")
    
    # Stop the engine
    await engine.stop_live_trading()
    assert engine.state == EngineState.STOPPED, "Engine should be in STOPPED state"
    print("✓ Engine stopped successfully")
    
    print("\n✅ Bridge test PASSED: Signals successfully transferred from StrategyCoordinator to LiveTradingEngine")


@pytest.mark.asyncio
async def test_bridge_handles_empty_queue():
    """Test that bridge handles empty coordinator queue gracefully."""
    
    # Create simple mocks
    pm = Mock()
    pm.cfg = {}
    pm.portfolio_state = {}
    
    rm = AsyncMock()
    
    # Create coordinator and engine
    coordinator = StrategyCoordinator(portfolio_manager=pm, risk_manager=rm)
    
    engine = LiveTradingEngine(
        mode='paper',
        portfolio_manager=pm,
        risk_manager=rm,
        exchange_clients={'bingx': Mock()},
        strategy_coordinator=coordinator
    )
    
    # Mock engine methods
    engine._initialize_risk_management = AsyncMock(return_value={'success': True})
    engine._initialize_portfolio_management = AsyncMock(return_value={'success': True})
    engine._prefetch_historical_data = AsyncMock()
    engine._position_monitoring_loop = AsyncMock()
    engine._order_management_loop = AsyncMock()
    engine._performance_reporting_loop = AsyncMock()
    engine._signal_processing_loop = AsyncMock()
    
    # Start engine with empty coordinator queue
    await engine.start_live_trading(mode='paper')
    
    # Wait to ensure bridge doesn't crash
    await asyncio.sleep(2)
    
    # Verify engine is still running
    assert engine.state == EngineState.RUNNING, "Engine should still be running"
    
    # Verify queues are empty
    assert coordinator.signal_queue.qsize() == 0, "Coordinator queue should be empty"
    assert engine.signal_queue.qsize() == 0, "Engine queue should be empty"
    
    # Stop the engine
    await engine.stop_live_trading()
    
    print("✅ Empty queue test PASSED: Bridge handles empty queue gracefully")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
