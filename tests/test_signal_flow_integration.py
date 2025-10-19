#!/usr/bin/env python3
"""
Integration test for complete signal flow.

Tests the full signal execution pipeline in a realistic scenario.
"""

import sys
import os
import asyncio
import pytest

from unittest.mock import Mock, AsyncMock
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator
from core.live_trading_engine import LiveTradingEngine
from core.strategy_coordinator import StrategyCoordinator
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager


@pytest.mark.asyncio
@pytest.mark.integration
async def test_complete_signal_execution_pipeline():
    """
    Integration test: Complete signal execution pipeline.
    
    Flow: Signal Generation → Validation → Queue → Forward → Execute
    
    This test validates that:
    1. Signals are properly validated by StrategyCoordinator
    2. Enriched signals are forwarded to LiveTradingEngine
    3. Signals are tracked through all lifecycle stages
    4. Counters are updated correctly
    5. Queue monitoring works
    """
    # Setup mock components with realistic behavior
    mock_risk_manager = Mock(spec=RiskManager)
    mock_risk_manager.portfolio_value = 10000.0
    mock_risk_manager.calculate_position_size = AsyncMock(return_value=0.01)
    mock_risk_manager.validate_new_position = AsyncMock(return_value=(True, "Valid", {
        'position_value': 500.0,
        'risk_amount': 50.0,
        'portfolio_risk_pct': 0.005
    }))
    mock_risk_manager.active_positions = {}
    
    mock_exchange_client = Mock()
    mock_exchange_client.fetch_ticker = Mock(return_value={'last': 50000.0, 'close': 50000.0})
    
    mock_portfolio_manager = Mock(spec=PortfolioManager)
    mock_portfolio_manager.strategies = {}
    mock_portfolio_manager.get_strategy_allocation = Mock(return_value=0.25)
    mock_portfolio_manager.performance_monitor = None
    mock_portfolio_manager.exchange_clients = {'test_exchange': mock_exchange_client}
    
    # Create coordinator and components
    coordinator = ProductionCoordinator()
    coordinator.strategy_coordinator = StrategyCoordinator(
        mock_portfolio_manager,
        mock_risk_manager
    )
    coordinator.trading_engine = LiveTradingEngine(
        mode='paper',
        portfolio_manager=mock_portfolio_manager,
        risk_manager=mock_risk_manager,
        exchange_clients={'test_exchange': mock_exchange_client}
    )
    coordinator.is_running = True
    
    # Create realistic test signals
    signals = [
        {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': 'adaptive_ob',
            'reason': 'Oversold bounce setup',
            'confidence': 0.75
        },
        {
            'symbol': 'ETH/USDT:USDT',
            'side': 'short',
            'entry': 3000.0,
            'stop': 3100.0,
            'target': 2800.0,
            'strategy': 'adaptive_str',
            'reason': 'Short the rip setup',
            'confidence': 0.80
        }
    ]
    
    submitted_ids = []
    
    # Submit multiple signals
    for signal in signals:
        result = await coordinator.submit_signal(signal)
        assert result['success'] == True, f"Signal submission failed: {result.get('reason')}"
        submitted_ids.append(result['signal_id'])
    
    # Verify signals are in engine queue
    engine_queue_size = coordinator.trading_engine.signal_queue.qsize()
    assert engine_queue_size == len(signals), \
        f"Expected {len(signals)} signals in engine queue, found {engine_queue_size}"
    
    # Verify lifecycle tracking
    for signal_id in submitted_ids:
        assert signal_id in coordinator.signal_lifecycle, \
            f"Signal {signal_id} should be tracked in lifecycle"
        
        lifecycle = coordinator.signal_lifecycle[signal_id]
        stages = [stage['stage'] for stage in lifecycle['stages']]
        
        # Verify all expected stages
        expected_stages = ['generated', 'validated', 'queued', 'forwarded']
        for expected_stage in expected_stages:
            assert expected_stage in stages, \
                f"Signal {signal_id} should have '{expected_stage}' stage"
    
    # Verify coordinator queue is also populated (signals go through coordinator first)
    coordinator_stats = coordinator.strategy_coordinator.get_processing_stats()
    assert coordinator_stats['stats']['accepted_signals'] == len(signals), \
        f"Expected {len(signals)} accepted signals in coordinator stats"
    
    # Process signals from engine queue (simulating engine execution)
    processed_count = 0
    while not coordinator.trading_engine.signal_queue.empty():
        signal = await coordinator.trading_engine.signal_queue.get()
        
        # Verify it's an enriched signal
        assert 'strategy_name' in signal, "Signal should be enriched with strategy_name"
        assert 'signal_timestamp' in signal, "Signal should be enriched with signal_timestamp"
        assert 'priority' in signal, "Signal should be enriched with priority"
        
        processed_count += 1
    
    assert processed_count == len(signals), \
        f"Expected to process {len(signals)} signals, processed {processed_count}"
    
    # Verify final state
    assert coordinator.is_running == True, "Coordinator should still be running"
    
    print(f"\n✅ Integration test passed:")
    print(f"   - Submitted: {len(signals)} signals")
    print(f"   - Validated: {len(submitted_ids)} signals")
    print(f"   - Forwarded: {engine_queue_size} signals")
    print(f"   - Processed: {processed_count} signals")
    print(f"   - Lifecycle tracked: {len(coordinator.signal_lifecycle)} signals")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_signal_flow_with_rejection():
    """
    Integration test: Signal flow with rejection handling.
    
    Tests that rejected signals are properly tracked and don't reach the engine.
    """
    # Setup with strict risk limits
    mock_risk_manager = Mock(spec=RiskManager)
    mock_risk_manager.portfolio_value = 100.0  # Very small portfolio
    mock_risk_manager.calculate_position_size = AsyncMock(return_value=0.0)  # Zero = rejection
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
        exchange_clients={}
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
    assert result['success'] == False, "Signal should be rejected due to risk limits"
    assert 'reason' in result, "Rejection should include reason"
    
    # Verify signal did NOT reach engine queue
    engine_queue_size = coordinator.trading_engine.signal_queue.qsize()
    assert engine_queue_size == 0, \
        f"Rejected signal should not be in engine queue, but found {engine_queue_size} signals"
    
    # Verify rejection was tracked
    rejected_signals = [
        sig_id for sig_id, data in coordinator.signal_lifecycle.items()
        if any(stage['stage'] == 'rejected' for stage in data['stages'])
    ]
    assert len(rejected_signals) > 0, "Rejection should be tracked in lifecycle"
    
    print(f"\n✅ Rejection test passed:")
    print(f"   - Signal rejected: {result['reason']}")
    print(f"   - Engine queue size: {engine_queue_size}")
    print(f"   - Tracked rejections: {len(rejected_signals)}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_queue_monitoring_integration():
    """
    Integration test: Queue monitoring with signal flow.
    
    Tests that queue monitoring correctly reports queue sizes during signal processing.
    """
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
        exchange_clients={}
    )
    coordinator.is_running = True
    
    # Start monitoring (it will run in background)
    monitoring_task = asyncio.create_task(coordinator._monitor_signal_queues())
    
    # Submit signals while monitoring is running
    for i in range(3):
        signal = {
            'symbol': f'TEST{i}/USDT:USDT',
            'side': 'long',
            'entry': 1000.0 + i,
            'strategy': 'test_strategy'
        }
        await coordinator.submit_signal(signal)
    
    # Let monitoring run briefly
    await asyncio.sleep(0.2)
    
    # Check queue sizes
    coordinator_queue_size = coordinator.strategy_coordinator.signal_queue.qsize()
    engine_queue_size = coordinator.trading_engine.signal_queue.qsize()
    
    # Stop monitoring
    coordinator.is_running = False
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    # Verify signals were queued
    assert engine_queue_size == 3, f"Expected 3 signals in engine queue, found {engine_queue_size}"
    
    print(f"\n✅ Queue monitoring test passed:")
    print(f"   - Coordinator queue: {coordinator_queue_size}")
    print(f"   - Engine queue: {engine_queue_size}")
    print(f"   - Monitoring task: completed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
