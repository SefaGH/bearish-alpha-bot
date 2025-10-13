"""
Comprehensive tests for Phase 3.3: Portfolio Management Engine.
Tests portfolio manager and strategy coordinator functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone

from core.portfolio_manager import PortfolioManager
from core.strategy_coordinator import StrategyCoordinator, SignalPriority, ConflictResolutionStrategy
from core.risk_manager import RiskManager
from core.performance_monitor import RealTimePerformanceMonitor


class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, name):
        self.name = name
    
    async def execute(self):
        return {'status': 'executed'}


class TestPortfolioManager:
    """Test portfolio manager functionality."""
    
    def test_initialization(self):
        """Test portfolio manager initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        assert portfolio_manager.risk_manager is not None
        assert portfolio_manager.performance_monitor is not None
        assert portfolio_manager.portfolio_state['total_value'] == 10000
        assert len(portfolio_manager.strategies) == 0
    
    def test_register_strategy(self):
        """Test strategy registration."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        strategy = MockStrategy('test_strategy')
        result = portfolio_manager.register_strategy('test_strategy', strategy, 0.25)
        
        assert result['status'] == 'success'
        assert result['allocation'] == 0.25
        assert result['allocated_capital'] == 2500.0
        assert 'test_strategy' in portfolio_manager.strategies
        assert portfolio_manager.strategy_allocations['test_strategy'] == 0.25
    
    def test_register_multiple_strategies(self):
        """Test registering multiple strategies."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register three strategies
        strategies = ['strategy_1', 'strategy_2', 'strategy_3']
        allocations = [0.3, 0.3, 0.2]
        
        for name, allocation in zip(strategies, allocations):
            strategy = MockStrategy(name)
            result = portfolio_manager.register_strategy(name, strategy, allocation)
            assert result['status'] == 'success'
        
        assert len(portfolio_manager.strategies) == 3
        total_allocation = sum(portfolio_manager.strategy_allocations.values())
        assert abs(total_allocation - 0.8) < 0.01  # 80% allocated
    
    def test_invalid_allocation(self):
        """Test invalid allocation handling."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        strategy = MockStrategy('test')
        result = portfolio_manager.register_strategy('test', strategy, 1.5)  # Invalid: > 1.0
        
        assert result['status'] == 'error'
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio_allocation_markowitz(self):
        """Test Markowitz portfolio optimization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register strategies
        for i in range(3):
            strategy = MockStrategy(f'strategy_{i}')
            portfolio_manager.register_strategy(f'strategy_{i}', strategy, 0.33)
            
            # Add some mock performance data
            for j in range(20):
                performance_monitor.track_strategy_performance(
                    f'strategy_{i}',
                    {'pnl': np.random.randn() * 10 + 5}  # Positive expected return
                )
        
        # Run optimization
        result = await portfolio_manager.optimize_portfolio_allocation('markowitz')
        
        assert result['status'] == 'success'
        assert 'new_allocations' in result
        assert len(result['new_allocations']) == 3
        
        # Check allocations sum to approximately 1.0
        total = sum(result['new_allocations'].values())
        assert abs(total - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio_allocation_risk_parity(self):
        """Test risk parity optimization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register strategies with different volatilities
        for i in range(2):
            strategy = MockStrategy(f'strategy_{i}')
            portfolio_manager.register_strategy(f'strategy_{i}', strategy, 0.5)
            
            # Strategy 0: low volatility, Strategy 1: high volatility
            volatility = 5 if i == 0 else 20
            for j in range(30):
                performance_monitor.track_strategy_performance(
                    f'strategy_{i}',
                    {'pnl': np.random.randn() * volatility}
                )
        
        # Run optimization
        result = await portfolio_manager.optimize_portfolio_allocation('risk_parity')
        
        assert result['status'] == 'success'
        # Low volatility strategy should get higher allocation
        assert result['new_allocations']['strategy_0'] > result['new_allocations']['strategy_1']
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio_allocation_performance_based(self):
        """Test performance-based optimization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register strategies
        for i in range(2):
            strategy = MockStrategy(f'strategy_{i}')
            portfolio_manager.register_strategy(f'strategy_{i}', strategy, 0.5)
            
            # Strategy 0: good performance, Strategy 1: poor performance
            win_pnl = 10 if i == 0 else 5
            loss_pnl = -5 if i == 0 else -10
            
            for j in range(30):
                pnl = win_pnl if j % 2 == 0 else loss_pnl
                performance_monitor.track_strategy_performance(
                    f'strategy_{i}',
                    {'pnl': pnl}
                )
        
        # Run optimization
        result = await portfolio_manager.optimize_portfolio_allocation('performance_based')
        
        assert result['status'] == 'success'
        # Better performing strategy should get higher allocation
        assert result['new_allocations']['strategy_0'] > result['new_allocations']['strategy_1']
    
    @pytest.mark.asyncio
    async def test_rebalance_portfolio_scheduled(self):
        """Test scheduled portfolio rebalancing."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register strategies
        for i in range(2):
            strategy = MockStrategy(f'strategy_{i}')
            portfolio_manager.register_strategy(f'strategy_{i}', strategy, 0.5)
        
        # First rebalance should trigger (no previous rebalance)
        result = await portfolio_manager.rebalance_portfolio('scheduled', apply=False)
        
        assert result['status'] in ['success', 'not_needed']
        if result['status'] == 'success':
            assert 'rebalancing_actions' in result
    
    @pytest.mark.asyncio
    async def test_rebalance_portfolio_threshold(self):
        """Test threshold-based rebalancing."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register strategy
        strategy = MockStrategy('strategy_1')
        portfolio_manager.register_strategy('strategy_1', strategy, 0.5)
        
        # Test with high threshold (should not trigger)
        result = await portfolio_manager.rebalance_portfolio('threshold', threshold=0.9, apply=False)
        
        assert result['status'] in ['success', 'not_needed']
    
    def test_get_portfolio_summary(self):
        """Test portfolio summary generation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register strategy
        strategy = MockStrategy('strategy_1')
        portfolio_manager.register_strategy('strategy_1', strategy, 0.3)
        
        summary = portfolio_manager.get_portfolio_summary()
        
        assert 'portfolio_state' in summary
        assert 'registered_strategies' in summary
        assert 'strategy_allocations' in summary
        assert 'risk_metrics' in summary
        assert 'strategy_1' in summary['registered_strategies']
        assert summary['strategy_allocations']['strategy_1'] == 0.3
    
    def test_get_strategy_allocation(self):
        """Test getting strategy allocation."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        strategy = MockStrategy('strategy_1')
        portfolio_manager.register_strategy('strategy_1', strategy, 0.4)
        
        allocation = portfolio_manager.get_strategy_allocation('strategy_1')
        assert allocation == 0.4
        
        # Non-existent strategy
        allocation = portfolio_manager.get_strategy_allocation('non_existent')
        assert allocation is None
    
    def test_update_strategy_status(self):
        """Test updating strategy status."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        strategy = MockStrategy('strategy_1')
        portfolio_manager.register_strategy('strategy_1', strategy, 0.3)
        
        # Deactivate strategy
        result = portfolio_manager.update_strategy_status('strategy_1', False)
        assert result is True
        assert portfolio_manager.strategy_metadata['strategy_1']['active'] is False
        
        # Reactivate strategy
        result = portfolio_manager.update_strategy_status('strategy_1', True)
        assert result is True
        assert portfolio_manager.strategy_metadata['strategy_1']['active'] is True


class TestStrategyCoordinator:
    """Test strategy coordinator functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test strategy coordinator initialization."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        assert coordinator.portfolio_manager is not None
        assert coordinator.risk_manager is not None
        assert coordinator.signal_queue is not None
        assert coordinator.processing_stats['total_signals'] == 0
    
    @pytest.mark.asyncio
    async def test_process_strategy_signal_valid(self):
        """Test processing valid strategy signal."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register strategy
        strategy = MockStrategy('test_strategy')
        portfolio_manager.register_strategy('test_strategy', strategy, 0.3)
        
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # Valid signal
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000,
            'stop': 49000,
            'target': 52000
        }
        
        result = await coordinator.process_strategy_signal('test_strategy', signal)
        
        assert result['status'] == 'accepted'
        assert 'signal_id' in result
        assert coordinator.processing_stats['total_signals'] == 1
        assert coordinator.processing_stats['accepted_signals'] == 1
    
    @pytest.mark.asyncio
    async def test_process_strategy_signal_invalid(self):
        """Test processing invalid strategy signal."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # Invalid signal (missing required fields)
        signal = {
            'symbol': 'BTC/USDT:USDT'
            # Missing 'side' and 'entry'
        }
        
        result = await coordinator.process_strategy_signal('test_strategy', signal)
        
        assert result['status'] == 'rejected'
        assert 'reason' in result
        assert coordinator.processing_stats['rejected_signals'] == 1
    
    @pytest.mark.asyncio
    async def test_signal_conflict_detection(self):
        """Test signal conflict detection."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        # Register strategies with performance data
        portfolio_manager.register_strategy('strategy_1', MockStrategy('s1'), 0.3)
        portfolio_manager.register_strategy('strategy_2', MockStrategy('s2'), 0.3)
        
        # Add performance data for priority calculation
        for i in range(20):
            performance_monitor.track_strategy_performance('strategy_1', {'pnl': 10})
            performance_monitor.track_strategy_performance('strategy_2', {'pnl': 5})
        
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # First signal
        signal1 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000,
            'stop': 49000,
            'target': 52000
        }
        
        result1 = await coordinator.process_strategy_signal('strategy_1', signal1)
        assert result1['status'] == 'accepted'
        
        # Conflicting signal (opposite side, same symbol)
        signal2 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'short',
            'entry': 50000,
            'stop': 51000,
            'target': 48000
        }
        
        result2 = await coordinator.process_strategy_signal('strategy_2', signal2)
        
        # Check if conflict was detected (either rejected or conflict counter increased)
        # With better strategy_1, it should win and strategy_2 should be rejected
        if result2['status'] == 'rejected':
            assert 'conflict' in result2.get('stage', '').lower() or 'conflict' in result2.get('reason', '').lower()
        
        # Either way, conflict should be recorded
        assert coordinator.processing_stats['conflicted_signals'] >= 1 or coordinator.processing_stats['rejected_signals'] >= 1
    
    @pytest.mark.asyncio
    async def test_resolve_signal_conflicts_highest_priority(self):
        """Test conflict resolution by highest priority."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # Create signals with different priorities
        high_priority_signal = {
            'signal_id': 'high',
            'strategy_name': 'high_strategy',
            'priority': SignalPriority.HIGH,
            'symbol': 'BTC/USDT:USDT',
            'side': 'long'
        }
        
        low_priority_signal = {
            'signal_id': 'low',
            'strategy_name': 'low_strategy',
            'priority': SignalPriority.LOW,
            'symbol': 'BTC/USDT:USDT',
            'side': 'short'
        }
        
        result = await coordinator.resolve_signal_conflicts(
            high_priority_signal,
            [low_priority_signal],
            ConflictResolutionStrategy.HIGHEST_PRIORITY
        )
        
        assert result['action'] == 'accept'
        assert result['winner']['signal_id'] == 'high'
    
    @pytest.mark.asyncio
    async def test_resolve_signal_conflicts_best_risk_reward(self):
        """Test conflict resolution by best risk/reward."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # Signal with better risk/reward
        good_rr_signal = {
            'signal_id': 'good_rr',
            'strategy_name': 'good_strategy',
            'entry': 50000,
            'stop': 49000,
            'target': 53000,  # 3:1 RR
            'symbol': 'BTC/USDT:USDT',
            'side': 'long'
        }
        
        # Signal with worse risk/reward
        poor_rr_signal = {
            'signal_id': 'poor_rr',
            'strategy_name': 'poor_strategy',
            'entry': 50000,
            'stop': 49000,
            'target': 51000,  # 1:1 RR
            'symbol': 'BTC/USDT:USDT',
            'side': 'short'
        }
        
        result = await coordinator.resolve_signal_conflicts(
            good_rr_signal,
            [poor_rr_signal],
            ConflictResolutionStrategy.BEST_RISK_REWARD
        )
        
        assert result['action'] == 'accept'
        assert result['winner']['signal_id'] == 'good_rr'
    
    @pytest.mark.asyncio
    async def test_get_next_signal(self):
        """Test getting next signal from queue."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        portfolio_manager.register_strategy('test_strategy', MockStrategy('test'), 0.3)
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # Process a signal
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000,
            'stop': 49000,
            'target': 52000
        }
        
        await coordinator.process_strategy_signal('test_strategy', signal)
        
        # Get signal from queue
        queued_signal = await coordinator.get_next_signal(timeout=1.0)
        
        assert queued_signal is not None
        assert 'signal_id' in queued_signal
        assert 'signal' in queued_signal
    
    @pytest.mark.asyncio
    async def test_mark_signal_executed(self):
        """Test marking signal as executed."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        
        portfolio_manager.register_strategy('test_strategy', MockStrategy('test'), 0.3)
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        # Process signal
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000,
            'stop': 49000,
            'target': 52000
        }
        
        result = await coordinator.process_strategy_signal('test_strategy', signal)
        signal_id = result['signal_id']
        
        # Mark as executed
        execution_result = {'status': 'filled', 'price': 50000}
        coordinator.mark_signal_executed(signal_id, execution_result)
        
        assert coordinator.active_signals[signal_id]['status'] == 'executed'
        assert coordinator.active_signals[signal_id]['execution_result'] == execution_result
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        stats = coordinator.get_processing_stats()
        
        assert 'stats' in stats
        assert 'active_signals' in stats
        assert 'queued_signals' in stats
        assert stats['stats']['total_signals'] == 0
    
    def test_get_active_signals_summary(self):
        """Test getting active signals summary."""
        portfolio_config = {'equity_usd': 10000}
        risk_manager = RiskManager(portfolio_config)
        performance_monitor = RealTimePerformanceMonitor()
        portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        
        summary = coordinator.get_active_signals_summary()
        
        assert isinstance(summary, list)
        assert len(summary) == 0  # No signals yet


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
