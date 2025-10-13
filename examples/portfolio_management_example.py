#!/usr/bin/env python3
"""
Example: Phase 3.3 Portfolio Management Engine
Demonstrates advanced multi-strategy portfolio optimization and coordination.
"""

import asyncio
import logging
import sys
import os
import numpy as np
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.portfolio_manager import PortfolioManager
from core.strategy_coordinator import StrategyCoordinator, SignalPriority, ConflictResolutionStrategy
from core.risk_manager import RiskManager
from core.performance_monitor import RealTimePerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockStrategy:
    """Mock trading strategy for demonstration."""
    
    def __init__(self, name: str, win_rate: float = 0.6):
        self.name = name
        self.win_rate = win_rate
    
    async def generate_signal(self, symbol: str = 'BTC/USDT:USDT'):
        """Generate a mock trading signal."""
        entry = 50000 + np.random.randn() * 1000
        stop = entry - 1000
        target = entry + 2000
        
        return {
            'symbol': symbol,
            'side': 'long' if np.random.rand() > 0.5 else 'short',
            'entry': entry,
            'stop': stop,
            'target': target,
            'confidence': np.random.uniform(0.6, 0.9)
        }


async def demonstrate_portfolio_manager():
    """Demonstrate portfolio manager functionality."""
    logger.info("\n" + "="*70)
    logger.info("1. Portfolio Manager - Strategy Registration & Allocation")
    logger.info("="*70)
    
    # Initialize components
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    logger.info(f"✓ Portfolio Manager initialized with ${portfolio_config['equity_usd']:,.2f}")
    
    # Register multiple strategies
    strategies = {
        'momentum_strategy': (MockStrategy('momentum', 0.65), 0.30),
        'mean_reversion': (MockStrategy('mean_reversion', 0.58), 0.25),
        'breakout_strategy': (MockStrategy('breakout', 0.62), 0.25),
        'trend_following': (MockStrategy('trend', 0.55), 0.20)
    }
    
    logger.info(f"\n[Registering {len(strategies)} Strategies]")
    for strategy_name, (strategy_instance, allocation) in strategies.items():
        result = portfolio_manager.register_strategy(
            strategy_name,
            strategy_instance,
            allocation
        )
        
        if result['status'] == 'success':
            logger.info(f"✓ {strategy_name}: {allocation:.1%} allocation (${result['allocated_capital']:,.2f})")
        else:
            logger.error(f"✗ {strategy_name}: Registration failed - {result.get('error')}")
    
    # Display portfolio summary
    summary = portfolio_manager.get_portfolio_summary()
    logger.info(f"\n[Portfolio State]")
    logger.info(f"  Total value: ${summary['portfolio_state']['total_value']:,.2f}")
    logger.info(f"  Allocated: ${summary['portfolio_state']['allocated_capital']:,.2f}")
    logger.info(f"  Available: ${summary['portfolio_state']['available_capital']:,.2f}")
    logger.info(f"  Active strategies: {len(summary['registered_strategies'])}")


async def demonstrate_portfolio_optimization():
    """Demonstrate portfolio optimization methods."""
    logger.info("\n" + "="*70)
    logger.info("2. Portfolio Optimization - Multiple Methods")
    logger.info("="*70)
    
    # Setup
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    
    # Register strategies
    strategies = ['strategy_A', 'strategy_B', 'strategy_C']
    for strategy_name in strategies:
        portfolio_manager.register_strategy(
            strategy_name,
            MockStrategy(strategy_name),
            0.33
        )
    
    # Generate mock performance data
    logger.info("\n[Generating Performance Data]")
    for i, strategy_name in enumerate(strategies):
        # Different performance characteristics
        win_rate = 0.5 + (i * 0.05)  # 50%, 55%, 60%
        volatility = 10 + (i * 5)  # 10, 15, 20
        
        for j in range(50):
            pnl = np.random.randn() * volatility + (5 if np.random.rand() < win_rate else -5)
            performance_monitor.track_strategy_performance(strategy_name, {'pnl': pnl})
        
        metrics = performance_monitor.get_strategy_summary(strategy_name)['metrics']
        logger.info(f"  {strategy_name}: Win Rate={metrics['win_rate']:.1%}, "
                   f"Sharpe={metrics['sharpe_ratio']:.2f}")
    
    # Test different optimization methods
    optimization_methods = ['markowitz', 'risk_parity', 'performance_based']
    
    for method in optimization_methods:
        logger.info(f"\n[{method.replace('_', ' ').title()} Optimization]")
        result = await portfolio_manager.optimize_portfolio_allocation(method)
        
        if result['status'] == 'success':
            logger.info(f"✓ Optimization successful")
            for strategy, allocation in result['new_allocations'].items():
                change = result['allocation_changes'].get(strategy, 0)
                logger.info(f"  {strategy}: {allocation:.1%} (change: {change:+.1%})")
        else:
            logger.warning(f"✗ Optimization failed: {result.get('error', 'Unknown error')}")


async def demonstrate_portfolio_rebalancing():
    """Demonstrate portfolio rebalancing."""
    logger.info("\n" + "="*70)
    logger.info("3. Portfolio Rebalancing - Dynamic Allocation")
    logger.info("="*70)
    
    # Setup
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    
    # Register strategies
    strategies = ['high_performer', 'low_performer']
    allocations = [0.4, 0.4]
    
    for strategy_name, allocation in zip(strategies, allocations):
        portfolio_manager.register_strategy(
            strategy_name,
            MockStrategy(strategy_name),
            allocation
        )
    
    # Generate performance data (high performer does better)
    logger.info("\n[Simulating Strategy Performance]")
    for i in range(40):
        # High performer: 70% win rate, good returns
        pnl_high = 15 if np.random.rand() < 0.7 else -5
        performance_monitor.track_strategy_performance('high_performer', {'pnl': pnl_high})
        
        # Low performer: 40% win rate, poor returns
        pnl_low = 8 if np.random.rand() < 0.4 else -10
        performance_monitor.track_strategy_performance('low_performer', {'pnl': pnl_low})
    
    # Display performance
    for strategy_name in strategies:
        metrics = performance_monitor.get_strategy_summary(strategy_name)['metrics']
        logger.info(f"  {strategy_name}: Win Rate={metrics['win_rate']:.1%}, "
                   f"Total PnL=${metrics['total_pnl']:.2f}, "
                   f"Sharpe={metrics['sharpe_ratio']:.2f}")
    
    # Test rebalancing triggers
    rebalance_triggers = ['scheduled', 'performance', 'risk']
    
    for trigger in rebalance_triggers:
        logger.info(f"\n[Rebalancing: {trigger}]")
        result = await portfolio_manager.rebalance_portfolio(trigger, apply=True)
        
        if result['status'] == 'success':
            logger.info(f"✓ Rebalancing executed ({result['reason']})")
            for action in result.get('rebalancing_actions', []):
                logger.info(f"  {action['strategy']}: {action['old_allocation']:.1%} → "
                           f"{action['new_allocation']:.1%} "
                           f"(${action['capital_change']:+,.2f})")
        elif result['status'] == 'not_needed':
            logger.info(f"⊘ Rebalancing not needed: {result['reason']}")
        else:
            logger.warning(f"✗ Rebalancing failed: {result.get('error', 'Unknown error')}")


async def demonstrate_strategy_coordinator():
    """Demonstrate strategy coordinator functionality."""
    logger.info("\n" + "="*70)
    logger.info("4. Strategy Coordinator - Signal Processing")
    logger.info("="*70)
    
    # Setup
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    
    # Register strategies
    strategies = ['strategy_alpha', 'strategy_beta']
    for strategy_name in strategies:
        portfolio_manager.register_strategy(
            strategy_name,
            MockStrategy(strategy_name),
            0.4
        )
        
        # Add performance data
        for i in range(30):
            pnl = 10 if np.random.rand() < 0.6 else -5
            performance_monitor.track_strategy_performance(strategy_name, {'pnl': pnl})
    
    coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
    logger.info(f"✓ Strategy Coordinator initialized")
    
    # Process signals
    logger.info(f"\n[Processing Strategy Signals]")
    
    # Valid signal
    signal1 = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'long',
        'entry': 50000,
        'stop': 49000,
        'target': 52000
    }
    
    result1 = await coordinator.process_strategy_signal('strategy_alpha', signal1)
    if result1['status'] == 'accepted':
        logger.info(f"✓ Signal 1 accepted: {signal1['symbol']} {signal1['side']}")
        logger.info(f"  Signal ID: {result1['signal_id']}")
        logger.info(f"  Position size: {result1['risk_assessment'].get('position_size', 0):.4f}")
    else:
        logger.warning(f"✗ Signal 1 rejected: {result1.get('reason')}")
    
    # Conflicting signal (opposite side, same symbol)
    signal2 = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'short',
        'entry': 50000,
        'stop': 51000,
        'target': 48000
    }
    
    result2 = await coordinator.process_strategy_signal('strategy_beta', signal2)
    if result2['status'] == 'accepted':
        logger.info(f"✓ Signal 2 accepted: {signal2['symbol']} {signal2['side']}")
    elif result2['status'] == 'rejected':
        logger.info(f"⊘ Signal 2 rejected (conflict): {result2.get('reason')}")
    
    # Display coordinator stats
    stats = coordinator.get_processing_stats()
    logger.info(f"\n[Coordinator Statistics]")
    logger.info(f"  Total signals: {stats['stats']['total_signals']}")
    logger.info(f"  Accepted: {stats['stats']['accepted_signals']}")
    logger.info(f"  Rejected: {stats['stats']['rejected_signals']}")
    logger.info(f"  Conflicts: {stats['stats']['conflicted_signals']}")
    logger.info(f"  Active signals: {stats['active_signals']}")
    logger.info(f"  Queued signals: {stats['queued_signals']}")


async def demonstrate_conflict_resolution():
    """Demonstrate signal conflict resolution."""
    logger.info("\n" + "="*70)
    logger.info("5. Signal Conflict Resolution - Multiple Strategies")
    logger.info("="*70)
    
    # Setup
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
    
    # Create conflicting signals with different characteristics
    signals = [
        {
            'signal_id': 'high_priority',
            'strategy_name': 'premium_strategy',
            'symbol': 'ETH/USDT:USDT',
            'side': 'long',
            'entry': 3000,
            'stop': 2900,
            'target': 3300,  # 3:1 RR
            'priority': SignalPriority.HIGH
        },
        {
            'signal_id': 'low_priority',
            'strategy_name': 'basic_strategy',
            'symbol': 'ETH/USDT:USDT',
            'side': 'short',
            'entry': 3000,
            'stop': 3100,
            'target': 2900,  # 1:1 RR
            'priority': SignalPriority.LOW
        }
    ]
    
    # Test different resolution strategies
    resolution_strategies = [
        ConflictResolutionStrategy.HIGHEST_PRIORITY,
        ConflictResolutionStrategy.BEST_RISK_REWARD
    ]
    
    for strategy in resolution_strategies:
        logger.info(f"\n[Conflict Resolution: {strategy.value}]")
        
        result = await coordinator.resolve_signal_conflicts(
            signals[0],
            [signals[1]],
            strategy
        )
        
        logger.info(f"  Winner: {result['winner']['strategy_name']}")
        logger.info(f"  Action: {result['action']}")
        logger.info(f"  Reason: {result['reason']}")


async def demonstrate_integration():
    """Demonstrate full integration of portfolio management."""
    logger.info("\n" + "="*70)
    logger.info("6. Full Integration - Portfolio Management in Action")
    logger.info("="*70)
    
    # Setup complete system
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
    
    # Register strategies
    logger.info("\n[System Setup]")
    strategies_config = {
        'momentum': (0.30, 0.65),
        'reversal': (0.25, 0.58),
        'breakout': (0.25, 0.62)
    }
    
    for strategy_name, (allocation, win_rate) in strategies_config.items():
        strategy = MockStrategy(strategy_name, win_rate)
        portfolio_manager.register_strategy(strategy_name, strategy, allocation)
        
        # Generate performance data
        for i in range(40):
            pnl = 12 if np.random.rand() < win_rate else -6
            performance_monitor.track_strategy_performance(strategy_name, {'pnl': pnl})
        
        logger.info(f"✓ {strategy_name}: {allocation:.1%} allocation, {win_rate:.1%} win rate")
    
    # Simulate trading workflow
    logger.info("\n[Simulating Trading Workflow]")
    
    # 1. Generate signals from strategies
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    for i, (strategy_name, _) in enumerate(strategies_config.items()):
        strategy_instance = portfolio_manager.strategies[strategy_name]
        signal = await strategy_instance.generate_signal(symbols[i])
        
        logger.info(f"\n  [{strategy_name}] Generated signal:")
        logger.info(f"    Symbol: {signal['symbol']}, Side: {signal['side']}")
        
        # 2. Process signal through coordinator
        result = await coordinator.process_strategy_signal(strategy_name, signal)
        
        if result['status'] == 'accepted':
            logger.info(f"    ✓ Accepted - Position size: {result['risk_assessment'].get('position_size', 0):.4f}")
        else:
            logger.info(f"    ✗ Rejected - {result.get('reason')}")
    
    # 3. Run optimization
    logger.info("\n[Running Portfolio Optimization]")
    opt_result = await portfolio_manager.optimize_portfolio_allocation('performance_based')
    
    if opt_result['status'] == 'success':
        logger.info("✓ Portfolio optimized successfully")
        for strategy, new_alloc in opt_result['new_allocations'].items():
            old_alloc = opt_result['old_allocations'][strategy]
            logger.info(f"  {strategy}: {old_alloc:.1%} → {new_alloc:.1%}")
    
    # 4. Display final summary
    logger.info("\n[Final Portfolio Summary]")
    summary = portfolio_manager.get_portfolio_summary()
    logger.info(f"  Portfolio value: ${summary['portfolio_state']['total_value']:,.2f}")
    logger.info(f"  Active strategies: {len(summary['registered_strategies'])}")
    logger.info(f"  Optimizations run: {summary['optimization_history_count']}")
    
    stats = coordinator.get_processing_stats()
    logger.info(f"  Signals processed: {stats['stats']['total_signals']}")
    logger.info(f"  Signals accepted: {stats['stats']['accepted_signals']}")
    logger.info(f"  Conflicts resolved: {stats['stats']['conflicted_signals']}")


async def main():
    """Run all portfolio management demonstrations."""
    logger.info("="*70)
    logger.info("Phase 3.3: Portfolio Management Engine Demonstration")
    logger.info("="*70)
    logger.info("Advanced multi-strategy portfolio optimization and coordination")
    logger.info("")
    
    try:
        # Run demonstrations
        await demonstrate_portfolio_manager()
        await demonstrate_portfolio_optimization()
        await demonstrate_portfolio_rebalancing()
        await demonstrate_strategy_coordinator()
        await demonstrate_conflict_resolution()
        await demonstrate_integration()
        
        logger.info("\n" + "="*70)
        logger.info("Portfolio Management Engine Demonstration Complete")
        logger.info("="*70)
        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ Multi-strategy registration and capital allocation")
        logger.info("✓ Portfolio optimization (Markowitz, Risk Parity, Performance-based)")
        logger.info("✓ Dynamic portfolio rebalancing with multiple triggers")
        logger.info("✓ Strategy signal coordination and validation")
        logger.info("✓ Intelligent signal conflict resolution")
        logger.info("✓ Full integration with Risk Manager and Performance Monitor")
        logger.info("\nPhase 3.3 Implementation: COMPLETE")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)


if __name__ == '__main__':
    asyncio.run(main())
