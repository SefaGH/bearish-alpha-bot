# Portfolio Management Engine Guide

**Phase 3.3: Advanced Multi-Strategy Portfolio Optimization**

Complete guide to the Portfolio Management Engine - coordinating multiple trading strategies with intelligent capital allocation and dynamic rebalancing.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Components](#components)
4. [Usage Examples](#usage-examples)
5. [Optimization Methods](#optimization-methods)
6. [Rebalancing Strategies](#rebalancing-strategies)
7. [Signal Coordination](#signal-coordination)
8. [Integration](#integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Portfolio Management Engine provides advanced multi-strategy coordination and optimization capabilities:

- **Multi-Strategy Registration**: Register and manage multiple trading strategies with individual capital allocations
- **Portfolio Optimization**: Optimize capital allocation using Modern Portfolio Theory (Markowitz), Risk Parity, and performance-based methods
- **Dynamic Rebalancing**: Automatically rebalance portfolio based on performance, risk, or schedule
- **Signal Coordination**: Intelligent signal processing and conflict resolution across strategies
- **Risk Integration**: Full integration with Phase 3.2 Risk Management Engine
- **Performance Tracking**: Continuous monitoring via Phase 2 Performance Monitor

### Key Features

- ✅ Strategy registration with flexible capital allocation
- ✅ Multiple optimization algorithms (Markowitz, Risk Parity, Performance-based)
- ✅ Automated portfolio rebalancing with multiple triggers
- ✅ Signal validation and enrichment
- ✅ Intelligent conflict resolution between competing signals
- ✅ Real-time portfolio state tracking
- ✅ Optimization history and audit trail

---

## Quick Start

### Basic Setup

```python
from core.portfolio_manager import PortfolioManager
from core.strategy_coordinator import StrategyCoordinator
from core.risk_manager import RiskManager
from core.performance_monitor import RealTimePerformanceMonitor

# Initialize components
portfolio_config = {'equity_usd': 10000}
risk_manager = RiskManager(portfolio_config)
performance_monitor = RealTimePerformanceMonitor()

# Create portfolio manager
portfolio_manager = PortfolioManager(
    risk_manager,
    performance_monitor,
    websocket_manager=None  # Optional
)

# Register strategies
portfolio_manager.register_strategy('momentum', strategy_instance, 0.30)
portfolio_manager.register_strategy('reversal', strategy_instance, 0.25)

# Create strategy coordinator
coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
```

### Process Trading Signals

```python
# Generate and process signal
signal = {
    'symbol': 'BTC/USDT:USDT',
    'side': 'long',
    'entry': 50000,
    'stop': 49000,
    'target': 52000
}

result = await coordinator.process_strategy_signal('momentum', signal)

if result['status'] == 'accepted':
    print(f"Signal accepted: {result['signal_id']}")
    print(f"Position size: {result['risk_assessment']['position_size']}")
else:
    print(f"Signal rejected: {result['reason']}")
```

---

## Components

### 1. Portfolio Manager (`src/core/portfolio_manager.py`)

Main component for portfolio-level operations.

#### Key Methods

**Strategy Registration**

```python
result = portfolio_manager.register_strategy(
    strategy_name='momentum_strategy',
    strategy_instance=strategy,
    initial_allocation=0.30  # 30% of portfolio
)
```

**Portfolio Optimization**

```python
# Markowitz optimization
result = await portfolio_manager.optimize_portfolio_allocation('markowitz')

# Risk parity
result = await portfolio_manager.optimize_portfolio_allocation('risk_parity')

# Performance-based
result = await portfolio_manager.optimize_portfolio_allocation('performance_based')
```

**Portfolio Rebalancing**

```python
# Scheduled rebalancing
result = await portfolio_manager.rebalance_portfolio('scheduled')

# Threshold-based rebalancing
result = await portfolio_manager.rebalance_portfolio('threshold', threshold=0.05)

# Performance-triggered
result = await portfolio_manager.rebalance_portfolio('performance')

# Risk-triggered
result = await portfolio_manager.rebalance_portfolio('risk')
```

**Portfolio Summary**

```python
summary = portfolio_manager.get_portfolio_summary()
print(f"Portfolio value: ${summary['portfolio_state']['total_value']:,.2f}")
print(f"Active strategies: {len(summary['registered_strategies'])}")
print(f"Allocations: {summary['strategy_allocations']}")
```

### 2. Strategy Coordinator (`src/core/strategy_coordinator.py`)

Coordinates signals across multiple strategies.

#### Key Methods

**Signal Processing**

```python
result = await coordinator.process_strategy_signal(
    strategy_name='momentum',
    signal={
        'symbol': 'BTC/USDT:USDT',
        'side': 'long',
        'entry': 50000,
        'stop': 49000,
        'target': 52000
    }
)
```

**Conflict Resolution**

```python
from core.strategy_coordinator import ConflictResolutionStrategy

result = await coordinator.resolve_signal_conflicts(
    new_signal,
    conflicting_signals,
    resolution_strategy=ConflictResolutionStrategy.HIGHEST_PRIORITY
)
```

**Available Resolution Strategies:**
- `HIGHEST_PRIORITY`: Use signal with highest priority
- `BEST_RISK_REWARD`: Use signal with best risk/reward ratio
- `PERFORMANCE_WEIGHTED`: Weight by strategy performance
- `FIRST_IN_FIRST_OUT`: Existing signals take precedence

**Get Queued Signals**

```python
# Get next signal from queue
signal = await coordinator.get_next_signal(timeout=1.0)

if signal:
    print(f"Processing signal: {signal['signal_id']}")
```

**Processing Statistics**

```python
stats = coordinator.get_processing_stats()
print(f"Total signals: {stats['stats']['total_signals']}")
print(f"Accepted: {stats['stats']['accepted_signals']}")
print(f"Rejected: {stats['stats']['rejected_signals']}")
print(f"Conflicts: {stats['stats']['conflicted_signals']}")
```

---

## Usage Examples

### Example 1: Multi-Strategy Portfolio Setup

```python
import asyncio
from core.portfolio_manager import PortfolioManager
from core.risk_manager import RiskManager
from core.performance_monitor import RealTimePerformanceMonitor

async def setup_portfolio():
    # Initialize
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    
    # Register strategies with allocations
    strategies = {
        'momentum': 0.30,
        'mean_reversion': 0.25,
        'breakout': 0.25,
        'trend_following': 0.20
    }
    
    for name, allocation in strategies.items():
        result = portfolio_manager.register_strategy(
            name,
            strategy_instance,  # Your strategy object
            allocation
        )
        print(f"{name}: {allocation:.1%} - ${result['allocated_capital']:,.2f}")
    
    return portfolio_manager

asyncio.run(setup_portfolio())
```

### Example 2: Optimizing Portfolio Allocation

```python
async def optimize_and_rebalance():
    # ... portfolio_manager setup ...
    
    # Run optimization
    result = await portfolio_manager.optimize_portfolio_allocation('performance_based')
    
    if result['status'] == 'success':
        print("Optimization successful!")
        for strategy, new_alloc in result['new_allocations'].items():
            old_alloc = result['old_allocations'][strategy]
            change = result['allocation_changes'][strategy]
            print(f"{strategy}: {old_alloc:.1%} → {new_alloc:.1%} (change: {change:+.1%})")
        
        # Apply rebalancing
        rebalance_result = await portfolio_manager.rebalance_portfolio(
            'scheduled',
            apply=True
        )
        
        if rebalance_result['status'] == 'success':
            print(f"Rebalanced {len(rebalance_result['rebalancing_actions'])} strategies")

asyncio.run(optimize_and_rebalance())
```

### Example 3: Processing and Coordinating Signals

```python
async def process_strategy_signals():
    # ... setup portfolio_manager and coordinator ...
    
    # Strategy 1 generates signal
    signal1 = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'long',
        'entry': 50000,
        'stop': 49000,
        'target': 52000
    }
    
    result1 = await coordinator.process_strategy_signal('momentum', signal1)
    
    if result1['status'] == 'accepted':
        # Get signal from queue for execution
        queued_signal = await coordinator.get_next_signal()
        
        # Execute trade (your execution logic)
        execution_result = await execute_trade(queued_signal)
        
        # Mark as executed
        coordinator.mark_signal_executed(
            queued_signal['signal_id'],
            execution_result
        )

asyncio.run(process_strategy_signals())
```

---

## Optimization Methods

### 1. Markowitz Optimization (Modern Portfolio Theory)

Maximizes Sharpe ratio or achieves target return with minimum variance.

```python
result = await portfolio_manager.optimize_portfolio_allocation(
    'markowitz',
    target_return=0.02  # Optional 2% target
)
```

**Best for:**
- Balanced risk-adjusted returns
- Diversified portfolio
- When historical returns are available

### 2. Risk Parity

Equal risk contribution from each strategy.

```python
result = await portfolio_manager.optimize_portfolio_allocation('risk_parity')
```

**Best for:**
- Risk-balanced portfolios
- Strategies with different volatilities
- Conservative capital preservation

### 3. Performance-Based Optimization

Allocates based on recent performance metrics (win rate, Sharpe, profit factor).

```python
result = await portfolio_manager.optimize_portfolio_allocation('performance_based')
```

**Best for:**
- Adaptive allocation
- Rewarding top performers
- Dynamic market conditions

### 4. Black-Litterman Model

Combines market equilibrium with investor views (simplified implementation).

```python
result = await portfolio_manager.optimize_portfolio_allocation(
    'black_litterman',
    target_return=0.015
)
```

**Note:** Current implementation uses Markowitz as fallback. Full Black-Litterman requires market views.

---

## Rebalancing Strategies

### Scheduled Rebalancing

Rebalances at regular intervals (default: 24 hours).

```python
result = await portfolio_manager.rebalance_portfolio('scheduled', apply=True)
```

**Triggers when:**
- More than 24 hours since last rebalance
- Initial rebalance (no previous rebalance)

### Threshold-Based Rebalancing

Rebalances when allocation drift exceeds threshold.

```python
result = await portfolio_manager.rebalance_portfolio(
    'threshold',
    threshold=0.05,  # 5% drift threshold
    apply=True
)
```

**Triggers when:**
- Any strategy's allocation drifts > 5% from target

### Performance-Triggered Rebalancing

Rebalances when strategy performance degrades significantly.

```python
result = await portfolio_manager.rebalance_portfolio('performance', apply=True)
```

**Triggers when:**
- Recent win rate < 70% of historical win rate
- Significant performance degradation detected

### Risk-Triggered Rebalancing

Rebalances when portfolio risk exceeds acceptable levels.

```python
result = await portfolio_manager.rebalance_portfolio('risk', apply=True)
```

**Triggers when:**
- Portfolio heat > 8%
- Risk metrics exceed thresholds

---

## Signal Coordination

### Signal Processing Pipeline

1. **Validation**: Check required fields and value ranges
2. **Enrichment**: Add strategy metadata, allocation, priority
3. **Conflict Detection**: Check for conflicts with existing signals/positions
4. **Risk Assessment**: Calculate position size and validate with risk manager
5. **Priority Routing**: Route based on priority and execution method
6. **Queue Management**: Add to execution queue

### Signal Priority Levels

```python
from core.strategy_coordinator import SignalPriority

# Priorities (highest to lowest):
# - CRITICAL: Excellent strategy performance + high confidence
# - HIGH: Good strategy performance
# - MEDIUM: Average performance (default)
# - LOW: Poor strategy performance
```

Priority is automatically calculated based on:
- Strategy win rate
- Sharpe ratio
- Profit factor
- Signal confidence (if provided)

### Conflict Resolution

When signals conflict (same symbol, opposite sides), the coordinator resolves using:

**1. Highest Priority**
```python
ConflictResolutionStrategy.HIGHEST_PRIORITY
```
Selects signal with highest priority level.

**2. Best Risk/Reward**
```python
ConflictResolutionStrategy.BEST_RISK_REWARD
```
Selects signal with best risk/reward ratio.

**3. Performance Weighted**
```python
ConflictResolutionStrategy.PERFORMANCE_WEIGHTED
```
Selects signal from best-performing strategy.

**4. First-In-First-Out**
```python
ConflictResolutionStrategy.FIRST_IN_FIRST_OUT
```
Existing signals take precedence.

---

## Integration

### Integration with Phase 3.2 Risk Management

Portfolio Manager fully integrates with Risk Manager for:

- Position size calculation
- Position validation
- Risk limit enforcement
- Portfolio heat monitoring
- Drawdown tracking

```python
# Risk manager validates all positions
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal,
    current_portfolio
)
```

### Integration with Phase 2 Performance Monitor

Tracks strategy performance for optimization decisions:

- Win rate tracking
- Sharpe ratio calculation
- Profit factor analysis
- Drawdown monitoring
- Performance-based allocation

```python
# Performance monitor provides metrics
summary = performance_monitor.get_strategy_summary('momentum')
metrics = summary['metrics']
```

### Integration with Phase 3.1 WebSocket Manager

Optional integration for real-time data:

```python
portfolio_manager = PortfolioManager(
    risk_manager,
    performance_monitor,
    websocket_manager  # Optional WebSocket integration
)
```

---

## Best Practices

### 1. Strategy Registration

- Allocate conservatively (leave 10-20% unallocated)
- Register strategies with proven track records
- Monitor strategy performance continuously
- Deactivate underperforming strategies

```python
# Good allocation
strategies = {
    'momentum': 0.25,
    'reversal': 0.25,
    'breakout': 0.20,
    # 30% unallocated for new opportunities
}
```

### 2. Portfolio Optimization

- Run optimization after sufficient performance data (30+ trades per strategy)
- Use appropriate method for your goals:
  - Risk Parity: Conservative, risk-balanced
  - Markowitz: Risk-adjusted returns
  - Performance-based: Adaptive, momentum-driven
- Don't optimize too frequently (weekly is reasonable)

### 3. Rebalancing

- Use multiple triggers:
  - Scheduled for regular maintenance
  - Performance for adaptive allocation
  - Risk for protection
- Set reasonable thresholds (5-10% drift)
- Apply rebalancing during low-volatility periods
- Keep rebalancing costs in mind

### 4. Signal Processing

- Always validate signals before processing
- Monitor conflict resolution decisions
- Track processing statistics
- Handle rejected signals appropriately

### 5. Monitoring

```python
# Regular monitoring
summary = portfolio_manager.get_portfolio_summary()
stats = coordinator.get_processing_stats()

# Log important metrics
print(f"Portfolio value: ${summary['portfolio_state']['total_value']:,.2f}")
print(f"Signal acceptance rate: {stats['stats']['accepted_signals'] / stats['stats']['total_signals']:.1%}")
```

---

## Troubleshooting

### Common Issues

**1. Optimization returns insufficient data**

- **Cause**: Not enough performance history
- **Solution**: Ensure 20+ trades per strategy before optimizing
- **Check**: `performance_monitor.get_strategy_summary(strategy_name)`

**2. Signals always rejected**

- **Cause**: Risk limits too strict or position sizing issues
- **Solution**: Review risk manager configuration
- **Check**: Risk limits, stop-loss distances, position sizes

**3. Portfolio never rebalances**

- **Cause**: Triggers not met or thresholds too high
- **Solution**: Adjust thresholds or use scheduled rebalancing
- **Check**: `rebalance_portfolio(..., apply=False)` to see trigger status

**4. Conflicts not resolving properly**

- **Cause**: Priority calculation or resolution strategy
- **Solution**: Ensure performance data is available for priority calculation
- **Check**: Signal priorities and conflict history

**5. Allocations don't sum to 1.0**

- **Cause**: By design - allows unallocated capital
- **Solution**: This is intentional for flexibility
- **Note**: Unallocated capital remains available for new strategies

### Debugging Tips

```python
# Enable debug logging
import logging
logging.getLogger('core.portfolio_manager').setLevel(logging.DEBUG)
logging.getLogger('core.strategy_coordinator').setLevel(logging.DEBUG)

# Check portfolio state
summary = portfolio_manager.get_portfolio_summary()
print(f"Allocated: ${summary['portfolio_state']['allocated_capital']:,.2f}")
print(f"Available: ${summary['portfolio_state']['available_capital']:,.2f}")

# Check coordinator stats
stats = coordinator.get_processing_stats()
print(f"Processing stats: {stats['stats']}")

# Review active signals
active = coordinator.get_active_signals_summary()
for signal in active:
    print(f"Signal: {signal['symbol']} {signal['side']} - {signal['status']}")
```

---

## API Reference

See individual module documentation:
- [Portfolio Manager API](../src/core/portfolio_manager.py)
- [Strategy Coordinator API](../src/core/strategy_coordinator.py)
- [Risk Manager API](../src/core/risk_manager.py)
- [Performance Monitor API](../src/core/performance_monitor.py)

---

## Support

For issues or questions:
1. Check the [tests](../tests/test_portfolio_management.py) for usage examples
2. Run the [example](../examples/portfolio_management_example.py)
3. Review logs for detailed error messages
4. Open an issue on GitHub

---

**Phase 3.3 Portfolio Management Engine** - Advanced Multi-Strategy Coordination for Algorithmic Trading
