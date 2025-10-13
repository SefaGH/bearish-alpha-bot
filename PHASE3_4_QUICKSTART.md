# Phase 3.4: Live Trading Engine - Quick Start Guide

## Overview
Phase 3.4 provides a production-ready live trading execution system. This guide helps you get started quickly.

## Installation

```bash
# Clone repository (if not already done)
git clone https://github.com/SefaGH/bearish-alpha-bot.git
cd bearish-alpha-bot

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Paper Trading Mode (Recommended First Step)

Test the system without risking real money:

```bash
# Set environment variables
export TRADING_MODE=paper
export TRADING_DURATION=300  # Run for 5 minutes (0 = indefinite)

# Run live trading mode
python src/main.py --live
```

### 2. Run Examples

Explore all capabilities through comprehensive examples:

```bash
python examples/live_trading_example.py
```

This runs 6 examples demonstrating:
- Basic live trading setup
- Strategy registration
- Signal execution
- Position management
- Execution analytics
- Production trading loop

### 3. Run Tests

Verify everything is working:

```bash
pytest tests/test_live_trading_engine.py -v
```

## Basic Usage

### Initialize Production System

```python
from core.production_coordinator import ProductionCoordinator
from core.ccxt_client import CcxtClient

# Create exchange clients
exchange_clients = {
    'kucoinfutures': CcxtClient('kucoinfutures')
}

# Initialize coordinator
coordinator = ProductionCoordinator()

# Initialize production system
await coordinator.initialize_production_system(
    exchange_clients=exchange_clients,
    portfolio_config={'equity_usd': 10000}
)
```

### Register Trading Strategies

```python
# Register multiple strategies with capital allocation
coordinator.register_strategy('momentum', momentum_strategy, 0.30)
coordinator.register_strategy('mean_reversion', mr_strategy, 0.30)
coordinator.register_strategy('breakout', breakout_strategy, 0.40)
```

### Execute Trading Signals

```python
# Create trading signal
signal = {
    'symbol': 'BTC/USDT:USDT',
    'side': 'buy',
    'entry': 50000.0,
    'stop': 49000.0,
    'target': 52000.0,
    'strategy': 'momentum',
    'exchange': 'kucoinfutures'
}

# Submit signal
result = await coordinator.submit_signal(signal)
```

### Run Production Loop

```python
# Run paper trading for testing
await coordinator.run_production_loop(mode='paper', duration=3600)

# Run live trading (when ready)
# await coordinator.run_production_loop(mode='live')
```

## Configuration

### Environment Variables

```bash
# Trading mode
export TRADING_MODE=paper        # Options: paper, live, simulation

# Duration (seconds, 0 = indefinite)
export TRADING_DURATION=0

# Risk parameters
export RISK_EQUITY_USD=10000
export RISK_PER_TRADE_RISK_PCT=0.01
export RISK_MAX_NOTIONAL_PER_TRADE=500
export RISK_DAILY_MAX_TRADES=5

# Exchange credentials (for live trading)
export EXCHANGES=kucoinfutures
export KUCOIN_KEY=your_api_key
export KUCOIN_SECRET=your_api_secret
export KUCOIN_PASSWORD=your_api_password
```

### Configuration Files

Edit `src/config/live_trading_config.py` for advanced settings:

```python
EXECUTION_CONFIG = {
    'default_execution_algo': 'limit',
    'max_slippage_tolerance': 0.005,  # 0.5%
    'order_timeout': 300,
}

MONITORING_CONFIG = {
    'position_check_interval': 10,   # seconds
    'pnl_update_frequency': 5,       # seconds
    'risk_check_frequency': 1,       # seconds
}

EMERGENCY_CONFIG = {
    'max_daily_loss': 0.05,          # 5%
    'enable_circuit_breaker': True,
}
```

## Trading Modes

### Paper Trading
- **Purpose**: Testing without real money
- **Use**: Development, strategy validation
- **Risk**: None (simulated)
- **Command**: `TRADING_MODE=paper python src/main.py --live`

### Simulation
- **Purpose**: Historical data replay
- **Use**: Strategy backtesting with live engine
- **Risk**: None (historical data)
- **Command**: `TRADING_MODE=simulation python src/main.py --live`

### Live Trading
- **Purpose**: Real money trading
- **Use**: Production deployment
- **Risk**: Real capital at risk
- **Command**: `TRADING_MODE=live python src/main.py --live`
- **⚠️ Warning**: Only use after thorough paper trading validation!

## Execution Algorithms

The system supports 4 execution algorithms:

### 1. Market Orders
- **Use**: Immediate execution needed
- **Pros**: Fast execution
- **Cons**: Higher slippage
- **Best for**: Small orders, urgent exits

### 2. Limit Orders (Default)
- **Use**: Normal trading
- **Pros**: Better pricing, low slippage
- **Cons**: May not fill immediately
- **Best for**: Most trading scenarios

### 3. Iceberg Orders
- **Use**: Large orders
- **Pros**: Reduced market impact
- **Cons**: Slower execution
- **Best for**: Large positions (>$50k)

### 4. TWAP (Time-Weighted Average Price)
- **Use**: Distributed execution
- **Pros**: Minimized market impact
- **Cons**: Takes time to complete
- **Best for**: Large orders over time

## Monitoring

### Check System Status

```python
# Get engine status
status = coordinator.trading_engine.get_engine_status()
print(f"State: {status['state']}")
print(f"Active positions: {status['active_positions']}")
print(f"Total trades: {status['total_trades']}")

# Get execution statistics
stats = coordinator.trading_engine.order_manager.get_execution_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg execution time: {stats['avg_execution_time']:.2f}s")

# Get position summary
summary = coordinator.trading_engine.position_manager.get_position_summary()
print(f"Total P&L: ${summary['total_pnl']:.2f}")
```

### Generate Reports

```python
# Execution quality report
analytics = coordinator.trading_engine.execution_analytics
report = analytics.generate_execution_report('1d')

if report['success']:
    print(report['report'])
```

## Emergency Stop

### Graceful Shutdown

```python
# Stop trading loop
await coordinator.stop_system()
```

### Emergency Shutdown

```python
# Emergency stop with position closure
await coordinator.handle_emergency_shutdown('manual_intervention')
```

### Keyboard Interrupt

Press `Ctrl+C` to trigger graceful shutdown.

## Best Practices

### Before Live Trading

1. ✅ Run paper trading for at least 1 week
2. ✅ Test with small capital first
3. ✅ Validate risk parameters
4. ✅ Monitor system stability
5. ✅ Review execution quality reports
6. ✅ Test emergency shutdown procedures

### During Live Trading

1. 📊 Monitor positions regularly
2. 🔍 Review execution quality daily
3. ⚡ Keep risk limits conservative
4. 🛡️ Enable circuit breaker
5. 📱 Set up alerts/notifications
6. 💾 Backup system state regularly

### Risk Management

1. **Start Small**: Use minimal capital initially
2. **Conservative Limits**: Max 2% risk per trade
3. **Daily Loss Limit**: Stop trading at 5% daily loss
4. **Position Size**: Max 10% of portfolio per position
5. **Diversification**: Use multiple strategies
6. **Emergency Plan**: Have exit strategy ready

## Troubleshooting

### System Won't Start

```bash
# Check dependencies
pip install -r requirements.txt

# Verify exchange credentials
python -c "from core.ccxt_client import CcxtClient; c = CcxtClient('kucoinfutures'); print('OK')"

# Run tests
pytest tests/test_live_trading_engine.py
```

### Orders Not Executing

- Check exchange connectivity
- Verify API credentials
- Review risk validation logs
- Check order validation errors

### High Slippage

- Switch to limit orders
- Reduce order size
- Use TWAP for large orders
- Check market liquidity

## Support

- **Documentation**: See `PHASE3_4_LIVE_TRADING_SUMMARY.md`
- **Examples**: Run `python examples/live_trading_example.py`
- **Tests**: Run `pytest tests/test_live_trading_engine.py -v`
- **Issues**: Check GitHub issues for known problems

## What's Next?

1. **Validate**: Run paper trading extensively
2. **Optimize**: Fine-tune risk parameters
3. **Deploy**: Start live trading with minimal capital
4. **Monitor**: Track performance and adjust
5. **Scale**: Gradually increase capital allocation

---

**Status**: ✅ Production Ready (Paper Trading Validated)

**Warning**: ⚠️ Always start with paper trading before live deployment!
