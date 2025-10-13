# Risk Management Engine Guide

## Phase 3.2: Advanced Portfolio Protection

This guide covers the comprehensive risk management system implemented in Phase 3.2, providing real-time portfolio protection, position sizing optimization, and emergency stop mechanisms.

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Integration](#integration)
6. [Best Practices](#best-practices)

---

## Overview

The Risk Management Engine provides:

- **Portfolio-level risk management**: Validates positions against multiple risk criteria
- **Advanced position sizing**: Kelly Criterion, fixed risk, volatility-adjusted, and regime-based algorithms
- **Real-time monitoring**: Continuous risk assessment using WebSocket feeds
- **Correlation management**: Portfolio diversification optimization
- **Emergency protocols**: Automatic position closure on risk threshold breach
- **VaR calculation**: Historical, Parametric, and Expected Shortfall

### Key Features

✅ **Multi-layer risk validation**  
✅ **4 position sizing algorithms**  
✅ **Real-time alerts and monitoring**  
✅ **Circuit breaker system**  
✅ **Correlation-aware position management**  
✅ **Integration with Phase 1 (Multi-Exchange) and Phase 2 (Market Intelligence)**

---

## Components

### 1. Risk Configuration (`src/config/risk_config.py`)

Centralized risk parameter management.

```python
from config.risk_config import RiskConfiguration

# Create default configuration
config = RiskConfiguration()

# Custom configuration
custom_config = RiskConfiguration({
    'max_portfolio_risk': 0.015,  # 1.5% max risk per trade
    'max_position_size': 0.08,    # 8% max position size
    'max_drawdown': 0.12,         # 12% max drawdown
    'daily_loss_limit': 0.04      # 4% daily loss limit
})

# Access limits
risk_limits = config.get_risk_limits()
breaker_limits = config.get_circuit_breaker_limits()
```

**Default Risk Limits:**
- Max portfolio risk per trade: 2%
- Max position size: 10% of portfolio
- Max drawdown: 15%
- Max correlation: 70%
- Stop-loss multiplier: 2x ATR
- Risk/reward ratio minimum: 2:1

**Circuit Breaker Limits:**
- Daily loss limit: 5%
- Position loss limit: 3%
- Volatility spike threshold: 3σ
- Correlation spike threshold: 90%

### 2. Risk Manager (`src/core/risk_manager.py`)

Main risk management engine for position validation and monitoring.

```python
from core.risk_manager import RiskManager

# Initialize
portfolio_config = {
    'equity_usd': 10000,
    'max_portfolio_risk': 0.02,
    'max_position_size': 0.10
}

risk_manager = RiskManager(portfolio_config, websocket_manager, performance_monitor)

# Validate new position
signal = {
    'symbol': 'BTC/USDT:USDT',
    'entry': 50000,
    'stop': 49000,
    'target': 52000,
    'position_size': 0.1,
    'side': 'long'
}

is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})

if is_valid:
    # Register position
    risk_manager.register_position('pos_1', {
        'symbol': signal['symbol'],
        'entry_price': signal['entry'],
        'stop_loss': signal['stop'],
        'size': signal['position_size'],
        'side': signal['side'],
        'risk_amount': metrics['risk_amount']
    })
```

**Validation Checks:**
1. Position size vs. portfolio limit
2. Risk amount vs. portfolio risk limit
3. Risk/reward ratio (minimum 1.5:1)
4. Current drawdown vs. max drawdown
5. Portfolio heat (total risk exposure)
6. Strategy performance (if available)

### 3. Position Sizing (`src/core/position_sizing.py`)

Advanced algorithms for optimal capital allocation.

```python
from core.position_sizing import AdvancedPositionSizing

sizing = AdvancedPositionSizing(risk_manager)

# Kelly Criterion
position_size = await sizing.calculate_optimal_size(
    signal,
    method='kelly',
    performance_history={'win_rate': 0.6, 'avg_win': 100, 'avg_loss': 50}
)

# Fixed Risk
position_size = await sizing.calculate_optimal_size(
    signal,
    method='fixed_risk',
    risk_per_trade=200
)

# Volatility Adjusted
position_size = await sizing.calculate_optimal_size(
    signal,
    method='volatility_adjusted',
    target_risk=200,
    market_volatility=500
)

# Regime Based
position_size = await sizing.calculate_optimal_size(
    signal,
    method='regime_based',
    market_regime={'trend': 'bullish', 'risk_multiplier': 1.2}
)
```

**Sizing Methods:**

1. **Kelly Criterion**: Optimal fraction based on win rate and win/loss ratio
2. **Fixed Risk**: Fixed dollar amount at risk per trade
3. **Volatility Adjusted**: Scales inversely with market volatility (ATR)
4. **Regime Based**: Adjusts for market regime, trend alignment, and performance

### 4. Real-Time Risk Monitor (`src/core/realtime_risk.py`)

Continuous monitoring using WebSocket feeds.

```python
from core.realtime_risk import RealTimeRiskMonitor

monitor = RealTimeRiskMonitor(risk_manager, websocket_manager)

# Start monitoring
await monitor.start_risk_monitoring()

# Get alerts
alerts = await monitor.get_risk_alerts(count=10)

# Calculate VaR
var_metrics = monitor.calculate_portfolio_var(confidence=0.05)
```

**Monitoring Features:**
- Stop-loss breach detection
- Unrealized P&L tracking
- Portfolio heat assessment
- Price spike detection
- VaR calculation (Historical, Parametric, Expected Shortfall)

### 5. Correlation Monitor (`src/core/correlation_monitor.py`)

Portfolio diversification and correlation tracking.

```python
from core.correlation_monitor import CorrelationMonitor

monitor = CorrelationMonitor(websocket_manager)

# Update correlation matrix
await monitor.update_correlation_matrix(['BTC/USDT:USDT', 'ETH/USDT:USDT'])

# Calculate diversification
metrics = monitor.calculate_portfolio_diversification(positions)

# Validate new position
is_valid, reason, corr_data = await monitor.validate_new_position_correlation(
    'SOL/USDT:USDT',
    existing_positions,
    max_correlation=0.7
)
```

**Diversification Metrics:**
- Effective number of positions (Herfindahl index)
- Concentration risk (max single position weight)
- Diversification ratio
- Correlation matrix

### 6. Circuit Breaker System (`src/core/circuit_breaker.py`)

Emergency protocols for extreme conditions.

```python
from core.circuit_breaker import CircuitBreakerSystem

breaker = CircuitBreakerSystem(risk_manager, websocket_manager)

# Configure breakers
breaker.set_circuit_breakers(
    daily_loss_limit=0.05,
    position_loss_limit=0.03,
    volatility_spike_threshold=3.0
)

# Start monitoring
breaker.start_monitoring()

# Manual trigger (if needed)
await breaker.trigger_circuit_breaker('daily_loss', severity='critical')

# Emergency protocols
await breaker.execute_emergency_protocol('close_all')
```

**Emergency Protocols:**
- `close_all`: Close all positions immediately
- `close_positions`: Close specific positions
- `reduce_positions`: Reduce all position sizes by 50%
- `redistribute_positions`: Move positions across exchanges

---

## Configuration

### Risk Configuration File

Create a risk configuration in your main config:

```yaml
risk_management:
  # Portfolio limits
  max_portfolio_risk: 0.02      # 2% max risk per trade
  max_position_size: 0.10       # 10% max position size
  max_drawdown: 0.15            # 15% max drawdown
  max_correlation: 0.70         # 70% max correlation
  
  # Position parameters
  stop_loss_multiplier: 2.0     # 2x ATR for stop loss
  take_profit_ratio: 2.0        # 2:1 risk/reward minimum
  
  # Circuit breakers
  daily_loss_limit: 0.05        # 5% daily loss limit
  position_loss_limit: 0.03     # 3% per position loss limit
  volatility_spike_threshold: 3.0  # 3 sigma threshold
  
  # Position sizing
  default_method: 'fixed_risk'  # kelly, fixed_risk, volatility_adjusted, regime_based
  kelly_fraction: 0.5           # 50% of Kelly for safety
```

### Environment Variables

```bash
# Risk management settings
RISK_MAX_PORTFOLIO_RISK=0.02
RISK_MAX_POSITION_SIZE=0.10
RISK_DAILY_LOSS_LIMIT=0.05

# Circuit breaker settings
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_DAILY_LOSS=0.05
CIRCUIT_BREAKER_POSITION_LOSS=0.03
```

---

## Usage Examples

### Complete Risk Management Workflow

```python
import asyncio
from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager
from core.position_sizing import AdvancedPositionSizing
from core.realtime_risk import RealTimeRiskMonitor
from core.circuit_breaker import CircuitBreakerSystem
from core.websocket_manager import WebSocketManager
from core.performance_monitor import RealTimePerformanceMonitor

async def main():
    # 1. Initialize components
    config = RiskConfiguration()
    
    portfolio_config = {
        'equity_usd': 10000,
        'max_portfolio_risk': 0.02,
        'max_position_size': 0.10
    }
    
    ws_manager = WebSocketManager()
    perf_monitor = RealTimePerformanceMonitor()
    
    risk_manager = RiskManager(portfolio_config, ws_manager, perf_monitor)
    sizing = AdvancedPositionSizing(risk_manager)
    risk_monitor = RealTimeRiskMonitor(risk_manager, ws_manager)
    breaker = CircuitBreakerSystem(risk_manager, ws_manager)
    
    # 2. Configure risk limits
    risk_manager.set_risk_limits(max_portfolio_risk=0.02)
    breaker.set_circuit_breakers(daily_loss_limit=0.05)
    
    # 3. Start monitoring
    await risk_monitor.start_risk_monitoring()
    breaker.start_monitoring()
    
    # 4. Process trading signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'entry': 50000,
        'stop': 49000,
        'target': 52000,
        'side': 'long',
        'strategy': 'oversold_bounce'
    }
    
    # 5. Calculate position size
    market_regime = {'trend': 'bullish', 'risk_multiplier': 1.2}
    position_size = await sizing.calculate_optimal_size(
        signal,
        method='regime_based',
        market_regime=market_regime
    )
    
    signal['position_size'] = position_size
    
    # 6. Validate position
    is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
    
    if is_valid:
        # 7. Execute trade and register position
        # (Execute via exchange API here)
        
        risk_manager.register_position('pos_1', {
            'symbol': signal['symbol'],
            'entry_price': signal['entry'],
            'stop_loss': signal['stop'],
            'size': position_size,
            'side': signal['side'],
            'risk_amount': metrics['risk_amount']
        })
        
        print(f"✓ Position opened: {position_size:.4f} @ {signal['entry']}")
    else:
        print(f"✗ Position rejected: {reason}")
    
    # 8. Monitor and manage
    while True:
        # Check for risk alerts
        alerts = await risk_monitor.get_risk_alerts(count=5)
        
        for alert in alerts:
            if alert['type'] == 'stop_loss_trigger':
                # Close position
                risk_manager.close_position(alert['position_id'], alert['current_price'], 0)
        
        # Get portfolio status
        summary = risk_manager.get_portfolio_summary()
        print(f"Portfolio: ${summary['portfolio_value']:,.2f}, "
              f"Heat: {summary['portfolio_heat']:.2%}")
        
        await asyncio.sleep(30)

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Integration

### With Phase 1 (Multi-Exchange)

```python
from core.multi_exchange_manager import MultiExchangeManager

# Risk management across multiple exchanges
exchanges = {
    'kucoinfutures': CcxtClient('kucoinfutures'),
    'bingx': CcxtClient('bingx')
}

multi_exchange = MultiExchangeManager(exchanges)

# Position sizing per exchange
for exchange_name in exchanges.keys():
    position_size = await risk_manager.calculate_position_size(
        signal,
        portfolio_state={'exchange': exchange_name}
    )
```

### With Phase 2 (Market Intelligence)

```python
from core.market_regime import MarketRegimeAnalyzer

# Regime-aware risk management
regime_analyzer = MarketRegimeAnalyzer()
regime = regime_analyzer.analyze_market_regime(df_30m, df_1h, df_4h)

# Adjust sizing based on regime
position_size = await sizing.calculate_optimal_size(
    signal,
    method='regime_based',
    market_regime=regime
)
```

### With Phase 3.1 (WebSocket)

```python
# Real-time risk monitoring with WebSocket feeds
async def price_callback(exchange, symbol, ticker):
    await risk_monitor.on_price_update(symbol, ticker)

await ws_manager.stream_tickers(
    symbols_per_exchange={'kucoinfutures': ['BTC/USDT:USDT']},
    callback=price_callback
)
```

---

## Best Practices

### 1. Risk Limit Configuration

- Start conservative (1-2% risk per trade)
- Gradually increase as system proves profitable
- Never exceed 5% risk per trade
- Keep max drawdown under 20%

### 2. Position Sizing

- Use **Fixed Risk** for consistency
- Use **Kelly Criterion** only with proven edge (>55% win rate)
- Use **Volatility Adjusted** in ranging markets
- Use **Regime Based** with market intelligence integration

### 3. Correlation Management

- Keep position correlation under 70%
- Aim for 3+ effective positions for diversification
- Monitor concentration risk (no single position >30%)
- Update correlation matrix daily

### 4. Circuit Breakers

- Always enable daily loss limit (5%)
- Set position loss limit (3%)
- Enable volatility spike detection
- Test emergency protocols in paper trading

### 5. Monitoring

- Review risk alerts every 15 minutes during trading
- Calculate VaR daily
- Monitor portfolio heat continuously
- Adjust limits based on market conditions

### 6. Testing

- Backtest position sizing algorithms
- Stress test circuit breakers
- Validate correlation calculations
- Test emergency protocols regularly

---

## Performance Metrics

### Key Risk Metrics

```python
# Get comprehensive risk metrics
summary = risk_manager.get_portfolio_summary()

print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
print(f"Current Drawdown: {summary['current_drawdown']:.2%}")
print(f"Active Positions: {summary['active_positions']}")
print(f"Total Risk: ${summary['total_risk']:.2f}")
print(f"Portfolio Heat: {summary['portfolio_heat']:.2%}")

# VaR calculation
var_metrics = risk_monitor.calculate_portfolio_var()
print(f"VaR (95%): ${var_metrics['historical_var']:.2f}")
print(f"Expected Shortfall: ${var_metrics['expected_shortfall']:.2f}")

# Diversification
div_metrics = corr_monitor.calculate_portfolio_diversification(positions)
print(f"Effective Positions: {div_metrics['effective_positions']:.2f}")
print(f"Concentration Risk: {div_metrics['concentration_risk']:.2%}")
```

---

## Troubleshooting

### Common Issues

**1. Position validation always fails**
- Check if portfolio_value is set correctly
- Verify stop loss is not too wide
- Ensure risk/reward ratio is >1.5

**2. Circuit breaker triggers too often**
- Increase daily_loss_limit
- Review position sizing
- Check for correlated positions

**3. VaR calculation returns 0**
- Need at least 20 price points
- Check price history buffer
- Verify symbols are updating

**4. Correlation matrix empty**
- Ensure price history is being updated
- Need minimum 30 price points per symbol
- Check symbol names match exactly

---

## API Reference

See individual module documentation:
- [Risk Manager API](../src/core/risk_manager.py)
- [Position Sizing API](../src/core/position_sizing.py)
- [Real-Time Risk Monitor API](../src/core/realtime_risk.py)
- [Correlation Monitor API](../src/core/correlation_monitor.py)
- [Circuit Breaker API](../src/core/circuit_breaker.py)

---

## Support

For issues or questions:
1. Check the [tests](../tests/test_risk_management.py) for usage examples
2. Run the [example](../examples/risk_management_example.py)
3. Review logs for detailed error messages
4. Open an issue on GitHub

---

**Phase 3.2 Risk Management Engine** - Advanced Portfolio Protection for Algorithmic Trading
