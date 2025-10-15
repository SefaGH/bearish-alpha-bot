# Monitoring and Alerting System

## Overview

The Bearish Alpha Bot includes a comprehensive monitoring and alerting system that provides real-time visibility into trading operations, performance metrics, and system health.

![Monitoring Dashboard](https://github.com/user-attachments/assets/db5671f6-fc1b-4388-8aa0-d4e8a1009aa6)

## Components

### 1. Real-Time Dashboard

A web-based monitoring dashboard with live updates via WebSocket.

**Features:**
- Real-time metrics display
- Health status monitoring
- P&L tracking
- Win rate calculation
- Active positions view
- Recent signals table
- System information

**Usage:**

```python
from src.monitoring.dashboard import MonitoringDashboard

# Initialize dashboard
dashboard = MonitoringDashboard(port=8080)

# Start dashboard server
await dashboard.start()

# Update metrics
dashboard.update_metrics(
    total_signals=10,
    total_trades=8,
    win_rate=0.625,
    total_pnl=150.50,
    health_status='healthy'
)

# Stop dashboard
await dashboard.stop()
```

**Access:** Open `http://localhost:8080` in your browser

### 2. Alert Manager

Multi-channel alert management with intelligent rate limiting and grouping.

**Supported Channels:**
- Telegram
- Discord (webhook)
- Email (placeholder for future implementation)
- Generic webhook

**Priority Levels:**
- `CRITICAL` - Rate limit: 1 minute
- `HIGH` - Rate limit: 5 minutes
- `MEDIUM` - Rate limit: 15 minutes
- `LOW` - Rate limit: 1 hour
- `INFO` - Rate limit: 4 hours

**Usage:**

```python
from src.monitoring.alert_manager import (
    AlertManager, AlertPriority, AlertChannel
)

# Configure alert channels
config = {
    'telegram': {
        'enabled': True,
        'bot_token': 'YOUR_BOT_TOKEN',
        'chat_id': 'YOUR_CHAT_ID'
    },
    'discord': {
        'enabled': True,
        'webhook_url': 'YOUR_DISCORD_WEBHOOK'
    }
}

# Initialize alert manager
alert_manager = AlertManager(config)

# Send an alert
await alert_manager.send_alert(
    title="Trading Alert",
    message="Significant price movement detected",
    priority=AlertPriority.HIGH,
    metadata={'symbol': 'BTC/USDT', 'price': 50000}
)

# Get alert summary
summary = alert_manager.get_alert_summary(hours=24)
print(f"Total alerts in last 24h: {summary['total_alerts']}")
```

**Alert Grouping:**

Similar alerts can be automatically grouped to prevent spam:

```python
# Alerts with same group_key will be grouped
await alert_manager.send_alert(
    title="Price Alert",
    message="Price crossed threshold",
    priority=AlertPriority.MEDIUM,
    group_key="price_alerts"
)
```

### 3. Performance Analytics

Calculate and track comprehensive trading performance metrics.

**Metrics Supported:**
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Risk/Reward Ratio
- Calmar Ratio

**Usage:**

```python
from src.monitoring.performance_analytics import PerformanceAnalytics
import pandas as pd

# Initialize analytics
analytics = PerformanceAnalytics(data_dir='data')

# Calculate Sharpe ratio
returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
sharpe = analytics.calculate_sharpe_ratio(returns)

# Calculate win rate
trades = [
    {'pnl': 100.0},
    {'pnl': -50.0},
    {'pnl': 75.0}
]
win_rate = analytics.calculate_win_rate(trades)

# Generate comprehensive report
report = analytics.generate_performance_report(
    trades=trades,
    equity_curve=equity_series
)

# Save report
analytics.save_report(report, 'performance_report.json')
```

## Integration with Existing Components

The monitoring system integrates seamlessly with existing bot components:

### With State Tracking

```python
from src.core.state import load_state, load_day_stats
from src.monitoring.dashboard import MonitoringDashboard

dashboard = MonitoringDashboard()

# Load current state
state = load_state()
day_stats = load_day_stats()

# Update dashboard
dashboard.update_metrics(
    open_positions=list(state['open'].values()),
    total_pnl=day_stats.get('pnl', 0.0),
    total_signals=day_stats.get('signals', 0)
)
```

### With Telegram Notifications

```python
from src.core.notify import Telegram
from src.monitoring.alert_manager import AlertManager, AlertChannel

# Configure with existing Telegram bot
config = {
    'telegram': {
        'enabled': True,
        'bot_token': 'YOUR_EXISTING_TOKEN',
        'chat_id': 'YOUR_CHAT_ID'
    }
}

alert_manager = AlertManager(config)
```

### With Live Trading Engine

```python
from src.monitoring.dashboard import MonitoringDashboard
from src.monitoring.alert_manager import AlertManager, AlertPriority

# In your trading loop
async def trading_loop():
    dashboard = MonitoringDashboard()
    alert_manager = AlertManager(config)
    
    await dashboard.start()
    
    while True:
        # Execute trades...
        
        # Update monitoring
        dashboard.update_metrics(
            total_trades=trade_count,
            win_rate=calculate_win_rate(),
            total_pnl=calculate_pnl()
        )
        
        # Send alerts for important events
        if significant_event:
            await alert_manager.send_alert(
                title="Trade Executed",
                message=f"Opened position on {symbol}",
                priority=AlertPriority.INFO
            )
        
        await asyncio.sleep(60)
```

## Demo and Testing

### Run the Demo

```bash
python examples/monitoring_demo.py
```

This will:
1. Start the dashboard on port 8080
2. Simulate trading activity
3. Update metrics in real-time
4. Generate sample alerts

### Run Tests

```bash
pytest tests/test_monitoring.py -v
```

All 18 tests should pass:
- Alert manager tests (5 tests)
- Performance analytics tests (9 tests)
- Dashboard tests (4 tests)

## Configuration

### Dashboard Configuration

```python
dashboard = MonitoringDashboard(
    port=8080  # Web server port
)
```

### Alert Manager Configuration

```python
config = {
    'telegram': {
        'enabled': True,
        'bot_token': 'YOUR_TOKEN',
        'chat_id': 'YOUR_CHAT_ID'
    },
    'discord': {
        'enabled': True,
        'webhook_url': 'https://discord.com/api/webhooks/...'
    },
    'email': {
        'enabled': False  # Not yet implemented
    },
    'webhook': {
        'enabled': True,
        'url': 'https://your-webhook-endpoint.com'
    }
}
```

### Performance Analytics Configuration

```python
analytics = PerformanceAnalytics(
    data_dir='data'  # Directory for storing reports
)
```

## Best Practices

1. **Rate Limiting**: Use appropriate priority levels to prevent alert spam
2. **Grouping**: Group similar alerts using `group_key` parameter
3. **Monitoring**: Check dashboard regularly for system health
4. **Performance Reports**: Generate and review performance reports weekly
5. **Error Handling**: Monitor alert delivery failures in logs

## Architecture

```
src/monitoring/
├── __init__.py           # Module exports
├── dashboard.py          # Web dashboard with WebSocket
├── alert_manager.py      # Multi-channel alerts
└── performance_analytics.py  # Performance metrics

Integration Points:
├── src/core/state.py     # State tracking
├── src/core/notify.py    # Telegram integration
└── scripts/live_trading_launcher.py  # HealthMonitor
```

## Troubleshooting

### Dashboard not accessible

1. Check if port 8080 is available
2. Verify dashboard is started: `await dashboard.start()`
3. Check firewall settings

### Alerts not sending

1. Verify channel is enabled in config
2. Check API credentials (Telegram token, Discord webhook)
3. Review logs for error messages
4. Test with a simple alert at CRITICAL priority

### Performance metrics incorrect

1. Ensure trade data includes required fields (`pnl`, `timestamp`)
2. Check data types (numeric values for calculations)
3. Verify equity curve is properly formatted as pandas Series

## Future Enhancements

- [ ] Email notification support
- [ ] Custom alert templates
- [ ] Historical metrics visualization
- [ ] Alert acknowledgment system
- [ ] Mobile app integration
- [ ] Prometheus/Grafana integration
- [ ] Advanced anomaly detection
- [ ] Multi-language support

## Support

For issues or questions:
1. Check logs in `live_trading_*.log`
2. Run tests to verify installation
3. Review this documentation
4. Open an issue on GitHub
