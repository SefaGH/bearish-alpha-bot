# Live Trading Launcher

Comprehensive production-ready launcher script for the Bearish Alpha Bot live trading system.

## Overview

The Live Trading Launcher (`live_trading_launcher.py`) is a complete integration of all Phase 1-4 components, designed for live trading on BingX with real USDT capital.

### Key Features

- **Full System Integration**: Combines all phases of the trading system
  - Phase 1: Multi-exchange framework (BingX focus)
  - Phase 2: Adaptive strategies with live signals
  - Phase 3: Portfolio management, risk engine, live execution
  - Phase 4: Complete AI enhancement (regime, adaptive learning, optimization, price prediction)

- **Production-Ready**: Enterprise-grade error handling, logging, and monitoring
- **Safety First**: Comprehensive pre-flight checks and emergency shutdown protocols
- **Live USDT Trading**: Designed for real money trading with proper risk management

## Configuration

### Trading Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Capital | 100 USDT | Real trading capital |
| Exchange | BingX | Single exchange focus |
| Trading Pairs | 8 pairs | Diversified crypto portfolio |
| Max Position Size | 15% | Maximum position relative to capital |
| Stop Loss | 5% | Stop loss percentage |
| Take Profit | 10% | Take profit percentage |
| Max Drawdown | 15% | Maximum portfolio drawdown |
| Risk Per Trade | 2% | Portfolio risk per trade |

### Trading Pairs

The launcher trades 8 diversified cryptocurrency pairs:

1. BTC/USDT:USDT (Bitcoin)
2. ETH/USDT:USDT (Ethereum)
3. SOL/USDT:USDT (Solana)
4. BNB/USDT:USDT (Binance Coin)
5. ADA/USDT:USDT (Cardano)
6. DOT/USDT:USDT (Polkadot)
7. LTC/USDT:USDT (Litecoin)
8. AVAX/USDT:USDT (Avalanche)

## Prerequisites

### Required Environment Variables

```bash
# BingX API credentials (required)
BINGX_KEY=your_api_key
BINGX_SECRET=your_api_secret

# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### Dependencies

All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Start live trading on BingX
python scripts/live_trading_launcher.py

# Run in paper trading mode (recommended for testing)
python scripts/live_trading_launcher.py --paper

# Run for 1 hour
python scripts/live_trading_launcher.py --paper --duration 3600

# Perform pre-flight checks only (dry run)
python scripts/live_trading_launcher.py --dry-run
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--paper` | Run in paper trading mode (simulated) | live mode |
| `--duration SECONDS` | Trading duration in seconds | indefinite |
| `--dry-run` | Perform checks only, no trading | false |
| `-h, --help` | Show help message | - |

## Initialization Phases

The launcher initializes the system in 8 sequential phases:

### Phase 1: Environment Configuration
- Loads and validates environment variables
- Checks for required BingX credentials
- Initializes optional Telegram notifications

### Phase 2: Exchange Connection
- Establishes connection to BingX
- Verifies API credentials
- Confirms all 8 trading pairs are available

### Phase 3: Risk Management
- Configures risk limits (15% max position, 5% stop loss, 10% take profit)
- Sets up portfolio drawdown protection (15% max)
- Initializes risk monitoring systems

### Phase 4: AI Components (Phase 4 Integration)
- **ML Regime Prediction**: Market regime forecasting
- **Adaptive Learning**: Real-time strategy adaptation
- **Strategy Optimization**: Multi-objective optimization
- **Price Prediction**: LSTM/Transformer price forecasting

### Phase 5: Trading Strategies
- Initializes Adaptive Oversold Bounce strategy
- Initializes Adaptive Short The Rip strategy
- Configures strategy parameters

### Phase 6: Production System
- Initializes ProductionCoordinator (Phase 3.4)
- Sets up portfolio manager with 100 USDT capital
- Configures live trading engine
- Establishes websocket connections
- Activates circuit breaker system

### Phase 7: Strategy Registration
- Registers strategies with portfolio manager
- Allocates capital across strategies (equal allocation)
- Validates strategy configurations

### Phase 8: Pre-Flight Checks
- ✓ Exchange connectivity test
- ✓ System state validation
- ✓ Risk limits verification
- ✓ Strategy registration confirmation
- ✓ Emergency protocols check

## Safety Features

### Pre-Flight Checks

Before trading begins, the launcher performs comprehensive checks:

1. **Exchange Connectivity**: Verifies BingX API connection
2. **System State**: Confirms all components initialized
3. **Risk Limits**: Validates risk management configuration
4. **Strategy Registration**: Ensures strategies are properly loaded
5. **Emergency Protocols**: Checks circuit breaker activation

### Emergency Shutdown

The system includes multiple emergency shutdown triggers:

- Manual interruption (Ctrl+C)
- Exchange disconnection
- Risk limit breach
- System error
- Max daily loss reached
- Circuit breaker activation

Emergency shutdown process:
1. Stop new signal processing
2. Cancel pending orders
3. Close positions (configurable method)
4. Disconnect websockets
5. Save system state
6. Generate emergency report
7. Send Telegram alert (if configured)

### Risk Management

Built-in risk controls:

- **Position Sizing**: Maximum 15% of capital per position
- **Stop Loss**: Automatic 5% stop loss on all positions
- **Take Profit**: 10% take profit targets
- **Drawdown Protection**: System pauses at 15% drawdown
- **Correlation Limits**: Maximum 70% correlation between positions
- **Daily Loss Limit**: 5% maximum daily loss

## Monitoring

### Real-Time Logging

The launcher creates timestamped log files:

```
live_trading_YYYYMMDD_HHMMSS.log
```

Logs include:
- System initialization steps
- Trading signals and executions
- Position management events
- Risk alerts and warnings
- Performance metrics
- Error messages and stack traces

### Telegram Notifications

When configured, the system sends Telegram alerts for:

- Trading system startup
- Position entries and exits
- Risk limit warnings
- Emergency shutdowns
- Daily performance summaries

### Performance Metrics

Real-time tracking of:
- Portfolio value and P&L
- Active positions count
- Win rate and profit factor
- Sharpe ratio and max drawdown
- Strategy performance breakdown
- Execution quality metrics

## Integration Architecture

### Phase 1: Multi-Exchange Framework
- `CcxtClient`: BingX connection with USDT support
- Server time synchronization
- Dynamic contract discovery
- Bulk OHLCV fetching

### Phase 2: Market Intelligence
- `MarketRegimeAnalyzer`: Real-time regime detection
- Adaptive strategies with regime awareness
- Dynamic parameter adjustment

### Phase 3: Portfolio & Risk Management
- `ProductionCoordinator`: Central orchestration
- `PortfolioManager`: Multi-strategy allocation
- `RiskManager`: Comprehensive risk controls
- `LiveTradingEngine`: Order execution
- `WebSocketManager`: Real-time data streaming
- `CircuitBreakerSystem`: Emergency protection

### Phase 4: AI Enhancement
- `MLRegimePredictor`: ML-based regime forecasting
- `AdvancedPricePredictionEngine`: LSTM/Transformer predictions
- `AIEnhancedStrategyAdapter`: Strategy signal enhancement
- `StrategyOptimizer`: Genetic algorithm optimization

## Examples

### Paper Trading Test

```bash
# Safe testing mode
python scripts/live_trading_launcher.py --paper --duration 3600
```

### Dry Run Validation

```bash
# Check system without trading
python scripts/live_trading_launcher.py --dry-run
```

### Production Live Trading

```bash
# Full live trading (ensure API credentials are correct!)
python scripts/live_trading_launcher.py
```

## Troubleshooting

### Common Issues

**Issue**: Missing environment variables
```
❌ Missing required environment variables: ['BINGX_KEY', 'BINGX_SECRET']
```
**Solution**: Set up `.env` file or export variables

**Issue**: Exchange connection failed
```
❌ Failed to connect to BingX: Authentication failed
```
**Solution**: Verify API credentials are correct and have proper permissions

**Issue**: Pre-flight checks failed
```
❌ Pre-flight checks failed - aborting launch
```
**Solution**: Review logs for specific check failures, resolve issues

**Issue**: Trading pair not available
```
⚠ Some trading pairs not available: ['SYMBOL/USDT:USDT']
```
**Solution**: Verify BingX supports the trading pair and it's in the correct format

### Debug Mode

For detailed debugging, set logging level:

```bash
export LOG_LEVEL=DEBUG
python scripts/live_trading_launcher.py --dry-run
```

## Testing

Run the test suite:

```bash
# Run launcher tests
pytest tests/test_live_trading_launcher.py -v

# Run all tests
pytest tests/ -v
```

## Best Practices

1. **Always Start with Dry Run**: Test configuration before live trading
   ```bash
   python scripts/live_trading_launcher.py --dry-run
   ```

2. **Use Paper Mode First**: Validate strategies in simulation
   ```bash
   python scripts/live_trading_launcher.py --paper
   ```

3. **Monitor Logs**: Keep logs directory clean and review regularly

4. **Set Up Telegram**: Configure alerts for important events

5. **Small Capital First**: Start with minimal capital to test

6. **Regular Backups**: Save state files regularly

## Support

For issues or questions:
- Review logs in `live_trading_*.log` files
- Check pre-flight check output
- Verify environment variables
- Consult main README.md
- Review Phase 3.4 and Phase 4 documentation

## License

See main repository LICENSE file.

## Disclaimer

Trading cryptocurrencies involves significant risk. This software is provided as-is without warranties. Always test thoroughly in paper trading mode before using real capital. The authors are not responsible for any financial losses.
