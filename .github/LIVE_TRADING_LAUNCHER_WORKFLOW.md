# Live Trading Launcher Workflow

**Enterprise-Grade 24/7 Trading Capabilities with Ultimate Continuous Mode**

## Overview

The Live Trading Launcher workflow provides a comprehensive GitHub Actions-based solution for running the Bearish Alpha Bot with full integration of:
- **PR #49**: Live Trading Launcher with 8-phase initialization
- **PR #50**: Ultimate Continuous Trading Mode with 3-layer defense system

## Features

### 🚀 Core Capabilities

- **Manual Dispatch**: Trigger trading sessions on-demand with full control
- **Dual Trading Modes**: Paper trading (safe testing) and Live trading (real money)
- **Pre-Flight Checks**: Comprehensive system validation before trading
- **BingX USDT Trading**: 100 USDT capital with 8 diversified trading pairs
- **Ultimate Continuous Mode**: Three-layer defense for uninterrupted operations
- **Safety Features**: Confirmation requirements, timeouts, and emergency protocols

### 🛡️ Three-Layer Defense System

#### Layer 1: TRUE CONTINUOUS MODE (`--infinite`)
- Never-ending loop that ignores duration limits
- Smart circuit breaker (only critical issues stop the bot)
- Auto-recovery from API/network errors
- Trading engine auto-restart capability
- Manual override always available (Ctrl+C)

#### Layer 2: AUTO-RESTART FAILSAFE (`--auto-restart`)
- External process monitoring wrapper
- Exponential backoff: 30s → 60s → 120s → 240s → ... → 3600s max
- Maximum restart limit (default: 1000 attempts)
- Consecutive failure protection (stops at 10)
- State preservation across restarts
- Real-time Telegram alerts

#### Layer 3: HEALTH MONITORING (Automatic)
- Heartbeat monitoring (every 5 minutes)
- Performance metrics tracking
- Health status management (healthy → degraded → critical)
- Error recording and analysis
- Hourly Telegram updates
- Comprehensive health reporting

## Configuration

### Trading Configuration

```yaml
Capital: 100 USDT (Real trading capital)
Exchange: BingX (Single exchange focus)
Trading Pairs: 8 (BTC, ETH, SOL, BNB, ADA, DOT, LTC, AVAX)
Max Position Size: 15%
Stop Loss: 5%
Take Profit: 10%
Max Drawdown: 15%
Risk Per Trade: 2%
```

### Environment Variables

Required secrets to configure in GitHub repository settings:

```bash
BINGX_KEY          # BingX API key
BINGX_SECRET       # BingX API secret
```

Optional (for notifications):
```bash
TELEGRAM_BOT_TOKEN # Telegram bot token
TELEGRAM_CHAT_ID   # Telegram chat ID
```

## Usage

### Access the Workflow

1. Navigate to **Actions** tab in your GitHub repository
2. Select **"Live Trading Launcher (Ultimate Continuous Mode)"**
3. Click **"Run workflow"**

### Input Parameters

#### Mode
- **Type**: Choice (paper/live)
- **Default**: paper
- **Description**: Choose trading mode - paper for testing, live for real trading

#### Confirm Live Trading
- **Type**: String
- **Required for live mode**: Must type "YES"
- **Description**: Safety confirmation for live trading

#### Duration
- **Type**: String (seconds)
- **Default**: Empty (indefinite)
- **Description**: Trading session duration in seconds

#### Dry Run
- **Type**: Boolean
- **Default**: false
- **Description**: Perform pre-flight checks only without starting trading

#### Infinite Mode (Layer 1)
- **Type**: Boolean
- **Default**: false
- **Description**: Enable TRUE CONTINUOUS MODE - never stops, auto-recovers from errors

#### Auto-Restart (Layer 2)
- **Type**: Boolean
- **Default**: false
- **Description**: Enable AUTO-RESTART FAILSAFE - external monitoring and restart

#### Max Restarts
- **Type**: String
- **Default**: 1000
- **Description**: Maximum restart attempts when auto-restart is enabled

#### Restart Delay
- **Type**: String
- **Default**: 30
- **Description**: Base delay between restarts in seconds (uses exponential backoff)

## Usage Examples

### Paper Trading (Safe Testing)

```yaml
Mode: paper
Dry Run: false
Infinite: false
Auto-Restart: false
Duration: 3600  # 1 hour
```

**Command executed:**
```bash
python scripts/live_trading_launcher.py --paper --duration 3600
```

### Live Trading with Duration

```yaml
Mode: live
Confirm Live: YES
Dry Run: false
Infinite: false
Auto-Restart: false
Duration: 7200  # 2 hours
```

**Command executed:**
```bash
python scripts/live_trading_launcher.py --duration 7200
```

### ULTIMATE MODE: Maximum Resilience

```yaml
Mode: live
Confirm Live: YES
Dry Run: false
Infinite: true
Auto-Restart: true
Max Restarts: 1000
Restart Delay: 30
Duration: (empty - indefinite)
```

**Command executed:**
```bash
python scripts/live_trading_launcher.py --infinite --auto-restart --max-restarts 1000 --restart-delay 30
```

### Pre-Flight Checks Only

```yaml
Mode: paper
Dry Run: true
Infinite: false
Auto-Restart: false
```

**Command executed:**
```bash
python scripts/live_trading_launcher.py --paper --dry-run
```

## Workflow Jobs

### 1. Validate Inputs
- Validates live trading confirmation
- Ensures "YES" is typed for live mode
- Outputs `should_proceed` flag

### 2. Pre-Flight Checks
- Installs dependencies
- Runs dry-run validation
- Tests exchange connectivity
- Verifies system state
- Uploads pre-flight logs

### 3. Live Trading
- Displays configuration
- Dynamically builds command with selected flags
- Executes live trading launcher
- Monitors for 12 hours maximum (safety timeout)
- Generates post-session summary
- Uploads multiple artifact types:
  - Trading logs with timestamps
  - Trading data and state files
  - Health monitoring reports
- Sends Telegram notifications on success/failure

### 4. Summary
- Generates comprehensive session summary
- Always runs (even on failure)
- Displays configuration and job statuses
- Lists available artifacts

## Safety Features

### 🔒 Confirmation Requirement
- Live trading requires typing "YES" in the confirm_live field
- Prevents accidental live trading execution
- Validation occurs before any trading starts

### ⏱️ Timeout Protection
- Maximum execution time: 12 hours (720 minutes)
- Prevents infinite runs from consuming resources
- Automatic termination after timeout

### 🔍 Pre-Flight Checks
- Exchange connectivity test
- System state verification
- Strategy registration validation
- Risk management checks
- Emergency shutdown protocols

### 💾 State Preservation
- Trading state saved continuously
- Recovery from unexpected terminations
- Historical data maintained
- Position tracking across restarts

### 🛑 Emergency Stop
- Manual override always available (workflow cancel)
- Graceful shutdown procedures
- State saved before termination
- Comprehensive exit logs

## Monitoring & Artifacts

### Logs
- **Trading Logs**: Timestamped logs of all trading activities
- **Pre-Flight Logs**: Validation and check results
- **Restart Logs**: Restart attempts and reasons
- **Circuit Breaker Logs**: Safety trigger events

### Data
- **Trading Data**: Order history, fills, positions
- **State Files**: Current bot state (state.json)
- **Statistics**: Daily performance metrics (day_stats.json)

### Health Reports
- **Health Status**: System health snapshots
- **Performance Metrics**: Loop counts, errors, signals
- **Failure Analysis**: Error patterns and frequencies

### Retention
- **Trading Logs & Data**: 30 days
- **Health Reports**: 7 days

## Telegram Notifications

### Notification Types

#### Session Started
Sent when trading session begins:
- Mode (paper/live)
- Configuration summary
- Expected duration

#### Heartbeat Updates (Hourly)
Sent every hour during trading:
- Health status
- Performance metrics
- Loops completed
- Errors caught

#### Success Notification
Sent on successful completion:
- Final statistics
- Session duration
- Artifacts available

#### Failure Notification
Sent on session failure:
- Error details
- Failure reason
- Log location

#### Restart Notifications
Sent on each restart attempt:
- Restart count
- Wait time (exponential backoff)
- Reason for restart

## Best Practices

### For Paper Trading
1. Always start with paper trading to test strategies
2. Use shorter durations (1-2 hours) for initial tests
3. Review logs and metrics before going live
4. Test both normal and failure scenarios

### For Live Trading
1. **ALWAYS** type "YES" carefully for confirmation
2. Start with small capital (stick to 100 USDT)
3. Monitor Telegram notifications closely
4. Keep max restarts reasonable (100-500 for testing)
5. Use infinite mode only when confident

### For Ultimate Mode
1. Enable Telegram notifications (highly recommended)
2. Set reasonable max_restarts (1000 is safe)
3. Monitor restart frequency - frequent restarts indicate issues
4. Review health status regularly
5. Have manual intervention plan ready

### Monitoring Guidelines
- ✅ Check logs within first 10 minutes of starting
- ✅ Monitor Telegram hourly updates
- ✅ Review health status if status becomes "degraded"
- ✅ Investigate if restart count exceeds 100 in short time
- ✅ Cancel workflow if consecutive failures exceed 5

## Troubleshooting

### Workflow Fails at Validation
**Cause**: Live trading without "YES" confirmation
**Solution**: Type "YES" in confirm_live field

### Pre-Flight Checks Fail
**Cause**: Missing credentials or exchange connection issues
**Solution**: Verify BINGX_KEY and BINGX_SECRET are set in secrets

### Trading Session Stops Early
**Cause**: Circuit breaker triggered or critical error
**Solution**: Review logs to identify issue, fix, and retry

### Too Many Restarts
**Cause**: Persistent error or configuration issue
**Solution**: Review restart logs, fix root cause, reduce max_restarts

### Timeout Reached
**Cause**: Session exceeded 12 hours
**Solution**: Normal for long sessions, check final state and restart if needed

## Security Considerations

- ✅ API keys stored as GitHub secrets (encrypted)
- ✅ Never log sensitive credentials
- ✅ Confirmation required for live trading
- ✅ Timeout limits prevent runaway processes
- ✅ State files do not contain credentials

## Related Documentation

- [Live Trading Launcher](../LIVE_TRADING_LAUNCHER_SUMMARY.md)
- [Ultimate Continuous Mode](../ULTIMATE_CONTINUOUS_MODE.md)
- [Implementation Complete](../IMPLEMENTATION_COMPLETE.md)
- [BingX Integration](../BINGX_ULTIMATE_INTEGRATION.md)

## Support

For issues or questions:
1. Check workflow logs in the Actions tab
2. Review Telegram notifications for errors
3. Examine artifacts (logs, data, health reports)
4. Consult related documentation
5. Open an issue with relevant logs

---

**⚠️ Important**: This workflow enables autonomous trading. Always:
- Start with paper trading
- Use small capital amounts
- Monitor closely
- Have a stop plan
- Never risk more than you can afford to lose

**Version**: 1.0.0
**Last Updated**: 2025-10-14
**Status**: Production Ready ✅
