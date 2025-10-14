# Ultimate Continuous Trading Mode

**TRUE CONTINUOUS MODE + AUTO-RESTART FAILSAFE**

A multi-layer defense strategy for uninterrupted, autonomous trading operations with the Bearish Alpha Bot.

## Overview

The Ultimate Continuous Trading Mode implements a three-layer defense system to ensure the bot never stops trading and always recovers from any failure:

1. **Layer 1: TRUE CONTINUOUS MODE** (Primary Defense) - Internal resilience
2. **Layer 2: AUTO-RESTART FAILSAFE** (Backup Defense) - External monitoring and restart
3. **Layer 3: HEALTH MONITORING** (Guardian Layer) - Continuous health checks and alerts

## Features

### 🛡️ Multi-Layer Defense Strategy

#### Layer 1: TRUE CONTINUOUS MODE
- **Infinite Loop**: Bot runs indefinitely without time-based exits
- **Auto-Recovery**: Automatically recovers from API/network errors
- **Smart Circuit Breaker**: Bypasses non-critical circuit breaker triggers
- **Duration Bypass**: Ignores duration parameters when enabled
- **Intelligent Error Handling**: Restarts internal components on failure

#### Layer 2: AUTO-RESTART FAILSAFE
- **External Monitoring**: Monitors bot process from outside
- **Smart Restart Logic**: Intelligently decides when to restart
- **Exponential Backoff**: Delays between restarts increase exponentially (30s → 60s → 120s → ... → max 1 hour)
- **State Preservation**: Maintains trading state across restarts
- **Maximum Restart Limit**: Prevents infinite restart loops (default: 1000 attempts)
- **Failure Analysis**: Tracks consecutive failures and restart patterns

#### Layer 3: HEALTH MONITORING
- **Heartbeat System**: Regular health checks every 5 minutes
- **Performance Metrics**: Tracks loops, errors, and signals
- **Telegram Notifications**: Real-time alerts for restarts and health status
- **Status Tracking**: Monitors health status (healthy, degraded, critical)
- **Manual Override**: Ctrl+C always works for manual shutdown

## Usage

### Basic Commands

```bash
# Standard live trading (original behavior)
python scripts/live_trading_launcher.py

# Layer 1: TRUE CONTINUOUS MODE (never stops, auto-recovers)
python scripts/live_trading_launcher.py --infinite

# Layer 2: AUTO-RESTART FAILSAFE (external monitoring)
python scripts/live_trading_launcher.py --auto-restart

# ULTIMATE MODE: Both layers enabled (maximum resilience)
python scripts/live_trading_launcher.py --infinite --auto-restart
```

### Advanced Options

```bash
# Custom restart parameters
python scripts/live_trading_launcher.py \
    --infinite \
    --auto-restart \
    --max-restarts 500 \
    --restart-delay 60

# Paper trading with ultimate mode
python scripts/live_trading_launcher.py \
    --paper \
    --infinite \
    --auto-restart

# Time-limited ultimate mode (runs for duration, then restarts)
python scripts/live_trading_launcher.py \
    --infinite \
    --auto-restart \
    --duration 3600  # 1 hour per cycle
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--infinite` | flag | False | Enable TRUE CONTINUOUS mode (Layer 1) |
| `--auto-restart` | flag | False | Enable auto-restart failsafe (Layer 2) |
| `--max-restarts` | int | 1000 | Maximum restart attempts |
| `--restart-delay` | int | 30 | Base delay between restarts (seconds) |
| `--paper` | flag | False | Run in paper trading mode |
| `--duration` | float | None | Trading duration per cycle (seconds) |
| `--dry-run` | flag | False | Pre-flight checks only |

## How It Works

### Layer 1: Internal Resilience

The production coordinator's main loop has been enhanced with continuous mode:

```python
# In src/core/production_coordinator.py
await coordinator.run_production_loop(
    mode='live',
    duration=None,      # Ignored in continuous mode
    continuous=True     # Enable infinite loop
)
```

**Behavior:**
- ✅ Bypasses duration checks
- ✅ Only stops on critical circuit breakers
- ✅ Auto-recovers from API errors
- ✅ Restarts trading engine if it stops
- ✅ Continues after non-critical failures

### Layer 2: External Restart Wrapper

The auto-restart manager wraps the entire bot process:

```python
# In scripts/live_trading_launcher.py
while True:
    try:
        # Run bot
        exit_code = await run_once()
        
        # Check if should restart
        if should_restart():
            delay = calculate_exponential_backoff()
            await asyncio.sleep(delay)
            continue  # Restart
        else:
            break  # Max restarts reached
            
    except Exception as e:
        record_failure(e)
        # Continue to restart
```

**Behavior:**
- ✅ Monitors bot from outside
- ✅ Restarts on crashes
- ✅ Exponential backoff: 30s → 60s → 120s → 240s → ... → 3600s
- ✅ Tracks consecutive failures
- ✅ Stops after max restarts (default: 1000)

### Layer 3: Health Guardian

Continuous monitoring and alerting system:

```python
# Health monitoring runs in parallel
async def health_monitor():
    while monitoring:
        await asyncio.sleep(300)  # Every 5 minutes
        
        # Send heartbeat
        log_health_status()
        send_telegram_update()
        
        # Track metrics
        update_performance_metrics()
```

**Behavior:**
- ✅ Heartbeat every 5 minutes
- ✅ Telegram updates every hour
- ✅ Error tracking and health status
- ✅ Performance metrics logging

## Safety Features

### Emergency Stop

The bot always respects manual stop signals:

```python
# Ctrl+C or SIGINT
try:
    await run_bot()
except KeyboardInterrupt:
    logger.info("Manual stop requested")
    await graceful_shutdown()
    sys.exit(0)  # No restart
```

### Maximum Restart Limit

Prevents infinite restart loops:

```python
if restart_count >= max_restarts:
    logger.critical("Max restarts reached, manual intervention required")
    send_telegram_alert("Manual intervention required")
    sys.exit(1)
```

### Consecutive Failure Protection

Stops if too many failures occur in a row:

```python
if consecutive_failures > 10:
    logger.critical("Too many consecutive failures, stopping")
    send_telegram_alert("Manual intervention required")
    sys.exit(1)
```

## Monitoring & Notifications

### Telegram Alerts

The bot sends Telegram notifications for:
- 🚀 **Bot start**: Initial launch notification
- 🔄 **Restart triggered**: Each restart attempt with details
- 💓 **Heartbeat**: Hourly health status updates
- 🛑 **Shutdown**: Graceful or emergency shutdown
- ⚠️ **Warnings**: Circuit breaker trips, errors

Example notification:
```
🔄 AUTO-RESTART TRIGGERED
Attempt: 3/1000
Reason: Connection timeout
Consecutive Failures: 2
Uptime: 2.5h
Next restart in: 60s
```

### Health Reports

The health monitor tracks:
- **Status**: healthy, degraded, critical
- **Uptime**: Total time running
- **Loops**: Successful iterations
- **Errors**: Error count and last error
- **Heartbeat**: Last heartbeat timestamp

## State Preservation

The bot preserves state across restarts:

```python
# Before restart
save_state({
    'positions': open_positions,
    'performance': metrics,
    'timestamp': current_time
})

# After restart
state = load_state()
restore_positions(state['positions'])
```

**Preserved:**
- ✅ Open positions
- ✅ Trading state
- ✅ Performance metrics
- ✅ Configuration

## Performance Characteristics

### Restart Timing (Exponential Backoff)

| Failures | Delay | Cumulative Time |
|----------|-------|-----------------|
| 1 | 30s | 30s |
| 2 | 60s | 1m 30s |
| 3 | 2m | 3m 30s |
| 4 | 4m | 7m 30s |
| 5 | 8m | 15m 30s |
| 6 | 16m | 31m 30s |
| 7+ | 1h (max) | Varies |

### Resource Usage

- **Memory**: Minimal overhead (~10MB for monitoring)
- **CPU**: Negligible (monitoring runs every 5 minutes)
- **Network**: Only for health checks and Telegram

## Best Practices

### Recommended Configuration

For maximum reliability:

```bash
python scripts/live_trading_launcher.py \
    --infinite \
    --auto-restart \
    --max-restarts 1000 \
    --restart-delay 30
```

### Monitoring

1. **Enable Telegram notifications**: Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
2. **Check logs regularly**: Review `live_trading_*.log` files
3. **Monitor health status**: Watch for "degraded" or "critical" status
4. **Track restart count**: If restarts are frequent, investigate root cause

### Manual Intervention

Stop the bot manually when:
- ✋ Maximum restarts approaching (>900/1000)
- ✋ Consecutive failures increasing rapidly
- ✋ Health status is "critical" for extended period
- ✋ Market conditions require strategy adjustment

## Troubleshooting

### Bot Keeps Restarting

**Symptoms**: High restart count, increasing consecutive failures

**Causes**:
- API credentials invalid
- Network connectivity issues
- Exchange maintenance
- Insufficient funds

**Solution**:
1. Check logs for error details
2. Verify API credentials
3. Check exchange status
4. Monitor network connectivity

### No Telegram Notifications

**Symptoms**: Bot running but no Telegram messages

**Causes**:
- Missing `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID`
- Invalid bot token
- Chat ID incorrect

**Solution**:
1. Verify environment variables are set
2. Test Telegram bot separately
3. Check bot permissions

### Bot Stopped After Max Restarts

**Symptoms**: Bot exits with "Max restarts reached"

**Causes**:
- Persistent error preventing startup
- Configuration issue
- Exchange API problem

**Solution**:
1. Review logs to identify root cause
2. Fix underlying issue
3. Reset and restart bot

## Implementation Details

### Files Modified

1. **`src/core/production_coordinator.py`**
   - Added `continuous` parameter to `run_production_loop()`
   - Implemented intelligent circuit breaker bypass
   - Added auto-recovery for trading engine
   - Enhanced error handling

2. **`scripts/live_trading_launcher.py`**
   - Created `HealthMonitor` class
   - Created `AutoRestartManager` class
   - Added `--infinite` and `--auto-restart` flags
   - Implemented restart wrapper logic
   - Enhanced shutdown procedures

### Key Classes

**HealthMonitor**:
- Heartbeat monitoring
- Performance metrics tracking
- Health status management
- Telegram notifications

**AutoRestartManager**:
- Restart decision logic
- Exponential backoff calculation
- Failure tracking
- State management

## Testing

Comprehensive tests validate all layers:

```bash
# Run validation tests
python tests/test_continuous_mode_simple.py
```

**Tests include**:
- ✅ HealthMonitor initialization and error handling
- ✅ AutoRestartManager restart logic
- ✅ Exponential backoff calculation
- ✅ Failure tracking and limits
- ✅ Command-line argument parsing
- ✅ Integration scenarios

## Examples

### Example 1: 24/7 Live Trading

```bash
# Maximum resilience for production
python scripts/live_trading_launcher.py \
    --infinite \
    --auto-restart
```

### Example 2: Paper Trading Test

```bash
# Test ultimate mode in paper trading
python scripts/live_trading_launcher.py \
    --paper \
    --infinite \
    --auto-restart \
    --max-restarts 100
```

### Example 3: Limited Duration Cycles

```bash
# Run 1-hour cycles, auto-restart between cycles
python scripts/live_trading_launcher.py \
    --duration 3600 \
    --auto-restart \
    --max-restarts 24  # Run for ~24 hours
```

## Conclusion

The Ultimate Continuous Trading Mode provides enterprise-grade reliability for autonomous trading operations. With three layers of defense, exponential backoff, and comprehensive monitoring, the bot is designed to never stop trading and always recover from failures.

**Key Benefits**:
- 🔒 **Reliability**: Multi-layer defense ensures continuous operation
- 🔄 **Auto-Recovery**: Automatic recovery from all non-critical failures
- 📊 **Monitoring**: Real-time health tracking and alerts
- 🛡️ **Safety**: Manual override and maximum restart limits
- 📈 **Performance**: Minimal overhead, exponential backoff

**Next Steps**:
1. Test in paper trading mode first
2. Enable Telegram notifications
3. Monitor logs and health status
4. Gradually increase confidence
5. Deploy to production with ultimate mode

---

*For questions or issues, check the logs and health reports first. The system is designed to be self-healing and transparent about its status.*
