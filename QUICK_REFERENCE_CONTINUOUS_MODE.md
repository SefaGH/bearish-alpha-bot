# Ultimate Continuous Mode - Quick Reference

## 🚀 Quick Start

### Production (24/7 Trading)
```bash
python scripts/live_trading_launcher.py --infinite --auto-restart
```

### Testing (Paper Trading)
```bash
python scripts/live_trading_launcher.py --paper --infinite --auto-restart
```

## 📋 Command Reference

| Command | Description |
|---------|-------------|
| `--infinite` | Layer 1: Never stops, auto-recovers |
| `--auto-restart` | Layer 2: External monitoring, auto-restart |
| `--max-restarts N` | Maximum restart attempts (default: 1000) |
| `--restart-delay N` | Base restart delay in seconds (default: 30) |
| `--paper` | Paper trading mode |
| `--duration N` | Duration per cycle in seconds |
| `--dry-run` | Pre-flight checks only |

## 🛡️ Three-Layer Defense

### Layer 1: TRUE CONTINUOUS MODE (`--infinite`)
- ✅ Infinite loop, never stops on duration
- ✅ Auto-recovery from API/network errors
- ✅ Bypasses non-critical circuit breakers
- ✅ Restarts trading engine on failure

### Layer 2: AUTO-RESTART FAILSAFE (`--auto-restart`)
- ✅ External process monitoring
- ✅ Smart restart with exponential backoff
- ✅ State preservation across restarts
- ✅ Max restart limit (default: 1000)

### Layer 3: HEALTH MONITORING (Auto-enabled)
- ✅ Heartbeat every 5 minutes
- ✅ Telegram notifications (if configured)
- ✅ Performance metrics tracking
- ✅ Health status monitoring

## 🔧 Configuration

### Required Environment Variables
```bash
BINGX_KEY=your_api_key
BINGX_SECRET=your_api_secret
```

### Optional (Recommended)
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 📊 Monitoring

### Telegram Notifications
- 🚀 Bot start
- 🔄 Restart attempts
- 💓 Hourly health checks
- 🛑 Shutdown events

### Log Files
```bash
# View live logs
tail -f live_trading_*.log

# Search for errors
grep ERROR live_trading_*.log

# Check restarts
grep "RESTART" live_trading_*.log
```

## ⚡ Emergency Actions

### Manual Stop
```
Press Ctrl+C
```
*Bot will shutdown gracefully, no restart*

### Check Status
Look for these in logs:
- `ULTIMATE CONTINUOUS MODE: AUTO-RESTART WRAPPER ACTIVE`
- `Health Monitor initialized`
- `Restart N/1000`

## 📈 Restart Timing (Exponential Backoff)

| Failure # | Delay | Total Time |
|-----------|-------|------------|
| 1 | 30s | 30s |
| 2 | 1m | 1m 30s |
| 3 | 2m | 3m 30s |
| 4 | 4m | 7m 30s |
| 5 | 8m | 15m 30s |
| 6+ | 16m-1h | Varies |

## 🚨 Safety Features

✅ **Manual Override**: Ctrl+C always stops the bot  
✅ **Max Restarts**: Prevents infinite loops (1000 limit)  
✅ **Consecutive Failures**: Stops after 10 consecutive failures  
✅ **State Preservation**: Positions maintained across restarts  
✅ **Telegram Alerts**: Real-time notifications  

## 📖 Examples

### Example 1: Maximum Resilience
```bash
python scripts/live_trading_launcher.py \
    --infinite \
    --auto-restart
```

### Example 2: Custom Settings
```bash
python scripts/live_trading_launcher.py \
    --infinite \
    --auto-restart \
    --max-restarts 500 \
    --restart-delay 60
```

### Example 3: Timed Cycles
```bash
# 1-hour cycles, auto-restart for 24 hours
python scripts/live_trading_launcher.py \
    --duration 3600 \
    --auto-restart \
    --max-restarts 24
```

### Example 4: Paper Trading Test
```bash
python scripts/live_trading_launcher.py \
    --paper \
    --infinite \
    --auto-restart \
    --max-restarts 10
```

## 🔍 Troubleshooting

### High Restart Count
**Problem**: Bot restarting frequently  
**Check**: 
- API credentials
- Network connectivity
- Exchange status
- Log files for errors

### No Telegram Notifications
**Problem**: Not receiving alerts  
**Check**:
- `TELEGRAM_BOT_TOKEN` set
- `TELEGRAM_CHAT_ID` set
- Bot token valid
- Chat ID correct

### Bot Stopped
**Problem**: "Max restarts reached"  
**Solution**:
1. Check logs for root cause
2. Fix underlying issue
3. Reset and restart

## 📚 Full Documentation

For complete details, see:
- **ULTIMATE_CONTINUOUS_MODE.md** - Full guide
- **examples/ultimate_mode_demo.sh** - Usage examples

## ⚙️ Technical Details

### Files Modified
- `src/core/production_coordinator.py` - Layer 1 implementation
- `scripts/live_trading_launcher.py` - Layers 2 & 3 implementation

### Classes Added
- `HealthMonitor` - Health monitoring and metrics
- `AutoRestartManager` - Restart logic and tracking

### Key Functions
- `run_production_loop(continuous=True)` - Layer 1
- `_run_with_auto_restart()` - Layer 2
- `start_monitoring()` - Layer 3

---

**Version**: 1.0  
**Last Updated**: 2025-10-14  
**Status**: ✅ Production Ready
