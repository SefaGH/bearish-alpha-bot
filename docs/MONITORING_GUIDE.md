# 📊 GitHub Actions Live Trading Monitoring Guide

## Overview

This guide explains how to use the GitHub Actions-based monitoring system for live trading. The system provides automated monitoring, Telegram alerts, and HTML reports without requiring a local machine.

## Features

### 1. **Telegram Monitoring** 📱
- Automated reports every 30 minutes
- Real-time trading statistics
- P&L tracking
- Win rate calculations
- Daily performance metrics

### 2. **HTML Reports** 📈
- Visual dashboard with metrics
- Color-coded P&L display
- Recent trade history
- Downloadable from GitHub Actions artifacts

### 3. **State Persistence** 💾
- Trading state preserved between runs
- 7-day artifact retention for quick access
- Automatic state download/upload

## Setup

### 1. Configure Telegram Bot (Optional)

To receive Telegram notifications, add these secrets to your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add the following secrets:
   - `TELEGRAM_BOT_TOKEN`: Your Telegram bot token (from @BotFather)
   - `TELEGRAM_CHAT_ID`: Your Telegram chat ID

### 2. Enable Workflows

The monitoring workflow is automatically enabled. It runs:
- **Every 30 minutes** via cron schedule
- **On-demand** via manual trigger

## Usage

### Running Monitoring Reports

#### Automatic (Scheduled)
The monitoring report runs automatically every 30 minutes. No action needed.

#### Manual Trigger
1. Go to **Actions** → **Trading Monitoring Report**
2. Click **Run workflow**
3. Select branch and click **Run workflow**

### Viewing Reports

#### Telegram
If configured, you'll receive messages like:
```
📊 LIVE TRADING REPORT
    
🔸 Status: RUNNING
🔸 Total Trades: 15
🔸 Open Positions: 2
🔸 Total P&L: $45.50
🔸 Win Rate: 73.3%

📅 Daily Stats:
• Signals: 12
• P&L: $5.50

⏰ 2025-10-15 22:00:00 UTC
```

#### HTML Reports
1. Go to **Actions** → **Trading Monitoring Report**
2. Click on the most recent run
3. Scroll to **Artifacts**
4. Download `monitoring-report-[run-id]`
5. Open `report.html` in your browser

### Checking Trading State

1. Go to **Actions** → **Live Trading Launcher (Ultimate Continuous Mode)**
2. Click on the most recent run
3. Scroll to **Artifacts**
4. Download `live-trading-state`
5. Review `state.json` and `day_stats.json`

## File Structure

### Trading State Files

#### `data/state.json`
```json
{
  "open": {
    "BTC-USDT": {
      "symbol": "BTC-USDT",
      "side": "long",
      "entry_price": 42000.0,
      "amount": 0.001
    }
  },
  "closed": [
    {
      "symbol": "ETH-USDT",
      "side": "long",
      "entry_price": 2200.0,
      "exit_price": 2300.0,
      "pnl": 10.5,
      "timestamp": "2025-10-15T20:00:00Z",
      "status": "closed"
    }
  ]
}
```

#### `data/day_stats.json`
```json
{
  "pnl": 5.5,
  "signals": 12,
  "date": "2025-10-15"
}
```

### Report Files

#### `data/report.html`
Visual HTML dashboard with:
- Total P&L metric card (green/red)
- Total trades counter
- Open positions counter
- Win rate percentage
- Recent trades table

#### `data/stats.json`
```json
{
  "generated_at": "2025-10-15T22:00:00",
  "total_pnl": 5.5,
  "total_trades": 3,
  "open_positions": 1,
  "win_rate": 0.666
}
```

## Workflows

### Monitoring Report Workflow
**File**: `.github/workflows/monitoring_report.yml`

**Trigger**: 
- Schedule: Every 30 minutes (`*/30 * * * *`)
- Manual: workflow_dispatch

**Steps**:
1. Download latest trading state (if available)
2. Generate monitoring report
3. Send Telegram notification (if configured)
4. Generate HTML report
5. Upload reports as artifacts

### Live Trading Launcher Updates
**File**: `.github/workflows/live_trading_launcher.yml`

**Added**:
- State upload step after trading session
- Separate artifact for monitoring (7-day retention)
- Preserved existing trading data artifact (30-day retention)

## Scripts

### `scripts/telegram_monitor.py`
**Purpose**: Generate and send Telegram monitoring reports

**Usage**:
```bash
python scripts/telegram_monitor.py
```

**Environment Variables**:
- `TELEGRAM_BOT_TOKEN`: Bot token for sending messages
- `TELEGRAM_CHAT_ID`: Target chat ID

**Output**:
- Sends formatted message to Telegram
- Prints success/failure status

### `scripts/generate_html_report.py`
**Purpose**: Generate HTML and JSON reports from trading data

**Usage**:
```bash
python scripts/generate_html_report.py
```

**Output**:
- `data/report.html`: Visual dashboard
- `data/stats.json`: Machine-readable statistics

## Troubleshooting

### No State Data Available
**Cause**: No live trading session has run yet, or artifacts expired

**Solution**: 
- Run a live trading session first
- Check artifact retention (7 days for state)

### Telegram Not Working
**Cause**: Missing or incorrect credentials

**Solution**:
1. Verify `TELEGRAM_BOT_TOKEN` is set correctly
2. Verify `TELEGRAM_CHAT_ID` is your chat ID (not username)
3. Test bot with `/start` command in Telegram

### Reports Show Zero Data
**Cause**: No trading activity or state files missing

**Solution**:
- Verify live trading is running
- Check logs in live trading workflow
- Ensure state files are being generated

### Workflow Not Running on Schedule
**Cause**: GitHub Actions schedule limitations

**Note**: GitHub Actions scheduled workflows may experience delays during high-load periods

**Solution**:
- Use manual trigger if needed
- Check workflow run history for execution times

## Best Practices

1. **Monitor Regularly**: Check Telegram messages for trading updates
2. **Download Reports**: Keep historical HTML reports for analysis
3. **Check State Files**: Verify trading state periodically
4. **Set Alerts**: Configure Telegram for immediate notifications
5. **Archive Data**: Download artifacts before they expire (7-day retention)

## Advanced Usage

### Custom Monitoring Schedule
Edit `.github/workflows/monitoring_report.yml`:
```yaml
schedule:
  - cron: "*/15 * * * *"  # Every 15 minutes
  - cron: "0 */2 * * *"   # Every 2 hours
```

### Multiple Telegram Channels
Add multiple Telegram secrets:
- `TELEGRAM_BOT_TOKEN_ALERTS`
- `TELEGRAM_CHAT_ID_ALERTS`

Update `scripts/telegram_monitor.py` to send to both.

### Custom Report Format
Edit `scripts/generate_html_report.py`:
- Modify CSS styles
- Add/remove metric cards
- Customize table columns
- Add charts or graphs

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Open an issue in the repository

## Summary

| Feature | Location | Frequency |
|---------|----------|-----------|
| Telegram Alerts | Telegram App | Every 30 min |
| HTML Reports | GitHub Artifacts | On-demand |
| State Files | GitHub Artifacts | 7-day retention |
| Trading Logs | GitHub Artifacts | 30-day retention |

The monitoring system runs completely on GitHub Actions, requiring no local infrastructure. All reports and state are accessible through GitHub's interface or Telegram notifications.
