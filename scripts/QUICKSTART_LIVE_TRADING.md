# Live Trading Quick Start Guide

Get started with the Bearish Alpha Bot live trading launcher in 5 minutes.

## Step 1: Prerequisites

Ensure you have:
- ✅ Python 3.8+ installed
- ✅ BingX account with API credentials
- ✅ VST test tokens allocated (or real USDT for live trading)

## Step 2: Installation

```bash
# Clone repository
git clone https://github.com/SefaGH/bearish-alpha-bot.git
cd bearish-alpha-bot

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configuration

Create a `.env` file in the project root:

```bash
# Required: BingX credentials
BINGX_KEY=your_api_key_here
BINGX_SECRET=your_api_secret_here

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

**⚠️ Security Note**: Never commit your `.env` file to git!

## Step 4: Test Configuration

Run a dry-run to validate your setup:

```bash
python scripts/live_trading_launcher.py --dry-run
```

Expected output:
```
✓ BingX credentials found
✓ Connected to BingX - XXX markets available
✓ VST/USDT:USDT contract verified
✓ All 8 trading pairs verified
✓ Risk configuration loaded
✓ ALL PRE-FLIGHT CHECKS PASSED
```

## Step 5: Start Trading

### Option A: Paper Trading (Recommended First)

Safe simulation mode with no real trades:

```bash
python scripts/live_trading_launcher.py --paper
```

### Option B: Limited Time Paper Trading

Run for 1 hour to test:

```bash
python scripts/live_trading_launcher.py --paper --duration 3600
```

### Option C: Live Trading (Production)

⚠️ **Only after successful paper trading!**

```bash
python scripts/live_trading_launcher.py
```

## What to Expect

### Initialization (8 phases)

You'll see the launcher go through:

```
[1/8] Loading Environment Configuration...
✓ BingX credentials found

[2/8] Initializing BingX Exchange Connection...
✓ Connected to BingX - 500+ markets available
✓ VST/USDT:USDT contract verified
✓ All 8 trading pairs verified

[3/8] Initializing Risk Management System...
✓ Risk configuration loaded
  - Max position size: 15.0%
  - Stop loss: 5.0%
  - Take profit: 10.0%

[4/8] Initializing Phase 4 AI Components...
✓ ML Regime Predictor initialized
✓ Strategy Optimizer initialized
✓ Price Prediction Engine initialized
✓ AI-Enhanced Strategy Adapter initialized

[5/8] Initializing Trading Strategies...
✓ Adaptive Oversold Bounce strategy initialized
✓ Adaptive Short The Rip strategy initialized

[6/8] Initializing Production Trading System...
✓ Production system initialized successfully

[7/8] Registering Trading Strategies...
✓ adaptive_ob: 50.0% allocation
✓ adaptive_str: 50.0% allocation

[8/8] Performing Pre-Flight System Checks...
✓ ALL PRE-FLIGHT CHECKS PASSED

======================================================================
STARTING LIVE TRADING
======================================================================
```

### During Trading

The system will:
- 📊 Monitor 8 trading pairs in real-time
- 🤖 Generate AI-enhanced trading signals
- 💰 Execute trades with risk management
- 📈 Track performance and P&L
- 🔔 Send Telegram notifications (if configured)

### Stopping Trading

Press `Ctrl+C` for graceful shutdown:

```
⚠ Keyboard interrupt received - initiating shutdown...

======================================================================
INITIATING GRACEFUL SHUTDOWN
======================================================================
1. Stopping signal processing...
2. Closing active positions...
3. Disconnecting websockets...
✓ Production system stopped

======================================================================
SHUTDOWN COMPLETE
======================================================================
```

## Trading Parameters

The launcher uses these pre-configured settings:

| Parameter | Value |
|-----------|-------|
| Capital | 100 VST |
| Exchange | BingX |
| Trading Pairs | 8 (BTC, ETH, SOL, BNB, ADA, DOT, MATIC, AVAX) |
| Max Position | 15% of capital |
| Stop Loss | 5% |
| Take Profit | 10% |
| Max Drawdown | 15% |

## Monitoring

### Log Files

Check timestamped logs:
```bash
tail -f live_trading_YYYYMMDD_HHMMSS.log
```

### Telegram Alerts

If configured, you'll receive:
- 🚀 Trading started notification
- 📈 Position entry/exit alerts
- ⚠️ Risk warnings
- 🛑 Shutdown notifications

## Troubleshooting

### Problem: "Missing required environment variables"

**Solution**: Check your `.env` file has `BINGX_KEY` and `BINGX_SECRET`

```bash
# Load environment variables
source .env  # or use python-dotenv
```

### Problem: "Failed to connect to BingX"

**Solution**: 
1. Verify API credentials are correct
2. Check API key has trading permissions
3. Ensure IP whitelist includes your IP (if configured)

### Problem: "Pre-flight checks failed"

**Solution**: 
1. Review specific failed check in logs
2. Verify exchange connectivity
3. Check risk parameters are valid
4. Ensure strategies are properly initialized

### Problem: "VST contract not available"

**Solution**: 
1. Verify BingX account has VST trading enabled
2. Check symbol format is correct: `VST/USDT:USDT`
3. Try other pairs like `BTC/USDT:USDT`

## Best Practices

### 🎯 Start Small
Begin with paper trading mode for at least 24 hours

### 📊 Monitor Closely
Keep logs open and watch initial trades closely

### 🔔 Enable Alerts
Configure Telegram for important notifications

### 💰 Test Capital
Start with minimal capital (100 VST recommended)

### 📝 Review Logs
Regularly check log files for warnings/errors

### 🛑 Know Emergency Stop
Familiarize yourself with `Ctrl+C` shutdown

## Next Steps

After successful paper trading:

1. **Review Performance**
   ```bash
   # Check logs for win rate, P&L, Sharpe ratio
   grep "Performance" live_trading_*.log
   ```

2. **Adjust Parameters** (optional)
   - Modify risk parameters in launcher code
   - Adjust strategy allocations
   - Fine-tune AI components

3. **Scale Up Gradually**
   - Start with 100 VST
   - Monitor for 1-2 weeks
   - Gradually increase capital if performing well

4. **Continuous Monitoring**
   - Set up log rotation
   - Create performance dashboards
   - Track key metrics

## Support

- 📖 Full Documentation: `scripts/README_LIVE_TRADING_LAUNCHER.md`
- 🧪 Demo Script: `examples/live_trading_launcher_demo.py`
- 🧪 Run Tests: `pytest tests/test_live_trading_launcher.py -v`
- 📝 Review Logs: `live_trading_*.log`

## Safety Reminders

⚠️ **Important Safety Notes**:

1. Always test in paper mode first
2. Start with minimal capital
3. Monitor trades closely initially
4. Set up stop losses (automatic)
5. Enable Telegram alerts
6. Keep logs for analysis
7. Review performance regularly
8. Never commit API keys to git

## Example Session

```bash
# 1. Set up environment
export BINGX_KEY=your_key
export BINGX_SECRET=your_secret

# 2. Dry run validation
python scripts/live_trading_launcher.py --dry-run

# 3. Paper trade for 1 hour
python scripts/live_trading_launcher.py --paper --duration 3600

# 4. Review logs
tail -100 live_trading_*.log

# 5. If all good, start live (carefully!)
python scripts/live_trading_launcher.py
```

## License & Disclaimer

Trading cryptocurrencies involves significant risk. This software is provided as-is without warranties. Always test thoroughly before using real capital. The authors are not responsible for any financial losses.

Happy trading! 🚀
