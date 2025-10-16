# Debug Logging Quick Reference

## 🚀 Quick Start

```bash
# Enable debug mode
python scripts/live_trading_launcher.py --debug --paper

# Or with environment variable
export LOG_LEVEL=DEBUG
python scripts/live_trading_launcher.py --paper
```

## 📋 Log Tags Cheat Sheet

| Tag | Meaning | Example |
|-----|---------|---------|
| `[UNIVERSE]` | Symbol filtering | `[UNIVERSE] ✅ BTC/USDT:USDT accepted` |
| `[PROCESSING]` | Symbol scan started | `[PROCESSING] Symbol: BTC/USDT:USDT` |
| `[DATA]` | Market data fetched | `[DATA] BTC/USDT:USDT: 30m=200 bars, last_close=67845.50` |
| `[INDICATORS]` | Technical indicators | `[INDICATORS] BTC/USDT:USDT: RSI=45.2, ATR=234.56` |
| `[STRATEGY-CHECK]` | Strategy evaluation | `[STRATEGY-CHECK] adaptive_ob for BTC/USDT:USDT` |
| `[SIGNAL]` | Signal generated ✅ | `[SIGNAL] BTC/USDT:USDT: {'side': 'buy', ...}` |
| `[NO-SIGNAL]` | No signal | `[NO-SIGNAL] BTC/USDT:USDT (adaptive_ob): RSI=55.3` |

## 🔍 Useful Grep Commands

```bash
# Show only signals
grep "\[SIGNAL\]" debug.log

# Show why no signals
grep "\[NO-SIGNAL\]" debug.log

# Show RSI values
grep "\[INDICATORS\].*RSI" debug.log

# Show one symbol's journey
grep "BTC/USDT:USDT" debug.log | grep -E "\[(PROCESSING|SIGNAL)\]"

# Show accepted symbols
grep "\[UNIVERSE\]" debug.log | grep "✅"

# Show rejected symbols
grep "\[UNIVERSE\]" debug.log | grep -v "✅"
```

## ⚙️ Test Configuration

File: `config/config.debug.yaml`

Key settings for easier signal generation:
- `min_quote_volume_usdt: 50000` (more symbols)
- `rsi_max: 50` (OversoldBounce - easier to trigger)
- `rsi_min: 50` (ShortTheRip - easier to trigger)
- `ignore_regime: true` (bypass regime filter)

## 🐛 Troubleshooting

### No signals generated?
1. Check universe: `grep "\[UNIVERSE\]" debug.log`
2. Check RSI values: `grep "\[INDICATORS\].*RSI" debug.log`
3. Check strategy calls: `grep "\[STRATEGY-CHECK\]" debug.log`

### Universe empty?
- Lower `min_quote_volume_usdt` to 0 or 50000
- Increase `max_symbols_per_exchange`

### RSI never in range?
- Use test config: `config/config.debug.yaml`
- Or wait for market conditions to change

## 📚 Documentation

- Full guide: `DEBUG_LOGGING_GUIDE.md`
- Demo script: `python scripts/demo_debug_logging.py`
- Tests: `pytest tests/test_debug_logging.py -v`

## 💡 Pro Tips

1. Always save logs: `python script.py 2>&1 | tee debug.log`
2. Use `--debug` flag when troubleshooting
3. Check `[INDICATORS]` to understand why no signals
4. Start with test config, tighten for production
5. Filter logs by tag for focused analysis
