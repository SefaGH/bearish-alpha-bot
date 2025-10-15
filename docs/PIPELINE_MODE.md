# Market Data Pipeline Mode

## Overview

The Market Data Pipeline mode is an optimized operation mode for Bearish Alpha Bot that provides:

- **Faster signal generation**: 30-second iterations (vs 30-minute traditional mode)
- **Data caching**: Market data is cached in memory, reducing API calls
- **Health monitoring**: Built-in health checks and status reporting
- **Automatic failover**: If one exchange fails, data is fetched from others
- **Memory management**: Circular buffers prevent memory overflow

## Usage

### Running Locally

```bash
# Pipeline mode (optimized, continuous)
python src/main.py --pipeline

# Traditional mode (one-shot)
python src/main.py

# Live trading mode (Phase 3.4)
python src/main.py --live
```

### Required Environment Variables

```bash
# Exchange configuration
EXCHANGES=binance,bingx,kucoin

# Exchange credentials (at least one required)
BINANCE_KEY=your_key
BINANCE_SECRET=your_secret

BINGX_KEY=your_key
BINGX_SECRET=your_secret

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### GitHub Actions

The pipeline mode can run automatically via GitHub Actions:

- **Workflow**: `.github/workflows/bot_pipeline.yml`
- **Schedule**: Every 15 minutes
- **Duration**: 30 minutes per run (with 35-minute workflow timeout)
- **Manual trigger**: Can be triggered via workflow_dispatch

To use:
1. Configure secrets in GitHub repository settings
2. Enable the workflow
3. Monitor via Actions tab

## Configuration

Edit `config/config.example.yaml` to customize:

```yaml
signals:
  oversold_bounce:
    enable: true
    ignore_regime: false
  short_the_rip:
    enable: true

indicators:
  rsi_period: 14
  ema_periods: [21, 50, 200]
```

## Features

### Data Pipeline

- **Symbols tracked**: BTC/USDT, ETH/USDT, SOL/USDT (configurable)
- **Timeframes**: 30m, 1h, 4h
- **Buffer limits**: 
  - 30m: 1000 candles
  - 1h: 500 candles
  - 4h: 200 candles
  - 1d: 100 candles

### Health Monitoring

The pipeline continuously monitors:
- API request success/failure rates
- Data freshness
- Memory usage
- Active data streams

Health status:
- **healthy**: Error rate < 20%
- **degraded**: Error rate 20-50%
- **critical**: Error rate > 50%

### Signal Generation

The pipeline checks for trading signals using:
1. **Regime filter**: Bearish market detection on 4h timeframe
2. **OversoldBounce**: Long signals on oversold bounces
3. **ShortTheRip**: Short signals on exhausted rallies

Signals are sent via Telegram (if configured).

## Testing

### Unit Tests

```bash
# Run pipeline tests
python -m pytest tests/test_market_data_pipeline.py -v

# Run integration tests
python -m pytest tests/test_pipeline_integration.py -v

# Run all tests
python -m pytest tests/ -v
```

### Integration Test

```bash
# Test pipeline with live data (requires credentials)
python scripts/test_pipeline_integration.py
```

Expected output:
```
============================================================
Testing Market Data Pipeline Integration
============================================================
Building exchange clients...
✓ Clients: ['binance', 'bingx']

Initializing pipeline...
✓ Pipeline initialized

Starting data feeds...
✓ Feeds started

Waiting for data (10 seconds)...

Fetching data from pipeline...
✓ 30m data: 250 candles
  Latest: 2025-10-15 18:30:00 - Close: $66789.50
✓ 1h data: 250 candles
  Latest: 2025-10-15 18:00:00 - Close: $66798.20

Health status:
  Overall: healthy
  Active feeds: 6
  Error rate: 0.0%
  Memory: 2.45 MB

✅ Pipeline integration test complete!
```

## Performance

### Traditional Mode (run_once)
- Frequency: Every 30 minutes (cron)
- API calls: ~60-100 per run
- Response time: 30-60 seconds
- Signals delay: Up to 30 minutes

### Pipeline Mode (run_with_pipeline)
- Frequency: Continuous, 30-second iterations
- API calls: ~10-20 per iteration (cached data)
- Response time: 1-3 seconds per iteration
- Signals delay: 30 seconds max

**Performance improvement**: 60x faster signal generation with 5x fewer API calls.

## Troubleshooting

### No signals generated
- Check regime filter: Market may not be bearish
- Verify data availability: Check logs for API errors
- Test strategies: Use backtesting to verify signal logic

### High error rate
- Check API credentials and limits
- Verify network connectivity
- Review exchange-specific issues in logs

### Memory usage high
- Buffer limits are enforced automatically
- Normal usage: 2-10 MB for 3 symbols
- If > 50 MB, check for leaks and restart

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    run_with_pipeline()                   │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         MarketDataPipeline                      │    │
│  │                                                 │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐    │    │
│  │  │ Binance  │  │  BingX   │  │  KuCoin  │    │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘    │    │
│  │       │             │              │           │    │
│  │       └─────────────┴──────────────┘           │    │
│  │                     │                          │    │
│  │              ┌──────▼───────┐                  │    │
│  │              │ Data Storage │                  │    │
│  │              │   (Memory)   │                  │    │
│  │              └──────┬───────┘                  │    │
│  └─────────────────────┼──────────────────────────┘    │
│                        │                               │
│         ┌──────────────▼───────────────┐              │
│         │   Strategy Execution          │              │
│         │  - OversoldBounce             │              │
│         │  - ShortTheRip                │              │
│         └──────────────┬────────────────┘              │
│                        │                               │
│         ┌──────────────▼───────────────┐              │
│         │    Signal Notification        │              │
│         │      (Telegram)               │              │
│         └───────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

## Future Enhancements

- [ ] WebSocket streaming for real-time data
- [ ] Trade execution integration
- [ ] Multi-strategy portfolio management
- [ ] Advanced ML-based regime detection
- [ ] Performance analytics dashboard

## References

- [Phase 2.1 Market Data Pipeline](../WEBSOCKET_PHASE_2_1_SUMMARY.md)
- [Main Bot Architecture](../README.md)
- [Multi-Exchange Integration](../MULTI_EXCHANGE_INTEGRATION_SUMMARY.md)
