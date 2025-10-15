# Market Data Pipeline Usage Guide

## Overview

The `MarketDataPipeline` class provides a robust foundation for multi-exchange data collection with automatic memory management, health monitoring, and error handling.

## Quick Start

```python
from core.multi_exchange import build_clients_from_env
from core.market_data_pipeline import MarketDataPipeline

# Initialize with exchanges
exchanges = build_clients_from_env()
pipeline = MarketDataPipeline(exchanges)

# Start data feeds
symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
timeframes = ['30m', '1h', '4h']
results = pipeline.start_feeds(symbols, timeframes)

# Get latest data (with indicators already added)
df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')

# Monitor health
health = pipeline.health_check()
print(f"Status: {health['status']}, Error rate: {health['error_rate']}%")

# Shutdown gracefully
pipeline.shutdown()
```

## Key Features

### 1. Multi-Exchange Support
- Automatically tries multiple exchanges with fallback
- Selects best data source based on freshness and completeness
- Handles exchange-specific failures gracefully

### 2. Memory Management
Circular buffers prevent memory overflow:
- **30m**: 1,000 candles (~21 days)
- **1h**: 500 candles (~21 days)
- **4h**: 200 candles (~33 days)
- **1d**: 100 candles (~3 months)

### 3. Health Monitoring
```python
status = pipeline.get_pipeline_status()
# Returns:
# {
#   'status': 'healthy',
#   'total_requests': 100,
#   'error_rate': 2.5,
#   'active_streams': 6,
#   'memory_estimate_mb': 1.2,
#   'data_freshness': {'fresh': 5, 'stale': 1, 'expired': 0},
#   'exchanges': {'kucoinfutures': {'symbols': 2, 'streams': 3}}
# }
```

### 4. Error Handling
- Automatic retry with exponential backoff (0.5s, 1.0s, 2.0s)
- Graceful degradation on exchange failures
- Comprehensive error logging with emoji prefixes

### 5. Indicator Integration
Data is automatically enriched with technical indicators:
- RSI (default: 14 period)
- ATR (default: 14 period)
- EMA 21, 50, 200

## Advanced Usage

### Specific Exchange Selection
```python
# Get data from specific exchange
df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='kucoinfutures')
```

### Custom Configuration
```python
config = {
    'indicators': {
        'rsi_period': 14,
        'ema_fast': 21,
        'ema_mid': 50,
        'ema_slow': 200
    }
}
pipeline = MarketDataPipeline(exchanges, config)
```

### Rate Limiting
Built-in rate limiting with 0.1s delay between symbol fetches to respect API limits.

## Integration Points

### With Existing Systems
- **CcxtClient**: Uses `ohlcv()` and `fetch_ohlcv_bulk()`
- **Indicators**: Uses `add_indicators()` from `core.indicators`
- **Symbol Validation**: Uses `validate_and_get_symbol()`
- **Authentication**: Works with BingX authenticator and standard exchanges

### For Phase 2.2 WebSocket
The pipeline is designed to be extended with WebSocket support:
```python
# Future: start_websocket_feeds() will update data_streams in real-time
# Current: start_feeds() provides REST API foundation
```

## Performance Considerations

### Memory Usage
Estimated memory per stream: ~200 bytes per candle
- 6 streams × 500 candles avg = ~0.6 MB
- Scales linearly with streams

### Request Efficiency
- Batch requests when possible
- Caches validated symbols
- Rate limiting prevents API throttling

## Error Messages

The pipeline uses emoji prefixes for quick visual scanning:
- 🔄 = Starting/Processing
- ✅ = Success
- ⚠️ = Warning/Retry
- ❌ = Error/Failure

## Best Practices

1. **Initialize once**: Create one pipeline instance per application
2. **Monitor health**: Check `health_check()` periodically
3. **Handle None**: Always check if `get_latest_ohlcv()` returns data
4. **Graceful shutdown**: Call `shutdown()` before exit
5. **Log review**: Monitor logs for patterns in failures

## Troubleshooting

### No data returned
```python
df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
if df is None:
    # Check if feeds were started
    # Check exchange connectivity
    status = pipeline.get_pipeline_status()
    print(f"Active streams: {status['active_streams']}")
```

### High error rate
```python
health = pipeline.health_check()
if health['error_rate'] > 20:
    # Check exchange API status
    # Verify credentials
    # Review error logs
```

### Memory concerns
```python
status = pipeline.get_pipeline_status()
print(f"Memory: {status['memory_estimate_mb']} MB")
# Adjust buffer limits if needed
```
