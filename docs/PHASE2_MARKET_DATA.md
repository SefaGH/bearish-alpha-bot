# Phase 2.1: Market Data Pipeline - Comprehensive Documentation

## Overview

Phase 2.1 introduces a robust **Market Data Pipeline** system that provides multi-exchange data collection, automatic memory management, health monitoring, and seamless integration with the existing Bearish Alpha Bot infrastructure.

### Key Components

1. **MarketDataPipeline** - Core data collection and management engine
2. **DataAggregator** - Multi-exchange data quality assessment and consensus generation
3. **Integration Layer** - Seamless connection with existing bot components

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Market Data Pipeline                     │
│                         (Phase 2.1)                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   BingX      │     │ KuCoin Fut   │     │   Binance    │
│  (CcxtClient)│     │ (CcxtClient) │     │ (CcxtClient) │
└──────────────┘     └──────────────┘     └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Data Aggregator   │
                    │  (Quality & Best)  │
                    └────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Indicators  │     │   Strategies │     │  Intelligence│
│  (RSI/EMA)   │     │   (OB/STR)   │     │  (Regime)    │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## Quick Start

### Basic Setup

```python
from core.multi_exchange import build_clients_from_env
from core.market_data_pipeline import MarketDataPipeline

# Initialize with exchanges from environment
clients = build_clients_from_env()
pipeline = MarketDataPipeline(clients)

# Start data feeds
symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
timeframes = ['30m', '1h']
result = pipeline.start_feeds(symbols, timeframes)

# Get latest data with indicators
df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')

# Clean shutdown
pipeline.shutdown()
```

### With Data Aggregator

```python
from core.data_aggregator import DataAggregator

# Create aggregator with pipeline
aggregator = DataAggregator(pipeline)

# Get best data source
best_exchange = aggregator.get_best_data_source('BTC/USDT:USDT', '30m')

# Get consensus data from multiple exchanges
consensus_df = aggregator.get_consensus_data('BTC/USDT:USDT', '30m', min_sources=2)
```

---

## Core Features

### 1. Multi-Exchange Support

The pipeline automatically manages data from multiple exchanges with intelligent fallback:

```python
# Exchanges are tried in order, with automatic failover
pipeline = MarketDataPipeline({
    'bingx': bingx_client,
    'kucoinfutures': kucoin_client,
    'binance': binance_client
})

# Fetches from first available exchange
data = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')

# Or specify a particular exchange
data = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m', exchange='bingx')
```

**Features:**
- Automatic exchange fallback on failures
- Exchange-specific error handling
- Symbol validation per exchange
- Concurrent multi-exchange data collection

### 2. Automatic Memory Management

Circular buffers prevent memory overflow with timeframe-specific limits:

| Timeframe | Buffer Limit | Approximate Coverage |
|-----------|--------------|---------------------|
| 30m       | 1,000 candles | ~21 days           |
| 1h        | 500 candles   | ~21 days           |
| 4h        | 200 candles   | ~33 days           |
| 1d        | 100 candles   | ~3 months          |

**Memory Estimation:**
- ~200 bytes per candle
- 6 streams × 500 candles = ~0.6 MB
- Scales linearly with number of streams

```python
# Buffer limits are automatically enforced
pipeline.start_feeds(['BTC/USDT:USDT'], ['30m', '1h', '4h'])

# Check memory usage
status = pipeline.get_pipeline_status()
print(f"Memory estimate: {status['memory_estimate_mb']:.2f} MB")
```

### 3. Integrated Indicator System

All data is automatically enriched with technical indicators:

**Default Indicators:**
- **RSI** (Relative Strength Index) - 14 period
- **ATR** (Average True Range) - 14 period
- **EMA21** (Exponential Moving Average) - 21 period
- **EMA50** (Exponential Moving Average) - 50 period
- **EMA200** (Exponential Moving Average) - 200 period

```python
# Data comes with indicators pre-calculated
df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')

# Access indicators directly
print(f"RSI: {df['rsi'].iloc[-1]:.2f}")
print(f"EMA21: ${df['ema21'].iloc[-1]:.2f}")
print(f"ATR: ${df['atr'].iloc[-1]:.2f}")
```

**Custom Indicator Configuration:**

```python
config = {
    'indicators': {
        'rsi_period': 14,
        'ema_fast': 21,
        'ema_mid': 50,
        'ema_slow': 200
    }
}

pipeline = MarketDataPipeline(clients, config=config)
```

### 4. Health Monitoring

Comprehensive health tracking and status reporting:

```python
# Quick health check
health = pipeline.health_check()
print(f"Status: {health['status']}")  # healthy, degraded, or critical
print(f"Error rate: {health['error_rate']:.2f}%")
print(f"Active streams: {health['active_streams']}")

# Detailed pipeline status
status = pipeline.get_pipeline_status()
print(f"Memory: {status['memory_estimate_mb']:.2f} MB")
print(f"Fresh streams: {status['data_freshness']['fresh']}")
print(f"Stale streams: {status['data_freshness']['stale']}")

# Per-exchange breakdown
for exchange, info in status['exchanges'].items():
    print(f"{exchange}: {info['streams']} streams, {len(info['symbols'])} symbols")
```

**Health Status Levels:**
- **healthy**: Error rate < 20%
- **degraded**: Error rate 20-50%
- **critical**: Error rate > 50%

### 5. Error Handling & Resilience

Built-in error handling with exponential backoff:

```python
# Automatic retry with exponential backoff
# Delays: 0.5s → 1.0s → 2.0s
# Maximum 3 retries per exchange

result = pipeline.start_feeds(['BTC/USDT:USDT'], ['30m'])

# Check for errors
print(f"Successful: {result['successful_fetches']}")
print(f"Failed: {result['failed_fetches']}")
print(f"Errors: {len(result['errors'])}")

# Pipeline continues operating despite individual failures
health = pipeline.health_check()
print(f"Status: {health['status']}")  # Pipeline stays operational
```

---

## Data Aggregator Features

### Quality Scoring System

The DataAggregator assigns quality scores to data from each exchange:

**Quality Metrics:**
- **Candle Count**: More data = higher score
- **Data Gaps**: Fewer gaps = higher score  
- **Freshness**: Recent data = higher score
- **OHLC Integrity**: Valid OHLC relationships = higher score

```python
aggregator = DataAggregator(pipeline)

# Aggregate with quality scoring
result = aggregator.aggregate_multi_exchange('BTC/USDT:USDT', '30m')

for exchange, data in result['sources'].items():
    print(f"{exchange}: Quality {data['quality_score']:.2f}, "
          f"Freshness: {data['freshness']}")
```

### Best Data Source Selection

Automatically select the most reliable exchange:

```python
# Get best exchange for symbol/timeframe
best = aggregator.get_best_data_source('BTC/USDT:USDT', '30m')
print(f"Best source: {best}")

# Optionally filter to specific exchanges
best = aggregator.get_best_data_source(
    'BTC/USDT:USDT', '30m',
    exchanges=['bingx', 'kucoinfutures']
)
```

### Consensus Data Generation

Create weighted consensus from multiple exchanges:

```python
# Generate consensus from 2+ sources
consensus = aggregator.get_consensus_data(
    'BTC/USDT:USDT', '30m',
    min_sources=2
)

if consensus is not None:
    # Weighted average of OHLCV data
    # Weight based on quality scores
    print(f"Consensus close: ${consensus['close'].iloc[-1]:.2f}")
```

---

## Integration with Existing Systems

### CcxtClient Integration

```python
from core.ccxt_client import CcxtClient

# CcxtClient handles all exchange communication
client = CcxtClient('bingx')

# Pipeline uses client's methods:
# - ohlcv() for standard fetches
# - fetch_ohlcv_bulk() for large datasets
# - validate_and_get_symbol() for symbol validation
```

### Indicator Integration

```python
from core.indicators import add_indicators

# Indicators are automatically added by pipeline
# But can also be used independently:

import pandas as pd
df = pd.DataFrame(...)  # Your OHLCV data
df = add_indicators(df, config={'rsi_period': 14})
```

### Strategy Integration

```python
from strategies.oversold_bounce import OversoldBounce
from strategies.short_the_rip import ShortTheRip

# Strategies work directly with pipeline data
pipeline.start_feeds(['BTC/USDT:USDT'], ['30m', '1h', '4h'])

df_30m = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
df_1h = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '1h')
df_4h = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '4h')

# Use with strategies
ob = OversoldBounce(config)
signals = ob.scan([df_30m])
```

### Market Intelligence Integration

```python
from core.market_regime import MarketRegimeAnalyzer

# Pipeline data feeds intelligence systems
analyzer = MarketRegimeAnalyzer()

df_30m = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
df_1h = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '1h')
df_4h = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '4h')

regime = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
```

---

## Advanced Usage

### Multi-Timeframe Analysis

```python
# Fetch multiple timeframes for comprehensive analysis
symbol = 'BTC/USDT:USDT'
timeframes = ['30m', '1h', '4h', '1d']

pipeline.start_feeds([symbol], timeframes)

# Analyze across timeframes
for tf in timeframes:
    data = pipeline.get_latest_ohlcv(symbol, tf)
    print(f"{tf}: RSI={data['rsi'].iloc[-1]:.2f}, "
          f"Close=${data['close'].iloc[-1]:.2f}")
```

### Custom Buffer Limits

```python
# Modify buffer limits if needed
pipeline.BUFFER_LIMITS['30m'] = 2000  # Increase to 2000 candles

# Affects memory usage accordingly
status = pipeline.get_pipeline_status()
print(f"Memory: {status['memory_estimate_mb']:.2f} MB")
```

### Rate Limiting

Built-in rate limiting respects exchange API limits:

```python
# Automatic 0.1s delay between symbol fetches
pipeline.start_feeds(['BTC/USDT:USDT', 'ETH/USDT:USDT'], ['30m'])
# Total time: ~0.2s for 2 symbols
```

### Data Freshness Tracking

```python
# Check data freshness
status = pipeline.get_pipeline_status()

print(f"Fresh data: {status['data_freshness']['fresh']}")
print(f"Stale data: {status['data_freshness']['stale']}")
print(f"Expired data: {status['data_freshness']['expired']}")

# Data is considered:
# - Fresh: < 1 hour old
# - Stale: 1-2 hours old
# - Expired: > 2 hours old
```

---

## Best Practices

### 1. Environment Setup

```bash
# Set up exchange credentials
export EXCHANGES='bingx,kucoinfutures,binance'
export BINGX_KEY='your_bingx_key'
export BINGX_SECRET='your_bingx_secret'
export KUCOIN_KEY='your_kucoin_key'
export KUCOIN_SECRET='your_kucoin_secret'
export KUCOIN_PASSWORD='your_kucoin_password'
```

### 2. Error Handling

```python
try:
    clients = build_clients_from_env()
    pipeline = MarketDataPipeline(clients)
    result = pipeline.start_feeds(symbols, timeframes)
    
    # Always check results
    if result['failed_fetches'] > 0:
        print(f"Warning: {result['failed_fetches']} fetches failed")
        
except Exception as e:
    print(f"Error initializing pipeline: {e}")
finally:
    if 'pipeline' in locals():
        pipeline.shutdown()
```

### 3. Resource Management

```python
# Always shutdown pipeline when done
pipeline = MarketDataPipeline(clients)
try:
    # Use pipeline
    pipeline.start_feeds(symbols, timeframes)
    data = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
finally:
    pipeline.shutdown()
```

### 4. Health Monitoring

```python
# Regular health checks in production
import time

while True:
    health = pipeline.health_check()
    
    if health['status'] == 'critical':
        print("⚠️ Pipeline in critical state!")
        # Take action: restart, alert, etc.
    
    time.sleep(60)  # Check every minute
```

### 5. Memory Management

```python
# Monitor memory usage
status = pipeline.get_pipeline_status()

if status['memory_estimate_mb'] > 100:  # Over 100 MB
    print("⚠️ High memory usage")
    # Consider: fewer symbols, shorter timeframes, or cleanup
```

---

## Troubleshooting

### Issue: No data returned

**Symptoms:**
```python
df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
print(df)  # None or empty
```

**Solutions:**
1. Check if feeds were started: `pipeline.start_feeds()`
2. Verify symbol format: Use `'BTC/USDT:USDT'` (not `'BTCUSDT'`)
3. Check exchange connectivity: `health = pipeline.health_check()`
4. Verify exchange credentials in environment

### Issue: High error rate

**Symptoms:**
```python
health = pipeline.health_check()
print(health['error_rate'])  # > 20%
```

**Solutions:**
1. Check exchange API status
2. Verify API credentials and permissions
3. Reduce request rate (fewer symbols)
4. Check network connectivity
5. Review error logs for specific failures

### Issue: Stale data

**Symptoms:**
```python
status = pipeline.get_pipeline_status()
print(status['data_freshness']['stale'])  # High number
```

**Solutions:**
1. Restart feeds: `pipeline.start_feeds()`
2. Check if exchanges are responding
3. Verify timeframe availability on exchange
4. Consider using different exchange

### Issue: Memory concerns

**Symptoms:**
```python
status = pipeline.get_pipeline_status()
print(status['memory_estimate_mb'])  # Very high
```

**Solutions:**
1. Reduce number of symbols
2. Use fewer timeframes
3. Adjust buffer limits: `pipeline.BUFFER_LIMITS`
4. Regular cleanup with `pipeline.shutdown()`

### Issue: Symbol not found

**Symptoms:**
```
ValueError: Symbol not found on exchange
```

**Solutions:**
1. Verify symbol format per exchange
2. Use `validate_and_get_symbol()` first
3. Check if symbol is available on exchange
4. Try alternative symbol format

---

## Performance Considerations

### Request Efficiency

- **Batch Requests**: Use `fetch_ohlcv_bulk()` for > 500 candles
- **Symbol Caching**: Validated symbols are cached per session
- **Rate Limiting**: Built-in delays prevent throttling

### Memory Optimization

| Configuration | Memory Usage | Coverage |
|--------------|--------------|----------|
| 2 symbols × 2 TF | ~0.4 MB | Basic |
| 5 symbols × 3 TF | ~1.5 MB | Standard |
| 10 symbols × 4 TF | ~4.0 MB | Heavy |

### Scaling Guidelines

- **< 10 symbols**: No special considerations
- **10-50 symbols**: Monitor memory and error rates
- **> 50 symbols**: Consider multiple pipeline instances

---

## Future Enhancements (Phase 2.2+)

### WebSocket Integration

```python
# Future: Real-time WebSocket updates
await pipeline.start_websocket_feeds(symbols, timeframes)

# Current: REST API foundation is in place
# WebSocket will update data_streams in real-time
```

### Advanced Features Planned

- **Real-time streaming** via WebSocket
- **Order book integration**
- **Trade data collection**
- **Tick-level data support**
- **Advanced caching strategies**

---

## Examples

See `/examples/market_data_pipeline_example.py` for complete working examples:

1. **Basic Usage** - Simple pipeline setup and data fetching
2. **Health Monitoring** - Status tracking and health checks
3. **Data Aggregator** - Multi-exchange aggregation
4. **Custom Configuration** - Custom indicator settings
5. **Multi-Timeframe** - Cross-timeframe analysis
6. **Error Handling** - Resilience demonstrations

Run examples:
```bash
# Set up environment first
export EXCHANGES='bingx,kucoinfutures'
export BINGX_KEY='your_key'
export BINGX_SECRET='your_secret'

# Run examples
python3 examples/market_data_pipeline_example.py
```

---

## Testing

Comprehensive test suite available in `tests/test_market_data_pipeline.py`:

```bash
# Run all pipeline tests
pytest tests/test_market_data_pipeline.py -v

# Run specific test
pytest tests/test_market_data_pipeline.py::test_pipeline_initialization -v

# Run with coverage
pytest tests/test_market_data_pipeline.py --cov=core.market_data_pipeline
```

**Test Coverage:**
- ✅ Pipeline initialization
- ✅ Data feed management
- ✅ Health monitoring
- ✅ Error handling
- ✅ Buffer management
- ✅ Multi-exchange support
- ✅ Integration with CcxtClient
- ✅ Memory optimization

---

## API Reference

### MarketDataPipeline

#### `__init__(exchanges, config=None)`
Initialize pipeline with exchange clients.

**Parameters:**
- `exchanges` (Dict[str, CcxtClient]): Exchange name to client mapping
- `config` (Dict, optional): Configuration dict for indicators and settings

#### `start_feeds(symbols, timeframes=['30m', '1h'])`
Start data feeds for symbols and timeframes.

**Parameters:**
- `symbols` (List[str]): Trading symbols (e.g., `['BTC/USDT:USDT']`)
- `timeframes` (List[str]): Timeframes to fetch (e.g., `['30m', '1h']`)

**Returns:**
- Dict with summary: `symbols_processed`, `successful_fetches`, `failed_fetches`, `exchanges_used`, `errors`

#### `get_latest_ohlcv(symbol, timeframe, exchange=None)`
Get latest OHLCV data with indicators.

**Parameters:**
- `symbol` (str): Trading symbol
- `timeframe` (str): Timeframe
- `exchange` (str, optional): Specific exchange name

**Returns:**
- DataFrame with OHLCV + indicators, or None if not available

#### `health_check()`
Get pipeline health metrics.

**Returns:**
- Dict with: `status`, `uptime_seconds`, `total_requests`, `failed_requests`, `error_rate`, `active_streams`, `is_running`

#### `get_pipeline_status()`
Get detailed pipeline status.

**Returns:**
- Dict with: `exchanges`, `memory_estimate_mb`, `data_freshness`, `buffer_limits`, `active_streams`, `total_requests`

#### `shutdown()`
Gracefully shutdown pipeline.

### DataAggregator

#### `__init__(pipeline)`
Initialize aggregator with pipeline.

**Parameters:**
- `pipeline` (MarketDataPipeline): Pipeline instance

#### `aggregate_multi_exchange(symbol, timeframe)`
Aggregate data from all exchanges.

**Parameters:**
- `symbol` (str): Trading symbol
- `timeframe` (str): Timeframe

**Returns:**
- Dict with `sources` containing exchange data and quality scores

#### `get_best_data_source(symbol, timeframe, exchanges=None)`
Get best exchange for symbol/timeframe.

**Parameters:**
- `symbol` (str): Trading symbol
- `timeframe` (str): Timeframe
- `exchanges` (List[str], optional): Limit to specific exchanges

**Returns:**
- str: Exchange name with best quality score, or None

#### `get_consensus_data(symbol, timeframe, min_sources=2)`
Create consensus data from multiple sources.

**Parameters:**
- `symbol` (str): Trading symbol
- `timeframe` (str): Timeframe
- `min_sources` (int): Minimum required sources

**Returns:**
- DataFrame with weighted consensus OHLCV data, or None

---

## Summary

Phase 2.1 Market Data Pipeline provides:

✅ **Multi-exchange data collection** with automatic fallback  
✅ **Automatic memory management** with circular buffers  
✅ **Integrated indicators** (RSI, ATR, EMA21/50/200)  
✅ **Health monitoring** and status tracking  
✅ **Error resilience** with exponential backoff  
✅ **Quality scoring** and data aggregation  
✅ **Seamless integration** with existing bot components  

**Ready for Production:**
- Thoroughly tested (16 tests passing)
- Well-documented with examples
- Production-grade error handling
- Scalable architecture

**Next Steps:**
- Phase 2.2: WebSocket real-time streaming
- Phase 2.3: Advanced order book integration
- Phase 3: Live trading integration

---

## Additional Resources

- **Usage Guide**: `docs/market_data_pipeline_usage.md`
- **Examples**: `examples/market_data_pipeline_example.py`
- **Tests**: `tests/test_market_data_pipeline.py`
- **Implementation**: `IMPLEMENTATION_DATA_AGGREGATOR.md`
- **Main README**: `README.md`

For questions or issues, refer to the troubleshooting section or check the test suite for usage patterns.
