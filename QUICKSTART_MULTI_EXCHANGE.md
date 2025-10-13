# Quick Start: Multi-Exchange ULTIMATE Integration

## Installation

```bash
# Install dependencies
pip3 install -r requirements.txt
```

## Verify Installation

```bash
# Run all test suites (22 tests total)
python3 tests/test_kucoin_ultimate_integration.py    # 6 tests
python3 tests/test_bingx_ultimate_integration.py     # 9 tests
python3 tests/test_complete_ultimate_integration.py  # 7 tests
```

Expected output:
```
✓ All KuCoin Ultimate Integration tests passed!  (6/6)
✓ All BingX ULTIMATE Integration tests passed!   (9/9)
✓ Phase 1 Complete - BingX ULTIMATE Integration SUCCESS! (7/7)
```

## Quick Usage Examples

### 1. Simple BingX Data Fetch

```python
from core.ccxt_client import CcxtClient

# Initialize BingX client
client = CcxtClient('bingx')

# Fetch 1000 candles (auto-batches into 2x500)
candles = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', 1000)

print(f"Fetched {len(candles)} candles")
print(f"Latest price: ${candles[-1][4]:.2f}")
```

### 2. Multi-Exchange Portfolio

```python
from core.multi_exchange_manager import MultiExchangeManager

# Auto-initializes KuCoin + BingX
manager = MultiExchangeManager()

# Define portfolio across exchanges
portfolio = {
    'kucoinfutures': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
    'bingx': ['BTC/USDT:USDT']
}

# Fetch all data
data = manager.fetch_unified_data(portfolio, timeframe='1h', limit=500)

# Access data
btc_kucoin = data['kucoinfutures']['BTC/USDT:USDT']
btc_bingx = data['bingx']['BTC/USDT:USDT']
```

### 3. VST Contract Validation

```python
from core.multi_exchange_manager import MultiExchangeManager

manager = MultiExchangeManager()

# Check if VST is available on BingX
vst_info = manager.validate_vst_contract('bingx')

if vst_info['available']:
    print("✓ VST/USDT ready for trading on BingX")
    print(f"Contract type: {vst_info['contract_type']}")
else:
    print("VST not yet available")
```

### 4. Cross-Exchange Price Comparison

```python
from core.multi_exchange_manager import MultiExchangeManager

manager = MultiExchangeManager()

# Fetch BTC from both exchanges
data = manager.fetch_unified_data({
    'kucoinfutures': ['BTC/USDT:USDT'],
    'bingx': ['BTC/USDT:USDT']
}, timeframe='1h', limit=10)

# Compare latest prices
kucoin_price = data['kucoinfutures']['BTC/USDT:USDT'][-1][4]
bingx_price = data['bingx']['BTC/USDT:USDT'][-1][4]

print(f"KuCoin BTC: ${kucoin_price:.2f}")
print(f"BingX BTC: ${bingx_price:.2f}")
print(f"Difference: ${abs(kucoin_price - bingx_price):.2f}")
```

## Exchange-Specific Features

### KuCoin Futures

```python
from core.ccxt_client import CcxtClient

client = CcxtClient('kucoinfutures')

# Server time sync
server_time = client._get_kucoin_server_time()

# Dynamic contracts
contracts = client._get_dynamic_symbol_mapping()
print(f"Available contracts: {len(contracts)}")

# Bulk fetch with native format
candles = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', 1500)
```

### BingX Perpetual

```python
from core.ccxt_client import CcxtClient

client = CcxtClient('bingx')

# Server time sync
server_time = client._get_bingx_server_time()

# Dynamic contracts (including VST)
contracts = client._get_bingx_contracts()
print(f"Available contracts: {len(contracts)}")

# Bulk fetch with native format
candles = client.fetch_ohlcv_bulk('VST/USDT:USDT', '30m', 1000)
```

## Advanced Features

### Timestamp Alignment

```python
from core.multi_exchange_manager import MultiExchangeManager

manager = MultiExchangeManager()

# Fetch data from multiple exchanges
data = manager.fetch_unified_data({
    'kucoinfutures': ['BTC/USDT:USDT'],
    'bingx': ['BTC/USDT:USDT']
}, timeframe='1h', limit=100)

# Align timestamps for synchronized analysis
aligned = manager.align_timestamps(data, tolerance_ms=60000)

# Now all exchanges have matching timestamps
print(f"Aligned candles: {len(aligned['kucoinfutures']['BTC/USDT:USDT'])}")
```

### Exchange Status Monitoring

```python
from core.multi_exchange_manager import MultiExchangeManager

manager = MultiExchangeManager()

# Get status of all exchanges
summary = manager.get_exchange_summary()

print(f"Total exchanges: {summary['total_exchanges']}")
for name, info in summary['exchanges'].items():
    status = info['status']
    markets = info.get('markets', 'N/A')
    print(f"  {name}: {status} ({markets} markets)")
```

## Common Patterns

### Pattern 1: Backtest Data Collection

```python
from core.ccxt_client import CcxtClient
import pandas as pd

# Collect historical data for backtesting
client = CcxtClient('bingx')
candles = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '1h', 2000)

# Convert to DataFrame
df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### Pattern 2: Multi-Timeframe Analysis

```python
from core.multi_exchange_manager import MultiExchangeManager

manager = MultiExchangeManager()

# Fetch same symbol on different timeframes
timeframes = ['15m', '30m', '1h', '4h']
data_by_tf = {}

for tf in timeframes:
    data = manager.fetch_unified_data(
        {'bingx': ['BTC/USDT:USDT']},
        timeframe=tf,
        limit=500
    )
    data_by_tf[tf] = data['bingx']['BTC/USDT:USDT']

print(f"Collected {len(timeframes)} timeframes")
```

### Pattern 3: Real-time Monitoring

```python
from core.multi_exchange_manager import MultiExchangeManager
import time

manager = MultiExchangeManager()

while True:
    # Fetch latest data
    data = manager.fetch_unified_data({
        'kucoinfutures': ['BTC/USDT:USDT'],
        'bingx': ['BTC/USDT:USDT']
    }, timeframe='1m', limit=1)
    
    # Process latest prices
    for exchange, exchange_data in data.items():
        for symbol, candles in exchange_data.items():
            if candles:
                price = candles[-1][4]
                print(f"{exchange} {symbol}: ${price:.2f}")
    
    time.sleep(60)  # Update every minute
```

## Configuration

### VST Trading Configuration

```python
VST_CONFIG = {
    'symbol': 'VST/USDT:USDT',
    'exchange': 'bingx',
    'contract_type': 'perpetual',
    'test_mode': True,
    'allocation_pct': 0.1,  # 10% for testing
    'timeframe': '30m',
    'limit': 1000
}
```

### Exchange Selection

```python
from core.ccxt_client import CcxtClient
from core.multi_exchange_manager import MultiExchangeManager

# Custom exchange selection
custom_exchanges = {
    'kucoinfutures': CcxtClient('kucoinfutures'),
    'bingx': CcxtClient('bingx')
}

manager = MultiExchangeManager(exchanges=custom_exchanges)
```

## Troubleshooting

### Network Errors

The implementation handles network errors gracefully with fallback mechanisms:

```python
# Server time sync falls back to local time
# Contract discovery falls back to essential symbols (BTC, ETH, VST)
# All methods log warnings but continue operation
```

### Rate Limiting

Conservative rate limits are built-in:
- 0.7 seconds between batches
- 0.5 seconds between symbols
- 1-hour caching for contract discovery

## Documentation

- **Full BingX Docs:** [`BINGX_ULTIMATE_INTEGRATION.md`](BINGX_ULTIMATE_INTEGRATION.md)
- **KuCoin Docs:** [`KUCOIN_ULTIMATE_INTEGRATION.md`](KUCOIN_ULTIMATE_INTEGRATION.md)
- **Implementation Summary:** [`MULTI_EXCHANGE_INTEGRATION_SUMMARY.md`](MULTI_EXCHANGE_INTEGRATION_SUMMARY.md)

## Examples

Run the comprehensive example file:

```bash
python3 examples/bingx_multi_exchange_example.py
```

This demonstrates:
1. BingX bulk OHLCV fetch
2. VST contract validation
3. Multi-exchange unified fetch
4. Timestamp alignment
5. Exchange status summary
6. Complete VST trading setup

## Support

For issues or questions:
- Check test results: `python3 tests/test_complete_ultimate_integration.py`
- Review logs for detailed error messages
- Ensure all dependencies are installed
- Verify network connectivity (or accept fallback behavior)

---

**Status:** ✅ Phase 1 Complete - Ready for production use
