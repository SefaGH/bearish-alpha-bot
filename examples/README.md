# Examples

This directory contains example scripts demonstrating various features of the Bearish Alpha Bot.

## KuCoin Futures Bulk Fetch Example

**File:** `kucoin_bulk_fetch_example.py`

Demonstrates the new KuCoin Futures integration features:
- Server time synchronization
- Dynamic symbol discovery
- Bulk OHLCV fetching (up to 2000 candles)
- Automatic batch management
- Multiple symbol handling

### Usage

```bash
python3 examples/kucoin_bulk_fetch_example.py
```

### What You'll Learn

1. **Simple Bulk Fetch** - Basic usage to fetch 1000 candles
2. **Maximum Bulk Fetch** - Fetching the maximum 2000 candles
3. **DataFrame Integration** - Converting to pandas DataFrames
4. **Automatic Selection** - How the system chooses between regular and bulk fetching
5. **Multiple Symbols** - Iterating over multiple trading pairs
6. **Server Time Sync** - Understanding time synchronization
7. **Dynamic Symbols** - Exploring automatic symbol discovery

### Requirements

- Python 3.10+
- ccxt
- pandas
- requests

All dependencies are in `requirements.txt` at the repository root.

## Future Examples

More examples will be added here as the bot evolves:
- Live trading examples
- Strategy customization
- Risk management configuration
- Multi-exchange scanning
