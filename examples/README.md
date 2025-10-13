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

## BingX Multi-Exchange Example

**File:** `bingx_multi_exchange_example.py`

Demonstrates Phase 1 multi-exchange integration features:
- BingX server time synchronization
- Dynamic contract discovery (including VST/USDT)
- Unified data fetching across KuCoin and BingX
- Cross-exchange timestamp alignment
- VST contract validation

### Usage

```bash
python3 examples/bingx_multi_exchange_example.py
```

### What You'll Learn

1. **BingX Bulk Fetch** - Fetching OHLCV data from BingX
2. **VST Contract Validation** - Checking VST/USDT availability
3. **Multi-Exchange Unified Fetch** - Fetching from multiple exchanges
4. **Timestamp Alignment** - Synchronizing data across exchanges
5. **Exchange Status Summary** - Monitoring exchange health
6. **VST Trading Setup** - Complete VST trading configuration

---

## Phase 2: Market Intelligence Example

**File:** `market_intelligence_example.py`

Comprehensive demonstration of Phase 2 Market Intelligence Engine:
- Multi-timeframe market regime detection
- Adaptive OversoldBounce and ShortTheRip strategies
- VST market intelligence and optimization
- Real-time performance monitoring
- Complete integrated workflow

### Usage

```bash
python3 examples/market_intelligence_example.py
```

### What You'll Learn

1. **Market Regime Detection** - 4H/1H/30m multi-timeframe analysis
2. **Adaptive OversoldBounce** - Dynamic RSI thresholds and position sizing
3. **Adaptive ShortTheRip** - Regime-aware short strategy
4. **VST Intelligence** - BingX VST-specific optimization
5. **Performance Monitoring** - Real-time metrics and optimization feedback
6. **Integrated Workflow** - Complete end-to-end trading intelligence system

### Key Features Demonstrated

- **Intelligent Market Analysis:** Trend, momentum, and volatility classification
- **Dynamic Parameter Adaptation:** Real-time RSI/EMA threshold adjustment
- **VST Market Intelligence:** Conservative test trading parameters (10% allocation)
- **Performance Monitoring:** Win rate, Sharpe ratio, parameter drift detection
- **Self-Learning System:** Optimization recommendations based on performance

---

## Future Examples

More examples will be added here as the bot evolves:
- Phase 3: Live trading with adaptive strategies
- WebSocket integration for real-time updates
- Advanced multi-symbol portfolio management
- Custom strategy development templates
