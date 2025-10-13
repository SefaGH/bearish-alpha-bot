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

## Phase 3.1: WebSocket Streaming Example

**File:** `websocket_streaming_example.py`

Comprehensive demonstration of Phase 3.1 WebSocket Infrastructure:
- Real-time OHLCV candle streaming
- Real-time ticker data streaming
- Multi-symbol streaming
- Multi-exchange coordination
- Data collection and buffering

### Usage

```bash
python3 examples/websocket_streaming_example.py
```

### What You'll Learn

1. **Basic OHLCV Streaming** - Stream 1-minute candles from single exchange
2. **Multi-Symbol Streaming** - Stream BTC and ETH simultaneously
3. **Multi-Exchange Streaming** - Stream from KuCoin and BingX simultaneously
4. **Ticker Streaming** - Real-time bid/ask spread monitoring

### Key Features Demonstrated

- **Real-Time Data:** WebSocket-based streaming with < 100ms latency
- **Async Architecture:** Non-blocking asyncio-based implementation
- **Stream Management:** Automatic lifecycle handling and cleanup
- **Data Collection:** Built-in buffering with configurable limits
- **Multi-Exchange:** Unified interface across KuCoin, BingX, and more

### Note

WebSocket examples require live exchange connections. In sandbox environments with DNS restrictions, connections will fail gracefully with appropriate error handling.

---

## Phase Integration Example

**File:** `phase_integration_example.py`

Complete demonstration of Phase 1 + Phase 2 + Phase 3.1 integration:
- Multi-exchange REST API data fetching (Phase 1)
- Market regime analysis (Phase 2)
- Real-time WebSocket streaming (Phase 3.1)

### Usage

```bash
python3 examples/phase_integration_example.py
```

### What You'll Learn

1. **Phase 1 Integration** - Multi-exchange REST API for historical data
2. **Phase 2 Integration** - Intelligent market regime detection
3. **Phase 3.1 Integration** - Real-time WebSocket streaming
4. **Complete Workflow** - How all phases work together seamlessly

### Integration Architecture

```
Phase 3.1: WebSocket Streaming (Real-time data)
           ↓
Phase 2: Market Intelligence (Analysis & adaptation)
           ↓
Phase 1: Multi-Exchange Framework (Data fetching & coordination)
```

---

## Future Examples

More examples will be added here as the bot evolves:
- Phase 3.2: Advanced WebSocket features (order books, trades)
- Phase 3.3: Live trading with real-time regime adaptation
- Advanced multi-symbol portfolio management
- Custom strategy development templates
