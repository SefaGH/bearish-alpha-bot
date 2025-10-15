# WebSocket Manager Phase 2.1 Integration

## Overview

Enhanced WebSocket Manager with seamless integration to Phase 1 Multi-Exchange Framework using `build_clients_from_env()` pattern.

## Key Features

### 1. CcxtClient Integration

The WebSocketManager now accepts `Dict[str, CcxtClient]` instances, enabling better integration with the existing multi-exchange framework:

```python
from core.multi_exchange import build_clients_from_env
from core.websocket_manager import WebSocketManager

# Phase 1: Build clients from environment
clients = build_clients_from_env()

# Phase 3.1: Initialize WebSocket manager with CcxtClient instances
config = {
    'reconnect_delay': 5,
    'max_retries': 3,
    'timeout': 30
}
ws_manager = WebSocketManager(clients, config=config)
```

### 2. New Subscription API

Enhanced API methods for real-time data subscriptions:

```python
# Subscribe to tickers
await ws_manager.subscribe_tickers(['BTC/USDT:USDT', 'ETH/USDT:USDT'])

# Subscribe to orderbook (planned for future)
await ws_manager.subscribe_orderbook('BTC/USDT:USDT', depth=20)

# Start all subscribed streams
subscriptions = {
    'tickers': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
    'ohlcv': ['BTC/USDT:USDT'],
    'timeframe': '1m'
}
await ws_manager.start_streams(subscriptions)

# Graceful shutdown
await ws_manager.shutdown()
```

### 3. Callback Registration System

Method chaining for registering callbacks:

```python
async def on_ticker(exchange, symbol, ticker):
    print(f"{symbol} @ ${ticker['last']}")

async def on_orderbook(exchange, symbol, orderbook):
    print(f"{symbol} orderbook updated")

# Register callbacks with method chaining
ws_manager.on_ticker_update(on_ticker).on_orderbook_update(on_orderbook)
```

### 4. Backward Compatibility

Maintains full backward compatibility with existing credential-based initialization:

```python
# Legacy mode - still works!
exchanges = {
    'kucoinfutures': {'apiKey': '...', 'secret': '...'},
    'bingx': {'apiKey': '...', 'secret': '...'}
}
ws_manager = WebSocketManager(exchanges)
```

## Multi-Exchange Support

Priority exchanges from production bot:
- ✅ **BingX** - With Phase 1 BingXAuthenticator integration
- ✅ **KuCoin Futures** (kucoinfutures)
- ✅ **Binance**
- ✅ **Bitget**
- ✅ All other CCXT Pro supported exchanges

## API Reference

### Constructor

```python
WebSocketManager(
    exchanges: Optional[Union[Dict[str, Dict[str, str]], Dict[str, CcxtClient]]] = None,
    config: Dict[str, Any] = None
)
```

**Parameters:**
- `exchanges`: Can be either:
  - `Dict[str, CcxtClient]` - Exchange clients from `build_clients_from_env()` (recommended)
  - `Dict[str, Dict[str, str]]` - Credential dictionaries (legacy)
  - `None` - Uses default exchanges (KuCoin + BingX, unauthenticated)
- `config`: Configuration dictionary with options like:
  - `reconnect_delay`: Delay in seconds before reconnection attempt
  - `max_retries`: Maximum reconnection attempts
  - `timeout`: Connection timeout in seconds

### Methods

#### `subscribe_tickers(symbols: List[str], callback=None) -> List[Task]`

Subscribe to real-time ticker updates for multiple symbols.

**Example:**
```python
symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
tasks = await ws_manager.subscribe_tickers(symbols)
```

#### `subscribe_orderbook(symbol: str, depth: int = 20, callback=None) -> Task`

Subscribe to L2 orderbook stream for a symbol.

**Note:** Currently uses ticker streams as proxy. Full orderbook support coming in future phase.

#### `start_streams(subscriptions: Dict[str, List[str]]) -> Dict[str, List[Task]]`

Start WebSocket streams for specified subscriptions.

**Example:**
```python
subscriptions = {
    'tickers': ['BTC/USDT:USDT'],
    'ohlcv': ['BTC/USDT:USDT'],
    'timeframe': '1m'
}
tasks = await ws_manager.start_streams(subscriptions)
```

#### `on_ticker_update(callback) -> self`

Register callback for ticker updates. Returns self for method chaining.

**Callback signature:**
```python
async def callback(exchange: str, symbol: str, ticker: Dict)
```

#### `on_orderbook_update(callback) -> self`

Register callback for orderbook updates. Returns self for method chaining.

**Callback signature:**
```python
async def callback(exchange: str, symbol: str, orderbook: Dict)
```

#### `async shutdown()`

Graceful shutdown of all WebSocket connections.

### Attributes

- `exchanges`: Original exchange parameter
- `config`: Configuration dictionary
- `connections`: Active WebSocket connections
- `callbacks`: Registered callback functions (defaultdict)
- `is_running`: Boolean indicating if streams are running
- `reconnect_delays`: Track reconnection delays per exchange
- `clients`: Dictionary of WebSocketClient instances

## Integration Patterns

### Pattern 1: Production Coordinator

```python
from core.multi_exchange import build_clients_from_env
from core.websocket_manager import WebSocketManager

# Phase 1: Multi-exchange setup
exchange_clients = build_clients_from_env()

# Phase 3.1: WebSocket streaming
ws_manager = WebSocketManager(exchange_clients, config={'reconnect_delay': 5})
```

### Pattern 2: Market Data Pipeline

```python
from core.websocket_manager import WebSocketManager, StreamDataCollector

# Initialize
ws_manager = WebSocketManager(clients)
collector = StreamDataCollector(buffer_size=1000)

# Register collector
ws_manager.on_ticker_update(collector.ticker_callback)

# Start streaming
await ws_manager.subscribe_tickers(['BTC/USDT:USDT'])
await ws_manager.run_streams(duration=60)
```

### Pattern 3: Live Trading Integration

```python
from core.live_trading_engine import LiveTradingEngine

# All Phase 3 components
trading_engine = LiveTradingEngine(
    portfolio_manager=portfolio_mgr,
    risk_manager=risk_mgr,
    websocket_manager=ws_manager,  # Enhanced manager with CcxtClient support
    exchange_clients=exchange_clients
)
```

## Testing

Run all WebSocket tests:
```bash
pytest tests/test_websocket_infrastructure.py -v
pytest tests/test_phase_integration_websocket.py -v
```

Run examples:
```bash
python examples/websocket_ccxt_integration_example.py
python examples/websocket_streaming_example.py
```

## Configuration Examples

### Basic Configuration
```python
config = {
    'reconnect_delay': 5,
    'max_retries': 3
}
ws_manager = WebSocketManager(clients, config=config)
```

### Production Configuration
```python
config = {
    'reconnect_delay': 10,
    'max_retries': 5,
    'timeout': 30,
    'enable_heartbeat': True,
    'heartbeat_interval': 20
}
ws_manager = WebSocketManager(clients, config=config)
```

## Compatibility

- ✅ **Phase 1**: `build_clients_from_env()` pattern
- ✅ **Phase 2**: MarketDataPipeline, MarketRegimeAnalyzer
- ✅ **Phase 3.1**: Existing WebSocket infrastructure
- ✅ **Phase 3.2**: RiskManager integration
- ✅ **Phase 3.3**: PortfolioManager integration
- ✅ **Phase 3.4**: LiveTradingEngine integration
- ✅ **GitHub Actions**: Compatible with CI/CD workflows
- ✅ **main.py**: Compatible with orchestration patterns

## Migration Guide

### From Legacy Mode to CcxtClient Mode

**Before:**
```python
exchanges = {
    'kucoinfutures': {'apiKey': '...', 'secret': '...'},
    'bingx': {'apiKey': '...', 'secret': '...'}
}
ws_manager = WebSocketManager(exchanges)
```

**After:**
```python
from core.multi_exchange import build_clients_from_env

clients = build_clients_from_env()
ws_manager = WebSocketManager(clients)
```

**Note:** Legacy mode still works - no breaking changes!

## Future Enhancements

- [ ] Full L2 orderbook streaming support
- [ ] L3 orderbook (full depth) streaming
- [ ] Trade stream support
- [ ] Funding rate streams
- [ ] Liquidation data streams
- [ ] Advanced reconnection strategies
- [ ] Connection pooling
- [ ] Stream multiplexing optimization

## See Also

- [Phase 3.1 WebSocket Infrastructure](../PHASE3_WEBSOCKET_INFRASTRUCTURE.md)
- [Phase 1 Multi-Exchange Integration](../MULTI_EXCHANGE_INTEGRATION_SUMMARY.md)
- [examples/websocket_ccxt_integration_example.py](../examples/websocket_ccxt_integration_example.py)
- [examples/websocket_streaming_example.py](../examples/websocket_streaming_example.py)
