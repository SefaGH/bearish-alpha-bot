# Phased Initialization Guide

## Overview

The bot's initialization has been refactored into clear, sequential phases to ensure data layer health before ML initialization.

## Initialization Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 0: LAUNCHER START                   │
│                  (scripts/live_trading_launcher.py)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 1: CORE SYSTEMS                        │
│                                                               │
│  ✓ Exchange Connection (CcxtClient)                          │
│  ✓ WebSocket Manager (stream setup)                          │
│  ✓ Market Data Pipeline (historical + streaming)             │
│  ✓ Risk Manager                                              │
│  ✓ Portfolio Manager                                         │
│  ✓ Trading Engine                                            │
│  ✓ Circuit Breaker                                           │
│                                                               │
│  Methods:                                                     │
│  - LiveTradingLauncher._initialize_production_system_core()  │
│  - ProductionCoordinator.initialize_core_systems()           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           PHASE 1.5: DATA LAYER HEALTH CHECK                 │
│                                                               │
│  Validates BEFORE ML initialization:                          │
│  ✅ WebSocket Connection: Active?                            │
│  ✅ Subscriptions: Successful?                               │
│  ✅ Data Flow: Receiving live data?                          │
│                                                               │
│  Methods:                                                     │
│  - LiveTradingLauncher._perform_data_health_check()          │
│  - ProductionCoordinator.is_data_layer_healthy()             │
│                                                               │
│  ⚠️  System continues with REST API fallback if unhealthy    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               PHASE 2: ML SYSTEMS                            │
│                                                               │
│  ✓ Feature Engineering Pipeline                              │
│  ✓ Price Prediction Engine                                   │
│  ✓ Regime Predictor                                          │
│  ✓ Reinforcement Learning Agent                              │
│  ✓ ML Strategy Integration                                   │
│                                                               │
│  Methods:                                                     │
│  - LiveTradingLauncher._initialize_production_system_ml()    │
│  - ProductionCoordinator.initialize_ml_systems()             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: FINALIZE & START                       │
│                                                               │
│  ✓ Register Strategies                                       │
│  ✓ Pre-flight Checks                                         │
│  ✓ Start Trading Loop                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                  ▶️ TRADING ACTIVE
```

## Expected Log Output

### Phase 1: Core Systems
```
[PHASE 1] INITIALIZING CORE SYSTEMS
==============================================================================
✓ Received 1 exchange client(s): ['bingx']
✓ WebSocket manager received from launcher (external).
✓ Market data pipeline initialized
✓ Performance monitor initialized
✓ Risk config prepared: $100.0 equity
✓ Risk manager initialized (portfolio value: $100.00)
✓ Order manager initialized (dependencies pending)
✓ Position manager initialized and linked with OrderManager
✓ Portfolio manager initialized
✓ All managers have been interlinked
✓ WebSocket collector verified ready
✓ Strategy coordinator initialized
✓ Circuit breaker system initialized
✓ Live trading engine initialized (mode: paper)
✓ Active symbols set: 3 symbols

[PHASE 1] ✅ CORE SYSTEMS INITIALIZATION COMPLETE
==============================================================================
Components initialized: 7
Portfolio value: $100.00
Active symbols: 3
Mode: paper
==============================================================================
```

### Phase 1.5: Data Layer Health Check
```
[PHASE 1.5] DATA LAYER HEALTH CHECK
==============================================================================
🏥 [HEALTH-CHECK] Performing data layer health check...
   ✅ WebSocket connection: Active
   ✅ Subscriptions: 3 active streams
   ✅ Data flow: Confirmed (3 symbols)
🏥 [HEALTH-CHECK] ✅ Data layer is HEALTHY

[PHASE 1.5] ✅ DATA LAYER IS HEALTHY
==============================================================================
```

### Phase 2: ML Systems
```
[PHASE 2] INITIALIZING ML SYSTEMS
==============================================================================
🧠 [ML-INIT] Initializing ML system...
✅ Feature engineering pipeline ready
✅ Price prediction engine initialized
✅ Regime predictor initialized
✅ Reinforcement learning agent initialized
✅ ML strategy integration manager initialized with MarketDataPipeline
🔗 ML connected to StrategyCoordinator
🔗 ML connected to LiveTradingEngine
🧠 [ML-INIT] ✅ ML SYSTEM INITIALIZED
   Components: feature_pipeline, price_engine, regime_predictor, rl_agent, ml_integration

[PHASE 2] ✅ ML SYSTEMS INITIALIZATION COMPLETE
==============================================================================
```

### Phase 3: Trading Start
```
[PHASE 3] FINALIZING SETUP
==============================================================================
✓ adaptive_ob: 0.5 allocation
✓ adaptive_str: 0.5 allocation

STARTING PRODUCTION TRADING LOOP
==============================================================================
Mode: PAPER
Duration: Indefinite
Active Symbols: 3
==============================================================================

🔄 [LOOP-START] Main trading loop entered successfully
```

## Health Check Details

### WebSocket Connection Check
- Verifies at least one WebSocket client is connected
- Uses `websocket_manager.is_any_client_connected()`
- Status: `healthy`, `unhealthy`, or `not_available`

### Subscriptions Check
- Verifies active stream count > 0
- Uses `websocket_manager.get_active_stream_count()`
- Status: `healthy`, `unhealthy`, or `not_available`

### Data Flow Check
- Verifies actual data received from collector
- Checks first 3 symbols for data availability
- Uses `websocket_manager.get_latest_data(symbol, timeframe)`
- Status: `healthy`, `degraded`, `not_available`, or `error`

## Fallback Behavior

If health check fails:
1. System logs warning but does NOT fail
2. ML initialization still proceeds
3. System uses REST API for data fetching
4. WebSocket is used opportunistically when available

## API Reference

### ProductionCoordinator Methods

#### `initialize_core_systems()`
```python
async def initialize_core_systems(
    self,
    exchange_clients: Optional[Dict] = None,
    portfolio_config: Optional[Dict] = None,
    mode: str = 'paper',
    trading_symbols: Optional[List[str]] = None,
    websocket_manager: Optional[Any] = None
) -> Dict[str, Any]
```
Initializes all non-ML components (Phase 1).

**Returns:**
```python
{
    'success': bool,
    'components': List[str],
    'active_symbols_count': int,
    'reason': str  # only if success=False
}
```

#### `is_data_layer_healthy()`
```python
async def is_data_layer_healthy(self) -> Dict[str, Any]
```
Performs health check on data layer (Phase 1.5).

**Returns:**
```python
{
    'healthy': bool,
    'checks': {
        'websocket_connection': {
            'status': str,  # 'healthy', 'unhealthy', 'not_available', 'error'
            'details': str
        },
        'subscriptions': {
            'status': str,
            'details': str
        },
        'data_flow': {
            'status': str,
            'details': str
        }
    },
    'timestamp': datetime
}
```

#### `initialize_ml_systems()`
```python
async def initialize_ml_systems(
    self,
    price_engine: Optional[Any] = None,
    regime_predictor: Optional[Any] = None
) -> Dict[str, Any]
```
Initializes ML components (Phase 2).

**Returns:**
```python
{
    'success': bool,
    'components': List[str],
    'reason': str  # only if success=False
}
```

### LiveTradingLauncher Methods

#### `_initialize_production_system_core()`
Wrapper for Phase 1 initialization.

#### `_perform_data_health_check()`
Wrapper for Phase 1.5 health check with detailed logging.

#### `_initialize_production_system_ml()`
Wrapper for Phase 2 ML initialization.

## Testing

Run tests:
```bash
python3.11 -m pytest tests/test_phased_initialization.py -v
```

## Migration Guide

### Old Code
```python
# Old monolithic initialization
result = await coordinator.initialize_production_system(
    exchange_clients=clients,
    websocket_manager=ws_manager,
    price_engine=price_engine,
    regime_predictor=regime_predictor
)
```

### New Phased Code
```python
# Phase 1: Core Systems
core_result = await coordinator.initialize_core_systems(
    exchange_clients=clients,
    websocket_manager=ws_manager
)

# Phase 1.5: Health Check
health = await coordinator.is_data_layer_healthy()

# Phase 2: ML Systems (only if healthy)
if health['healthy']:
    ml_result = await coordinator.initialize_ml_systems(
        price_engine=price_engine,
        regime_predictor=regime_predictor
    )
```

### Backward Compatibility

The old `initialize_production_system()` method still works and now internally calls the phased methods in sequence, so existing code continues to function.

## Benefits

1. **Clear Phase Separation**: Each phase has a specific responsibility
2. **Early Failure Detection**: Data layer issues caught before ML initialization
3. **Better Debugging**: Clear log markers show exactly where initialization fails
4. **Graceful Degradation**: System can continue with REST API if WebSocket fails
5. **ML Safety**: ML only initializes after data layer is confirmed healthy

## Troubleshooting

### ML initialization fails
- Check Phase 1.5 health check logs
- Verify WebSocket connections are active
- Check data flow from collector

### Health check always fails
- Verify WebSocket manager is properly initialized
- Check exchange credentials
- Verify network connectivity
- Review WebSocket logs for connection errors

### System slow to start
- Health check includes retries with delays
- WebSocket connection may take time to establish
- Review timeout settings in configuration
