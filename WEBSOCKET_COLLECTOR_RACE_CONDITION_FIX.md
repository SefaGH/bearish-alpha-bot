# WebSocket Collector Race Condition Fix

## Problem Description

The bot was logging repeated warnings during startup:
```
⚠️ [INJECT] WebSocket manager or collector not found. Skipping data injection.
```

This error indicated that `MarketDataPipeline` was trying to inject historical OHLCV data into the WebSocket buffer before `WebSocketManager`'s collector was fully initialized, creating a **race condition**.

## Root Cause

The race condition occurred because:

1. `WebSocketManager` initialized its internal `_data_collector` in `__init__`
2. `MarketDataPipeline` tried to access it via `websocket_manager.collector`
3. The `collector` property didn't exist, making the collector appear "not found"
4. This happened during async initialization, where timing was unpredictable

## Solution

### 1. WebSocketManager Improvements

#### Added `collector` Property
```python
@property
def collector(self):
    """Property to access the data collector."""
    return getattr(self, '_data_collector', None)
```

This exposes the internal `_data_collector` through a clean public interface.

#### Added `is_collector_ready()` Method
```python
def is_collector_ready(self) -> bool:
    """Check if the data collector is ready to accept data."""
    return self.collector is not None
```

Provides a safe way to check if the collector is initialized.

#### Added `prime_buffer_with_dataframe()` Method to StreamDataCollector
```python
def prime_buffer_with_dataframe(self, exchange: str, symbol: str, timeframe: str, df):
    """Prime the buffer with historical data from a DataFrame."""
    # Converts DataFrame to OHLCV format and stores in buffer
```

This method allows `MarketDataPipeline` to inject historical data in the correct format.

### 2. MarketDataPipeline Improvements

#### Added `_wait_for_websocket_ready()` Helper
```python
async def _wait_for_websocket_ready(self, timeout: float = 10.0) -> bool:
    """Wait for WebSocket manager's collector to be ready."""
    # Polls every 100ms up to timeout
    # Returns True if ready, False on timeout
```

This prevents the race condition by waiting for the collector to be ready before proceeding.

#### Updated `prime_data_buffers_async()`
```python
async def prime_data_buffers_async(self, symbols: List[str], timeframes: List[str]):
    # CRITICAL: Wait for WebSocket collector to be ready before priming
    if not await self._wait_for_websocket_ready(timeout=10.0):
        logger.warning("[PRIME] WebSocket collector not ready - proceeding without WebSocket injection")
```

Now waits for collector readiness before attempting injection.

#### Enhanced Defensive Checks
Both `_fetch_and_store_async()` and `_fetch_and_store()` now include comprehensive null checks:

```python
# Check if websocket_manager exists
if not self.websocket_manager:
    logger.debug("[INJECT] No WebSocket manager - skipping")
    return True

# Check if collector exists and is ready
if not hasattr(self.websocket_manager, 'collector') or not self.websocket_manager.collector:
    logger.warning("[INJECT] Collector not found - skipping")
    return True

try:
    # Inject data
    self.websocket_manager.collector.prime_buffer_with_dataframe(...)
except Exception as e:
    logger.error(f"[INJECT] Failed: {e}")
    # Don't fail - continue without injection
```

### 3. ProductionCoordinator Verification

Added verification step after pipeline initialization:

```python
# STEP 8.5: VERIFY WEBSOCKET COLLECTOR READY
if self.websocket_manager:
    if hasattr(self.websocket_manager, 'is_collector_ready'):
        if self.websocket_manager.is_collector_ready():
            logger.info("✓ WebSocket collector verified ready")
        else:
            logger.warning("⚠️ WebSocket manager exists but collector not ready")
```

This provides early detection of initialization issues.

## Testing

Added comprehensive test suite (`tests/test_websocket_collector_race_condition.py`):

- ✅ `test_websocket_manager_has_collector_property` - Verifies property exposure
- ✅ `test_websocket_manager_is_collector_ready` - Tests readiness check
- ✅ `test_stream_data_collector_prime_buffer` - Tests data priming
- ✅ `test_market_data_pipeline_wait_for_websocket_ready` - Tests wait mechanism
- ✅ `test_market_data_pipeline_wait_timeout` - Tests timeout behavior
- ✅ `test_market_data_pipeline_no_websocket_manager` - Tests graceful degradation
- ✅ `test_defensive_null_checks_in_fetch_and_store` - Tests error handling
- ✅ `test_integration_websocket_collector_ready_flow` - Tests full flow

**Result**: All 8 tests passing

## Benefits

1. **Eliminates Race Condition**: The warning no longer appears in logs
2. **Graceful Degradation**: System continues to work even if WebSocket is unavailable
3. **Better Debugging**: Clear log messages indicate what's happening
4. **No Breaking Changes**: Backward compatible with existing code
5. **Improved Reliability**: REST API fallback ensures data is always available

## Usage

No changes required in existing code. The fix is transparent and automatic.

However, developers can now:

1. Check if collector is ready:
   ```python
   if websocket_manager.is_collector_ready():
       # Safe to inject data
   ```

2. Wait for collector to be ready:
   ```python
   ready = await pipeline._wait_for_websocket_ready(timeout=10.0)
   if ready:
       # Collector is ready
   ```

3. Prime buffer with historical data:
   ```python
   websocket_manager.collector.prime_buffer_with_dataframe(
       exchange='bingx',
       symbol='BTC/USDT:USDT',
       timeframe='1h',
       df=historical_dataframe
   )
   ```

## Lifecycle Flow

### Before Fix
```
1. WebSocketManager.__init__() creates _data_collector
2. MarketDataPipeline.__init__() stores reference to websocket_manager
3. MarketDataPipeline.prime_data_buffers_async() starts
   ↓
4. Pipeline tries to access websocket_manager.collector ❌ (doesn't exist)
   ↓
5. WARNING: "WebSocket manager or collector not found"
   ↓
6. Data injection skipped, REST API fallback used
```

### After Fix
```
1. WebSocketManager.__init__() creates _data_collector
   ↓
2. collector property exposes _data_collector ✅
   ↓
3. MarketDataPipeline.__init__() stores reference to websocket_manager
   ↓
4. MarketDataPipeline.prime_data_buffers_async() starts
   ↓
5. Pipeline calls _wait_for_websocket_ready(timeout=10.0)
   ↓
6. Wait loop checks websocket_manager.is_collector_ready() every 100ms
   ↓
7. Collector ready ✅ (or timeout after 10s)
   ↓
8. If ready: Data injection proceeds
   If timeout: Log warning, continue with REST API
```

## Performance Impact

- **Startup Time**: Adds up to 10s maximum wait (typically resolves in <100ms)
- **Runtime**: No performance impact after initialization
- **Memory**: Minimal overhead (one property accessor, one boolean check method)

## Security

- ✅ CodeQL scan: No vulnerabilities detected
- ✅ No sensitive data exposure
- ✅ No new attack vectors introduced

## Maintenance Notes

### For Developers

When working with WebSocketManager:

1. **Always check readiness** before accessing collector in async contexts
2. **Use defensive checks** when collector might not be available
3. **Log appropriately** - use INFO for normal flow, WARNING for degraded mode
4. **Test with timeout scenarios** to ensure graceful degradation

### Future Improvements

Potential enhancements (not critical):

1. Make timeout configurable via environment variable
2. Add metrics for collector initialization time
3. Add Prometheus metrics for WebSocket readiness
4. Consider event-based notification instead of polling

## Related Files

- `src/core/websocket_manager.py` - WebSocketManager and StreamDataCollector
- `src/core/market_data_pipeline.py` - MarketDataPipeline with wait logic
- `src/core/production_coordinator.py` - Initialization verification
- `tests/test_websocket_collector_race_condition.py` - Test suite
- `scripts/live_trading_launcher.py` - Production launcher

## References

- **Issue**: WebSocket Manager collector hazır olmadan veri enjeksiyonu: race condition ve tam çözüm
- **PR**: [Link to PR]
- **Labels**: `bug`, `critical`, `websocket`, `race-condition`, `data-pipeline`, `performance`
- **Assignee**: @SefaGH
