# Signal Execution Pipeline - Comprehensive Fix

## Problem Summary
Signals were being validated and queued but never executed. In paper mode testing, 5 signals were queued but 0 positions were opened.

## Root Cause
The ProductionCoordinator was accepting signals into the StrategyCoordinator queue but was not properly forwarding them to the LiveTradingEngine queue. Additionally:
- Raw signals were being forwarded instead of enriched signals
- No lifecycle tracking to monitor signal flow
- No queue monitoring to detect bottlenecks
- Signal counters in LiveTradingEngine were not being displayed correctly

## Solution Components

### 1. Signal Forwarding (Critical Fix)
**File**: `src/core/production_coordinator.py`

**Change**: Modified `submit_signal()` method to forward the **enriched signal** (not raw signal) from StrategyCoordinator to LiveTradingEngine:

```python
# Before (line 774):
await self.trading_engine.signal_queue.put(signal)

# After:
enriched_signal = result['enriched_signal']
await self.trading_engine.signal_queue.put(enriched_signal)
```

**Why this matters**: The enriched signal contains critical metadata added by StrategyCoordinator:
- `strategy_name`: Strategy that generated the signal
- `signal_timestamp`: When signal was created
- `priority`: Signal priority level
- `strategy_allocation`: Capital allocation for this strategy
- `position_size`: Calculated position size
- Risk assessment metrics

### 2. Lifecycle Tracking
**File**: `src/core/production_coordinator.py`

**Added**: Signal lifecycle tracking through all stages:

```python
self.signal_lifecycle = {}  # signal_id -> {stage, timestamp, details}
```

**Stages tracked**:
1. `generated` - Signal created
2. `validated` - Signal passed validation
3. `queued` - Signal added to StrategyCoordinator queue
4. `forwarded` - Signal forwarded to LiveTradingEngine queue
5. `executed` - Signal executed (position opened)
6. `rejected` - Signal rejected (if applicable)

**Method**: `_track_signal_lifecycle(signal_id, stage, details)`

### 3. Queue Monitoring
**File**: `src/core/production_coordinator.py`

**Added**: Background monitoring task that logs queue sizes every 30 seconds:

```python
async def _monitor_signal_queues(self):
    """Monitor signal queues and log their sizes periodically."""
    while self.is_running:
        coordinator_queue_size = self.strategy_coordinator.signal_queue.qsize()
        engine_queue_size = self.trading_engine.signal_queue.qsize()
        
        logger.info(f"📊 [QUEUE-MONITOR] StrategyCoordinator: {coordinator_queue_size} signals | "
                   f"LiveTradingEngine: {engine_queue_size} signals")
        
        # Log lifecycle summary
        await asyncio.sleep(30)
```

### 4. Fixed Counters
**File**: `src/core/live_trading_engine.py`

**Changes**:
- Added `signals_received` counter to engine status output
- Fixed signal processing loop to correctly increment counter
- Added `signal_queue_size` to status for real-time monitoring

```python
def get_engine_status(self) -> Dict[str, Any]:
    return {
        'signals_received': self._signal_count,  # Now correctly displayed
        'signal_queue_size': self.signal_queue.qsize(),
        ...
    }
```

### 5. Enhanced Logging
**Files**: `src/core/production_coordinator.py`, `src/core/live_trading_engine.py`

**Added stage markers** throughout signal flow:

```python
logger.info(f"[STAGE:GENERATED] Signal {signal_id} for {symbol}")
logger.info(f"[STAGE:VALIDATED] Signal {signal_id} validated")
logger.info(f"[STAGE:QUEUED] Signal {signal_id} in StrategyCoordinator queue")
logger.info(f"[STAGE:FORWARDED] Signal {signal_id} forwarded to LiveTradingEngine queue")
logger.info(f"[STAGE:RECEIVED] 📤 Signal received from queue: {symbol}")
logger.info(f"[STAGE:EXECUTED] ✅ Signal executed: {symbol} - Position opened")
```

## Testing

### Unit Tests
Created `tests/test_signal_execution_flow.py` with 7 comprehensive tests:

1. ✅ `test_signal_forwarding_to_engine` - Validates signals reach engine queue
2. ✅ `test_signal_lifecycle_tracking` - Verifies all stages are tracked
3. ✅ `test_queue_monitoring` - Tests monitoring task functionality
4. ✅ `test_signal_rejection_tracking` - Validates rejection handling
5. ✅ `test_engine_status_shows_signals_received` - Tests counter display
6. ✅ `test_enriched_signal_forwarding` - Verifies enrichment metadata
7. ✅ `test_monitor_logs_queue_sizes` - Tests queue size reporting

### Integration Tests
Created `tests/test_signal_flow_integration.py` with 3 integration tests:

1. ✅ `test_complete_signal_execution_pipeline` - End-to-end signal flow
2. ✅ `test_signal_flow_with_rejection` - Rejection path validation
3. ✅ `test_queue_monitoring_integration` - Monitoring during signal processing

**All 10 tests passing** ✅

## Expected Result After Fix

### Before Fix:
```
Signals generated: 5
Signals validated: 5
Signals queued (Coordinator): 5
Signals queued (Engine): 0  ❌
Positions opened: 0  ❌
```

### After Fix:
```
Signals generated: 5
Signals validated: 5
Signals queued (Coordinator): 5
Signals queued (Engine): 5  ✅
Positions opened: 5  ✅
```

## Signal Flow Diagram

```
┌─────────────────┐
│ Strategy Signal │
└────────┬────────┘
         │
         ▼
[STAGE:GENERATED]
         │
         ▼
┌─────────────────────┐
│ StrategyCoordinator │
│  - Validate format  │
│  - Enrich metadata  │
│  - Risk assessment  │
└────────┬────────────┘
         │
         ▼
[STAGE:VALIDATED]
         │
         ▼
┌─────────────────────┐
│ Coordinator Queue   │
└────────┬────────────┘
         │
         ▼
[STAGE:QUEUED]
         │
         ▼
┌─────────────────────┐
│ Forward to Engine   │  ← CRITICAL FIX HERE
│ (enriched signal)   │
└────────┬────────────┘
         │
         ▼
[STAGE:FORWARDED]
         │
         ▼
┌─────────────────────┐
│ LiveTradingEngine   │
│  - Signal queue     │
└────────┬────────────┘
         │
         ▼
[STAGE:RECEIVED]
         │
         ▼
┌─────────────────────┐
│ Execute Signal      │
│  - Place order      │
│  - Open position    │
└────────┬────────────┘
         │
         ▼
[STAGE:EXECUTED]
```

## Files Modified

1. **src/core/production_coordinator.py**
   - Added signal lifecycle tracking dictionary
   - Modified `submit_signal()` to forward enriched signals
   - Added `_track_signal_lifecycle()` method
   - Added `_monitor_signal_queues()` background task
   - Enhanced logging with stage markers

2. **src/core/live_trading_engine.py**
   - Fixed signal processing loop (removed duplicate queue insertion)
   - Added `signals_received` to status output
   - Added `signal_queue_size` to status
   - Enhanced logging with stage markers

3. **tests/test_signal_execution_flow.py** (NEW)
   - 7 comprehensive unit tests
   - Tests all aspects of signal forwarding and tracking

4. **tests/test_signal_flow_integration.py** (NEW)
   - 3 integration tests
   - End-to-end signal flow validation

## Verification

To verify the fix is working:

1. **Check logs for stage markers**:
   ```
   [STAGE:GENERATED] Signal adaptive_ob_BTC/USDT:USDT_...
   [STAGE:VALIDATED] Signal adaptive_ob_BTC/USDT:USDT_...
   [STAGE:QUEUED] Signal adaptive_ob_BTC/USDT:USDT_...
   [STAGE:FORWARDED] Signal adaptive_ob_BTC/USDT:USDT_...
   [STAGE:RECEIVED] 📤 Signal received from queue: BTC/USDT:USDT
   [STAGE:EXECUTED] ✅ Signal executed: BTC/USDT:USDT - Position opened
   ```

2. **Check queue monitoring logs** (every 30s):
   ```
   📊 [QUEUE-MONITOR] StrategyCoordinator: 2 signals | LiveTradingEngine: 2 signals
   📊 [LIFECYCLE] Total tracked: 5 | Stages: {'executed': 3, 'forwarded': 2}
   ```

3. **Check engine status**:
   ```python
   status = trading_engine.get_engine_status()
   print(f"Signals received: {status['signals_received']}")
   print(f"Signal queue size: {status['signal_queue_size']}")
   print(f"Active positions: {status['active_positions']}")
   ```

4. **Run tests**:
   ```bash
   pytest tests/test_signal_execution_flow.py -v
   pytest tests/test_signal_flow_integration.py -v
   ```

## Summary

This comprehensive fix ensures signals flow correctly through the entire pipeline:

✅ Signals are properly validated by StrategyCoordinator
✅ Enriched signals (with metadata) are forwarded to LiveTradingEngine
✅ All stages are tracked and logged
✅ Queue sizes are monitored every 30 seconds
✅ Counters accurately reflect signal processing
✅ Complete test coverage validates the fix

The root cause was that raw signals were being forwarded instead of enriched signals, and there was no visibility into the signal flow. With these fixes, signals now flow correctly from generation to execution, with full tracking and monitoring.
