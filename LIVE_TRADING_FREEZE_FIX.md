# Fix for Live Trading Bot Freeze Issue

## Problem Summary

The live trading bot was freezing after startup, completing all initialization and pre-flight checks but then becoming unresponsive with no trades executed and no error messages.

## Root Cause

**Location:** `src/core/production_coordinator.py`, line 1123

The `_fetch_ohlcv()` method was making synchronous blocking network calls within an async context:

```python
def _fetch_ohlcv(self, client, symbol: str, timeframe: str) -> pd.DataFrame:
    """Helper method to fetch OHLCV data via REST API."""
    try:
        rows = client.ohlcv(symbol, timeframe, limit=200)  # ❌ BLOCKING CALL
        return self._ohlcv_to_dataframe(rows)
```

The `client.ohlcv()` method calls `self.ex.fetch_ohlcv()` which:
1. Makes synchronous HTTP requests that can take several seconds
2. Includes `time.sleep(0.8)` for retry delays (also blocking)
3. Blocks the entire async event loop while waiting

### Impact

When the trading loop called `process_symbol()` → `_fetch_ohlcv()`, the blocking I/O prevented:
- WebSocket health monitoring from running
- Trading loop from processing other symbols
- Any async tasks from making progress
- The bot appeared "frozen" with no error or activity

## Solution

### Changes Made

1. **Made `_fetch_ohlcv` async** (`src/core/production_coordinator.py:1120-1133`)
   - Used `asyncio.to_thread()` to run blocking calls in a thread pool
   - Prevents event loop blocking while maintaining compatibility with synchronous CCXT library

```python
async def _fetch_ohlcv(self, client, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Helper method to fetch OHLCV data via REST API.
    
    This method runs the blocking client.ohlcv() call in a thread pool to prevent
    blocking the async event loop, which was causing the bot to freeze.
    """
    try:
        # Run blocking I/O in thread pool to prevent event loop blocking
        rows = await asyncio.to_thread(client.ohlcv, symbol, timeframe, limit=200)
        return self._ohlcv_to_dataframe(rows)
    except Exception as e:
        logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
        return None
```

2. **Updated all calls to `_fetch_ohlcv`** (line 300-302)
   - Added `await` keywords for proper async handling

3. **Added timeout protection** (line 456-495)
   - 30-second timeout per symbol in `_process_trading_loop()`
   - Prevents indefinite blocking on any single symbol
   - Logs timeout errors and continues processing

4. **Enhanced debug logging** (lines 697-703, 769)
   - Loop iteration tracking
   - Symbol processing start/end logging
   - Helps identify exactly where execution hangs

### Testing

Created comprehensive test suite (`tests/test_async_blocking_fix.py`):
- ✅ Verifies `_fetch_ohlcv` is properly async
- ✅ Tests concurrent fetches run in parallel (not sequentially)
- ✅ Validates timeout protection
- ✅ Confirms error handling
- ✅ All 6 tests passing

## Technical Explanation

### The Problem: Blocking in Async

When you call a blocking function (like `time.sleep()` or synchronous I/O) in an async context:
```python
async def async_function():
    result = blocking_call()  # ❌ Blocks entire event loop
```

The event loop cannot process any other tasks until the blocking call completes. This includes:
- Other coroutines waiting to run
- Network events (WebSocket messages)
- Timers and callbacks

### The Solution: Thread Pool

Using `asyncio.to_thread()`:
```python
async def async_function():
    result = await asyncio.to_thread(blocking_call)  # ✅ Runs in thread pool
```

The blocking call runs in a separate thread, allowing the event loop to continue processing other tasks.

## Prevention Guidelines

To prevent similar issues:

1. **Never call blocking functions in async code without `asyncio.to_thread()`**
   - Network I/O (`requests`, `ccxt` sync methods)
   - File I/O (unless using `aiofiles`)
   - CPU-intensive operations
   - `time.sleep()` (use `asyncio.sleep()` instead)

2. **Always use `await` with async functions**
   ```python
   result = await async_function()  # ✅ Correct
   result = async_function()         # ❌ Returns coroutine object, doesn't execute
   ```

3. **Add timeouts to prevent indefinite waiting**
   ```python
   result = await asyncio.wait_for(long_operation(), timeout=30.0)
   ```

4. **Use async libraries when available**
   - `aiohttp` instead of `requests`
   - `asyncio.sleep()` instead of `time.sleep()`
   - `aiofiles` instead of `open()`

## Testing Instructions

### Run the fix validation tests:
```bash
pytest tests/test_async_blocking_fix.py -v
```

### Verify bot doesn't freeze:
1. Start bot in paper trading mode
2. Monitor logs for loop iteration messages
3. Confirm symbols are processed continuously
4. No hanging or unresponsiveness

### Check for blocking calls:
```bash
# Search for potentially blocking patterns
grep -r "time.sleep" src/ --include="*.py"
grep -r "def.*fetch.*(" src/core/ --include="*.py" | grep -v "async def"
```

## Security

Ran CodeQL security analysis - **0 vulnerabilities found** ✅

## Related Issues

- Original issue: Live Trading Bot Freezes or Gets Stuck After Startup (WebSocket/Connection Issue)
- Phase 3.4: Critical Fixes - WebSocket race conditions and limits

## Verification

- [x] Root cause identified (blocking I/O in async context)
- [x] Fix implemented with `asyncio.to_thread()`
- [x] Comprehensive tests created and passing
- [x] Timeout protection added
- [x] Debug logging enhanced
- [x] Security scan passed (0 issues)
- [x] Documentation completed

## Future Improvements

1. Consider migrating to async CCXT library (`ccxt.pro`) for native async support
2. Add performance monitoring for I/O operations
3. Implement circuit breaker for repeated fetch failures
4. Add metrics dashboard showing event loop health
