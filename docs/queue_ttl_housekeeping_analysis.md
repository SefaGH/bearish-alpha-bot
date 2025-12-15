# PrioritySignalQueue TTL & Pending Housekeeping Analysis (Paper Run anomaly)

## Data Structures & Config Inputs
- Queue storage: `_heap: List[Tuple[-priority, enqueued_at, seq, payload]]` with `_condition` lock; priorities recomputed on refresh.
- Pending counters: `_pending_by_symbol: defaultdict(lambda: {"total": 0, "scale_in": 0})` tracking per-symbol total and scale-in counts.
- Config keys read in `PrioritySignalQueue.__init__`:
  - `risk.queue.ttl_seconds` → `_ttl`
  - `risk.queue.max_queue_depth` → `_max_depth`
  - `risk.queue.batch_dequeue` (used by callers, not the queue)
  - `risk.queue.max_pending_per_symbol` → `_max_pending_per_symbol` (entry/reentry and baseline cap)
  - `risk.queue.max_pending_scale_in_per_symbol` → `_max_pending_scale_in_per_symbol` (extra slots for scale_in when pyramiding enabled)
  - `pyramiding_enabled` flag (passed from StrategyCoordinator) toggles scale-in allowance logic.

## Where Pending Is Incremented/Decremented
- Incremented:
  - `put`: after enqueue, increments `pending["total"]`, and `pending["scale_in"]` if intent=scale_in.
  - `requeue`: same increments.
  - `_maybe_replace_lowest`: when replacing, increments for the new entry via earlier paths; decrements removed entry.
- Decremented:
  - `get`: before returning payload, decrements `pending["total"]` and `["scale_in"]` if intent=scale_in (recently fixed bug).
  - `_maybe_replace_lowest`: decrements removed payload’s counts.
  - `_purge_expired_locked`: decrements for expired entries.
- Not decremented:
  - If expired entries remain in heap without a call that triggers `_purge_expired_locked`.

## TTL / Expiration Behavior
- `_ttl` stored from config.
- Expiration check occurs in `_purge_expired_locked`, which is called inside `get` (and before priority refresh), and not inside `put`.
- `put` sets `queue_meta.expiration = now + _ttl` but does not purge.
- If no `get` or purge call occurs (e.g., dispatch loop crashed), expired entries remain in heap and pending counters stay > 0, causing `Queue limit reached` on subsequent `put`.

## Observed Log Pattern vs Code
- Logs: enqueue then immediate “Queue limit reached …” minutes later, with no positions executed, implies pending counters never cleared (likely because the consumer loop crashed and `get` never ran to purge TTL-expired items).
- Pending limits checked on `put` before any purge; TTL housekeeping only in `get`.

## Unit Tests (diagnostic)
- Added `tests/unit/test_priority_signal_queue_ttl.py` (xfail) to reproduce expectation that after TTL, a new enqueue should succeed:
  - `test_pending_cleared_after_ttl_for_same_symbol` (xfail): enqueue, advance fake time past TTL, attempt second enqueue without a `get`; expected accept, actual current behavior rejects due to uncleared pending.
  - `test_scale_in_pending_cleared_after_ttl` (xfail): same for scale_in with scale-in pending caps.
- Tests use monkeypatched `time.time` to control clock; xfail because current implementation does not purge on enqueue.

## Conclusion
- There is a real housekeeping gap: TTL expiration is only processed on `get`/purge, not on `put`; if the consumer stops (or lags) and items expire, per-symbol pending counters remain > 0 and block new signals, producing “Queue limit reached” even after TTL.

## Proposed Fix (design, not applied)
1) Trigger TTL purge before enforcing pending limits in `put` (and optionally `requeue`):
   ```python
   async with self._condition:
       self._purge_expired_locked()
       # then check per-symbol limits and enqueue
   ```
2) Ensure `_purge_expired_locked` decrements pending counters (already implemented) and uses safe defaults; avoid creating new entries on read by using `.get(symbol, {...})`.
3) Optionally schedule periodic purge or expose a housekeeping call if `get` is idle.
4) Edge cases: replacing entries should continue to decrement removed payload counts; requeue should respect current limits after purge.

These changes would allow expired items to be dropped and counters cleared even when the consumer loop is stalled, preventing spurious “Queue limit reached” blocks after TTL.
