# Volume Policy & Execution System v2.0

## 1. System Overview
- Goal: move from hard rejects on low-volume signals to a defer-and-rescue model that preserves intent while enforcing safeguards.
- Signals now flow through a two-stage queue, enabling timed deferral instead of immediate rejection.
- Volume scores moved from legacy 0–10 scaling to normalized 0–1 (sigmoid-style) inputs used by the matrix logic.

## 2. Architecture: Dual-Stage Priority Queue
- Problem: deferred high-priority signals were reprocessed in tight loops, starving newer work.
- Solution: `PrioritySignalQueue` keeps two structures:
  - Ready Heap: `_queue` (heap) for immediately processable signals.
  - Waiting Room: `_waiting_room` (list) for signals scheduled for future processing.
- Lifecycle: `put(signal, process_after=ts)` → waiting room → `_check_waiting_room_locked` promotes when `now >= process_after` → heap → consumer `get`/`try_dispatch_next`.
- Busy-loop prevention: waiting items never re-enter the ready heap until their timer expires; promotions mark `queue_meta.is_deferred = True` for downstream logic.

## 3. The Brain: Volume Policy Matrix (Decision Logic)
Logic lives in `process_strategy_signal` (post-enrichment, pre-duplicate/risk):

- Case A (Normal volume): accept (standard profile).
- Case B (Low volume + tight stop < 0.15%): defer 1 bar (default 300s) via `signal_queue.put(..., process_after=now+300)`.
- Case C (Deferred returning, still Low volume): rescue path
  - Widen stop to at least 0.15%.
  - Apply `position_size_multiplier = 0.25`.
  - Enforce `execution_params = {type: LIMIT, post_only: True}`.
  - Recheck RR; require `rr_ratio >= 3.0` or reject.
- Case D (Low volume + wide stop > 0.50%): accept with limit/post-only and `position_size_multiplier = 0.35` (safety).
- Case E (Low volume + normal stop 0.15–0.50%): accept with limit/post-only and `position_size_multiplier = 0.50`.

Text decision tree:
- Volume >= NORMAL → Accept.
- Volume == LOW:
  - Stop < 0.15% → Defer 300s.
  - Stop > 0.50% → Accept (limit, post-only, mult 0.35).
  - Else (0.15–0.50%) → Accept (limit, post-only, mult 0.50).
- Deferred + Volume LOW → Rescue (widen to 0.15%, mult 0.25, limit/post-only, RR >= 3).
- Deferred + Volume >= NORMAL → Accept (standard).

## 4. Execution Layer: Intent-Based Simulator
- `SmartOrderManager._limit_order_execution` now reads `execution_params` and, in paper mode, logs intent: `[PAPER] Processing POST_ONLY Limit Order request`.
- Intent logging ensures pre-live verification that strategies are choosing the correct order style (LIMIT/POST_ONLY) even though fills remain simulated/instant.

## 5. Configuration Reference (volume_policy_matrix.yaml anchors)
- `risk_multipliers`: safety caps applied per branch (e.g., 0.25 rescue, 0.35 low-wide, 0.50 low-normal).
- `stop_bands`: tight (<0.15%), normal (0.15–0.50%), wide (>0.50%) thresholds.
- `execution` profiles: LIMIT + POST_ONLY enforced on low-volume accepts and rescues.
- `defer_time`: 1 bar (default 300s) for low-volume tight-stop deferral.

## 6. Observability & Logs
- Defer: `⏳ Deferring signal (Low Vol + Tight Stop < 0.15%)`
- Rescue RR check: log includes rescue recheck outcome and RR.
- Deferred promotion flag: `queue_meta.is_deferred` set when moved from waiting room to ready heap.
- Paper intent: `[PAPER] Processing POST_ONLY Limit Order request`
- Queue promotions: ready-heap depth logs (`[QUEUE] Signal enqueued... depth=...`) show waiting + ready counts.
