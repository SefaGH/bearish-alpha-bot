# QUALITY-BEFORE-RISK NameError Analysis

## 1. Overview
- Runtime error: `NameError: name 'strategy_name' is not defined` during `_assess_signal_risk` when emitting the new `[QUALITY-BEFORE-RISK]` log.
- Observed after sizing in a paper-mode extreme-bypass + pyramiding scenario (post `[Signal Enriched]` and `[SIZING-PROPOSED]` logs), so the risk pipeline aborts before RiskManager executes.

## 2. Code Path & Root Cause
- The log statement in `src/core/strategy_coordinator.py:2982-2989`:
  ```python
  logger.info(
      "[QUALITY-BEFORE-RISK] strat=%s | sym=%s | intent=%s | extreme_bypass=%s | quality=%.3f",
      sized_signal.get('strategy_name', strategy_name),  # <- NameError
      sized_signal.get('symbol'),
      sized_signal.get('intent'),
      sized_signal.get('extreme_bypass', False),
      float(sized_signal.get('quality_score', 0.0) or 0.0),
  )
  ```
- `_assess_signal_risk(self, signal)` does not accept `strategy_name` and never defines it locally. The default argument `strategy_name` in the `get` call therefore raises `NameError` before the log is emitted.
- `sized_signal` comes from `AdvancedPositionSizing.calculate_optimal_size(signal, return_signal=True)` and is the same signal dict augmented with sizing fields (`amount`, `notional`, `position_size`, `sizing_meta`, etc.). It preserves whatever fields were already present on the incoming `signal`.
  - In the full strategy path (`process_strategy_signal`), `_enrich_signal` adds `strategy_name` to the signal before `_assess_signal_risk` is invoked, so `sized_signal` typically carries it.
  - In the simplified `process_signal` path, `strategy_name` might be missing unless provided by the caller, but the current failure happens even before any fallback is considered because the variable is undefined in scope.

## 3. Impact
- The `NameError` is caught by the `_assess_signal_risk` try/except and returned as `{'acceptable': False, 'reason': 'Risk assessment error: name ...'}`.
- Callers treat this as a risk rejection (`process_strategy_signal` logs a rejection at the risk stage), so:
  - `risk_manager.size_and_validate_position` never runs for that signal.
  - Planner/pyramiding scaling logic is skipped, masking real behavior in paper-mode runs.
- Effectively, any signal that hits this log path is dropped from the pipeline once sizing finishes.

## 4. Fix Options

**Option A: Use only in-scope, safe fallbacks inside `_assess_signal_risk` (no signature change).**
- Change the log to derive a label from available dict fields, e.g.:
  - `strategy_label = sized_signal.get('strategy_name') or signal.get('strategy_name') or signal.get('strategy') or "unknown"`
  - Use `strategy_label` in the log instead of `sized_signal.get(..., strategy_name)`.
- Pros: Minimal change, no call-site updates, removes `NameError` while keeping helpful context when present.
- Cons: Still depends on upstream data being populated; falls back to `"unknown"` if absent (less explicit than passing the name).
- Tests: No signature changes; only log expectation snapshots (if any) could need updates.

**Option B: Pass `strategy_name` explicitly into `_assess_signal_risk`.**
- Update signature to `_assess_signal_risk(self, signal: Dict, strategy_name: Optional[str] = None)` and use the parameter in the `[QUALITY-BEFORE-RISK]` log (with a simple default like `"unknown"`).
- Update call sites:
  - `process_strategy_signal(...): await self._assess_signal_risk(enriched_signal, strategy_name)`
  - `process_signal(...): await self._assess_signal_risk(signal, strategy_name)` (the local `strategy_name` is already computed earlier in the method).
- Pros: Explicit contract that the function needs the strategy name; clearer logging; resilient even if signal dict is missing the field.
- Cons: Larger change surface (all call sites and any tests mocking `_assess_signal_risk` must be updated); slightly more boilerplate.
- Tests: Any direct calls in unit tests will need the new argument (or accept the optional default).

**Option C: Define a local `strategy_name` at the top of `_assess_signal_risk` (no signature change).**
- Add `strategy_name = signal.get('strategy_name') or signal.get('strategy') or "unknown"` before the log and keep using `sized_signal.get('strategy_name', strategy_name)`.
- Pros: Very small change; keeps current call sites untouched.
- Cons: Still relies on the signal carrying the field; less explicit than Option B; partially redundant because `sized_signal` already carries the same fields.
- Tests: Minimal risk aside from log expectations.

## 5. Recommendation
- Prefer **Option A** for the first fix: it is the smallest, removes the `NameError`, and keeps the log useful by leaning on the signal contents while providing a safe fallback.
- If we want stronger contracts around strategy metadata, follow up with **Option B** to make the dependency explicit and enforce passing the name from callers.
