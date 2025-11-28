# Fix Summary: Issue #436 - Log & Signal Explainability

## Overview
Implemented comprehensive signal quality scoring and structured logging to enhance observability and explainability of trading signals.

## Changes Implemented

### 1. Quality Calculation (`src/quality/quality_calculator.py`)
- Validated `compute_quality` function.
- Calculates a 0.0-1.0 score based on:
  - **ML Component** (60% weight)
  - **Volume** (20% weight)
  - **Momentum** (15% weight)
  - **Spread** (5% weight)
- Provides detailed breakdown and reasons for low scores.

### 2. Strategy Coordinator (`src/core/strategy_coordinator.py`)
- Updated `emit_signal_breakdown` to:
  - Log structured JSON `signal_breakdown` events.
  - **Alert** (Warning log) when `quality_score` is 0.0.
- Integrated `compute_quality` into `process_strategy_signal` pipeline.

### 3. Position Manager (`src/core/position_manager.py`)
- Updated `_extract_entry_metadata` to capture `quality_score` and `quality_breakdown` from signals.
- Updated `open_position` to persist quality metrics in active position state.
- Updated `close_position` to include quality metrics in `TRADE_CLOSED` logs for historical analysis.

## Verification
- Created unit tests in `tests/unit/test_issue_436.py`.
- Verified:
  - Quality score calculation logic and fallbacks.
  - Alert triggering on zero quality.
  - Metadata extraction and persistence in Position Manager.
- All tests passed.

## Next Steps
- Monitor `signal_breakdown` logs in ELK/CloudWatch.
- Analyze `quality_score` distribution to tune weights if necessary.
