# Pyramiding Pre‑flight & Calibration (Paper Mode)

Operational checklist for safely enabling and tuning pyramiding (multi-layer scale-ins) in **paper trading**. No live deployment until all Go/No-Go criteria are green.

## Overview
- Pyramiding here: multiple scale-in layers per symbol using `intent=scale_in`, governed by StrategyCoordinator intent classification → duplicate soft guard (for scale_in) → RiskManager dynamic scaling + pyramiding caps → intent-aware queue → LiveTradingEngine execution.
- Default state is shadow/off (`pyramiding.enabled=false`); enabling requires explicit config.
- Goal: confirm configs, tests, and paper-mode scenarios before any live use.

## Config Sanity Checklist
1) **Pyramiding block**
   - Ensure block exists; keep `pyramiding.enabled=false` until tests pass.
   - Set conservative caps: `max_layers_per_symbol` (recommend 2–3).
   - Threshold overlays used by RiskManager for scale-ins: `min_scale_in_quality`, `min_scale_in_unrealized_pnl_pct`, `min_scale_in_distance_pct`.
2) **Risk concurrent limits & dynamic scaling**
   - `risk.concurrent_limits.max_positions_per_symbol`
   - `risk.concurrent_limits.dynamic_scaling.enabled`
   - `risk.concurrent_limits.dynamic_scaling.max_additional_positions`
   - `risk.concurrent_limits.dynamic_scaling.quality_threshold`
   - `risk.concurrent_limits.dynamic_scaling.min_unrealized_pnl_pct`
   - `risk.concurrent_limits.dynamic_scaling.min_distance_pct` (or code default)
   - Interaction: effective slots = min(`max_positions_per_symbol + max_additional_positions`, `pyramiding.max_layers_per_symbol` when enabled). Threshold overlays come from `pyramiding.*` when intent=scale_in.
3) **Queue limits**
   - `risk.queue.max_pending_per_symbol` (entry/reentry cap, always strict)
   - `risk.queue.max_pending_scale_in_per_symbol` (extra scale-in pending slots only when pyramiding enabled)
   - Effect: at most 1 pending entry per symbol; limited extra pending scale_ins when enabled.
4) **Global risk constraints**
   - `risk.concurrent_limits.max_total_risk_pct`, daily trade limits, circuit breakers/daily loss limits.
   - These can still block scale-ins even if local checks pass.

Example conservative paper-mode snippet (do **not** replace defaults; use as override template):
```yaml
pyramiding:
  enabled: true
  max_layers_per_symbol: 2
  min_scale_in_quality: 0.82
  min_scale_in_unrealized_pnl_pct: 0.006   # 0.6%
  min_scale_in_distance_pct: 0.005         # 0.5%
risk:
  concurrent_limits:
    max_positions_per_symbol: 1
    dynamic_scaling:
      enabled: true
      quality_threshold: 0.8
      min_unrealized_pnl_pct: 0.005
      min_distance_pct: 0.005
      max_additional_positions: 2
  queue:
    max_pending_per_symbol: 1
    max_pending_scale_in_per_symbol: 2
```

### Central Config + Azure App Config / ENV
- Keys (YAML path → Azure key):  
  - `pyramiding.enabled` → `BearishAlphaBot/pyramiding.enabled` (ENV: `PYRAMIDING_ENABLED`)  
  - `pyramiding.max_layers_per_symbol` → `BearishAlphaBot/pyramiding.max_layers_per_symbol` (ENV: `PYRAMIDING_MAX_LAYERS_PER_SYMBOL`)  
  - `pyramiding.min_scale_in_quality` → `BearishAlphaBot/pyramiding.min_scale_in_quality` (ENV: `PYRAMIDING_MIN_SCALE_IN_QUALITY`)  
  - `pyramiding.min_scale_in_unrealized_pnl_pct` → `BearishAlphaBot/pyramiding.min_scale_in_unrealized_pnl_pct` (ENV: `PYRAMIDING_MIN_SCALE_IN_PNL`)  
  - `pyramiding.min_scale_in_distance_pct` → `BearishAlphaBot/pyramiding.min_scale_in_distance_pct` (ENV: `PYRAMIDING_MIN_SCALE_IN_DISTANCE`)  
  - `risk.queue.max_pending_scale_in_per_symbol` → `BearishAlphaBot/risk.queue.max_pending_scale_in_per_symbol` (ENV: `SIGNAL_QUEUE_MAX_PENDING_SCALE_IN_PER_SYMBOL`)
- Paper label example (Azure App Config label `paper`):  
  - `pyramiding.enabled = true`  
  - `pyramiding.max_layers_per_symbol = 2`  
  - `risk.queue.max_pending_scale_in_per_symbol = 1`
- Verification: on startup, check “Pyramiding Settings” section in logs; values should match intended overrides. Defaults remain production-safe (`enabled=false`).

## Test Suite Checklist
Run before and after toggling pyramiding to true in paper config:
- `pytest tests/unit/test_issue_103_duplicate_prevention.py` (duplicate baseline)
- `pytest tests/unit/test_duplicate_intents.py` (intent constants/plumbing)
- `pytest tests/unit/test_duplicate_scale_in.py` (scale_in soft guard behavior)
- `pytest tests/unit/test_intent_classification.py` (intent classification helper)
- `pytest tests/unit/test_risk_pyramiding.py` (RiskManager dynamic scaling + pyramiding caps)
- `pytest tests/unit/test_pyramiding_queue.py` (queue intent-aware pending limits)
- Optional: integration/smoke of trading loop in sandbox.
All must pass before proceeding.

## Paper-Mode Scenario Checklist
Run with `pyramiding.enabled=true` in paper config.

1) **Single symbol, trend-up (positive PnL pyramiding)**
   - Preconditions: strategy emits additional buy signals while first layer in profit; quality high; distance meets threshold.
   - Watch logs: `[PYRAMID] scale-in allowed/rejected`, `[PYRAMID-QUEUE]`, duplicate spam logs.
   - Pass: scale-in attempts reach RiskManager; accepted when quality/PnL/distance met; queue holds limited pending scale_ins.
2) **Single symbol, noisy/high-frequency signals**
   - Aim: ensure spam-window in duplicate rejects ultra-fast, tiny-delta scale-ins; queue caps per symbol.
   - Watch: duplicate spam rejections vs risk/queue rejections; pending counts.
   - Pass: spammy repeats rejected; non-spam scale-ins proceed to risk; queue does not grow unbounded.
3) **Multi-symbol portfolio**
   - Aim: confirm global limits (max_open_positions, portfolio heat) block scale-ins when needed.
   - Watch: risk logs indicating global caps; queue/duplicate not primary blockers in these cases.
   - Pass: clear risk-based rejections when caps hit; no unexplained drops.

For each scenario, if expected acceptances/rejections differ, adjust config (quality/PnL/distance/layers/queue caps) and rerun.

## Log Inspection & Calibration Guide
Key log tags/patterns:
- `[PYRAMID] scale-in allowed` – dynamic scaling accepted; shows slots, quality, PnL, distance.
- `[PYRAMID] scale-in rejected by risk` or specific reason strings (`scale_in_quality_below_threshold`, `scale_in_pnl_below_threshold`, `scale_in_distance_below_threshold`, `pyramiding_max_layers_reached`).
- `[PYRAMID-QUEUE] scale-in enqueued/rejected` – per-symbol pending logic.
- Duplicate spam/cooldown warnings with `intent=scale_in` and reasons (`duplicate_scale_in_spam_window`).

Calibration loop:
- Many rejections for quality → consider lowering `min_scale_in_quality` slightly or improve quality computation.
- Many rejections for PnL → adjust `min_scale_in_unrealized_pnl_pct` or strategy timing.
- Distance too small → lower `min_scale_in_distance_pct` cautiously; consider strategy-specific tuning.
- Queue rejections frequent → review `max_pending_scale_in_per_symbol`; increase slightly if justified, else keep tight.
- Layer cap reached often → raise `max_layers_per_symbol` only after verifying risk tolerance and outcomes.

Optional helper: see `scripts/analyze_pyramiding_logs.py` below for quick counts.

## Optional Helper: Log Analysis
Script: `scripts/analyze_pyramiding_logs.py`
- Purpose: parse log files for `[PYRAMID]` and `[PYRAMID-QUEUE]` lines; report counts of attempts, accepted, rejected by reason, and queue rejections.
- Usage example:
  ```bash
  python scripts/analyze_pyramiding_logs.py --files logs/run.log
  ```
- Output: summary counts (attempts/allowed/rejected by reason; queue rejections by intent).
Use to spot dominant rejection reasons and tune thresholds accordingly.

## Startup Verification (Paper Mode)
- After applying overrides (Azure label or ENV), start in paper mode and check logs:
  - “Pyramiding Settings:” section should show `Enabled=true`, expected `max_layers_per_symbol`, and `max_pending_scale_in_per_symbol`.
  - Risk/ML summaries should still load successfully (sanity check).
- Local smoke check note: short paper-mode start with `pyramiding.enabled=true`, small caps (e.g., layers=2, max_pending_scale_in=1) confirmed that startup summary prints intended values and no init errors. No live trades executed in this check.

## Go / No-Go Criteria
- ✅ All unit tests (duplicate/intent/risk/queue/pyramiding) passing.
- ✅ Paper-mode scenarios (trend, noisy, multi-symbol) run with expected outcomes; no unexplained rejections or unbounded pending signals.
- ✅ Conservative parameters: small `max_layers_per_symbol`, reasonable quality/PnL/distance thresholds, tight queue limits for scale_in.
- ✅ Logs show healthy distribution of attempts vs accepted vs rejected; no signs of uncontrolled risk growth or queue starvation.

Proceed to extended paper runs only when all above are green; live experiments should follow only after stable paper results.
