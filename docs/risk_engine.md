# Risk Engine (Sprint 1) Overview

## Goals
- Prevent oversized positions from microscopic stops (stop floor).
- Enforce deterministic sizing → clip → rules → optional auto-resize flow.
- Provide safe rejection paths (min notional, sizing errors) without crashes.

## Flow
1) **Position sizing** (stop floor applied).
2) **Clip to limits** (max notional, min notional, policy clip/reject).
3) **Risk rules** (capital, position size, heat, R/R, etc.).
4) **Optional auto-resize** on capital/margin errors (single attempt, clamped to max notional). 

## Key Mechanics
- **Stop floor:** `effective_stop = max(raw_stop, entry * min_stop_pct)`; sets `floor_triggered` flag.
- **Min notional:** sizing raises `ValueError` if proposed notional < `min_notional_threshold`; RiskManager returns blocked result.
- **Config priority:** `min_stop_pct` read as ENV `RISK_MIN_STOP_PCT` → YAML → default `0.005`. Values ≤1 are treated as decimals (0.005 = 0.5%); values >1 are treated as percentages and divided by 100 (e.g., 50 → 0.5).
- **Resize tracking:** deterministic key (symbol/entry/stop/qty/leverage) prevents multiple retries per position.
- **Auto-resize clamp:** affordable notional is clamped by `max_position_notional_usd`; marks `resize_failed` when balance is too low.
- **Health check:** `run_health_check()` validates critical config (min_stop_pct >0, max_position_notional_usd, min_notional_threshold, max_risk_per_trade_usd, etc.) and logs HEALTHY/UNHEALTHY.

## Logging & Metadata
- Sizing meta: `raw_stop_pct`, `effective_stop_pct`, `floor_triggered`, `min_stop_pct`, `original_notional`, `proposed_notional`.
- Limits meta: `action` (clip/reject/none), `final_notional`, `allowed_notional`, `min_notional`, `policy`.
- Resize meta: `max_affordable`, `available_balance`, `used_notional`, `clipped_after_resize`, `resize_failed` when not salvageable.

## Recommended overrides for Azure App Config
Use these keys/values when setting cloud overrides (App Config applies last after ENV and YAML):

- `capital_usdt`: base equity in USD (e.g., `1000`)
- `per_trade_risk_pct`: per-trade risk percent (e.g., `1` for 1%)
- `risk.min_stop_pct`: stop floor (e.g., `0.5` → 0.5%; values >1 are treated as percent/100)
- `max_position_notional_usd`: hard cap per position in USD (set explicit number, e.g., `250`)

Example key-value list (App Config):
```
capital_usdt = 1000
per_trade_risk_pct = 1
risk.min_stop_pct = 0.5
max_position_notional_usd = 250
```

## Tests (coverage snapshot)
- Stop floor notional (~200 USD band on tight stop).
- Clip-before-rules ordering.
- Auto-resize success and failure paths.
- Min-notional rejection from sizing.
- Config priority (ENV → YAML → default) for `min_stop_pct`.
- Tight-stop regression scenario (11:30-style) passes with stop floor.

## Notes / gaps
- `min_notional_threshold` is currently internal (default 5.0). It is not exposed via YAML/ENV/App Config; it can only be changed via code/custom_limits. If external tuning is needed, expose as `RISK_MIN_NOTIONAL_THRESHOLD` / `risk.min_notional_threshold` in a future sprint.

## Suggested Next (non-blocking)
- Emit telemetry/metrics for floor triggers, clips, auto-resize attempts/outcomes.
- Expose health/metrics endpoint for external monitoring.
