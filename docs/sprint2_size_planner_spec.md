# Sprint 2 Spec Stub: Option B Size Planner

## Scope
- Implement Option B Size Planner inside `RiskManager`, behind feature flag `RISK_SIZE_PLANNER_ENABLED`.
- Add `plan_position_size` helper; move canonical `min_notional_threshold` enforcement into the planner.
- Wire planner into existing `size_and_validate_position` / `validate_new_position` flow.
- Keep AdvancedPositionSizing as the stop/risk/volatility engine; planner only caps.

## Non-goals
- No changes to exchange adapter normalization (lot size, tick size, min notional); still adapter responsibility.
- No new auto-resize strategies beyond current capital/margin retry.

## API contract
- `PlannedSizeResult`: `planned_notional`, `planned_qty`, `capped_by_size_pct`, `capped_by_max_notional`, `capped_by_capital`, `capped_by_heat`, `below_min_notional`, `reason`.
- `plan_position_size` inputs: `raw_notional`, `symbol`, `equity`, `risk_limits` (including `max_position_size`, `max_position_notional_usd`, `position_size_policy`, `min_notional_threshold`), `available_balance`, `leverage`, `price`, `current_open_risk_usd`, `max_portfolio_risk_usd`, `risk.min_notional_threshold`.
- Invariants: planner never increases `raw_notional`; never overrides APS risk cap; enforces `min_notional_threshold` post-caps; shares `compute_max_affordable_notional` with CapitalLimitRule; uses shared `compute_portfolio_open_risk_usd` for heat.

## Rollout
- `RISK_SIZE_PLANNER_ENABLED`: `false` → Sprint 1 behavior (planner may run in shadow for logs/metrics). `true` → planner active.
- Shadow mode: run planner, emit `size_planner.decision` logs/metrics; do not alter live notional. Flip to active after observation window.

## Test matrix (high level)
- Tight-stop small account: capped but accepted when above `min_notional_threshold`.
- Too-small account: early rejection with `reason="notional_below_min_threshold"` (or heat-specific when heat binds).
- Heat exhaustion scenario: rejects with `capped_by_heat=True`, `reason="portfolio_heat_exhausted"`.
- Normal account: planner is a no-op; PositionSizeRule remains quiet.
- Capital-bound scenario: capital cap binds and planner emits capped notional consistent with CapitalLimitRule.
