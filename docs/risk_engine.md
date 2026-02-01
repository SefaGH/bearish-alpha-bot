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
- **Config priority:** `min_stop_pct` read as ENV `RISK_MIN_STOP_PCT` → YAML → default `0.001`. Values ≤1 are treated as decimals (0.001 = 0.1%); values >1 are treated as percentages and divided by 100 (e.g., 50 → 0.5).
- **Per-trade risk pct:** `risk.per_trade_risk_pct` is a fraction end-to-end (0.01 = 1%). Operators can supply 1 or 2 in YAML/App Config; `LiveTradingConfiguration` normalizes values >1 to fractions (1 → 0.01). `RiskConfiguration` now consumes the fraction as-is for USD caps.
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
 - `risk.max_notional_pct_per_trade`: raw multiplier that derives `computed_max_notional_usd` (e.g., `0.75` or `10.0`)

Example key-value list (App Config):
```
capital_usdt = 1000
per_trade_risk_pct = 1
risk.min_stop_pct = 0.5
max_position_notional_usd = 250
risk.max_notional_pct_per_trade = 0.75
risk.size_planner_enabled = true
```

**Max notional precedence:**
- If `max_position_notional_usd` is provided (YAML/ENV/App Config), it is used directly.
- Else, if `computed_max_notional_usd` exists (from `risk.max_notional_pct_per_trade * equity_usd`), it is used.
- Else, no clamp is applied (default None).

Balanced preset (equity 500):

| Param | Value | USD |
|---|---|---|
| `risk.equity_usd` | 500 | — |
| `risk.per_trade_risk_pct` | 0.003 (0.3%) | `max_risk_per_trade_usd` = 1.50 |
| `risk.max_portfolio_risk_pct` | 0.06 (6%) | `max_portfolio_risk_usd` = 30.0 |
| `risk.daily_loss_limit_pct` | 0.02 (2%) | `daily_loss_limit_usd` = 10.0 |
| `risk.max_position_size` | 0.25 (25%) | `max_position_notional_usd` = 125.0 |
| `risk.min_stop_pct` | 0.001 (0.1%) | stop floor applied in sizing |

Example (sanity check, equity 100):

| Input | Value | Resulting USD |
|---|---|---|
| `risk.equity_usd` | 100 | — |
| `risk.per_trade_risk_pct` | 0.01 (or 1 via env/App Config) | `max_risk_per_trade_usd` = 1.00 |
| `risk.daily_loss_limit_pct` | 0.02 | `daily_loss_limit_usd` = 2.00 |
| `risk.max_drawdown` | 0.10 | `max_drawdown_usd` = 10.00 |

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

---

# Option B (Sprint 2 Plan) – Size Planner in RiskManager

**Status:** Planned, gated by `RISK_SIZE_PLANNER_ENABLED`. When the flag is **false**, Sprint 1 behavior applies. When **true**, the Size Planner becomes active. A shadow mode can run the planner for logs/metrics only before activation.

Supported keys to enable planner:
- ENV: `RISK_SIZE_PLANNER_ENABLED=true|false`
- App Config: `risk.size_planner_enabled=true|false` (preferred nested key)

## End-to-end pipeline (planner enabled)
1) **Config inputs:** `risk.per_trade_risk_pct` (fraction), `risk.risk_usd_cap`, `risk.max_position_size`, `max_position_notional_usd` or `risk.max_notional_pct_per_trade`, `risk.min_stop_pct`, `risk.min_notional_threshold`, `risk.max_portfolio_risk_usd`, `equity_usd`, `available_balance`, `leverage`, `price`.
2) **APS (stop/risk/volatility expert):**
	- `computed_max_risk_usd = equity * per_trade_risk_pct`
	- `effective_risk_usd = min(computed_max_risk_usd, risk_usd_cap)` (cap optional)
	- `effective_stop_pct = max(requested_stop_pct_after_vol_multipliers, min_stop_pct)`
	- `raw_notional = effective_risk_usd / effective_stop_pct` (APS may output `< min_notional_threshold`; it does **not** reject on that)
3) **Size Planner (RiskManager):**
	- Caps: `cap_size_pct = equity * max_position_size`; `cap_notional = max_position_notional_usd` or derived `equity * risk.max_notional_pct_per_trade` or ∞; `cap_capital = compute_max_affordable_notional(available_balance, leverage, 0.95)` (same logic as CapitalLimitRule); `cap_heat = max_portfolio_risk_usd - current_open_risk_usd` (∞ if disabled)
	- `planned_notional = min(raw_notional, cap_size_pct, cap_notional, cap_capital, cap_heat)`
	- If `planned_notional < min_notional_threshold`: early reject with clear reason (`REJECT_TOO_SMALL_AFTER_CAP` when heat not binding; `portfolio_heat_exhausted` when heat binds)
	- `planned_qty = planned_notional / price` (exchange normalization happens later)
4) **Risk rules:** consume `planned_notional`; PositionSizeRule expected quiet; PortfolioHeatRule acts as guard-rail using the same portfolio heat helper.
5) **Auto-resize:** only for capital/margin failures; no size%-based auto-resize. Optional broker-min retry can be considered separately.

## `position_size_policy` behavior with planner
- Config key: `risk.position_size_policy` (allowed: "clip", "reject"; default "clip" if unset).
- "clip" (default): planner clips to the tightest cap (`planned_notional = min(...)`); accepts if ≥ `min_notional_threshold`.
- "reject": if a **size-driven** cap binds (size_pct or max_notional), planner rejects instead of clipping (`reason="REJECT_SIZE_CAP"`). Capital and heat caps remain safety caps and still clip by default. If `planned_notional < min_notional_threshold`, reject per min-notional rules.

## Canonical portfolio heat
- `current_open_risk_usd` is computed via a shared helper (e.g., `compute_portfolio_open_risk_usd()`), summing per-position risk in USD using the same definition as PortfolioHeatRule (risk = position size × |entry - stop|, aligned with APS risk semantics).
- Both the planner and PortfolioHeatRule must call this helper; no divergent calculations.
- `cap_heat = max_portfolio_risk_usd - current_open_risk_usd`; if `cap_heat <= 0`, planner rejects with `capped_by_heat=True`, `reason="portfolio_heat_exhausted"` even if other caps pass.

## Invariants (planner vs APS)
- APS: `effective_risk_usd = min(equity * per_trade_risk_pct, risk_usd_cap?)`; `raw_notional = effective_risk_usd / effective_stop_pct`; APS never increases risk beyond `effective_risk_usd` and does not enforce `min_notional_threshold`.
- Planner: never increases `raw_notional`; always `planned_notional <= raw_notional`, `<=` size cap, `<=` max_notional cap, `<=` capital cap (shared with CapitalLimitRule), `<=` heat cap (if enabled). Planner enforces `min_notional_threshold` post-caps.

## Capital cap alignment
- `compute_max_affordable_notional(available_balance, leverage, safety_factor=0.95)` must match the CapitalLimitRule formula exactly (same balance source, leverage handling, safety factor). Planner and rule share this helper to stay consistent.

## Example: capital-bound scenario
- Inputs: equity=100, per_trade_risk_pct=0.01, risk_usd_cap unset, effective_stop_pct=0.005 → APS `raw_notional=200`; leverage=3; available_balance=40; max_position_size set high (e.g., 200%); max_position_notional_usd unset; heat disabled.
- Caps: `cap_size_pct=200`, `cap_notional=∞`, `cap_capital=40*3*0.95=114`, `cap_heat=∞` → planner picks 114 (capital binds); `planned_qty=114/price`.

## Out of scope (Option B)
- Planner does **not** handle exchange microstructure: min order/notional, qty step size, tick size, rounding. Exchange adapters remain responsible for normalizing `planned_qty`/`planned_notional` to broker constraints.
- Order manager/exchange adapter will round/normalize the planner output; risk logs refer to pre-normalized planner values. Any material drift should be handled by the adapter (future enhancement if needed).

## Rollout / feature flag
- `RISK_SIZE_PLANNER_ENABLED`: `false` → Sprint 1 behavior (planner optional shadow logging). `true` → planner active as described. Shadow mode logs `size_planner.decision` (including deltas) and metrics without changing live behavior; flip after acceptable observation.

## Observability (planner)
- Structured log `size_planner.decision` includes: symbol, equity, raw_notional, planned_notional, price, planned_qty, cap flags, heat_remaining_usd, max_portfolio_risk_usd, below_min_notional, position_size_policy, reason, and delta fields `notional_delta_abs`, `notional_delta_ratio`, plus `shadow_mode`.
