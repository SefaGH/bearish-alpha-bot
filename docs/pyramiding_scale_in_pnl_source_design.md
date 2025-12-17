# PnL Source Design for `_can_dynamic_scale`

## Root cause (recap)
- Live P&L is updated inside `LiveTradingEngine._monitor_positions()` via `PositionManager.monitor_position_pnl()`, which mutates `PositionManager.positions[position_id]` with `unrealized_pnl`, `unrealized_pnl_pct`, etc. The `[P&L-UPDATE]` log reflects this store.
- `_can_dynamic_scale` pulls positions from `PortfolioManager` (`get_open_positions_for_symbol` / `get_open_positions`) and looks for `unrealized_pnl_pct` (or `metrics.unrealized_pnl_pct`). These snapshots are not refreshed by the monitoring loop, so missing PnL fields default to `0.0`, producing `avgPnL=0.00%` while live PnL logs show profit.

## Current PnL data flow
1) **LiveTradingEngine → PositionManager**
   - `_monitor_positions` iterates `self.active_positions` (engine) and calls `position_manager.monitor_position_pnl(position_id, current_price)`.
   - `monitor_position_pnl` updates `PositionManager.positions[position_id]`: `current_price`, `unrealized_pnl`, `unrealized_pnl_pct`, excursions, trailing stops; records snapshots in `pnl_tracker`; returns `pnl_pct`.
   - Logs: `💰 [P&L-UPDATE] ... (+X.XX%)` come from these PositionManager-side updates.

2) **PortfolioManager snapshots**
   - Holds `active_positions` and exposes `get_open_positions` / `get_open_positions_for_symbol` (returns shallow copies).
   - These positions typically lack live `unrealized_pnl_pct` because PortfolioManager is not synced with PositionManager’s updates.

3) **RiskManager `_can_dynamic_scale`**
   - Fetches positions via PortfolioManager getters.
   - For each position: `pnl_val = pos.get('unrealized_pnl_pct')` else `pos.get('metrics', {}).get('unrealized_pnl_pct')`; on failure, `0.0`.
   - Averages these values → `avg_pnl_pct`; compares to `min_scale_in_unrealized_pnl_pct`. Missing data → `avgPnL=0.00%`.

## PortfolioManager responsibilities (relevant to scale-in)
- State summary: open-position counts, per-symbol counts, max_open_positions, max_positions_per_symbol, portfolio heat/drawdown caps.
- Not a reliable mark-to-market source today: does not ingest/refresh `unrealized_pnl_pct` from PositionManager.
- For scale-in, it should remain the place for counts/limits, but PnL gating needs a canonical mark-to-market source.

## Position matching constraints
- Stable identifiers: `position_id` (best), plus `symbol` and possibly `strategy_name`/`side`.
- Edge cases: multiple positions per symbol (pyramiding layers), multiple strategies on same symbol, partial closes/status changes. Matching by `position_id` avoids ambiguity; matching by `symbol` alone is lossy when layers exist.

## Option A – RiskManager reads PnL directly from PositionManager
- Sketch:
  - `_can_dynamic_scale` (or a helper) asks a PositionManager/PnL view for open positions for a symbol (optionally strategy/side).
  - Reads `unrealized_pnl_pct` from that source, computes `avgPnL`.
  - PortfolioManager still supplies counts/limits/heat; PnL gating uses PositionManager as the single source.
- Coupling mitigation:
  - Define a lightweight interface (e.g., `PnLProvider` with `get_open_positions_with_pnl(symbol, strategy=None)`), implemented by PositionManager and injected into RiskManager.
- Pros: Always up-to-date PnL; no reliance on snapshot sync; clear ownership of mark-to-market.
- Cons: Introduces a new dependency from RiskManager to PositionManager (or an interface); needs plumbing/injection; must ensure current prices are reflected or accessible at decision time.
- Impact surface: `_can_dynamic_scale`, RiskManager init/wiring, tests to mock PnL provider; PortfolioManager still used for counts/limits.

## Option B – Canonical PnL sync: PositionManager → PortfolioManager
- Sketch:
  - On each `monitor_position_pnl` (or in a dedicated sync hook), push `unrealized_pnl_pct` (and optionally a timestamp) into `PortfolioManager.active_positions[position_id]` (or shared object).
  - `_can_dynamic_scale` continues to read from PortfolioManager, but now the PnL fields are the same ones `[P&L-UPDATE]` logs.
- Pros: Minimal change to RiskManager logic; keeps RiskManager dependent only on PortfolioManager; centralizes state for other consumers.
- Cons: Requires reliable synchronization (shared objects or explicit copy); risk of staleness if sync fails; adds coupling from PositionManager to PortfolioManager.
- Impact surface: PositionManager update path, PortfolioManager storage, possibly LiveTradingEngine to orchestrate the sync; tests around PnL sync and scale-in gating.

## Option C (supplementary) – Recency guard
- Enforce `min_scale_in_unrealized_pnl_pct` only if PnL data is recent (timestamp within N seconds or N monitoring ticks); otherwise skip/allow or re-evaluate after refresh.
- Pros: Reduces false negatives from stale data.
- Cons: Does not fix dual-store inconsistency; requires timestamp propagation; adds a new condition to scaling.

## Recommendation
- Primary: Option A (PositionManager as PnL source via a small injected interface). It directly uses the authoritative mark-to-market data that drives `[P&L-UPDATE]`, eliminating sync gaps, while keeping PortfolioManager for counts/heat.
- Secondary: If architectural constraints favor PortfolioManager-only consumption, adopt Option B with a clear, robust sync from PositionManager to PortfolioManager and consider adding a recency guard (Option C) to avoid stale reads.***
