# Pyramiding Scale-In PnL Flow Analysis

## Summary of the discrepancy
- LiveTradingEngine’s monitoring loop logs positive unrealized P&L for BTC/USDT (e.g., +0.33%) immediately before a scale-in attempt.
- `_can_dynamic_scale` in `RiskManager` logs `avgPnL=0.00%` and rejects with `scale_in_pnl_below_threshold`.
- Root cause: `_can_dynamic_scale` pulls open positions from `PortfolioManager` (`get_open_positions_for_symbol` / `get_open_positions`), whose snapshots **do not carry the live `unrealized_pnl_pct` updates** coming from `PositionManager` inside LiveTradingEngine. Missing PnL fields default to `0.0`, so `avgPnL` collapses to zero even while the live P&L loop shows profit.

## PnL data flow (current)
1. **Monitoring / update source (LiveTradingEngine + PositionManager)**
   - `LiveTradingEngine._monitor_positions()` iterates `self.active_positions` (internal to the engine) and calls `position_manager.monitor_position_pnl(position_id, current_price)`.
   - `monitor_position_pnl` (PositionManager) updates in-place on its own `self.positions[position_id]`:
     - `current_price`, `unrealized_pnl`, `unrealized_pnl_pct`, excursions, trailing stops.
   - Logs: `💰 [P&L-UPDATE] ... P&L: $... (+X.XX%)` reflect these PositionManager-side fields.
   - This path does **not** update `PortfolioManager.active_positions`.

2. **Position storage consumed by RiskManager**
   - `_can_dynamic_scale` fetches positions via `portfolio_manager.get_open_positions_for_symbol(symbol)` or `get_open_positions()`.
   - It inspects each position dict for `unrealized_pnl_pct`, else `metrics.unrealized_pnl_pct`; missing/parse errors become `0.0`.
   - `avg_pnl_pct` = mean of those values; if `< min_scale_in_unrealized_pnl_pct`, scale-in is denied.

3. **Resulting mismatch**
   - `PortfolioManager.active_positions` snapshots lack `unrealized_pnl_pct` updates because live P&L updates occur on a separate `PositionManager` store inside LiveTradingEngine.
   - Thus `_can_dynamic_scale` sees empty/zero PnL values and logs `avgPnL=0.00%`, while the monitoring loop shows a positive P&L from the other store.

## Ordering considerations
- The monitoring loop runs on its own cadence inside LiveTradingEngine, updating only PositionManager positions.
- Scale-in validation via RiskManager reads PortfolioManager snapshots that are not refreshed by that loop.
- Even if timing were aligned, the two stores are not synchronized, so the PnL seen by `_can_dynamic_scale` can remain stale/zero.

## Design fix options (no code changes yet)

### Option A: Compute PnL on the fly inside `_can_dynamic_scale`
- For scale-in checks, derive unrealized PnL per position from `entry_price`, `amount`, and **latest price** (via PortfolioManager/MarketDataPipeline or PositionManager) instead of relying on stored `unrealized_pnl_pct`.
- Pros: Single-source calculation at decision time; no reliance on stale snapshots.
- Cons: Requires access to a reliable current price; introduces pricing dependency into RiskManager; needs careful handling of multiple positions/sides.

### Option B: Align on a canonical PnL field/source
- Ensure `PortfolioManager.active_positions` carries and refreshes `unrealized_pnl_pct` (and/or `metrics.unrealized_pnl_pct`) from the same PositionManager updates that drive `[P&L-UPDATE]`.
- Have `_can_dynamic_scale` read that canonical field only.
- Pros: Minimal change to scaling logic; keeps RiskManager simple; consistent with monitoring logs.
- Cons: Requires syncing PositionManager updates into PortfolioManager (shared objects or explicit refresh); needs wiring in the execution layer.

### Option C: Guard scale-in checks on PnL recency (supplementary)
- Track a “last PnL update” timestamp per position and require a recent update before enforcing `min_scale_in_unrealized_pnl_pct`; otherwise, skip/allow or re-evaluate after refresh.
- Pros: Avoids false negatives when data is stale.
- Cons: Does not solve the underlying dual-store issue; needs timestamp plumbing.

## Recommendation
- Treat the zero PnL as a data-source mismatch: `_can_dynamic_scale` should either compute PnL directly (Option A) or consume the same updated fields that the monitoring loop maintains (Option B). Either approach ensures the scale-in gate evaluates the real unrealized PnL seen in `[P&L-UPDATE]` logs. Option C can supplement but should not be the primary fix.
