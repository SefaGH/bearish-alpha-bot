# PnL Provider Wiring Analysis for `_can_dynamic_scale`

## 1) Wiring overview (text diagram)
- **LiveTradingEngine → PositionManager**  
  - `_monitor_positions()` calls `position_manager.monitor_position_pnl(position_id, current_price)`.  
  - Updates `PositionManager.positions[position_id]` with `current_price`, `unrealized_pnl`, `pnl_pct` (percentage), excursions, trailing stop, and appends to `pnl_tracker`.  
  - `[P&L-UPDATE]` logs are emitted from this path using `unrealized_pnl` and `pnl_pct` stored on the PositionManager-side positions.

- **RiskManager → PnL provider (PositionManager-backed)**  
  - RiskManager now accepts an injected `PositionPnlProviderProtocol`. ProductionCoordinator wires `RiskManager` to `PositionManager` via `PositionManagerPnlProvider(self.position_manager)`.  
  - `_can_dynamic_scale` calls `pnl_provider.get_positions_for_symbol(symbol, strategy_name, side)` to fetch positions for PnL aggregation, then reads `unrealized_pnl_pct` (or `metrics.unrealized_pnl_pct`) from those snapshots.

- **PortfolioManager**  
  - Still used for counts/limits (`count_open_positions`, `max_open_positions`, `max_positions_per_symbol`, portfolio heat); not a PnL source for scaling.  
  - `_can_dynamic_scale` no longer uses PortfolioManager for PnL aggregation; only for counts/limits.

## 2) Field mapping

| Layer                        | Structure / field path                          | Meaning / usage                                           |
| ---------------------------- | ----------------------------------------------- | --------------------------------------------------------- |
| PositionManager (runtime)    | `positions[pos_id]["unrealized_pnl"]`           | mark-to-market PnL (quote)                                |
| PositionManager (runtime)    | `positions[pos_id]["pnl_pct"]` (a.k.a. %P&L)    | PnL percentage used in `[P&L-UPDATE]` logging             |
| PositionManagerPnlProvider   | Iterates `position_manager.positions` by symbol | Builds snapshots per symbol                               |
| Provider PnL extraction      | `snapshot["unrealized_pnl_pct"]` else `metrics["unrealized_pnl_pct"]` | PnL % fed to `_can_dynamic_scale`                         |
| RiskManager `_can_dynamic_scale` | `avg_pnl_pct = mean(pnls_from_provider)`      | Compared to `min_scale_in_unrealized_pnl_pct`             |

**Mismatch observed:** PositionManager writes percentage under `pnl_pct` (and `unrealized_pnl`) but `PositionManagerPnlProvider` looks for `unrealized_pnl_pct` (or `metrics.unrealized_pnl_pct`). Unless some other code copies `pnl_pct` into `unrealized_pnl_pct`, provider sees no PnL and returns “unavailable/invalid.”

## 3) Why PnL was “unavailable/invalid” despite `[P&L-UPDATE] +0.4x%`
- In the reference run:  
  - Live PnL was computed in `monitor_position_pnl`, stored as `pnl_pct`, logged in `[P&L-UPDATE]`.  
  - `_can_dynamic_scale` called the provider; the provider searched for `unrealized_pnl_pct`/`metrics.unrealized_pnl_pct`.  
  - No such field existed on the position snapshot (only `pnl_pct` was present), so `pnl_values` list was empty → `_can_dynamic_scale` logged `PnL data unavailable/invalid ...` and rejected with `scale_in_pnl_data_unavailable`.
- This is a **field/path mismatch** (same PositionManager instance is wired, but the provider reads a key that is never set).

## 4) Light implementation plan (no code changes yet)

### Minimal fix outline (preferred)
- **Unify field name:**  
  - In `PositionManagerPnlProvider.get_positions_for_symbol`, when building snapshots, map `pnl_pct` to `unrealized_pnl_pct` if the latter is missing.  
  - Alternatively (or additionally), in `monitor_position_pnl`, set `position['unrealized_pnl_pct'] = pnl_pct` to make the field explicit for consumers.
- **Validate wiring:**  
  - Confirm the same PositionManager instance is injected into RiskManager via ProductionCoordinator (already done) and used by LiveTradingEngine.  
  - No change to PortfolioManager usage for counts/limits.
- **Logging sanity:**  
  - Add a debug-level log in the provider when PnL is missing to surface field mismatches during tests (optional).

### Patch sketch (conceptual)
- File: `src/core/position_manager.py` (provider class)  
  - In `get_positions_for_symbol`, before returning snapshots:  
    - If `"unrealized_pnl_pct"` not in snapshot and `"pnl_pct"` in snapshot: `snapshot["unrealized_pnl_pct"] = snapshot["pnl_pct"]`.
- (Optional) File: `src/core/position_manager.py` (`monitor_position_pnl`)  
  - After computing `pnl_pct`, set `position["unrealized_pnl_pct"] = pnl_pct` to keep the field present at source.
- File: `src/core/risk_manager.py` (`_can_dynamic_scale`)  
  - Keep current logic; with the above mapping, `pnl_values` will be populated and `avgPnL` will match `[P&L-UPDATE]`.

### Validation steps (post-fix)
- Trigger a scale-in when `[P&L-UPDATE]` shows a non-zero PnL; expect `[RISK-SCALING] ... avgPnL=+0.4x%` (matching the last P&L update) instead of “PnL data unavailable.”
- Unit tests: extend/mimic provider snapshots containing `pnl_pct` only and assert `_can_dynamic_scale` uses it via the mapping.

## 5) PositionManager vs PortfolioManager as PnL source
- Given the current architecture, PositionManager should be the single source of truth for mark-to-market PnL (it owns price updates and emits `[P&L-UPDATE]`). PortfolioManager should remain responsible for counts/limits/heat, not PnL snapshots. The minimal field-mapping fix above preserves this separation while ensuring RiskManager sees the same PnL that PositionManager computes.
