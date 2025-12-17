**Producer structure (PositionManager)**
- `AdvancedPositionManager.positions`: dict keyed by `position_id`.
- `open_position` seeds PnL fields with `unrealized_pnl: 0.0` only (no `%` fields).
- `monitor_position_pnl` updates `unrealized_pnl` and computes `pnl_pct` (percent form, e.g., `0.44` -> `+0.44%` log). The tracker appends `{'timestamp', 'price', 'unrealized_pnl', 'pnl_pct'}` snapshots.
- Before the fix, that percent never lived on the position dict; neither `pnl_pct` nor `unrealized_pnl_pct` nor a `metrics` entry was persisted.

**Consumer expectations**
- `PositionManagerPnlProvider.get_positions_for_symbol` walks `position_manager.positions` and populates `unrealized_pnl_pct` on snapshots. `RiskManager._can_dynamic_scale` then reads `pos['unrealized_pnl_pct']` (or `metrics['unrealized_pnl_pct']`) and drops positions with missing/invalid values, logging `scale_in_pnl_data_unavailable`.
- Pyramiding thresholds in `RiskManager` are fractional (e.g., `0.005` = `0.5%`) and are compared against the provider-sourced `unrealized_pnl_pct`.

**Exact mismatch**
- Live loop logs `[P&L-UPDATE] ... (+0.44%)` because `monitor_position_pnl` returns the percent value, but the position object only carried `unrealized_pnl`. When the provider queried that position, there was no `unrealized_pnl_pct`/`pnl_pct`/`metrics` entry to read, so `_can_dynamic_scale` saw zero usable PnL values and emitted `PnL data unavailable/invalid for BTC/USDT:USDT (positions=1)`.
- Wiring is otherwise correct: `ProductionCoordinator` injects the same `AdvancedPositionManager` instance into both `LiveTradingEngine` and `RiskManager` via `set_pnl_provider(PositionManagerPnlProvider(self.position_manager))`.

**Fix applied**
- Persist PnL to the position: `monitor_position_pnl` now stores `pnl_pct` (percent for logs) and a normalized `unrealized_pnl_pct` fraction on the position plus `position['metrics']['unrealized_pnl_pct']`.
- Harden the provider: new `_extract_unrealized_pnl_pct` helper normalizes candidates in priority order (`unrealized_pnl_pct`, `metrics.unrealized_pnl_pct`, `pnl_pct`/`pnl.pct` converted from percent → fraction, or a computed fallback from `unrealized_pnl`, `entry_price`, `amount`) and rejects non-finite values.
- `_can_dynamic_scale` now receives the same mark-to-market PnL that drives `[P&L-UPDATE]`, so scale-in gating no longer reports PnL as unavailable when the live loop is logging it.

**Minimal follow-up options**
- Keep PnL normalization here; thresholds remain unchanged. If future producers emit percent-style `unrealized_pnl_pct`, the provider normalization will downscale values >10 to fractions to stay aligned with RiskManager thresholds.
