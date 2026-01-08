# Trailing “Activation” Semantics Audit (Stage-3)

## What “activation” means

The bot tracks trailing activation per-position using:

- `position["trailing_stop_activation_threshold"]` (decimal, e.g. `0.003` = 0.3%)
- `position["trailing_stop_activated"]` (bool)

Activation is evaluated in `AdvancedPositionManager._is_trailing_stop_active()` (`src/core/position_manager.py:2176`):

- If threshold `<= 0`: trailing activates immediately (`trailing_stop_activated=True`).
- Else: activates only after price moves in favor by the threshold relative to entry.

## What sets activation today (precedence)

Activation threshold is resolved per-position in `LiveTradingEngine._resolve_execution_config()` (`src/core/live_trading_engine.py:402`), with precedence:

1) **Signal overrides** (`signal["execution"]["trailing_stop"]`), or legacy alias `signal["trailing_stop_config"]`:
   - `activation_threshold_pct`
   - or alias `activation_threshold`
   - or `activation_price` (converted into a pct using `signal["entry"]`)
   (`src/core/live_trading_engine.py:482`)

2) **Strategy execution profile** (`strategies.<name>.execution_profile → execution_profiles.<profile>.trailing_stop.activation_threshold_pct`)
   (`src/core/live_trading_engine.py:454`)

3) **Global defaults** (`position_management.trailing_stop.activation_threshold`)
   (`src/core/live_trading_engine.py:428`)

Config defaults (example):
- Global: `position_management.trailing_stop.activation_threshold: 0.003` (`config/config.example.yaml:623`)
- Profile: `execution_profiles.scalp_tight.trailing_stop.activation_threshold_pct: 0.003` (`config/config.example.yaml:534`)

After the position is opened, LiveTradingEngine applies the resolved value to the position via
`AdvancedPositionManager.configure_trailing_stop(..., activation_threshold_pct=...)` (`src/core/live_trading_engine.py:934` → `src/core/position_manager.py:2669`).

## When native trailing is placed (Stage-2)

Native trailing is placed **only when trailing is active** inside `AdvancedPositionManager.monitor_position_pnl()`:

- It calls `_is_trailing_stop_active(...)` and only when `is_active=True` will it attempt native trailing placement.
- Placement is feature-flagged (`BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED`) and requires real execution.
  (`src/core/position_manager.py:1676`)

So, with a positive activation threshold (e.g. 0.3%), native trailing is not placed until the activation move is reached.

## Why diagnostics showed activation=0.0000

The Stage-2 diagnostics runners intentionally configure activation as `0.0` to force immediate trailing placement for VST validation:

- `diagnostics/bingx_vst_stage2_native_stops_evidence.py` defaults `--trailing-activation-threshold 0.0`
- `diagnostics/bingx_vst_stage2_edge_cases.py` uses activation `0.0` internally

That is test harness behavior, not the production default.

## Recommended default + safety implications

Recommended default: keep activation **positive** (e.g. `0.003` = 0.3%) unless you explicitly want “always-on” trailing.

Safety implications:
- `activation_threshold_pct <= 0` is not unsafe, but it changes behavior materially (native trailing can be placed immediately at entry).
- In hedge mode, **do not rely on reduceOnly** for safety; rely on `positionSide` + correct close side + `qty <= position size`.
  (validated by VST matrix artifacts; see `diagnostics/vst/bingx_vst_matrix_summary_latest.json`)

