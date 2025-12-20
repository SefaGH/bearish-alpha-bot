# Implementation Report: AppConfig Full Compatibility (Key-Path + Schema Coercion)

## Summary
- Added schema-driven recursive coercion after merge, while keeping allowlist-first for high-risk paths.
- AppConfig key normalization now preserves/uppercases symbol segments to match YAML symbols.
- Added unknown AppConfig key warnings and allowlist type validation (warn-only by default, strict mode via CONFIG_STRICT_TYPE_CHECK).

## Changes
### 1) Schema-driven coercion
- File: src/config/live_trading_config.py.
- New flow: build schema from config/config.example.yaml -> warn unknown AppConfig keys -> allowlist coercion -> schema-driven coercion -> allowlist type validation -> defaults/risk normalization.
- Coercion handles: bool, int/float, JSON list/dict, comma lists; JSON parse failures warn and keep raw values.
- List element typing: JSON/comma lists are cast using schema element types when available.

### 2) Allowlist + validation
- Allowlist-first remains; now expanded via prefix risk. and explicit high-risk paths (MTF flags, pyramiding.enabled, RL training_mode).
- Allowlist type validation added: warn-only by default; strict mode with CONFIG_STRICT_TYPE_CHECK=true fails fast on mismatched types.

### 3) AppConfig key-path validation
- AppConfig raw keys are normalized with the same segment rules as nesting.
- Any normalized AppConfig key not in canonical YAML schema is logged as deprecated/warn-only.
- Legacy keys are still warn-only (no compat mapping added because not consumed).

### 4) Symbol-segment normalization
- AppConfig nesting no longer lowercases every segment.
- Segments that match symbol patterns (e.g., BTC/USDT:USDT) are normalized to uppercase.
- Fixes signals.*.symbols.<SYMBOL> and leverage overrides key access.

### 5) YAML inline list overrides
- Added explicit schema type overrides for inline list fields:
  - ml.features.volatility_windows (list[int])
  - ml.features.momentum_windows (list[int])
  - ml.price_prediction.timeframes (list[str])
  - ml.price_prediction.models (list[str])
  - ml.reinforcement_learning.ppo_lookback_windows (list[int])
  - strategies.regime_routing.*.preferred_strategies (list[str])

## Backward-compat decisions
- No compat mapping added for legacy AppConfig keys (none consumed in code).
- Deprecated keys are warn-only: ml_rl_training_mode (use canonical nested path).
- TRADING_MODE / TRADING_SYMBOLS remain env-only (deprecate/cleanup backlog).

## Tests
- pytest tests/test_live_trading_config.py -k "type_coercion or appconfig_symbol" -> PASS
