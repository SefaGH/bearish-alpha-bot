# Implementation Report: Key-Path + Type Coercion

## Legacy Key Consumption Audit
- Config-based legacy keys: no direct consumption found.
  - Evidence: `rg --line-number "ml_rl_training_mode" src` -> only hit is the deprecation list in `src/config/live_trading_config.py:88`.
- Env-only legacy usage (not mapped in this iteration):
  - `TRADING_MODE` is read from env in `src/main.py:467`.
  - `TRADING_SYMBOLS` is read from env in `src/core/production_coordinator.py:912`.

## Compat Mapping Decision
- Compat mapping list: none (no legacy keys consumed from config dicts).
- Warn-only policy applied for deprecated keys if present in config:
  - `ml_rl_training_mode` is flagged as deprecated in `src/config/live_trading_config.py` (warn, no mapping).
- `TRADING_MODE` / `TRADING_SYMBOLS` are left as deprecate/cleanup backlog per request (env-only).

## Central Type Coercion (Allowlist-first)
- Location: `src/config/live_trading_config.py`.
- Apply point: after YAML+ENV+AppConfig merge, before `_apply_universe_defaults` and `_normalize_risk_config`.
- Allowlist (high-risk paths):
  - `ml.reinforcement_learning.training_mode` (bool)
  - `signals.short_the_rip.mtf_confirmation.enabled` (bool)
  - `signals.short_the_rip.mtf_confirmation.require_15m` (bool)
  - `signals.short_the_rip.mtf_confirmation.require_1h` (bool)
  - `pyramiding.enabled` (bool)
  - `strategies.regime_routing.bullish.preferred_strategies` (list)
  - `strategies.regime_routing.bearish.preferred_strategies` (list)
  - `strategies.regime_routing.neutral.preferred_strategies` (list)
  - `strategies.regime_routing.volatile.preferred_strategies` (list)
  - `risk.min_stop_pct` (float)
  - `risk.per_trade_risk_pct` (float)
  - `risk.max_position_size_pct` (float)
  - `risk.max_notional_pct_per_trade` (float)
  - `risk.max_margin_pct_per_trade` (float)
  - `risk.daily_loss_limit_pct` (float)
- Coercion rules:
  - Bool: `true/false/1/0/yes/no/on/off` (case-insensitive) -> bool.
  - Numeric: `int`/`float` parse; invalid -> warn, keep raw.
  - JSON list/dict: if string starts with `[` or `{`, attempt `json.loads`.
  - Comma-separated list: `a,b,c` -> `['a','b','c']` when expected type is list.
  - JSON parse failure: warn and keep raw value (no hard error).
- YAML parser note:
  - Inline lists in `config/config.example.yaml` are loaded as strings by the repo’s custom `yaml` module.
  - For `strategies.regime_routing.*.preferred_strategies`, an explicit type override is added so JSON strings coerce to lists.

## RiskConfiguration Defensive Cast (Startup Crash Fix)
- Location: `src/config/risk_config.py`.
- Defensive cast added for:
  - `risk.min_stop_pct` (string -> float) before `> 1` comparison.
  - `risk.max_position_notional_usd` / `risk.computed_max_notional_usd` before numeric comparisons.
- Behavior: invalid numeric strings -> warning + default fallback.

## Tests
- `pytest tests/test_live_trading_config.py -k "allowlist_type_coercion"` -> PASS
- `pytest tests/test_risk_config_usd_amounts.py -k "min_stop_pct_string"` -> PASS
