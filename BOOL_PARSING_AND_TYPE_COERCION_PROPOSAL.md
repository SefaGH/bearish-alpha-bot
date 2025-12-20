# Bool Parsing and Type Coercion Proposal (No Code Changes)

## Executive Summary
- AppConfig overrides are loaded as strings and merged without type casting, so string values like "false" are truthy in many runtime checks.
- The canonical type source is the YAML defaults; env overrides are cast using that schema, but AppConfig overrides are not.
- Multiple high-impact subsystems use `bool(cfg.get(...))` or truthy `if cfg.get(...)` patterns that will flip feature toggles when a string is provided.
- AppConfig key normalization lowercases and splits on dot, so env-style keys like `ML_RL_TRAINING_MODE` become top-level fields and do not override nested YAML paths.
- JSON-like list strings (for routing and strategy selection) are parsed in some adapters but not consistently across the app layer.

## Risk Inventory
| Subsystem | Key(s) (examples) | Pattern | Code Pointer(s) | Expected Type | Likely Type from AppConfig | Severity |
| --- | --- | --- | --- | --- | --- | --- |
| MTF enable | `signals.short_the_rip.mtf_confirmation.enabled` | `bool(cfg.get(...))` and truthy check | `src/core/production_coordinator.py:406` `src/strategies/adaptive_str.py:803` | bool | str | High |
| MTF require flags | `signals.short_the_rip.mtf_confirmation.require_15m`, `require_1h`, `require_1h_bearish_ema_stack` | `bool(cfg.get(...))` | `src/strategies/adaptive_str.py:422` `src/strategies/adaptive_str.py:540` `src/strategies/adaptive_str.py:551` | bool | str | High |
| RL training mode | `ml.reinforcement_learning.training_mode` | direct assignment + `not cfg.get(...)` | `src/ml/reinforcement_learning.py:248` `src/core/production_coordinator.py:1216` | bool | str | High |
| PPO enable | `ml.reinforcement_learning.ppo_enabled` | `bool(cfg.get(...))` and truthy checks | `src/ml/adapters/ppo_trading_adapter.py:115` `src/core/production_coordinator.py:1244` `src/core/strategy_coordinator.py:1557` | bool | str | High |
| Legacy RL enable | `ml.reinforcement_learning.legacy_dqn_enabled` | `bool(cfg.get(...))` | `src/core/strategy_coordinator.py:491` `src/core/production_coordinator.py:1201` | bool | str | Medium |
| Reward clipping | `ml.reinforcement_learning.reward_clip_enabled` | `bool(cfg.get(...))` | `src/ml/reinforcement_learning.py:233` | bool | str | Medium |
| Pyramiding | `pyramiding.enabled` | `bool(cfg.get(...))` | `src/core/risk_manager.py:373` `src/core/strategy_coordinator.py:573` | bool | str | High |
| Dynamic scaling | `concurrent_limits.dynamic_scaling.enabled` | truthy check | `src/core/risk_manager.py:435` | bool | str | Medium |
| Bypass enable | `signals.bypass.enabled` | truthy check | `src/core/strategy_coordinator.py:2264` | bool | str | Medium |
| Bypass force swap | `signals.bypass.force_swap_enabled` | truthy check | `src/core/strategy_coordinator.py:2291` | bool | str | Medium |
| Duplicate prevention bypass | `signals.duplicate_prevention.price_delta_bypass_enabled` | truthy check | `src/core/strategy_coordinator.py:652` | bool | str | Medium |
| Regime routing lists | `strategies.regime_routing.*.preferred_strategies` | JSON string treated as single string | `src/core/strategy_coordinator.py:2786` | list[str] | str | Medium |
| Universe filters | `universe.only_linear`, `universe.exclude_stables` | `bool(cfg.get(...))` | `src/universe.py:105` `src/universe.py:106` | bool | str | Medium |
| Regime ignore | `signals.oversold_bounce.ignore_regime` | `bool(cfg.get(...))` | `src/main.py:295` | bool | str | Medium |
| Volatility sizing | `risk.volatility_sizing.enabled` | `bool(cfg.get(...))` | `src/config/live_trading_config.py:699` `src/config/risk_config.py:761` | bool | str | Medium |
| Alerts | `telegram.enabled`, `discord.enabled`, `email.enabled`, `webhook.enabled` | truthy checks | `src/monitoring/alert_manager.py:86` `src/monitoring/alert_manager.py:98` `src/monitoring/alert_manager.py:103` `src/monitoring/alert_manager.py:108` | bool | str | Low-Medium |

## Root Cause Analysis
**Config loader and override flow**
- YAML is loaded and normalized first, and provides the canonical default types.
- Environment overrides are cast using the YAML value type via `_get_env_overrides` and `_cast_value`.
- AppConfig overrides are loaded through REST and kept as raw strings, then merged into the config without casting.
- No post-merge type coercion or validation step exists for AppConfig values.

**Key paths and schema**
- `_flatten_to_nested` lowercases and splits only on dots. Keys without dots (for example `ML_RL_TRAINING_MODE`) become top-level `ml_rl_training_mode`, which does not override `ml.reinforcement_learning.training_mode`.
- The canonical schema is implicit (YAML types) rather than explicit, which makes it easy to skip casting when a new source is added.

**JSON list handling is inconsistent**
- PPO adapter parses JSON-like strings, but routing, strategy lists, and several other subsystems do not parse list or dict strings at the app layer.

## Solution Options
### Option 1: Central Type Coercion After Merge
**Approach**
- After the YAML + env + AppConfig merge, walk the resulting config and coerce values based on a canonical type schema (derived from YAML defaults).

**Pros**
- Single place to fix the behavior; minimal changes to call sites.
- Keeps existing config shapes and downstream code intact.

**Cons**
- Requires a robust type-walk to handle missing keys and partial overrides.
- Risk of breaking cases where strings are intentionally passed through.

**Regression risk**
- Medium: may change behavior for existing string values that are currently relied on as truthy.

**Back-compat plan**
- Start with an allowlist of critical keys, then expand.

**Observability**
- Log per-key cast events and parsing failures; expose counts as metrics.

### Option 2: Schema-Driven Casting (Explicit Schema)
**Approach**
- Define an explicit schema (or derive it from YAML in a structured way) and cast all incoming values from AppConfig and env using that schema.

**Pros**
- Predictable and testable; provides strong guarantees about runtime types.
- Easy to validate for missing keys or unexpected types.

**Cons**
- Requires schema maintenance and synchronization with YAML.
- More up-front design effort than Option 1.

**Regression risk**
- Low to Medium: depends on schema coverage and strictness.

**Back-compat plan**
- Start with a "warn-only" mode, then enforce for a key allowlist.

**Observability**
- Track schema mismatches and unknown keys; provide a summary at startup.

### Option 3: Strict Validation (Warn-Fast or Fail-Fast)
**Approach**
- Validate types at startup and either warn or fail when a bool/list/dict field is a string.

**Pros**
- Clear signal when AppConfig data is malformed.
- Prevents silent behavior changes in production.

**Cons**
- Does not fix the values by itself; still needs remediation in AppConfig or a follow-up cast step.
- Fail-fast can cause downtime if the data is not cleaned first.

**Regression risk**
- Medium for warn-only; High for fail-fast without a staged rollout.

**Back-compat plan**
- Warn-only in production, fail-fast in CI or staging.

**Observability**
- Emit per-key validation warnings and a summary counter.

## Recommended Roadmap (Phased Rollout)
1. Phase 0: Inventory and measurement
   - Build a read-only report of AppConfig keys and their types, plus an allowlist of critical bool/list keys.
2. Phase 1: Allowlist type coercion for critical keys
   - Target MTF, RL training_mode, pyramiding, and bypass flags first.
3. Phase 2: Expand to all bools and JSON lists
   - Include alert toggles, routing lists, and ML sub-flags.
4. Phase 3: Add validation gates
   - Warn-fast in production, fail-fast in staging/CI.

## Test and Validation Plan
**Unit tests**
- Bool parsing cases: "false", "true", "0", "1", "no", "yes", "", "True", "ON".
- JSON list parsing: '["trend_follower","breakout_hunter"]', "trend_follower,breakout_hunter".
- Negative cases: invalid JSON and mixed-type lists.

**Integration tests**
- Simulate AppConfig overrides and assert the effective config snapshot matches expected types.
- Include both dotted keys and env-style keys to verify key normalization behavior.

**Canary plan**
- Enable type coercion on one bot instance, compare:
  - MTF block/pass rates
  - RL inference vs training logs
  - Strategy selection distribution
  - Pyramiding scale-in frequency

## Appendix: Code Pointers
**AppConfig load and merge**
- `src/config/live_trading_config.py:722` AppConfig load and flat key ingest.
- `src/config/live_trading_config.py:815` `_flatten_to_nested` key normalization.
- `src/config/live_trading_config.py:251` env override casting based on YAML type.
- `src/config/live_trading_config.py:399` `_cast_value` rules.

**MTF gating**
- `src/core/production_coordinator.py:406` mtf_enabled bool cast.
- `src/strategies/adaptive_str.py:422` require_15m bool cast.
- `src/strategies/adaptive_str.py:540` require_1h bool cast.
- `src/strategies/adaptive_str.py:803` mtf enabled truthy check.

**ML/RL toggles**
- `src/ml/reinforcement_learning.py:248` training_mode assignment.
- `src/core/production_coordinator.py:1216` inference mode gating.
- `src/ml/adapters/ppo_trading_adapter.py:115` ppo_enabled bool cast.

**Pyramiding and bypass**
- `src/core/risk_manager.py:373` pyramiding_enabled bool cast.
- `src/core/strategy_coordinator.py:2264` bypass enabled truthy check.
- `src/core/strategy_coordinator.py:652` price_delta_bypass_enabled truthy check.

**Routing**
- `src/core/strategy_coordinator.py:2786` JSON-like routing strings treated as single string.

