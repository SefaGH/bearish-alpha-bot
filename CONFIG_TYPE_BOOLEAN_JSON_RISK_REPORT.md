# Config Types and Boolean/JSON Parsing Risk Report

## Executive Summary
- AppConfig overrides are flattened to nested keys without type casting, so boolean and list/dict values can remain strings and flow into truthy checks across critical areas.
- Multiple modules use `bool(cfg.get(...))` or truthy `if cfg.get(...)` patterns for MTF, pyramiding, bypass, alerts, and ML/RL flags, which will misinterpret string values like "false".
- Runtime snapshot was not obtainable in this environment (no running containers), so the string-value inventory is based on code paths and user-reported runtime examples.
- Regime routing is implemented in `strategy_coordinator`; risk is not "unused config" but JSON-like string rules being treated as single-string strategy names.
- Log evidence shows RL inference forcing in at least one run, but no MTF gating logs were found in the available log set.

## Observations (Runtime Config Types)
**Observation**
- No running Docker containers were found, so `get_config()` runtime snapshot and `docker logs` could not be collected here.
- Local log files contain config dumps with booleans shown as `True/False` (no quoted `"false"` strings observed in these dumps).

**Reported (from user context, not reproduced here)**
- Examples like `ml_rl_training_mode='false'`, `debug_mode='false'`, and `signals.short_the_rip.mtf_confirmation.require_15m='false'` indicate AppConfig values arriving as strings.

**Gap**
- No live type inventory was captured for "bool-like strings" or JSON-like strings. This is needed to quantify scope and confirm which keys are affected at runtime.

## Code Pattern Findings (Locations and Patterns)
**AppConfig load and type handling**
- `src/config/live_trading_config.py:722` loads AppConfig values and returns raw strings from REST.
- `src/config/live_trading_config.py:796` flattens AppConfig into nested keys without type casting.
- `src/config/live_trading_config.py:815` only lowercases and splits on dots; env-style keys (e.g., `ML_RL_TRAINING_MODE`) become top-level `ml_rl_training_mode` and do not map to nested config paths.
- `src/config/live_trading_config.py:251` applies `_cast_value` only for environment-variable overrides, not AppConfig values.
- `src/config/live_trading_config.py:399` defines `_cast_value` (bool/list parsing) but it is not applied to AppConfig overrides.

**MTF gating and require_* flags**
- `src/core/production_coordinator.py:406` uses `mtf_enabled = bool(mtf_cfg.get('enabled', False))`.
- `src/core/production_coordinator.py:751` repeats `mtf_enabled` bool casting in async path.
- `src/strategies/adaptive_str.py:422` `require_15m = bool(cfg.get('require_15m', False))`.
- `src/strategies/adaptive_str.py:540` `require_1h = bool(cfg.get('require_1h', False))`.
- `src/strategies/adaptive_str.py:551` `require_1h_bearish_ema_stack = bool(cfg.get('require_1h_bearish_ema_stack', True))`.
- `src/strategies/adaptive_str.py:803` checks `mtf_cfg.get("enabled", False)` without casting; string "false" is truthy.

**ML/RL mode and toggles**
- `src/ml/reinforcement_learning.py:248` assigns `self.training_mode = self.config.get('training_mode', False)` without casting.
- `src/ml/reinforcement_learning.py:253` uses `self.training_mode` to select epsilon values.
- `src/core/production_coordinator.py:1201` `legacy_rl_enabled = rl_config.get('enabled', True) and rl_config.get('legacy_dqn_enabled', False)` uses truthiness.
- `src/core/production_coordinator.py:1216` `not rl_config.get('training_mode', False)` controls inference-mode forcing.
- `src/core/strategy_coordinator.py:491` `legacy_dqn_enabled = bool(rl_cfg.get('legacy_dqn_enabled', False))`.

**Pyramiding and risk**
- `src/core/risk_manager.py:373` `pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False))`.
- `src/core/risk_manager.py:438` repeats bool cast when validating cfg shape.
- `src/core/strategy_coordinator.py:76` stores `_pyramiding_enabled = bool(queue_config.get('pyramiding_enabled', False))`.
- `src/core/strategy_coordinator.py:425` merges `pyramiding_enabled` into queue config using `bool(...)`.
- `src/core/strategy_coordinator.py:573` `pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False))`.

**Bypass and monitoring toggles**
- `src/core/strategy_coordinator.py:2264` `if not bypass_config.get('enabled', True)` uses truthiness (string "false" => enabled).
- `src/core/risk_rules.py:705` `if not config.get('enabled', False)` uses truthiness.
- `src/core/production_coordinator.py:2163` `enable_circuit_breaker` uses truthiness.
- `src/monitoring/alert_manager.py:86` `telegram.enabled` checked with truthy `if`.
- `src/monitoring/alert_manager.py:98` `discord.enabled` checked with truthy `if`.
- `src/monitoring/alert_manager.py:103` `email.enabled` checked with truthy `if`.
- `src/monitoring/alert_manager.py:108` `webhook.enabled` checked with truthy `if`.

**Regime routing rules and JSON-like strings**
- `config/config.example.yaml:402` defines `strategies.regime_routing` keys.
- `config/config.example.yaml:404` uses list literals for `preferred_strategies`.
- `src/core/strategy_coordinator.py:405` loads `self.regime_routing_rules` from config.
- `src/core/strategy_coordinator.py:2754` applies regime routing to signals.
- `src/core/strategy_coordinator.py:2786` treats string rules as a single strategy name.
- `src/core/strategy_coordinator.py:2794` reads `preferred_strategies` and applies priorities.

## MTF Findings (Behavior, Frequency, Gating)
**Observation**
- No MTF-related log entries were found in `logs/live_trading_*.log` for `No Signal: MTF-15m`, `No Signal: MTF-1h`, `mtf_15m_missing`, or `mtf_1h_missing`.
- No `mtf` tokens were found in those logs, implying MTF may be disabled or logging suppressed in the available runs.

**Hypothesis (based on code paths and reported string values)**
- If AppConfig delivers `signals.short_the_rip.mtf_confirmation.require_15m = "false"`, then `bool("false")` evaluates to `True` and the MTF requirement will be enforced.
- If `mtf_confirmation.enabled` is a string "false", the truthy check in `adaptive_str` will still enable MTF.

**Gating points**
- MTF enablement in coordinator: `src/core/production_coordinator.py:406`.
- MTF require flags: `src/strategies/adaptive_str.py:422`, `src/strategies/adaptive_str.py:540`.
- MTF block logs exist in code but were not observed in logs: `src/strategies/adaptive_str.py:818`, `src/strategies/adaptive_str.py:825`.

## ML/RL Mode Findings
**Observation**
- Logs show RL agent forced into inference mode in at least one run:
  - `logs/live_trading_20251116_234446_517426.log:362` shows "training_mode=True detected ... Forcing inference mode".
  - `logs/live_trading_20251116_234446_517426.log:365` shows `training_mode: False`.
  - `logs/live_trading_20251116_234446_517426.log:373` shows `RL Agent Config: training_mode=False`.

**Hypothesis (based on code paths and reported string values)**
- If AppConfig delivers `ml.reinforcement_learning.training_mode = "false"` as a string, then:
  - `self.training_mode` becomes a truthy string (`src/ml/reinforcement_learning.py:248`).
  - Inference forcing can be skipped because `not rl_config.get('training_mode', False)` becomes False (`src/core/production_coordinator.py:1216`).
- If AppConfig uses env-style keys like `ML_RL_TRAINING_MODE`, the override may land as top-level `ml_rl_training_mode` (unused), leaving the nested `training_mode` unchanged.

## Regime Routing Findings
**Observation**
- Regime routing is implemented and used in signal processing:
  - Routing rules loaded at `src/core/strategy_coordinator.py:405`.
  - Applied to signals at `src/core/strategy_coordinator.py:2754`.
  - Strategy urgency derived at `src/core/strategy_coordinator.py:2794`.
- No explicit routing or preferred-strategy logs were found in the available `logs/live_trading_*.log` set; only risk config dumps include the word `strategy_urgency`.

**Hypothesis**
- If AppConfig provides JSON-like strings for `preferred_strategies`, `_normalize_route_hint` will treat the full JSON string as a single strategy name (`src/core/strategy_coordinator.py:2786`), causing no preferred match and no intended priority boost.

## Risk Matrix (Prioritized)
| Risk | Affected Areas | Evidence | Likelihood | Impact |
| --- | --- | --- | --- | --- |
| AppConfig values remain strings and flow into truthy checks, flipping toggle semantics | MTF, pyramiding, bypass, alerts, emergency flags | Code patterns in `src/config/live_trading_config.py:722` and bool/truthy usage across modules | High (if AppConfig used) | High |
| AppConfig key naming mismatch (env-style keys not mapped to nested paths) | Any override using env-style keys | `_flatten_to_nested` only lowercases and splits on dots: `src/config/live_trading_config.py:815` | Medium | Medium |
| RL training_mode string interpreted as truthy, inference forcing skipped | ML/RL training vs inference, epsilon behavior | `src/ml/reinforcement_learning.py:248`, `src/core/production_coordinator.py:1216`, log evidence of inference forcing | Medium-High | High |
| MTF require flags and enablement treat string "false" as True | MTF gating, signal blocking | `src/strategies/adaptive_str.py:422`, `src/strategies/adaptive_str.py:803`, no MTF logs found | Medium | Medium-High |
| Pyramiding enabled string interpreted as True | Exposure, scaling risk | `src/core/risk_manager.py:373`, `src/core/strategy_coordinator.py:76` | Medium | High |
| Regime routing list strings not parsed as lists | Strategy selection/priority | `src/core/strategy_coordinator.py:2786`, `config/config.example.yaml:404` | Medium | Medium |
| Alert/bypass flags treated as truthy when string "false" | Monitoring and risk bypass | `src/monitoring/alert_manager.py:86`, `src/core/strategy_coordinator.py:2264` | Medium | Low-Medium |

## Appendix: Commands and Summary Outputs
**Container availability**
- `docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"` -> no running containers.
- `docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"` -> no containers found.

**Code searches**
- `rg -n "def get_config|load_appconfig|AppConfig" src config` -> config loader in `src/config/live_trading_config.py`.
- `rg -n "bool\\(.*get\\(" src` -> bool/truthy patterns in MTF, pyramiding, ML/RL, risk.
- `rg -n "regime_routing|preferred_strategies|preferred_priority|queue_priority_boost" src config` -> routing config and usage in `src/core/strategy_coordinator.py`.
- `rg -n "training_mode|inference_mode|ml_rl_training_mode" src` -> RL mode usage and gating.

**Log searches**
- `rg -g "live_trading_*.log" -c -i "No Signal: MTF-15m" logs` -> no matches.
- `rg -g "live_trading_*.log" -c -i "No Signal: MTF-1h" logs` -> no matches.
- `rg -g "live_trading_*.log" -c -i "mtf_15m_missing|mtf_1h_missing|mtf_15m_block|mtf_1h_block" logs` -> no matches.
- `rg -g "live_trading_*.log" -c -i "training_mode|inference_mode|RL Agent forced" logs` -> multiple files with 2-3 matches.
- `rg -n -i "training_mode|inference_mode|RL Agent forced" logs\\live_trading_20251116_234446_517426.log` -> evidence of inference forcing.

## Birlikte değerlendirme için soru listesi
1. AppConfig keys are stored as dot-paths (e.g., `ml.reinforcement_learning.training_mode`) or env-style keys (e.g., `ML_RL_TRAINING_MODE`)?
2. Which container/environment should be used to capture a live `get_config()` snapshot and type inventory?
3. Should MTF be active in production? If yes, do we expect `No Signal: MTF-*` logs and do we need to confirm data availability?
4. Is RL training ever intended in live/paper runs, or should training_mode always be forced to inference?
5. Do you expect visible logs for regime routing/priority decisions, or should we treat routing as silent behavior?
