# Implementation Report: Systemic Config Type Coercion

## Summary
- Reworked coercion to be schema-first + heuristic fallback, without requiring per-key allowlists.
- Only string values are parsed; already-typed bool/int/float/list/dict values remain untouched.
- Added strict-mode schema validation gate (CONFIG_STRICT_TYPE_CHECK) with warn-only default.

## Coercion Pipeline
- Location: `src/config/live_trading_config.py`.
- Order after merge:
  1) build schema from `config/config.example.yaml`
  2) warn on unknown AppConfig keys
  3) schema-first coercion for all schema paths
  4) heuristic fallback for schema-unknown string values
  5) deprecated key warnings
  6) schema validation (warn-only or strict)

## Schema-first coercion
- Types inferred from canonical YAML values.
- Inline list/dict strings are parsed (JSON -> literal_eval) to derive schema types.
- Coercion applies only when the runtime value is a string and matches a schema path.

## Heuristic fallback (schema-unknown keys)
- Applied to any string value not found in schema (including nested dict values).
- Parsing order:
  1) bool tokens: true/false/1/0/yes/no/on/off
  2) numeric: int, then float
  3) structured: json.loads if looks like [..] or {..}
     - fallback: ast.literal_eval for list/dict literals like ['a','b'] or {'k':1}
  4) comma lists: "a,b" -> list; if all numeric, cast numeric list
  5) else keep string

## Strict mode validation
- Env: `CONFIG_STRICT_TYPE_CHECK=true`.
- Fail-fast only for schema-known paths if types still mismatch after coercion.
- Default is warn-only.

## Deprecated key detection
- Warn-only for legacy keys (e.g., `ml_rl_training_mode`) when present.
- No compat mapping added unless code consumes the legacy key directly.

## Tests
- `pytest tests/test_live_trading_config.py -k "coercion or heuristic or inline_list"` -> PASS
- Covers:
  - schema-first: `risk.rr_dynamic.base_target_rr` and `signals.short_the_rip.mtf_confirmation.require_15m`
  - heuristic fallback: int/float/bool/list literal + comma list
  - regression: `ml.price_prediction.models` stays list (no JSON warning)
