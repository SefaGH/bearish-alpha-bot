# Canary Go/No-Go Metrics

This document defines the operational canary gates based on runtime telemetry events:

- `order_decision_trace` (LiveTradingEngine)
- `order_decision_outcome` (LiveTradingEngine)
- `order_manager_decision` (OrderManager)
- `[RECON-WATCHDOG] ...` (LiveTradingEngine / PositionManager)

Script: `scripts/canary_go_no_go_report.py`

## Scope

Recommended canary scope:

- `symbol = BTC/USDT:USDT`
- `strategy = adaptive_ob`

The script supports filtering with `--symbol` and `--strategy`.

## Event Keys

Primary join key:

- `signal_id` (trace <-> outcome)

Secondary keys (when needed):

- `symbol`
- `timestamp`

## Metrics

1. `smart_entry_applied_rate`
- Source: `order_decision_trace.policy_applied`
- Formula: `count(policy_applied=true) / count(trace)`
- Target: `> 0.50` (preferred `> 0.70`)

2. `missing_atr_force_market_rate`
- Source: `order_decision_trace.policy_decision`, `order_decision_trace.fallback_reason`
- Formula: `count(contains("missing_atr_force_market")) / count(trace)`
- Target: near zero

3. `market_rate_by_bucket[EXTREME]`
- Source: `order_decision_trace.effective_order_type`, `order_decision_trace.bucket`
- Formula: `count(bucket=EXTREME and order=market) / count(bucket=EXTREME)`
- Target: near zero

4. `fallback_reason` distribution
- Source: `order_decision_outcome.fallback_reason`
- Use to confirm expected causes:
  - `limit_timeout_market_fallback_disabled:*` in extreme/fast-move contexts
  - `limit_timeout_market_fallback` only where acceptable

5. `abort_no_fill_timeout_rate`
- Source: `order_decision_outcome.reason`
- Formula: `count(reason startswith ABORT:NO_FILL_TIMEOUT) / count(outcome)`
- Used as missed-fill proxy

6. `env_forced_order_type=market` count
- Source: `order_decision_trace.env_forced_order_type` (and outcome cross-check)
- Target: `0` in canary analysis windows

7. `atr_freshness_market_violations`
- Source: `order_decision_trace.atr_age_ms`, `order_decision_trace.effective_order_type`
- Formula: `count(atr_age_ms > threshold and order=market)`
- Target: `0`

8. `entry_slippage_bps p90/p95` (trade-weighted + notional-weighted)
- Source: `order_decision_outcome.entry_slippage_bps`, `order_decision_outcome.entry_notional_usd`
- Targets: baseline-relative improvement (p90/p95 lower)

9. `stop_overshoot_bps p90/p95`
- Source: `TRADE_CLOSED.stop_overshoot_bps`
- Target: baseline-relative improvement (p90/p95 lower)

10. `time_to_fill_ms p50/p90/p95`
- Source: `order_decision_outcome.time_to_fill_ms`
- Target: should remain within acceptable latency envelope while slippage improves

11. `planned_vs_realized_rr_drift` (abs p90/p95)
- Source: `TRADE_CLOSED.planned_vs_realized_rr_drift`
- Fallback derive: `abs(TRADE_CLOSED.rr_achieved - TRADE_CLOSED.rr_after_fill)`
- Target: lower is better (less RR drift)

12. `recon_orphans_detected_total`
- Source: `[RECON-WATCHDOG] stale_removed=... orphans_detected=... orphans_adopted=...`
- Target (pre-adopt stage): `0`

13. `recon_stale_removed_total`
- Source: `[RECON-WATCHDOG] stale_removed=...`
- Target: close to `0` (unexpected spikes indicate local/exchange drift)

14. `recon_orphans_adopted_total`
- Source: `[RECON-WATCHDOG] ... orphans_adopted=...` or adopt logs
- Target (pre-adopt stage): `0`

## Go/No-Go Gates

Hard gates:

- `env_forced_order_type=market == 0`
- `smart_entry_applied_rate >= threshold`
- `missing_atr_force_market_rate <= threshold`
- `market_rate_by_bucket[EXTREME] <= threshold`
- `atr_freshness_market_violations == 0`

Pre-adopt (watchdog hard gates):

- `recon_orphans_detected_total <= threshold` (typically `0`)
- `recon_orphans_adopted_total <= threshold` (typically `0`)
- `recon_stale_removed_total <= threshold` (environment-dependent; often `0` for canary)

Optional baseline gate:

- `ABORT:NO_FILL_TIMEOUT` increase vs baseline `<= +20%`

## Known Telemetry Gaps

The report marks a telemetry gap when one of these fields is absent in runtime logs:

- `order_decision_outcome.entry_slippage_bps`
- `order_decision_outcome.time_to_fill_ms`
- `TRADE_CLOSED.stop_overshoot_bps`
- `TRADE_CLOSED.planned_vs_realized_rr_drift` (or `rr_after_fill` + `rr_achieved`)
- `[RECON-WATCHDOG]` cycle logs (required only if `--require-recon-watchdog-events` is set)

If they are present, p90/p95 metrics are computed automatically.

## Usage

Example:

```powershell
python scripts/canary_go_no_go_report.py `
  --symbol "BTC/USDT:USDT" `
  --strategy adaptive_ob `
  --atr-age-threshold-ms 5000 `
  --require-recon-watchdog-events `
  --max-recon-orphans-detected 0 `
  --max-recon-orphans-adopted 0 `
  --max-recon-stale-removed 0 `
  --output-json artifacts/canary/go_no_go_latest.json
```

With baseline:

```powershell
python scripts/canary_go_no_go_report.py `
  --symbol "BTC/USDT:USDT" `
  --strategy adaptive_ob `
  --baseline-json artifacts/canary/go_no_go_prev.json `
  --output-json artifacts/canary/go_no_go_latest.json
```

## Simulation Flows

For CI-safe validation without live exchange execution:

1. Test-based simulation logs:

```powershell
pytest tests/test_smart_entry_engine_integration.py -q -o log_cli=true --log-cli-level=INFO -o color=no *> artifacts/canary/sim_engine.log
pytest tests/test_execution_backend.py -q -o log_cli=true --log-cli-level=INFO -o color=no *> artifacts/canary/sim_order_manager.log
python scripts/canary_go_no_go_report.py `
  --log-file artifacts/canary/sim_engine.log `
  --log-file artifacts/canary/sim_order_manager.log `
  --log-glob "" `
  --symbol "BTC/USDT:USDT" `
  --output-json artifacts/canary/go_no_go_sim_tests.json
```

2. Deterministic execution harness:

```powershell
python scripts/simulate_canary_execution.py --log-file artifacts/canary/sim_harness.log
python scripts/canary_go_no_go_report.py `
  --log-file artifacts/canary/sim_harness.log `
  --log-glob "" `
  --symbol "BTC/USDT:USDT" `
  --strategy adaptive_ob `
  --output-json artifacts/canary/go_no_go_sim_harness.json
```
