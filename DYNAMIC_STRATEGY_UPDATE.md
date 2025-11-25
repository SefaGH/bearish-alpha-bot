# Dynamic Strategy Update

This document summarizes the latest enhancements shipped on branch `feature/dynamic-strategy-enhancements` and explains how to configure and operate the new dynamic-strategy stack.

## 1. Highlights
- **Deterministic Dynamic R/R Tests**: `tests/test_dynamic_rr.py` now asserts the exact thresholds returned by `RiskRewardRatioRule` and verifies failure reasons, ensuring future changes to the math are immediately caught.
- **Per-Strategy Risk/Reward Overrides**: `risk.rr_dynamic.strategy_overrides` lets each strategy define custom R/R bounds, weights, or regime multipliers without touching the global baseline (see `src/config/risk_config.py`, `src/core/risk_rules.py`).
- **Regime-Aware Routing**: `strategies.regime_routing` defines preferred strategy bundles per market regime. Enriched signals now carry routing metadata that influences queue priority (`src/core/strategy_coordinator.py`).
- **Queue Intelligence Boost**: `PrioritySignalQueue` ingests regime alignment and strategy urgency so high-fit signals get dispatched faster.
- **Telemetry Hooks**: The coordinator records rolling averages of actual vs. target R/R per strategy, unlocking downstream analytics.

## 2. Configuring the New Features

### 2.1 Risk/Reward Strategy Overrides
Add overrides under `risk.rr_dynamic.strategy_overrides` in `config/config.example.yaml` (or your live config). Only specify values that differ from the global defaults; everything else inherits automatically.

```yaml
risk:
  rr_dynamic:
    base_target_rr: 2.0
    ...
    strategy_overrides:
      scalper:
        base_target_rr: 1.5
        lower_bound_rr: 1.0
        weights:
          ml_confidence: 0.25
          regime_clarity: 0.10
      mean_reversion:
        upper_bound_rr: 2.2
        regime_multipliers:
          neutral: 0.9
          volatile: 1.3
      breakout_hunter:
        base_target_rr: 2.2
        lower_bound_rr: 1.4
        weights:
          ml_confidence: 0.45
          volume_strength: 0.15
```

**Notes**
- Keys are case-insensitive (`Scalper` == `scalper`).
- Overrides merge recursively; nested dicts (e.g., `weights`) only need the fields you want to change.
- The rule annotates each signal with `dynamic_rr_target` and `rr_ratio`, which now feed telemetry.

### 2.2 Regime Routing & Queue Priority
Define routing hints under `strategies.regime_routing`:

```yaml
strategies:
  regime_routing:
    bullish:
      preferred_strategies: ["trend_follower", "breakout_hunter"]
      preferred_priority: 0.8
      fallback_priority: 0.4
      queue_priority_boost: 0.75
    bearish:
      preferred_strategies: ["short_the_rip", "mean_reversion"]
      preferred_priority: 0.85
    neutral:
      preferred_strategies: ["range_sniper", "mean_reversion"]
      preferred_priority: 0.7
    volatile:
      preferred_strategies: ["scalper", "volatility_breakout"]
      preferred_priority: 0.9
      queue_priority_boost: 0.9
    default:
      preferred_priority: 0.5
      fallback_priority: 0.4
```

What happens at runtime:
1. `StrategyCoordinator._apply_regime_route_hint` attaches `regime_route_hint` and `strategy_urgency` based on the current `regime_name`.
2. `PrioritySignalQueue`’s scoring function uses `regime_alignment` and `strategy_urgency` weights (see `risk.queue.priority_weights`). Example defaults:
   ```yaml
   risk:
     queue:
       priority_weights:
         explicit_priority: 0.4
         risk_reward: 0.3
         ml_confidence: 0.2
         urgency: 0.1
         regime_alignment: 0.05
         strategy_urgency: 0.05
   ```
3. Signals that match their preferred regime-strategy combo climb the queue faster, while fallback strategies still retain a tunable priority floor.

## 3. Operational Checklist
1. **Edit Configs**: Update your environment-specific YAML (and/or GitHub variables) with the new blocks above.
2. **Run Focused Tests**: `./pytest.cmd tests/test_dynamic_rr.py -k dynamic` verifies the R/R math and overrides.
3. **Monitor Telemetry**: During dry runs, inspect logs for `dynamic_rr_target`, `rr_ratio`, and the new `regime_route_hint`/`strategy_urgency` fields. Hook `StrategyCoordinator.rr_telemetry` into dashboards if deeper analytics are needed.
4. **Tune Weights Iteratively**: Start with the provided defaults, then adjust per strategy/regime as you gather performance data.

## 4. Next Steps
- Surface `rr_telemetry` summaries via an admin endpoint or periodic log to make override efficacy visible.
- Automate configuration suggestions by correlating telemetry with realized PnL.
- Expand routing beyond regimes (e.g., volatility buckets or exchange health) using the same metadata pattern.

For questions or follow-up work, ping the strategy team or reference the implementation files noted above.
