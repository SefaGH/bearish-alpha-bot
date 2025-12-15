# Pyramiding / Scale-In Quality Analysis

## Overview
- Paper-mode runs with `pyramiding.enabled=true` show `[RISK-SCALING] Denied scale-in ... quality=0.00 < 0.80`, so every scale-in candidate is rejected on the quality gate.
- Pyramiding config is loaded correctly (min quality 0.80, min PnL 0.5%, min distance 0.5%, queue limits set), and signals are reaching the pyramiding path (intent classified as `scale_in`, queue sees scale-in counters).
- Enriched-signal logs elsewhere show non-zero scoring components, so quality exists but is not present when RiskManager evaluates scale-ins.

## Code Map (Quality Wiring)
- **Intent classification (StrategyCoordinator._determine_intent @ src/core/strategy_coordinator.py:553)**  
  - If pyramiding is enabled in config/portfolio_manager.cfg and a same-side position already exists for the strategy/symbol, intent is set to `scale_in`; otherwise `entry`. No quality mutation here.

-.signal enrichment & ML/RL context (StrategyCoordinator._enrich_signal_for_dynamic_rr, same file)**  
  - Adds `ml_confidence`, regime data, PPO/RL flags, volume/momentum metrics.  
  - Sets an initial `quality_score` from ML context if available; **fallback is 0.0 when ML data is missing** (lines ~2710–2742).  
  - This happens **before** risk validation.

- **Risk sizing/validation (StrategyCoordinator._assess_signal_risk → RiskManager.size_and_validate_position)**  
  - `_assess_signal_risk` calls `risk_manager.size_and_validate_position` **before quality is computed from features.**
  - Inside `RiskManager.validate_new_position` → `_check_concurrent_limits` → `_can_dynamic_scale` (src/core/risk_manager.py:399):  
    - Reads `quality_score = float(signal.get('quality_score', 0) or 0)`; values >1 normalized to 0–1.  
    - Uses pyramiding thresholds (`pyramiding.min_scale_in_quality`, `min_scale_in_unrealized_pnl_pct`, `min_scale_in_distance_pct`).  
    - Emits denial log `[RISK-SCALING] Denied scale-in for %s | quality=%.2f < %.2f` if `quality_score` below threshold.
  - Because quality hasn’t been computed yet, `quality_score` is whatever the ML enrichment set (often 0.0 when ML context is absent), so scale-ins fail here.

- **Quality computation (StrategyCoordinator.process_strategy_signal, step after risk assessment)**  
  - After risk validation and duplicate checks, builds `quality_features` (ml_confidence, volume/momentum/spread) and calls `compute_quality` (src/quality/quality_calculator.py).  
  - Sets `enriched_signal["quality_score"] = quality_result["value"]` (non-zero even with fallbacks) and adds `quality_breakdown`.  
  - This happens **after** RiskManager has already decided to accept/reject the signal.

- **Queue/bridge/engine wiring**  
  - Accepted signals are enqueued with the full `enriched_signal` payload (includes the computed `quality_score`) and passed to LiveTradingEngine via `_strategy_coordinator_bridge_loop` (src/core/live_trading_engine.py:830+).  
  - Bridge simply merges payload fields; no quality mutation.  
  - PositionManager stores `quality_score`/`quality_breakdown` on position entry if present (src/core/position_manager.py:204–212).

## Findings: Why quality is 0.00 at scale-in gate
- **Timing bug:** RiskManager evaluates pyramiding quality during `size_and_validate_position` *before* StrategyCoordinator computes the final quality via `compute_quality`.  
  - At that moment, `signal['quality_score']` is only the ML fallback (0.0 when ML context missing/disabled), so `_can_dynamic_scale` sees 0.0 and rejects.  
  - The later `compute_quality` result never influences this initial decision; the signal is already rejected and never reaches the queue/engine.
- **Field alignment:** RiskManager expects `quality_score` (0–1). StrategyCoordinator also uses `quality_score`, so naming is consistent; the issue is strictly ordering/timing, not a key mismatch.
- **Observation alignment:** The denial log matches `_can_dynamic_scale`’s early read of `quality_score` (0.00) and the configured threshold 0.80, confirming the above flow.

## Proposed Positive Test Scenarios (design only)
1) **Unit: RiskManager dynamic scaling allows high-quality scale-in**  
   - Preconditions: `pyramiding.enabled=true`, `min_scale_in_quality=0.80`, `min_scale_in_unrealized_pnl_pct=0.005`, `min_scale_in_distance_pct=0.005`; `max_positions_per_symbol=1`, `dynamic_scaling.enabled=true`.  
   - PortfolioManager mock: 1 open position for symbol X/side long, `unrealized_pnl_pct=0.01`, `entry_price=100`.  
   - Signal: `intent=scale_in`, `quality_score=0.90`, `entry=101` (distance ~1%), side=long.  
   - Expect: `_can_dynamic_scale` returns allow; log `[PYRAMID] scale-in allowed ... quality=0.90/0.80 ...`.

2) **Integration-ish: StrategyCoordinator scale-in path**  
   - Step 1: Submit an entry signal; it passes risk and opens a position.  
   - Step 2: Submit another same-side signal with positive unrealized PnL on the open position, pyramiding enabled.  
   - Ensure quality is computed **before** calling `size_and_validate_position` (post-fix) so `quality_score>=0.85`.  
   - Expect: intent classified as `scale_in`; risk validation passes dynamic scaling; signal enqueued with `[PYRAMID]` allow log; `quality_score` persists into queue payload.

3) **Queue + engine bridge flow for scale-in**  
   - Config: `risk.queue.max_pending_scale_in_per_symbol` set >0; pyramiding enabled.  
   - With an open position and a qualifying scale-in signal (`quality_score=0.90`, distance/PnL thresholds met):  
     - StrategyCoordinator enqueues the signal (queue stats show `scale_in` pending).  
     - LiveTradingEngine bridge pulls it; RiskManager `can_open_new_position` sees the non-zero `quality_score` and allows execution.  
   - Expected logs: `[PYRAMID-QUEUE] ...` for enqueue, `[PYRAMID] scale-in allowed ...`, followed by execution logs.

## Suggested Next Implementation Steps
- Move or duplicate quality computation so `quality_score` is populated **before** RiskManager’s `size_and_validate_position` / `_can_dynamic_scale` runs (e.g., compute right after `_enrich_signal_for_dynamic_rr` or pass a precomputed quality into risk validation).  
- Ensure ML fallback does not zero out quality when ML context is missing; rely on `compute_quality` fallbacks instead.  
- Add targeted tests for the scenarios above (unit for `_can_dynamic_scale`, integration for StrategyCoordinator → queue → bridge).  
- Re-run paper mode with pyramiding enabled to confirm `[RISK-SCALING]` logs show non-zero quality and allow paths when thresholds are met.
