# Pyramiding Scale-In Quality – Extreme Bypass Path Analysis

## Overview
- Problem: In paper runs with extreme-oversold bypass + same-side scale-in, RiskManager logs `📉 [RISK-SCALING] Denied scale-in ... | quality=0.00 < 0.80`, even though enrichment logs show non-trivial ML/regime/volume/momentum context and pyramiding tests pass.
- Goal: Identify why `quality_score` is 0.00 specifically on the EXTREME-BYPASS + SCALE_IN path and propose design-level fixes (no code changes made).

## Code Path Map (Extreme-Bypass → Scale-In → Risk)
1) **Signal intake (`StrategyCoordinator.process_strategy_signal`)**
   - Intent initially set via `_determine_intent` (entry vs scale_in).
   - Enrichment: `_enrich_signal` (volume/momentum/ctx) → optional early duplicate check → `_enhance_signal_with_ml`.
2) **Extreme bypass logic (`_enhance_signal_with_ml` + `_check_extreme_condition_bypass`)**
   - `_check_extreme_condition_bypass` runs *before* ML/RL enhancement.
   - If RSI triggers extreme oversold/overbought:
     - Logs `[EXTREME-OVERSOLD-BYPASS] ... Bypassing all ML/RL checks`.
     - Calls `_prepare_force_swap_slot`:
       - If same-side position exists and pyramiding enabled: sets `signal['intent'] = INTENT_SCALE_IN` (`[EXTREME-SAME-SIDE] ... converting bypass signal to SCALE_IN`).
     - `_enhance_signal_with_ml` returns early with minimal ML fields (ml_confidence default 0.8, bypass flags).
3) **Post-bypass pipeline**
   - Conflict check.
   - `_compute_signal_quality` (added earlier) computes quality from `ml_confidence`, volume/momentum/spread features and writes `quality_score` + `quality_breakdown` onto the same `enriched_signal` dict.
   - `_assess_signal_risk` is called next.
4) **Risk assessment (`_assess_signal_risk`)**
   - First step: `_enrich_signal_for_dynamic_rr(signal)` runs again and **re-writes `quality_score`** from ML context:
     - If ML integration is absent/disabled/bypassed, it sets `signal['quality_score'] = 0.0` (fallback) (see lines ~2716–2749).
   - Position sizing → `risk_manager.size_and_validate_position` → `RiskManager.validate_new_position` → `_check_concurrent_limits` → `_can_dynamic_scale`.
5) **RiskManager scale-in gate (`_can_dynamic_scale`, `src/core/risk_manager.py`)**
   - Reads `quality_score = float(signal.get('quality_score', 0) or 0)`; logs `[RISK-SCALING] ... quality=...`.
   - For `intent=scale_in` with pyramiding enabled, compares against `pyramiding.min_scale_in_quality`.

## Field Wiring Analysis
- **Enrichment logs**: `[Signal Enriched] ... ML=0.50 ... Regime={..., 'quality_score': 1.0, ...}, Vol=1.00, Mom=0.50 ...`
  - These values come from ML context/regime metadata and volume/momentum fields; they are not the same as the final `signal['quality_score']` used by RiskManager.
- **`_compute_signal_quality`**:
  - Inputs: `ml_confidence`, `features.volume_score`/`volume_24h`, `features.momentum`, `features.spread`.
  - Outputs: writes `signal["quality_score"]` and `signal["quality_breakdown"]` (intended to be consumed by RiskManager).
- **Overwrite point**:
  - `_enrich_signal_for_dynamic_rr` (inside `_assess_signal_risk`) unconditionally sets `quality_score` from ML context defaults when ML is missing/bypassed (0.0), overwriting the value produced by `_compute_signal_quality`.
  - In extreme-bypass cases (ML skipped by design), this reset drives `quality_score` to 0.0 just before RiskManager is called.
- **RiskManager**:
  - Uses `signal.get('quality_score')` directly; no special handling for extreme bypass or alternative quality fields.

## Root Cause Hypothesis
- In the EXTREME-BYPASS + SCALE_IN path, `quality_score` computed earlier is overwritten to `0.0` by `_enrich_signal_for_dynamic_rr` because ML was bypassed and the fallback sets `quality_score` to zero. RiskManager then reads this zeroed value in `_can_dynamic_scale`, leading to `quality=0.00 < 0.80` despite earlier enrichment showing strong context.
- The dict instance passed into RiskManager is the same one whose `quality_score` was reset; there is no separate “scale_in_quality” field, so this overwrite is decisive.

## Solution Options (Design-Level)
1) **Preserve existing quality in `_enrich_signal_for_dynamic_rr`**
   - Change `_enrich_signal_for_dynamic_rr` to only set `quality_score` if it is absent; do not overwrite a precomputed value from `_compute_signal_quality`.
   - Pros: Minimal surface area; keeps single `quality_score` source of truth; aligns risk view with earlier computation.
   - Cons: If ML context arrives later, may need an explicit merge rule (e.g., prefer ML when present).

2) **Recompute/restore quality after dynamic RR enrichment but before risk**
   - After `_enrich_signal_for_dynamic_rr` inside `_assess_signal_risk`, re-run `_compute_signal_quality` (or stash the precomputed value and restore it) so the final signal handed to `size_and_validate_position` carries the intended quality.
   - Pros: Guarantees RiskManager sees the final quality even if intermediate steps mutate it.
   - Cons: Extra compute; must ensure no double-counting/oscillation if volume_score adjustments happen later.

3) **Explicit extreme-bypass override**
   - In extreme-bypass cases (RSI-triggered, ML skipped), set `quality_score` to a derived baseline (e.g., from volume/momentum/regime confidence) and guard against later resets.
   - Pros: Ensures non-zero quality for bypassed signals without requiring ML.
   - Cons: Special-case logic; must be carefully documented to avoid inflating quality for non-bypass flows.

4) **Position-aware fallback (optional)**
   - For `intent=scale_in`, if `quality_score` is missing/zero post-enrichment, blend in the existing position’s stored `quality_score` (PositionManager stores entry quality metadata).
   - Pros: Leverages prior entry quality when new signal lacks ML data.
   - Cons: More invasive; needs clear rules to avoid overstating quality when position quality is stale.

**Recommendation:** Start with option 1 (preserve precomputed quality in `_enrich_signal_for_dynamic_rr`) and, if needed, add option 2 to reassert the computed quality right before risk validation. Both keep a single `quality_score` field and avoid special-casing risk logic.
