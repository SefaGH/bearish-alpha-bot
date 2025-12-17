# Signal Quality Computation Analysis

## 1) Function Overview

### Location
`src/core/strategy_coordinator.py` → `StrategyCoordinator._compute_signal_quality(signal: Dict[str, Any]) -> Dict[str, Any]`

### Branches
- **Extreme bypass path** (`signal.get("extreme_bypass")` truthy): custom weights from config (`signals.signal_scoring.extreme_weights`, defaults w_regime=0.4, w_vol=0.3, w_mom=0.3, w_rr=0.0), clamps inputs to [0,1], computes a weighted average, applies an optional `extreme_min_quality` floor, writes `quality_score` and `quality_breakdown`.
- **Normal path** (default): builds a `quality_features` dict and delegates to `quality.compute_quality`. Inputs:
  - `ml_component` ← `signal.get("ml_confidence")`
  - `volume_component` ← `features.get("volume_score")` else `signal.get("volume_24h")`
  - `momentum_component` ← `features.get("momentum")`
  - `spread_component` ← `features.get("spread")`
  - (Note: ignores `volume_strength`, `momentum_strength`, `regime_confidence`, PPO/RL, R/R, etc.)
  - Writes `quality_score` and `quality_breakdown` from `compute_quality`.

### `compute_quality` (src/quality/quality_calculator.py)
- Weights: ml 0.60, volume 0.20, momentum 0.15, spread 0.05.
- Fallbacks (when a component is missing/None):
  - ml: 0.10
  - volume: 0.05 (normalized 0..2 → 0..1, but fallback is 0.05)
  - momentum: 0.05
  - spread: 0.05
- With all fallbacks, final `quality_value` = `0.6*0.10 + 0.2*0.05 + 0.15*0.05 + 0.05*0.05 = 0.08`.
- If `quality_value` ≤ 0.001 it tags reasons; otherwise it just returns the weighted sum (rounded).

### Inputs expected vs. provided
- `_compute_signal_quality` expects ML confidence, volume_score/volume_24h, momentum, spread under `signal["features"]` (or top-level `ml_confidence` and `volume_24h`).
- The enrichment pipeline (prior to `_compute_signal_quality`):
  - `_enrich_signal` adds `volume_strength`, `volume_bucket`, optional `volume_score` (only if strategy config enables `use_volume_strength_in_score`).
  - `_enhance_signal_with_ml` may **not** populate `ml_confidence`; ML context commonly lives in `predicted_regime`, `regime_confidence`, etc.
  - `_compute_signal_quality` runs **before** `_enrich_signal_for_dynamic_rr`, which does populate `ml_confidence`/`regime_confidence` fallbacks (0.5/0.3) but too late for the initial quality calculation.
- Result: for most “normal” signals, `ml_confidence` is missing and `features` is empty → all components take the default fallbacks → quality locks at **0.08**.

## 2) Reproducing the Example

### Normal path (example log, adaptive_ob, scale-in, non-extreme)
Inputs visible in logs:
- ML shown later as 0.50 (set by `_enrich_signal_for_dynamic_rr`, which runs **after** quality calculation).
- Volume strength 0.52 (bucket NORMAL), momentum_strength 0.50, PPO score 1.0, RL agree, regime neutral/conf=1.0, R/R 3.14.
- At `_compute_signal_quality` time:
  - `ml_confidence` likely **missing** (not set yet by ML/Dynamic RR).
  - `features` likely empty (no `volume_score`, `momentum`, `spread`).
  - `volume_24h` not set.

Computation:
- `ml_component`: fallback 0.10
- `volume_component`: fallback 0.05
- `momentum_component`: fallback 0.05
- `spread_component`: fallback 0.05
- Weighted sum = 0.08 → logged as `[QUALITY-BEFORE-RISK] ... quality=0.080`.

This aligns with the repeated 0.08 values in normal (non-extreme) signals: the ML/volume/momentum inputs the scorer expects are not present when it runs.

### Extreme-bypass path (~0.65 observed)
- Branch uses actual `regime_confidence` (or `regime_weight`), `volume_strength`, `momentum_strength`, and optional `rr_ratio` with weights (0.4/0.3/0.3/0.0).
- Example with regime_conf=1.0, volume_strength=0.52, momentum_strength=0.5:
  - q_regime=1.0 → 0.4
  - q_vol=0.52 → 0.156
  - q_mom=0.5 → 0.15
  - Total ≈ 0.706 (clamped/min-applied), consistent with the ~0.65–0.70 qualities seen for extreme-bypass signals.
- Here, because the extreme branch explicitly reads the enriched strengths, it produces a meaningful score.

## 3) Findings / Issues
1. **Systematic fallback to 0.08**: In the normal path, missing `ml_confidence`, `features.volume_score`, `features.momentum`, and `features.spread` causes all components to use the default fallbacks, yielding the fixed 0.08 score regardless of real ML/volume/regime/ppo context.
2. **Input mismatch / timing**:
   - `_compute_signal_quality` runs before `_enrich_signal_for_dynamic_rr` sets `ml_confidence`/`regime_confidence`.
   - It ignores `volume_strength`/`momentum_strength` already present from volume analyzer/market metrics.
   - It relies on `features.volume_score`/`momentum` that are rarely populated in the current pipeline.
3. **Unutilized intelligence**: ML (0.50), RL agree, PPO=1.0, regime_conf=1.0, volume_strength=0.52, momentum_strength=0.5, R/R=3.14 → none of these affect the normal-path quality because the mapping doesn’t use them.
4. **Extreme path behaves as intended** because it directly uses the available strengths; hence the visible gap (0.08 normal vs. ~0.65 extreme).

## 4) Design Options (no code changes yet)

### Option A — Fix input mapping, keep existing weights
- Make `_compute_signal_quality` in the normal path consume the enriched fields already present: `ml_confidence`, `volume_strength` (or analyzer `volume_score`), `momentum_strength`, and optionally `regime_confidence`/`rr_ratio`.
- Ensure `ml_confidence` is set (from ML integration or default 0.5) **before** calling `_compute_signal_quality`.
- **Pros**: Minimal conceptual change; preserves current weighting shape; immediately lifts scores off the 0.08 floor.
- **Cons**: Still couples behavior to the current weight choices; distribution may remain narrow without rebalancing.

### Option B — Redesign as explicit weighted average on 0–1 scale
- Define a clear component set and weights (e.g., ML 0.35, RL/PPO 0.15, regime 0.15, volume 0.15, momentum 0.10, R/R 0.10), all normalized to [0,1] with sane defaults (0.5 neutral).
- Target interpretation: 0.2=weak, 0.5=neutral, 0.8+=strong. Use mid-point defaults instead of tiny fallbacks.
- **Pros**: Produces a wider, more informative distribution; transparent semantics; easier to tune for pyramiding thresholds (e.g., 0.65 meaningful).
- **Cons**: Larger change; requires aligning callers and any downstream thresholds; may need migration/testing of historical expectations.

### Option C — Preserve extreme profile; widen normal distribution
- Keep the extreme-bypass branch as-is; for normal path, swap fallbacks to neutral (e.g., 0.5) and include `volume_strength`/`momentum_strength` plus `ml_confidence` before calling.
- **Pros**: Small change with big practical effect; normal signals would cluster in a mid-range instead of at 0.08; extreme path remains differentiated.
- **Cons**: Less principled than a full redesign; still dependent on current component set and weights in `compute_quality`.

## 5) Conclusion
- The current computation is **structurally flawed for the normal path**: it almost always falls back to the 0.08 default because the inputs it expects are missing at the time it runs. Extreme-bypass works because it directly uses the available enriched metrics. A remap (Option A/C) or a fuller redesign (Option B) is needed to make `quality_score` reflect ML/RL/volume/regime signals meaningfully for pyramiding/risk decisions.
