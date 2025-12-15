# Scale-In Quality Ownership & Extreme-Bypass Model — Design Review

## Overview
- Problem: In extreme-oversold/overbought bypass + scale-in, RiskManager still sees `quality=0.00` despite enrichment context and earlier quality computation. Root cause from prior analyses: `_enrich_signal_for_dynamic_rr` overwrites `quality_score` with an ML fallback (0.0 when bypassed), so `_can_dynamic_scale` reads the reset value.
- Proposed direction to review: make `_compute_signal_quality` the sole owner of `quality_score`; treat ML-derived quality separately or only as a fallback; add an extreme-bypass profile so bypassed signals have meaningful non-zero quality derived from regime/volume/momentum (and optionally R/R/PnL).

## Feasibility Assessment
- **Single ownership of `quality_score`:** Realistic. `_compute_signal_quality` already exists and runs before risk. The main conflict is `_enrich_signal_for_dynamic_rr` (inside `_assess_signal_risk`) reassigning `quality_score` from ML fallbacks. Converting that to a “set if missing” or to write into a separate `ml_quality_score` is localized and low-risk.
- **Call graph alignment:** The signal instance flowing into RiskManager is the same dict mutated by `_enrich_signal_for_dynamic_rr`; preventing that function from overwriting `quality_score` lets RiskManager see the computed value. No public API change needed; just internal ownership discipline.
- **Extreme-bypass profile:** At the point `_compute_signal_quality` is invoked (before risk), we already have volume_strength/bucket, momentum, regime info, and bypass flags. That’s enough to compute a non-ML quality profile. Adding a branch keyed on `signal.get("extreme_bypass")` is straightforward. Optional inputs (unrealized PnL) are available from PositionManager via RiskManager, but not guaranteed in the signal; keeping PnL out of `quality_score` preserves separation of concerns unless explicitly desired.
- **Tests impact:** Existing pyramiding/dynamic scaling tests assert behavior on `quality_score`; moving ownership to `_compute_signal_quality` and stopping overwrites should keep them passing. No test appears to rely on the ML fallback zeroing quality; if any do, they can be updated to assert presence of `quality_score` instead of zeroing.

## Risks / Edge Cases
- **Double-write/oscillation:** If `_compute_signal_quality` is called multiple times (e.g., re-enrichment) we must ensure deterministic, idempotent behavior or store once and reuse.
- **Volume/feature availability:** Extreme-bypass may still lack some features; the quality function needs robust fallbacks (already present via `compute_quality` defaults). Ensure momentum/volume fields exist or guard them.
- **PnL blending:** If PnL is injected into quality, clarify weighting to avoid duplicating RiskManager’s own PnL gate; otherwise, leave PnL purely in RiskManager.
- **Logging clarity:** Without explicit logging of pre/post-risk `quality_score`, regressions could hide. Add structured logs for quality at (a) after `_compute_signal_quality`, (b) right before RiskManager call.

## Recommendations
1) **Ownership rule:** In `_enrich_signal_for_dynamic_rr`, never overwrite `quality_score`. Either:
   - Only set it if missing (`if 'quality_score' not in signal: ...`), or
   - Write ML-derived quality into `signal['ml_quality_score']` and leave `quality_score` untouched.

2) **Extreme-bypass profile in `_compute_signal_quality`:**
   - Detect `signal.get('extreme_bypass')`.
   - Compute quality from available non-ML inputs: regime confidence/quality, volume_strength/bucket, momentum, optional R/R metric; ignore ML/RL entirely.
   - Optional: clamp to a config-driven floor (e.g., `extreme_min_quality`) so bypassed signals don’t default to 0.0 when ML is absent.

3) **Preserve/restore quality pre-risk:** After `_enrich_signal_for_dynamic_rr` (inside `_assess_signal_risk`), ensure the signal heading into `size_and_validate_position` retains the authoritative `quality_score` (restore cached value or re-run `_compute_signal_quality` as a guard).

4) **Logging:** Add structured log of `quality_score` immediately before calling `size_and_validate_position`, and keep `[RISK-SCALING]` log as-is to verify propagation. For bypass flows, log both `extreme_bypass` flag and the computed `quality_score`.

5) **PnL handling:** Keep PnL gating in RiskManager; if later desired, add a small optional PnL bump in the extreme profile with a dedicated weight and config key to avoid hidden coupling.

Preferred path to implement:
- Apply ownership rule (no overwrite in dynamic RR; optional `ml_quality_score` field).
- Add extreme-bypass branch in `_compute_signal_quality` using regime/volume/momentum with a configurable floor.
- Guard quality integrity before RiskManager call and improve logging around that boundary.
