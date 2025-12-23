# Stage 6 — Retrain Roadmap (planned)

Current state:
- Live-parity dataset built from the live feature pipeline (82+5+2 obs).
- PPO model (300k steps) remains flat-biased; gating vetoes ~100% of longs.
- VecNormalize + spec parity in place.

Planned improvements:
1) Data: regenerate NPZ with a longer history (more regimes) via `scripts/build_ppo_dataset_from_live_pipeline.py` (increase `--candles`, adjust date ranges).
2) Reward tweaks (configurable):
   - Mix absolute log-return with benchmark-relative term.
   - Raise/relax clip range (e.g., clip_obs=10 remains, but reward clip to 5.0) — controlled via config.
3) Training sweep (target ≥1M steps, multi-seed):
   - learning_rate ∈ {2.5e-4, 1e-4}, ent_coef ∈ {0.001, 0.005}, clip_range ∈ {0.2, 0.3}, gamma ∈ {0.99, 0.995}.
   - Track p_long distribution, entropy, and gating-pass rate on the test split after each run.
4) Selection criteria:
   - Non-degenerate p_long (std > 0.05, mean not collapsed to ~0).
   - Gating backtest improves return/DD relative to baseline longs without killing participation (>10% of longs pass threshold).
5) Validate:
   - Parity tests, threshold sweep, gating impact backtest.
   - Update artifacts (`zip`, `.obs_spec.json`, `.vecnormalize.pkl`) and reports when the candidate is chosen.

Status: pending execution (requires longer training run and possibly reward config changes).*** End Patch
