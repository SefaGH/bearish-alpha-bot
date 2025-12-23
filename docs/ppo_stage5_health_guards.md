# PPO Health Guards (Stage 5)

Purpose: fail safe to neutral when PPO telemetry looks unhealthy (stuck probs, missing normalization, or heavy clipping).

## Guard Conditions (adapter)
- `vecnorm_missing` when `ppo_require_vecnorm=true` but the `.vecnormalize.pkl` sidecar is not loaded.
- `p_long_low_variance` when std of the last `ppo_health_window` p_long samples `< ppo_health_min_std` (default 1e-3).
- `obs_clip_high` when the mean `obs_clip_frac` over the last window `>` `ppo_health_clip_frac_limit` (default 0.30).

## Behavior
- When any guard triggers: `decision=GUARD_FALLBACK`, `score=ppo_fallback_score`, `metadata.reason=health_guard`, `guarded_score=<pre-guard>`.
- Metadata now carries `health_ok`, `health_reasons`, `health_stats` (includes `p_long_std`, optional `clip_mean`), plus thresholds (`conf`, `min_margin`).
- Histories are windowed to `ppo_health_window` to keep memory bounded.

## Config Keys (config/config.example.yaml)
- `ppo_conf_threshold`, `ppo_min_margin`
- `ppo_health_min_std`, `ppo_health_window`, `ppo_health_clip_frac_limit`
- `ppo_require_vecnorm`

## Log/Telemetry Signals
- `[PPO-DEBUG]` / `[PPO-MONITOR]` metadata includes `health_ok`, `health_reasons`, `obs_norm_present`, `obs_clip_frac`, `z_abs_mean`, `z_abs_p99`.
- A guard hit surfaces as `action=GUARD_FALLBACK` and `reason=health_guard` in PPO metadata.
