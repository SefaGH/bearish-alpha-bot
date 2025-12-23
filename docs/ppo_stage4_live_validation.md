# PPO Stage 4 Live/Paper Validation Checklist (VecNormalize)

Artifact set deployed:
- `artifacts/ppo/ppo_trading_agent.zip`
- `artifacts/ppo/ppo_trading_agent.obs_spec.json`
- `artifacts/ppo/ppo_trading_agent.vecnormalize.pkl`

What to check in logs (paper/staging run):
- `[PPO-INIT]` shows `spec_obs_dim=89` and no obs_dim mismatch.
- `[PPO-DEBUG]` shows:
  - `obs_norm_present=True`
  - `obs_clip_frac` low/stable (<< 1.0)
  - `z_abs_mean` / `z_abs_p99` reasonable (not exploding)
  - `p_long` varies over time (not identical to 6 decimals across ticks)
  - Entropy is non-zero and not constantly minimal.
- `[PPO-MONITOR]` continues to show actions/confidence; ensure they are not constant.

How to run and inspect quickly:
```bash
docker logs --since 30m --timestamps bearish-bot | grep "[PPO-DEBUG]" | tail -n 50
```

What would trigger rollback:
- `obs_norm_present=False` (sidecars not loaded)
- `obs_clip_frac` ~1.0 or clearly stuck/clipped everywhere
- `p_long` constant across multiple candles/ticks
- Entropy near zero consistently
- Explicit `obs_dim_mismatch` / `spec_missing` / `obs_build_failed` errors

If rollback needed:
- Point config/model path back to the previous PPO artifact set.
- Restart the service and confirm `[PPO-INIT]` loads the prior model (and logs the prior obs_dim/spec).
