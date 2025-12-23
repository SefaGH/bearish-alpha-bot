# PPO Stage 4 Rollout / Rollback Plan (VecNormalize-enabled)

## Rollout Steps
1) Artifacts
   - Ensure these files exist together under `artifacts/ppo/`:
     - `ppo_trading_agent.zip`
     - `ppo_trading_agent.obs_spec.json`
     - `ppo_trading_agent.vecnormalize.pkl`
2) Config
   - Point PPO model path to `artifacts/ppo/ppo_trading_agent.zip` (if configurable).
   - Keep PPO enabled for the target symbol(s).
3) Deploy / restart
   - Restart the service/bot so the adapter reloads the model + sidecars.
4) Verify init logs
   - Look for `[PPO-INIT] model_obs_dim=89 spec_obs_dim=89 ...` and `obs_norm_present=True`.
   - No `obs_dim_mismatch` or `spec_missing` errors.
5) Verify live telemetry
   - `[PPO-DEBUG]` shows `obs_norm_present=True`, reasonable `obs_clip_frac`, and non-constant `p_long`.
   - Entropy is non-zero; `p_long` varies over ticks/candles.

## Rollback Triggers
- `obs_norm_present=False` (vecnorm not loaded)
- `obs_dim_mismatch` / `obs_build_failed` errors
- `obs_clip_frac` ~1.0 or exploding values
- `p_long` constant across time; entropy near zero

## Rollback Steps
1) Point PPO model path back to the previous artifact set (zip/spec/vecnorm trio).
2) Restart the service/bot.
3) Confirm `[PPO-INIT]` loads the prior model and `[PPO-DEBUG]` resumes normal telemetry.
