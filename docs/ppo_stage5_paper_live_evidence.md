# Stage 5 – Paper/Shadow Live Evidence (to be filled after run)

Run instructions:
- Deploy artifacts: `ppo_trading_agent.zip` + `.obs_spec.json` + `.vecnormalize.pkl` from `artifacts/ppo/`.
- Enable PPO monitor with current config (ppo_enabled=true) and defaults:
  - `ppo_conf_threshold=0.60`, `ppo_min_margin=0.0`
  - Health guards on (require_vecnorm=true, health_window=30, health_min_std=1e-3, clip_limit=0.30)
- Optional: `PPO_DUMP_LIMIT=20` if you need a small `obs.jsonl` sample (leave off by default).

What to capture (paste snippets below):
- `[PPO-INIT]` confirming spec + vecnorm loaded (obs_dim=89, obs_norm_present=True).
- 20–50 `[PPO-DEBUG]` lines across at least one new candle showing:
  - `p_long` varies (not identical)
  - `obs_norm_present=True`, `obs_clip_frac` reasonable (<0.3), `z_abs_mean/p99` stable
  - `health_ok=True` (no `health_reasons`)
  - tail overrides (pf/pv) non-default if portfolio/equity available
- `[PPO-MONITOR]` or decision logs that show non-constant scores.

Quick stats to compute from the captured window:
- p_long mean/std:
- entropy mean/std:
- obs_clip_frac mean/p95:
- health_ok rate:

Log snippets (paste):
```
<paste here>
```
