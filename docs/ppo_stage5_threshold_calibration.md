# Stage 5 – PPO Threshold Calibration

Dataset/model used:
- Model: `artifacts/ppo/ppo_trading_agent.zip` (300k steps on live-parity dataset)
- VecNormalize: `artifacts/ppo/ppo_trading_agent.vecnormalize.pkl`
- Spec: `artifacts/ppo/ppo_trading_agent.obs_spec.json` (82 + 5 extras + 2 tail = 89)
- Eval dataset: `data/training/BTC_USDT_USDT_1h_liveparity_test.npz` (900 rows)

Distribution summary (normalized obs, policy head):
- `p_long`: min 1.41e-07, max 0.8946, mean 0.0088, std 0.0495
- `entropy`: mean 0.0250, std 0.0824 (often very low → sharp preference for FLAT)

Threshold sweep (margin = p_long - p_flat >= 0):
- `p_long` > 0.1 → 1.9% of samples, but only 0.1% also have margin ≥ 0
- `p_long` > 0.5 .. 0.8 → pass rate ~0.1% (essentially none)
- Even at 0.3, pass rate ~0.1% (margin requirement eliminates most)

Recommendation (for paper/live canary):
- Keep `ppo_conf_threshold` at 0.60 and `ppo_min_margin` at 0.0 for now; PPO acts as a **soft veto** and will rarely force LONG given the current model.
- If you want more PPO vetoes (rather than approvals), keep the high threshold; if you want occasional approvals, you would need a stronger model or drop the threshold dramatically (<0.2) which is not advised given current low entropy.
- Revisit after retraining with live-parity dataset (or richer data) to lift the p_long distribution; rerun `src/tools/ppo_threshold_sweep.py` to refresh pass-rate curves.

How to reproduce:
```
python src/tools/ppo_threshold_sweep.py \
  --model artifacts/ppo/ppo_trading_agent.zip \
  --dataset data/training/BTC_USDT_USDT_1h_liveparity_test.npz \
  --spec artifacts/ppo/ppo_trading_agent.obs_spec.json \
  --threshold-start 0.1 --threshold-stop 0.8 --threshold-step 0.02 --margin 0.0
```
