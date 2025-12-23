# PPO Stage 4 Evaluation (VecNormalize-enabled PPO Artifact)

Artifacts produced (artifacts/ppo):
- `ppo_trading_agent.zip`
- `ppo_trading_agent.obs_spec.json` (87 features + tail, obs_dim=89)
- `ppo_trading_agent.vecnormalize.pkl` (norm_obs=True, norm_reward=False, clip_obs=10.0)

Training run
- Command: `python scripts/train_ppo_agent.py --dataset data/training/BTC_USDT_USDT_1h_train.npz --model-dir artifacts/ppo --model-name ppo_trading_agent --timesteps 10000`
- VecNormalize applied during training (norm_obs=True, norm_reward=False, clip_obs=10.0)
- Sidecars saved next to the model.

Offline evaluation (data/training/BTC_USDT_USDT_1h_test.npz)
- Command: `python scripts/evaluate_ppo_agent.py --model artifacts/ppo/ppo_trading_agent.zip --dataset data/training/BTC_USDT_USDT_1h_test.npz --output-summary data/training/ppo_eval_summary_new.json --output-equity-curve data/training/ppo_eval_equity_curve_new.csv`
- Summary (data/training/ppo_eval_summary_new.json):
  - steps: 899
  - PnL: -2652.16 (return -0.2652), maxDD -0.3003
  - unique actions: [0, 1], num_trades: 214
- Probability / entropy stats (same eval run, deterministic policy over the episode):
  - `p_long` min/max/mean/std: 0.2041 / 0.8360 / 0.5533 / 0.1280
  - Counts: `p_long >= 0.90`: 0; `>=0.95`: 0; `>=0.98`: 0; `<0.90`: 899
  - Entropy min/max/mean/std: 0.4462 / 0.6931 / 0.6534 / 0.0474
  - Entropy buckets: `<0.2`: 0, `0.2–0.4`: 0, `>0.4`: 899
- Interpretation: distribution is no longer “stuck” at a single probability; `p_long` varies meaningfully across the episode, and entropy is moderate (no collapse to zero).

Parity checks
- `pytest tests/test_ppo_observation_parity.py -q` (passes): raw parity, normalized parity (VecNormalize), and “probabilities vary” sanity check on fixture.
- `src/tools/ppo_observation_parity_check.py --model artifacts/ppo/ppo_trading_agent.zip --dataset data/training/BTC_USDT_USDT_1h_test.npz --index -1`
  - Observations align in shape (89) and spec; tool shows non-identical values when using live feature pipeline vs dataset features (manifest differences). Use for manual diffs; adapter enforces spec/vecnorm at runtime.

Next steps
- Use this artifact set for paper/staging rollout with VecNormalize enabled (adapter loads sidecars automatically).
- Monitor `[PPO-DEBUG]` for `obs_norm_present=True`, reasonable `obs_clip_frac`, and non-constant `p_long`.

---

## Stage 5 retrain (live-parity dataset, 82+5+2=89 obs)

Artifacts (overwrites same filenames under `artifacts/ppo/`):
- `ppo_trading_agent.zip`
- `ppo_trading_agent.obs_spec.json` (82 features from GEMMA manifest + 5 price extras + 2 tail → obs_dim=89)
- `ppo_trading_agent.vecnormalize.pkl`

Data + training:
- Dataset rebuilt via `scripts/build_ppo_dataset_from_live_pipeline.py --input-file data/training/BTC_USDT_USDT_1h_combined_prices.csv --output-dir data/training`
- Train command: `python scripts/train_ppo_agent.py --dataset data/training/BTC_USDT_USDT_1h_liveparity_train.npz --model-dir artifacts/ppo --model-name ppo_trading_agent --timesteps 300000 --obs-spec data/training/BTC_USDT_USDT_1h_liveparity.obs_spec.json`

Offline eval (data/training/BTC_USDT_USDT_1h_liveparity_test.npz):
- Summary (data/training/ppo_eval_summary_liveparity.json):
  - steps: 899
  - PnL: -59.71 (return -0.0060), maxDD -0.0060
  - unique actions: [0, 1], num_trades: 2
- Probability / entropy stats (threshold sweep via `src/tools/ppo_threshold_sweep.py`):
  - `p_long` min/max/mean/std: 1.41e-07 / 0.8946 / 0.0088 / 0.0495
  - Entropy min/max/mean/std: 2.23e-06 / 0.6917 / 0.0250 / 0.0824
  - Pass-rate (margin>=0): `p_long>=0.6` → ~0.1% of samples (effectively none); even at 0.3 pass rate ~0.1%.

Interpretation:
- Observation/spec parity now matches the live manifest (82 base features + 5 extras).
- The current policy remains very flat-biased (low entropy, low p_long); PPO acts as a soft veto rather than an approver. Threshold calibration doc outlines the pass-rate curves and recommends keeping `ppo_conf_threshold=0.60` for now.
