# RL Training Workflow

This guide captures the full pipeline for preparing data, training the RL agent, and validating checkpoints offline. All commands assume the repository root as the working directory and Python 3.11.*.

## 1. Prepare Datasets
Use `scripts/prepare_rl_training_data.py` to fetch OHLCV candles (or consume an existing CSV/Parquet file), run them through the feature pipeline, and emit train/val/test archives under `data/training/`.

```pwsh
C:/Users/sefaa/AppData/Local/Programs/Python/Python311/python.exe scripts/prepare_rl_training_data.py `
  --exchange bingx `
  --symbol BTC/USDT:USDT `
  --timeframe 1h `
  --candles 1800 `
  --output-dir data/training `
  --config config/config.example.yaml `
  --train-ratio 0.7 `
  --val-ratio 0.15
```

Key outputs:
- `data/training/BTC_USDT_USDT_1h_train.npz`
- `data/training/BTC_USDT_USDT_1h_val.npz`
- `data/training/BTC_USDT_USDT_1h_test.npz`
- `data/training/BTC_USDT_USDT_1h_metadata.json`

## 2. Train the RL Agent
Point the trainer at any of the prepared splits (typically `*_train.npz`). Hyperparameters come from `ml.reinforcement_learning` in the config unless overridden on the CLI.

```pwsh
C:/Users/sefaa/AppData/Local/Programs/Python/Python311/python.exe scripts/train_rl_agent.py `
  --dataset data/training/BTC_USDT_USDT_1h_train.npz `
  --config config/config.example.yaml `
  --episodes 250 `
  --save-every 25 `
  --model-dir data/checkpoints `
  --model-name rl_agent_train.pth
```

Results:
- Rolling checkpoints under `data/checkpoints/`
- Final checkpoint `data/checkpoints/rl_agent_final.pth`
- Summary JSON `data/training/rl_training_summary.json`

### Experimentation shortcuts
- Override optimizer hyperparameters without touching config YAML:
  `scripts/train_rl_agent.py --dataset ... --learning-rate 0.0001 --gradient-clip 10` speeds up updates while guarding against explosions.
- Normalize rewards on the fly: `--reward-clip 10 --reward-scale 0.01` (or `--reward-clip-range -5 5`) compresses the TD target and matches the scale used in `inspect_replay_and_td.py`.
- The new `ml.reinforcement_learning` keys (`reward_clip_*`, `reward_scale`, `gradient_clip_norm`) expose the same knobs when you want enduring defaults.

## 3. Validate a Checkpoint
Replay the agent on a validation or test split to gather reward/PnL telemetry and Q-value distributions. The script produces a JSON report you can diff across runs.

```pwsh
C:/Users/sefaa/AppData/Local/Programs/Python/Python311/python.exe scripts/validate_rl_model.py `
  --dataset data/training/BTC_USDT_USDT_1h_val.npz `
  --checkpoint data/checkpoints/rl_agent_final.pth `
  --config config/config.example.yaml `
  --report-file data/training/rl_validation_report.json
```

Outputs:
- `data/training/rl_validation_report.json` with action distribution, reward totals, and Q-value stats.

## Tips
- Use `--input-file` on the prepare script to bootstrap from cached CSVs instead of live fetches.
- `--max-steps` on `validate_rl_model.py` lets you run quick smoke tests over a subset of the dataset.
- The helper scripts automatically add `scripts/` to `sys.path`, so modules like `rl_dataset_utils` remain reusable across CLIs.
- `scripts/inspect_replay_and_td.py` now supports `--reward-clip`, `--reward-clip-range`, and `--reward-scale` so you can see TD normalization effects before retraining.

## 4. Verify Learnable Head Scale (Optional Diagnostics)
Two quick checks confirm that the learnable head-scale parameter is wired up correctly before launching sweeps or long training jobs.

### 4.1 Inspect Parameter Registration
```pwsh
$env:PYTHONPATH='src;.'
python scripts/inspect_head_param.py `
  --model data/checkpoints/rl_agent_head_scale_canonical.pth `
  --state-size 87 `
  --head-scale-learnable `
  --initial-head-scale 1.0 `
  --head-scale-min 0.1
```
- Prints all parameters with their `requires_grad` flags and surfaces optimizer param-groups.
- Confirms `head_scale_raw` appears in `model.named_parameters()` and in at least one optimizer group, and reports the resolved head-scale (computed as `head_scale_min + softplus(raw)`).

### 4.2 Probe Gradient Flow
```pwsh
$env:PYTHONPATH='src;.'
python scripts/inspect_head_gradient.py `
  --model data/checkpoints/rl_agent_head_scale_canonical.pth `
  --state-size 87 `
  --head-scale-learnable `
  --initial-head-scale 1.0 `
  --head-scale-min 0.1 `
  --batch-size 32
```
- Runs a synthetic single update, printing the loss, current `head_scale` value, the softplus input (`head_scale_raw`), and gradient magnitude.
- A non-zero gradient indicates backprop reaches the parameter; if it stays zero, investigate the forward wiring or optimizer groups.

### 4.3 Validate Migration with Pytest
```pwsh
$env:PYTHONPATH='src;.'
pytest tests/unit/test_head_scale_migration.py -q
```
- Test skips automatically when the legacy checkpoint fixture is absent (`tests/data/legacy_rl_agent_head_scale.pth`).
- When the checkpoint and optional JSON meta are present it ensures the migrated head-scale matches the recorded legacy value within tight tolerance.
