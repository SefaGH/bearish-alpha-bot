# PPO Stage 1 — Observation Parity & Diagnostics (No Behavior Change)

## Model Artifact Summary
- File: `artifacts/ppo/ppo_trading_agent.zip`
- Loaded via SB3 `PPO.load`
- `observation_space.shape`: `(89,)`
- `action_space`: `Discrete(2)`
- Policy: `MlpPolicy` (standard SB3 MLP), no VecNormalize wrapper detected in training script.

## Training Observation Spec
- Entrypoint: `scripts/train_ppo_agent.py`
- Env: `RLTradingEnvGym` -> `RLTradingEnv`
- Observation: `features_df.iloc[step]` (feature columns from the `.npz` dataset) **+ tail** `[position_fraction, normalized_pv]`
- No normalization/VecNormalize/scaler; raw feature magnitudes.
- No extra 5 price-derived features; no clipping; tail is unclamped in env.
- Expected dim during training: `len(feature_columns) + 2` (likely 84 if 82 features in `.npz`).

## Live Observation Spec
- Adapter: `src/ml/adapters/ppo_trading_adapter.py`
- Features: manifest-selected GEMMA price features (82) via `FeatureEngineeringPipeline.extract_features`.
- Extra: 5 OHLCV-derived features `_compute_extra_features_from_price` (log-returns, range, vol10, EMA spread).
- Tail: `[position_fraction, normalized_pv]` with clamping to [0,1] and [0.1,5.0]; defaults to `[0,1]` if unavailable.
- State concatenation: 82 + 5 + 2 = **89**; `_align_state_dim` pads/truncates only if mismatch.
- No normalization/scaler; raw magnitudes; debug-only scaled/clip variants are logged but not used for decisions.

## Parity Tool
- Script: `src/tools/ppo_observation_parity_check.py`
- Usage:
  ```bash
  python src/tools/ppo_observation_parity_check.py \
    --model artifacts/ppo/ppo_trading_agent.zip \
    --dataset data/training/BTC_USDT_USDT_1h_train.npz \
    --index -1
  ```
- What it does:
  - Builds training-style obs (features + tail) and live-style obs (features + 5 extras + tail) on the same snapshot.
  - Aligns to model obs_dim if needed.
  - Prints vector stats, top diffs (value + feature name), and PPO distribution stats (probs/entropy/logits) for both.

## Live Dump Hook (debug only)
- Env vars:
  - `PPO_DUMP_OBS=/tmp/ppo_obs.jsonl`
  - `PPO_DUMP_LIMIT=50` (optional)
- Effect: appends JSONL entries with `state` and `metadata` per PPO call (off by default). No decision/logic change.

## Stage 2 Updates (Observation Parity)
- Shared observation spec (`src/ml/ppo/observation_spec.py`) defines ordered feature/extra/tail lists and obs_dim; training saves a sidecar spec (`*.obs_spec.json`) next to the PPO model.
- Both training env (`RLTradingEnv`) and live adapter (`PPOTradingAdapter`) now build observations via the same spec; mismatches raise explicit errors instead of silent padding.
- Fixture for CI: `tests/fixtures/ppo_parity_fixture.npz` (82 features, 5 price columns, 300 rows, 1h), used in parity test.
- Parity test: `tests/test_ppo_observation_parity.py` asserts env vs adapter observation equality (max abs diff ≤ 1e-6) using the fixture.
- Parity tool can still be pointed at the live/eval NPZ (e.g., `data/training/BTC_USDT_USDT_1h_train.npz` or the latest evaluation NPZ) via `--dataset`.

## Stage 3 — Normalization & Live Robustness
- Training: `train_ppo_agent.py` wraps the env with `VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10.0)` and saves sidecars: `*.obs_spec.json`, `*.vecnormalize.pkl` next to the PPO zip.
- Evaluation: `evaluate_ppo_agent.py` loads `vecnormalize.pkl`, sets `training=False`, `norm_reward=False`, and uses the saved observation spec.
- Live: `ppo_trading_adapter.py` loads spec + vecnormalize sidecars; if present, observations are normalized (same stats) before inference. Telemetry now reports `obs_norm_present`, pre/post norm summaries, clip fraction, and z-score drift stats.
- Tests: `tests/test_ppo_observation_parity.py` now covers raw parity, normalized parity (VecNormalize), and a sanity check that probabilities vary across fixture indices.
- Fixture: `tests/fixtures/ppo_parity_fixture.npz` is the committed CI fixture; the parity tool can also point at the latest evaluation NPZ (`data/training/BTC_USDT_USDT_1h_train.npz` or similar).
- Live validation checklist:
  - Logs show `obs_norm_present=True`.
  - `obs_clip_frac` low/stable; `z_abs_mean`/`z_abs_p99` reasonable.
  - `p_long` varies over time (not constant to 6 decimals).
  - Gating decisions align with expected thresholds (no always-LONG due to saturation).

## Root-Cause Hypothesis (evidence-driven)
- Previous runs likely suffered from scale mismatch: training had no normalization and different feature mix; live fed large-magnitude/raw inputs. Normalization + unified spec/sidecars remove that mismatch; if saturation persists, focus on drift in input distribution vs training stats.

## Stage 2 Requirements (fix, not yet applied)
- Align observation construction: either retrain with 82+5+2 live spec or remove extras in inference to match training spec.
- Decide on normalization: introduce and persist a scaler/VecNormalize (train + load in inference) or standardize inputs consistently.
- Align tail semantics: match clamping/defaults between train and live or retrain with current live tail behavior.
- Validate manifest/feature order consistency; ensure `.npz` generation uses the same manifest selection as live.

## Stage 5 Additions (live-parity + guards)
- Dataset builder: `scripts/build_ppo_dataset_from_live_pipeline.py` recomputes features via the live FeatureEngineeringPipeline (GEMMA manifest), enforces the shared ObservationSpec, and emits train/val/test NPZs (`data/training/BTC_USDT_USDT_1h_liveparity_*`) plus spec sidecar.
- ObservationSpec now standardizes on **82 manifest features + 5 price extras + 2 tail = 89**; extras computed in both training env and adapter via `compute_price_extras`.
- Health guards in adapter: require vecnorm (configurable), guard on low p_long variance and high obs_clip_frac; fallback with `reason=health_guard` and `action=GUARD_FALLBACK`.
- Calibration tool: `src/tools/ppo_threshold_sweep.py` prints pass-rate curves for any model/dataset/spec to tune `ppo_conf_threshold` and `ppo_min_margin`.
