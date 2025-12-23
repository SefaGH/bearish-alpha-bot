#!/usr/bin/env python3
"""
Compute PPO pass-rate curves over a dataset for threshold calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as th
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_dataset_utils import load_npz_dataset
from src.ml.ppo.observation_spec import (
    DEFAULT_EXTRA_FEATURE_NAMES,
    build_observation,
    compute_price_extras,
    load_spec,
    spec_from_feature_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO threshold sweep for pass-rate calibration.")
    parser.add_argument("--model", type=Path, required=True, help="Path to PPO model .zip")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to NPZ dataset")
    parser.add_argument("--spec", type=Path, help="Optional ObservationSpec JSON (defaults to model sidecar or dataset)")
    parser.add_argument("--threshold-start", type=float, default=0.5)
    parser.add_argument("--threshold-stop", type=float, default=0.8)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.0, help="Minimum p_long - p_flat to count as pass")
    parser.add_argument("--max-samples", type=int, help="Optional cap on samples for quick runs")
    return parser.parse_args()


def load_spec_auto(model_path: Path, dataset_cols: List[str], explicit: Path | None) -> Tuple[Path, object]:
    spec_path = explicit if explicit and explicit.exists() else model_path.with_suffix(".obs_spec.json")
    if spec_path.exists():
        return spec_path, load_spec(spec_path)
    return spec_path, spec_from_feature_columns(dataset_cols, extra_feature_names=DEFAULT_EXTRA_FEATURE_NAMES)


def main() -> None:
    args = parse_args()

    features_df, price_df, _ = load_npz_dataset(args.dataset, min_rows=None)
    model = PPO.load(str(args.model))

    spec_path, spec = load_spec_auto(args.model, features_df.columns.tolist(), args.spec)

    vecnorm_path = args.model.with_suffix(".vecnormalize.pkl")
    vecnorm = None
    if vecnorm_path.exists():
        class _SimpleEnv(gym.Env):
            def __init__(self, obs_dim: int):
                super().__init__()
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(2)
            def reset(self):
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            def step(self, action):
                return self.reset(), 0.0, True, False, {}

        dummy_env = DummyVecEnv([lambda: _SimpleEnv(spec.obs_dim)])
        vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
        vecnorm.training = False
        vecnorm.norm_reward = False

    p_longs: List[float] = []
    p_flats: List[float] = []
    entropies: List[float] = []

    max_idx = len(features_df)
    if args.max_samples:
        max_idx = min(max_idx, args.max_samples)

    for idx in range(max_idx):
        feature_row = features_df.iloc[idx]
        price_slice = price_df.iloc[: idx + 1]
        extra_arr = compute_price_extras(price_slice)
        extra_values = {name: float(extra_arr[i]) for i, name in enumerate(spec.extra_feature_names or [])}
        tail = {"position_fraction": 0.0, "normalized_pv": 1.0}
        obs = build_observation(spec, feature_row, extra_values=extra_values, tail_values=tail)
        obs_batch = obs[np.newaxis, :]
        if vecnorm:
            obs_batch = vecnorm.normalize_obs(obs_batch.copy())
        obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        p_flats.append(float(probs[0]))
        p_longs.append(float(probs[1]))
        entropies.append(float(dist.distribution.entropy().mean().item()))

    p_long_arr = np.array(p_longs)
    p_flat_arr = np.array(p_flats)
    margins = p_long_arr - p_flat_arr
    entropy_arr = np.array(entropies)

    thresholds = np.arange(args.threshold_start, args.threshold_stop + 1e-9, args.threshold_step)
    sweep = []
    for th in thresholds:
        pass_mask = (p_long_arr >= th) & (margins >= args.margin)
        sweep.append(
            {
                "threshold": round(float(th), 4),
                "pass_rate": float(pass_mask.mean()) if len(pass_mask) else 0.0,
                "long_rate": float((p_long_arr >= th).mean()) if len(p_long_arr) else 0.0,
            }
        )

    summary: Dict[str, object] = {
        "samples": int(len(p_long_arr)),
        "spec_path": str(spec_path),
        "p_long": {
            "min": float(p_long_arr.min()) if p_long_arr.size else None,
            "max": float(p_long_arr.max()) if p_long_arr.size else None,
            "mean": float(p_long_arr.mean()) if p_long_arr.size else None,
            "std": float(p_long_arr.std()) if p_long_arr.size else None,
        },
        "entropy": {
            "min": float(entropy_arr.min()) if entropy_arr.size else None,
            "max": float(entropy_arr.max()) if entropy_arr.size else None,
            "mean": float(entropy_arr.mean()) if entropy_arr.size else None,
            "std": float(entropy_arr.std()) if entropy_arr.size else None,
        },
        "sweep": sweep,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
