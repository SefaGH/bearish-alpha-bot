#!/usr/bin/env python3
"""
PPO observation parity checker (diagnostics only).

Compares training-style observations (as produced by RLTradingEnvGym) against
live-style observations (as produced by PPOTradingAdapter logic) on the same
snapshot. Prints vector diffs and PPO distribution stats for both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_dataset_utils import load_npz_dataset
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.adapters.ppo_trading_adapter import PPOTradingAdapter
from src.ml.ppo.observation_spec import load_spec, spec_from_feature_columns, build_observation


EXTRA_FEATURE_NAMES: List[str] = [
    "extra_ret_1",
    "extra_ret_3",
    "extra_range_norm",
    "extra_vol_10",
    "extra_trend_ema_ratio",
]
TAIL_NAMES: List[str] = ["position_fraction", "normalized_pv"]


def build_training_obs(features_df: pd.DataFrame, idx: int, spec: Optional[Any]) -> Tuple[np.ndarray, List[str]]:
    if spec is None:
        row = features_df.iloc[idx].to_numpy(dtype=np.float32)
        tail = np.array([0.0, 1.0], dtype=np.float32)
        obs = np.concatenate([row, tail]).astype(np.float32)
        names = list(features_df.columns) + TAIL_NAMES
        return obs, names
    tail_values = {"position_fraction": 0.0, "normalized_pv": 1.0}
    extra_values = {name: 0.0 for name in spec.extra_feature_names}
    obs = build_observation(spec, features_df.iloc[idx], extra_values=extra_values, tail_values=tail_values)
    names = list(spec.feature_names) + list(spec.extra_feature_names) + list(spec.tail_names)
    return obs, names


def build_live_obs(price_df: pd.DataFrame, feature_pipeline: FeatureEngineeringPipeline, idx: int, spec: Optional[Any]) -> Tuple[np.ndarray, List[str]]:
    features_df = feature_pipeline.extract_features(price_df, mode="price")
    if features_df is None or features_df.empty:
        raise RuntimeError("FeatureEngineeringPipeline returned empty features for live obs.")
    features_df = features_df.iloc[: len(price_df)]
    latest_row = features_df.iloc[idx]
    if spec is None:
        extra_values = {name: 0.0 for name in EXTRA_FEATURE_NAMES}
        tail_values = {"position_fraction": 0.0, "normalized_pv": 1.0}
        obs = build_observation(
            spec_from_feature_columns(features_df.columns, extra_feature_names=EXTRA_FEATURE_NAMES),
            latest_row,
            extra_values=extra_values,
            tail_values=tail_values,
        )
        names = list(features_df.columns) + EXTRA_FEATURE_NAMES + TAIL_NAMES
        return obs, names

    extra_values: Dict[str, float] = {name: 0.0 for name in spec.extra_feature_names}
    tail_values = {"position_fraction": 0.0, "normalized_pv": 1.0}
    obs = build_observation(
        spec,
        latest_row,
        extra_values=extra_values,
        tail_values=tail_values,
    )
    names = list(spec.feature_names) + list(spec.extra_feature_names) + list(spec.tail_names)
    return obs, names


def align_for_model(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[0] == target_dim:
        return vec.astype(np.float32)
    if vec.shape[0] > target_dim:
        return vec[:target_dim].astype(np.float32)
    padded = np.concatenate([vec, np.zeros(target_dim - vec.shape[0], dtype=np.float32)])
    return padded.astype(np.float32)


def describe_vector(vec: np.ndarray) -> Dict[str, float]:
    return {
        "len": int(vec.shape[0]),
        "mean": float(np.mean(vec)),
        "std": float(np.std(vec)),
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
    }


def top_diffs(a: np.ndarray, b: np.ndarray, names: List[str], top_k: int = 20) -> List[Tuple[int, str, float, float, float]]:
    diffs = np.abs(a - b)
    idxs = np.argsort(diffs)[::-1][:top_k]
    rows: List[Tuple[int, str, float, float, float]] = []
    for i in idxs:
        name = names[i] if i < len(names) else f"idx_{i}"
        rows.append((int(i), name, float(a[i]), float(b[i]), float(diffs[i])))
    return rows


def get_policy_stats(model: PPO, obs: np.ndarray) -> Dict[str, float]:
    with th.no_grad():
        obs_tensor = th.as_tensor(obs[np.newaxis, :], device=model.device)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]
        entropy = float(dist.distribution.entropy().mean().item())
        logits = getattr(dist.distribution, "logits", None)
        logits_vals = logits.cpu().numpy()[0].tolist() if logits is not None else None
    return {
        "p_flat": float(probs[0]) if probs.size > 0 else None,
        "p_long": float(probs[1]) if probs.size > 1 else None,
        "entropy": entropy,
        "logits": logits_vals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO observation parity checker.")
    parser.add_argument("--model", type=Path, required=True, help="Path to ppo_trading_agent.zip")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to *_train/test npz dataset")
    parser.add_argument("--index", type=int, default=-1, help="Row index to inspect (default: last)")
    args = parser.parse_args()

    features_df, price_df, _ = load_npz_dataset(args.dataset, min_rows=None)
    idx = args.index if args.index >= 0 else len(features_df) - 1
    feature_pipe = FeatureEngineeringPipeline()

    spec_path = args.model.with_suffix(".obs_spec.json")
    spec = load_spec(spec_path) if spec_path.exists() else None

    training_obs, training_names = build_training_obs(features_df, idx, spec)
    live_obs, live_names = build_live_obs(price_df, feature_pipe, idx, spec)

    model = PPO.load(str(args.model))
    target_dim = int(model.observation_space.shape[0])

    training_obs_aligned = align_for_model(training_obs, target_dim)
    live_obs_aligned = align_for_model(live_obs, target_dim)

    print("=== Observation Shapes ===")
    print(f"Training_obs_len={len(training_obs)} (aligned {len(training_obs_aligned)})")
    print(f"Live_obs_len={len(live_obs)} (aligned {len(live_obs_aligned)})")

    print("\n=== Vector Stats ===")
    print("Training:", describe_vector(training_obs_aligned))
    print("Live    :", describe_vector(live_obs_aligned))

    diffs = top_diffs(training_obs_aligned, live_obs_aligned, live_names)
    print("\n=== Top Diff Indices (training vs live aligned) ===")
    for i, name, a, b, d in diffs:
        print(f"{i:3d} {name:32s} train={a:+.6f} live={b:+.6f} diff={d:+.6f}")

    print("\n=== Policy Stats ===")
    stats_train = get_policy_stats(model, training_obs_aligned)
    stats_live = get_policy_stats(model, live_obs_aligned)
    print("Training obs ->", json.dumps(stats_train))
    print("Live obs     ->", json.dumps(stats_live))


if __name__ == "__main__":
    main()
