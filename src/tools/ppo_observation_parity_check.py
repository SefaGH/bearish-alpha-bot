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
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_dataset_utils import load_npz_dataset
def _load_price_df_with_ts(npz_path: Path) -> pd.DataFrame:
    with np.load(npz_path, allow_pickle=True) as data:
        prices = data["prices"]
        price_columns = data["price_columns"].tolist()
        timestamps = pd.to_datetime(data["timestamps"])
    pdf = pd.DataFrame(prices, columns=price_columns)
    pdf["timestamp"] = timestamps
    return pdf.set_index("timestamp").sort_index()


def _load_full_price_history_from_sibling_splits(dataset_path: Path) -> pd.DataFrame:
    """
    If dataset is a split like <base>_{train|val|test}.npz, load sibling split prices
    and concatenate to reconstruct full history. Fallback: return this dataset's price_df.
    """
    features_df, price_df, _ = load_npz_dataset(dataset_path, min_rows=None)
    name = dataset_path.name
    suffixes = ["_train.npz", "_val.npz", "_test.npz"]
    base = None
    for s in suffixes:
        if name.endswith(s):
            base = name[: -len(s)]
            break
    if base is None:
        return price_df

    paths = [dataset_path.parent / f"{base}{s}" for s in suffixes]
    if not all(p.exists() for p in paths):
        return price_df

    parts = []
    for p in paths:
        parts.append(_load_price_df_with_ts(p))

    full = pd.concat(parts, axis=0)
    full = full[~full.index.duplicated(keep="first")]
    full = full.sort_index()
    return full
from src.config.live_trading_config import LiveTradingConfiguration
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.ppo.deterministic_scaler import DeterministicScaler
from src.ml.ppo.observation_spec import (
    compute_price_extras,
    load_spec,
)

TAIL_DEFAULTS: Dict[str, float] = {"position_fraction": 0.0, "normalized_pv": 1.0}
EXTRA_LOOKBACK_BARS = 256  # parity/debug amaçlı; adapter lookback'ine yakın tutulur


def _extras_for_index(price_df: pd.DataFrame, idx: int, extra_names: List[str]) -> Dict[str, float]:
    if not extra_names:
        return {}
    window = price_df.iloc[: idx + 1].tail(EXTRA_LOOKBACK_BARS)
    arr = compute_price_extras(window)
    try:
        if getattr(arr, "ndim", 1) > 1:
            arr = arr[-1]
    except Exception:
        pass
    return {name: float(arr[i]) for i, name in enumerate(extra_names)}


def build_training_obs(
    features_df: pd.DataFrame,
    price_df: pd.DataFrame,
    idx: int,
    spec: Any,
    scaler: DeterministicScaler,
    *,
    price_idx: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    series = features_df.iloc[idx]
    row_dict = {name: float(series.get(name, 0.0)) for name in spec.feature_names}
    missing = [name for name in spec.feature_names if name not in series]
    if missing:
        logging.warning("Training obs missing %d features; filling 0.0: %s", len(missing), missing[:5])
    price_row_idx = idx if price_idx is None else price_idx
    row_dict.update(_extras_for_index(price_df, price_row_idx, list(spec.extra_feature_names or [])))
    row_dict.update(TAIL_DEFAULTS)
    close_price = float(price_df.iloc[price_row_idx]["close"])
    obs = scaler.transform(row_dict, close_price).astype(np.float32)
    return obs, list(scaler.feature_names)


def build_live_obs(
    price_df: pd.DataFrame,
    feature_pipeline: FeatureEngineeringPipeline,
    idx: int,
    spec: Any,
    scaler: DeterministicScaler,
) -> Tuple[np.ndarray, List[str]]:
    features_df = feature_pipeline.extract_features(price_df, mode="price")
    if features_df is None or features_df.empty:
        raise RuntimeError("FeatureEngineeringPipeline returned empty features for live obs.")
    features_df = features_df.iloc[: len(price_df)]
    latest_row = features_df.iloc[idx]
    row_dict = {name: float(latest_row.get(name, 0.0)) for name in spec.feature_names}
    missing = [name for name in spec.feature_names if name not in latest_row]
    if missing:
        logging.warning("Live obs missing %d features; filling 0.0: %s", len(missing), missing[:5])
    row_dict.update(_extras_for_index(price_df, idx, list(spec.extra_feature_names or [])))
    row_dict.update(TAIL_DEFAULTS)
    close_price = float(price_df.iloc[idx]["close"])
    obs = scaler.transform(row_dict, close_price).astype(np.float32)
    return obs, list(scaler.feature_names)


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
    parser.add_argument("--config", type=Path, default=Path("config/config.example.yaml"), help="Config path for FeatureEngineeringPipeline.")
    args = parser.parse_args()

    features_df, price_df, _ = load_npz_dataset(args.dataset, min_rows=None)
    idx = args.index if args.index >= 0 else len(features_df) - 1

    # Reconstruct full history for cumulative indicators (e.g., OBV)
    price_df_full = _load_full_price_history_from_sibling_splits(args.dataset)

    # Map split-index -> full-history index by timestamp
    with np.load(args.dataset, allow_pickle=True) as data:
        ts_split = pd.to_datetime(data["timestamps"])
    ts = ts_split[idx]
    if ts not in price_df_full.index:
        raise SystemExit(f"Timestamp {ts} not found in reconstructed full price history.")
    idx_full = int(price_df_full.index.get_loc(ts))

    cfg = LiveTradingConfiguration.load(config_path=args.config, log_summary=False)
    feature_pipe = FeatureEngineeringPipeline(config=cfg)

    spec_path = args.model.with_suffix(".obs_spec.json")
    if not spec_path.exists():
        raise SystemExit(f"Missing spec sidecar: {spec_path}")
    spec = load_spec(spec_path)
    scaler = DeterministicScaler(spec_path)

    training_obs, training_names = build_training_obs(
        features_df,
        price_df_full,
        idx,
        spec,
        scaler,
        price_idx=idx_full,
    )
    live_obs, live_names = build_live_obs(price_df_full, feature_pipe, idx_full, spec, scaler)

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
