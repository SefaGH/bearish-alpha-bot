#!/usr/bin/env python3
"""
Reward audit for PPO environment:
- Evaluate always-flat, always-long, and random policies on the PPO dataset/env.
- Report reward stats, equity returns, drawdown, trades, and exposure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gym
from stable_baselines3.common.vec_env import DummyVecEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_dataset_utils import load_npz_dataset
from src.ml.ppo.observation_spec import DEFAULT_EXTRA_FEATURE_NAMES, load_spec, spec_from_feature_columns
from src.ml.rl_trading_env_gym import RLTradingEnvGym


def compute_max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(dd.min())


def run_policy(env: RLTradingEnvGym, actions: np.ndarray) -> Dict[str, float]:
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    equity_curve: List[float] = [info.get("portfolio_value", 0.0)]
    rewards: List[float] = []
    positions: List[float] = []
    trades = 0

    for idx, act in enumerate(actions):
        step_out = env.step(int(act))
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done, info = step_out
        else:
            raise RuntimeError("Unexpected step output format")

        rewards.append(float(reward))
        positions.append(info.get("position_fraction", 0.0) if isinstance(info, dict) else 0.0)
        equity_curve.append(info.get("portfolio_value", equity_curve[-1]) if isinstance(info, dict) else equity_curve[-1])
        if idx + 1 < len(actions) and actions[idx + 1] != act:
            trades += 1
        if done:
            break

    equity_np = np.array(equity_curve, dtype=float)
    total_return = equity_np[-1] / equity_np[0] - 1.0 if equity_np.size > 1 else 0.0
    return {
        "steps": len(rewards),
        "trades": trades,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "total_return": total_return,
        "max_drawdown": compute_max_drawdown(equity_np),
        "exposure": float(np.mean(positions)) if positions else 0.0,
        "final_equity": float(equity_np[-1]) if equity_np.size else 0.0,
    }


def build_env(features_df, price_df, spec):
    env_cfg = {"fee_pct": 0.0006, "idle_cost": 0.0, "reward_clip_enabled": True, "reward_clip_min": -1.0, "reward_clip_max": 1.0}
    return RLTradingEnvGym(features_df=features_df, raw_df=price_df, config=env_cfg, initial_balance=10_000.0, observation_spec=spec)


def load_spec_auto(model_path: Path | None, dataset_cols: List[str]):
    if model_path:
        sidecar = model_path.with_suffix(".obs_spec.json")
        if sidecar.exists():
            return load_spec(sidecar)
    return spec_from_feature_columns(dataset_cols, extra_feature_names=DEFAULT_EXTRA_FEATURE_NAMES)


def load_vecnorm(model_path: Path, obs_dim: int) -> VecNormalize | None:
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PPO reward structure with reference policies.")
    parser.add_argument("--dataset", type=Path, required=True, help="NPZ dataset")
    parser.add_argument("--model", type=Path, help="Optional model path to load spec/vecnorm sidecars")
    parser.add_argument("--max-steps", type=int, help="Optional cap on steps")
    args = parser.parse_args()

    features_df, price_df, _ = load_npz_dataset(args.dataset, min_rows=None)
    spec = load_spec_auto(args.model, features_df.columns.tolist())
    env = build_env(features_df, price_df, spec)

    steps = len(features_df)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    actions_flat = np.zeros(steps, dtype=int)
    actions_long = np.ones(steps, dtype=int)
    rng = np.random.default_rng(seed=42)
    actions_random = rng.integers(0, 2, size=steps)

    results = {
        "always_flat": run_policy(env, actions_flat),
        "always_long": run_policy(env, actions_long),
        "random": run_policy(env, actions_random),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
