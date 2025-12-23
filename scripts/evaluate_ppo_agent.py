#!/usr/bin/env python3
"""
Evaluate a trained PPO trading agent on a given dataset (backtest-style).

- *_test.npz datasetini kullanır.
- RLTradingEnvGym üzerinde deterministic (greedy) policy ile tek episode koşar.
- PnL, MaxDD, trade sayısı ve equity curve üretir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_dataset_utils import load_npz_dataset
from src.ml.rl_trading_env_gym import RLTradingEnvGym
from src.ml.ppo.observation_spec import load_spec, spec_from_feature_columns


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def compute_trade_stats(actions: List[int]) -> Dict[str, Any]:
    if not actions:
        return {"num_trades": 0, "position_changes": 0}
    changes = sum(1 for a, b in zip(actions[:-1], actions[1:]) if a != b)
    return {"num_trades": changes, "position_changes": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO trading agent on dataset.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to PPO model .zip file (from Stable-Baselines3).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to *_test.npz dataset.",
    )
    parser.add_argument(
        "--obs-spec",
        type=Path,
        help="Optional ObservationSpec JSON. Defaults to model sidecar, then dataset sibling.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("data/training/ppo_eval_summary.json"),
    )
    parser.add_argument(
        "--output-equity-curve",
        type=Path,
        default=Path("data/training/ppo_eval_equity_curve.csv"),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    logging.info("Loading dataset from %s", args.dataset)
    features_df, price_df, metadata = load_npz_dataset(args.dataset)

    env_config = {
        "fee_pct": metadata.get("fee_pct", 0.0006) if isinstance(metadata, dict) else 0.0006,
        "trade_penalty_alpha": 0.0,
        "idle_cost": 0.0,
        "reward_clip_enabled": True,
        "reward_clip_min": -1.0,
        "reward_clip_max": 1.0,
    }
    initial_balance = float(metadata.get("initial_balance", 10_000.0)) if isinstance(metadata, dict) else 10_000.0

    def _resolve_spec_path() -> Optional[Path]:
        if args.obs_spec:
            return args.obs_spec
        candidates = [args.model.with_suffix(".obs_spec.json")]
        stem = args.dataset.stem
        for suffix in ("_train", "_test", "_val"):
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                candidates.append(args.dataset.parent / f"{base}.obs_spec.json")
                break
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    spec_path = _resolve_spec_path()
    if spec_path and spec_path.exists():
        obs_spec = load_spec(spec_path)
    else:
        obs_spec = spec_from_feature_columns(features_df.columns)

    def _env_fn():
        return RLTradingEnvGym(
            features_df=features_df,
            raw_df=price_df,
            config=env_config,
            initial_balance=initial_balance,
            observation_spec=obs_spec,
        )

    vec_env = DummyVecEnv([_env_fn])
    vecnorm_path = args.model.with_suffix(".vecnormalize.pkl")
    if vecnorm_path.exists():
        logging.info("Loading VecNormalize stats from %s", vecnorm_path)
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        logging.warning("VecNormalize stats not found; running evaluation without normalization.")

    logging.info("Loading PPO model from %s", args.model)
    model = PPO.load(str(args.model))
    if vecnorm_path.exists():
        model.set_env(vec_env)

    obs = vec_env.reset()
    done = False

    equity_curve: List[float] = []
    rewards: List[float] = []
    actions: List[int] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_array, info = vec_env.step([int(action)])
        done = bool(done_array[0]) if isinstance(done_array, (list, np.ndarray)) else bool(done_array)

        rewards.append(float(reward[0] if isinstance(reward, (list, np.ndarray)) else reward))
        actions.append(int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action))
        info_obj = info[0] if isinstance(info, (list, tuple)) else info
        equity_curve.append(float(info_obj.get("portfolio_value")))

    equity_np = np.array(equity_curve, dtype=float)
    final_value = equity_np[-1] if equity_np.size > 0 else initial_balance
    pnl = final_value - initial_balance
    total_return = (final_value / initial_balance - 1.0) if initial_balance > 0 else 0.0
    max_dd = compute_max_drawdown(equity_np) if equity_np.size > 0 else 0.0
    trade_stats = compute_trade_stats(actions)

    unique_actions = sorted(set(actions)) if actions else []

    summary = {
        "steps": len(equity_curve),
        "initial_balance": initial_balance,
        "final_portfolio_value": float(final_value),
        "total_pnl": float(pnl),
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "num_trades": trade_stats["num_trades"],
        "position_changes": trade_stats["position_changes"],
        "unique_actions": unique_actions,
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    args.output_equity_curve.parent.mkdir(parents=True, exist_ok=True)
    with args.output_equity_curve.open("w", encoding="utf-8") as f:
        f.write("step,equity\n")
        for i, v in enumerate(equity_curve):
            f.write(f"{i},{v}\n")

    logging.info(
        "Evaluation done. PnL=%.2f, return=%.4f, maxDD=%.4f, trades=%d",
        summary["total_pnl"],
        summary["total_return"],
        summary["max_drawdown"],
        summary["num_trades"],
    )


if __name__ == "__main__":
    main()
