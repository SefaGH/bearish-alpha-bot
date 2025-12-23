#!/usr/bin/env python3
"""
Offline gating impact backtest using PPO on the same NPZ dataset used for training/eval.

Computes baseline long decisions, applies PPO gating (conf + margin), and reports
P&L/metrics with and without gating plus veto statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as th
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


@dataclass
class BacktestMetrics:
    total_return: float
    max_drawdown: float
    trades: int
    win_rate: float
    avg_trade_pnl: float
    exposure: float
    fees_paid: float


def compute_max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(dd.min())


def simulate_trades(prices: np.ndarray, actions: np.ndarray, fee_pct: float = 0.0006) -> Tuple[np.ndarray, BacktestMetrics]:
    balance = 10_000.0
    position = 0.0  # units of asset
    equity_curve: List[float] = []
    trades = 0
    fees_paid = 0.0
    trade_pnls: List[float] = []
    last_action = 0

    for i, act in enumerate(actions):
        price = float(prices[i])
        # execute action at close of bar i (simplified)
        if act != last_action:
            trades += 1
            if act == 1:  # buy full
                notional = balance
                fee = notional * fee_pct
                qty = (notional - fee) / price if price > 0 else 0.0
                position += qty
                balance -= notional
                fees_paid += fee
            else:  # sell to flat
                notional = position * price
                fee = notional * fee_pct
                balance += max(0.0, notional - fee)
                fees_paid += fee
                # trade PnL relative to previous position
                trade_pnls.append(notional - fee)
                position = 0.0
            last_action = act
        equity = balance + position * price
        equity_curve.append(equity)

    equity_np = np.array(equity_curve, dtype=float)
    total_return = equity_np[-1] / equity_np[0] - 1.0 if equity_np.size > 1 else 0.0
    dd = compute_max_drawdown(equity_np) if equity_np.size else 0.0
    wins = [p for p in trade_pnls if p > 0]
    win_rate = (len(wins) / len(trade_pnls)) if trade_pnls else 0.0
    avg_trade_pnl = float(np.mean(trade_pnls)) if trade_pnls else 0.0
    exposure = float(np.mean(actions)) if actions.size else 0.0
    return equity_np, BacktestMetrics(
        total_return=total_return,
        max_drawdown=dd,
        trades=trades,
        win_rate=win_rate,
        avg_trade_pnl=avg_trade_pnl,
        exposure=exposure,
        fees_paid=fees_paid,
    )


def build_spec(dataset_cols: List[str], spec_path: Path | None) -> Tuple[Path | None, object]:
    if spec_path and spec_path.exists():
        return spec_path, load_spec(spec_path)
    return spec_path, spec_from_feature_columns(dataset_cols, extra_feature_names=DEFAULT_EXTRA_FEATURE_NAMES)


def load_vecnorm(model_path: Path, obs_dim: int | None) -> VecNormalize | None:
    vecnorm_path = model_path.with_suffix(".vecnormalize.pkl")
    if not vecnorm_path.exists():
        return None
    class _SimpleEnv(gym.Env):
        def __init__(self, obs_dim: int):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)
        def reset(self):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        def step(self, action):
            return self.reset(), 0.0, True, False, {}
    dummy = DummyVecEnv([lambda: _SimpleEnv(obs_dim or 1)])
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def compute_baseline_actions(prices: pd.Series, mode: str) -> np.ndarray:
    closes = prices.to_numpy(dtype=float)
    actions = np.zeros_like(closes, dtype=int)
    if mode == "always_long":
        actions[:] = 1
        return actions
    if mode == "price_up":
        actions[1:] = (closes[1:] > closes[:-1]).astype(int)
        return actions
    if mode == "ema_cross":
        ema_fast = pd.Series(closes).ewm(span=12, adjust=False).mean().to_numpy()
        ema_slow = pd.Series(closes).ewm(span=26, adjust=False).mean().to_numpy()
        actions[:] = (ema_fast > ema_slow).astype(int)
        return actions
    return actions  # default flat


def build_state(spec, features_df: pd.DataFrame, price_df: pd.DataFrame, idx: int):
    feature_row = features_df.iloc[idx]
    extra_arr = compute_price_extras(price_df.iloc[: idx + 1])
    extra_values = {name: float(extra_arr[i]) for i, name in enumerate(spec.extra_feature_names or [])}
    tail = {"position_fraction": 0.0, "normalized_pv": 1.0}
    return build_observation(spec, feature_row, extra_values=extra_values, tail_values=tail)


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO gating impact backtest.")
    parser.add_argument("--model", type=Path, required=True, help="Path to PPO model .zip")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to NPZ dataset")
    parser.add_argument("--spec", type=Path, help="Optional obs spec (defaults to model sidecar)")
    parser.add_argument("--baseline-mode", choices=["always_long", "price_up", "ema_cross", "flat"], default="always_long")
    parser.add_argument("--conf-threshold", type=float, default=0.60)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, help="Optional path to write summary JSON")
    args = parser.parse_args()

    features_df, price_df, _ = load_npz_dataset(args.dataset, min_rows=None)
    spec_path, spec = build_spec(features_df.columns.tolist(), args.spec or args.model.with_suffix(".obs_spec.json"))
    model = PPO.load(str(args.model))
    vecnorm = load_vecnorm(args.model, spec.obs_dim if hasattr(spec, "obs_dim") else None)

    closes = price_df["close"].reset_index(drop=True)
    baseline_actions = compute_baseline_actions(closes, args.baseline_mode)

    p_flat_list: List[float] = []
    p_long_list: List[float] = []
    gated_actions: List[int] = []
    vetoes = 0
    veto_losses = 0

    for idx in range(len(features_df)):
        state = build_state(spec, features_df, price_df, idx)
        obs_batch = state[np.newaxis, :]
        if vecnorm:
            obs_batch = vecnorm.normalize_obs(obs_batch.copy())
        obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        p_flat = float(probs[0])
        p_long = float(probs[1])
        p_flat_list.append(p_flat)
        p_long_list.append(p_long)

        base_act = int(baseline_actions[idx])
        gated_act = base_act
        if base_act == 1:
            margin = p_long - p_flat
            if not (p_long >= args.conf_threshold and margin >= args.min_margin and np.argmax(probs) == 1):
                gated_act = 0
                vetoes += 1
                # track whether the baseline trade would have lost on the next bar
                if idx + 1 < len(closes):
                    pnl = (closes.iloc[idx + 1] - closes.iloc[idx]) / closes.iloc[idx]
                    if pnl < 0:
                        veto_losses += 1
        gated_actions.append(gated_act)

    baseline_equity, baseline_metrics = simulate_trades(closes.to_numpy(), baseline_actions)
    gated_equity, gated_metrics = simulate_trades(closes.to_numpy(), np.array(gated_actions))

    summary: Dict[str, object] = {
        "dataset": str(args.dataset),
        "model": str(args.model),
        "spec": str(spec_path) if spec_path else None,
        "baseline_mode": args.baseline_mode,
        "thresholds": {"conf": args.conf_threshold, "min_margin": args.min_margin},
        "p_long": {
            "min": float(np.min(p_long_list)),
            "max": float(np.max(p_long_list)),
            "mean": float(np.mean(p_long_list)),
            "std": float(np.std(p_long_list)),
        },
        "vetoes": int(vetoes),
        "veto_on_losing_trades": int(veto_losses),
        "veto_rate": float(vetoes / max(1, int((baseline_actions == 1).sum()))),
        "baseline": baseline_metrics.__dict__,
        "gated": gated_metrics.__dict__,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
