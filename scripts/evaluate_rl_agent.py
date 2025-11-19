#!/usr/bin/env python3
"""
Evaluate a trained RL trading agent on a given dataset (backtest-style).

- Loads a *.npz dataset (features + prices) via rl_dataset_utils.load_npz_dataset
- Builds RLTradingEnv with the same config style as training
- Loads a trained TradingRLAgent model
- Runs a single deterministic episode (greedy policy, no epsilon exploration)
- Computes and prints basic performance metrics
- Optionally saves equity curve and summary to disk
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project imports
from scripts.rl_dataset_utils import load_npz_dataset
from src.ml.rl_trading_env import RLTradingEnv
from src.ml.reinforcement_learning import TradingRLAgent


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_max_drawdown(equity: np.ndarray) -> float:
    """Compute max drawdown from an equity curve."""
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def compute_trade_stats(actions: List[int]) -> Dict[str, Any]:
    """
    Rough trade stats from action sequence.
    For target-position actions [0,1,2], we count a trade whenever action changes.
    """
    if not actions:
        return {"num_trades": 0, "position_changes": 0}

    changes = 0
    for prev, curr in zip(actions[:-1], actions[1:]):
        if prev != curr:
            changes += 1

    return {
        "num_trades": changes,
        "position_changes": changes,
    }


def run_evaluation_episode(
    env: RLTradingEnv,
    agent: TradingRLAgent,
) -> Dict[str, Any]:
    """Run a single greedy episode and collect metrics."""
    state = env.reset()
    done = False

    equity_curve: List[float] = []
    rewards: List[float] = []
    actions: List[int] = []

    step_count = 0
    initial_balance = env.initial_balance

    while not done:
        # Deterministic (greedy) action selection:
        # Prefer an explicit greedy flag if TradingRLAgent supports it,
        # otherwise temporarily force epsilon=0.
        if hasattr(agent, "act"):
            try:
                action = agent.act(state, greedy=True)
            except TypeError:
                # Fallback: try without greedy arg, but force epsilon=0
                old_eps = getattr(agent, "epsilon", 0.0)
                setattr(agent, "epsilon", 0.0)
                action = agent.act(state)
                setattr(agent, "epsilon", old_eps)
        else:
            raise RuntimeError("TradingRLAgent must implement an 'act' method.")

        next_state, reward, done, info = env.step(action)

        rewards.append(float(reward))
        actions.append(int(action))
        equity_curve.append(float(info.get("portfolio_value", np.nan)))

        state = next_state
        step_count += 1

    equity_np = np.array(equity_curve, dtype=float)
    final_value = equity_np[-1] if equity_np.size > 0 else initial_balance
    pnl = final_value - initial_balance
    ret = (final_value / initial_balance - 1.0) if initial_balance > 0 else 0.0
    max_dd = compute_max_drawdown(equity_np) if equity_np.size > 0 else 0.0

    trade_stats = compute_trade_stats(actions)

    metrics: Dict[str, Any] = {
        "steps": step_count,
        "initial_balance": float(initial_balance),
        "final_portfolio_value": float(final_value),
        "total_pnl": float(pnl),
        "total_return": float(ret),
        "max_drawdown": float(max_dd),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "num_trades": trade_stats["num_trades"],
        "position_changes": trade_stats["position_changes"],
    }

    # Keep raw sequences in case caller wants to save them
    metrics["equity_curve"] = equity_curve
    metrics["rewards"] = rewards
    metrics["actions"] = actions

    return metrics


def save_equity_curve(
    path: Path,
    equity_curve: List[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("step,equity\n")
        for i, v in enumerate(equity_curve):
            f.write(f"{i},{v}\n")


def save_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Remove large arrays before saving summary; keep only aggregates
    summary_to_save = dict(summary)
    summary_to_save.pop("equity_curve", None)
    summary_to_save.pop("rewards", None)
    summary_to_save.pop("actions", None)

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary_to_save, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL trading agent on a given dataset.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained RL agent model file (e.g. rl_agent_final.pth)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to *.npz dataset file (typically *_test.npz).",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=None,
        help="Override initial_balance for the environment (default: use env default, usually 10000).",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("data/training/rl_eval_summary.json"),
        help="Where to save evaluation summary JSON.",
    )
    parser.add_argument(
        "--output-equity-curve",
        type=Path,
        default=Path("data/training/rl_eval_equity_curve.csv"),
        help="Where to save equity curve CSV.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)

    logging.info("Loading dataset from %s", args.dataset)
    features_df, price_df, metadata = load_npz_dataset(args.dataset)

    # Build env config from metadata if available, with sane defaults
    env_config = {
        "fee_pct": metadata.get("fee_pct", 0.0006) if isinstance(metadata, dict) else 0.0006,
        "reward_clip_enabled": True,
        "reward_clip_min": -1.0,
        "reward_clip_max": 1.0,
        "reward_scale": 1.0,
        "trade_penalty_alpha": 0.001,
        # idle_cost can be tuned here if desired
        # "idle_cost": 0.0,
    }

    initial_balance = (
        float(args.initial_balance)
        if args.initial_balance is not None
        else 10000.0
    )

    logging.info(
        "Building RLTradingEnv with initial_balance=%.2f, fee_pct=%.6f",
        initial_balance,
        env_config["fee_pct"],
    )

    env = RLTradingEnv(
        features_df=features_df,
        raw_df=price_df,
        config=env_config,
        initial_balance=initial_balance,
    )

    state_size = env.state_dim
    action_size = 3  # TARGET_0.0, TARGET_0.5, TARGET_1.0

    logging.info(
        "Initializing TradingRLAgent with state_size=%d, action_size=%d",
        state_size,
        action_size,
    )
    agent = TradingRLAgent(state_size=state_size, action_size=action_size)

    logging.info("Loading trained model from %s", args.model)
    agent.load_model(str(args.model))

    logging.info("Running deterministic evaluation episode (greedy policy, no exploration)...")
    metrics = run_evaluation_episode(env, agent)

    # Print human-readable summary
    print("=== RL Agent Evaluation Summary ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Steps: {metrics['steps']}")
    print(f"Initial balance: {metrics['initial_balance']:.2f}")
    print(f"Final portfolio value: {metrics['final_portfolio_value']:.2f}")
    print(f"Total PnL: {metrics['total_pnl']:.2f}")
    print(f"Total return: {metrics['total_return']:.4f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.4f}")
    print(f"Num trades (position changes): {metrics['num_trades']}")
    print(f"Average reward per step: {metrics['avg_reward']:.6f}")
    print(f"Reward std: {metrics['std_reward']:.6f}")

    # Save artifacts
    if metrics.get("equity_curve"):
        save_equity_curve(args.output_equity_curve, metrics["equity_curve"])
        logging.info("Equity curve saved to %s", args.output_equity_curve)

    save_summary(args.output_summary, metrics)
    logging.info("Evaluation summary saved to %s", args.output_summary)


if __name__ == "__main__":
    main()
