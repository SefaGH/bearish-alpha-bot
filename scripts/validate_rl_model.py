#!/usr/bin/env python3
"""Offline evaluator for trained RL checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
def _json_default(obj: Any) -> Any:
    """Coerce numpy values to native Python types for JSON serialization."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rl_dataset_utils import load_npz_dataset  # noqa: E402
from src.config.live_trading_config import LiveTradingConfiguration  # noqa: E402
from src.ml.reinforcement_learning import TradingRLAgent  # noqa: E402
from src.ml.rl_trading_env import RLTradingEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained RL checkpoint against a dataset split.")
    parser.add_argument("--dataset", required=True, help="Path to *.npz dataset (val/test split)")
    parser.add_argument(
        "--checkpoint",
        default="data/models/rl_agent_final.pth",
        help="Path to trained RL checkpoint",
    )
    parser.add_argument("--config", default="config/config.example.yaml", help="Config file for manifests")
    parser.add_argument("--max-steps", type=int, help="Optional cap on evaluation steps")
    parser.add_argument(
        "--report-file",
        default="data/training/rl_validation_report.json",
        help="Where to store evaluation metrics",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...)")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_agent(config: Dict[str, Any], state_size: int, checkpoint: Path | None) -> TradingRLAgent:
    rl_cfg = (config.get("ml") or {}).get("reinforcement_learning", {})
    rl_cfg = dict(rl_cfg)
    rl_cfg["training_mode"] = False
    agent = TradingRLAgent(state_size=state_size, action_size=3, config=rl_cfg)
    agent.set_inference_mode(epsilon=rl_cfg.get("epsilon_inference", 0.01))
    if checkpoint:
        logging.info("Loading checkpoint %s", checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        try:
            agent.load_model(str(checkpoint))
        except Exception as exc:  # noqa: BLE001 - surface detailed context for diagnostics
            raise RuntimeError(f"Failed to load checkpoint {checkpoint}: {exc}") from exc
    return agent


def evaluate_agent(agent: TradingRLAgent, env: RLTradingEnv, max_steps: int | None) -> Dict[str, Any]:
    state = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    last_info: Dict[str, Any] = {"pnl": 0.0}
    action_counts = {0: 0, 1: 0, 2: 0}
    q_history: List[List[float]] = []
    prob_history: List[List[float]] = []

    while not done:
        action, meta = agent.get_action_with_meta(state, training=False)
        action_counts[action] += 1
        q_vals = meta.get("raw_q_values") or meta.get("adjusted_q_values")
        probs = meta.get("probabilities")
        if q_vals is not None:
            q_history.append(q_vals)
        if probs is not None:
            prob_history.append(probs)

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        last_info = info
        state = next_state
        step += 1
        if max_steps and step >= max_steps:
            logging.warning("Max step cap reached (%d). Ending evaluation early.", max_steps)
            break

    pnl = last_info.get("pnl", 0.0)
    q_stats = summarize_vector_history(q_history)
    prob_stats = summarize_vector_history(prob_history)

    return {
        "steps": step,
        "total_reward": total_reward,
        "final_pnl": pnl,
        "action_counts": action_counts,
        "action_distribution": {k: v / max(step, 1) for k, v in action_counts.items()},
        "q_values": q_stats,
        "probabilities": prob_stats,
    }


def summarize_vector_history(history: List[List[float]]) -> Dict[str, Any]:
    if not history:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(history, dtype=np.float32)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    logging.info("Validation report written to %s", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    dataset_path = Path(args.dataset)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    report_path = Path(args.report_file)

    logging.info("Loading dataset %s", dataset_path)
    features_df, price_df, meta = load_npz_dataset(dataset_path)

    config = LiveTradingConfiguration.load(config_path=args.config, log_summary=False)
    agent = build_agent(config, state_size=features_df.shape[1], checkpoint=checkpoint)
    env = RLTradingEnv(features_df=features_df, raw_df=price_df)

    metrics = evaluate_agent(agent, env, args.max_steps)
    payload = {
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "symbol": meta.get("symbol"),
        "timeframe": meta.get("timeframe"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
    }

    logging.info(
        "Evaluation complete: steps=%d reward=%.4f pnl=%.2f",
        metrics["steps"],
        metrics["total_reward"],
        metrics["final_pnl"],
    )
    write_report(report_path, payload)


if __name__ == "__main__":
    main()
