#!/usr/bin/env python3
"""
Offline evaluator for trained RL checkpoints.

Validation Report JSON Schema:
    {
        "dataset": str,              # Path to validation dataset
        "checkpoint": str | None,    # Path to checkpoint or None
        "symbol": str,               # Trading symbol (e.g., "BTC/USDT:USDT")
        "timeframe": str,            # Timeframe (e.g., "1h")
        "generated_at": str,         # ISO timestamp with 'Z' suffix
        "metrics": {
            "steps": int,            # Number of steps executed
            "total_reward": float,   # Cumulative reward over episode
            "final_pnl": float,      # Final profit/loss
            "action_counts": {       # Raw action counts by index
                "0": int,            # HOLD count
                "1": int,            # BUY count
                "2": int             # SELL count
            },
            "action_distribution": { # Normalized action frequencies
                "0": float,          # HOLD rate (0.0-1.0)
                "1": float,          # BUY rate
                "2": float           # SELL rate
            },
            "q_values": {            # Q-value statistics (per action)
                "mean": [float, float, float],
                "std": [float, float, float],
                "min": [float, float, float],
                "max": [float, float, float]
            },
            "probabilities": {       # Probability statistics (per action)
                "mean": [float, float, float],
                "std": [float, float, float],
                "min": [float, float, float],
                "max": [float, float, float]
            }
        }
    }
"""

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


def _infer_checkpoint_state_size(path: Path) -> int | None:
    """Best-effort inference of checkpoint input dimension for validation."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch may be unavailable in some envs
        logging.debug("PyTorch not available; skipping checkpoint dimension inference for %s", path)
        return None

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - diagnostic path
        logging.warning("Unable to inspect checkpoint %s for dimension check: %s", path, exc)
        return None

    state_dict = checkpoint.get("q_network")
    if not isinstance(state_dict, dict):
        return None

    for key, tensor in state_dict.items():
        if key.endswith("network.0.weight") and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[1])

    return None


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
        inferred_state = _infer_checkpoint_state_size(checkpoint)
        if inferred_state is not None and inferred_state != state_size:
            logging.error(
                "Checkpoint %s expects state size %d, but dataset provides %d features.",
                checkpoint,
                inferred_state,
                state_size,
            )
            raise ValueError(
                f"Checkpoint expects state size {inferred_state}, but dataset provides {state_size} features."
            )
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
    
    action_distribution = {k: v / max(step, 1) for k, v in action_counts.items()}
    
    # === DIAGNOSTICS: Detect degenerate policies ===
    _diagnose_policy_health(action_distribution, total_reward, pnl, q_stats)

    return {
        "steps": step,
        "total_reward": total_reward,
        "final_pnl": pnl,
        "action_counts": action_counts,
        "action_distribution": action_distribution,
        "q_values": q_stats,
        "probabilities": prob_stats,
    }


def _diagnose_policy_health(
    action_distribution: Dict[int, float],
    total_reward: float,
    pnl: float,
    q_stats: Dict[str, Any]
) -> None:
    """
    Emit warnings if the RL policy appears degenerate or frozen.
    
    Checks for:
    - Extreme HOLD dominance (>95% HOLD actions)
    - Zero or near-zero reward/PnL with high HOLD rate
    - Very low Q-value standard deviation (frozen model)
    """
    hold_rate = action_distribution.get(0, 0.0)
    buy_rate = action_distribution.get(1, 0.0)
    sell_rate = action_distribution.get(2, 0.0)
    
    warnings_logged = False
    
    # Check for extreme HOLD dominance
    if hold_rate > 0.95:
        logging.warning("=" * 70)
        logging.warning("⚠️  POLICY HEALTH WARNING: Extreme HOLD dominance detected")
        logging.warning("=" * 70)
        logging.warning(f"   HOLD rate:  {hold_rate:.2%}")
        logging.warning(f"   BUY rate:   {buy_rate:.2%}")
        logging.warning(f"   SELL rate:  {sell_rate:.2%}")
        warnings_logged = True
    
    # Check for zero/near-zero reward and PnL
    if abs(total_reward) < 1e-6 and abs(pnl) < 1e-6 and hold_rate > 0.99:
        if not warnings_logged:
            logging.warning("=" * 70)
        logging.warning("⚠️  POLICY HEALTH WARNING: No-trade / Always-HOLD policy")
        logging.warning(f"   Total Reward: {total_reward:.6f}")
        logging.warning(f"   Final PnL:    {pnl:.6f}")
        logging.warning(f"   HOLD rate:    {hold_rate:.2%}")
        logging.warning("   → Policy appears to be degenerate (no trades, ~0 reward/PnL)")
        warnings_logged = True
    
    # Check Q-value variance (frozen model detection)
    if q_stats.get("std") is not None:
        q_std_array = q_stats["std"]
        if isinstance(q_std_array, list) and len(q_std_array) >= 3:
            avg_q_std = sum(q_std_array) / len(q_std_array)
            if avg_q_std < 1e-5:
                if not warnings_logged:
                    logging.warning("=" * 70)
                logging.warning("⚠️  POLICY HEALTH WARNING: Frozen Q-values detected")
                logging.warning(f"   Average Q-std: {avg_q_std:.2e}")
                logging.warning(f"   Q-std per action: {q_std_array}")
                logging.warning("   → Q-values show almost no variance across states")
                warnings_logged = True
    
    if warnings_logged:
        logging.warning("=" * 70)
        logging.warning("💡 Consider:")
        logging.warning("   - Inspecting training logs for reward scale and Q-value evolution")
        logging.warning("   - Reviewing hyperparameters (learning_rate, reward_scale, epsilon)")
        logging.warning("   - Checking if training converged to a local minimum")
        logging.warning("=" * 70)


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

    # 1. Önce Environment'ı oluştur (ki state_dim'i öğrenebilelim)
    logging.info("Building RL environment...")
    rl_cfg = (config.get("ml") or {}).get("reinforcement_learning", {})
    idle_cost = rl_cfg.get("idle_cost", 0.0)
    
    env = RLTradingEnv(features_df=features_df, raw_df=price_df, idle_cost=idle_cost)
    
    # 2. Gerçek state boyutunu Environment'tan al
    # (Eğer env.state_dim yoksa fallback olarak feature sayısını kullan)
    env_state_size = getattr(env, "state_dim", features_df.shape[1])
    
    if env_state_size != features_df.shape[1]:
        logging.info(
            "🧠 Validation State Size Adjusted: Dataset(%d) -> Agent(%d) (from Env)", 
            features_df.shape[1], env_state_size
        )

    # 3. Ajanı doğru boyutla oluştur
    agent = build_agent(config, state_size=env_state_size, checkpoint=checkpoint)

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
