#!/usr/bin/env python3
"""Command-line entrypoint for training the RL agent on prepared datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.config.live_trading_config import LiveTradingConfiguration  # noqa: E402
from src.ml.reinforcement_learning import ExperienceReplay, TradingRLAgent  # noqa: E402
from src.ml.rl_model_trainer import RLModelTrainer  # noqa: E402
from src.ml.rl_trading_env import RLTradingEnv  # noqa: E402

from rl_dataset_utils import PRICE_COLUMNS, load_npz_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RL agent using a prepared dataset.")
    parser.add_argument("--dataset", help="Path to *.npz dataset from prepare_rl_training_data.py")
    parser.add_argument("--state-size", type=int, help="Override the state vector size (default = dataset features)")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Symbol metadata for logs")
    parser.add_argument("--timeframe", default="1h", help="Timeframe metadata for logs")
    parser.add_argument("--config", default="config/config.example.yaml", help="Config file for RL hyperparams")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count (default from config)")
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint frequency (default from config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default from config)")
    parser.add_argument("--model-dir", default="data/checkpoints", help="Directory for intermediate checkpoints")
    parser.add_argument("--model-name", default="rl_agent_cli.pth", help="Filename for periodic checkpoints")
    parser.add_argument("--resume", help="Optional checkpoint path to resume training")
    parser.add_argument("--summary-file", default="data/training/rl_training_summary.json", help="Where to store training summary JSON")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--learning-rate", type=float, help="Experiment-friendly override for optimizer lr")
    parser.add_argument("--gradient-clip", type=float, help="Clamp gradient norm (max_norm) during training")
    parser.add_argument("--reward-clip", type=float, help="Symmetric reward clipping (±value) for quick experiments")
    parser.add_argument("--reward-clip-min", type=float, help="Explicit minimum reward clamp")
    parser.add_argument("--reward-clip-max", type=float, help="Explicit maximum reward clamp")
    parser.add_argument("--reward-scale", type=float, help="Scale factor applied to rewards before TD computation")
    parser.add_argument("--reinit-head", action="store_true", help="Reinitialize the output head before training")
    parser.add_argument(
        "--head-scale",
        type=float,
        help="Scale factor applied to the head weights/bias after reinit and seeds initial_head_scale",
    )
    parser.add_argument("--head-scale-min", type=float, help="Lower bound for learnable head scale (min + softplus(raw))")
    parser.add_argument("--output-scale", type=float, help="Multiply q-value outputs by this factor")
    parser.add_argument("--head-only", action="store_true", help="Only train the output head (freezes other layers).")
    parser.add_argument("--head-lr", type=float, help="Learning rate to use when head-only training is enabled.")
    parser.add_argument("--head-scale-learnable", action="store_true", help="Allow the head multiplier to be learned as trainable parameter")
    parser.add_argument("--reset-optimizer", action="store_true", help="Reinitialize optimizer state after head adjustments")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )



def resolve_training_params(config: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], int, int, int]:
    ml_cfg = (config.get("ml") or {}).get("reinforcement_learning", {})
    training_cfg = ml_cfg.get("training", {})

    episodes = args.episodes or int(training_cfg.get("episodes", 250))
    save_every = args.save_every or int(training_cfg.get("save_every", max(5, episodes // 10)))
    batch_size = args.batch_size or int(ml_cfg.get("batch_size", 64))

    agent_cfg = dict(ml_cfg)
    agent_cfg["training_mode"] = True
    models_cfg = config.get("models", {})
    agent_cfg.setdefault("active_bundle", models_cfg.get("active_bundle", "artifacts/gemma/final"))

    if args.learning_rate is not None:
        agent_cfg["learning_rate"] = args.learning_rate
    if args.gradient_clip is not None:
        agent_cfg["gradient_clip_norm"] = args.gradient_clip

    reward_clip_min = args.reward_clip_min
    reward_clip_max = args.reward_clip_max
    if args.reward_clip is not None:
        reward_clip_min = -args.reward_clip
        reward_clip_max = args.reward_clip
    if reward_clip_min is not None or reward_clip_max is not None:
        agent_cfg["reward_clip_enabled"] = True
        if reward_clip_min is not None:
            agent_cfg["reward_clip_min"] = reward_clip_min
        if reward_clip_max is not None:
            agent_cfg["reward_clip_max"] = reward_clip_max

    if args.reward_scale is not None:
        agent_cfg["reward_scale"] = args.reward_scale

    if args.output_scale is not None:
        agent_cfg["output_scale"] = args.output_scale
    if getattr(args, "head_scale_learnable", False):
        agent_cfg["head_scale_learnable"] = True
    if args.head_scale is not None:
        agent_cfg["initial_head_scale"] = args.head_scale
    if args.head_scale_min is not None:
        agent_cfg["head_scale_min_multiplier"] = args.head_scale_min

    return agent_cfg, episodes, save_every, batch_size


def write_summary(summary_path: Path, payload: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logging.info("Training summary written to %s", summary_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    dataset_path = Path(args.dataset) if args.dataset else None
    if dataset_path is None:
        raise SystemExit("--dataset path is required")

    logging.info("Loading dataset %s", dataset_path)
    features_df, price_df, meta = load_npz_dataset(dataset_path)

    dataset_state_size = features_df.shape[1]
    state_size = args.state_size or dataset_state_size
    if args.state_size and args.state_size != dataset_state_size:
        logging.warning(
            "--state-size override (%d) does not match dataset feature count (%d); agent will expect the provided dimension",
            args.state_size,
            dataset_state_size,
        )

    config = LiveTradingConfiguration.load(config_path=args.config, log_summary=False)
    agent_cfg, episodes, save_every, batch_size = resolve_training_params(config, args)

    logging.info("Building RL environment (rows=%d, state_size=%d)", len(features_df), state_size)
    env = RLTradingEnv(features_df=features_df, raw_df=price_df)

    agent = TradingRLAgent(state_size=state_size, action_size=3, config=agent_cfg)
    replay = ExperienceReplay(agent_cfg.get("buffer_size", 100000))

    if args.reinit_head:
        can_inspect_head = hasattr(agent, 'reinit_last_layer') and hasattr(agent, 'get_last_layer_stats')
        if can_inspect_head:
            get_stats: Callable[[], Any] = getattr(agent, 'get_last_layer_stats')
            pre_stats = get_stats()
            logging.info("🛠 Reinitializing RL output head (pre-stats: %s)", pre_stats)
        reinit_head: Callable[[], Any] = getattr(agent, 'reinit_last_layer')
        reinit_head()
        if args.head_scale is not None and hasattr(agent, 'scale_last_layer'):
            scale_head: Callable[[float], Any] = getattr(agent, 'scale_last_layer')
            scale_head(args.head_scale)
        if can_inspect_head:
            get_stats = getattr(agent, 'get_last_layer_stats')
            post_stats = get_stats()
            logging.info("🧪 Output head stats after reset: %s", post_stats)

    if args.head_only:
        head_lr = args.head_lr or agent_cfg.get('learning_rate', 1e-4)
        if head_lr <= 0:
            logging.warning("Invalid head learning rate %.6f, falling back to default 1e-4", head_lr)
            head_lr = 1e-4
        if hasattr(agent, 'enable_head_only_training'):
            enable_head = getattr(agent, 'enable_head_only_training')
            success = enable_head(head_lr)
            if success:
                logging.info("🧠 Head-only training active at lr=%.6f", head_lr)
            else:
                logging.warning("⚠️ Could not enable head-only training; continuing full-model updates.")
        else:
            logging.warning("⚠️ Agent does not support head-only training; running full updates.")

    if args.reset_optimizer:
        reset_lr = args.head_lr if args.head_lr else agent_cfg.get('learning_rate', 1e-4)
        logging.info("🔁 --reset-optimizer provided; reinitializing optimizer (lr=%.6f)", reset_lr)
        agent.reset_optimizer(reset_lr)

    trainer = RLModelTrainer(
        agent=agent,
        env=env,
        experience_replay=replay,
        model_save_path=str(Path(args.model_dir)),
        model_name=args.model_name,
    )

    trainer.train(
        num_episodes=episodes,
        batch_size=batch_size,
        save_every=save_every,
        checkpoint_path=args.resume,
    )

    final_checkpoint = Path(args.model_dir) / "rl_agent_final.pth"
    payload = {
        "dataset": str(dataset_path),
        "symbol": meta.get("symbol", args.symbol),
        "timeframe": meta.get("timeframe", args.timeframe),
        "rows": len(features_df),
        "state_size": state_size,
        "episodes": episodes,
        "save_every": save_every,
        "batch_size": batch_size,
        "checkpoint_dir": str(Path(args.model_dir)),
        "final_checkpoint": str(final_checkpoint),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "agent_summary": agent.get_training_summary(),
    }

    write_summary(Path(args.summary_file), payload)
    logging.info("RL training finished. Final checkpoint: %s", final_checkpoint)


if __name__ == "__main__":
    main()
