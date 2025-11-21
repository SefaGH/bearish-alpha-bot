#!/usr/bin/env python3
"""
Train a PPO agent on the prepared RL trading dataset using Stable-Baselines3.

- prepare_rl_training_data.py tarafından üretilen *_train.npz datasetini kullanır.
- RLTradingEnvGym üzerinde PPO ("MlpPolicy") eğitir.
- Modeli data/checkpoints altına kaydeder.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.ml.rl_trading_env_gym import RLTradingEnvGym
from scripts.rl_dataset_utils import load_npz_dataset  # mevcut util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent on RL trading dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to *_train.npz dataset (from prepare_rl_training_data.py).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/checkpoints"),
        help="Directory to save PPO models.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_trading_agent",
        help="Base name for PPO model (without extension).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total timesteps for PPO training.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_env(features_df, price_df, metadata):
    env_config = {
        "fee_pct": metadata.get("fee_pct", 0.0006) if isinstance(metadata, dict) else 0.0006,
        "trade_penalty_alpha": 0.0,
        "idle_cost": 0.0,
        "reward_clip_enabled": True,
        "reward_clip_min": -1.0,
        "reward_clip_max": 1.0,
    }
    initial_balance = float(metadata.get("initial_balance", 10_000.0)) if isinstance(metadata, dict) else 10_000.0
    env = RLTradingEnvGym(
        features_df=features_df,
        raw_df=price_df,
        config=env_config,
        initial_balance=initial_balance,
    )
    return env


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    logging.info("Loading dataset from %s", args.dataset)
    features_df, price_df, metadata = load_npz_dataset(args.dataset)

    logging.info("Creating Gym-compatible trading environment...")
    def _env_fn():
        return make_env(features_df, price_df, metadata)

    vec_env = DummyVecEnv([_env_fn])

    logging.info("Initializing PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=str(args.model_dir / "ppo_tb"),
        # HAFİF DAHA STABİL VE BİRAZ DAHA KEŞİFÇİ AYARLAR
        learning_rate=2.5e-4,    # 3e-4'ten çok az aşağı, biraz daha stabil öğrenme
        n_steps=1024,            # rollout'u yarıya indir, daha sık güncelleme
        batch_size=64,           # aynı kalabilir (1024'ün böleni)
        gamma=0.99,              # aynen
        gae_lambda=0.95,         # aynen
        ent_coef=0.005,          # 0.001 → 0.005: biraz daha exploration
        vf_coef=0.5,             # aynen
        max_grad_norm=0.5,       # aynen
    )

    logging.info("Training PPO for %d timesteps...", args.timesteps)
    model.learn(total_timesteps=args.timesteps)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / f"{args.model_name}.zip"
    logging.info("Saving PPO model to %s", model_path)
    model.save(str(model_path))

    logging.info("PPO training finished.")


if __name__ == "__main__":
    main()
