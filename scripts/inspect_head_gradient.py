#!/usr/bin/env python3
"""Quick probe to verify head-scale gradient after a backward pass."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Proje kök dizinini yola ekle
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Gerekli importlar (Environment ve Dataset utils eklendi)
from src.ml.rl_trading_env import RLTradingEnv
from scripts.rl_dataset_utils import load_npz_dataset

def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def _extract_checkpoint_state_size(model_path: Path) -> int | None:
    """Checkpoint dosyasından beklenen state boyutunu çıkarmaya çalışır."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception:
        return None

    state_dict = checkpoint.get("q_network")
    if not isinstance(state_dict, dict):
        return None
    for key, tensor in state_dict.items():
        if key.endswith("network.0.weight") and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[1])
    return None


def load_agent(
    model_path: Path,
    *,
    state_size: int,
    head_scale_learnable: bool,
    initial_head_scale: float,
    head_scale_min: float,
):
    """Ajanı belirtilen state boyutu ve config ile yükler."""
    try:
        from src.ml.reinforcement_learning import TradingRLAgent
    except Exception:
        from ml.reinforcement_learning import TradingRLAgent

    # Checkpoint içindeki boyutu kontrol et
    checkpoint_state = _extract_checkpoint_state_size(model_path)
    if checkpoint_state is not None and checkpoint_state != state_size:
        print(f"WARN: Checkpoint expects {checkpoint_state}, but calculated/provided size is {state_size}.")
        print(f"WARN: Switching to Checkpoint size ({checkpoint_state}) to avoid crash.")
        state_size = checkpoint_state

    agent = TradingRLAgent(
        state_size=state_size,
        action_size=3,
        config={
            "training_mode": True,
            "head_scale_learnable": head_scale_learnable,
            "initial_head_scale": initial_head_scale,
            "head_scale_min_multiplier": head_scale_min,
        },
    )
    try:
        agent.load_model(str(model_path))
    except Exception as exc:
        print(
            "WARN: Failed to load checkpoint optimizer state"
            f" ({exc}). Proceeding with initialized optimizer."
        )
    return agent


def run_probe(agent, *, batch_size: int) -> None:
    """Model üzerinde tek bir backward pass yaparak gradyanları kontrol eder."""
    model = getattr(agent, "q_network", None)
    if model is None:
        raise AttributeError("TradingRLAgent is missing q_network; cannot inspect gradients.")

    model.train()
    agent.optimizer.zero_grad()

    state_dim = agent.state_size
    action_dim = agent.action_size

    # Rastgele veri oluştur
    states = torch.randn(batch_size, state_dim)
    next_states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)

    # Forward pass
    scaled_current_q = agent._scale_q(model(states))
    current_q = scaled_current_q.gather(1, actions.unsqueeze(1))

    with torch.no_grad():
        target_net = getattr(agent, "target_network", model)
        next_actions = agent._scale_q(model(next_states)).argmax(1, keepdim=True)
        next_q = agent._scale_q(target_net(next_states)).gather(1, next_actions)
        targets = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * agent.gamma * next_q

    # Loss ve Backward
    loss = F.mse_loss(current_q, targets)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")

    # Head scale parametrelerini kontrol et
    scale_tensor = getattr(model, "head_scale", None)
    if isinstance(scale_tensor, torch.Tensor):
        print(f"head_scale value: {float(scale_tensor.detach().view(-1)[0]):.6f}")
    else:
        print("head_scale value unavailable")

    raw_param = getattr(model, "head_scale_raw", None)
    if isinstance(raw_param, torch.nn.Parameter):
        raw_val = float(raw_param.detach().view(-1)[0])
        print(f"head_scale_raw (softplus input): {raw_val:.6f}")
        grad = raw_param.grad
        if grad is None:
            print("head_scale_raw grad is None")
        else:
            grad_sample = float(grad.view(-1)[0])
            print(f"head_scale_raw grad sample: {grad_sample:.6e}")
            print(f"head_scale_raw grad norm: {grad.norm().item():.6e}")
        return

    alpha_param = getattr(model, "head_scale_alpha", None)
    if isinstance(alpha_param, torch.nn.Parameter):
        alpha_val = float(alpha_param.detach().view(-1)[0])
        print(f"head_scale_alpha (legacy pre-clamp): {alpha_val:.6f}")
        grad = alpha_param.grad
        if grad is None:
            print("head_scale_alpha grad is None")
        else:
            grad_sample = float(grad.view(-1)[0])
            print(f"head_scale_alpha grad sample: {grad_sample:.6e}")
            print(f"head_scale_alpha grad norm: {grad.norm().item():.6e}")
    else:
        print("No head_scale_raw parameter on q_network.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect head gradients.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the model checkpoint")
    # Dataset argümanı eklendi: Env boyutunu öğrenmek için gerekli
    parser.add_argument("--dataset", type=Path, required=True, help="Path to .npz dataset to initialize environment")
    
    parser.add_argument("--state-size", type=int, default=None, help="Override state size (optional)")
    parser.add_argument("--head-scale-learnable", action="store_true")
    parser.add_argument("--initial-head-scale", type=float, default=1.0)
    parser.add_argument("--head-scale-min", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    # 1. Dataset'i yükle
    logging.info("Loading dataset %s to determine state size...", args.dataset)
    features_df, price_df, _ = load_npz_dataset(args.dataset)

    # 2. Environment'ı kur ve gerçek boyutu öğren (Feature + Portfolio)
    env = RLTradingEnv(features_df=features_df, raw_df=price_df)
    env_state_size = getattr(env, "state_dim", features_df.shape[1])
    
    logging.info(f"🔍 Detected State Size from Env: {env_state_size}")

    # 3. Kullanılacak boyutu belirle (CLI override yoksa env boyutunu kullan)
    final_state_size = args.state_size if args.state_size is not None else env_state_size

    # 4. Ajanı yükle ve testi çalıştır
    agent = load_agent(
        args.model,
        state_size=final_state_size,
        head_scale_learnable=args.head_scale_learnable,
        initial_head_scale=args.initial_head_scale,
        head_scale_min=args.head_scale_min,
    )

    run_probe(agent, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
