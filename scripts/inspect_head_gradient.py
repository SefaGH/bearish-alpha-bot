#!/usr/bin/env python3
"""Quick probe to verify head-scale gradient after a backward pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def _extract_checkpoint_state_size(model_path: Path) -> int | None:
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception:  # pragma: no cover - handled by caller
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
    try:
        from src.ml.reinforcement_learning import TradingRLAgent
    except Exception:  # pragma: no cover
        from ml.reinforcement_learning import TradingRLAgent  # type: ignore[import-not-found]

    checkpoint_state = _extract_checkpoint_state_size(model_path)
    if checkpoint_state is not None and checkpoint_state != state_size:
        raise ValueError(
            f"Checkpoint expects state size {checkpoint_state}, but --state-size={state_size}. "
            "Re-run with --state-size set to the checkpoint's dimension."
        )

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
    except Exception as exc:  # pylint: disable=broad-except
        print(
            "WARN: Failed to load checkpoint optimizer state"
            f" ({exc}). Proceeding with initialized optimizer."
        )
    return agent


def run_probe(agent, *, batch_size: int) -> None:
    model = getattr(agent, "q_network", None)
    if model is None:
        raise AttributeError("TradingRLAgent is missing q_network; cannot inspect gradients.")

    model.train()
    agent.optimizer.zero_grad()

    state_dim = agent.state_size
    action_dim = agent.action_size

    states = torch.randn(batch_size, state_dim)
    next_states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)

    scaled_current_q = agent._scale_q(model(states))
    current_q = scaled_current_q.gather(1, actions.unsqueeze(1))

    with torch.no_grad():
        next_actions = agent._scale_q(model(next_states)).argmax(1, keepdim=True)
        next_q = agent._scale_q(agent.target_network(next_states)).gather(1, next_actions)
        targets = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * agent.gamma * next_q

    loss = F.mse_loss(current_q, targets)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-size", type=int, default=82)
    parser.add_argument("--head-scale-learnable", action="store_true")
    parser.add_argument("--initial-head-scale", type=float, default=1.0)
    parser.add_argument("--head-scale-min", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    agent = load_agent(
        args.model,
        state_size=args.state_size,
        head_scale_learnable=args.head_scale_learnable,
        initial_head_scale=args.initial_head_scale,
        head_scale_min=args.head_scale_min,
    )

    run_probe(agent, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
