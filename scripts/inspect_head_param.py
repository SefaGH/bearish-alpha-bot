#!/usr/bin/env python3
"""
Inspect head-scale parameter registration and optimizer membership.
Usage example:
    python scripts/inspect_head_param.py --model data/checkpoints/rl_agent_head_scale_canonical.pth --head-scale-learnable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


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

    agent = TradingRLAgent(
        state_size=state_size,
        action_size=3,
        config={
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
            f" ({exc}). Inspecting model parameters only."
        )
    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-size", type=int, default=87)
    parser.add_argument("--head-scale-learnable", action="store_true")
    parser.add_argument("--initial-head-scale", type=float, default=1.0)
    parser.add_argument("--head-scale-min", type=float, default=0.1)
    args = parser.parse_args()

    model_path = args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    agent = load_agent(
        model_path,
        state_size=args.state_size,
        head_scale_learnable=args.head_scale_learnable,
        initial_head_scale=args.initial_head_scale,
        head_scale_min=args.head_scale_min,
    )
    model = getattr(agent, "q_network", None)
    if model is None:
        raise AttributeError("TradingRLAgent is missing q_network; cannot inspect parameters.")

    print("Listing named parameters and trainability:")
    for name, param in model.named_parameters():
        sample = float(param.detach().view(-1)[0]) if param.numel() > 0 else 0.0
        print(
            f"PARAM: {name} | shape={tuple(param.shape)} | requires_grad={param.requires_grad}"
            f" | dtype={param.dtype} | device={param.device} | value(sample)={sample:.6e}"
        )

    print("\nSearching for head-scale parameter:")
    found = False
    for name, param in model.named_parameters():
        key = name.lower()
        if "head_scale" in key or ("scale" in key and ("head" in key or "output" in key)):
            found = True
            sample_value = float(param.detach().cpu().view(-1)[0]) if param.numel() > 0 else 0.0
            print("FOUND candidate:", name, "value:", f"{sample_value:.6e}", "requires_grad:", param.requires_grad)
            if param.grad is not None:
                grad_sample = float(param.grad.detach().view(-1)[0]) if param.grad.numel() > 0 else 0.0
                print("    grad sample:", grad_sample)
            else:
                print("    grad is None (no backward pass executed or not computed yet)")
    if not found:
        print("No parameter matching 'head_scale' (or similar) found. Check registration name.")
    else:
        scale_tensor = getattr(model, "head_scale", None)
        if isinstance(scale_tensor, torch.Tensor):
            sample_value = float(scale_tensor.detach().cpu().view(-1)[0])
            print(f"Resolved head_scale (min + softplus(raw)): {sample_value:.6e}")
        raw_param = getattr(model, "head_scale_raw", None)
        if isinstance(raw_param, torch.nn.Parameter):
            raw_sample = float(raw_param.detach().cpu().view(-1)[0])
            print(f"head_scale_raw (softplus input) sample: {raw_sample:.6e}")
        alpha_param = getattr(model, "head_scale_alpha", None)
        if isinstance(alpha_param, torch.nn.Parameter):
            alpha_sample = float(alpha_param.detach().cpu().view(-1)[0])
            print(f"Legacy head_scale_alpha (pre-clamp) sample: {alpha_sample:.6e}")

    optimizer = getattr(agent, "optimizer", None)
    if optimizer is None:
        print("\nNo optimizer object attached to agent.")
        return

    print("\nOptimizer param groups:")
    param_name_map = {id(p): name for name, p in model.named_parameters()}
    for idx, group in enumerate(optimizer.param_groups):
        lr = group.get("lr")
        params = group.get("params", [])
        names = [param_name_map.get(id(p), "<unknown>") for p in params]
        print(
            f"  group[{idx}] lr={lr} params_count={len(params)} names(sample up to 10)={names[:10]}"
        )


if __name__ == "__main__":
    main()
