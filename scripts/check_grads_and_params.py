#!/usr/bin/env python3
"""
Quick check: prints optimizer lr, param std, requires_grad, does a single forward/backward on one minibatch
Usage:
  python scripts/check_grads_and_params.py --model data/models/rl_agent_final.pth --scaler artifacts/gemma/gemma_price_scaler.joblib --replay data/replay_sample.pkl
Adjust imports paths if your project layout differs.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_agent(model_path, state_size):
    from src.ml.reinforcement_learning import TradingRLAgent  # noqa: E402
    agent = TradingRLAgent(state_size=state_size, action_size=3)
    agent.load_model(str(model_path))
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--replay",
        required=False,
        help="Optional replay pickle (trusted project artifact only)",
    )
    p.add_argument("--dataset", help="Optional dataset to infer state_size and replay if needed")
    p.add_argument("--state-size", type=int, help="Force state size when initializing the agent")
    args = p.parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print("Model not found:", model_path)
        sys.exit(1)

    state_size = args.state_size
    if state_size is None and args.dataset:
        from scripts.rl_dataset_utils import load_npz_dataset  # noqa: E402
        features_df, _, _ = load_npz_dataset(Path(args.dataset))
        state_size = features_df.shape[1]
        print("Inferred state_size from dataset:", state_size)
    if state_size is None:
        state_size = 82
        print("Defaulting state_size to", state_size)
    agent = load_agent(model_path, state_size)
    model = getattr(agent, "q_network", None)
    if model is None:
        print("Agent has no q_network to inspect")
        sys.exit(1)
    print("PARAM STATS:")
    for name, p in model.named_parameters():
        print(
            f"{name} | requires_grad={p.requires_grad} | std={float(p.detach().cpu().std()):.6e} | mean={float(p.detach().cpu().mean()):.6e}"
        )

    opt = getattr(agent, "optimizer", None)
    if opt is None:
        print("No optimizer found on agent object (agent.optimizer missing).")
    else:
        for i, pg in enumerate(opt.param_groups):
            print(
                f"optimizer.param_group[{i}] lr={pg.get('lr')} weight_decay={pg.get('weight_decay')} params={len(pg['params'])}"
            )

    if args.replay:
        with open(args.replay, "rb") as f:
            replay = pickle.load(f)
        batch = replay[: min(32, len(replay))]
        states = np.array([t[0] for t in batch], dtype=float)
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
    else:
        states = np.random.randn(8, state_size).astype(float)
        actions = np.random.randint(0, 3, size=(8,))
        rewards = np.random.randn(8).astype(float)
        next_states = np.random.randn(8, state_size).astype(float)

    model.train()
    states_t = torch.FloatTensor(states)
    next_t = torch.FloatTensor(next_states)
    q = model(states_t)
    q_selected = q.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = model(next_t)
        q_next_max = q_next.max(dim=1)[0]
    gamma = 0.99
    target = torch.FloatTensor(rewards) + gamma * q_next_max

    loss_fn = torch.nn.MSELoss(reduction="mean")
    loss = loss_fn(q_selected, target)
    print("Sample loss (before backward):", float(loss.item()))
    loss.backward()

    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            gn = 0.0
        else:
            gn = float(p.grad.data.norm(2).cpu().item())
        total_norm += gn
        print(f"{name} grad_norm={gn:.6e}")
    print("total_grad_norm:", total_norm)
    print("Done single-step check. If all grad_norms are 0 -> backward or loss graph broken.")


if __name__ == "__main__":
    main()
