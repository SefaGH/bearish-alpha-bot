#!/usr/bin/env python3
"""Compare replay/td diagnostics between two RL checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_replay_and_td import (
    inspect_checkpoint,
    load_agent,
    load_dataset,
)


def choose_states(dataset_features: Any, sample_size: int) -> torch.FloatTensor:
    total = len(dataset_features)
    if total == 0:
        raise ValueError("Dataset contains no states for sampling q histograms.")
    indices = np.random.choice(total, min(sample_size, total), replace=False)
    states = torch.FloatTensor(dataset_features.iloc[indices].to_numpy(dtype=float))
    return states


def evaluate_q_values(agent: Any, states: torch.FloatTensor) -> np.ndarray:
    with torch.no_grad():
        q_values = agent.q_network(states)
        if hasattr(agent, "_scale_q"):
            q_values = agent._scale_q(q_values)
    return q_values.cpu().numpy()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def earth_movers_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def compute_hist_divergence(base_q: np.ndarray, exp_q: np.ndarray, bins: int) -> List[Dict[str, Any]]:
    num_actions = base_q.shape[1]
    divergences: List[Dict[str, Any]] = []
    for action_idx in range(num_actions):
        combined = np.concatenate([base_q[:, action_idx], exp_q[:, action_idx]])
        bin_edges = np.histogram_bin_edges(combined, bins=bins)
        base_counts, _ = np.histogram(base_q[:, action_idx], bins=bin_edges)
        exp_counts, _ = np.histogram(exp_q[:, action_idx], bins=bin_edges)
        base_probs = base_counts / base_counts.sum() if base_counts.sum() > 0 else base_counts
        exp_probs = exp_counts / exp_counts.sum() if exp_counts.sum() > 0 else exp_counts
        divergences.append({
            "action": action_idx,
            "js_divergence": js_divergence(base_probs, exp_probs),
            "earth_mover": earth_movers_distance(base_probs, exp_probs),
        })
    return divergences


def print_summary(title: str, metrics: Dict[str, Any]) -> None:
    print(f"=== {title} ===")
    print(f"Avg TD error: {metrics['avg_td']:.6f}")
    print(f"Median TD error: {metrics['median_td']:.6f}")
    print(f"Median q_std: {metrics['median_q_std']:.6f}")
    reward_stats = metrics['reward_stats']
    print(
        "Reward (mean/std/min/max/1%/99%):",
        reward_stats['mean'],
        reward_stats['std'],
        reward_stats['min'],
        reward_stats['max'],
        reward_stats['1%'],
        reward_stats['99%'],
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two RL checkpoint inspections.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--state-size", type=int, default=None)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--histogram-sample", type=int, default=1024)
    parser.add_argument("--histogram-bins", type=int, default=20)
    parser.add_argument("--baseline-reward-clip", type=float, default=None)
    parser.add_argument("--baseline-reward-scale", type=float, default=1.0)
    parser.add_argument("--experiment-reward-clip", type=float, default=None)
    parser.add_argument("--experiment-reward-scale", type=float, default=1.0)
    parser.add_argument("--state-sample-seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.state_sample_seed)

    baseline_metrics = inspect_checkpoint(
        model_path=args.baseline,
        state_size=args.state_size,
        dataset_path=args.dataset,
        reward_clip=args.baseline_reward_clip,
        reward_scale=args.baseline_reward_scale,
        batch_size=args.batch_size,
        samples=args.samples,
        histogram_bins=args.histogram_bins,
        histogram_sample=args.histogram_sample,
    )
    experiment_metrics = inspect_checkpoint(
        model_path=args.experiment,
        state_size=args.state_size,
        dataset_path=args.dataset,
        reward_clip=args.experiment_reward_clip,
        reward_scale=args.experiment_reward_scale,
        batch_size=args.batch_size,
        samples=args.samples,
        histogram_bins=args.histogram_bins,
        histogram_sample=args.histogram_sample,
    )

    print_summary("Baseline", baseline_metrics)
    print_summary("Experiment", experiment_metrics)

    dataset_features, _ = load_dataset(args.dataset)
    states = choose_states(dataset_features, args.histogram_sample)

    baseline_agent = load_agent(args.baseline, baseline_metrics['state_size'])
    experiment_agent = load_agent(args.experiment, experiment_metrics['state_size'])

    baseline_q = evaluate_q_values(baseline_agent, states)
    experiment_q = evaluate_q_values(experiment_agent, states)

    divergences = compute_hist_divergence(baseline_q, experiment_q, args.histogram_bins)
    print("Per-action histogram divergences:")
    for div in divergences:
        print(
            f"Action {div['action']}: JS={div['js_divergence']:.6f}, EMD={div['earth_mover']:.6f}"
        )

    print("q_std (baseline):", float(np.std(baseline_q)))
    print("q_std (experiment):", float(np.std(experiment_q)))


if __name__ == "__main__":
    main()
