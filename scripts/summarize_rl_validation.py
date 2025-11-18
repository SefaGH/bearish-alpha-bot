#!/usr/bin/env python3
import json, sys, os
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_rl_validation.py <report_path>", file=sys.stderr)
        sys.exit(1)

    report_path = Path(sys.argv[1])
    if not report_path.is_file():
        print(f"Validation report not found at {report_path}", file=sys.stderr)
        sys.exit(0)

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {}) or {}

    print("\n--- RL VALIDATION REPORT ---")
    total_reward = metrics.get("total_reward", "N/A")
    pnl_total = metrics.get("final_pnl", "N/A")
    action_dist = metrics.get("action_distribution", {})
    q_stats = metrics.get("q_values", {})

    print(f"Total Reward: {total_reward}")
    print(f"Total PnL:    {pnl_total}")
    print(f"Action Distribution: {action_dist}")
    print(f"Q-Value Stats: {q_stats}")
    print("--- END REPORT ---\n")

if __name__ == "__main__":
    main()
