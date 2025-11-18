#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: summarize_rl_validation.py <report_path>", file=sys.stderr)
        sys.exit(1)

    report_path = Path(sys.argv[1])

    if not report_path.is_file():
        print(f"Validation report not found at {report_path}", file=sys.stderr)
        # Graceful exit: workflow adımını fail etmeyelim, sadece bilgi verelim
        sys.exit(0)

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n--- RL VALIDATION REPORT ---")
    total_reward = data.get("total_reward") or data.get("reward_total") or "N/A"
    pnl_total = data.get("pnl_total", "N/A")
    action_dist = data.get("action_distribution", {})
    q_stats = data.get("q_stats", {})

    print(f"Total Reward: {total_reward}")
    print(f"Total PnL:    {pnl_total}")
    print(f"Action Distribution: {action_dist}")
    print(f"Q-Value Stats: {q_stats}")
    print("--- END REPORT ---\n")


if __name__ == "__main__":
    main()
