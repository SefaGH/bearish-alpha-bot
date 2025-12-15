"""
Quick log analyzer for pyramiding-related decisions.

Parses log files for tags such as:
- [PYRAMID] scale-in allowed/rejected
- [PYRAMID-QUEUE] scale-in enqueued/rejected
- duplicate spam window rejections with intent=scale_in

Usage:
    python scripts/analyze_pyramiding_logs.py --files logs/run.log other.log
"""

import argparse
import re
from collections import Counter, defaultdict


PATTERNS = {
    "pyramid": re.compile(r"\[PYRAMID\]", re.IGNORECASE),
    "queue": re.compile(r"\[PYRAMID-QUEUE\]", re.IGNORECASE),
    "duplicate_spam": re.compile(r"duplicate_scale_in_spam_window", re.IGNORECASE),
    "reason": re.compile(r"reason=([A-Za-z0-9_]+)", re.IGNORECASE),
}


def analyze_logs(paths):
    counters = {
        "pyramid_lines": 0,
        "queue_lines": 0,
        "duplicate_spam": 0,
        "pyramid_reasons": Counter(),
        "queue_reasons": Counter(),
    }

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if PATTERNS["pyramid"].search(line):
                        counters["pyramid_lines"] += 1
                        m = PATTERNS["reason"].search(line)
                        if m:
                            counters["pyramid_reasons"][m.group(1)] += 1
                    if PATTERNS["queue"].search(line):
                        counters["queue_lines"] += 1
                        m = PATTERNS["reason"].search(line)
                        if m:
                            counters["queue_reasons"][m.group(1)] += 1
                    if PATTERNS["duplicate_spam"].search(line):
                        counters["duplicate_spam"] += 1
        except FileNotFoundError:
            print(f"Warning: file not found: {path}")
            continue
    return counters


def main():
    parser = argparse.ArgumentParser(description="Analyze pyramiding-related log lines.")
    parser.add_argument("--files", nargs="+", required=True, help="Log file paths to analyze")
    args = parser.parse_args()

    counters = analyze_logs(args.files)

    print("=== Pyramiding Log Summary ===")
    print(f"[PYRAMID] lines: {counters['pyramid_lines']}")
    if counters["pyramid_reasons"]:
        print("  Reasons:")
        for reason, count in counters["pyramid_reasons"].most_common():
            print(f"    {reason}: {count}")
    print(f"[PYRAMID-QUEUE] lines: {counters['queue_lines']}")
    if counters["queue_reasons"]:
        print("  Queue reasons:")
        for reason, count in counters["queue_reasons"].most_common():
            print(f"    {reason}: {count}")
    print(f"Duplicate spam rejections (scale_in): {counters['duplicate_spam']}")


if __name__ == "__main__":
    main()
