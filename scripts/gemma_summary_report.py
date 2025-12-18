#!/usr/bin/env python3
"""Summarize GEMMA monitoring lines from a live trading log file."""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"\[GEMMA\]\s+conf_p50=(?P<p50>[0-9.]+)\s+conf_p95=(?P<p95>[0-9.]+)\s+"
    r"p95_latency=(?P<lat>[0-9.]+)ms\s+class_counts=(?P<counts>\{[^}]*\})"
)
COUNT_RE = re.compile(r"['\"]?([a-zA-Z_]+)['\"]?\s*:\s*([0-9]+)")
REGIME_RE = re.compile(
    r"\[ml\.regime_predictor\].*?Prediction:\s+"
    r"(bearish|neutral|bullish)\s+\(confidence:\s*([0-9.]+)\)"
)


def parse_counts(raw: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label, value in COUNT_RE.findall(raw):
        counts[label] = int(value)
    return counts


def find_latest_log() -> Optional[Path]:
    latest_link = Path("logs/live_trading_latest.log")
    if latest_link.exists():
        return latest_link
    candidates = list(Path("logs").glob("live_trading_*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_log(log_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = LINE_RE.search(line)
            if not match:
                continue
            counts = parse_counts(match.group("counts"))
            rows.append(
                {
                    "timestamp": match.group("ts"),
                    "conf_p50": float(match.group("p50")),
                    "conf_p95": float(match.group("p95")),
                    "p95_latency_ms": float(match.group("lat")),
                    "class_counts": counts,
                }
            )
    return rows


def parse_regime_predictions(log_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = REGIME_RE.search(line)
            if not match:
                continue
            timestamp = line[:19] if len(line) >= 19 else ""
            rows.append(
                {
                    "timestamp": timestamp,
                    "label": match.group(1),
                    "confidence": float(match.group(2)),
                }
            )
    return rows


def summarize(rows: List[Dict[str, object]], window: int) -> Dict[str, object]:
    if not rows:
        return {
            "summary_lines": 0,
            "latest": None,
            "window": None,
        }

    latest = rows[-1]
    latest_counts = latest["class_counts"]
    totals = {
        "bearish": latest_counts.get("bearish", 0),
        "neutral": latest_counts.get("neutral", 0),
        "bullish": latest_counts.get("bullish", 0),
    }
    total_n = sum(totals.values())
    latest_summary = {
        "timestamp": latest["timestamp"],
        "conf_p50": latest["conf_p50"],
        "conf_p95": latest["conf_p95"],
        "p95_latency_ms": latest["p95_latency_ms"],
        "class_counts": totals,
        "class_total": total_n,
        "class_ratio": {
            k: (v / total_n) if total_n else 0.0 for k, v in totals.items()
        },
    }

    window_summary = None
    if window > 1 and len(rows) >= window:
        prev = rows[-window]
        prev_counts = prev["class_counts"]
        delta = {
            "bearish": latest_counts.get("bearish", 0) - prev_counts.get("bearish", 0),
            "neutral": latest_counts.get("neutral", 0) - prev_counts.get("neutral", 0),
            "bullish": latest_counts.get("bullish", 0) - prev_counts.get("bullish", 0),
        }
        window_total = sum(delta.values())
        window_summary = {
            "window_size": window,
            "from_timestamp": prev["timestamp"],
            "to_timestamp": latest["timestamp"],
            "class_counts": delta,
            "class_total": window_total,
            "class_ratio": {
                k: (v / window_total) if window_total else 0.0
                for k, v in delta.items()
            },
        }

    return {
        "summary_lines": len(rows),
        "latest": latest_summary,
        "window": window_summary,
    }


def summarize_regime(rows: List[Dict[str, object]], window: int) -> Dict[str, object]:
    if not rows:
        return {
            "predictions": 0,
            "latest": None,
            "all": None,
            "window": None,
        }

    confs = [row["confidence"] for row in rows]
    labels = [row["label"] for row in rows]
    counts = {
        "bearish": labels.count("bearish"),
        "neutral": labels.count("neutral"),
        "bullish": labels.count("bullish"),
    }
    total = sum(counts.values())
    ratios = {k: (v / total) if total else 0.0 for k, v in counts.items()}
    latest = rows[-1]

    all_summary = {
        "class_counts": counts,
        "class_total": total,
        "class_ratio": ratios,
        "conf_p50": float(np.percentile(confs, 50)),
        "conf_p95": float(np.percentile(confs, 95)),
        "conf_max": float(max(confs)),
        "high_conf_ratio_gt_0_95": float(sum(c > 0.95 for c in confs) / len(confs)),
    }

    window_summary = None
    if window > 1 and len(rows) >= window:
        recent = rows[-window:]
        r_confs = [row["confidence"] for row in recent]
        r_labels = [row["label"] for row in recent]
        r_counts = {
            "bearish": r_labels.count("bearish"),
            "neutral": r_labels.count("neutral"),
            "bullish": r_labels.count("bullish"),
        }
        r_total = sum(r_counts.values())
        r_ratios = {k: (v / r_total) if r_total else 0.0 for k, v in r_counts.items()}
        window_summary = {
            "window_size": window,
            "from_timestamp": recent[0]["timestamp"],
            "to_timestamp": recent[-1]["timestamp"],
            "class_counts": r_counts,
            "class_total": r_total,
            "class_ratio": r_ratios,
            "conf_p50": float(np.percentile(r_confs, 50)),
            "conf_p95": float(np.percentile(r_confs, 95)),
        }

    return {
        "predictions": len(rows),
        "latest": latest,
        "all": all_summary,
        "window": window_summary,
    }


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "conf_p50",
                "conf_p95",
                "p95_latency_ms",
                "bearish",
                "neutral",
                "bullish",
                "total",
            ]
        )
        for row in rows:
            counts = row["class_counts"]
            bearish = counts.get("bearish", 0)
            neutral = counts.get("neutral", 0)
            bullish = counts.get("bullish", 0)
            writer.writerow(
                [
                    row["timestamp"],
                    row["conf_p50"],
                    row["conf_p95"],
                    row["p95_latency_ms"],
                    bearish,
                    neutral,
                    bullish,
                    bearish + neutral + bullish,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize GEMMA monitoring lines from a log file."
    )
    parser.add_argument(
        "--log-file",
        help="Path to live trading log (default: logs/live_trading_latest.log).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Window size for recent class distribution (default: 20 summaries).",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path to write per-summary CSV output.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file) if args.log_file else find_latest_log()
    if not log_path:
        print("No log file provided and no logs found under ./logs.")
        return 1
    if log_path.is_symlink():
        resolved = Path(os.path.realpath(log_path))
        if resolved.exists():
            log_path = resolved

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return 1

    rows = parse_log(log_path)
    summary = summarize(rows, args.window)
    regime_rows = parse_regime_predictions(log_path)
    regime_summary = summarize_regime(regime_rows, args.window)

    print(f"log: {log_path}")
    print(f"summary_lines: {summary['summary_lines']}")
    if summary["latest"] is None:
        print("no GEMMA summaries found")
    else:
        latest = summary["latest"]
        print(
            "latest:",
            f"timestamp={latest['timestamp']}",
            f"conf_p50={latest['conf_p50']:.3f}",
            f"conf_p95={latest['conf_p95']:.3f}",
            f"p95_latency_ms={latest['p95_latency_ms']:.1f}",
            f"class_counts={latest['class_counts']}",
            f"class_total={latest['class_total']}",
            f"class_ratio={latest['class_ratio']}",
        )

    if summary["window"]:
        window = summary["window"]
        print(
            "window:",
            f"size={window['window_size']}",
            f"from={window['from_timestamp']}",
            f"to={window['to_timestamp']}",
            f"class_counts={window['class_counts']}",
            f"class_total={window['class_total']}",
            f"class_ratio={window['class_ratio']}",
        )

    print(f"regime_predictions: {regime_summary['predictions']}")
    if regime_summary["latest"] is None:
        print("no regime predictions found")
    else:
        latest_regime = regime_summary["latest"]
        print(
            "regime_latest:",
            f"timestamp={latest_regime['timestamp']}",
            f"label={latest_regime['label']}",
            f"confidence={latest_regime['confidence']:.3f}",
        )
        all_regime = regime_summary["all"]
        print(
            "regime_all:",
            f"conf_p50={all_regime['conf_p50']:.3f}",
            f"conf_p95={all_regime['conf_p95']:.3f}",
            f"conf_max={all_regime['conf_max']:.3f}",
            f"high_conf_ratio_gt_0_95={all_regime['high_conf_ratio_gt_0_95']:.3f}",
            f"class_counts={all_regime['class_counts']}",
            f"class_total={all_regime['class_total']}",
            f"class_ratio={all_regime['class_ratio']}",
        )
        if regime_summary["window"]:
            window = regime_summary["window"]
            print(
                "regime_window:",
                f"size={window['window_size']}",
                f"from={window['from_timestamp']}",
                f"to={window['to_timestamp']}",
                f"conf_p50={window['conf_p50']:.3f}",
                f"conf_p95={window['conf_p95']:.3f}",
                f"class_counts={window['class_counts']}",
                f"class_total={window['class_total']}",
                f"class_ratio={window['class_ratio']}",
            )

    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
        print(f"wrote csv: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
