"""Offline analysis of TRADE_CLOSED events grouped by volume bucket.

Expects JSONL log lines containing `event: "TRADE_CLOSED"` payloads. Supports
filtering by `run_id` and `timeframe`, aggregates by `volume_bucket_at_entry`
(and strategy when present), and emits JSON plus optional CSV.

CLI (examples):
    python -m scripts.analyze_volume_buckets --log-dir ./logs/run_2025_12_10 \
        --run-id run-123 --output ./reports/volume_bucket_report.json \
        --csv-output ./reports/volume_bucket_report.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TRADE_EVENT_NAME = "TRADE_CLOSED"


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_json(line: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from a line that may have a prefix before the JSON payload."""
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        brace_idx = line.find("{")
        if brace_idx != -1:
            candidate = line[brace_idx:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
    return None


def _iter_log_files(log_dir: Optional[Path], log_file: Optional[Path]) -> List[Path]:
    files: List[Path] = []
    if log_file:
        files.append(log_file)
    if log_dir:
        for pattern in ("*.log", "*.jsonl"):
            files.extend(sorted(log_dir.glob(pattern)))
    return files


def load_trades_from_files(files: Iterable[Path], run_id: Optional[str] = None, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load TRADE_CLOSED events from the given files."""
    trades: List[Dict[str, Any]] = []
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    payload = _extract_json(line)
                    if not payload:
                        continue
                    event_name = str(payload.get("event", "")).upper()
                    if event_name != TRADE_EVENT_NAME:
                        continue
                    if run_id and payload.get("run_id") != run_id:
                        continue
                    if timeframe and payload.get("timeframe") != timeframe:
                        continue
                    trades.append(payload)
        except FileNotFoundError:
            continue
    return trades


def _init_acc() -> Dict[str, Any]:
    return {
        "n_trades": 0,
        "n_wins": 0,
        "n_losses": 0,
        "pnl_sum": 0.0,
        "rr_sum": 0.0,
        "rr_count": 0,
    }


def _record(acc: Dict[str, Any], pnl: float, rr: Optional[float]) -> None:
    acc["n_trades"] += 1
    if pnl > 0:
        acc["n_wins"] += 1
    elif pnl < 0:
        acc["n_losses"] += 1
    acc["pnl_sum"] += pnl
    if rr is not None:
        acc["rr_sum"] += rr
        acc["rr_count"] += 1


def _finalize(acc: Dict[str, Any]) -> Dict[str, Any]:
    n_trades = acc.get("n_trades", 0)
    rr_count = acc.get("rr_count", 0)
    return {
        "n_trades": n_trades,
        "n_wins": acc.get("n_wins", 0),
        "n_losses": acc.get("n_losses", 0),
        "win_rate": (acc["n_wins"] / n_trades) if n_trades else 0.0,
        "avg_pnl": (acc["pnl_sum"] / n_trades) if n_trades else 0.0,
        "avg_rr": (acc["rr_sum"] / rr_count) if rr_count else None,
    }


def aggregate_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = _init_acc()
    by_bucket: Dict[str, Dict[str, Any]] = {}
    by_bucket_strategy: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for trade in trades:
        bucket = trade.get("volume_bucket_at_entry") or "UNKNOWN"
        strategy = trade.get("strategy") or trade.get("strategy_name") or "unknown"
        pnl_val = (
            trade.get("realized_pnl_usdt")
            or trade.get("realized_pnl_usd")
            or trade.get("pnl_usd")
        )
        pnl = _as_float(pnl_val)
        if pnl is None:
            continue
        rr = _as_float(trade.get("rr") or trade.get("rr_achieved"))

        _record(overall, pnl, rr)

        bucket_acc = by_bucket.setdefault(bucket, _init_acc())
        _record(bucket_acc, pnl, rr)

        strat_map = by_bucket_strategy.setdefault(bucket, {})
        strat_acc = strat_map.setdefault(strategy, _init_acc())
        _record(strat_acc, pnl, rr)

    report = {
        "overall": _finalize(overall),
        "by_volume_bucket": {k: _finalize(v) for k, v in by_bucket.items()},
        "by_bucket_and_strategy": {
            bucket: {strategy: _finalize(acc) for strategy, acc in strat_map.items()}
            for bucket, strat_map in by_bucket_strategy.items()
        },
    }
    report["total_trades"] = report["overall"].get("n_trades", 0)
    return report


def write_json(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_csv(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for bucket, stats in report.get("by_volume_bucket", {}).items():
        rows.append({"bucket": bucket, "strategy": "ALL", **stats})
    for bucket, strat_map in report.get("by_bucket_and_strategy", {}).items():
        for strategy, stats in strat_map.items():
            rows.append({"bucket": bucket, "strategy": strategy, **stats})
    fieldnames = [
        "bucket",
        "strategy",
        "n_trades",
        "n_wins",
        "n_losses",
        "win_rate",
        "avg_pnl",
        "avg_rr",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TRADE_CLOSED events by volume bucket.")
    parser.add_argument("--log-dir", type=str, help="Directory containing JSONL/log files.")
    parser.add_argument("--log-file", type=str, help="Single log file to process.")
    parser.add_argument("--output", type=str, help="Path to write JSON report.")
    parser.add_argument("--csv-output", type=str, help="Optional path to write CSV output.")
    parser.add_argument("--run-id", type=str, help="Filter trades by run_id.")
    parser.add_argument("--timeframe", type=str, help="Filter trades by timeframe (e.g., 5m).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else None
    log_file = Path(args.log_file) if args.log_file else None
    if not log_dir and not log_file:
        raise SystemExit("Please provide --log-dir or --log-file")

    files = _iter_log_files(log_dir, log_file)
    trades = load_trades_from_files(files, run_id=args.run_id, timeframe=args.timeframe)
    report = aggregate_trades(trades)
    if args.run_id:
        report["run_id"] = args.run_id
    if args.timeframe:
        report["timeframe"] = args.timeframe

    if args.output:
        write_json(report, Path(args.output))
    else:
        print(json.dumps(report, indent=2))

    if args.csv_output:
        write_csv(report, Path(args.csv_output))


if __name__ == "__main__":
    main()
