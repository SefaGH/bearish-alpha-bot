"""
Comprehensive Volume Analysis Script for Trading Bot Logs.

Features:
1. TRADE ANALYSIS: Analyzes TRADE_CLOSED events by volume bucket (Original feature).
2. THRESHOLD ANALYSIS: Scans `volume_decision_check` and `TRADE_CLOSED` events to analyze the distribution of volume_strength.
3. RECOMMENDATIONS: Suggests new volume thresholds based on actual market data percentiles (when data exists).

Usage:
    python analyze_volume_buckets.py --log-file live_trading_20251218.log
    python analyze_volume_buckets.py --log-dir logs
"""

import argparse
import csv
import json
import ast
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Constants ---
TRADE_EVENT_NAME = "TRADE_CLOSED"
VOLUME_CONTEXT_EVENTS = {"volume_context", "volume_decision_check"}
# VolumeAnalyzer buckets are derived from `volume_strength` (0..1). In logs we may see:
# - `volume_decision_check` events (signal-time audit)
# - `TRADE_CLOSED` events (trade record, carries *_at_entry fields)
# - legacy keys (`ratio_combined`) or renamed keys (`volume_ratio_combined`)
VOLUME_STRENGTH_KEYS = (
    "volume_strength",
    "volume_strength_at_entry",
    "volume_score",
    "volume_component",
)
VOLUME_RATIO_KEYS = (
    "volume_ratio_combined",
    "ratio_combined",
    "ratio",
)
VOLUME_BUCKET_KEYS = (
    "volume_bucket",
    "volume_bucket_at_entry",
    "bucket",
)

DEFAULT_SIGMOID_ALPHA = 1.2  # matches `src/core/volume_analyzer.py` default

# --- Helper Functions ---

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
    
    # Try parsing strictly as JSON first
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
        
    # Attempt to find JSON start
    brace_idx = line.find("{")
    if brace_idx != -1:
        candidate = line[brace_idx:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
            
    # Attempt to parse python dict string (common in some loggers: {'key': 'val'})
    # Only runs after JSON parsing fails; uses literal_eval (safe subset).
    if brace_idx != -1:
        try:
            candidate = line[brace_idx:]
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError):
            pass

    return None

def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

def _iter_log_files(log_dir: Optional[Path], log_file: Optional[Path]) -> List[Path]:
    files: List[Path] = []
    if log_file:
        files.append(log_file)
    if log_dir:
        for pattern in ("*.log", "*.jsonl", "*.txt"):
            files.extend(sorted(log_dir.glob(pattern)))
    return files

# --- Analysis Logic ---

def _sigmoid_volume_strength(ratio_combined: float, alpha: float = DEFAULT_SIGMOID_ALPHA) -> float:
    # matches: 1 / (1 + exp(-alpha * (ratio - 1.0)))
    x = alpha * (ratio_combined - 1.0)
    return float(1.0 / (1.0 + np.exp(-x)))

def _extract_volume_sample(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], str]:
    """
    Returns (volume_strength, bucket, source_label).
    Prefers true `volume_strength`; can derive from `*_ratio_combined` if needed.
    """
    event = str(payload.get("event") or "")

    bucket: Optional[str] = None
    for key in VOLUME_BUCKET_KEYS:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            bucket = val.strip()
            break

    for key in VOLUME_STRENGTH_KEYS:
        strength = _as_float(payload.get(key))
        if strength is not None:
            return strength, bucket, (event or f"key:{key}")

    for key in VOLUME_RATIO_KEYS:
        ratio = _as_float(payload.get(key))
        if ratio is not None:
            return _sigmoid_volume_strength(ratio), bucket, (event or f"key:{key}")

    return None, bucket, (event or "unknown")

def analyze_volume_thresholds(files: Iterable[Path]) -> Dict[str, Any]:
    """Scans logs for volume-related events and calculates distribution stats for volume_strength."""
    ratios = []
    buckets = []
    source_counts: Dict[str, int] = {}
    files_scanned = 0
    
    for file_path in files:
        try:
            files_scanned += 1
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Fast pre-filter: most lines are plain text.
                    if "{" not in line:
                        continue
                    lower = line.lower()
                    if ("volume" not in lower) and (TRADE_EVENT_NAME not in line):
                        continue
                    payload = _extract_json(line)
                    if not payload:
                        continue
                    strength, bucket, source = _extract_volume_sample(payload)
                    if strength is not None:
                        ratios.append(strength)
                        source_counts[source] = source_counts.get(source, 0) + 1
                    if bucket:
                        buckets.append(bucket)
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            continue

    if not ratios:
        # Avoid emitting an `"error"` key because upstream tooling often treats it as a hard failure.
        return {
            "status": "no_data",
            "message": (
                "No volume decision samples found. This usually means the run produced no trade signals "
                "(so `volume_decision_check` was never logged) and no `TRADE_CLOSED` events exist in the file(s)."
            ),
            "files_scanned": files_scanned,
            "source_counts": source_counts,
            "count": 0,
            "current_bucket_distribution": {},
            "recommended_thresholds": None,
        }

    # Calculate Statistics
    data = np.array(ratios)
    # Daha hassas dilimler alalım
    p25, p50, p60, p75, p90 = np.percentile(data, [25, 50, 60, 75, 90])
    max_val = float(np.max(data))
    
    # --- AKILLI ARALIK MANTIĞI (SMART BUCKETING) ---
    
    # 1. LOW: Alt %25 her zaman güvenlidir.
    low_max = round(float(p25), 2)
    
    # 2. NORMAL:
    # Eğer P75, Max değere çok yakınsa (doygunluk varsa), Normal'i biraz daraltıp (P60 veya P50)
    # High'a yer açmamız gerekir.
    if p75 >= (max_val - 0.1): # Doygunluk kontrolü (10.0 == 10.0)
        normal_max = round(float(p50), 2) # Normal'i Medyana çek
    else:
        normal_max = round(float(p75), 2)

    # 3. HIGH:
    # Eğer yukarıda Normal'i daralttıysak, High için P50 ile Max arası kalır.
    # Çakışmayı önlemek için kontrol:
    high_check = round(float(p90), 2)
    
    if high_check <= normal_max:
        # Hala çakışma varsa, Normal ile Max'in tam ortasını seç
        high_max = round(normal_max + ((max_val - normal_max) / 2), 2)
    else:
        high_max = high_check
        
    # Eğer hesaplanan High Max, mutlak Max'a eşitse, Extreme için küçücük bir pay bırak
    if high_max >= max_val:
         high_max = round(max(max_val - 0.1, 0.0), 2)

    # Bucket Distribution
    bucket_dist: Dict[str, float] = {}
    total_buckets = len(buckets)
    if total_buckets:
        bucket_counts = {b: buckets.count(b) for b in set(buckets)}
        bucket_dist = {b: (count / total_buckets * 100) for b, count in bucket_counts.items()}

    recommended_thresholds = {
        "LOW_MAX": low_max,
        "NORMAL_MAX": normal_max,
        "HIGH_MAX": high_max,
        "EXTREME_MIN": high_max # Extreme bu değerden başlar
    }

    return {
        "status": "ok",
        "count": len(data),
        "mean": float(np.mean(data)),
        "max": max_val,
        "percentiles": {"p25": p25, "p50": p50, "p75": p75, "p90": p90},
        "current_bucket_distribution": bucket_dist,
        "recommended_thresholds": recommended_thresholds,
        "source_counts": source_counts,
        "files_scanned": files_scanned,
    }

def load_trades_from_files(files: Iterable[Path], run_id: Optional[str] = None, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load TRADE_CLOSED events from the given files (Original Logic)."""
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TRADE_CLOSED events and Volume Thresholds.")
    parser.add_argument("--log-dir", type=str, help="Directory containing JSONL/log files.")
    parser.add_argument("--log-file", type=str, help="Single log file to process.")
    parser.add_argument("--output", type=str, help="Path to write JSON report.")
    parser.add_argument("--run-id", type=str, help="Filter trades by run_id.")
    parser.add_argument("--timeframe", type=str, help="Filter trades by timeframe (e.g., 5m).")
    return parser.parse_args()

def main() -> None:
    _configure_stdout()
    args = parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else None
    log_file = Path(args.log_file) if args.log_file else None
    if not log_dir and not log_file:
        raise SystemExit("Please provide --log-dir or --log-file")

    files = _iter_log_files(log_dir, log_file)
    
    # 1. Analyze Trades
    trades = load_trades_from_files(files, run_id=args.run_id, timeframe=args.timeframe)
    trade_report = aggregate_trades(trades)
    
    # 2. Analyze Volume Thresholds
    # We need to re-iterate files or handle differently, but _iter_log_files returns Paths so we can just pass it again
    volume_report = analyze_volume_thresholds(files)
    
    final_report = {
        "trade_analysis": trade_report,
        "volume_threshold_analysis": volume_report
    }

    if args.run_id:
        final_report["run_id"] = args.run_id
    if args.timeframe:
        final_report["timeframe"] = args.timeframe

    if args.output:
        write_json(final_report, Path(args.output))
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(final_report, indent=2))
        
    # Console Summary for the User
    if volume_report.get("status") == "ok":
        print("\n" + "="*60)
        print("📊 VOLUME THRESHOLD ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Data Points Scanned: {volume_report['count']}")
        print(f"Current Bucket Distribution: {json.dumps(volume_report['current_bucket_distribution'], indent=2)}")
        print("-" * 60)
        print("💡 RECOMMENDED THRESHOLDS (Based on P25/P75/P90):")
        rec = volume_report['recommended_thresholds']
        print(f"   LOW     : < {rec['LOW_MAX']}")
        print(f"   NORMAL  : {rec['LOW_MAX']} - {rec['NORMAL_MAX']}")
        print(f"   HIGH    : {rec['NORMAL_MAX']} - {rec['HIGH_MAX']}")
        print(f"   EXTREME : > {rec['EXTREME_MIN']}")
        print("="*60)
    elif volume_report.get("status") == "no_data":
        print("\n" + "="*60)
        print("?? VOLUME THRESHOLD ANALYSIS SUMMARY")
        print("="*60)
        print("No volume samples found in the provided log(s).")
        print(f"Files scanned: {volume_report.get('files_scanned', 0)}")
        print(f"Source counts: {json.dumps(volume_report.get('source_counts', {}), indent=2)}")
        print("="*60)

if __name__ == "__main__":
    main()
