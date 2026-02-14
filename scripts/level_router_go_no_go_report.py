"""Build observe/enforce Go/No-Go report for level-router rollout.

Parses structured telemetry lines:
- level_router_decision
- strategy_recheck_request
- soft_deferral_recheck_outcome
- waiting_room_drop

Focus:
- Observe-mode "would block" visibility
- Detector stability (UNKNOWN / AT_LEVEL rates)
- Recheck reliability for level-router pending flows
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LEVEL_ROUTER_DECISION_MARKER = "level_router_decision "
STRATEGY_RECHECK_REQUEST_MARKER = "strategy_recheck_request "
SOFT_DEFERRAL_RECHECK_OUTCOME_MARKER = "soft_deferral_recheck_outcome "
WAITING_ROOM_DROP_MARKER = "waiting_room_drop "


@dataclass(frozen=True)
class Thresholds:
    min_decisions: int
    min_would_block_count: int
    max_unknown_rate: float
    max_at_level_rate: float
    max_out_of_scope_rate: float
    max_level_recheck_error_rate: float
    min_recheck_outcome_coverage: float


def _extract_after_marker(line: str, marker: str) -> Optional[str]:
    idx = line.find(marker)
    if idx < 0:
        return None
    return line[idx + len(marker) :].strip()


def _parse_payload(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return bool(value)


def _iter_log_files(log_files: Iterable[str], log_glob: Optional[str]) -> List[Path]:
    out: List[Path] = []
    for p in log_files:
        path = Path(p)
        if path.exists() and path.is_file():
            out.append(path)
    if log_glob:
        out.extend(sorted(Path().glob(log_glob)))
    seen = set()
    uniq: List[Path] = []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _is_level_reason(value: Any) -> bool:
    return _safe_str(value).strip().lower().startswith("level_router.")


def _load_events(
    paths: List[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    decisions: List[Dict[str, Any]] = []
    rechecks: List[Dict[str, Any]] = []
    outcomes: List[Dict[str, Any]] = []
    drops: List[Dict[str, Any]] = []

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if LEVEL_ROUTER_DECISION_MARKER in line:
                    raw = _extract_after_marker(line, LEVEL_ROUTER_DECISION_MARKER)
                    payload = _parse_payload(raw or "")
                    if isinstance(payload, dict) and _safe_str(payload.get("event")) == "level_router_decision":
                        payload.setdefault("_source_file", str(path))
                        decisions.append(payload)
                if STRATEGY_RECHECK_REQUEST_MARKER in line:
                    raw = _extract_after_marker(line, STRATEGY_RECHECK_REQUEST_MARKER)
                    payload = _parse_payload(raw or "")
                    if isinstance(payload, dict) and _safe_str(payload.get("event")) == "strategy_recheck_request":
                        payload.setdefault("_source_file", str(path))
                        rechecks.append(payload)
                if SOFT_DEFERRAL_RECHECK_OUTCOME_MARKER in line:
                    raw = _extract_after_marker(line, SOFT_DEFERRAL_RECHECK_OUTCOME_MARKER)
                    payload = _parse_payload(raw or "")
                    if isinstance(payload, dict) and _safe_str(payload.get("event")) == "soft_deferral_recheck_outcome":
                        payload.setdefault("_source_file", str(path))
                        outcomes.append(payload)
                if WAITING_ROOM_DROP_MARKER in line:
                    raw = _extract_after_marker(line, WAITING_ROOM_DROP_MARKER)
                    payload = _parse_payload(raw or "")
                    if isinstance(payload, dict) and _safe_str(payload.get("event")) == "waiting_room_drop":
                        payload.setdefault("_source_file", str(path))
                        drops.append(payload)
    return decisions, rechecks, outcomes, drops


def _matches_scope(payload: Dict[str, Any], symbol: Optional[str], strategy: Optional[str]) -> bool:
    symbol_filter = _safe_str(symbol).strip().upper()
    strategy_filter = _safe_str(strategy).strip().lower()

    if symbol_filter:
        payload_symbol = _safe_str(payload.get("symbol")).strip().upper()
        if payload_symbol != symbol_filter:
            return False
    if strategy_filter:
        payload_strategy = _safe_str(payload.get("strategy") or payload.get("strategy_name")).strip().lower()
        if payload_strategy != strategy_filter:
            return False
    return True


def _apply_filters(
    decisions: List[Dict[str, Any]],
    rechecks: List[Dict[str, Any]],
    outcomes: List[Dict[str, Any]],
    drops: List[Dict[str, Any]],
    symbol: Optional[str],
    strategy: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    return (
        [x for x in decisions if _matches_scope(x, symbol, strategy)],
        [x for x in rechecks if _matches_scope(x, symbol, strategy)],
        [x for x in outcomes if _matches_scope(x, symbol, strategy)],
        [x for x in drops if _matches_scope(x, symbol, strategy)],
    )


def _compute_metrics(
    *,
    decisions: List[Dict[str, Any]],
    rechecks: List[Dict[str, Any]],
    outcomes: List[Dict[str, Any]],
    drops: List[Dict[str, Any]],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    total = len(decisions)
    if total == 0:
        return {
            "status": "NO_RESULT",
            "reason": "No level_router_decision events matched the selected scope.",
            "metrics": {},
            "go_no_go": {},
            "failed_gates": [],
        }

    scope_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    zone_counts: Counter[str] = Counter()
    rollout_counts: Counter[str] = Counter()

    would_block_count = 0
    blocked_count = 0
    unknown_count = 0
    at_level_count = 0
    out_of_scope_count = 0

    for d in decisions:
        scope = _safe_str(d.get("scope")).strip().lower() or "unknown"
        reason = _safe_str(d.get("reason_code")).strip().lower()
        zone = _safe_str(d.get("zone")).strip().upper() or "UNKNOWN"
        rollout_mode = _safe_str(d.get("rollout_mode")).strip().lower() or "unknown"
        allowed = _safe_bool(d.get("allowed"))

        scope_counts[scope] += 1
        reason_counts[reason] += 1
        zone_counts[zone] += 1
        rollout_counts[rollout_mode] += 1

        if reason == "level_router.observe_would_block":
            would_block_count += 1
        if not allowed:
            blocked_count += 1
        if zone == "UNKNOWN":
            unknown_count += 1
        if zone == "AT_LEVEL":
            at_level_count += 1
        if reason == "level_router.rollout_out_of_scope":
            out_of_scope_count += 1

    would_block_rate = would_block_count / total
    blocked_rate = blocked_count / total
    unknown_rate = unknown_count / total
    at_level_rate = at_level_count / total
    out_of_scope_rate = out_of_scope_count / total

    level_rechecks = [
        r
        for r in rechecks
        if _is_level_reason(r.get("pending_reason_code")) or _is_level_reason(r.get("reason_code"))
    ]
    level_rechecks_at_level = [
        r for r in level_rechecks if _safe_str(r.get("pending_reason_code")).strip().lower() == "level_router.at_level"
    ]

    level_outcomes = [
        o
        for o in outcomes
        if _is_level_reason(o.get("pending_reason_code")) or _is_level_reason(o.get("final_reason"))
    ]
    level_outcome_total = len(level_outcomes)
    level_outcome_error_count = sum(1 for o in level_outcomes if _safe_str(o.get("outcome")).strip().lower() == "error")
    level_outcome_error_rate = (level_outcome_error_count / level_outcome_total) if level_outcome_total > 0 else None
    level_loop_prevented_count = sum(
        1 for o in level_outcomes if _safe_str(o.get("error_code")).strip().lower() == "loop_prevented"
    )
    level_rearm_fast_watch_count = sum(1 for o in level_outcomes if _safe_bool(o.get("rearm_fast_watch")))
    level_signal_emitted_count = sum(
        1 for o in level_outcomes if _safe_str(o.get("outcome")).strip().lower() == "signal_emitted"
    )
    level_no_signal_count = sum(1 for o in level_outcomes if _safe_str(o.get("outcome")).strip().lower() == "no_signal")
    level_final_reason_counts: Counter[str] = Counter(
        _safe_str(o.get("final_reason")).strip().lower() or "unknown" for o in level_outcomes
    )

    level_waiting_room_drops = [d for d in drops if _is_level_reason(d.get("reason_code"))]
    level_drop_reason_counts: Counter[str] = Counter(
        _safe_str(d.get("drop_reason")).strip().lower() or "unknown" for d in level_waiting_room_drops
    )

    req_parent_ids = {
        _safe_str(r.get("parent_pending_id")).strip()
        for r in level_rechecks_at_level
        if _safe_str(r.get("parent_pending_id")).strip()
    }
    out_parent_ids = {
        _safe_str(o.get("parent_pending_id")).strip()
        for o in level_outcomes
        if _safe_str(o.get("parent_pending_id")).strip()
    }
    recheck_outcome_coverage = None
    if req_parent_ids:
        recheck_outcome_coverage = len(req_parent_ids & out_parent_ids) / len(req_parent_ids)

    go_no_go: Dict[str, Any] = {
        "min_decisions_met": total >= thresholds.min_decisions,
        "would_block_samples_met": would_block_count >= thresholds.min_would_block_count,
        "unknown_rate_ok": unknown_rate <= thresholds.max_unknown_rate,
        "at_level_rate_ok": at_level_rate <= thresholds.max_at_level_rate,
        "out_of_scope_rate_ok": out_of_scope_rate <= thresholds.max_out_of_scope_rate,
        "loop_prevented_zero": level_loop_prevented_count == 0,
    }

    if level_outcome_error_rate is None:
        go_no_go["level_recheck_error_rate_ok"] = "no_level_rechecks"
    else:
        go_no_go["level_recheck_error_rate_ok"] = level_outcome_error_rate <= thresholds.max_level_recheck_error_rate

    if recheck_outcome_coverage is None:
        go_no_go["recheck_outcome_coverage_ok"] = "no_level_rechecks"
    else:
        go_no_go["recheck_outcome_coverage_ok"] = recheck_outcome_coverage >= thresholds.min_recheck_outcome_coverage

    failed_gates = [key for key, value in go_no_go.items() if value is False]
    status = "GO" if not failed_gates else "NO_GO"

    return {
        "status": status,
        "metrics": {
            "total_level_router_decisions": total,
            "scope_counts": dict(scope_counts),
            "reason_counts": dict(reason_counts),
            "zone_counts": dict(zone_counts),
            "rollout_mode_counts": dict(rollout_counts),
            "would_block_count": would_block_count,
            "would_block_rate": would_block_rate,
            "blocked_count": blocked_count,
            "blocked_rate": blocked_rate,
            "unknown_count": unknown_count,
            "unknown_rate": unknown_rate,
            "at_level_count": at_level_count,
            "at_level_rate": at_level_rate,
            "out_of_scope_count": out_of_scope_count,
            "out_of_scope_rate": out_of_scope_rate,
            "level_recheck_request_count": len(level_rechecks),
            "level_recheck_at_level_request_count": len(level_rechecks_at_level),
            "level_recheck_outcome_count": level_outcome_total,
            "level_recheck_outcome_error_count": level_outcome_error_count,
            "level_recheck_outcome_error_rate": level_outcome_error_rate,
            "level_loop_prevented_count": level_loop_prevented_count,
            "level_rearm_fast_watch_count": level_rearm_fast_watch_count,
            "level_signal_emitted_count": level_signal_emitted_count,
            "level_no_signal_count": level_no_signal_count,
            "level_final_reason_counts": dict(level_final_reason_counts),
            "level_waiting_room_drop_count": len(level_waiting_room_drops),
            "level_waiting_room_drop_reason_counts": dict(level_drop_reason_counts),
            "level_recheck_outcome_coverage": recheck_outcome_coverage,
        },
        "go_no_go": go_no_go,
        "failed_gates": failed_gates,
    }


def _print_summary(report: Dict[str, Any], files: List[Path], symbol: Optional[str], strategy: Optional[str]) -> None:
    print("=== Level Router Go/No-Go Report ===")
    print(f"files={len(files)} symbol={symbol or '*'} strategy={strategy or '*'}")
    print(f"status={report.get('status')}")
    if report.get("status") == "NO_RESULT":
        print(f"reason={report.get('reason')}")
        return

    metrics = report.get("metrics", {})
    print(
        "decisions={} would_block_rate={:.2%} unknown_rate={:.2%} at_level_rate={:.2%} blocked_rate={:.2%}".format(
            int(metrics.get("total_level_router_decisions") or 0),
            float(metrics.get("would_block_rate") or 0.0),
            float(metrics.get("unknown_rate") or 0.0),
            float(metrics.get("at_level_rate") or 0.0),
            float(metrics.get("blocked_rate") or 0.0),
        )
    )
    print(
        "level_recheck: req={} outcomes={} error_rate={} coverage={}".format(
            int(metrics.get("level_recheck_request_count") or 0),
            int(metrics.get("level_recheck_outcome_count") or 0),
            metrics.get("level_recheck_outcome_error_rate"),
            metrics.get("level_recheck_outcome_coverage"),
        )
    )
    print(f"failed_gates={report.get('failed_gates') or []}")
    print("gate_status:")
    for key, value in sorted((report.get("go_no_go") or {}).items()):
        print(f"  - {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build level-router observe/enforce Go/No-Go report from telemetry logs.")
    parser.add_argument("--log-file", action="append", default=[], help="Specific log file path. Repeatable.")
    parser.add_argument(
        "--log-glob",
        default="logs/live_trading_*.log",
        help="Glob pattern for log files (default: logs/live_trading_*.log)",
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help='Filter symbol (default: "BTC/USDT:USDT").')
    parser.add_argument("--strategy", default=None, help='Optional strategy filter (e.g. "adaptive_ob").')
    parser.add_argument("--min-decisions", type=int, default=500, help="Minimum required level_router_decision samples.")
    parser.add_argument(
        "--min-would-block-count",
        type=int,
        default=20,
        help="Minimum observe_would_block sample count for rollout confidence.",
    )
    parser.add_argument("--max-unknown-rate", type=float, default=0.40, help="Maximum allowed UNKNOWN zone rate.")
    parser.add_argument("--max-at-level-rate", type=float, default=0.65, help="Maximum allowed AT_LEVEL zone rate.")
    parser.add_argument("--max-out-of-scope-rate", type=float, default=0.00, help="Maximum rollout_out_of_scope rate.")
    parser.add_argument(
        "--max-level-recheck-error-rate",
        type=float,
        default=0.02,
        help="Maximum allowed error rate in level-related soft_deferral_recheck_outcome events.",
    )
    parser.add_argument(
        "--min-recheck-outcome-coverage",
        type=float,
        default=0.95,
        help="Minimum parent_pending_id coverage between level recheck requests and outcomes.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write report json.")
    args = parser.parse_args()

    files = _iter_log_files(args.log_file, args.log_glob)
    if not files:
        print("No log files matched.")
        return 2

    decisions, rechecks, outcomes, drops = _load_events(files)
    decisions, rechecks, outcomes, drops = _apply_filters(
        decisions=decisions,
        rechecks=rechecks,
        outcomes=outcomes,
        drops=drops,
        symbol=args.symbol,
        strategy=args.strategy,
    )

    thresholds = Thresholds(
        min_decisions=max(int(args.min_decisions), 1),
        min_would_block_count=max(int(args.min_would_block_count), 0),
        max_unknown_rate=float(args.max_unknown_rate),
        max_at_level_rate=float(args.max_at_level_rate),
        max_out_of_scope_rate=float(args.max_out_of_scope_rate),
        max_level_recheck_error_rate=float(args.max_level_recheck_error_rate),
        min_recheck_outcome_coverage=float(args.min_recheck_outcome_coverage),
    )

    report = _compute_metrics(
        decisions=decisions,
        rechecks=rechecks,
        outcomes=outcomes,
        drops=drops,
        thresholds=thresholds,
    )
    report["scope"] = {
        "files": [str(p) for p in files],
        "symbol": args.symbol,
        "strategy": args.strategy,
    }
    report["event_counts"] = {
        "level_router_decision": len(decisions),
        "strategy_recheck_request": len(rechecks),
        "soft_deferral_recheck_outcome": len(outcomes),
        "waiting_room_drop": len(drops),
    }

    _print_summary(report, files, args.symbol, args.strategy)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"wrote={out}")

    return 0 if report.get("status") in {"GO", "NO_GO", "NO_RESULT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
