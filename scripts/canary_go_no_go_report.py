"""Generate an operational Go/No-Go canary report from execution telemetry logs.

This script parses structured telemetry events emitted by:
- LiveTradingEngine: `order_decision_trace`, `order_decision_outcome`
- OrderManager: `order_manager_decision` (optional cross-check)

It computes hard-gate metrics for canary rollout decisions:
- env_forced_order_type guard
- smart_entry_applied_rate
- missing_atr_force_market rate
- market_rate_by_bucket (focus: EXTREME)
- ABORT:NO_FILL_TIMEOUT rate
- ABORT:STOP_HIT_BEFORE_ENTRY_CANCEL_UNCONFIRMED count/rate
- FILLED_DURING_STOP_ABORT count/rate
- ATR freshness violations (market while atr_age_ms > threshold)
- time_to_fill_ms distribution (p50/p90/p95)
- planned_vs_realized_rr_drift distribution from TRADE_CLOSED

Notes:
- Some requested KPIs are not fully derivable in all runs. The script flags
  missing fields as telemetry gaps instead of fabricating numbers.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRACE_MARKER = "order_decision_trace "
OUTCOME_MARKER = "order_decision_outcome "
ORDER_MGR_MARKER = "order_manager_decision "
TRADE_CLOSED_MARKER = "TRADE_CLOSED "
RECON_WATCHDOG_MARKER = "[RECON-WATCHDOG]"


@dataclass(frozen=True)
class Thresholds:
    smart_entry_applied_min_rate: float
    missing_atr_force_market_max_rate: float
    extreme_market_max_rate: float
    atr_age_threshold_ms: int
    missed_fill_increase_max_pct: float
    max_stop_abort_cancel_unconfirmed: Optional[int]
    max_recon_orphans_detected: Optional[int]
    max_recon_stale_removed: Optional[int]
    max_recon_orphans_adopted: Optional[int]
    require_recon_watchdog_events: bool


def _extract_after_marker(line: str, marker: str) -> Optional[str]:
    idx = line.find(marker)
    if idx < 0:
        return None
    return line[idx + len(marker) :].strip()


def _parse_python_dict_payload(raw: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    vs = sorted(float(v) for v in values)
    if len(vs) == 1:
        return vs[0]
    q = max(0.0, min(1.0, float(q)))
    idx = q * (len(vs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vs[lo]
    frac = idx - lo
    return vs[lo] + (vs[hi] - vs[lo]) * frac


def _weighted_quantile(values: List[float], weights: List[float], q: float) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    pairs = sorted((float(v), max(float(w), 0.0)) for v, w in zip(values, weights))
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    target = max(0.0, min(1.0, float(q))) * total_w
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            return v
    return pairs[-1][0]


def _iter_log_files(log_files: Iterable[str], log_glob: Optional[str]) -> List[Path]:
    out: List[Path] = []
    for p in log_files:
        path = Path(p)
        if path.exists() and path.is_file():
            out.append(path)
    if log_glob:
        out.extend(sorted(Path().glob(log_glob)))
    # Unique while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _parse_recon_watchdog_line(line: str) -> Optional[Dict[str, Any]]:
    if RECON_WATCHDOG_MARKER not in line:
        return None

    # Cycle summary from LiveTradingEngine:
    # [RECON-WATCHDOG] stale_removed=0 orphans_detected=0 orphans_adopted=0 active_positions=0
    cycle_match = re.search(
        r"stale_removed=(?P<stale>\d+)\s+orphans_detected=(?P<orphans>\d+)\s+orphans_adopted=(?P<adopted>\d+)",
        line,
    )
    if cycle_match:
        return {
            "kind": "cycle",
            "stale_removed": int(cycle_match.group("stale")),
            "orphans_detected": int(cycle_match.group("orphans")),
            "orphans_adopted": int(cycle_match.group("adopted")),
        }

    # PositionManager orphan discovery:
    # [RECON-WATCHDOG] Orphan exchange positions detected: count=1 details=[...]
    detect_match = re.search(r"Orphan exchange positions detected:\s*count=(?P<count>\d+)", line)
    if detect_match:
        return {
            "kind": "orphans_detected",
            "orphans_detected": int(detect_match.group("count")),
        }

    # PositionManager adopt log:
    # [RECON-WATCHDOG] Adopted orphan position: ...
    if "Adopted orphan position:" in line:
        return {
            "kind": "orphans_adopted",
            "orphans_adopted": 1,
        }

    return {"kind": "other"}


def _load_events(
    paths: List[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    traces: List[Dict[str, Any]] = []
    outcomes: List[Dict[str, Any]] = []
    mgr: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    recon: List[Dict[str, Any]] = []

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if TRACE_MARKER in line:
                    raw = _extract_after_marker(line, TRACE_MARKER)
                    if raw:
                        payload = _parse_python_dict_payload(raw)
                        if payload and payload.get("event") == "order_decision_trace":
                            payload.setdefault("_source_file", str(path))
                            traces.append(payload)
                if OUTCOME_MARKER in line:
                    raw = _extract_after_marker(line, OUTCOME_MARKER)
                    if raw:
                        payload = _parse_python_dict_payload(raw)
                        if payload and payload.get("event") == "order_decision_outcome":
                            payload.setdefault("_source_file", str(path))
                            outcomes.append(payload)
                if ORDER_MGR_MARKER in line:
                    raw = _extract_after_marker(line, ORDER_MGR_MARKER)
                    if raw:
                        payload = _parse_python_dict_payload(raw)
                        if payload and payload.get("event") == "order_manager_decision":
                            payload.setdefault("_source_file", str(path))
                            mgr.append(payload)
                if TRADE_CLOSED_MARKER in line:
                    raw = _extract_after_marker(line, TRADE_CLOSED_MARKER)
                    if raw:
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            payload = None
                        if isinstance(payload, dict) and str(payload.get("event", "")).upper() == "TRADE_CLOSED":
                            payload.setdefault("_source_file", str(path))
                            trades.append(payload)
                recon_payload = _parse_recon_watchdog_line(line)
                if recon_payload:
                    recon_payload.setdefault("_source_file", str(path))
                    recon.append(recon_payload)
    return traces, outcomes, mgr, trades, recon


def _apply_filters(
    traces: List[Dict[str, Any]],
    outcomes: List[Dict[str, Any]],
    mgr: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    symbol: Optional[str],
    strategy: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sym = _safe_str(symbol).strip()
    strat = _safe_str(strategy).strip()

    def ok_trace(p: Dict[str, Any]) -> bool:
        if sym and _safe_str(p.get("symbol")) != sym:
            return False
        if strat and _safe_str(p.get("strategy_name")) != strat:
            return False
        return True

    def ok_outcome(p: Dict[str, Any]) -> bool:
        if sym and _safe_str(p.get("symbol")) != sym:
            return False
        if strat:
            out_strat = _safe_str(p.get("strategy_name")).strip()
            if out_strat and out_strat != strat:
                return False
        return True

    def ok_mgr(p: Dict[str, Any]) -> bool:
        if sym and _safe_str(p.get("symbol")) != sym:
            return False
        return True

    def ok_trade(p: Dict[str, Any]) -> bool:
        if sym and _safe_str(p.get("symbol")) != sym:
            return False
        if strat:
            t_strat = _safe_str(p.get("strategy_name") or p.get("strategy")).strip()
            if t_strat and t_strat != strat:
                return False
        return True

    return (
        [x for x in traces if ok_trace(x)],
        [x for x in outcomes if ok_outcome(x)],
        [x for x in mgr if ok_mgr(x)],
        [x for x in trades if ok_trade(x)],
    )


def _compute_metrics(
    traces: List[Dict[str, Any]],
    outcomes: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    recon_events: List[Dict[str, Any]],
    thresholds: Thresholds,
    baseline: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    total = len(traces)
    if total == 0:
        return {
            "status": "NO_RESULT",
            "reason": "No order_decision_trace events matched the selected scope.",
            "metrics": {},
            "go_no_go": {},
            "telemetry_gaps": [
                "No trace events in scope.",
                "Slippage/overshoot metrics require order_decision_outcome and/or TRADE_CLOSED events.",
            ],
        }

    applied_cnt = sum(1 for t in traces if bool(t.get("policy_applied")))
    smart_entry_applied_rate = applied_cnt / total

    missing_atr_cnt = 0
    env_forced_market_cnt = 0
    atr_stale_market_violations = 0

    bucket_totals: Dict[str, int] = defaultdict(int)
    bucket_markets: Dict[str, int] = defaultdict(int)

    for t in traces:
        decision = _safe_str(t.get("policy_decision"))
        fb = _safe_str(t.get("fallback_reason"))
        if "missing_atr_force_market" in decision or "missing_atr_force_market" in fb:
            missing_atr_cnt += 1

        if _safe_str(t.get("env_forced_order_type")).lower() == "market":
            env_forced_market_cnt += 1

        bucket = _safe_str(t.get("bucket")).upper().strip() or "UNKNOWN"
        bucket_totals[bucket] += 1
        eff = _safe_str(t.get("effective_order_type")).lower().strip()
        if eff == "market":
            bucket_markets[bucket] += 1

        atr_age_ms = _safe_int(t.get("atr_age_ms"))
        if atr_age_ms is not None and atr_age_ms > thresholds.atr_age_threshold_ms and eff == "market":
            atr_stale_market_violations += 1

    missing_atr_rate = missing_atr_cnt / total

    market_rate_by_bucket: Dict[str, float] = {}
    for b, n in bucket_totals.items():
        m = bucket_markets.get(b, 0)
        market_rate_by_bucket[b] = (m / n) if n > 0 else 0.0

    extreme_market_rate = market_rate_by_bucket.get("EXTREME", 0.0)

    outcome_total = len(outcomes)
    fallback_reason_counts: Counter[str] = Counter()
    outcome_reason_counts: Counter[str] = Counter()
    abort_no_fill_timeout_cnt = 0
    abort_stop_hit_cancel_unconfirmed_cnt = 0
    filled_during_stop_abort_cnt = 0
    for o in outcomes:
        fr = _safe_str(o.get("fallback_reason")).strip()
        if fr:
            fallback_reason_counts[fr] += 1
        reason = _safe_str(o.get("reason")).strip()
        reason_upper = reason.upper()
        if reason:
            outcome_reason_counts[reason] += 1
        if reason_upper.startswith("ABORT:NO_FILL_TIMEOUT"):
            abort_no_fill_timeout_cnt += 1
        if reason_upper == "ABORT:STOP_HIT_BEFORE_ENTRY_CANCEL_UNCONFIRMED":
            abort_stop_hit_cancel_unconfirmed_cnt += 1
        if reason_upper == "FILLED_DURING_STOP_ABORT":
            filled_during_stop_abort_cnt += 1
    abort_no_fill_timeout_rate = (abort_no_fill_timeout_cnt / outcome_total) if outcome_total > 0 else 0.0
    abort_stop_hit_cancel_unconfirmed_rate = (
        abort_stop_hit_cancel_unconfirmed_cnt / outcome_total
    ) if outcome_total > 0 else 0.0
    filled_during_stop_abort_rate = (filled_during_stop_abort_cnt / outcome_total) if outcome_total > 0 else 0.0

    slippages: List[float] = []
    slippage_weights: List[float] = []
    time_to_fill_ms_values: List[float] = []
    for o in outcomes:
        s = _safe_float(o.get("entry_slippage_bps"))
        n = _safe_float(o.get("entry_notional_usd"))
        if s is not None:
            slippages.append(float(s))
            slippage_weights.append(float(n) if n is not None and n > 0 else 1.0)
        ttf = _safe_float(o.get("time_to_fill_ms"))
        if ttf is not None and ttf >= 0:
            time_to_fill_ms_values.append(float(ttf))

    trade_slippage_p90 = _quantile(slippages, 0.90) if slippages else None
    trade_slippage_p95 = _quantile(slippages, 0.95) if slippages else None
    notional_slippage_p90 = _weighted_quantile(slippages, slippage_weights, 0.90) if slippages else None
    notional_slippage_p95 = _weighted_quantile(slippages, slippage_weights, 0.95) if slippages else None
    time_to_fill_p50_ms = _quantile(time_to_fill_ms_values, 0.50) if time_to_fill_ms_values else None
    time_to_fill_p90_ms = _quantile(time_to_fill_ms_values, 0.90) if time_to_fill_ms_values else None
    time_to_fill_p95_ms = _quantile(time_to_fill_ms_values, 0.95) if time_to_fill_ms_values else None

    stop_overshoots: List[float] = []
    rr_drifts: List[float] = []
    for t in trades:
        v = _safe_float(t.get("stop_overshoot_bps"))
        if v is not None:
            stop_overshoots.append(float(v))
        drift = _safe_float(t.get("planned_vs_realized_rr_drift"))
        if drift is None:
            rr_after = _safe_float(t.get("rr_after_fill"))
            rr_achieved = _safe_float(t.get("rr_achieved"))
            if rr_after is not None and rr_achieved is not None:
                drift = float(rr_achieved) - float(rr_after)
        if drift is not None:
            rr_drifts.append(abs(float(drift)))
    stop_overshoot_p90 = _quantile(stop_overshoots, 0.90) if stop_overshoots else None
    stop_overshoot_p95 = _quantile(stop_overshoots, 0.95) if stop_overshoots else None
    rr_drift_abs_p90 = _quantile(rr_drifts, 0.90) if rr_drifts else None
    rr_drift_abs_p95 = _quantile(rr_drifts, 0.95) if rr_drifts else None

    recon_cycles = [e for e in recon_events if _safe_str(e.get("kind")) == "cycle"]
    if recon_cycles:
        recon_stale_removed_total = sum(int(e.get("stale_removed") or 0) for e in recon_cycles)
        recon_orphans_detected_total = sum(int(e.get("orphans_detected") or 0) for e in recon_cycles)
        recon_orphans_adopted_total = sum(int(e.get("orphans_adopted") or 0) for e in recon_cycles)
    else:
        recon_stale_removed_total = 0
        recon_orphans_detected_total = sum(int(e.get("orphans_detected") or 0) for e in recon_events)
        recon_orphans_adopted_total = sum(int(e.get("orphans_adopted") or 0) for e in recon_events)

    missed_fill_increase_pct = None
    baseline_abort = None
    if baseline and isinstance(baseline, dict):
        baseline_abort = _safe_float(
            (baseline.get("metrics") or {}).get("abort_no_fill_timeout_rate")
            if isinstance(baseline.get("metrics"), dict)
            else baseline.get("abort_no_fill_timeout_rate")
        )
    if baseline_abort is not None:
        if baseline_abort > 0:
            missed_fill_increase_pct = ((abort_no_fill_timeout_rate - baseline_abort) / baseline_abort) * 100.0
        elif abort_no_fill_timeout_rate > 0:
            missed_fill_increase_pct = float("inf")
        else:
            missed_fill_increase_pct = 0.0

    go_no_go = {
        "env_forced_market_zero": env_forced_market_cnt == 0,
        "smart_entry_applied_rate_ok": smart_entry_applied_rate >= thresholds.smart_entry_applied_min_rate,
        "missing_atr_force_market_rate_ok": missing_atr_rate <= thresholds.missing_atr_force_market_max_rate,
        "extreme_market_rate_ok": extreme_market_rate <= thresholds.extreme_market_max_rate,
        "atr_freshness_violations_zero": atr_stale_market_violations == 0,
    }

    if missed_fill_increase_pct is not None:
        go_no_go["missed_fill_increase_ok"] = missed_fill_increase_pct <= thresholds.missed_fill_increase_max_pct
    else:
        go_no_go["missed_fill_increase_ok"] = "baseline_not_provided"

    if thresholds.max_stop_abort_cancel_unconfirmed is not None:
        go_no_go["stop_abort_cancel_unconfirmed_ok"] = (
            abort_stop_hit_cancel_unconfirmed_cnt <= int(thresholds.max_stop_abort_cancel_unconfirmed)
        )
    else:
        go_no_go["stop_abort_cancel_unconfirmed_ok"] = "not_configured"

    if thresholds.max_recon_orphans_detected is not None:
        go_no_go["recon_orphans_detected_ok"] = recon_orphans_detected_total <= int(thresholds.max_recon_orphans_detected)
    else:
        go_no_go["recon_orphans_detected_ok"] = "not_configured"
    if thresholds.max_recon_stale_removed is not None:
        go_no_go["recon_stale_removed_ok"] = recon_stale_removed_total <= int(thresholds.max_recon_stale_removed)
    else:
        go_no_go["recon_stale_removed_ok"] = "not_configured"
    if thresholds.max_recon_orphans_adopted is not None:
        go_no_go["recon_orphans_adopted_ok"] = recon_orphans_adopted_total <= int(thresholds.max_recon_orphans_adopted)
    else:
        go_no_go["recon_orphans_adopted_ok"] = "not_configured"
    if thresholds.require_recon_watchdog_events:
        go_no_go["recon_events_present"] = len(recon_events) > 0
    else:
        go_no_go["recon_events_present"] = "not_required"

    hard_failed = [k for k, v in go_no_go.items() if v is False]
    status = "GO" if not hard_failed else "NO_GO"

    telemetry_gaps: List[str] = []
    if not slippages:
        telemetry_gaps.append("No entry_slippage_bps observed in order_decision_outcome events.")
    if not time_to_fill_ms_values:
        telemetry_gaps.append("No time_to_fill_ms observed in order_decision_outcome events.")
    if not stop_overshoots:
        telemetry_gaps.append("No stop_overshoot_bps observed in TRADE_CLOSED events.")
    if not rr_drifts:
        telemetry_gaps.append("No planned_vs_realized_rr_drift (or rr_after_fill/rr_achieved) in TRADE_CLOSED events.")
    if thresholds.require_recon_watchdog_events and not recon_events:
        telemetry_gaps.append("No [RECON-WATCHDOG] events observed in selected logs.")

    return {
        "status": status,
        "metrics": {
            "total_traces": total,
            "total_outcomes": outcome_total,
            "smart_entry_applied_rate": smart_entry_applied_rate,
            "missing_atr_force_market_count": missing_atr_cnt,
            "missing_atr_force_market_rate": missing_atr_rate,
            "env_forced_order_type_market_count": env_forced_market_cnt,
            "market_rate_by_bucket": dict(sorted(market_rate_by_bucket.items())),
            "fallback_reason_counts": dict(fallback_reason_counts),
            "outcome_reason_counts": dict(outcome_reason_counts),
            "abort_no_fill_timeout_count": abort_no_fill_timeout_cnt,
            "abort_no_fill_timeout_rate": abort_no_fill_timeout_rate,
            "abort_stop_hit_cancel_unconfirmed_count": abort_stop_hit_cancel_unconfirmed_cnt,
            "abort_stop_hit_cancel_unconfirmed_rate": abort_stop_hit_cancel_unconfirmed_rate,
            "filled_during_stop_abort_count": filled_during_stop_abort_cnt,
            "filled_during_stop_abort_rate": filled_during_stop_abort_rate,
            "entry_slippage_trade_weighted_p90_bps": trade_slippage_p90,
            "entry_slippage_trade_weighted_p95_bps": trade_slippage_p95,
            "entry_slippage_notional_weighted_p90_bps": notional_slippage_p90,
            "entry_slippage_notional_weighted_p95_bps": notional_slippage_p95,
            "time_to_fill_ms_p50": time_to_fill_p50_ms,
            "time_to_fill_ms_p90": time_to_fill_p90_ms,
            "time_to_fill_ms_p95": time_to_fill_p95_ms,
            "stop_overshoot_p90_bps": stop_overshoot_p90,
            "stop_overshoot_p95_bps": stop_overshoot_p95,
            "planned_vs_realized_rr_drift_abs_p90": rr_drift_abs_p90,
            "planned_vs_realized_rr_drift_abs_p95": rr_drift_abs_p95,
            "recon_watchdog_events_count": len(recon_events),
            "recon_cycles_count": len(recon_cycles),
            "recon_stale_removed_total": recon_stale_removed_total,
            "recon_orphans_detected_total": recon_orphans_detected_total,
            "recon_orphans_adopted_total": recon_orphans_adopted_total,
            "atr_stale_market_violations": atr_stale_market_violations,
            "atr_age_threshold_ms": thresholds.atr_age_threshold_ms,
            "missed_fill_increase_pct_vs_baseline": missed_fill_increase_pct,
        },
        "go_no_go": go_no_go,
        "failed_gates": hard_failed,
        "telemetry_gaps": telemetry_gaps,
    }


def _print_summary(report: Dict[str, Any], files: List[Path], symbol: Optional[str], strategy: Optional[str]) -> None:
    print("=== Canary Go/No-Go Report ===")
    print(f"files={len(files)} symbol={symbol or '*'} strategy={strategy or '*'}")
    print(f"status={report.get('status')}")

    if report.get("status") == "NO_RESULT":
        print(f"reason={report.get('reason')}")
        return

    m = report.get("metrics", {})
    gates = report.get("go_no_go", {})
    print(
        "smart_entry_applied_rate={:.2%} missing_atr_force_market_rate={:.2%} extreme_market_rate={:.2%}".format(
            float(m.get("smart_entry_applied_rate") or 0.0),
            float(m.get("missing_atr_force_market_rate") or 0.0),
            float((m.get("market_rate_by_bucket") or {}).get("EXTREME") or 0.0),
        )
    )
    print(
        "env_forced_market_count={} abort_no_fill_timeout_rate={:.2%} stop_abort_cancel_unconfirmed_rate={:.2%} filled_during_stop_abort_rate={:.2%} atr_stale_market_violations={}".format(
            int(m.get("env_forced_order_type_market_count") or 0),
            float(m.get("abort_no_fill_timeout_rate") or 0.0),
            float(m.get("abort_stop_hit_cancel_unconfirmed_rate") or 0.0),
            float(m.get("filled_during_stop_abort_rate") or 0.0),
            int(m.get("atr_stale_market_violations") or 0),
        )
    )
    if m.get("entry_slippage_trade_weighted_p90_bps") is not None:
        print(
            "entry_slippage_bps: trade_p90={:.2f} trade_p95={:.2f} notional_p90={:.2f} notional_p95={:.2f}".format(
                float(m.get("entry_slippage_trade_weighted_p90_bps") or 0.0),
                float(m.get("entry_slippage_trade_weighted_p95_bps") or 0.0),
                float(m.get("entry_slippage_notional_weighted_p90_bps") or 0.0),
                float(m.get("entry_slippage_notional_weighted_p95_bps") or 0.0),
            )
        )
    if m.get("stop_overshoot_p90_bps") is not None:
        print(
            "stop_overshoot_bps: p90={:.2f} p95={:.2f}".format(
                float(m.get("stop_overshoot_p90_bps") or 0.0),
                float(m.get("stop_overshoot_p95_bps") or 0.0),
            )
        )
    if m.get("time_to_fill_ms_p90") is not None:
        print(
            "time_to_fill_ms: p50={:.0f} p90={:.0f} p95={:.0f}".format(
                float(m.get("time_to_fill_ms_p50") or 0.0),
                float(m.get("time_to_fill_ms_p90") or 0.0),
                float(m.get("time_to_fill_ms_p95") or 0.0),
            )
        )
    if m.get("planned_vs_realized_rr_drift_abs_p90") is not None:
        print(
            "planned_vs_realized_rr_drift_abs: p90={:.3f} p95={:.3f}".format(
                float(m.get("planned_vs_realized_rr_drift_abs_p90") or 0.0),
                float(m.get("planned_vs_realized_rr_drift_abs_p95") or 0.0),
            )
        )
    if m.get("recon_watchdog_events_count") is not None:
        print(
            "recon_watchdog: events={} cycles={} stale_removed_total={} orphans_detected_total={} orphans_adopted_total={}".format(
                int(m.get("recon_watchdog_events_count") or 0),
                int(m.get("recon_cycles_count") or 0),
                int(m.get("recon_stale_removed_total") or 0),
                int(m.get("recon_orphans_detected_total") or 0),
                int(m.get("recon_orphans_adopted_total") or 0),
            )
        )
    print(f"failed_gates={report.get('failed_gates') or []}")
    print("gate_status:")
    for k, v in sorted(gates.items()):
        print(f"  - {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build operational Go/No-Go report from execution telemetry logs.")
    parser.add_argument("--log-file", action="append", default=[], help="Specific log file path. Repeatable.")
    parser.add_argument(
        "--log-glob",
        default="logs/live_trading_*.log",
        help="Glob pattern for log files (default: logs/live_trading_*.log)",
    )
    parser.add_argument("--symbol", default=None, help='Filter by symbol (e.g. "BTC/USDT:USDT").')
    parser.add_argument("--strategy", default=None, help='Filter by strategy (e.g. "adaptive_ob").')
    parser.add_argument("--atr-age-threshold-ms", type=int, default=5000, help="Stale ATR threshold for freshness gate.")
    parser.add_argument("--smart-entry-applied-min-rate", type=float, default=0.50, help="Minimum smart_entry_applied_rate.")
    parser.add_argument(
        "--missing-atr-force-market-max-rate",
        type=float,
        default=0.001,
        help="Maximum allowed missing_atr_force_market_rate.",
    )
    parser.add_argument(
        "--extreme-market-max-rate",
        type=float,
        default=0.01,
        help="Maximum allowed market rate for EXTREME bucket.",
    )
    parser.add_argument(
        "--missed-fill-increase-max-pct",
        type=float,
        default=20.0,
        help="Maximum allowed increase of ABORT:NO_FILL_TIMEOUT rate vs baseline.",
    )
    parser.add_argument(
        "--max-stop-abort-cancel-unconfirmed",
        type=int,
        default=None,
        help="Optional max allowed count of ABORT:STOP_HIT_BEFORE_ENTRY_CANCEL_UNCONFIRMED outcomes.",
    )
    parser.add_argument("--baseline-json", default=None, help="Optional previous report json for delta checks.")
    parser.add_argument("--output-json", default=None, help="Optional path to write report json.")
    parser.add_argument(
        "--max-recon-orphans-detected",
        type=int,
        default=None,
        help="Optional max allowed total orphan detections from [RECON-WATCHDOG].",
    )
    parser.add_argument(
        "--max-recon-stale-removed",
        type=int,
        default=None,
        help="Optional max allowed total stale local removals from [RECON-WATCHDOG].",
    )
    parser.add_argument(
        "--max-recon-orphans-adopted",
        type=int,
        default=None,
        help="Optional max allowed total orphan adoptions from [RECON-WATCHDOG].",
    )
    parser.add_argument(
        "--require-recon-watchdog-events",
        action="store_true",
        help="Fail gate when no [RECON-WATCHDOG] events are present in logs.",
    )
    args = parser.parse_args()

    files = _iter_log_files(args.log_file, args.log_glob)
    if not files:
        print("No log files matched.")
        return 2

    baseline = None
    if args.baseline_json:
        bpath = Path(args.baseline_json)
        if bpath.exists():
            baseline = json.loads(bpath.read_text(encoding="utf-8"))

    traces, outcomes, mgr, trades, recon = _load_events(files)
    traces, outcomes, mgr, trades = _apply_filters(
        traces=traces,
        outcomes=outcomes,
        mgr=mgr,
        trades=trades,
        symbol=args.symbol,
        strategy=args.strategy,
    )
    thresholds = Thresholds(
        smart_entry_applied_min_rate=float(args.smart_entry_applied_min_rate),
        missing_atr_force_market_max_rate=float(args.missing_atr_force_market_max_rate),
        extreme_market_max_rate=float(args.extreme_market_max_rate),
        atr_age_threshold_ms=int(args.atr_age_threshold_ms),
        missed_fill_increase_max_pct=float(args.missed_fill_increase_max_pct),
        max_stop_abort_cancel_unconfirmed=(
            int(args.max_stop_abort_cancel_unconfirmed) if args.max_stop_abort_cancel_unconfirmed is not None else None
        ),
        max_recon_orphans_detected=(
            int(args.max_recon_orphans_detected) if args.max_recon_orphans_detected is not None else None
        ),
        max_recon_stale_removed=(
            int(args.max_recon_stale_removed) if args.max_recon_stale_removed is not None else None
        ),
        max_recon_orphans_adopted=(
            int(args.max_recon_orphans_adopted) if args.max_recon_orphans_adopted is not None else None
        ),
        require_recon_watchdog_events=bool(args.require_recon_watchdog_events),
    )

    report = _compute_metrics(
        traces=traces,
        outcomes=outcomes,
        trades=trades,
        recon_events=recon,
        thresholds=thresholds,
        baseline=baseline,
    )
    report["scope"] = {
        "files": [str(p) for p in files],
        "symbol": args.symbol,
        "strategy": args.strategy,
    }
    report["event_counts"] = {
        "traces": len(traces),
        "outcomes": len(outcomes),
        "order_manager_decisions": len(mgr),
        "trade_closed": len(trades),
        "reconciliation_watchdog": len(recon),
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
