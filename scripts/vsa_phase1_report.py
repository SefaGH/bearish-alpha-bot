"""Generate Phase-1 VSA monitoring report from runtime logs.

Parses:
- SIGNAL_BREAKDOWN {...}
- TRADE_CLOSED {...}

Produces KPI summary aligned with docs/vsa_phase1_monitoring_checklist.md:
- short_trade_count
- short_stopout_rate
- short_net_pnl_per_trade
- short_win_rate
- expectancy_per_trade
- class_distribution (BA/GO/FR)
- ba_short_attempt_rate
- edge_calibration (E buckets)
- telemetry quality checks for vsa_shadow payload
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SIGNAL_BREAKDOWN_MARKER = "SIGNAL_BREAKDOWN "
TRADE_CLOSED_MARKER = "TRADE_CLOSED "


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _parse_iso_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_after_marker(line: str, marker: str) -> Optional[str]:
    idx = line.find(marker)
    if idx < 0:
        return None
    return line[idx + len(marker) :].strip()


def _iter_log_files(log_files: Iterable[str], log_glob: Optional[str]) -> List[Path]:
    out: List[Path] = []
    for item in log_files:
        p = Path(item)
        if p.exists() and p.is_file():
            out.append(p)
    if log_glob:
        out.extend(sorted(Path().glob(log_glob)))

    uniq: List[Path] = []
    seen = set()
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _side_is_short(side: Any) -> bool:
    s = str(side or "").strip().lower()
    return s in {"short", "sell"}


def _is_stop_exit(exit_reason: Any) -> bool:
    reason = str(exit_reason or "").strip().lower()
    if not reason:
        return False
    return ("stop" in reason) or (reason.startswith("sl")) or ("stop_loss" in reason)


def _bucket_edge(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    v = max(0.0, min(1.0, v))
    boundaries = [0.2, 0.4, 0.6, 0.8, 1.000001]
    lower = 0.0
    for upper in boundaries:
        if v < upper:
            return f"{lower:.1f}-{min(upper, 1.0):.1f}"
        lower = upper
    return "0.8-1.0"


def _load_events(
    paths: List[Path],
    *,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    signals: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if SIGNAL_BREAKDOWN_MARKER in line:
                    raw = _extract_after_marker(line, SIGNAL_BREAKDOWN_MARKER)
                    if raw:
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            payload = None
                        if isinstance(payload, dict) and str(payload.get("event", "")) == "signal_breakdown":
                            ts = _parse_iso_ts(payload.get("timestamp"))
                            if from_ts and ts and ts < from_ts:
                                continue
                            if to_ts and ts and ts > to_ts:
                                continue
                            payload["_source_file"] = str(path)
                            signals.append(payload)

                if TRADE_CLOSED_MARKER in line:
                    raw = _extract_after_marker(line, TRADE_CLOSED_MARKER)
                    if raw:
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            payload = None
                        if isinstance(payload, dict) and str(payload.get("event", "")).upper() == "TRADE_CLOSED":
                            ts = _parse_iso_ts(payload.get("timestamp"))
                            if from_ts and ts and ts < from_ts:
                                continue
                            if to_ts and ts and ts > to_ts:
                                continue
                            payload["_source_file"] = str(path)
                            trades.append(payload)
    return signals, trades


def _apply_filters(
    signals: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    *,
    symbol: Optional[str],
    strategy: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sym = str(symbol or "").strip()
    strat = str(strategy or "").strip().lower()

    def ok_symbol(value: Any) -> bool:
        if not sym:
            return True
        return str(value or "").strip() == sym

    def ok_strategy(value: Any) -> bool:
        if not strat:
            return True
        return str(value or "").strip().lower() == strat

    filtered_signals = [
        x for x in signals
        if ok_symbol(x.get("symbol")) and ok_strategy(x.get("strategy"))
    ]
    filtered_trades = [
        x for x in trades
        if ok_symbol(x.get("symbol")) and ok_strategy(x.get("strategy_name") or x.get("strategy"))
    ]
    return filtered_signals, filtered_trades


def _compute_metrics(signals: List[Dict[str, Any]], trades: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    short_trades = [t for t in trades if _side_is_short(t.get("side"))]
    short_trade_count = len(short_trades)
    short_pnls: List[float] = []
    short_wins = 0
    short_stopouts = 0
    for tr in short_trades:
        pnl = _safe_float(tr.get("pnl_usd"))
        if pnl is None:
            pnl = _safe_float(tr.get("realized_pnl_usd"))
        if pnl is not None and math.isfinite(pnl):
            short_pnls.append(float(pnl))
            if pnl > 0:
                short_wins += 1
        if _is_stop_exit(tr.get("exit_reason")):
            short_stopouts += 1

    short_net_pnl_per_trade = (sum(short_pnls) / len(short_pnls)) if short_pnls else None
    short_win_rate = (short_wins / short_trade_count) if short_trade_count > 0 else None
    short_stopout_rate = (short_stopouts / short_trade_count) if short_trade_count > 0 else None
    expectancy_per_trade = short_net_pnl_per_trade

    # vsa_shadow telemetry from signal_breakdown
    total_signal_breakdown = len(signals)
    vsa_payloads = [s.get("vsa_shadow") for s in signals if isinstance(s.get("vsa_shadow"), dict)]
    vsa_shadow_present = len(vsa_payloads)
    vsa_shadow_missing_rate = (
        1.0 - (vsa_shadow_present / total_signal_breakdown)
        if total_signal_breakdown > 0 else None
    )

    class_counter: Counter = Counter()
    ba_total = 0
    ba_short_attempts = 0
    prob_sum_out_of_bounds = 0
    score_out_of_range = 0

    signal_index_by_id: Dict[str, Dict[str, Any]] = {}
    for s in signals:
        sid = _safe_str(s.get("signal_id")).strip()
        if sid:
            signal_index_by_id[sid] = s

    for s in signals:
        vsa = s.get("vsa_shadow")
        if not isinstance(vsa, dict):
            continue
        cls = _safe_str(vsa.get("selected_class")).strip().upper()
        if cls in {"BA", "GO", "FR"}:
            class_counter[cls] += 1
            if cls == "BA":
                ba_total += 1
                if _side_is_short(s.get("side")):
                    ba_short_attempts += 1

        probs = vsa.get("probabilities")
        if isinstance(probs, dict):
            p_ba = _safe_float(probs.get("BA")) or 0.0
            p_go = _safe_float(probs.get("GO")) or 0.0
            p_fr = _safe_float(probs.get("FR")) or 0.0
            p_sum = float(p_ba + p_go + p_fr)
            if not (0.99 <= p_sum <= 1.01):
                prob_sum_out_of_bounds += 1

        scores = vsa.get("scores")
        if isinstance(scores, dict):
            ok = True
            for key in ("I", "T", "A", "R", "z_norm"):
                val = _safe_float(scores.get(key))
                if val is None or not (0.0 <= val <= 1.0):
                    ok = False
                    break
            if not ok:
                score_out_of_range += 1

    class_total = sum(class_counter.values())
    class_distribution = {
        k: {
            "count": int(class_counter.get(k, 0)),
            "rate": (class_counter.get(k, 0) / class_total) if class_total > 0 else None,
        }
        for k in ("BA", "GO", "FR")
    }
    ba_short_attempt_rate = (ba_short_attempts / ba_total) if ba_total > 0 else None

    # Edge calibration: join signal vsa_shadow.edge.E with TRADE_CLOSED by signal_id.
    edge_bucket_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "trades": 0,
        "wins": 0,
        "stopouts": 0,
        "pnl_sum": 0.0,
        "rr_achieved_values": [],
    })
    joined_trade_count = 0
    for tr in short_trades:
        sid = _safe_str(tr.get("signal_id")).strip()
        if not sid:
            continue
        s = signal_index_by_id.get(sid)
        if not s:
            continue
        vsa = s.get("vsa_shadow")
        if not isinstance(vsa, dict):
            continue
        edge = vsa.get("edge")
        if not isinstance(edge, dict):
            continue
        e_val = _safe_float(edge.get("E"))
        bucket = _bucket_edge(e_val)
        if bucket is None:
            continue
        joined_trade_count += 1
        rec = edge_bucket_stats[bucket]
        rec["trades"] += 1
        pnl = _safe_float(tr.get("pnl_usd"))
        if pnl is None:
            pnl = _safe_float(tr.get("realized_pnl_usd"))
        if pnl is not None and math.isfinite(pnl):
            rec["pnl_sum"] += float(pnl)
            if pnl > 0:
                rec["wins"] += 1
        if _is_stop_exit(tr.get("exit_reason")):
            rec["stopouts"] += 1
        rr = _safe_float(tr.get("rr_achieved"))
        if rr is not None and math.isfinite(rr):
            rec["rr_achieved_values"].append(float(rr))

    edge_calibration: Dict[str, Any] = {}
    for bucket in sorted(edge_bucket_stats.keys()):
        rec = edge_bucket_stats[bucket]
        trades_n = int(rec["trades"])
        edge_calibration[bucket] = {
            "trades": trades_n,
            "win_rate": (rec["wins"] / trades_n) if trades_n > 0 else None,
            "stopout_rate": (rec["stopouts"] / trades_n) if trades_n > 0 else None,
            "avg_pnl_usd": (rec["pnl_sum"] / trades_n) if trades_n > 0 else None,
            "avg_rr_achieved": (
                (sum(rec["rr_achieved_values"]) / len(rec["rr_achieved_values"]))
                if rec["rr_achieved_values"] else None
            ),
        }

    # Baseline deltas
    baseline_metrics = baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
    baseline_short_trade_count = _safe_float(baseline_metrics.get("short_trade_count"))
    no_trade_drift = None
    if baseline_short_trade_count is not None and baseline_short_trade_count > 0:
        no_trade_drift = (float(short_trade_count) - float(baseline_short_trade_count)) / float(baseline_short_trade_count)

    telemetry_gaps: List[str] = []
    if total_signal_breakdown == 0:
        telemetry_gaps.append("No SIGNAL_BREAKDOWN events in selected scope.")
    if len(trades) == 0:
        telemetry_gaps.append("No TRADE_CLOSED events in selected scope.")
    if total_signal_breakdown > 0 and vsa_shadow_present == 0:
        telemetry_gaps.append("SIGNAL_BREAKDOWN exists but vsa_shadow payload missing.")
    if joined_trade_count == 0 and len(short_trades) > 0:
        telemetry_gaps.append("No signal_id join between vsa_shadow and short TRADE_CLOSED events.")

    return {
        "metrics": {
            "short_trade_count": short_trade_count,
            "short_stopout_rate": short_stopout_rate,
            "short_net_pnl_per_trade": short_net_pnl_per_trade,
            "short_win_rate": short_win_rate,
            "expectancy_per_trade": expectancy_per_trade,
            "ba_short_attempt_rate": ba_short_attempt_rate,
            "no_trade_drift": no_trade_drift,
        },
        "shadow": {
            "class_distribution": class_distribution,
            "edge_calibration": edge_calibration,
            "joined_trade_count": joined_trade_count,
        },
        "telemetry_quality": {
            "signal_breakdown_events": total_signal_breakdown,
            "vsa_shadow_present_events": vsa_shadow_present,
            "vsa_shadow_missing_rate": vsa_shadow_missing_rate,
            "prob_sum_out_of_bounds_count": prob_sum_out_of_bounds,
            "score_out_of_range_count": score_out_of_range,
        },
        "counts": {
            "trade_closed_events": len(trades),
            "short_trade_closed_events": short_trade_count,
            "ba_total_signals": ba_total,
            "ba_short_attempts": ba_short_attempts,
        },
        "telemetry_gaps": telemetry_gaps,
    }


def _render_markdown(report: Dict[str, Any], *, symbol: Optional[str], strategy: Optional[str]) -> str:
    m = report.get("metrics", {})
    tq = report.get("telemetry_quality", {})
    sh = report.get("shadow", {})
    lines: List[str] = []
    lines.append("# VSA Phase-1 Report")
    lines.append("")
    lines.append(f"- symbol: `{symbol or 'ALL'}`")
    lines.append(f"- strategy: `{strategy or 'ALL'}`")
    lines.append("")
    lines.append("## Core KPI")
    lines.append(f"- short_trade_count: `{m.get('short_trade_count')}`")
    lines.append(f"- short_stopout_rate: `{m.get('short_stopout_rate')}`")
    lines.append(f"- short_net_pnl_per_trade: `{m.get('short_net_pnl_per_trade')}`")
    lines.append(f"- short_win_rate: `{m.get('short_win_rate')}`")
    lines.append(f"- expectancy_per_trade: `{m.get('expectancy_per_trade')}`")
    lines.append(f"- ba_short_attempt_rate: `{m.get('ba_short_attempt_rate')}`")
    lines.append(f"- no_trade_drift: `{m.get('no_trade_drift')}`")
    lines.append("")
    lines.append("## Telemetry Quality")
    lines.append(f"- signal_breakdown_events: `{tq.get('signal_breakdown_events')}`")
    lines.append(f"- vsa_shadow_present_events: `{tq.get('vsa_shadow_present_events')}`")
    lines.append(f"- vsa_shadow_missing_rate: `{tq.get('vsa_shadow_missing_rate')}`")
    lines.append(f"- prob_sum_out_of_bounds_count: `{tq.get('prob_sum_out_of_bounds_count')}`")
    lines.append(f"- score_out_of_range_count: `{tq.get('score_out_of_range_count')}`")
    lines.append("")
    lines.append("## Class Distribution")
    class_dist = sh.get("class_distribution", {}) if isinstance(sh.get("class_distribution"), dict) else {}
    for cls in ("BA", "GO", "FR"):
        item = class_dist.get(cls, {})
        lines.append(f"- {cls}: count=`{item.get('count')}` rate=`{item.get('rate')}`")
    lines.append("")
    lines.append("## Edge Calibration")
    edge = sh.get("edge_calibration", {}) if isinstance(sh.get("edge_calibration"), dict) else {}
    if not edge:
        lines.append("- no joined edge/trade sample")
    else:
        for bucket in sorted(edge.keys()):
            item = edge.get(bucket, {})
            lines.append(
                f"- {bucket}: trades=`{item.get('trades')}` win_rate=`{item.get('win_rate')}` "
                f"stopout_rate=`{item.get('stopout_rate')}` avg_pnl_usd=`{item.get('avg_pnl_usd')}` "
                f"avg_rr_achieved=`{item.get('avg_rr_achieved')}`"
            )
    gaps = report.get("telemetry_gaps") or []
    lines.append("")
    lines.append("## Telemetry Gaps")
    if not gaps:
        lines.append("- none")
    else:
        for g in gaps:
            lines.append(f"- {g}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate VSA Phase-1 monitoring report from logs.")
    p.add_argument("--log-file", action="append", default=[], help="Path to a log file (repeatable).")
    p.add_argument("--log-glob", default="logs/*.log", help="Glob pattern for log files.")
    p.add_argument("--symbol", default=None, help="Exact symbol filter (e.g., BTC/USDT:USDT).")
    p.add_argument("--strategy", default="mean_reversion", help="Strategy filter (default: mean_reversion).")
    p.add_argument("--from-utc", default=None, help="Inclusive start timestamp (ISO8601 UTC).")
    p.add_argument("--to-utc", default=None, help="Inclusive end timestamp (ISO8601 UTC).")
    p.add_argument("--baseline-json", default=None, help="Optional previous report JSON for drift comparison.")
    p.add_argument("--output-json", default="artifacts/vsa/phase1_report.json", help="Output JSON path.")
    p.add_argument("--output-md", default="artifacts/vsa/phase1_report.md", help="Output markdown path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from_ts = _parse_iso_ts(args.from_utc) if args.from_utc else None
    to_ts = _parse_iso_ts(args.to_utc) if args.to_utc else None
    if args.from_utc and from_ts is None:
        raise SystemExit(f"Invalid --from-utc: {args.from_utc}")
    if args.to_utc and to_ts is None:
        raise SystemExit(f"Invalid --to-utc: {args.to_utc}")

    paths = _iter_log_files(args.log_file, args.log_glob)
    if not paths:
        raise SystemExit("No log files found. Use --log-file and/or --log-glob.")

    signals, trades = _load_events(paths, from_ts=from_ts, to_ts=to_ts)
    signals, trades = _apply_filters(
        signals,
        trades,
        symbol=args.symbol,
        strategy=args.strategy,
    )

    baseline = None
    if args.baseline_json:
        p = Path(args.baseline_json)
        if p.exists():
            baseline = json.loads(p.read_text(encoding="utf-8"))

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "symbol": args.symbol,
            "strategy": args.strategy,
            "from_utc": args.from_utc,
            "to_utc": args.to_utc,
            "log_files": [str(p) for p in paths],
        },
    }
    report.update(_compute_metrics(signals, trades, baseline))

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        _render_markdown(report, symbol=args.symbol, strategy=args.strategy),
        encoding="utf-8",
    )

    print(f"Wrote JSON: {out_json}")
    print(f"Wrote Markdown: {out_md}")
    print(
        f"short_trade_count={report.get('metrics', {}).get('short_trade_count')} "
        f"short_stopout_rate={report.get('metrics', {}).get('short_stopout_rate')} "
        f"ba_short_attempt_rate={report.get('metrics', {}).get('ba_short_attempt_rate')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

