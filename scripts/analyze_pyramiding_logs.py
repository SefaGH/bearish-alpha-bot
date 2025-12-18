"""
Log analyzer for signal quality & pyramiding calibration.

New capabilities (keeps legacy counters):
- Parse [QUALITY-BEFORE-RISK] to build quality distributions by intent/extreme_bypass.
- Parse [RISK-SCALING] decisions for scale-in outcomes (quality, reason).
- Parse [PYRAMID] / [PYRAMID-QUEUE] for queue/slot decisions.
- Optional filters: symbol/strategy; multi-file support; optional JSON output.

Usage:
    python scripts/analyze_pyramiding_logs.py --files logs/run.log other.log \
        [--symbol BTC/USDT:USDT] [--strategy adaptive_ob] [--json-out out.json]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple


QUALITY_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+)? .*?\[QUALITY-BEFORE-RISK\] strat=(?P<strategy>[^|]+)\s+\|\s+sym=(?P<symbol>[^|]+)\s+\|\s+intent=(?P<intent>[^|]+)\s+\|\s+extreme_bypass=(?P<bypass>[^|]+)\s+\|\s+quality=(?P<quality>[0-9.]+)",
    re.IGNORECASE,
)
RISK_SCALING_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+)? .*?\[RISK-SCALING\]\s+(?P<decision>Allowed|Denied|Rejected)[^|]*?\sfor\s+(?P<symbol>[^|]+)\s*\|\s*(?P<tail>.*)",
    re.IGNORECASE,
)
PYRAMID_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+)? .*?\[PYRAMID\].*", re.IGNORECASE)
PYRAMID_QUEUE_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+)? .*?\[PYRAMID-QUEUE\].*", re.IGNORECASE)
REASON_RE = re.compile(r"reason=([A-Za-z0-9_\-\.]+)", re.IGNORECASE)
QUALITY_IN_TAIL_RE = re.compile(r"quality=([0-9.]+)")
PNL_RE = re.compile(r"(?:avgPnL|pnl)[=_](-?[0-9.]+)", re.IGNORECASE)
DIST_RE = re.compile(r"(?:dist(?:ance)?|diff)[=_]([0-9.]+)", re.IGNORECASE)

COMPARE_RE = re.compile(
    r"(?P<metric>avgPnL|pnl|quality|dist(?:ance)?|dist|diff)\s*=\s*(?P<value>-?[0-9.]+)%?\s*(?P<op><=|<|>=|>)\s*(?P<threshold>-?[0-9.]+)%?",
    re.IGNORECASE,
)
RISK_SCALING_PNL_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+)? .*?\[RISK-SCALING-PNL\]\s+sym=(?P<symbol>[^|]+)\s*\|\s*layers=(?P<layers>\d+)\s*\|\s*pnls=\[(?P<pnls>[^\]]*)\]",
    re.IGNORECASE,
)
PYRAMID_DETAIL_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+)? .*?\[PYRAMID\]\s+(?P<msg>.*)", re.IGNORECASE)
RISK_ENGINE_BLOCK_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+)? .*?\[RISK-ENGINE\].*?:\s*(?P<reason>scale_in_[A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)
STRATEGY_REJECT_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+)? .*?\[(?P<strategy>[A-Z0-9_]+)/(?P<symbol>[^\]]+)\].*REJECTED.*:\s*(?P<reason>scale_in_[A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)



def parse_ts(ts_raw: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_raw)
    except Exception:
        return None


def percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def summarize_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0}
    vals = sorted(values)
    mean = sum(vals) / len(vals)
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "p10": percentile(vals, 0.10),
        "p25": percentile(vals, 0.25),
        "p50": percentile(vals, 0.50),
        "p75": percentile(vals, 0.75),
        "p90": percentile(vals, 0.90),
    }


def line_passes_filters(line: str, symbol_filter: str, strategy_filter: str) -> bool:
    # Note: some lines (e.g., [RISK-ENGINE]) do not contain symbol/strategy text. We still want them for correlation.
    if symbol_filter and symbol_filter not in line:
        if "[RISK-ENGINE]" in line:
            return True
        return False
    if strategy_filter and strategy_filter.lower() not in line.lower():
        # Some lines (risk scaling / pyramiding / PnL updates) do not carry strategy text; keep them for analysis.
        passthrough_tags = ("[RISK-SCALING]", "[RISK-SCALING-PNL]", "[PYRAMID]", "[RISK-ENGINE]", "[P&L-UPDATE]")
        if any(tag in line for tag in passthrough_tags):
            return True
        return False
    return True


def analyze_logs(paths: List[str], symbol_filter: str = None, strategy_filter: str = None) -> Dict[str, Any]:
    quality_entries: List[Dict[str, Any]] = []
    scaling_entries: List[Dict[str, Any]] = []
    scaling_pnl_entries: List[Dict[str, Any]] = []
    pyramid_events: List[Dict[str, Any]] = []
    risk_engine_blocks: List[Dict[str, Any]] = []
    strategy_rejects: List[Dict[str, Any]] = []
    last_scaling_event: Dict[str, Any] = None

    pyramid_queue = Counter()
    pyramid_reasons = Counter()
    queue_reasons = Counter()
    duplicate_spam = 0

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line_passes_filters(line, symbol_filter, strategy_filter):
                        continue

                    m_q = QUALITY_RE.search(line)
                    if m_q:
                        entry = {
                            "ts": m_q.group("ts"),
                            "strategy": m_q.group("strategy").strip(),
                            "symbol": m_q.group("symbol").strip(),
                            "intent": m_q.group("intent").strip(),
                            "extreme_bypass": m_q.group("bypass").strip().lower() == "true",
                            "quality": float(m_q.group("quality")),
                        }
                        quality_entries.append(entry)
                        continue

                    m_rsp = RISK_SCALING_PNL_RE.search(line)
                    if m_rsp:
                        pnls_raw = m_rsp.group("pnls").strip()
                        pnls = []
                        if pnls_raw:
                            for tok in pnls_raw.split(","):
                                tok = tok.strip()
                                if tok:
                                    try:
                                        pnls.append(float(tok))
                                    except Exception:
                                        pass
                        avg_pnl = (sum(pnls) / len(pnls)) if pnls else None
                        scaling_pnl_entries.append(
                            {
                                "ts": m_rsp.group("ts"),
                                "symbol": m_rsp.group("symbol").strip(),
                                "layers": int(m_rsp.group("layers")),
                                "pnls": pnls,
                                "avg_pnl": avg_pnl,
                                "raw": line,
                            }
                        )
                        continue

                    m_pyr = PYRAMID_DETAIL_RE.search(line)
                    if m_pyr:
                        msg = m_pyr.group("msg").strip()
                        if "scale-in" in msg.lower():
                            parts = [p.strip() for p in msg.split("|")]
                            head = parts[0].lower() if parts else ""
                            decision = None
                            if "allowed" in head:
                                decision = "allowed"
                            elif "denied" in head:
                                decision = "denied"
                            kv = {}
                            for p in parts[1:]:
                                if "=" in p:
                                    k, v = p.split("=", 1)
                                    kv[k.strip()] = v.strip()
                            pyramid_events.append(
                                {
                                    "ts": m_pyr.group("ts"),
                                    "decision": decision,
                                    "symbol": (kv.get("sym") or kv.get("symbol")),
                                    "slots": kv.get("slots"),
                                    "quality": kv.get("quality"),
                                    "avgPnL": kv.get("avgPnL"),
                                    "dist": (kv.get("dist") or kv.get("distance") or kv.get("diff")),
                                    "raw": line,
                                }
                            )
                        continue

                    m_block = RISK_ENGINE_BLOCK_RE.search(line)
                    if m_block:
                        reason = m_block.group("reason")
                        risk_engine_blocks.append({"ts": m_block.group("ts"), "symbol": None, "reason": reason, "raw": line})
                        if last_scaling_event is not None and not last_scaling_event.get("reason"):
                            last_scaling_event["reason"] = reason
                        continue

                    m_srej = STRATEGY_REJECT_RE.search(line)
                    if m_srej:
                        strategy_rejects.append(
                            {
                                "ts": m_srej.group("ts"),
                                "strategy": m_srej.group("strategy").lower(),
                                "symbol": m_srej.group("symbol").strip(),
                                "reason": m_srej.group("reason"),
                                "raw": line,
                            }
                        )
                        continue

                    m_rs = RISK_SCALING_RE.search(line)
                    if m_rs:
                        tail = m_rs.group("tail")
                        quality_match = QUALITY_IN_TAIL_RE.search(tail)
                        pnl_match = PNL_RE.search(tail)
                        dist_match = DIST_RE.search(tail)
                        reason_match = REASON_RE.search(tail)
                        cmp = COMPARE_RE.search(tail)
                        metric = cmp.group("metric").lower() if cmp else None
                        value = float(cmp.group("value")) if cmp else None
                        threshold = float(cmp.group("threshold")) if cmp else None

                        # Prefer explicit reason=... if present; otherwise infer from metric.
                        reason = reason_match.group(1) if reason_match else None
                        if not reason and metric:
                            if metric == "quality":
                                reason = "scale_in_quality_below_threshold"
                            elif metric in ("avgpnl", "pnl"):
                                reason = "scale_in_pnl_below_threshold"
                            elif metric in ("diff", "dist", "distance"):
                                reason = "scale_in_distance_below_threshold"

                        last_scaling_event = {
                            "ts": m_rs.group("ts"),
                            "symbol": m_rs.group("symbol").strip(),
                            "decision": m_rs.group("decision").lower(),
                            "metric": metric,
                            "value": value,
                            "threshold": threshold,
                            "quality": float(quality_match.group(1)) if quality_match else None,
                            "pnl": float(pnl_match.group(1)) if pnl_match else None,
                            "distance": float(dist_match.group(1)) if dist_match else None,
                            "reason": reason,
                            "raw": line,
                        }
                        scaling_entries.append(last_scaling_event)
                        continue

                    if PYRAMID_RE.search(line):
                        pyramid_reasons[REASON_RE.search(line).group(1) if REASON_RE.search(line) else "unknown"] += 1
                        continue

                    if PYRAMID_QUEUE_RE.search(line):
                        queue_reasons[REASON_RE.search(line).group(1) if REASON_RE.search(line) else "unknown"] += 1
                        continue

                    if "duplicate_scale_in_spam_window" in line.lower():
                        duplicate_spam += 1

        except FileNotFoundError:
            print(f"Warning: file not found: {path}", file=sys.stderr)
            continue

    # Quality distribution by intent/extreme flag
    dist_by_group: Dict[Tuple[str, bool], List[float]] = defaultdict(list)
    for q in quality_entries:
        key = (q["intent"], q["extreme_bypass"])
        dist_by_group[key].append(q["quality"])

    quality_summary = {f"{k[0]}|extreme={k[1]}": summarize_distribution(v) for k, v in dist_by_group.items()}

    # Scale-in outcomes
    scale_in_entries = list(scaling_entries)
    quality_rejects = [s for s in scale_in_entries if s.get("reason") and "quality" in s["reason"]]
    quality_reject_dist = summarize_distribution(
        [s["quality"] for s in quality_rejects if s.get("quality") is not None]
    )

    # Queue stats (legacy compat)
    pyramid_lines = sum(pyramid_reasons.values())
    queue_lines = sum(queue_reasons.values())

    return {
        "quality_entries": quality_entries,
        "quality_summary": quality_summary,
        "scaling_entries": scaling_entries,
        "scale_in_quality_reject_summary": quality_reject_dist,
        "pyramid_lines": pyramid_lines,
        "queue_lines": queue_lines,
        "pyramid_reasons": pyramid_reasons,
        "queue_reasons": queue_reasons,
        "duplicate_spam": duplicate_spam,
        "scaling_pnl_entries": scaling_pnl_entries,
        "pyramid_events": pyramid_events,
        "risk_engine_blocks": risk_engine_blocks,
        "strategy_rejects": strategy_rejects,

    }


def print_summary(data: Dict[str, Any]) -> None:
    print("=== Quality Distribution by Intent & Bypass ===")
    if not data["quality_summary"]:
        print("No [QUALITY-BEFORE-RISK] entries found for given filters.")
    for group, stats in data["quality_summary"].items():
        print(
            f"{group}: n={stats['count']}, min={stats['min']:.3f}, p25={stats['p25']:.3f}, "
            f"p50={stats['p50']:.3f}, p75={stats['p75']:.3f}, p90={stats['p90']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}"
        )

    print("\n=== Scale-in Outcomes (RISK-SCALING) ===")
    scaling = data["scaling_entries"]
    if not scaling:
        print("No [RISK-SCALING] entries found.")
    else:
        total = len(scaling)
        decisions = Counter([s["decision"] for s in scaling if s.get("decision")])
        reasons = Counter([s["reason"] for s in scaling if s.get("reason")])
        print(f"Total scale-in related lines: {total}")
        if decisions:
            print("Decisions:")
            for k, v in decisions.most_common():
                print(f"  {k}: {v}")
        if reasons:
            print("Reasons:")
            for k, v in reasons.most_common():
                print(f"  {k}: {v}")

        if data["scale_in_quality_reject_summary"]["count"] > 0:
            s = data["scale_in_quality_reject_summary"]
            print(
                f"Quality rejects: n={s['count']} mean={s['mean']:.3f} p75={s['p75']:.3f} max={s['max']:.3f}"
            )

    print("\n=== Pyramiding Queue ===")
    print(f"[PYRAMID] lines: {data['pyramid_lines']}")
    if data["pyramid_reasons"]:
        print("  Reasons:")
        for reason, count in data["pyramid_reasons"].most_common():
            print(f"    {reason}: {count}")
    print(f"[PYRAMID-QUEUE] lines: {data['queue_lines']}")
    if data["queue_reasons"]:
        print("  Queue reasons:")
        for reason, count in data["queue_reasons"].most_common():
            print(f"    {reason}: {count}")
    print(f"Duplicate spam rejections (scale_in): {data['duplicate_spam']}")

    # Calibration hints
    print("\n=== Calibration Hints ===")
    for group, stats in data["quality_summary"].items():
        if stats["count"] == 0:
            continue
        p50 = stats["p50"]
        p75 = stats["p75"]
        print(
            f"{group}: median={p50:.3f}, p75={p75:.3f} -> "
            f"Conservative min_scale_in_quality ~{p75:.2f}, moderate ~{p50:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze quality and pyramiding logs.")
    parser.add_argument("--files", nargs="+", required=True, help="Log file paths to analyze")
    parser.add_argument("--symbol", help="Filter by symbol substring (e.g., BTC/USDT:USDT)")
    parser.add_argument("--strategy", help="Filter by strategy substring (e.g., adaptive_ob)")
    parser.add_argument("--json-out", help="Optional path to write JSON output")
    args = parser.parse_args()

    data = analyze_logs(args.files, symbol_filter=args.symbol, strategy_filter=args.strategy)
    print_summary(data)

    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"\nJSON output written to {args.json_out}")
        except Exception as exc:
            print(f"Failed to write JSON output: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()