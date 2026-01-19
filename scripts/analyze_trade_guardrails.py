"""Analyze a live trading log and estimate which trades would be gated by new guardrails.

This script is intentionally conservative: older logs won't contain the new
reversal/downtrend metadata fields, so we report "applies" and "evidence" rather
than claiming definite would-pass/would-fail.

Usage (Windows):
  C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/analyze_trade_guardrails.py \
    --log logs/live_trading_20260118_202844_781510.log \
    --strategy adaptive_ob

Optional:
  --trade-ids 2c6d277d 90dd9316 ...
  --context-lines 400
  --json-out out_guardrails.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bisect import bisect_left


TRADE_IDS_DEFAULT = [
    "2c6d277d",
    "90dd9316",
    "2f878e35",
    "c192b8ec",
    "1df4f67e",
    "d9f37d44",
    "eaac032b",
    "bea733d6",
    "492e1195",
    "d69e0891",
    "89cf2f56",
]


@dataclass
class TradeRecord:
    trade_id: str
    strategy: Optional[str] = None
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    volume_bucket_at_entry: Optional[str] = None

    # Derived from nearby log context
    downtrend_veto_seen: bool = False
    downtrend_adx: Optional[float] = None
    extreme_bypass_seen: bool = False
    volume_override_seen: bool = False

    # Guardrail application
    rule1_low_volume_requires_reversal_applies: bool = False
    rule2_downtrend_low_volume_requires_strong_applies: bool = False
    reversal_evidence: str = "unknown"  # old logs don't have the new fields
    strong_reversal_evidence: str = "unknown"


def _try_parse_json_from_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of JSON object from a log line."""
    brace = line.find("{")
    if brace < 0:
        return None
    candidate = line[brace:].strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _find_line_indexes(lines: List[str], needle: str) -> List[int]:
    out: List[int] = []
    for idx, line in enumerate(lines):
        if needle in line:
            out.append(idx)
    return out


def _find_best_anchor_index(lines: List[str], trade_id: str) -> Optional[int]:
    """Choose an anchor line that is likely near trade ENTRY (not the shutdown summary table)."""
    indexes = _find_line_indexes(lines, trade_id)
    if not indexes:
        return None

    # Prefer explicit lifecycle events.
    preferred_tokens = [
        "TRADE_OPENED",
        "TRADE_OPEN",
        "POSITION_OPENED",
        "ENTRY_FILLED",
        "ENTRY ORDER",
        "OPENED",
        "OPEN POSITION",
    ]

    for idx in indexes:
        line = lines[idx]
        if any(tok in line for tok in preferred_tokens):
            return idx
        payload = _try_parse_json_from_log_line(line)
        if payload:
            event = str(payload.get("event", "")).upper()
            if event in {"TRADE_OPENED", "TRADE_OPEN", "POSITION_OPENED", "ENTRY_FILLED", "TRADE_CREATED"}:
                return idx

    # As a second best, pick the earliest occurrence that isn't in the session summary table.
    for idx in indexes:
        line = lines[idx]
        if "INDIVIDUAL TRADE HISTORY" in line:
            continue
        if "core.position_manager" in line and "INDIVIDUAL TRADE HISTORY" in line:
            continue
        return idx

    return indexes[0]


def _parse_line_ts(line: str) -> Optional[datetime]:
    """Parse the leading log timestamp 'YYYY-MM-DD HH:MM:SS' as UTC (best effort)."""
    if len(line) < 19:
        return None
    head = line[:19]
    try:
        dt = datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_iso_z(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _extract_adx_from_downtrend_veto(line: str) -> Optional[float]:
    # Example: "... ADX 38.3 > 30"
    m = re.search(r"ADX\s+(\d+(?:\.\d+)?)\s*>", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_extreme_bypass_value(line: str) -> Optional[bool]:
    """Return extreme_bypass boolean if present on this line.

    We intentionally only treat this as True when it is explicitly True.
    This avoids false-positives from config dumps or generic mentions.
    """
    m = re.search(r"extreme_bypass\s*=\s*(true|false)", line, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"

    payload = _try_parse_json_from_log_line(line)
    if payload and "extreme_bypass" in payload:
        v = payload.get("extreme_bypass")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "false"}:
                return s == "true"

    # Last-resort: look for a JSON-ish token '"extreme_bypass": true/false'
    m2 = re.search(r'"extreme_bypass"\s*:\s*(true|false)', line, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).lower() == "true"

    return None


def _parse_trade_closed_from_lines(lines: List[str], trade_id: str) -> Optional[TradeRecord]:
    """Locate and parse a TRADE_CLOSED JSON payload for a given trade_id."""
    for line in lines:
        if trade_id not in line:
            continue
        if "TRADE_CLOSED" not in line:
            continue
        payload = _try_parse_json_from_log_line(line)
        if not payload:
            continue

        # Support both "event":"TRADE_CLOSED" and other formats
        if str(payload.get("event", "")).upper() != "TRADE_CLOSED":
            # Still accept if it contains trade_id and looks like a closure payload
            pass

        rec = TradeRecord(trade_id=trade_id)
        rec.strategy = payload.get("strategy") or payload.get("strategy_name")
        rec.entry_time = payload.get("entry_time")
        rec.exit_time = payload.get("exit_time")
        rec.entry_price = _to_float(payload.get("entry_price"))
        rec.exit_price = _to_float(payload.get("exit_price"))
        rec.pnl_usd = _to_float(payload.get("pnl_usd"))
        rec.pnl_pct = _to_float(payload.get("pnl_pct"))
        rec.exit_reason = payload.get("exit_reason")
        rec.volume_bucket_at_entry = payload.get("volume_bucket_at_entry")
        return rec

    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _boolish(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def analyze_log(
    log_path: Path,
    trade_ids: List[str],
    strategy_filter: Optional[str],
    context_lines: int,
    context_seconds: Optional[int],
) -> List[TradeRecord]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Build a searchable timeline of log line timestamps for entry_time anchoring.
    ts_points: List[Tuple[datetime, int]] = []
    for i, line in enumerate(lines):
        dt = _parse_line_ts(line)
        if dt is not None:
            ts_points.append((dt, i))
    ts_points.sort(key=lambda x: x[0])
    ts_times = [t for t, _ in ts_points]
    ts_idxs = [i for _, i in ts_points]

    results: List[TradeRecord] = []

    for trade_id in trade_ids:
        rec = _parse_trade_closed_from_lines(lines, trade_id) or TradeRecord(trade_id=trade_id)

        # Prefer anchoring around entry_time (trade_id is often missing from entry logs).
        anchor: Optional[int] = None
        entry_dt = _parse_iso_z(rec.entry_time)
        if entry_dt is not None and ts_points:
            pos = bisect_left(ts_times, entry_dt)
            if pos >= len(ts_idxs):
                pos = len(ts_idxs) - 1
            anchor = ts_idxs[pos]

        if anchor is None:
            # Fallback: find an anchor line near entry based on trade_id.
            anchor = _find_best_anchor_index(lines, trade_id)
        if anchor is None:
            # Last fallback: try to anchor on TRADE_CLOSED occurrence.
            closed_indexes = [i for i in _find_line_indexes(lines, "TRADE_CLOSED") if trade_id in lines[i]]
            anchor = closed_indexes[0] if closed_indexes else None

        # Build a context window near entry_time to avoid picking up unrelated config dumps.
        if context_seconds and entry_dt is not None:
            start_dt = entry_dt - timedelta(seconds=context_seconds)
            end_dt = entry_dt + timedelta(seconds=max(30, context_seconds // 4))

            # Walk around anchor and collect only timestamped lines within the window.
            window_lines: List[str] = []
            if anchor is None:
                anchor = 0
            i = anchor
            while i >= 0:
                dt = _parse_line_ts(lines[i])
                if dt is not None and dt < start_dt:
                    break
                i -= 1
            lo = max(0, i)
            j = anchor
            while j < len(lines):
                dt = _parse_line_ts(lines[j])
                if dt is not None and dt > end_dt:
                    break
                j += 1
            hi = min(len(lines), j)
            window = lines[lo:hi]
        else:
            lo = max(0, (anchor or 0) - context_lines)
            hi = min(len(lines), (anchor or 0) + context_lines)
            window = lines[lo:hi]

        # Detect context signals
        downtrend_lines = [ln for ln in window if "Downtrend Veto" in ln]
        rec.downtrend_veto_seen = bool(downtrend_lines)
        if downtrend_lines:
            rec.downtrend_adx = _extract_adx_from_downtrend_veto(downtrend_lines[-1])

        rec.volume_override_seen = any("[VOLUME-OVERRIDE]" in ln for ln in window)

        # Only treat extreme_bypass as present when it's explicitly True.
        rec.extreme_bypass_seen = any(_extract_extreme_bypass_value(ln) is True for ln in window)

        # Determine volume bucket at entry (best effort)
        vol_bucket = (rec.volume_bucket_at_entry or "").strip()
        if not vol_bucket:
            # Fallback: parse last volume_decision_check in window
            vol_checks = [ln for ln in window if "volume_decision_check" in ln]
            for ln in reversed(vol_checks):
                payload = _try_parse_json_from_log_line(ln)
                if not payload:
                    continue
                if payload.get("event") != "volume_decision_check":
                    continue
                if str(payload.get("strategy_name")) and strategy_filter and str(payload.get("strategy_name")) != strategy_filter:
                    continue
                vol_bucket = str(payload.get("volume_bucket") or "").strip()
                if vol_bucket:
                    break
        rec.volume_bucket_at_entry = vol_bucket or rec.volume_bucket_at_entry

        # Strategy filter
        if strategy_filter and rec.strategy and rec.strategy != strategy_filter:
            # Keep it but mark strategy mismatch; caller can filter.
            pass

        is_low = (rec.volume_bucket_at_entry or "").strip().lower() in {"low", "very_low"}
        if rec.volume_override_seen:
            is_low = True

        # Rule 1: applies whenever entry volume is LOW/very_low AND override is used.
        # In this session, adaptive_ob is configured with allow_low_volume=true and min_bucket=NORMAL,
        # so LOW implies override path.
        rec.rule1_low_volume_requires_reversal_applies = is_low

        # Rule 2: applies only when LOW AND downtrend context; and bypass if extreme_bypass.
        rec.rule2_downtrend_low_volume_requires_strong_applies = bool(
            is_low and rec.downtrend_veto_seen and not rec.extreme_bypass_seen
        )

        # Evidence fields are unknown for older logs (no meta emitted yet)
        results.append(rec)

    # If requested, filter by strategy here (only after we assembled records)
    if strategy_filter:
        results = [r for r in results if (r.strategy == strategy_filter or r.strategy is None)]

    return results


def _format_table(rows: List[TradeRecord]) -> str:
    headers = [
        "trade_id",
        "pnl_usd",
        "reason",
        "vol",
        "override",
        "downtrend",
        "extreme",
        "rule1(low->rev)",
        "rule2(low+dt->strong)",
    ]

    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, bool):
            return "Y" if v else "N"
        return str(v)

    matrix: List[List[str]] = []
    for r in rows:
        matrix.append(
            [
                r.trade_id,
                fmt(r.pnl_usd),
                fmt(r.exit_reason),
                fmt(r.volume_bucket_at_entry),
                fmt(r.volume_override_seen),
                fmt(r.downtrend_veto_seen),
                fmt(r.extreme_bypass_seen),
                fmt(r.rule1_low_volume_requires_reversal_applies),
                fmt(r.rule2_downtrend_low_volume_requires_strong_applies),
            ]
        )

    widths = [len(h) for h in headers]
    for row in matrix:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    out = [render_row(headers), render_row(["-" * w for w in widths])]
    out.extend(render_row(r) for r in matrix)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate which trades would be gated by new guardrails")
    parser.add_argument("--log", required=True, help="Path to live trading log")
    parser.add_argument("--strategy", default="adaptive_ob", help="Strategy name filter (default: adaptive_ob)")
    parser.add_argument("--trade-ids", nargs="*", default=None, help="Trade IDs to analyze (default: last 11 known)")
    parser.add_argument("--context-lines", type=int, default=500, help="Lines to look around entry anchor")
    parser.add_argument(
        "--context-seconds",
        type=int,
        default=180,
        help="Time window (seconds) around entry_time (default: 180). Set 0 to disable and use --context-lines.",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON output path")

    args = parser.parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    trade_ids = args.trade_ids if args.trade_ids else TRADE_IDS_DEFAULT

    rows = analyze_log(
        log_path=log_path,
        trade_ids=trade_ids,
        strategy_filter=args.strategy if args.strategy else None,
        context_lines=int(args.context_lines),
        context_seconds=(int(args.context_seconds) if int(args.context_seconds) > 0 else None),
    )

    print(_format_table(rows))

    if args.json_out:
        out_path = Path(args.json_out)
        payload = [r.__dict__ for r in rows]
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
