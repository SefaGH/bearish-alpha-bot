#!/usr/bin/env python3
"""Offline analysis: compute adaptive_ob persistency bounce-at-entry from a live trading log.

Why:
- Persistency time-mode previously only checked dwell+samples.
- We introduced an optional min-bounce gate; to tune it without missing good entries,
  we want to measure actual bounce sizes at the decision timestamps.

What it does:
- Parses a log file.
- Tracks last seen trigger_price from adaptive_ob "Hybrid meta" lines.
- Detects persistency sample=1 (state start) and the next persistency passed=True.
- Computes bounce_from_min = (trigger_at_pass - min_trigger_seen_between_start_and_pass) / min_trigger.

Notes:
- This is an approximation because logs are discrete; it uses the last observed trigger price
  before each persistency line.
- It focuses on a single symbol/strategy in the provided log.

Usage:
  python tools/analyze_ob_persistency_bounce_from_log.py --log logs/live_trading_*.log --symbol BTC/USDT:USDT
  python tools/analyze_ob_persistency_bounce_from_log.py --log ... --symbol BTC/USDT:USDT --start "2026-01-31 14:20:00" --end "2026-01-31 14:40:00"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TS_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - \[(?P<logger>[^\]]+)\] - (?P<level>[A-Z]+) - (?P<msg>.*)$"
)

HYBRID_META_RE = re.compile(
    r"\[ADAPTIVE_OB/(?P<symbol>[^\]]+)\] Hybrid meta .*?forming_open_time=(?P<bucket>\d+) .*?trigger_price=(?P<trigger>-?\d+(?:\.\d+)?)"
)

PERSIST_RE = re.compile(
    r"\[ADAPTIVE_OB/(?P<symbol>[^\]]+)\] Persistency \| mode=(?P<mode>\w+) condition_true=(?P<cond>True|False) elapsed_s=(?P<elapsed>-?\d+(?:\.\d+)?) samples=(?P<samples>\d+) passed=(?P<passed>True|False)"
)

SIGNAL_RE = re.compile(
    r"\[ADAPTIVE_OB/(?P<symbol>[^\]]+)\] All checks passed\. Generating BUY signal\."
)

DOWNTREND_VETO_RE = re.compile(
    r"(?P<symbol>\S+) Downtrend Veto: lowering RSI threshold .*?\(Price < EMA50 & ADX (?P<adx>-?\d+(?:\.\d+)?) > 30\)"
)


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _iter_log(path: Path) -> Iterable[Tuple[datetime, str, str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = TS_LINE_RE.match(line)
            if not m:
                continue
            yield _parse_ts(m.group("ts")), m.group("logger"), m.group("level"), m.group("msg")


@dataclass
class Attempt:
    symbol: str
    start_ts: datetime
    start_bucket: Optional[int]
    pass_ts: Optional[datetime] = None
    pass_bucket: Optional[int] = None
    trigger_at_pass: Optional[float] = None
    min_trigger: Optional[float] = None
    max_trigger: Optional[float] = None
    bounce_from_min: Optional[float] = None
    generated_signal: bool = False
    downtrend_adx: Optional[float] = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to live_trading log")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. BTC/USDT:USDT")
    ap.add_argument("--start", default=None, help='UTC start, e.g. "2026-01-31 14:20:00"')
    ap.add_argument("--end", default=None, help='UTC end, e.g. "2026-01-31 14:40:00"')
    args = ap.parse_args()

    path = Path(args.log)
    symbol = args.symbol

    start_dt = _parse_ts(args.start).astimezone(timezone.utc) if args.start else None
    end_dt = _parse_ts(args.end).astimezone(timezone.utc) if args.end else None

    last_trigger: Dict[str, float] = {}
    last_bucket: Dict[str, int] = {}
    last_downtrend_adx: Dict[str, float] = {}

    active_attempt: Optional[Attempt] = None
    completed: List[Attempt] = []

    def _upd_attempt_with_trigger(ts: datetime) -> None:
        nonlocal active_attempt
        if active_attempt is None:
            return
        trg = last_trigger.get(active_attempt.symbol)
        if trg is None:
            return
        if active_attempt.min_trigger is None or trg < active_attempt.min_trigger:
            active_attempt.min_trigger = trg
        if active_attempt.max_trigger is None or trg > active_attempt.max_trigger:
            active_attempt.max_trigger = trg

    for ts, logger, level, msg in _iter_log(path):
        if start_dt and ts < start_dt:
            continue
        if end_dt and ts > end_dt:
            break

        m = HYBRID_META_RE.search(msg)
        if m and m.group("symbol") == symbol:
            try:
                last_trigger[symbol] = float(m.group("trigger"))
                last_bucket[symbol] = int(m.group("bucket"))
            except Exception:
                pass
            # update min/max in-flight
            _upd_attempt_with_trigger(ts)
            continue

        m = DOWNTREND_VETO_RE.search(msg)
        if m and m.group("symbol") == symbol:
            try:
                last_downtrend_adx[symbol] = float(m.group("adx"))
            except Exception:
                pass
            continue

        m = PERSIST_RE.search(msg)
        if m and m.group("symbol") == symbol:
            samples = int(m.group("samples"))
            passed = m.group("passed") == "True"

            # Ensure we have trigger sampled for this time slice.
            _upd_attempt_with_trigger(ts)

            if samples == 1 and not passed:
                # start a new attempt
                if active_attempt is not None and active_attempt.pass_ts is None:
                    completed.append(active_attempt)
                active_attempt = Attempt(
                    symbol=symbol,
                    start_ts=ts,
                    start_bucket=last_bucket.get(symbol),
                    downtrend_adx=last_downtrend_adx.get(symbol),
                )
                _upd_attempt_with_trigger(ts)
                continue

            if passed and active_attempt is not None and active_attempt.pass_ts is None:
                active_attempt.pass_ts = ts
                active_attempt.pass_bucket = last_bucket.get(symbol)
                active_attempt.trigger_at_pass = last_trigger.get(symbol)
                active_attempt.downtrend_adx = last_downtrend_adx.get(symbol)
                # finalize bounce
                if (
                    active_attempt.min_trigger
                    and active_attempt.trigger_at_pass
                    and active_attempt.min_trigger > 0
                    and active_attempt.trigger_at_pass > 0
                ):
                    active_attempt.bounce_from_min = (
                        active_attempt.trigger_at_pass - active_attempt.min_trigger
                    ) / active_attempt.min_trigger
                continue

        m = SIGNAL_RE.search(msg)
        if m and m.group("symbol") == symbol:
            if active_attempt is not None and active_attempt.pass_ts is not None:
                active_attempt.generated_signal = True
                completed.append(active_attempt)
                active_attempt = None
            continue

    if active_attempt is not None:
        completed.append(active_attempt)

    # Print summary
    print(f"log={path} symbol={symbol}")
    if start_dt or end_dt:
        print(f"window={start_dt.isoformat() if start_dt else '...'} -> {end_dt.isoformat() if end_dt else '...'}")
    print("attempts:")

    for a in completed:
        status = "SIGNAL" if a.generated_signal else "NO-SIGNAL"
        bounce = f"{a.bounce_from_min:.4%}" if a.bounce_from_min is not None else "n/a"
        minp = f"{a.min_trigger:.2f}" if a.min_trigger is not None else "n/a"
        maxp = f"{a.max_trigger:.2f}" if a.max_trigger is not None else "n/a"
        trgp = f"{a.trigger_at_pass:.2f}" if a.trigger_at_pass is not None else "n/a"
        adx = f"{a.downtrend_adx:.1f}" if a.downtrend_adx is not None else "n/a"
        print(
            f"- start={a.start_ts.isoformat()} bucket={a.start_bucket} pass={a.pass_ts.isoformat() if a.pass_ts else 'n/a'} "
            f"status={status} adx={adx} trigger_at_pass={trgp} min={minp} max={maxp} bounce_from_min={bounce}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
