#!/usr/bin/env python3
"""Offline stats: join adaptive_ob persistency bounce to realized trade outcomes.

Goal
----
Select `adaptive_ob_persistency_min_bounce_pct` based on historical distributions,
not intuition.

This script:
- Parses live trading logs.
- Computes bounce-from-intrabucket-min for each adaptive_ob *signal attempt* using
  Hybrid meta trigger prices between persistency start (samples=1) and persistency pass.
- Joins the nearest prior signal attempt to each `TRADE_CLOSED` trade (strategy=adaptive_ob).
- Emits percentiles + a simple histogram of bounce sizes, split by outcome and regime.

Notes
-----
- This is an approximation based on discrete log lines.
- It requires that the log contains `[ADAPTIVE_OB/<symbol>] Hybrid meta ... trigger_price=...`
  and `[ADAPTIVE_OB/<symbol>] Persistency ...` lines.

Usage
-----
  python tools/analyze_adaptive_ob_bounce_stats.py --logs "logs/**/*.log" --symbol "BTC/USDT:USDT"
  python tools/analyze_adaptive_ob_bounce_stats.py --logs "logs/**/*.log" --out tools/_ob_bounce_join.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


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

TRADE_CLOSED_RE = re.compile(r"\bTRADE_CLOSED\s+(?P<json>\{.*\})\s*$")


def _parse_ts_prefix(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _parse_iso_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Most logs: 2026-01-31T14:26:51.909024Z
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _iter_log_lines(path: Path) -> Iterator[Tuple[datetime, str, str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = TS_LINE_RE.match(line)
            if not m:
                continue
            yield _parse_ts_prefix(m.group("ts")), m.group("logger"), m.group("level"), m.group("msg")


@dataclass
class SignalAttempt:
    symbol: str
    signal_ts: datetime
    start_ts: datetime
    pass_ts: Optional[datetime]
    start_bucket: Optional[int]
    pass_bucket: Optional[int]
    trigger_at_pass: Optional[float]
    min_trigger: Optional[float]
    max_trigger: Optional[float]
    bounce_from_min: Optional[float]
    downtrend_adx: Optional[float]


@dataclass
class ClosedTrade:
    path: Path
    trade_id: Optional[str]
    position_id: Optional[str]
    symbol: str
    side: Optional[str]
    strategy: Optional[str]
    entry_time: Optional[datetime]
    exit_time: Optional[datetime]
    entry_price: Optional[float]
    exit_price: Optional[float]
    exit_reason: Optional[str]
    pnl_usd: Optional[float]
    pnl_pct: Optional[float]
    regime_at_entry: Optional[str]
    regime_conf: Optional[float]


@dataclass
class Joined:
    trade: ClosedTrade
    signal: Optional[SignalAttempt]


def _percentile(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 100:
        return float(sorted_vals[-1])
    # linear interpolation between closest ranks
    n = len(sorted_vals)
    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _format_pct(x: Optional[float]) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    return f"{x * 100:.3f}%"


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _normalize_reason(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None
    return str(reason).strip().lower()


def _regime_bin(conf: Optional[float]) -> str:
    if conf is None or math.isnan(conf):
        return "unknown"
    if conf < 0.30:
        return "low(<0.30)"
    if conf < 0.60:
        return "mid(0.30-0.60)"
    return "high(>=0.60)"


def _collect_signal_attempts(paths: Sequence[Path], symbol_filter: Optional[str]) -> List[SignalAttempt]:
    last_trigger: Dict[str, float] = {}
    last_bucket: Dict[str, int] = {}
    last_downtrend: Dict[str, Tuple[datetime, float]] = {}

    # Per-symbol in-flight attempt state
    active_start_ts: Dict[str, datetime] = {}
    active_start_bucket: Dict[str, Optional[int]] = {}
    active_min_trigger: Dict[str, Optional[float]] = {}
    active_max_trigger: Dict[str, Optional[float]] = {}
    active_pass_ts: Dict[str, Optional[datetime]] = {}
    active_pass_bucket: Dict[str, Optional[int]] = {}
    active_trigger_at_pass: Dict[str, Optional[float]] = {}

    def _upd_minmax(sym: str) -> None:
        trg = last_trigger.get(sym)
        if trg is None:
            return
        mn = active_min_trigger.get(sym)
        mx = active_max_trigger.get(sym)
        if mn is None or trg < mn:
            active_min_trigger[sym] = trg
        if mx is None or trg > mx:
            active_max_trigger[sym] = trg

    attempts: List[SignalAttempt] = []

    for path in paths:
        for ts, _logger, _level, msg in _iter_log_lines(path):
            m = HYBRID_META_RE.search(msg)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                try:
                    last_trigger[sym] = float(m.group("trigger"))
                    last_bucket[sym] = int(m.group("bucket"))
                except Exception:
                    continue
                if sym in active_start_ts:
                    _upd_minmax(sym)
                continue

            m = DOWNTREND_VETO_RE.search(msg)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                try:
                    last_downtrend[sym] = (ts, float(m.group("adx")))
                except Exception:
                    pass
                continue

            m = PERSIST_RE.search(msg)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                samples = int(m.group("samples"))
                passed = m.group("passed") == "True"

                # keep min/max updated on persistency checkpoints too
                if sym in active_start_ts:
                    _upd_minmax(sym)

                if samples == 1 and not passed:
                    active_start_ts[sym] = ts
                    active_start_bucket[sym] = last_bucket.get(sym)
                    active_min_trigger[sym] = None
                    active_max_trigger[sym] = None
                    active_pass_ts[sym] = None
                    active_pass_bucket[sym] = None
                    active_trigger_at_pass[sym] = None
                    _upd_minmax(sym)
                    continue

                if passed and sym in active_start_ts and active_pass_ts.get(sym) is None:
                    active_pass_ts[sym] = ts
                    active_pass_bucket[sym] = last_bucket.get(sym)
                    active_trigger_at_pass[sym] = last_trigger.get(sym)
                    _upd_minmax(sym)
                    continue

            m = SIGNAL_RE.search(msg)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                if sym not in active_start_ts:
                    continue

                start_ts = active_start_ts[sym]
                pass_ts = active_pass_ts.get(sym)
                min_trg = active_min_trigger.get(sym)
                trg_at_pass = active_trigger_at_pass.get(sym)
                bounce = None
                if (
                    min_trg is not None
                    and trg_at_pass is not None
                    and min_trg > 0
                    and trg_at_pass > 0
                ):
                    bounce = (trg_at_pass - min_trg) / min_trg

                adx = None
                ts_adx = last_downtrend.get(sym)
                if ts_adx is not None:
                    adx_ts, adx_val = ts_adx
                    # only attach if relatively recent to the signal
                    if abs((ts - adx_ts).total_seconds()) <= 15 * 60:
                        adx = adx_val

                attempts.append(
                    SignalAttempt(
                        symbol=sym,
                        signal_ts=ts,
                        start_ts=start_ts,
                        pass_ts=pass_ts,
                        start_bucket=active_start_bucket.get(sym),
                        pass_bucket=active_pass_bucket.get(sym),
                        trigger_at_pass=trg_at_pass,
                        min_trigger=min_trg,
                        max_trigger=active_max_trigger.get(sym),
                        bounce_from_min=bounce,
                        downtrend_adx=adx,
                    )
                )

                # reset attempt state after emitting a signal
                active_start_ts.pop(sym, None)
                active_start_bucket.pop(sym, None)
                active_min_trigger.pop(sym, None)
                active_max_trigger.pop(sym, None)
                active_pass_ts.pop(sym, None)
                active_pass_bucket.pop(sym, None)
                active_trigger_at_pass.pop(sym, None)
                continue

    # sort for fast join (time-ordered)
    attempts.sort(key=lambda a: (a.symbol, a.signal_ts))
    return attempts


def _collect_closed_trades(paths: Sequence[Path], symbol_filter: Optional[str]) -> List[ClosedTrade]:
    trades: List[ClosedTrade] = []
    for path in paths:
        for ts, _logger, _level, msg in _iter_log_lines(path):
            m = TRADE_CLOSED_RE.search(msg)
            if not m:
                continue
            payload = m.group("json")
            try:
                data = json.loads(payload)
            except Exception:
                continue

            strategy = data.get("strategy") or data.get("strategy_name")
            if strategy != "adaptive_ob":
                continue
            sym = data.get("symbol")
            if not isinstance(sym, str):
                continue
            if symbol_filter and sym != symbol_filter:
                continue

            entry_time = _parse_iso_ts(data.get("entry_time"))
            exit_time = _parse_iso_ts(data.get("exit_time") or data.get("timestamp"))
            trade = ClosedTrade(
                path=path,
                trade_id=data.get("trade_id"),
                position_id=data.get("position_id"),
                symbol=sym,
                side=data.get("side"),
                strategy=strategy,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=_safe_float(data.get("entry_price")),
                exit_price=_safe_float(data.get("exit_price")),
                exit_reason=_normalize_reason(data.get("exit_reason")),
                pnl_usd=_safe_float(data.get("pnl_usd") or data.get("realized_pnl_usd")),
                pnl_pct=_safe_float(data.get("pnl_pct")),
                regime_at_entry=(data.get("regime_at_entry") or data.get("ml_regime") or data.get("regime")),
                regime_conf=_safe_float(data.get("regime_conf") or data.get("regime_confidence")),
            )

            # Some variants keep these nested
            if trade.regime_at_entry is None or trade.regime_conf is None:
                md = data.get("entry_metadata")
                if isinstance(md, dict):
                    rd = md.get("regime_data")
                    if isinstance(rd, dict):
                        if trade.regime_at_entry is None:
                            trade.regime_at_entry = rd.get("regime")
                        if trade.regime_conf is None:
                            trade.regime_conf = _safe_float(rd.get("confidence"))

            # If entry_time missing, fall back to the file-prefix timestamp (worse)
            if trade.entry_time is None:
                trade.entry_time = ts

            trades.append(trade)

    return trades


def _join_trades_to_signals(
    trades: Sequence[ClosedTrade],
    attempts: Sequence[SignalAttempt],
    max_gap_seconds: int,
) -> List[Joined]:
    # Build per-symbol list of attempts for binary-ish search by walking pointer.
    by_symbol: Dict[str, List[SignalAttempt]] = {}
    for a in attempts:
        by_symbol.setdefault(a.symbol, []).append(a)

    joined: List[Joined] = []
    for t in trades:
        sym_attempts = by_symbol.get(t.symbol, [])
        best: Optional[SignalAttempt] = None
        if t.entry_time is not None and sym_attempts:
            # linear scan backwards from end; logs aren't huge, and this keeps it simple.
            for a in reversed(sym_attempts):
                if a.signal_ts <= t.entry_time:
                    gap = (t.entry_time - a.signal_ts).total_seconds()
                    if 0 <= gap <= max_gap_seconds:
                        best = a
                    break
        joined.append(Joined(trade=t, signal=best))
    return joined


def _histogram(values: Sequence[float], bin_edges: Sequence[float]) -> List[int]:
    # bins: [e0,e1), [e1,e2), ... [e_{n-2}, e_{n-1}) with last inclusive
    counts = [0 for _ in range(len(bin_edges) - 1)]
    for v in values:
        if math.isnan(v):
            continue
        placed = False
        for i in range(len(bin_edges) - 1):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if i == len(bin_edges) - 2:
                if lo <= v <= hi:
                    counts[i] += 1
                    placed = True
                    break
            else:
                if lo <= v < hi:
                    counts[i] += 1
                    placed = True
                    break
        if not placed and v < bin_edges[0]:
            counts[0] += 1
    return counts


def _print_percentiles(title: str, values: Sequence[float]) -> None:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    print(f"\n{title}")
    print(f"n={len(vals)}")
    if not vals:
        return
    for q in (5, 10, 25, 50, 75, 90, 95):
        p = _percentile(vals, q)
        print(f"p{q:02d}: {_format_pct(p)}")


def _print_hist(title: str, values: Sequence[float], bin_edges: Sequence[float]) -> None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    print(f"\n{title}")
    print(f"n={len(vals)}")
    if not vals:
        return
    counts = _histogram(vals, bin_edges)
    for i, c in enumerate(counts):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        print(f"{lo*100:6.3f}% .. {hi*100:6.3f}% : {c}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help='Glob for log files, e.g. "logs/**/*.log"')
    ap.add_argument("--symbol", default=None, help='Optional symbol filter, e.g. "BTC/USDT:USDT"')
    ap.add_argument("--max-gap-seconds", type=int, default=300, help="Max seconds between signal and entry_time")
    ap.add_argument("--out", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.logs))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(f"No log files matched: {args.logs}")

    attempts = _collect_signal_attempts(paths, args.symbol)
    trades = _collect_closed_trades(paths, args.symbol)
    joined = _join_trades_to_signals(trades, attempts, args.max_gap_seconds)

    # Extract bounce series
    all_bounce: List[float] = []
    tp_bounce: List[float] = []
    sl_bounce: List[float] = []
    profit_bounce: List[float] = []
    loss_bounce: List[float] = []

    by_regime: Dict[str, List[float]] = {}
    by_regime_bin: Dict[str, List[float]] = {}
    by_regime_bin_tp: Dict[str, List[float]] = {}
    by_regime_bin_sl: Dict[str, List[float]] = {}

    unmatched = 0
    for j in joined:
        if j.signal is None or j.signal.bounce_from_min is None:
            unmatched += 1
            continue
        b = j.signal.bounce_from_min
        all_bounce.append(b)

        reason = _normalize_reason(j.trade.exit_reason)
        if reason == "take_profit":
            tp_bounce.append(b)
        if reason == "stop_loss":
            sl_bounce.append(b)

        if (j.trade.pnl_usd is not None) and (j.trade.pnl_usd > 0):
            profit_bounce.append(b)
        if (j.trade.pnl_usd is not None) and (j.trade.pnl_usd < 0):
            loss_bounce.append(b)

        reg = (j.trade.regime_at_entry or "unknown")
        reg = str(reg).strip().lower() if reg is not None else "unknown"
        by_regime.setdefault(reg, []).append(b)

        bin_name = _regime_bin(j.trade.regime_conf)
        by_regime_bin.setdefault(bin_name, []).append(b)
        if reason == "take_profit":
            by_regime_bin_tp.setdefault(bin_name, []).append(b)
        if reason == "stop_loss":
            by_regime_bin_sl.setdefault(bin_name, []).append(b)

    print(f"logs={len(paths)} trades(adaptive_ob)={len(trades)} attempts(signals)={len(attempts)}")
    print(f"joined_with_bounce={len(all_bounce)} missing_bounce_or_match={unmatched}")

    # Default bins: 0%..0.50% step 0.05% plus an overflow bucket to 1.0%
    bin_edges = [i / 10000.0 for i in range(0, 51, 5)] + [0.01]

    _print_percentiles("ALL (joined)", all_bounce)
    _print_percentiles("TP only", tp_bounce)
    _print_percentiles("SL only", sl_bounce)
    _print_percentiles("Profit (pnl_usd>0)", profit_bounce)
    _print_percentiles("Loss (pnl_usd<0)", loss_bounce)

    _print_hist("Histogram ALL", all_bounce, bin_edges)
    _print_hist("Histogram TP", tp_bounce, bin_edges)
    _print_hist("Histogram SL", sl_bounce, bin_edges)

    print("\nBy regime_conf bins (ALL):")
    for k in ("low(<0.30)", "mid(0.30-0.60)", "high(>=0.60)", "unknown"):
        vals = by_regime_bin.get(k, [])
        print(f"- {k}: n={len(vals)} p50={_format_pct(_percentile(sorted(vals), 50)) if vals else 'n/a'}")
    print("\nBy regime_conf bins (TP):")
    for k in ("low(<0.30)", "mid(0.30-0.60)", "high(>=0.60)", "unknown"):
        vals = by_regime_bin_tp.get(k, [])
        print(f"- {k}: n={len(vals)} p50={_format_pct(_percentile(sorted(vals), 50)) if vals else 'n/a'}")
    print("\nBy regime_conf bins (SL):")
    for k in ("low(<0.30)", "mid(0.30-0.60)", "high(>=0.60)", "unknown"):
        vals = by_regime_bin_sl.get(k, [])
        print(f"- {k}: n={len(vals)} p50={_format_pct(_percentile(sorted(vals), 50)) if vals else 'n/a'}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "log_path",
                    "trade_id",
                    "position_id",
                    "symbol",
                    "side",
                    "entry_time",
                    "exit_time",
                    "exit_reason",
                    "pnl_usd",
                    "pnl_pct",
                    "regime_at_entry",
                    "regime_conf",
                    "signal_ts",
                    "bounce_from_min",
                    "min_trigger",
                    "trigger_at_pass",
                    "downtrend_adx",
                ]
            )
            for j in joined:
                s = j.signal
                w.writerow(
                    [
                        str(j.trade.path),
                        j.trade.trade_id,
                        j.trade.position_id,
                        j.trade.symbol,
                        j.trade.side,
                        j.trade.entry_time.isoformat() if j.trade.entry_time else None,
                        j.trade.exit_time.isoformat() if j.trade.exit_time else None,
                        j.trade.exit_reason,
                        j.trade.pnl_usd,
                        j.trade.pnl_pct,
                        j.trade.regime_at_entry,
                        j.trade.regime_conf,
                        s.signal_ts.isoformat() if s else None,
                        s.bounce_from_min if s else None,
                        s.min_trigger if s else None,
                        s.trigger_at_pass if s else None,
                        s.downtrend_adx if s else None,
                    ]
                )
        print(f"\nwrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
