"""Generate a concise session trade timeline + metrics report from a live trading log.

- Primary source of truth: `TRADE_CLOSED {json...}` lines.
- Optional enrichment: `[P&L-UPDATE]` lines (used only if TRADE_CLOSED lacks MFE/MAE).

Outputs:
- CSV timeline (one row per closed trade)
- Markdown summary + table

Designed for Windows + Python 3.11.
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
from typing import Any, Iterable


_LOG_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+-\s+")
_TRADE_CLOSED_RE = re.compile(r"\bTRADE_CLOSED\b")
_SIGNAL_INGRESS_RE = re.compile(
    r"\[(?P<label>[A-Z0-9_]+/[^\]]+)\]\s+Signal ingress\s+\|\s+"
    r"side=(?P<side>\w+)\s+\|\s+intent_hint=(?P<intent>\w+)"
    r"(?:\s+\|\s+reason=(?P<reason>.*))?$"
)
_STRAT_REJECT_RE = re.compile(
    r"\[(?P<label>[A-Z0-9_]+/[^\]]+)\]\s+REJECTED\b(?P<tail>.*)$"
)

_PNL_UPDATE_RE = re.compile(
    r"\[P&L-UPDATE\]\s+"
    r"(?P<position_id>\S+)\s+\|\s+"
    r"(?P<symbol>[^|]+)\|\s+"
    r"Entry:\s+\$(?P<entry>[0-9,]+(?:\.[0-9]+)?),\s+"
    r"Current:\s+\$(?P<current>[0-9,]+(?:\.[0-9]+)?)\s+\|\s+"
    r"P&L:\s+\$(?P<pnl_usd>[-+]?\d+(?:\.\d+)?)\s+\((?P<pnl_pct>[-+]?\d+(?:\.\d+)?)%\)"
)


@dataclass(frozen=True)
class TradeClosed:
    run_id: str
    trade_id: str
    position_id: str
    symbol: str
    timeframe: str
    side: str
    strategy: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    exit_reason: str
    position_size: float | None
    pnl_usd: float | None
    realized_pnl_usd: float | None
    pnl_pct: float | None
    duration_min: float | None

    # optional enrichments commonly present
    ml_regime: str | None
    regime_conf: float | None
    ml_price_direction: str | None
    quality_score: float | None
    volume_bucket_at_entry: str | None

    # excursions (percent, not fraction) if present
    mfe_pct: float | None
    mae_pct: float | None


def _parse_log_prefix_ts(line: str) -> datetime | None:
    m = _LOG_TS_RE.match(line)
    if not m:
        return None
    # Log prefix has no timezone; treat as UTC for consistent analysis.
    # (Run IDs and event JSON timestamps are UTC `...Z`.)
    try:
        naive = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc)


def _parse_iso8601(ts: str) -> datetime:
    # Handles both `...Z` and `+00:00`
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    # If no timezone info is provided (e.g. "2026-01-20 01:00:00"), assume UTC.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return float(v)
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            if not s:
                return None
            return float(s)
    except Exception:
        return None
    return None


def _extract_json_after_marker(line: str, marker: str) -> dict[str, Any] | None:
    idx = line.find(marker)
    if idx < 0:
        return None

    json_start = line.find("{", idx)
    if json_start < 0:
        return None

    try:
        decoder = json.JSONDecoder()
        payload, _end = decoder.raw_decode(line[json_start:])
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return None
    return None


def _parse_trade_closed_payload(payload: dict[str, Any]) -> TradeClosed | None:
    if payload.get("event") != "TRADE_CLOSED":
        return None

    try:
        entry_time = _parse_iso8601(str(payload["entry_time"]))
        exit_time = _parse_iso8601(str(payload["exit_time"]))
    except Exception:
        return None

    return TradeClosed(
        run_id=str(payload.get("run_id") or ""),
        trade_id=str(payload.get("trade_id") or ""),
        position_id=str(payload.get("position_id") or ""),
        symbol=str(payload.get("symbol") or ""),
        timeframe=str(payload.get("timeframe") or ""),
        side=str(payload.get("side") or ""),
        strategy=str(payload.get("strategy") or payload.get("strategy_name") or ""),
        entry_price=float(payload.get("entry_price")),
        entry_time=entry_time,
        exit_price=float(payload.get("exit_price")),
        exit_time=exit_time,
        exit_reason=str(payload.get("exit_reason") or ""),
        position_size=_safe_float(payload.get("position_size")),
        pnl_usd=_safe_float(payload.get("pnl_usd")),
        realized_pnl_usd=_safe_float(payload.get("realized_pnl_usd")),
        pnl_pct=_safe_float(payload.get("pnl_pct")),
        duration_min=_safe_float(payload.get("duration_min")),
        ml_regime=(str(payload.get("ml_regime")) if payload.get("ml_regime") is not None else None),
        regime_conf=_safe_float(payload.get("regime_conf")),
        ml_price_direction=(
            str(payload.get("ml_price_direction"))
            if payload.get("ml_price_direction") is not None
            else None
        ),
        quality_score=_safe_float(payload.get("quality_score")),
        volume_bucket_at_entry=(
            str(payload.get("volume_bucket_at_entry"))
            if payload.get("volume_bucket_at_entry") is not None
            else None
        ),
        mfe_pct=_safe_float(payload.get("mfe_pct")),
        mae_pct=_safe_float(payload.get("mae_pct")),
    )


@dataclass
class PnlSample:
    ts: datetime
    entry_price: float
    current_price: float


@dataclass(frozen=True)
class PhaseWindow:
    name: str
    start: datetime
    end: datetime


def _load_phases(path: str | None) -> list[PhaseWindow]:
    if not path:
        return []

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Phases JSON not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Phases JSON must be a list of {name,start,end}")

    phases: list[PhaseWindow] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Accept either canonical keys (name/start/end) or user-friendly keys
        # (phase_name/start_time/end_time).
        name = str(item.get("name") or item.get("phase_name") or "").strip()
        start = item.get("start") if item.get("start") is not None else item.get("start_time")
        end = item.get("end") if item.get("end") is not None else item.get("end_time")
        if not name or not start or not end:
            continue
        phases.append(PhaseWindow(name=name, start=_parse_iso8601(str(start)), end=_parse_iso8601(str(end))))

    phases.sort(key=lambda ph: ph.start)
    return phases


def _assign_phase(ts: datetime, phases: list[PhaseWindow]) -> str:
    if not phases:
        return ""
    for ph in phases:
        if ph.start <= ts <= ph.end:
            return ph.name
    return ""


def _parse_pnl_update(line: str) -> tuple[str, PnlSample] | None:
    ts = _parse_log_prefix_ts(line)
    if ts is None:
        return None

    m = _PNL_UPDATE_RE.search(line)
    if not m:
        return None

    position_id = m.group("position_id")
    entry = _safe_float(m.group("entry"))
    current = _safe_float(m.group("current"))
    if entry is None or current is None:
        return None

    return position_id, PnlSample(ts=ts, entry_price=entry, current_price=current)


def _compute_excursions_from_pnl(trade: TradeClosed, samples: list[PnlSample]) -> tuple[float | None, float | None]:
    if not samples:
        return None, None

    # Use entry_price from trade as the base; if missing, fall back to sample.
    entry_price = trade.entry_price if trade.entry_price else samples[0].entry_price
    if not entry_price:
        return None, None

    prices = [s.current_price for s in samples]
    if not prices:
        return None, None

    max_price = max(prices)
    min_price = min(prices)

    side = trade.side.upper()
    if side == "LONG":
        mfe = (max_price - entry_price) / entry_price * 100.0
        mae = (min_price - entry_price) / entry_price * 100.0
    elif side == "SHORT":
        mfe = (entry_price - min_price) / entry_price * 100.0
        mae = (entry_price - max_price) / entry_price * 100.0
        mae = -mae  # keep MAE as negative (adverse)
    else:
        return None, None

    return mfe, mae


def _fmt_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _write_csv(path: Path, trades: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trades:
        path.write_text("", encoding="utf-8")
        return

    # Stable column order
    fieldnames: list[str] = list(trades[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trades:
            writer.writerow(row)


def _pct(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.3f}"


def _usd(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.4f}"


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate session trade timeline + metrics from a live log")
    parser.add_argument("--log", required=True, help="Path to live trading log file")
    parser.add_argument("--run-id", default=None, help="Filter to a specific run_id")
    parser.add_argument("--start", default=None, help="UTC start time (inclusive), e.g. 2026-01-19T23:12:00Z")
    parser.add_argument("--end", default=None, help="UTC end time (inclusive), e.g. 2026-01-20T05:12:00Z")
    parser.add_argument(
        "--phases-json",
        default=None,
        help=(
            "Optional JSON file with chart phases: "
            "[{\"name\":\"selloff\",\"start\":\"...Z\",\"end\":\"...Z\"}, ...]"
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix path (without extension). Default: reports/session_<run_id>",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    start_dt = _parse_iso8601(args.start) if args.start else None
    end_dt = _parse_iso8601(args.end) if args.end else None
    phases = _load_phases(args.phases_json)

    trades: list[TradeClosed] = []
    pnl_by_position: dict[str, list[PnlSample]] = {}

    # Signal funnel + rejection analysis
    signal_ingress_raw: dict[str, int] = {}
    signal_ingress_dedup_keys: dict[str, set[tuple[str, str, str, str, datetime]]] = {}
    reject_counts: dict[str, int] = {}
    reject_reasons: dict[str, dict[str, int]] = {}
    reject_examples: dict[str, list[str]] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Signal ingress + rejections are based on log-prefix timestamps.
            prefix_ts = _parse_log_prefix_ts(line)
            phase_for_line = _assign_phase(prefix_ts, phases) if prefix_ts else ""

            if "Signal ingress" in line and prefix_ts is not None:
                m = _SIGNAL_INGRESS_RE.search(line.strip())
                if m:
                    label = m.group("label")
                    strategy = (label.split("/", 1)[0] if "/" in label else label).lower()
                    side = m.group("side").lower()
                    intent = m.group("intent").lower()
                    if intent == "entry" and side in {"long", "buy"}:
                        phase_key = phase_for_line or "(outside phases)"
                        signal_ingress_raw[phase_key] = signal_ingress_raw.get(phase_key, 0) + 1
                        # De-dup at 60s granularity to avoid spammy repeated ingress logs
                        ts_bucket = prefix_ts.replace(second=0, microsecond=0)
                        signal_ingress_dedup_keys.setdefault(phase_key, set()).add(
                            (phase_key, strategy, side, intent, ts_bucket)
                        )

            if "REJECTED" in line and prefix_ts is not None:
                # Most user-relevant rejections are StrategyCoordinator “REJECTED (Risk Check)” style.
                m = _STRAT_REJECT_RE.search(line.strip())
                if m:
                    phase_key = phase_for_line or "(outside phases)"
                    reject_counts[phase_key] = reject_counts.get(phase_key, 0) + 1

                    tail = (m.group("tail") or "").strip()
                    # Normalize to a compact reason token.
                    category = ""
                    detail = ""
                    if ":" in tail:
                        category, detail = tail.split(":", 1)
                        category = category.strip()
                        detail = detail.strip()
                    else:
                        category = tail.strip()

                    detail_l = detail.lower()
                    category_l = category.lower()
                    if "pyramiding_disabled_for_strategy" in detail_l:
                        reason = "pyramiding_disabled_for_strategy"
                    elif "risk/reward ratio" in detail_l or "riskrewardrati" in detail_l:
                        reason = "risk_reward_below_target"
                    elif "volume" in category_l and "gating" in category_l:
                        reason = "volume_gating"
                    elif detail:
                        reason = detail[:80]
                    elif category:
                        reason = category[:80]
                    else:
                        reason = "REJECTED"

                    reject_reasons.setdefault(phase_key, {})
                    reject_reasons[phase_key][reason] = reject_reasons[phase_key].get(reason, 0) + 1

                    # Keep a few examples for auditability
                    ex = line.strip()
                    reject_examples.setdefault(phase_key, [])
                    if len(reject_examples[phase_key]) < 5:
                        reject_examples[phase_key].append(ex)

            if _TRADE_CLOSED_RE.search(line):
                payload = _extract_json_after_marker(line, "TRADE_CLOSED")
                if payload:
                    trade = _parse_trade_closed_payload(payload)
                    if trade is not None:
                        if args.run_id and trade.run_id != args.run_id:
                            continue
                        if start_dt and trade.entry_time < start_dt:
                            continue
                        if end_dt and trade.entry_time > end_dt:
                            continue
                        trades.append(trade)
                continue

            if "[P&L-UPDATE]" in line:
                parsed = _parse_pnl_update(line)
                if parsed is None:
                    continue
                position_id, sample = parsed
                pnl_by_position.setdefault(position_id, []).append(sample)

    if not trades:
        print("No TRADE_CLOSED events matched filters.")
        return 2

    trades.sort(key=lambda t: t.entry_time)

    run_id = args.run_id or trades[0].run_id or "unknown"
    out_prefix = Path(args.out_prefix) if args.out_prefix else Path("reports") / f"session_{run_id}"
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    rows: list[dict[str, Any]] = []
    pnl_values: list[float] = []
    win_values: list[bool] = []
    pnl_by_phase: dict[str, float] = {}
    pnl_by_exit_reason: dict[str, float] = {}
    executed_by_phase: dict[str, int] = {}

    for t in trades:
        realized = t.realized_pnl_usd if t.realized_pnl_usd is not None else t.pnl_usd
        pnl_values.append(realized or 0.0)
        win_values.append((realized or 0.0) > 0)

        phase = _assign_phase(t.entry_time, phases)
        if phase:
            pnl_by_phase[phase] = pnl_by_phase.get(phase, 0.0) + (realized or 0.0)
            executed_by_phase[phase] = executed_by_phase.get(phase, 0) + 1
        reason = (t.exit_reason or "").strip() or "unknown"
        pnl_by_exit_reason[reason] = pnl_by_exit_reason.get(reason, 0.0) + (realized or 0.0)

        mfe = t.mfe_pct
        mae = t.mae_pct
        if mfe is None or mae is None:
            samples = pnl_by_position.get(t.position_id, [])
            computed_mfe, computed_mae = _compute_excursions_from_pnl(t, samples)
            mfe = mfe if mfe is not None else computed_mfe
            mae = mae if mae is not None else computed_mae

        duration_min = t.duration_min
        if duration_min is None:
            duration_min = (t.exit_time - t.entry_time).total_seconds() / 60.0

        rows.append(
            {
                "phase": phase,
                "run_id": t.run_id,
                "trade_id": t.trade_id,
                "position_id": t.position_id,
                "symbol": t.symbol,
                "timeframe": t.timeframe,
                "side": t.side,
                "strategy": t.strategy,
                "entry_time_utc": _fmt_dt(t.entry_time),
                "exit_time_utc": _fmt_dt(t.exit_time),
                "duration_min": f"{duration_min:.2f}",
                "entry_price": f"{t.entry_price:.2f}",
                "exit_price": f"{t.exit_price:.2f}",
                "position_size": "" if t.position_size is None else f"{t.position_size:.6f}",
                "exit_reason": t.exit_reason,
                "pnl_usd": _usd(realized),
                "pnl_pct": _pct(t.pnl_pct),
                "mfe_pct": _pct(mfe),
                "mae_pct": _pct(mae),
                "ml_regime": t.ml_regime or "",
                "regime_conf": "" if t.regime_conf is None else f"{t.regime_conf:.4f}",
                "ml_price_direction": t.ml_price_direction or "",
                "quality_score": "" if t.quality_score is None else f"{t.quality_score:.3f}",
                "volume_bucket": t.volume_bucket_at_entry or "",
            }
        )

    _write_csv(csv_path, rows)

    total_trades = len(rows)
    total_pnl = sum(pnl_values)
    wins = sum(1 for w in win_values if w)
    losses = total_trades - wins
    win_rate = wins / total_trades * 100.0 if total_trades else 0.0
    avg_pnl = _mean(pnl_values)
    avg_win = _mean(v for v, w in zip(pnl_values, win_values, strict=True) if w)
    avg_loss = _mean(v for v, w in zip(pnl_values, win_values, strict=True) if not w)

    start_str = _fmt_dt(trades[0].entry_time)
    end_str = _fmt_dt(trades[-1].exit_time)

    # Simple equity curve stats using cumulative realized PnL, starting at 0.
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for t in trades:
        realized = t.realized_pnl_usd if t.realized_pnl_usd is not None else t.pnl_usd
        equity += realized or 0.0
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_drawdown:
            max_drawdown = dd

    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Session Report: {run_id}\n\n")
        f.write(f"- Log: {log_path.as_posix()}\n")
        if args.start or args.end:
            f.write(f"- Filters: start={args.start or ''} end={args.end or ''}\n")
        f.write(f"- Span (from trades): {start_str} → {end_str} (UTC)\n\n")

        f.write("## Summary\n")
        f.write(f"- Trades: {total_trades} (wins={wins}, losses={losses}, win_rate={win_rate:.1f}%)\n")
        f.write(f"- Total realized PnL: ${total_pnl:.4f}\n")
        f.write(f"- Max drawdown (realized PnL curve): ${max_drawdown:.4f}\n")
        if avg_pnl is not None:
            f.write(f"- Avg PnL / trade: ${avg_pnl:.4f}\n")
        if avg_win is not None:
            f.write(f"- Avg win: ${avg_win:.4f}\n")
        if avg_loss is not None:
            f.write(f"- Avg loss: ${avg_loss:.4f}\n")
        f.write("\n")

        if pnl_by_phase:
            f.write("## PnL By Phase\n")
            for name, pnl in sorted(pnl_by_phase.items(), key=lambda kv: kv[0]):
                f.write(f"- {name}: ${pnl:.4f}\n")
            f.write("\n")

        if phases:
            f.write("## Signal Funnel By Phase (Entry LONG)\n")
            f.write("Counts are based on `Signal ingress` lines (raw + de-dup @60s), and executed trades via `TRADE_CLOSED` entries.\n\n")
            f.write("| Phase | Signals (raw) | Signals (dedup@60s) | Executed trades | Rejections | Top rejection reasons |\n")
            f.write("|---|---:|---:|---:|---:|---|\n")

            for ph in phases:
                key = ph.name
                raw = signal_ingress_raw.get(key, 0)
                dedup = len(signal_ingress_dedup_keys.get(key, set()))
                executed = executed_by_phase.get(key, 0)
                rej = reject_counts.get(key, 0)
                top_reasons = reject_reasons.get(key, {})
                if top_reasons:
                    top3 = sorted(top_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                    top_str = ", ".join([f"{name}×{cnt}" for name, cnt in top3])
                else:
                    top_str = ""
                f.write(f"| {key} | {raw} | {dedup} | {executed} | {rej} | {top_str} |\n")
            f.write("\n")

            # Provide a short audit trail for Dump phase specifically (as requested)
            dump_key = next((ph.name for ph in phases if "Dump" in ph.name or "Panik" in ph.name), None)
            if dump_key and reject_examples.get(dump_key):
                f.write(f"## Rejection Examples: {dump_key}\n")
                for ex in reject_examples[dump_key]:
                    f.write(f"- {ex}\n")
                f.write("\n")

        f.write("## PnL By Exit Reason\n")
        for name, pnl in sorted(pnl_by_exit_reason.items(), key=lambda kv: (-abs(kv[1]), kv[0])):
            f.write(f"- {name}: ${pnl:.4f}\n")
        f.write("\n")

        f.write("## Timeline\n")
        f.write(
            "| # | Phase | Entry (UTC) | Exit (UTC) | Side | Strategy | Exit Reason | PnL $ | PnL % | MFE % | MAE % | ML Regime | Q | Vol |\n"
        )
        f.write(
            "|---:|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|\n"
        )
        for i, r in enumerate(rows, start=1):
            f.write(
                "| "
                + " | ".join(
                    [
                        str(i),
                        r.get("phase", "") or "",
                        r["entry_time_utc"],
                        r["exit_time_utc"],
                        r["side"],
                        r["strategy"],
                        r["exit_reason"],
                        r["pnl_usd"] or "",
                        r["pnl_pct"] or "",
                        r["mfe_pct"] or "",
                        r["mae_pct"] or "",
                        r["ml_regime"] or "",
                        r["quality_score"] or "",
                        r["volume_bucket"] or "",
                    ]
                )
                + " |\n"
            )

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote MD:  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
