"""Extract trade entry/exit timestamps and recommend OHLCV time windows.

This script is intentionally dependency-free (stdlib only) so it can be run anywhere.

It scans `TRADE_CLOSED {json}` log lines and extracts:
- trade_id
- entry_time / exit_time
- symbol / timeframe / side / strategy (when available)

Then it prints a recommended OHLCV window that includes optional warmup time
(for indicators like ATR) plus configurable padding.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator


TRADE_CLOSED_MARKER = "TRADE_CLOSED "


@dataclass(frozen=True)
class TradeTimes:
    log_path: Path
    trade_id: str
    entry_time: datetime
    exit_time: datetime
    symbol: str | None = None
    timeframe: str | None = None
    side: str | None = None
    strategy: str | None = None


def _parse_iso8601_z(value: str) -> datetime:
    # Examples: "2026-01-31T01:35:33.504803Z" or with offset "+00:00".
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iter_log_files(logs_dir: Path) -> Iterator[Path]:
    if not logs_dir.exists():
        return
    yield from sorted(logs_dir.glob("**/*.log"))


def _extract_trade_closed_json(line: str) -> dict | None:
    idx = line.find(TRADE_CLOSED_MARKER)
    if idx == -1:
        return None

    json_part = line[idx + len(TRADE_CLOSED_MARKER) :].strip()
    if not json_part.startswith("{"):
        return None

    try:
        payload = json.loads(json_part)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("event") != "TRADE_CLOSED":
        return None
    return payload


def _scan_log_for_trades(log_path: Path, trade_ids: set[str]) -> dict[str, TradeTimes]:
    found: dict[str, TradeTimes] = {}

    # Using errors="replace" because logs occasionally include odd unicode.
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "TRADE_CLOSED" not in line:
                continue
            payload = _extract_trade_closed_json(line)
            if payload is None:
                continue

            trade_id = payload.get("trade_id")
            if not isinstance(trade_id, str) or trade_id not in trade_ids:
                continue

            entry_time_s = payload.get("entry_time")
            exit_time_s = payload.get("exit_time")
            if not isinstance(entry_time_s, str) or not isinstance(exit_time_s, str):
                continue

            entry_time = _parse_iso8601_z(entry_time_s)
            exit_time = _parse_iso8601_z(exit_time_s)

            found[trade_id] = TradeTimes(
                log_path=log_path,
                trade_id=trade_id,
                entry_time=entry_time,
                exit_time=exit_time,
                symbol=payload.get("symbol") if isinstance(payload.get("symbol"), str) else None,
                timeframe=payload.get("timeframe") if isinstance(payload.get("timeframe"), str) else None,
                side=payload.get("side") if isinstance(payload.get("side"), str) else None,
                strategy=payload.get("strategy") if isinstance(payload.get("strategy"), str) else None,
            )

            if len(found) == len(trade_ids):
                break

    return found


def find_trade_times(
    *,
    trade_ids: Iterable[str],
    log_path: Path | None,
    logs_dir: Path,
) -> tuple[dict[str, TradeTimes], list[str]]:
    trade_id_set = {t.strip() for t in trade_ids if t.strip()}
    if not trade_id_set:
        raise ValueError("No trade ids provided")

    if log_path is not None:
        found = _scan_log_for_trades(log_path, trade_id_set)
        missing = sorted(trade_id_set - set(found.keys()))
        return found, missing

    remaining = set(trade_id_set)
    all_found: dict[str, TradeTimes] = {}

    for candidate in _iter_log_files(logs_dir):
        if not remaining:
            break
        partial = _scan_log_for_trades(candidate, remaining)
        if partial:
            all_found.update(partial)
            remaining -= set(partial.keys())

    missing = sorted(remaining)
    return all_found, missing


def _format_dt(dt: datetime) -> str:
    # Use ISO 8601 with Z for easy copy/paste into APIs.
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat().replace("+00:00", "Z")


def _recommend_window(
    *,
    entry_time: datetime,
    exit_time: datetime,
    warmup_minutes: int,
    pre_pad_minutes: int,
    post_pad_minutes: int,
) -> tuple[datetime, datetime]:
    start = entry_time - timedelta(minutes=warmup_minutes + pre_pad_minutes)
    end = exit_time + timedelta(minutes=post_pad_minutes)
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract entry/exit timestamps for trades and recommend OHLCV windows.",
    )
    parser.add_argument(
        "trade_ids",
        nargs="+",
        help="One or more trade_id values (e.g. 45d7611a 6029d7f0)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Specific log file to scan. If omitted, scans all *.log under --logs-dir.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory to scan for logs when --log is not provided (default: logs).",
    )
    parser.add_argument(
        "--warmup-minutes",
        type=int,
        default=90,
        help="Extra history before entry for indicator warmup (default: 90).",
    )
    parser.add_argument(
        "--pre-pad-minutes",
        type=int,
        default=5,
        help="Extra padding before entry (default: 5).",
    )
    parser.add_argument(
        "--post-pad-minutes",
        type=int,
        default=10,
        help="Extra padding after exit (default: 10).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )

    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None
    logs_dir = Path(args.logs_dir)

    found, missing = find_trade_times(trade_ids=args.trade_ids, log_path=log_path, logs_dir=logs_dir)

    # Deterministic output order based on input order
    ordered: list[TradeTimes] = [found[t] for t in args.trade_ids if t in found]

    if args.format == "json":
        rows = []
        for t in ordered:
            start, end = _recommend_window(
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                warmup_minutes=args.warmup_minutes,
                pre_pad_minutes=args.pre_pad_minutes,
                post_pad_minutes=args.post_pad_minutes,
            )
            rows.append(
                {
                    "trade_id": t.trade_id,
                    "log_path": str(t.log_path),
                    "symbol": t.symbol,
                    "timeframe": t.timeframe,
                    "side": t.side,
                    "strategy": t.strategy,
                    "entry_time": _format_dt(t.entry_time),
                    "exit_time": _format_dt(t.exit_time),
                    "ohlcv_start": _format_dt(start),
                    "ohlcv_end": _format_dt(end),
                    "warmup_minutes": args.warmup_minutes,
                    "pre_pad_minutes": args.pre_pad_minutes,
                    "post_pad_minutes": args.post_pad_minutes,
                }
            )
        print(json.dumps({"trades": rows, "missing": missing}, indent=2))
    else:
        # Simple fixed-width-ish table.
        header = (
            "trade_id",
            "entry_time",
            "exit_time",
            "ohlcv_start",
            "ohlcv_end",
            "tf",
            "side",
            "strategy",
        )
        print(" | ".join(header))
        print("-" * 120)
        for t in ordered:
            start, end = _recommend_window(
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                warmup_minutes=args.warmup_minutes,
                pre_pad_minutes=args.pre_pad_minutes,
                post_pad_minutes=args.post_pad_minutes,
            )
            row = (
                t.trade_id,
                _format_dt(t.entry_time),
                _format_dt(t.exit_time),
                _format_dt(start),
                _format_dt(end),
                t.timeframe or "",
                t.side or "",
                t.strategy or "",
            )
            print(" | ".join(row))

        if missing:
            print("\nMissing trade_ids:")
            for m in missing:
                print(f"- {m}")

    # Exit non-zero if anything requested was missing.
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
