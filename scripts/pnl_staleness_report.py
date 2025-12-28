import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Run:
    current: str
    start_ts: datetime
    end_ts: datetime
    count: int


PUNCTUAL_TS_FMT = "%Y-%m-%d %H:%M:%S"

PANDL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - \[core\.live_trading_engine\] - INFO - .*\[P&L-UPDATE\] (?P<pos>pos_[^ ]+) .*?Entry: \$(?P<entry>\d+\.\d+), Current: \$(?P<current>\d+\.\d+) .*?P&L: \$(?P<pnl>[-\d\.]+) \((?P<pct>[-+\d\.]+)%\)"
)


def parse_ts(ts: str) -> datetime:
    # Logs appear to be local time without offset; use UTC tzinfo as a stable baseline.
    return datetime.strptime(ts, PUNCTUAL_TS_FMT).replace(tzinfo=timezone.utc)


def build_runs(series: list[tuple[datetime, str]]) -> list[Run]:
    if not series:
        return []
    series_sorted = sorted(series, key=lambda x: x[0])

    runs: list[Run] = []
    current_val = series_sorted[0][1]
    start_ts = series_sorted[0][0]
    end_ts = series_sorted[0][0]
    count = 1

    for ts, current in series_sorted[1:]:
        if current == current_val:
            count += 1
            end_ts = ts
        else:
            runs.append(Run(current=current_val, start_ts=start_ts, end_ts=end_ts, count=count))
            current_val = current
            start_ts = ts
            end_ts = ts
            count = 1

    runs.append(Run(current=current_val, start_ts=start_ts, end_ts=end_ts, count=count))
    return runs


def percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        raise ValueError("empty list")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = int(round(p * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def median(sorted_vals: list[int]) -> int:
    if not sorted_vals:
        raise ValueError("empty list")
    return sorted_vals[len(sorted_vals) // 2]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / "logs" / "live_trading_20251227_220109_746903.log"

    positions = [
        "pos_BTC/USDT:USDT_1766873782",
        "pos_BTC/USDT:USDT_1766879488",
    ]

    rows: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PANDL_RE.match(line)
            if not m:
                continue
            pos = m.group("pos")
            if pos not in positions:
                continue
            rows[pos].append((parse_ts(m.group("ts")), m.group("current")))

    for pos in positions:
        series = rows.get(pos, [])
        runs = build_runs(series)

        print(f"=== {pos} ===")
        if not series:
            print("No P&L updates found")
            print()
            continue

        run_lengths = sorted(r.count for r in runs)
        max_run = max(run_lengths)
        median_run = median(run_lengths)
        p95_run = percentile(run_lengths, 0.95)

        # Update cadence estimate from consecutive timestamps
        series_sorted = sorted(series, key=lambda x: x[0])
        deltas = [
            (t2 - t1).total_seconds()
            for (t1, _), (t2, _) in zip(series_sorted, series_sorted[1:])
            if (t2 - t1).total_seconds() > 0
        ]
        deltas.sort()
        median_dt = deltas[len(deltas) // 2] if deltas else None

        print(f"updates={len(series_sorted)} runs={len(runs)} median_dt_s={median_dt}")
        print(f"run_lengths: median={median_run} p95={p95_run} max={max_run}")

        top_runs = sorted(runs, key=lambda r: r.count, reverse=True)[:5]
        print("top runs:")
        for r in top_runs:
            dur_s = (r.end_ts - r.start_ts).total_seconds()
            print(
                f"  current={r.current} count={r.count} duration_s={dur_s:.0f} "
                f"start={r.start_ts.isoformat()} end={r.end_ts.isoformat()}"
            )
        print()


if __name__ == "__main__":
    main()
