import argparse
import json
import re
import sys
from datetime import datetime, timezone


def extract_json(line: str):
    m = re.search(r"(\{.*?\})", line)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def ts_to_dt(ts: str | None):
    if not ts:
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def iter_lines(log_file: str | None):
    if not log_file or log_file == "-":
        yield from sys.stdin
        return

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        yield from f


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Exit when the next collector_state last_closed_ot pivot happens."
    )
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tfs", default="1h,4h")
    ap.add_argument(
        "log_file",
        nargs="?",
        help="Path to log file (defaults to stdin). Use '-' to force stdin.",
    )
    ap.add_argument(
        "--log-file",
        dest="log_file_flag",
        help="Path to log file (same as positional arg).",
    )
    args = ap.parse_args(argv)

    tfs = {x.strip() for x in args.tfs.split(",") if x.strip()}
    log_file = args.log_file_flag or args.log_file

    last: dict[str, object] = {}

    for line in iter_lines(log_file):
        obj = extract_json(line)
        if not obj:
            continue
        if obj.get("event") != "collector_state":
            continue
        if obj.get("symbol") != args.symbol:
            continue
        tf = obj.get("timeframe")
        if tf not in tfs:
            continue

        ts = ts_to_dt(obj.get("ts"))
        last_closed_ot = obj.get("last_closed_ot")

        if tf not in last:
            last[tf] = last_closed_ot
            print(f"[WATCH] init tf={tf} last_closed_ot={last_closed_ot} at {ts}")
            continue

        if last_closed_ot != last[tf]:
            print(f"[PIVOT] tf={tf} {last[tf]} -> {last_closed_ot} at {ts}")
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
