import argparse
import json
import re
import sys
from datetime import datetime, timezone

EVENTS = {"collector_state", "gap_detected", "backfill_result"}


def extract_json(line: str):
    m = re.search(r'(\{.*?"event"\s*:\s*".*?".*?\})', line)
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
    parser = argparse.ArgumentParser(
        description="Print a health summary from a Bearish Alpha Bot log."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        help="Path to log file (defaults to stdin). Use '-' to force stdin.",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file_flag",
        help="Path to log file (same as positional arg).",
    )
    args = parser.parse_args(argv)

    log_file = args.log_file_flag or args.log_file

    for line in iter_lines(log_file):
        obj = extract_json(line)
        if not obj:
            continue
        ev = obj.get("event")
        if ev not in EVENTS:
            continue

        if ev == "collector_state":
            ts = ts_to_dt(obj.get("ts"))
            tf = obj.get("timeframe")
            sym = obj.get("symbol")
            conn = obj.get("connected")
            subs = obj.get("subs")
            msgs = obj.get("ws_messages")
            gaps = obj.get("gap_count")
            ooo = obj.get("out_of_order_drops")
            last_ot = obj.get("last_closed_ot")
            forming_ot = obj.get("forming_ot")

            tss = ts.strftime("%H:%M:%S") if ts else "na"
            print(
                f"[HEALTH] {tss} {sym} tf={tf} conn={conn} subs={subs} ws_msgs={msgs} "
                f"last_closed_ot={last_ot} forming_ot={forming_ot} gaps={gaps} ooo={ooo}"
            )
        else:
            print(f"[EVENT] {ev} {obj}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
