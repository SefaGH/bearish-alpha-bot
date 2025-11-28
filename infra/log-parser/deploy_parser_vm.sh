#!/bin/bash
set -euo pipefail

# Install dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-pip
pip3 install pyyaml python-dateutil

# Create directories
mkdir -p /opt/bearish/parser
mkdir -p /mnt/bearish/logs
mkdir -p /mnt/bearish/data/parsed
chown -R azureuser:azureuser /mnt/bearish/data/parsed

# Write parser.py
cat > /opt/bearish/parser/parser.py <<'PYTHON_EOF'
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dateutil import parser as date_parser
import yaml

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - \[(?P<logger>[^\]]+)\] - (?P<level>\w+) - (?P<message>.+)$"
)


@dataclass
class Config:
    input_glob: str
    output_dir: Path
    poll_interval: int

    @staticmethod
    def load(path: Path) -> "Config":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return Config(
            input_glob=data.get("input_glob", "/input/logs/live_trading_*.log"),
            output_dir=Path(data.get("output_dir", "/output/ndjson")),
            poll_interval=int(data.get("poll_interval", 60)),
        )


def parse_line(line: str) -> dict | None:
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None

    groups = match.groupdict()
    timestamp = date_parser.parse(groups["timestamp"]).astimezone(timezone.utc)

    event = {
        "timestamp_utc": timestamp.isoformat(),
        "logger": groups["logger"],
        "level": groups["level"],
        "message": groups["message"],
        "event_type": infer_event_type(groups["message"]),
        "extra": {},
    }

    enrich_event(event)
    return event


def infer_event_type(message: str) -> str:
    lowered = message.lower()
    if "signal generated" in lowered:
        return "signal_generated"
    if "order executed" in lowered or "order filled" in lowered:
        return "trade_entry"
    if "position closed" in lowered or "exit summary" in lowered:
        return "trade_exit"
    if "shutdown" in lowered:
        return "shutdown"
    if "exception" in lowered or "error" in lowered:
        return "exception"
    if "watchdog" in lowered:
        return "health_ping"
    return "log"


def enrich_event(event: dict) -> None:
    message = event["message"]
    symbol_match = re.search(r"([A-Z]+/[A-Z]+:[A-Z]+)", message)
    if symbol_match:
        event["symbol"] = symbol_match.group(1)

    pnl_match = re.search(r"P&L: \$([\-\d.]+)", message)
    if pnl_match:
        event["pnl_usd"] = float(pnl_match.group(1))


def generate_run_id(path: Path) -> str:
    stem = path.stem
    return stem.replace("live_trading_", "")


def iter_new_lines(files: Iterable[Path]) -> Iterable[tuple[Path, str]]:
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    yield file_path, line
        except FileNotFoundError:
            continue


def write_events(events: list[dict], output_dir: Path, run_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_id}.ndjson"
    with output_file.open("w", encoding="utf-8") as handle:
        for event in events:
            event["run_id"] = run_id
            handle.write(json.dumps(event) + "\n")


def main(config: Config) -> int:
    # Handle glob manually since Path.glob doesn't support absolute paths well in all versions/contexts
    # But here we use glob module or Path.glob
    # The original code used Path("/").glob(config.input_glob.lstrip("/"))
    # We will stick to that
    input_files = [Path(p) for p in Path("/").glob(config.input_glob.lstrip("/"))]
    
    while True:
        events_by_run: dict[str, list[dict]] = {}

        for file_path, line in iter_new_lines(input_files):
            event = parse_line(line)
            if not event:
                continue

            run_id = generate_run_id(file_path)
            events_by_run.setdefault(run_id, []).append(event)

        for run_id, events in events_by_run.items():
            write_events(events, config.output_dir, run_id)

        time.sleep(config.poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bearish Alpha log parser")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = Config.load(Path(args.config))
    # os.chdir("/")  # ensure absolute globs work
    raise SystemExit(main(config))
PYTHON_EOF

# Write config.yaml
cat > /opt/bearish/parser/config.yaml <<'YAML_EOF'
input_glob: "/mnt/bearish/logs/live_trading_*.log"
output_dir: "/mnt/bearish/data/parsed"
poll_interval: 60
YAML_EOF

# Create systemd service
cat > /etc/systemd/system/bearish-parser.service <<'SERVICE_EOF'
[Unit]
Description=Bearish Alpha Log Parser
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bearish/parser
ExecStart=/usr/bin/python3 /opt/bearish/parser/parser.py --config config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Start service
systemctl daemon-reload
systemctl enable bearish-parser
systemctl restart bearish-parser
