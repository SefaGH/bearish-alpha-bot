# Log Parser Service Requirements

This document captures the expected behavior and schema for the Azure reporting parser described in issue #430.

## Input
- Source directory: `/mnt/bearish/logs` (mounted from host)
- File pattern: `live_trading_*.log` (plain text; may be gzip when archived)
- Ingestion cadence: tail or periodic batch (minimum every 60 s)

## Processing
- Extract individual log entries and map to structured JSON events.
- Determine `run_id` per session: prefer `RUN_ID` field in logs; fallback to filename stem (`live_trading_<timestamp>_<nonce>`).
- Normalize timestamps to UTC ISO-8601 (`YYYY-MM-DDTHH:MM:SS.sssZ`).
- Capture P&L metrics, order lifecycle, strategy metadata, and shutdown markers.
- Attach raw message under `message` for debugging.

## Output
- Write newline-delimited JSON files (`*.ndjson`) to `/mnt/bearish/data/parsed/<run_id>.ndjson`.
- Each line conforms to the schema below and is self-contained.
- Ensure file rotation (truncate or create new files) to prevent unbounded growth.

## Event Schema (`bearish_events`)
| Field | Type | Notes |
| --- | --- | --- |
| `run_id` | string | Session identifier |
| `timestamp_utc` | datetime | Event timestamp (UTC) |
| `event_type` | string | Enum: `run_start`, `signal_generated`, `trade_entry`, `trade_exit`, `shutdown`, `exception`, etc. |
| `logger` | string | Logger name from source log |
| `level` | string | Severity level |
| `message` | string | Original log line |
| `symbol` | string? | Trading pair if applicable |
| `entry_price` | float? | For trade entries |
| `exit_price` | float? | For trade exits |
| `pnl_usd` | float? | Profit/loss for trade exit |
| `holding_time_s` | integer? | Seconds between entry/exit |
| `strategy` | string? | Strategy identifier |
| `ml_confidence` | float? | GEMMA/PPO confidence where present |
| `rl_confidence` | float? | PPO confidence |
| `signal_score` | float? | Additional scoring metrics |
| `extra` | object | Arbitrary properties (JSON) |

## Operational Notes
- Parser container runs with read-only access to logs directory; write access only to parsed output.
- Apply backoff when source log is locked or incomplete.
- Emit heartbeat events (`event_type=health_ping`) to confirm liveness.
- Parser should log metrics (processed lines, dropped lines) to stdout for Container Insights.

## TODO
- Collect representative `live_trading_*.log` sample to finalize regex patterns.
- Define severity mapping for `event_type = exception` and reconciliation with alerts.
- Add unit tests using recorded log fragments.
