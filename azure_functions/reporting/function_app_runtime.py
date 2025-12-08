import json
import os
import logging
import re
import traceback
import hashlib
import time
import uuid
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from io import StringIO

# External Azure imports (Azure Functions ortamında mevcut olacak)
import azure.functions as func  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore
from azure.core.exceptions import ResourceExistsError  # type: ignore
from azure.storage.blob import BlobServiceClient, BlobLeaseClient, ContentSettings  # type: ignore
from azure.mgmt.compute import ComputeManagementClient  # type: ignore
from azure.mgmt.compute.models import RunCommandInput  # type: ignore


# ============================================================================
# CONFIGURATION & VALIDATION
# ============================================================================

REQUIRED_VARS = [
    "bearishstorage_STORAGE",
    "RAW_LOGS_CONTAINER",
    "REPORTS_CONTAINER",
    "AZURE_SUBSCRIPTION_ID",
    "LOGUPLOADER_VM_NAME",
    "LOGUPLOADER_RESOURCE_GROUP",
    "LOGUPLOADER_STORAGE_ACCOUNT",
]


def validate_environment():
    """Startup environment validation with detailed error messages"""
    missing = [var for var in REQUIRED_VARS if not os.environ.get(var)]
    if missing:
        raise ValueError(f"CRITICAL: Missing environment variables: {', '.join(missing)}")

    sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
    try:
        uuid.UUID(sub_id)
    except ValueError:
        raise ValueError(f"Invalid AZURE_SUBSCRIPTION_ID format (expected UUID): {sub_id}")

    storage_acc = os.environ.get("LOGUPLOADER_STORAGE_ACCOUNT", "")
    if not re.match(r"^[a-z0-9]{3,24}$", storage_acc):
        raise ValueError(f"Invalid storage account name: {storage_acc}")


try:
    validate_environment()
    logging.info("✅ Environment validation passed")
except Exception as e:
    logging.critical(f"❌ Environment validation failed: {e}")

# Environment variables
STORAGE_CONNECTION = os.environ.get("bearishstorage_STORAGE")
RAW_LOGS_CONTAINER = os.environ.get("RAW_LOGS_CONTAINER", "raw-logs")
REPORTS_CONTAINER = os.environ.get("REPORTS_CONTAINER", "reports")
LOGUPLOADER_DEFAULT_VM = os.environ.get("LOGUPLOADER_VM_NAME")
LOGUPLOADER_DEFAULT_RG = os.environ.get("LOGUPLOADER_RESOURCE_GROUP")
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
STORAGE_ACCOUNT = os.environ.get("LOGUPLOADER_STORAGE_ACCOUNT")

# Operational settings
VM_LOG_DIR = os.environ.get("LOGUPLOADER_VM_LOG_DIR", "/mnt/bearish/logs")
LOG_FILE_PATTERN = os.environ.get("LOGUPLOADER_FILE_GLOB", "live_trading_*.log")
STORAGE_API_VERSION = os.environ.get("LOGUPLOADER_STORAGE_API_VERSION", "2021-08-06")
RUN_COMMAND_TIMEOUT = int(os.environ.get("LOGUPLOADER_TIMEOUT_SECONDS", "180"))
MAX_LOG_SIZE_BYTES = int(os.environ.get("MAX_LOG_SIZE_MB", "10")) * 1024 * 1024
LOCK_TIMEOUT_SECONDS = int(os.environ.get("LOCK_TIMEOUT_SECONDS", "300"))

LOGGER = logging.getLogger(__name__)


# ============================================================================
# SECURE BASH SCRIPT (VM tarafında son logu Storage'a atan script)
# ============================================================================

def generate_secure_bash_script() -> str:
    log_dir = VM_LOG_DIR.replace('"', '\\"')
    pattern = LOG_FILE_PATTERN.replace('"', '\\"')
    storage_acc = (STORAGE_ACCOUNT or "").replace('"', '\\"')
    container = RAW_LOGS_CONTAINER.replace('"', '\\"')
    api_ver = STORAGE_API_VERSION.replace('"', '\\"')

    return f"""#!/bin/bash
set -euo pipefail
LOG_DIR="{log_dir}"
PATTERN="{pattern}"
STORAGE_ACCOUNT="{storage_acc}"
CONTAINER="{container}"
API_VERSION="{api_ver}"

log_json() {{ echo "$1" >&2; echo "$1"; }}

LATEST_LOG=$(find "$LOG_DIR" -maxdepth 1 -name "$PATTERN" -type f -printf '%T@ %p\\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$LATEST_LOG" ]; then log_json '{{"status":"error","message":"No log files found"}}'; exit 1; fi

FILENAME=$(basename "$LATEST_LOG")
CONTENT_LENGTH=$(stat -c%s "$LATEST_LOG" 2>/dev/null || echo "0")
if [ "$CONTENT_LENGTH" -eq 0 ]; then log_json '{{"status":"error","message":"Empty log file"}}'; exit 1; fi

for i in $(seq 1 3); do
    TOKEN=$(curl -s --max-time 10 -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://storage.azure.com/" 2>/dev/null | grep -oP '(?<="access_token":")[^"]+' || echo "")
    if [ -n "$TOKEN" ]; then break; fi
    if [ $i -lt 3 ]; then sleep 2; fi
done

if [ -z "$TOKEN" ]; then log_json '{{"status":"error","message":"Token acquisition failed"}}'; exit 1; fi

BLOB_URL="https://$STORAGE_ACCOUNT.blob.core.windows.net/$CONTAINER/$FILENAME"
HTTP_CODE=$(curl -X PUT "$BLOB_URL" -H "Authorization: Bearer $TOKEN" -H "x-ms-blob-type: BlockBlob" -H "x-ms-version: $API_VERSION" -H "Content-Length: $CONTENT_LENGTH" --data-binary "@$LATEST_LOG" --max-time 120 -s -o /dev/null -w "%{{http_code}}")

if [ "$HTTP_CODE" = "201" ] || [ "$HTTP_CODE" = "409" ]; then
    log_json '{{"status":"success","file":"'"$FILENAME"'","size":'"$CONTENT_LENGTH"'}}'
else
    log_json '{{"status":"error","http_code":"'"$HTTP_CODE"'"}}'
    exit 1
fi
"""


BASH_TEMPLATE = generate_secure_bash_script()


# ============================================================================
# DISTRIBUTED LOCK (şu an kullanılmıyor ama ileride concurrency için hazır)
# ============================================================================

class DistributedLock:
    def __init__(self, blob_service_client: BlobServiceClient, lock_name: str):
        self.container_name = "system-locks"
        self.blob_name = f"{lock_name}.lock"
        self.blob_service = blob_service_client
        self.lease_client: Optional[BlobLeaseClient] = None
        self.acquired = False

    def __enter__(self):
        try:
            container_client = self.blob_service.get_container_client(self.container_name)
            try:
                container_client.create_container()
            except ResourceExistsError:
                pass

            blob_client = container_client.get_blob_client(self.blob_name)
            try:
                blob_client.upload_blob(b"lock", overwrite=False)
            except ResourceExistsError:
                pass

            self.lease_client = BlobLeaseClient(blob_client)
            self.lease_client.acquire(timeout=LOCK_TIMEOUT_SECONDS)
            self.acquired = True
            LOGGER.info("🔒 Lock acquired: %s", self.blob_name)
            return self
        except Exception as e:
            LOGGER.warning("⚠️ Failed to acquire lock: %s", e)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired and self.lease_client:
            try:
                self.lease_client.release()
                LOGGER.info("🔓 Lock released: %s", self.blob_name)
            except Exception as e:
                LOGGER.error("Error releasing lock: %s", e)


# ============================================================================
# ENHANCED LOG ANALYSIS - SINGLE PASS WITH RICH METRICS
# ============================================================================

class EnhancedLogAnalyzer:
    """Deep analysis with strategy breakdown, timeline, and actionable insights"""

    def __init__(self):
        self.stats = {
            "iterations": 0,
            "failures": defaultdict(int),
            "strategies": {
                "adaptive_ob": {"attempts": 0, "signals": 0},
                "adaptive_str": {"attempts": 0, "signals": 0},
                "ppo": {"signals": 0, "confidence_scores": [], "above_threshold": 0},
            },
            "rsi_values": [],
            "volume_data": [],
            "regimes": [],
            "trades": [],
            "timestamps": [],
            "signal_timeline": [],  # When did signals occur
        }

        # Compiled patterns for performance
        self.patterns = {
            "iteration": re.compile(r"\[ITERATION (\d+)\]"),
            "timestamp": re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"),
            "volume_fail": re.compile(r"Volume confirmation failed"),
            "rsi_fail": re.compile(r"RSI .* is below the threshold"),
            "rip_fail": re.compile(r"Rip Check Failed"),
            "ppo_weak": re.compile(r"Ignored weak PPO signal.*Conf: (\d+\.\d+)"),
            "ppo_strong": re.compile(r"PPO signal accepted.*Conf: (\d+\.\d+)"),
            "adaptive_ob": re.compile(r"\[ADAPTIVE_OB/([^\]]+)\]"),
            "adaptive_str": re.compile(r"\[ADAPTIVE_STR/([^\]]+)\]"),
            "ppo_monitor": re.compile(r"PPO-MONITOR.*Conf: (\d+\.\d+)"),
            "rsi_value": re.compile(r"RSI \((\d+\.?\d*)\)"),
            "volume": re.compile(r"current=(\d+), avg=(\d+)"),
            "regime": re.compile(r"Prediction: (\w+) \(confidence: (\d+\.\d+)\)"),
            "trade_closed": re.compile(r"TRADE_CLOSED\s+(\{.*?\})"),
            "signal_generated": re.compile(r"SIGNAL GENERATED.*side: (\w+)"),
        }

    def analyze_stream(self, content: str, max_lines: int = 50000) -> Dict[str, Any]:
        """Enhanced single-pass analysis with rich metrics"""
        lines_processed = 0
        current_hour_counts: Dict[int, int] = defaultdict(int)

        for line in content.split("\n"):
            if lines_processed >= max_lines:
                LOGGER.warning("⚠️ Line limit reached (%s)", max_lines)
                break
            lines_processed += 1

            # Timestamp extraction for timeline
            ts_match = self.patterns["timestamp"].search(line)
            if ts_match:
                ts_str = ts_match.group(1)
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                        self.stats["timestamps"].append(dt)
                        current_hour_counts[dt.hour] += 1
                        break
                    except ValueError:
                        continue

            # Iteration counting
            if self.patterns["iteration"].search(line):
                self.stats["iterations"] += 1

            # Failure tracking with context
            if self.patterns["volume_fail"].search(line):
                self.stats["failures"]["volume_confirmation"] += 1
            if self.patterns["rsi_fail"].search(line):
                self.stats["failures"]["rsi_threshold"] += 1
            if self.patterns["rip_fail"].search(line):
                self.stats["failures"]["rip_check"] += 1

            # PPO analysis (weak signals)
            ppo_weak = self.patterns["ppo_weak"].search(line)
            if ppo_weak:
                conf = float(ppo_weak.group(1))
                self.stats["failures"]["ppo_confidence"] += 1
                self.stats["strategies"]["ppo"]["confidence_scores"].append(conf)

            # PPO analysis (strong signals)
            ppo_strong = self.patterns["ppo_strong"].search(line)
            if ppo_strong:
                conf = float(ppo_strong.group(1))
                self.stats["strategies"]["ppo"]["signals"] += 1
                self.stats["strategies"]["ppo"]["confidence_scores"].append(conf)
                if conf >= 0.75:
                    self.stats["strategies"]["ppo"]["above_threshold"] += 1

            # Strategy attempts
            ob_match = self.patterns["adaptive_ob"].search(line)
            if ob_match:
                self.stats["strategies"]["adaptive_ob"]["attempts"] += 1
                if "signal" in line.lower():
                    self.stats["strategies"]["adaptive_ob"]["signals"] += 1

            str_match = self.patterns["adaptive_str"].search(line)
            if str_match:
                self.stats["strategies"]["adaptive_str"]["attempts"] += 1
                if "signal" in line.lower():
                    self.stats["strategies"]["adaptive_str"]["signals"] += 1

            # PPO monitoring
            ppo_mon = self.patterns["ppo_monitor"].search(line)
            if ppo_mon:
                conf = float(ppo_mon.group(1))
                self.stats["strategies"]["ppo"]["confidence_scores"].append(conf)

            # Market data sampling
            rsi_match = self.patterns["rsi_value"].search(line)
            if rsi_match and len(self.stats["rsi_values"]) < 1000:
                self.stats["rsi_values"].append(float(rsi_match.group(1)))

            vol_match = self.patterns["volume"].search(line)
            if vol_match and len(self.stats["volume_data"]) < 1000:
                self.stats["volume_data"].append(
                    (int(vol_match.group(1)), int(vol_match.group(2)))
                )

            regime_match = self.patterns["regime"].search(line)
            if regime_match:
                self.stats["regimes"].append(regime_match.group(1))

            # Signal generation tracking
            signal_match = self.patterns["signal_generated"].search(line)
            if signal_match and ts_match:
                self.stats["signal_timeline"].append(
                    {"time": ts_match.group(1), "side": signal_match.group(1)}
                )

            # Trade tracking
            trade_match = self.patterns["trade_closed"].search(line)
            if trade_match:
                try:
                    trade_data = json.loads(trade_match.group(1))
                    self.stats["trades"].append(trade_data)
                except json.JSONDecodeError:
                    pass

        return self._compile_enhanced_results(current_hour_counts)

    def _compile_enhanced_results(self, hour_counts: Dict[int, int]) -> Dict[str, Any]:
        """Compile with enhanced metrics and insights"""
        total_iter = self.stats["iterations"]
        ppo_scores = self.stats["strategies"]["ppo"]["confidence_scores"]

        # PPO confidence stats
        ppo_avg_conf = sum(ppo_scores) / len(ppo_scores) if ppo_scores else 0.0
        ppo_total_signals = len(ppo_scores)

        # RSI statistics
        rsi_values = self.stats["rsi_values"]
        rsi_stats = {
            "avg": (sum(rsi_values) / len(rsi_values)) if rsi_values else 0.0,
            "min": min(rsi_values) if rsi_values else 0.0,
            "max": max(rsi_values) if rsi_values else 0.0,
            "samples": len(rsi_values),
        }

        # Volume statistics
        vol_data = self.stats["volume_data"]
        if vol_data:
            current_sum = sum(v[0] for v in vol_data)
            baseline_sum = sum(v[1] for v in vol_data)
            current_avg = current_sum / len(vol_data)
            baseline_avg = baseline_sum / len(vol_data)
            ratio = (current_sum / baseline_sum) if baseline_sum > 0 else 0.0
        else:
            current_avg = baseline_avg = ratio = 0.0

        vol_stats = {
            "current_avg": current_avg,
            "baseline_avg": baseline_avg,
            "ratio": ratio,
        }

        # Market regime analysis
        regimes = self.stats["regimes"]
        regime_counter = Counter(regimes)
        dominant_regime = regime_counter.most_common(1)[0] if regime_counter else (
            "unknown",
            0,
        )

        # Timeline analysis
        peak_hour = (
            max(hour_counts.items(), key=lambda x: x[1]) if hour_counts else (None, 0)
        )

        timestamps = sorted(self.stats["timestamps"])
        if timestamps:
            start_ts = timestamps[0]
            end_ts = timestamps[-1]
            dur_sec = (end_ts - start_ts).total_seconds()
        else:
            start_ts = end_ts = None
            dur_sec = 0

        return {
            "total_iterations": total_iter,
            "failure_categories": dict(self.stats["failures"]),
            "strategy_performance": {
                "adaptive_ob": {
                    "attempts": self.stats["strategies"]["adaptive_ob"]["attempts"],
                    "signals": self.stats["strategies"]["adaptive_ob"]["signals"],
                    "success_rate": (
                        self.stats["strategies"]["adaptive_ob"]["signals"]
                        / self.stats["strategies"]["adaptive_ob"]["attempts"]
                        * 100.0
                    )
                    if self.stats["strategies"]["adaptive_ob"]["attempts"] > 0
                    else 0.0,
                },
                "adaptive_str": {
                    "attempts": self.stats["strategies"]["adaptive_str"]["attempts"],
                    "signals": self.stats["strategies"]["adaptive_str"]["signals"],
                    "success_rate": (
                        self.stats["strategies"]["adaptive_str"]["signals"]
                        / self.stats["strategies"]["adaptive_str"]["attempts"]
                        * 100.0
                    )
                    if self.stats["strategies"]["adaptive_str"]["attempts"] > 0
                    else 0.0,
                },
                "ppo": {
                    "total_signals": ppo_total_signals,
                    "above_threshold": self.stats["strategies"]["ppo"][
                        "above_threshold"
                    ],
                    "below_threshold": ppo_total_signals
                    - self.stats["strategies"]["ppo"]["above_threshold"],
                    "avg_confidence": round(ppo_avg_conf, 2),
                    "threshold_pass_rate": (
                        self.stats["strategies"]["ppo"]["above_threshold"]
                        / ppo_total_signals
                        * 100.0
                    )
                    if ppo_total_signals > 0
                    else 0.0,
                },
            },
            "market_conditions": {
                "rsi_stats": rsi_stats,
                "volume_stats": vol_stats,
                "regime_distribution": dict(regime_counter),
                "dominant_regime": dominant_regime[0],
                "regime_confidence": (dominant_regime[1] / len(regimes) * 100.0)
                if regimes
                else 0.0,
            },
            "timeline_analysis": {
                "peak_hour": peak_hour[0],
                "peak_hour_count": peak_hour[1],
                "total_hours_active": len(hour_counts),
                "signal_count": len(self.stats["signal_timeline"]),
                "run_start_utc": start_ts.isoformat().replace("+00:00", "Z") if start_ts else None,
                "run_end_utc": end_ts.isoformat().replace("+00:00", "Z") if end_ts else None,
                "run_duration_seconds": int(dur_sec),
                "run_duration_minutes": round(dur_sec / 60, 1) if dur_sec else 0,
            },
            "has_trades": len(self.stats["trades"]) > 0,
            "trades": self.stats["trades"],
        }


# ============================================================================
# ENHANCED HTML REPORT WITH RICH VISUALS
# ============================================================================

class EnhancedHTMLReportGenerator:
    """Production-grade HTML report with charts and insights"""

    @staticmethod
    def escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def generate(
        self, base: Dict[str, Any], sig: Dict[str, Any], recs: List[Dict[str, Any]]
    ) -> str:
        output = StringIO()

        run_id = self.escape_html(str(base.get("run_id", "Unknown")))
        report_filename = self.escape_html(str(base.get("report_filename", "Unknown")))
        message = self.escape_html(str(base.get("message", "")))

        # Extract enhanced metrics
        strat_perf = sig.get("strategy_performance", {})
        market = sig.get("market_conditions", {})
        timeline = sig.get("timeline_analysis", {})
        run_window = {
            "start": timeline.get("run_start_utc"),
            "end": timeline.get("run_end_utc"),
            "duration": timeline.get("run_duration_minutes"),
        }

        output.write(
            f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Bearish Bot Raporu - {run_id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #1a202c;
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1100px; 
            margin: 0 auto; 
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{ 
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white; 
            padding: 30px;
            border-bottom: 4px solid #3b82f6;
        }}
        .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 12px; }}
        .header .meta {{ 
            font-size: 13px; 
            opacity: 0.85; 
            font-family: 'Courier New', monospace;
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
        }}
        .content {{ padding: 30px; }}
        .card {{ 
            border: 2px solid #e5e7eb; 
            padding: 24px; 
            margin-bottom: 24px; 
            border-radius: 12px;
            background: #fafafa;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .card.success {{ background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-color: #22c55e; }}
        .card.warning {{ background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-color: #f59e0b; }}
        .card.info {{ background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-color: #3b82f6; }}
        .card.danger {{ background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-color: #ef4444; }}
        .card h3 {{ 
            margin-bottom: 18px; 
            font-size: 20px; 
            color: #1e293b;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}
        .stat-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
            gap: 16px; 
            margin-top: 18px; 
        }}
        .stat-box {{ 
            background: white; 
            padding: 18px; 
            border-radius: 10px; 
            text-align: center;
            border: 2px solid #e5e7eb;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }}
        .stat-box:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
        .stat-val {{ 
            font-size: 32px; 
            font-weight: 800; 
            color: #1e293b; 
            display: block;
            margin-bottom: 6px;
        }}
        .stat-lbl {{ 
            font-size: 12px; 
            color: #64748b; 
            text-transform: uppercase; 
            letter-spacing: 0.8px;
            font-weight: 600;
        }}
        .stat-sub {{ 
            font-size: 11px; 
            color: #94a3b8; 
            margin-top: 4px;
        }}
        .progress-bar {{
            width: 100%;
            height: 24px;
            background: #e5e7eb;
            border-radius: 12px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 11px;
            font-weight: 700;
        }}
        .tag {{ 
            display: inline-block;
            padding: 5px 14px; 
            border-radius: 20px; 
            font-size: 11px; 
            font-weight: 700; 
            color: white;
            margin-right: 8px;
            margin-bottom: 8px;
        }}
        .tag-HIGH {{ background: #dc2626; }} 
        .tag-MEDIUM {{ background: #d97706; }} 
        .tag-LOW {{ background: #16a34a; }}
        .recommendation {{ 
            margin: 15px 0; 
            padding: 18px; 
            background: white;
            border-radius: 10px;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            margin: 8px 0;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }}
        .metric-label {{ font-weight: 600; color: #475569; }}
        .metric-value {{ font-size: 18px; font-weight: 700; color: #1e293b; }}
        .section-divider {{
            height: 2px;
            background: linear-gradient(90deg, transparent 0%, #e5e7eb 50%, transparent 100%);
            margin: 24px 0;
        }}
        .footer {{ 
            text-align: center; 
            padding: 24px; 
            background: #f9fafb; 
            border-top: 2px solid #e5e7eb;
            font-size: 12px;
            color: #64748b;
        }}
        .insight-box {{
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            border: 2px solid #fb923c;
            border-radius: 10px;
            padding: 16px;
            margin: 16px 0;
        }}
        .insight-box h4 {{
            color: #ea580c;
            margin-bottom: 8px;
            font-size: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }}
        th {{
            background: #f1f5f9; 
            padding: 12px;
            text-align: left;
            font-weight: 700;
            color: #475569;
            border-bottom: 2px solid #cbd5e1;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:hover {{
            background: #f8fafc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Bearish Alpha Bot - Detaylı Analiz Raporu</h1>
            <div class="meta">
                <strong>Run ID:</strong> {run_id}<br>
                <strong>Rapor Dosyası:</strong> {report_filename}<br>
                <strong>Oluşturulma:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
                <strong>İşlem Süresi (analiz):</strong> {base.get('processing_time_ms', 0)}ms<br>
                {f"<strong>Run Penceresi (log):</strong> {run_window['start']} -> {run_window['end']} (~{run_window['duration']} dk)<br>" if run_window['start'] and run_window['end'] else ''}
            </div>
        </div>
        
        <div class="content">
            <!-- EXECUTIVE SUMMARY -->
            <div class="card {'success' if base.get('trade_count', 0) > 0 else 'warning'}">
                <h3>📊 Yönetici Özeti</h3>
                <p style="margin-bottom: 18px; font-size: 15px;">{message}</p>
                <div class="stat-grid">
                    <div class="stat-box">
                        <span class="stat-val">{base.get('trade_count', 0)}</span>
                        <span class="stat-lbl">Tamamlanan İşlem</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val" style="color: {'#16a34a' if base.get('total_pnl', 0) > 0 else '#dc2626'};">{base.get('total_pnl', 0):.2f}</span>
                        <span class="stat-lbl">Toplam PnL (USDT)</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val">{base.get('win_rate', 0):.1f}%</span>
                        <span class="stat-lbl">Başarı Oranı</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val">{sig.get('total_iterations', 0)}</span>
                        <span class="stat-lbl">Analiz Döngüsü</span>
                    </div>
                </div>
            </div>
"""
        )

        # STRATEGY PERFORMANCE BREAKDOWN
        output.write(
            f"""
            <div class="card info">
                <h3>🎯 Strateji Performans Analizi</h3>
                
                <h4 style="margin-top: 16px; color: #475569;">Adaptive Orderbook Strategy</h4>
                <div class="metric-row">
                    <span class="metric-label">Toplam Deneme</span>
                    <span class="metric-value">{strat_perf.get('adaptive_ob', {}).get('attempts', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Üretilen Sinyal</span>
                    <span class="metric-value">{strat_perf.get('adaptive_ob', {}).get('signals', 0)}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {strat_perf.get('adaptive_ob', {}).get('success_rate', 0):.1f}%">
                        Başarı: %{strat_perf.get('adaptive_ob', {}).get('success_rate', 0):.1f}
                    </div>
                </div>
                
                <div class="section-divider"></div>
                
                <h4 style="color: #475569;">Adaptive Structure Strategy</h4>
                <div class="metric-row">
                    <span class="metric-label">Toplam Deneme</span>
                    <span class="metric-value">{strat_perf.get('adaptive_str', {}).get('attempts', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Üretilen Sinyal</span>
                    <span class="metric-value">{strat_perf.get('adaptive_str', {}).get('signals', 0)}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {strat_perf.get('adaptive_str', {}).get('success_rate', 0):.1f}%">
                        Başarı: %{strat_perf.get('adaptive_str', {}).get('success_rate', 0):.1f}
                    </div>
                </div>
                
                <div class="section-divider"></div>
                
                <h4 style="color: #475569;">PPO ML Engine</h4>
                <div class="stat-grid">
                    <div class="stat-box">
                        <span class="stat-val">{strat_perf.get('ppo', {}).get('total_signals', 0)}</span>
                        <span class="stat-lbl">Toplam Sinyal</span>
                        <div class="stat-sub">ML analizi</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val" style="color: #16a34a;">{strat_perf.get('ppo', {}).get('above_threshold', 0)}</span>
                        <span class="stat-lbl">Güçlü Sinyal</span>
                        <div class="stat-sub">&gt;0.75 güven</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val" style="color: #dc2626;">{strat_perf.get('ppo', {}).get('below_threshold', 0)}</span>
                        <span class="stat-lbl">Zayıf Sinyal</span>
                        <div class="stat-sub">&lt;0.75 güven</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val">{strat_perf.get('ppo', {}).get('avg_confidence', 0):.2f}</span>
                        <span class="stat-lbl">Ort. Güven Skoru</span>
                        <div class="stat-sub">0-1 arası</div>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {strat_perf.get('ppo', {}).get('threshold_pass_rate', 0):.1f}%; background: linear-gradient(90deg, #16a34a 0%, #15803d 100%);">
                        Güçlü Sinyal Oranı: %{strat_perf.get('ppo', {}).get('threshold_pass_rate', 0):.1f}
                    </div>
                </div>
            </div>
"""
        )

        # MARKET CONDITIONS
        rsi_stats = market.get("rsi_stats", {})
        vol_stats = market.get("volume_stats", {})
        regime_dist = market.get("regime_distribution", {})

        output.write(
            f"""
            <div class="card info">
                <h3>🌍 Piyasa Koşulları ve Rejim Analizi</h3>
                
                <div class="stat-grid">
                    <div class="stat-box">
                        <span class="stat-val">{rsi_stats.get('avg', 0):.1f}</span>
                        <span class="stat-lbl">Ortalama RSI</span>
                        <div class="stat-sub">Min: {rsi_stats.get('min', 0):.1f} | Max: {rsi_stats.get('max', 0):.1f}</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val">{vol_stats.get('ratio', 0):.2f}x</span>
                        <span class="stat-lbl">Hacim Çarpanı</span>
                        <div class="stat-sub">Güncel/Baseline</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val">{market.get('dominant_regime', 'N/A').upper()}</span>
                        <span class="stat-lbl">Baskın Rejim</span>
                        <div class="stat-sub">%{market.get('regime_confidence', 0):.1f} güven</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val">{timeline.get('total_hours_active', 0)}</span>
                        <span class="stat-lbl">Aktif Saat</span>
                        <div class="stat-sub">Peak: {timeline.get('peak_hour', 'N/A')}:00</div>
                    </div>
                </div>
                
                <div class="insight-box">
                    <h4>💡 Market Rejim Dağılımı</h4>
                    <table>
                        <tr>
                            <th>Rejim</th>
                            <th>Gözlem Sayısı</th>
                            <th>Oran</th>
                        </tr>
"""
        )

        total_regimes = sum(regime_dist.values()) or 0
        for regime, count in sorted(
            regime_dist.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (count / total_regimes * 100.0) if total_regimes > 0 else 0.0
            output.write(
                f"""
                        <tr>
                            <td><strong>{self.escape_html(regime.upper())}</strong></td>
                            <td>{count}</td>
                            <td>
                                <div class="progress-bar" style="height: 18px;">
                                    <div class="progress-fill" style="width: {pct:.1f}%; font-size: 10px;">
                                        %{pct:.1f}
                                    </div>
                                </div>
                            </td>
                        </tr>
"""
            )

        output.write(
            """
                    </table>
                </div>
            </div>
"""
        )

        # FAILURE ANALYSIS
        failures = sig.get("failure_categories", {})
        total_failures = sum(failures.values())

        output.write(
            f"""
            <div class="card danger">
                <h3>🚫 Sinyal Engelleri ve Başarısızlık Analizi</h3>
                <p style="margin-bottom: 16px; color: #64748b;">
                    Toplam <strong>{total_failures}</strong> sinyal çeşitli nedenlerle engellenmiştir.
                </p>
                
                <table>
                    <tr>
                        <th>Engel Tipi</th>
                        <th>Frekans</th>
                        <th>Oran</th>
                        <th>Etki</th>
                    </tr>
"""
        )

        failure_labels = {
            "volume_confirmation": ("Hacim Onayı", "Yetersiz işlem hacmi"),
            "rsi_threshold": ("RSI Eşiği", "RSI değeri kritik seviyenin altında"),
            "rip_check": ("Rip Kontrolü", "Ani fiyat hareketleri tespit edildi"),
            "ppo_confidence": ("PPO Güveni", "ML modeli yetersiz güven seviyesi"),
        }

        for fail_type, count in sorted(
            failures.items(), key=lambda x: x[1], reverse=True
        ):
            if count == 0:
                continue
            label, desc = failure_labels.get(
                fail_type, (fail_type.replace("_", " ").title(), "Bilinmeyen")
            )
            pct = (count / total_failures * 100.0) if total_failures > 0 else 0.0
            impact = "🔴 Kritik" if pct > 40 else "🟡 Orta" if pct > 20 else "🟢 Düşük"

            output.write(
                f"""
                    <tr>
                        <td>
                            <strong>{self.escape_html(label)}</strong><br>
                            <small style="color: #64748b;">{self.escape_html(desc)}</small>
                        </td>
                        <td><span class="stat-val" style="font-size: 18px;">{count}</span></td>
                        <td>
                            <div class="progress-bar" style="height: 20px;">
                                <div class="progress-fill" style="width: {pct:.1f}%; background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);">
                                    %{pct:.1f}
                                </div>
                            </div>
                        </td>
                        <td>{impact}</td>
                    </tr>
"""
            )

        output.write(
            """
                </table>
            </div>
"""
        )

        # RECOMMENDATIONS
        if recs:
            output.write(
                """
            <div class="card warning">
                <h3>💡 Parametre Optimizasyon Önerileri</h3>
                <p style="margin-bottom: 16px; color: #64748b;">
                    Aşağıdaki öneriler performans analizi sonucunda otomatik olarak oluşturulmuştur.
                </p>
"""
            )
            for r in recs:
                urgency = self.escape_html(str(r.get("urgency", "LOW")))
                param = self.escape_html(str(r.get("parameter", "")))
                current = self.escape_html(str(r.get("current", "")))
                suggested = self.escape_html(str(r.get("suggested", "")))
                reason = self.escape_html(str(r.get("reason", "")))
                impact = self.escape_html(str(r.get("expected_impact", "")))

                output.write(
                    f"""
                <div class="recommendation">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div>
                            <span class="tag tag-{urgency}">{urgency} ÖNCELİK</span>
                            <strong style="font-size: 16px;">{param}</strong>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #64748b;">{current}</span>
                            <span style="margin: 0 8px;">→</span>
                            <strong style="color: #16a34a; font-size: 18px;">{suggested}</strong>
                        </div>
                    </div>
                    <div style="background: #f8fafc; padding: 10px; border-radius: 6px; margin-top: 8px;">
                        <strong style="color: #475569;">Neden:</strong> {reason}<br>
                        <strong style="color: #475569;">Beklenen Etki:</strong> {impact}
                    </div>
                </div>
"""
                )
            output.write("</div>")

        # FOOTER
        output.write(
            f"""
        </div>
        
        <div class="footer">
            <strong>Bearish Alpha Bot Analytics Engine v11</strong><br>
            Powered by Azure Functions | Generated in {base.get('processing_time_ms', 0)}ms<br>
            <small style="margin-top: 8px; display: block; color: #94a3b8;">
                Bu rapor otomatik olarak oluşturulmuştur. Yatırım tavsiyesi değildir.
            </small>
        </div>
    </div>
</body>
</html>
"""
        )

        return output.getvalue()


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class OptimizationRecommender:
    """Generate enhanced recommendations"""

    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        fails = analysis.get("failure_categories", {})
        total_iter = analysis.get("total_iterations", 0)

        if total_iter <= 0:
            return recs

        # Volume confirmation very restrictive
        if fails.get("volume_confirmation", 0) / total_iter > 0.5:
            recs.append(
                {
                    "parameter": "volume_factor",
                    "current": 1.5,
                    "suggested": 1.2,
                    "reason": f"Hacim şartı {fails['volume_confirmation']}/{total_iter} kez sağlanamadı",
                    "urgency": "HIGH",
                    "expected_impact": "Sinyal üretimini %30-40 artırabilir",
                }
            )

        # RSI threshold too strict
        if fails.get("rsi_threshold", 0) / total_iter > 0.6:
            recs.append(
                {
                    "parameter": "RSI_THRESHOLD_BTC",
                    "current": 50,
                    "suggested": 45,
                    "reason": f"RSI eşiği {fails['rsi_threshold']}/{total_iter} kez aşılamadı",
                    "urgency": "MEDIUM",
                    "expected_impact": "Short sinyallerini artırır",
                }
            )

        # PPO confidence gate
        ppo = analysis.get("strategy_performance", {}).get("ppo", {})
        if ppo.get("total_signals", 0) > 0:
            threshold_ratio = ppo.get("above_threshold", 0) / ppo.get(
                "total_signals", 1
            )
            if threshold_ratio < 0.3:
                recs.append(
                    {
                        "parameter": "PPO_CONFIDENCE_THRESHOLD",
                        "current": 0.75,
                        "suggested": 0.65,
                        "reason": f"PPO sinyallerinin %{(1-threshold_ratio)*100:.0f}'i güven eşiğinin altında",
                        "urgency": "LOW",
                        "expected_impact": "ML katılımını artırır",
                    }
                )

        return recs


# ============================================================================
# MAIN ANALYSIS LOGIC
# ============================================================================

def analyze_trading_logs(
    content: str,
    filename: str = "",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Enhanced analysis with rich reporting (run_id + runtime + HTML)."""
    stats: Dict[str, Any] = {
        "status": "no_data",
        "trade_count": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "message": "Log içeriği boş veya analiz edilemedi.",
        "run_id": run_id or "UNKNOWN",
        "report_filename": "UNKNOWN",
        "events_count": 0,
        "signal_analysis": {},
        "optimization_recommendations": [],
        "has_trades": False,
        "html_report": "",
        "processing_time_ms": 0,
        "log_filename": filename or "",
    }

    start_time = time.time()

    if not content or not content.strip():
        LOGGER.warning("Empty log content")
        stats["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return stats

    try:
        if stats["run_id"] == "UNKNOWN":
            run_id_match = re.search(r"runId[\"':\s]+([a-zA-Z0-9\-_]+)", content)
            if run_id_match:
                stats["run_id"] = run_id_match.group(1)

        match = re.search(r"live_trading_(\d{8}_\d{6}_\d+)\.log", filename or "")
        if match:
            file_id = match.group(1)
            if stats["run_id"] == "UNKNOWN":
                stats["run_id"] = file_id
            stats["report_filename"] = f"report_{file_id}.html"
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            stats["report_filename"] = f"report_{ts}.html"

        summary_match = re.search(r"Total Exits:\s*(\d+)", content)
        if summary_match and int(summary_match.group(1)) > 0:
            stats["trade_count"] = int(summary_match.group(1))

            pnl_match = re.search(r"Total P&L:\s*\S+\s*([+\-]?\d+\.\d+)", content)
            if pnl_match:
                stats["total_pnl"] = float(pnl_match.group(1))

            wr_match = re.search(r"Win Rate:\s*(\d+\.\d+)%", content)
            if wr_match:
                stats["win_rate"] = float(wr_match.group(1))

            stats["status"] = "success"
            stats["has_trades"] = True

        analyzer = EnhancedLogAnalyzer()
        sig_analysis = analyzer.analyze_stream(content)
        stats["signal_analysis"] = sig_analysis
        stats["events_count"] = sig_analysis.get("total_iterations", 0)

        if not stats["has_trades"] and sig_analysis.get("trades"):
            trades = sig_analysis["trades"]
            pnl_sum = sum(float(t.get("pnl_usd", 0)) for t in trades)
            wins = sum(1 for t in trades if float(t.get("pnl_usd", 0)) > 0)

            stats["trade_count"] = len(trades)
            stats["total_pnl"] = round(pnl_sum, 4)
            stats["win_rate"] = round((wins / len(trades)) * 100, 2) if trades else 0.0

            stats["status"] = "success"
            stats["has_trades"] = True

        recs: List[Dict[str, Any]] = []
        if not stats["has_trades"]:
            optimizer = OptimizationRecommender()
            recs = optimizer.generate_recommendations(sig_analysis)
            stats["optimization_recommendations"] = recs

            if sig_analysis.get("total_iterations", 0) > 0:
                stats["status"] = "signals_only"
                stats["message"] = "Bot çalıştı ancak işlem kriterleri sağlanamadı."
        else:
            stats["message"] = (
                f"✅ {stats['trade_count']} işlem tamamlandı. "
                f"PnL: {stats['total_pnl']:.2f} USDT"
            )

        stats["processing_time_ms"] = int((time.time() - start_time) * 1000)

        html_gen = EnhancedHTMLReportGenerator()
        stats["html_report"] = html_gen.generate(stats, sig_analysis, recs)

        return stats

    except Exception as e:
        LOGGER.error("Analysis error: %s\n%s", str(e), traceback.format_exc())
        stats["status"] = "error"
        stats["message"] = f"Analiz hatası: {str(e)}"
        stats["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return stats


# ============================================================================
# BLOB HELPERS FOR REPORTING
# ============================================================================

def _get_blob_service_client(
    credential: Optional[DefaultAzureCredential] = None,
) -> BlobServiceClient:
    """
    Storage Account'a Managed Identity ile bağlanmak için ortak client.
    """
    if not STORAGE_ACCOUNT:
        raise RuntimeError("LOGUPLOADER_STORAGE_ACCOUNT env değişkeni boş.")

    if credential is None:
        credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)

    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=credential)


def _fetch_latest_log_data(
    credential: DefaultAzureCredential,
) -> Tuple[Optional[str], Optional[str]]:
    """
    RAW_LOGS_CONTAINER içindeki en son log blob'unu (isim + içerik) döner.
    Hiç blob yoksa (None, None) döner.
    """
    service = _get_blob_service_client(credential)
    container = service.get_container_client(RAW_LOGS_CONTAINER)

    blobs = list(container.list_blobs())
    if not blobs:
        LOGGER.warning(
            "No blobs found in raw logs container '%s'", RAW_LOGS_CONTAINER
        )
        return None, None

    latest = max(blobs, key=lambda b: b.last_modified)
    LOGGER.info("Using latest log blob for analysis: %s", latest.name)

    data = container.get_blob_client(latest.name).download_blob().readall()
    try:
        content = data.decode("utf-8", errors="replace")
    except Exception:
        content = data.decode("latin-1", errors="replace")

    return content, latest.name


def _upload_report_blobs(
    credential: DefaultAzureCredential,
    base_id: str,
    analysis: Dict[str, Any],
) -> Dict[str, str]:
    """
    Analiz sonucunu hem JSON hem HTML olarak REPORTS_CONTAINER'a yazar.
    İsimler:
      - HTML: report_{base_id}.html
      - JSON: report_{base_id}.json
    """
    service = _get_blob_service_client(credential)
    container = service.get_container_client(REPORTS_CONTAINER)

    # Container yoksa oluştur (idempotent)
    if not container.exists():
        container.create_container()

    html_name = analysis.get("report_filename") or f"report_{base_id}.html"
    json_name = f"report_{base_id}.json"

    html_client = container.get_blob_client(html_name)
    json_client = container.get_blob_client(json_name)

    html_content = analysis.get("html_report", "") or ""

    # HTML rapor
    html_client.upload_blob(
        html_content.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(
            content_type="text/html; charset=utf-8"
        ),
    )

    # JSON rapor (ham analiz objesi)
    json_client.upload_blob(
        json.dumps(analysis, ensure_ascii=False, indent=2).encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

    base_url = (
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{REPORTS_CONTAINER}"
    )
    urls = {
        "html_blob": html_name,
        "json_blob": json_name,
        "html_url": f"{base_url}/{html_name}",
        "json_url": f"{base_url}/{json_name}",
    }

    LOGGER.info("Report uploaded: %s", urls)
    return urls


# ============================================================================
# MAIN ENTRY: run_report_logic (for /run_report HTTP trigger)
# ============================================================================

def run_report_logic(req: func.HttpRequest) -> func.HttpResponse:
    """
    Logic App'ten tetiklenen ana analiz fonksiyonu.

    Giriş mantığı:
      - Body içinde log_content + filename gelmişse → direkt onu analiz eder.
      - Aksi halde → RAW_LOGS_CONTAINER içindeki en son log'u indirir.
    Çıkış:
      - JSON body: analiz sonucu + HTML/JSON rapor blob bilgileri
      - HTTP Status:
          200 → Trade var (PnL hesaplanmış)
          202 → Trade yok ama sinyal/iteration var
          202 → Hiç veri yok (log bulunamadı)
          500 → Analiz hatası
    """
    LOGGER.info("Run Report Logic (v11) triggered")

    try:
        body = req.get_json()
    except ValueError:
        body = {}

    # Logic App body'den gelebilecek ek bilgiler
    run_id_from_req = body.get("run_id")
    inline_log_content = body.get("log_content")
    inline_filename = body.get("filename") or body.get("file_name") or ""

    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)

    # 1) Log içeriğini belirle
    if inline_log_content:
        log_content = inline_log_content
        filename = inline_filename or "inline.log"
        LOGGER.info("Using inline log content from request body.")
    else:
        log_content, blob_name = _fetch_latest_log_data(credential)
        filename = blob_name or "unknown.log"

    # 2) Log yoksa 202 NO DATA
    if not log_content:
        result = {
            "status": "no_data",
            "message": "Hiç log bulunamadı veya içerik boş.",
            "run_id": run_id_from_req or None,
            "file_name": filename,
        }
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            status_code=202,
            mimetype="application/json",
        )

    try:
        # 3) Asıl analiz
        analysis = analyze_trading_logs(log_content, filename, run_id=run_id_from_req)

        # Dışarıdan gelen run_id varsa ve analiz içinde boşsa set et
        if run_id_from_req and (
            not analysis.get("run_id") or analysis.get("run_id") == "UNKNOWN"
        ):
            analysis["run_id"] = run_id_from_req

        # Base ID: önce file_id, yoksa run_id, o da yoksa timestamp
        base_id = (
            analysis.get("file_id")
            or analysis.get("run_id")
            or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
        analysis["report_base_id"] = base_id

        # 4) Raporları blob'a yükle
        urls = _upload_report_blobs(
            credential=credential,
            base_id=str(base_id),
            analysis=analysis,
        )

        # HTML içeriği response'dan atalım (payload şişmesin)
        # analysis["html_report"] = None

        analysis.update(
            {
                "report_html_blob": urls["html_blob"],
                "report_json_blob": urls["json_blob"],
                "report_html_url": urls["html_url"],
                "report_json_url": urls["json_url"],
                # Logic App geriye dönük uyumluluk için:
                "report_url": urls["html_url"],
            }
        )

        # 5) HTTP status kodu belirle
        if analysis.get("status") == "error":
            http_status = 500
        elif (analysis.get("trade_count") or 0) > 0:
            http_status = 200
            analysis["status"] = analysis.get("status") or "trades"
        elif (analysis.get("events_count") or 0) > 0:
            http_status = 202
            if analysis.get("status") == "no_data":
                analysis["status"] = "signals_only"
        else:
            http_status = 202
            if analysis.get("status") == "no_data":
                analysis["message"] = (
                    analysis.get("message")
                    or "Analiz yapılacak veri bulunamadı."
                )

        return func.HttpResponse(
            json.dumps(analysis, ensure_ascii=False),
            status_code=http_status,
            mimetype="application/json",
        )
    except Exception as exc:
        LOGGER.error("run_report_logic failed: %s", exc, exc_info=True)
        error_body = {
            "status": "error",
            "message": str(exc),
            "run_id": run_id_from_req or None,
            "file_name": filename,
        }
        return func.HttpResponse(
            json.dumps(error_body, ensure_ascii=False),
            status_code=500,
            mimetype="application/json",
        )


# ============================================================================
# VM LOG UPLOADER HTTP ENTRY (for /loguploader HTTP trigger)
# ============================================================================

def log_uploader_http(req: func.HttpRequest) -> func.HttpResponse:
    """
    Logic App tarafından çağrılan VM log uploader endpoint'i.
    VM üzerinde secure bash script (BASH_TEMPLATE) ile son log dosyasını bulup
    RAW_LOGS_CONTAINER'a upload eder.
    """
    try:
        body = req.get_json()
    except ValueError:
        body = {}

    vm_name = body.get("vmName") or req.params.get("vmName") or LOGUPLOADER_DEFAULT_VM
    resource_group = (
        body.get("resourceGroup") or req.params.get("resourceGroup") or LOGUPLOADER_DEFAULT_RG
    )

    if not vm_name or not resource_group:
        return func.HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "message": "vmName ve resourceGroup zorunludur.",
                },
                ensure_ascii=False,
            ),
            status_code=400,
            mimetype="application/json",
        )

    try:
        command_result = _invoke_vm_log_sync(vm_name, resource_group)
        vm_payload = _parse_vm_output(command_result) or {}
        status = vm_payload.get("status", "error")
        http_code = 200 if status == "success" else 500
        response_body = {
            "status": status,
            "vm": vm_name,
            "resourceGroup": resource_group,
            "vmPayload": vm_payload,
        }
        return func.HttpResponse(
            json.dumps(response_body, ensure_ascii=False),
            status_code=http_code,
            mimetype="application/json",
        )
    except Exception as exc:
        LOGGER.exception("LogUploader HTTP trigger failed")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False),
            status_code=500,
            mimetype="application/json",
        )


def _invoke_vm_log_sync(vm_name: str, resource_group: str):
    """
    Azure VM üzerinde RunCommand ile BASH_TEMPLATE script'ini çalıştırır.
    """
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    compute_client = ComputeManagementClient(credential, SUBSCRIPTION_ID or "")
    command_input = RunCommandInput(command_id="RunShellScript", script=[BASH_TEMPLATE])
    poller = compute_client.virtual_machines.begin_run_command(
        resource_group_name=resource_group,
        vm_name=vm_name,
        parameters=command_input,
    )
    LOGGER.info("Triggered RunCommand on %s/%s", resource_group, vm_name)
    return poller.result(timeout=RUN_COMMAND_TIMEOUT)


def _parse_vm_output(result) -> Dict[str, Any]:
    """
    RunCommand sonucundan stdout içindeki JSON payload'ı parse eder.
    """
    try:
        if not result.value:
            return {"status": "error", "message": "VM RunCommand returned no output"}
        message = result.value[0].message or ""
    except Exception:
        return {"status": "error", "message": "Invalid VM RunCommand result structure"}

    if "[stdout]" in message and "[stderr]" in message:
        stdout_section = message.split("[stdout]", 1)[1].split("[stderr]", 1)[0].strip()
        stdout_section = stdout_section.replace("Enable succeeded:", "", 1).strip()
    else:
        stdout_section = message

    try:
        return json.loads(stdout_section)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "Failed to parse VM output",
            "rawOutput": stdout_section[:500],
        }