#!/usr/bin/env python3
"""Analyze GEMMA log files for quick health indicators."""
from __future__ import annotations

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Analyze GEMMA log output")
parser.add_argument("log_file", help="Path to log file to analyze")
args = parser.parse_args()

log_path = Path(args.log_file)
if not log_path.exists():
    raise SystemExit(f"Log file not found: {log_path}")

stats = {
    "gemma_active": False,
    "feature_82": 0,
    "fallback_warnings": 0,
    "trades": 0,
    "errors": 0,
    "ml_predictions": 0,
}

with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
    for line in handle:
        lower = line.lower()
        if "gemma-2.0.0" in line or "feature_count=82" in lower:
            stats["gemma_active"] = True
        if "feature" in lower and "82" in lower:
            stats["feature_82"] += 1
        if "fallback" in lower:
            if "'fallback':" not in lower:
                stats["fallback_warnings"] += 1
        if "[stage:executed]" in lower:
            stats["trades"] += 1
        if " - error - " in lower or "[error]" in lower:
            stats["errors"] += 1
        if (
            "ml prediction" in lower
            or "gemma prediction" in lower
            or ("prediction refreshed" in lower and "price-engine" in lower)
        ):
            stats["ml_predictions"] += 1

print("\n" + "=" * 50)
print("📊 GEMMA LOG ANALYSIS")
print("=" * 50)
print(f"✅ GEMMA Active: {'YES' if stats['gemma_active'] else 'NO'}")
print(f"📈 82-Feature References: {stats['feature_82']}")
print(f"⚠️  Fallback Warnings: {stats['fallback_warnings']}")
print(f"💰 Trades Executed: {stats['trades']}")
print(f"🧠 ML Predictions: {stats['ml_predictions']}")
print(f"❌ Errors: {stats['errors']}")
print("=" * 50)

if stats['gemma_active'] and stats['fallback_warnings'] == 0:
    print("🎉 GEMMA IS FULLY OPERATIONAL!")
elif stats['gemma_active']:
    print("⚠️  GEMMA is active but some components in fallback mode")
else:
    print("❌ GEMMA is not active - check configuration")