#!/usr/bin/env python3
"""
Forensic Probe: API Data Integrity Check (VST vs PROD)

Runs a 60s dual-WebSocket capture and compares:
- Kline volume (v) for BTC-USDT@kline_1m
- Ticker 24h volume (v) and quote volume (q) for BTC-USDT@ticker

Usage:
  python scripts/forensic_probe_api_integrity.py
  python scripts/forensic_probe_api_integrity.py --duration 60 --symbol BTC-USDT --timeframe 1m
  python scripts/forensic_probe_api_integrity.py --vst-url wss://vst-open-api-ws.bingx.com/swap-market
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.forensic_probe import run_dual_ws_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--vst-url", default="wss://open-api-vst.bingx.com/swap-market")
    parser.add_argument("--prod-url", default="wss://open-api-swap.bingx.com/swap-market")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    result = run_dual_ws_probe(
        symbol=args.symbol,
        timeframe=args.timeframe,
        duration_s=args.duration,
        vst_url=args.vst_url,
        prod_url=args.prod_url,
        output_dir=args.output_dir,
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

