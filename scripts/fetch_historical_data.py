#!/usr/bin/env python3
"""
Fetch historical market data from exchange and save to CSV.
Dedicated script for data ingestion step.
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
import pandas as pd

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.ccxt_client import CcxtClient

def configure_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - [%(levelname)s] - %(message)s')

def timeframe_to_minutes(tf: str) -> int:
    if tf.endswith('m'): return int(tf[:-1])
    if tf.endswith('h'): return int(tf[:-1]) * 60
    if tf.endswith('d'): return int(tf[:-1]) * 1440
    return 60

async def fetch_data(exchange_id: str, symbol: str, timeframe: str, days: int, output_path: str):
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Fetching {symbol} [{timeframe}] from {exchange_id} ({days} days)")
    
    client = CcxtClient(exchange_id)
    minutes = timeframe_to_minutes(timeframe)
    limit = int((days * 24 * 60) / minutes) + 100
    
    try:
        df = await client.ohlcv(symbol, timeframe=timeframe, limit=limit)
        if df is None or df.empty:
            logger.error("❌ No data received.")
            sys.exit(1)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=True)
        logger.info(f"💾 Saved {len(df)} rows to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Fetch error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    asyncio.run(fetch_data(args.exchange, args.symbol, args.timeframe, args.days, args.output))

if __name__ == "__main__":
    main()
