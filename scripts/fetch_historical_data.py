#!/usr/bin/env python3
"""
Fetch historical market data from exchange and save to CSV.
Dedicated script for data ingestion step.
Handles pagination for exchanges with limits (like BingX).
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
    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def timeframe_to_minutes(tf: str) -> int:
    if tf.endswith('m'): return int(tf[:-1])
    if tf.endswith('h'): return int(tf[:-1]) * 60
    if tf.endswith('d'): return int(tf[:-1]) * 1440
    if tf.endswith('w'): return int(tf[:-1]) * 10080
    return 60

async def fetch_data(exchange_id: str, symbol: str, timeframe: str, days: int, output_path: str):
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Fetching {symbol} [{timeframe}] from {exchange_id} ({days} days)")
    
    client = CcxtClient(exchange_id)
    
    # Toplam kaç mum gerektiğini hesapla
    minutes = timeframe_to_minutes(timeframe)
    total_minutes = days * 24 * 60
    # Buffer ekleyerek limiti belirle
    limit = int(total_minutes / minutes) + 100 
    
    try:
        # CcxtClient.fetch_ohlcv_bulk fonksiyonu senkrondur ve pagination (sayfalama) yapar.
        # Bu yüzden event loop'u bloklamaması için executor içinde çalıştırıyoruz.
        # Bu fonksiyon BingX'in 1440 limitini aşmak için istekleri böler.
        loop = asyncio.get_running_loop()
        
        logger.info(f"Requesting {limit} candles via bulk fetch strategy...")
        
        data = await loop.run_in_executor(
            None, 
            lambda: client.fetch_ohlcv_bulk(symbol, timeframe, target_limit=limit)
        )
        
        if not data:
            logger.error("❌ No data received.")
            sys.exit(1)
            
        # List of lists -> DataFrame çevrimi
        # CCXT standart format: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Timestamp formatlama
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Kaydet
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=True)
        
        logger.info(f"✅ Successfully fetched {len(df)} rows.")
        logger.info(f"💾 Saved to: {output_file}")
        
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
    
    asyncio.run(fetch_data(
        args.exchange,
        args.symbol,
        args.timeframe,
        args.days,
        args.output
    ))

if __name__ == "__main__":
    main()
