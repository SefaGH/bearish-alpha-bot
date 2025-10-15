#!/usr/bin/env python3
"""Test Market Data Pipeline integration"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from core.multi_exchange import build_clients_from_env
from core.market_data_pipeline import MarketDataPipeline

async def test_pipeline():
    print("=" * 60)
    print("Testing Market Data Pipeline Integration")
    print("=" * 60)
    
    # Build clients
    print("Building exchange clients...")
    clients = build_clients_from_env(required_symbols=['BTC/USDT:USDT'])
    print(f"✓ Clients: {list(clients.keys())}")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = MarketDataPipeline(clients)
    print("✓ Pipeline initialized")
    
    # Start feeds
    print("\nStarting data feeds...")
    await pipeline.start_feeds_async(['BTC/USDT:USDT'], ['30m', '1h'])
    print("✓ Feeds started")
    
    # Wait for data
    print("\nWaiting for data (10 seconds)...")
    await asyncio.sleep(10)
    
    # Get data
    print("\nFetching data from pipeline...")
    df_30m = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
    df_1h = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '1h')
    
    if df_30m is not None:
        print(f"✓ 30m data: {len(df_30m)} candles")
        print(f"  Latest: {df_30m.index[-1]} - Close: ${df_30m['close'].iloc[-1]:.2f}")
    else:
        print("✗ 30m data not available")
    
    if df_1h is not None:
        print(f"✓ 1h data: {len(df_1h)} candles")
        print(f"  Latest: {df_1h.index[-1]} - Close: ${df_1h['close'].iloc[-1]:.2f}")
    else:
        print("✗ 1h data not available")
    
    # Health check
    print("\nHealth status:")
    health = pipeline.get_health_status()
    print(f"  Overall: {health['overall_status']}")
    print(f"  Active feeds: {health['active_feeds']}")
    print(f"  Error rate: {health['error_rate']}%")
    print(f"  Memory: {health['memory_mb']:.2f} MB")
    
    # Shutdown
    pipeline.shutdown()
    
    print("\n✅ Pipeline integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
