#!/usr/bin/env python3
"""
Example: Using Market Data Pipeline Mode

This example demonstrates how to use the optimized pipeline mode
for continuous market monitoring and signal generation.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from core.multi_exchange import build_clients_from_env
from core.market_data_pipeline import MarketDataPipeline


async def simple_pipeline_example():
    """
    Simple example of using the Market Data Pipeline.
    
    This demonstrates:
    1. Initializing the pipeline with exchange clients
    2. Starting async data feeds
    3. Retrieving cached data
    4. Monitoring health status
    """
    print("=" * 60)
    print("Market Data Pipeline Example")
    print("=" * 60)
    
    # Step 1: Build exchange clients
    print("\n1. Building exchange clients...")
    try:
        clients = build_clients_from_env(required_symbols=['BTC/USDT:USDT'])
        print(f"   ✓ Connected to: {list(clients.keys())}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   ℹ Make sure EXCHANGES environment variable is set")
        return
    
    # Step 2: Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = MarketDataPipeline(clients)
    print("   ✓ Pipeline initialized")
    
    # Step 3: Start data feeds (async)
    print("\n3. Starting data feeds...")
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    timeframes = ['30m', '1h', '4h']
    
    results = await pipeline.start_feeds_async(symbols, timeframes)
    print(f"   ✓ Successful fetches: {results['successful_fetches']}")
    print(f"   ✓ Failed fetches: {results['failed_fetches']}")
    print(f"   ✓ Exchanges used: {results['exchanges_used']}")
    
    # Step 4: Wait for data to be fully loaded
    print("\n4. Waiting for data to be fully loaded...")
    await asyncio.sleep(5)
    print("   ✓ Data loaded")
    
    # Step 5: Retrieve cached data
    print("\n5. Retrieving cached data...")
    for symbol in symbols:
        print(f"\n   {symbol}:")
        for timeframe in timeframes:
            df = pipeline.get_latest_ohlcv(symbol, timeframe)
            if df is not None:
                last_price = df['close'].iloc[-1]
                candle_count = len(df)
                print(f"      {timeframe}: {candle_count} candles, last price: ${last_price:.2f}")
            else:
                print(f"      {timeframe}: No data")
    
    # Step 6: Check health status
    print("\n6. Health status:")
    health = pipeline.get_health_status()
    print(f"   Overall status: {health['overall_status']}")
    print(f"   Active feeds: {health['active_feeds']}")
    print(f"   Error rate: {health['error_rate']}%")
    print(f"   Memory usage: {health['memory_mb']:.2f} MB")
    
    # Step 7: Detailed pipeline status
    print("\n7. Detailed pipeline status:")
    status = pipeline.get_pipeline_status()
    print(f"   Total requests: {status['total_requests']}")
    print(f"   Failed requests: {status['failed_requests']}")
    print(f"   Uptime: {status['uptime_seconds']:.0f} seconds")
    print(f"   Data freshness:")
    print(f"      Fresh: {status['data_freshness']['fresh']}")
    print(f"      Stale: {status['data_freshness']['stale']}")
    print(f"      Expired: {status['data_freshness']['expired']}")
    
    # Step 8: Shutdown
    print("\n8. Shutting down pipeline...")
    pipeline.shutdown()
    print("   ✓ Pipeline shutdown complete")
    
    print("\n" + "=" * 60)
    print("✅ Example complete!")
    print("=" * 60)


async def continuous_monitoring_example():
    """
    Example of continuous monitoring with the pipeline.
    
    This demonstrates a simplified version of the main bot's
    run_with_pipeline() function.
    """
    print("=" * 60)
    print("Continuous Monitoring Example (5 iterations)")
    print("=" * 60)
    
    # Initialize
    clients = build_clients_from_env(required_symbols=['BTC/USDT:USDT'])
    pipeline = MarketDataPipeline(clients)
    
    symbols = ['BTC/USDT:USDT']
    timeframes = ['30m', '1h']
    
    await pipeline.start_feeds_async(symbols, timeframes)
    print("✓ Pipeline started\n")
    
    # Run for 5 iterations
    for i in range(1, 6):
        print(f"Iteration {i}:")
        
        # Get latest data
        for symbol in symbols:
            df_30m = pipeline.get_latest_ohlcv(symbol, '30m')
            if df_30m is not None:
                price = df_30m['close'].iloc[-1]
                print(f"  {symbol} (30m): ${price:.2f}")
        
        # Check health
        health = pipeline.get_health_status()
        print(f"  Health: {health['overall_status']} (error rate: {health['error_rate']}%)")
        
        # Wait before next iteration
        print("  Waiting 3 seconds...\n")
        await asyncio.sleep(3)
    
    pipeline.shutdown()
    print("✅ Continuous monitoring example complete!")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        # Run continuous monitoring example
        asyncio.run(continuous_monitoring_example())
    else:
        # Run simple example
        asyncio.run(simple_pipeline_example())


if __name__ == "__main__":
    main()
