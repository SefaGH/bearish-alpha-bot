#!/usr/bin/env python3
"""
Example: Using KuCoin Futures Bulk OHLCV Fetching

This script demonstrates how to use the new fetch_ohlcv_bulk method
to fetch large amounts of historical data (up to 2000 candles).
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient
import pandas as pd


def example_simple_bulk_fetch():
    """Simple example: Fetch 1000 candles."""
    print("=" * 60)
    print("Example 1: Simple Bulk Fetch (1000 candles)")
    print("=" * 60)
    
    client = CcxtClient('kucoinfutures')
    
    # Fetch 1000 30-minute candles
    candles = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', 1000)
    
    print(f"✓ Fetched {len(candles)} candles")
    print(f"  First timestamp: {candles[0][0]}")
    print(f"  Last timestamp: {candles[-1][0]}")
    print()


def example_max_bulk_fetch():
    """Maximum bulk fetch: 2000 candles."""
    print("=" * 60)
    print("Example 2: Maximum Bulk Fetch (2000 candles)")
    print("=" * 60)
    
    client = CcxtClient('kucoinfutures')
    
    # Fetch maximum 2000 1-hour candles
    candles = client.fetch_ohlcv_bulk('ETH/USDT:USDT', '1h', 2000)
    
    print(f"✓ Fetched {len(candles)} candles")
    print(f"  Time range: ~{len(candles) * 1 / 24:.1f} days")
    print()


def example_with_dataframe():
    """Convert to pandas DataFrame for analysis."""
    print("=" * 60)
    print("Example 3: With Pandas DataFrame")
    print("=" * 60)
    
    client = CcxtClient('kucoinfutures')
    
    # Fetch 1500 4-hour candles
    candles = client.fetch_ohlcv_bulk('BNB/USDT:USDT', '4h', 1500)
    
    # Convert to DataFrame
    df = pd.DataFrame(
        candles, 
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"✓ Fetched {len(df)} candles")
    print(f"  Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  First close price: ${df['close'].iloc[0]:.2f}")
    print(f"  Last close price: ${df['close'].iloc[-1]:.2f}")
    print()


def example_backtest_friendly():
    """Backtest-friendly usage with automatic selection."""
    print("=" * 60)
    print("Example 4: Backtest-Friendly (Automatic Selection)")
    print("=" * 60)
    
    client = CcxtClient('kucoinfutures')
    
    # Small fetch uses regular ohlcv
    small_candles = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', 250)
    print(f"✓ Small fetch (250 candles): {len(small_candles)} candles")
    print("  → Used regular ohlcv method (limit <= 500)")
    print()
    
    # Large fetch uses bulk method
    large_candles = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', 1500)
    print(f"✓ Large fetch (1500 candles): {len(large_candles)} candles")
    print("  → Used bulk method with 3 batches")
    print()


def example_multiple_symbols():
    """Fetch data for multiple symbols."""
    print("=" * 60)
    print("Example 5: Multiple Symbols")
    print("=" * 60)
    
    client = CcxtClient('kucoinfutures')
    
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    for symbol in symbols:
        try:
            candles = client.fetch_ohlcv_bulk(symbol, '1h', 1000)
            print(f"✓ {symbol}: {len(candles)} candles fetched")
        except Exception as e:
            print(f"✗ {symbol}: Failed - {e}")
    print()


def example_server_time_sync():
    """Demonstrate server time synchronization."""
    print("=" * 60)
    print("Example 6: Server Time Synchronization")
    print("=" * 60)
    
    client = CcxtClient('kucoinfutures')
    
    # Get server time
    server_time = client._get_kucoin_server_time()
    print(f"✓ Server time: {server_time}ms")
    print(f"  Cached offset: {client._server_time_offset}ms")
    print(f"  → Used in all bulk fetch requests")
    print()


def example_dynamic_symbols():
    """Demonstrate dynamic symbol discovery."""
    print("=" * 60)
    print("Example 7: Dynamic Symbol Discovery")
    print("=" * 60)
    
    client = CcxtClient('kucoinfutures')
    
    # Get symbol mapping
    symbol_map = client._get_dynamic_symbol_mapping()
    
    print(f"✓ Discovered {len(symbol_map)} active contracts")
    print(f"  Sample mappings:")
    for ccxt_sym, native_sym in list(symbol_map.items())[:5]:
        print(f"    {ccxt_sym} → {native_sym}")
    print()


if __name__ == "__main__":
    print()
    print("KuCoin Futures Bulk OHLCV Fetching Examples")
    print()
    
    examples = [
        example_simple_bulk_fetch,
        example_max_bulk_fetch,
        example_with_dataframe,
        example_backtest_friendly,
        example_multiple_symbols,
        example_server_time_sync,
        example_dynamic_symbols,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"✗ Example failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
