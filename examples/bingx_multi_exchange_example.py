#!/usr/bin/env python3
"""
BingX ULTIMATE Integration Examples
Demonstrates multi-exchange data fetching, VST contract validation, and unified portfolio management.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient
from core.multi_exchange_manager import MultiExchangeManager
import pandas as pd


def example_bingx_bulk_fetch():
    """Example 1: Simple BingX bulk OHLCV fetch."""
    print("=" * 60)
    print("Example 1: BingX Bulk OHLCV Fetch")
    print("=" * 60)
    
    try:
        client = CcxtClient('bingx')
        
        # Fetch 1000 30-minute candles for BTC
        candles = client.fetch_ohlcv_bulk('BTC/USDT:USDT', '30m', 1000)
        
        print(f"✓ Fetched {len(candles)} candles from BingX")
        print(f"  First timestamp: {candles[0][0]}")
        print(f"  Last timestamp: {candles[-1][0]}")
        print(f"  First close: ${candles[0][4]:.2f}")
        print(f"  Last close: ${candles[-1][4]:.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_vst_contract_validation():
    """Example 2: VST/USDT Contract Validation on BingX."""
    print("=" * 60)
    print("Example 2: VST Contract Validation")
    print("=" * 60)
    
    try:
        manager = MultiExchangeManager()
        
        # Validate VST contract on BingX
        vst_info = manager.validate_vst_contract('bingx')
        
        print(f"Symbol: {vst_info.get('symbol')}")
        print(f"Exchange: {vst_info.get('exchange')}")
        print(f"Available: {vst_info.get('available', 'checking...')}")
        print(f"Contract Type: {vst_info.get('contract_type')}")
        
        if vst_info.get('available'):
            market_info = vst_info.get('market_info', {})
            print(f"\nMarket Details:")
            print(f"  Active: {market_info.get('active')}")
            print(f"  Type: {market_info.get('type')}")
            print(f"  Settle: {market_info.get('settle')}")
            print(f"  Contract Size: {market_info.get('contract_size')}")
        elif vst_info.get('alternative_symbols'):
            print(f"\nAlternative VST symbols found:")
            for sym in vst_info.get('alternative_symbols', []):
                print(f"  - {sym}")
        
        print(f"\n✓ VST validation complete")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_multi_exchange_unified_fetch():
    """Example 3: Unified Data Fetch Across Multiple Exchanges."""
    print("=" * 60)
    print("Example 3: Multi-Exchange Unified Data Fetch")
    print("=" * 60)
    
    try:
        manager = MultiExchangeManager()
        
        # Define portfolio across exchanges
        portfolio = {
            'kucoinfutures': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
            'bingx': ['BTC/USDT:USDT']  # Can add VST/USDT:USDT when available
        }
        
        print(f"Fetching data from {len(portfolio)} exchanges...")
        print(f"Total symbols: {sum(len(syms) for syms in portfolio.values())}")
        
        # Fetch data (using smaller limit for example)
        data = manager.fetch_unified_data(
            portfolio,
            timeframe='30m',
            limit=100
        )
        
        # Display results
        print(f"\n✓ Data fetch complete:")
        for exchange, exchange_data in data.items():
            print(f"\n  {exchange}:")
            for symbol, candles in exchange_data.items():
                if candles:
                    print(f"    {symbol}: {len(candles)} candles")
                    print(f"      Latest close: ${candles[-1][4]:.2f}")
                else:
                    print(f"    {symbol}: No data")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_timestamp_alignment():
    """Example 4: Cross-Exchange Timestamp Alignment."""
    print("=" * 60)
    print("Example 4: Timestamp Alignment")
    print("=" * 60)
    
    try:
        manager = MultiExchangeManager()
        
        # Fetch data from both exchanges
        portfolio = {
            'kucoinfutures': ['BTC/USDT:USDT'],
            'bingx': ['BTC/USDT:USDT']
        }
        
        print("Fetching data from both exchanges...")
        data = manager.fetch_unified_data(portfolio, timeframe='1h', limit=50)
        
        # Count original candles
        original_counts = {}
        for exchange, exchange_data in data.items():
            for symbol, candles in exchange_data.items():
                original_counts[f"{exchange}:{symbol}"] = len(candles)
        
        # Align timestamps
        print("\nAligning timestamps (60s tolerance)...")
        aligned_data = manager.align_timestamps(data, tolerance_ms=60000)
        
        # Count aligned candles
        aligned_counts = {}
        for exchange, exchange_data in aligned_data.items():
            for symbol, candles in exchange_data.items():
                aligned_counts[f"{exchange}:{symbol}"] = len(candles)
        
        # Display results
        print("\n✓ Alignment complete:")
        for key in original_counts.keys():
            original = original_counts[key]
            aligned = aligned_counts.get(key, 0)
            print(f"  {key}:")
            print(f"    Original: {original} candles")
            print(f"    Aligned: {aligned} candles")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_exchange_summary():
    """Example 5: Exchange Status Summary."""
    print("=" * 60)
    print("Example 5: Exchange Status Summary")
    print("=" * 60)
    
    try:
        manager = MultiExchangeManager()
        
        # Get summary
        summary = manager.get_exchange_summary()
        
        print(f"Total Exchanges: {summary['total_exchanges']}")
        print(f"\nExchange Details:")
        
        for name, info in summary['exchanges'].items():
            status = info.get('status', 'unknown')
            print(f"\n  {name}:")
            print(f"    Status: {status}")
            
            if status == 'active':
                markets = info.get('markets', 'N/A')
                print(f"    Markets: {markets}")
            elif status == 'error':
                error = info.get('error', 'Unknown error')
                print(f"    Error: {error[:80]}...")
        
        print(f"\n✓ Summary complete")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_vst_trading_setup():
    """Example 6: Complete VST Trading Setup."""
    print("=" * 60)
    print("Example 6: VST Trading Setup (Test Mode)")
    print("=" * 60)
    
    try:
        # VST Configuration
        VST_CONFIG = {
            'symbol': 'VST/USDT:USDT',
            'exchange': 'bingx',
            'contract_type': 'perpetual',
            'test_mode': True,
            'allocation_pct': 0.1,  # 10% for testing
            'timeframe': '30m',
            'limit': 500
        }
        
        print("VST Trading Configuration:")
        for key, value in VST_CONFIG.items():
            print(f"  {key}: {value}")
        
        # Initialize manager
        manager = MultiExchangeManager()
        
        # Validate contract
        print("\nValidating VST contract...")
        vst_info = manager.validate_vst_contract(VST_CONFIG['exchange'])
        
        if vst_info.get('available'):
            print("✓ VST/USDT contract available")
            
            # Fetch historical data for backtesting
            print(f"\nFetching {VST_CONFIG['limit']} candles...")
            data = manager.fetch_unified_data(
                {VST_CONFIG['exchange']: [VST_CONFIG['symbol']]},
                timeframe=VST_CONFIG['timeframe'],
                limit=VST_CONFIG['limit']
            )
            
            vst_candles = data[VST_CONFIG['exchange']][VST_CONFIG['symbol']]
            
            if vst_candles:
                print(f"✓ Fetched {len(vst_candles)} VST candles")
                
                # Convert to DataFrame for analysis
                df = pd.DataFrame(
                    vst_candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                print(f"\nVST Data Summary:")
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"  Latest close: ${df['close'].iloc[-1]:.4f}")
                print(f"  Average volume: {df['volume'].mean():.2f}")
                print(f"\n✓ VST ready for test trading")
            else:
                print("⚠ No VST data available")
        else:
            print("⚠ VST/USDT not available")
            if vst_info.get('error'):
                print(f"   Error: {vst_info['error'][:80]}...")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("BingX ULTIMATE Integration Examples")
    print("=" * 60)
    print()
    
    examples = [
        example_bingx_bulk_fetch,
        example_vst_contract_validation,
        example_multi_exchange_unified_fetch,
        example_timestamp_alignment,
        example_exchange_summary,
        example_vst_trading_setup,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("Examples Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
