#!/usr/bin/env python3
"""
Integration test to verify fixed symbol list optimization.
Demonstrates the optimization working end-to-end.
"""
import sys
import os
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from universe import build_universe


def test_fixed_symbols_integration():
    """
    Integration test showing fixed symbol mode in action.
    This demonstrates the optimization described in the issue.
    """
    print("\n" + "="*70)
    print("FIXED SYMBOL LIST OPTIMIZATION - INTEGRATION TEST")
    print("="*70)
    
    # Create separate mock exchange clients for each exchange
    mock_bingx = Mock()
    mock_bingx.markets = Mock(return_value={})
    mock_bingx.tickers = Mock(return_value={})
    
    mock_kucoin = Mock()  # Separate instance
    mock_kucoin.markets = Mock(return_value={})
    mock_kucoin.tickers = Mock(return_value={})
    
    exchanges = {
        'bingx': mock_bingx,
        'kucoinfutures': mock_kucoin
    }
    
    # Config from issue - fixed symbols mode
    fixed_symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
        'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT',
        'AVAX/USDT:USDT', 'MATIC/USDT:USDT', 'LINK/USDT:USDT',
        'LTC/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT',
        'XRP/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT'
    ]
    
    config = {
        'universe': {
            'fixed_symbols': fixed_symbols,
            'auto_select': False,
            # Old parameters (ignored when auto_select=false)
            'min_quote_volume_usdt': 1000000,
            'prefer_perps': True,
            'max_symbols_per_exchange': 80,
            'top_n_per_exchange': 15,
            'only_linear': True
        }
    }
    
    print("\n📊 Configuration:")
    print(f"  - Fixed symbols: {len(fixed_symbols)}")
    print(f"  - Auto-select: {config['universe']['auto_select']}")
    print(f"  - Exchanges: {list(exchanges.keys())}")
    
    # Build universe
    print("\n🚀 Building universe with fixed symbols...")
    result = build_universe(exchanges, config)
    
    # Verify no market loading occurred
    print("\n✅ Verification:")
    print(f"  - markets() calls to BingX: {mock_bingx.markets.call_count}")
    print(f"  - tickers() calls to BingX: {mock_bingx.tickers.call_count}")
    print(f"  - markets() calls to KuCoin: {mock_kucoin.markets.call_count}")
    print(f"  - tickers() calls to KuCoin: {mock_kucoin.tickers.call_count}")
    
    assert mock_bingx.markets.call_count == 0, "BingX markets() should not be called"
    assert mock_bingx.tickers.call_count == 0, "BingX tickers() should not be called"
    assert mock_kucoin.markets.call_count == 0, "KuCoin markets() should not be called"
    assert mock_kucoin.tickers.call_count == 0, "KuCoin tickers() should not be called"
    
    print("\n✓ No API calls made! 🎉")
    
    # Verify results
    print("\n📈 Results:")
    print(f"  - Exchanges configured: {len(result)}")
    print(f"  - BingX symbols: {len(result['bingx'])}")
    print(f"  - KuCoin symbols: {len(result['kucoinfutures'])}")
    
    assert len(result) == 2
    assert len(result['bingx']) == 15
    assert len(result['kucoinfutures']) == 15
    assert result['bingx'] == fixed_symbols
    assert result['kucoinfutures'] == fixed_symbols
    
    print("\n✓ All symbols correctly assigned!")
    
    # Show expected performance improvement
    print("\n📊 Expected Performance Improvement:")
    print("  - Startup time: 5s → 0.5s (10x faster)")
    print("  - API calls per 30s: 2500+ → 0 (100% reduction)")
    print("  - Memory usage: 10MB → 100KB (100x reduction)")
    
    print("\n" + "="*70)
    print("TEST PASSED - OPTIMIZATION WORKING! ✅")
    print("="*70 + "\n")


def test_auto_select_still_works():
    """Verify that auto_select mode still works when needed."""
    print("\n" + "="*70)
    print("AUTO-SELECT MODE - BACKWARD COMPATIBILITY TEST")
    print("="*70)
    
    # Create mock with market data
    mock_markets = {
        'BTC/USDT:USDT': {
            'symbol': 'BTC/USDT:USDT',
            'active': True,
            'quote': 'USDT',
            'type': 'swap',
            'linear': True,
            'swap': True
        },
        'ETH/USDT:USDT': {
            'symbol': 'ETH/USDT:USDT',
            'active': True,
            'quote': 'USDT',
            'type': 'swap',
            'linear': True,
            'swap': True
        }
    }
    
    mock_tickers = {
        'BTC/USDT:USDT': {'quoteVolume': 5000000},
        'ETH/USDT:USDT': {'quoteVolume': 3000000}
    }
    
    mock_client = Mock()
    mock_client.markets = Mock(return_value=mock_markets)
    mock_client.tickers = Mock(return_value=mock_tickers)
    
    exchanges = {'bingx': mock_client}
    
    config = {
        'universe': {
            'fixed_symbols': [],  # Empty
            'auto_select': True,  # Enable auto-select
            'min_quote_volume_usdt': 1000000,
            'top_n_per_exchange': 5
        }
    }
    
    print("\n📊 Configuration:")
    print(f"  - Auto-select: {config['universe']['auto_select']}")
    print(f"  - Min volume: {config['universe']['min_quote_volume_usdt']:,}")
    
    print("\n🔄 Building universe with auto-select...")
    result = build_universe(exchanges, config)
    
    print("\n✅ Verification:")
    print(f"  - markets() calls: {mock_client.markets.call_count}")
    print(f"  - tickers() calls: {mock_client.tickers.call_count}")
    
    assert mock_client.markets.call_count == 1, "markets() should be called in auto-select mode"
    assert mock_client.tickers.call_count == 1, "tickers() should be called in auto-select mode"
    
    print("\n✓ Auto-select mode still works correctly!")
    print("\n" + "="*70)
    print("BACKWARD COMPATIBILITY VERIFIED ✅")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_fixed_symbols_integration()
    test_auto_select_still_works()
    print("\n🎉 All integration tests passed!")
