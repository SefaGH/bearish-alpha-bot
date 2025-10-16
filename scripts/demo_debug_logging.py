#!/usr/bin/env python3
"""
Demonstration of enhanced debug logging for signal generation and strategy control.

This script demonstrates the new debug logging features added to:
- universe.py: BingX perpetual filtering
- live_trading_engine.py: Signal scanning and strategy evaluation

Usage:
    python scripts/demo_debug_logging.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from universe import _is_usdt_candidate, build_universe


def demo_universe_logging():
    """Demonstrate universe building with debug logging."""
    print("="*70)
    print("DEMO 1: Universe Symbol Filtering with Debug Logging")
    print("="*70)
    print()
    
    # Test various market types
    test_markets = [
        {
            'symbol': 'BTC/USDT:USDT',
            'active': True,
            'quote': 'USDT',
            'swap': True,
            'linear': True
        },
        {
            'symbol': 'ETH/USDT:USDT',
            'active': True,
            'quote': 'USDT',
            'swap': True,
            'linear': True
        },
        {
            'symbol': 'BTC/USD',
            'active': True,
            'quote': 'USD',  # Should be rejected
            'swap': True,
            'linear': True
        },
        {
            'symbol': 'USDT/USDC',
            'active': True,
            'quote': 'USDC',  # Should be rejected
            'swap': False,
            'spot': True
        },
        {
            'symbol': 'SOL/USDT:USDT',
            'active': False,  # Should be rejected (inactive)
            'quote': 'USDT',
            'swap': True,
            'linear': True
        }
    ]
    
    print("Testing symbol filtering with only_linear=True...")
    print()
    
    accepted = []
    rejected = []
    
    for market in test_markets:
        result = _is_usdt_candidate(market, only_linear=True)
        if result:
            accepted.append(market['symbol'])
        else:
            rejected.append(market['symbol'])
        print()  # Add spacing between symbols
    
    print("="*70)
    print(f"Summary: {len(accepted)} accepted, {len(rejected)} rejected")
    print(f"Accepted: {', '.join(accepted)}")
    print(f"Rejected: {', '.join(rejected)}")
    print()


def demo_strategy_logging_format():
    """Demonstrate the logging format for strategy checks."""
    print("="*70)
    print("DEMO 2: Strategy Check Logging Format")
    print("="*70)
    print()
    
    print("Expected log format during signal scanning:")
    print()
    print("[PROCESSING] Symbol: BTC/USDT:USDT")
    print("[DATA] BTC/USDT:USDT: 30m=200 bars, last_close=67845.50")
    print("[INDICATORS] BTC/USDT:USDT: RSI=45.2, ATR=234.5678")
    print("[STRATEGY-CHECK] adaptive_ob for BTC/USDT:USDT")
    print("✅ [SIGNAL] BTC/USDT:USDT: {'side': 'buy', 'reason': 'RSI oversold', ...}")
    print("   Strategy: adaptive_ob")
    print("   Side: BUY")
    print("   Reason: RSI oversold")
    print()
    print("Or when no signal:")
    print()
    print("[PROCESSING] Symbol: ETH/USDT:USDT")
    print("[DATA] ETH/USDT:USDT: 30m=200 bars, last_close=3421.80")
    print("[INDICATORS] ETH/USDT:USDT: RSI=55.3, ATR=45.2341")
    print("[STRATEGY-CHECK] adaptive_ob for ETH/USDT:USDT")
    print("[NO-SIGNAL] ETH/USDT:USDT (adaptive_ob): RSI=55.3")
    print()


def demo_usage_examples():
    """Show usage examples for running with debug mode."""
    print("="*70)
    print("DEMO 3: Usage Examples")
    print("="*70)
    print()
    
    print("To enable debug logging in live_trading_launcher.py:")
    print()
    print("1. Using --debug flag:")
    print("   python scripts/live_trading_launcher.py --debug --paper")
    print()
    print("2. Using environment variable:")
    print("   export LOG_LEVEL=DEBUG")
    print("   python scripts/live_trading_launcher.py --paper")
    print()
    print("3. For testing signal generation:")
    print("   export LOG_LEVEL=DEBUG")
    print("   python scripts/test_signal_generation.py")
    print()
    print("Expected output will include:")
    print("  - [UNIVERSE] tags for symbol filtering")
    print("  - [PROCESSING] tags for each symbol scanned")
    print("  - [DATA] tags for market data fetching")
    print("  - [INDICATORS] tags for technical indicator values")
    print("  - [STRATEGY-CHECK] tags for strategy evaluation")
    print("  - [SIGNAL] tags for successful signals")
    print("  - [NO-SIGNAL] tags when conditions not met")
    print()


def main():
    """Run all demonstrations."""
    print()
    print("🔍 ENHANCED DEBUG LOGGING DEMONSTRATION")
    print()
    
    try:
        demo_universe_logging()
        print()
        demo_strategy_logging_format()
        print()
        demo_usage_examples()
        
        print("="*70)
        print("✓ Debug logging demonstration complete!")
        print("="*70)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
