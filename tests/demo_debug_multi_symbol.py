#!/usr/bin/env python3
"""
Demonstration script for multi-symbol debug logging
Shows how the debug output helps diagnose why signals are/aren't generated
"""

import pandas as pd
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies.adaptive_str import AdaptiveShortTheRip

# Configure logging to show INFO level
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def create_test_dataframe(rsi_value, ema_aligned=True):
    """Create a test dataframe with specified RSI and EMA alignment"""
    if ema_aligned:
        # Bearish alignment for short: ema21 < ema50 < ema200
        ema21, ema50, ema200 = 99.0, 100.0, 101.0
    else:
        # Not aligned
        ema21, ema50, ema200 = 101.0, 100.0, 99.0
    
    data = {
        'close': [100.0] * 50,
        'open': [99.0] * 50,
        'high': [101.0] * 50,
        'low': [98.0] * 50,
        'volume': [1000.0] * 50,
        'rsi': [rsi_value] * 50,
        'atr': [2.0] * 50,
        'ema21': [ema21] * 50,
        'ema50': [ema50] * 50,
        'ema200': [ema200] * 50,
    }
    return pd.DataFrame(data)


def demo_multi_symbol_debug():
    """Demonstrate debug logging for multiple symbols with different outcomes"""
    
    print("=" * 70)
    print("Multi-Symbol Trading Debug Demonstration")
    print("=" * 70)
    print("\nThis demo shows how debug logging helps diagnose signal generation")
    print("across different symbols with different RSI thresholds.\n")
    
    # Configure strategy with symbol-specific thresholds
    config = {
        'adaptive_rsi_base': 55,
        'adaptive_rsi_range': 10,
        'tp_atr_mult': 3.0,
        'sl_atr_mult': 1.5,
        'min_tp_pct': 0.01,
        'max_sl_pct': 0.02,
        'symbols': {
            'BTC/USDT:USDT': {'rsi_threshold': 55},
            'ETH/USDT:USDT': {'rsi_threshold': 50},
            'SOL/USDT:USDT': {'rsi_threshold': 50},
        }
    }
    
    strategy = AdaptiveShortTheRip(config)
    
    # Scenario 1: All symbols with RSI 52
    print("\n" + "=" * 70)
    print("SCENARIO 1: Market Scan - All symbols have RSI 52")
    print("=" * 70)
    print("Expected: ETH and SOL generate signals (52 >= 50)")
    print("          BTC does NOT generate signal (52 < 55)")
    print("-" * 70 + "\n")
    
    df_52 = create_test_dataframe(rsi_value=52.0, ema_aligned=True)
    
    # Test each symbol
    for symbol in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
        signal = strategy.signal(df_52, symbol=symbol)
        print()  # Add spacing
    
    # Scenario 2: BTC with high RSI, others with low RSI
    print("\n" + "=" * 70)
    print("SCENARIO 2: Different RSI values per symbol")
    print("=" * 70)
    print("BTC: RSI 60 (above threshold 55) - Should generate signal")
    print("ETH: RSI 48 (below threshold 50) - Should NOT generate signal")
    print("SOL: RSI 51 (above threshold 50) - Should generate signal")
    print("-" * 70 + "\n")
    
    # BTC with high RSI
    df_btc = create_test_dataframe(rsi_value=60.0, ema_aligned=True)
    signal_btc = strategy.signal(df_btc, symbol='BTC/USDT:USDT')
    print()
    
    # ETH with low RSI
    df_eth = create_test_dataframe(rsi_value=48.0, ema_aligned=True)
    signal_eth = strategy.signal(df_eth, symbol='ETH/USDT:USDT')
    print()
    
    # SOL with medium RSI
    df_sol = create_test_dataframe(rsi_value=51.0, ema_aligned=True)
    signal_sol = strategy.signal(df_sol, symbol='SOL/USDT:USDT')
    print()
    
    # Scenario 3: Good RSI but bad EMA alignment
    print("\n" + "=" * 70)
    print("SCENARIO 3: Good RSI but failed EMA alignment check")
    print("=" * 70)
    print("ETH: RSI 55 (above threshold) but EMA NOT bearish aligned")
    print("Expected: No signal due to EMA check failure")
    print("-" * 70 + "\n")
    
    df_bad_ema = create_test_dataframe(rsi_value=55.0, ema_aligned=False)
    signal_bad_ema = strategy.signal(df_bad_ema, symbol='ETH/USDT:USDT')
    print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✅ Debug logging shows for each symbol:")
    print("   • Symbol name")
    print("   • RSI value vs. threshold (with symbol-specific overrides)")
    print("   • EMA alignment status and values")
    print("   • Volume checks")
    print("   • ATR values")
    print("   • Final signal decision with clear reasoning")
    print("\n💡 This helps diagnose why ETH/SOL signals weren't being generated:")
    print("   • Too high RSI threshold → Lower it in config")
    print("   • EMA filter too strict → Adjust regime requirements")
    print("   • Missing data → Check data fetching")
    print("\n🎯 Solution: Use symbol-specific thresholds in config.example.yaml")
    print("   BTC: 55 (more selective)")
    print("   ETH/SOL: 50 (more sensitive)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_multi_symbol_debug()
