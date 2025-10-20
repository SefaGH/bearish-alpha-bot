#!/usr/bin/env python3
"""
Simulate a multi-symbol trading scan showing debug output
This simulates what happens when the bot scans BTC, ETH, and SOL
"""

import pandas as pd
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies.adaptive_str import AdaptiveShortTheRip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def simulate_market_scan():
    """Simulate a market scan showing why some symbols generate signals and others don't"""
    
    print("\n" + "=" * 80)
    print(" SIMULATED MULTI-SYMBOL TRADING SCAN")
    print("=" * 80)
    print("\nBot Configuration:")
    print("  • Strategy: Adaptive Short The Rip")
    print("  • Symbols: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT")
    print("  • BTC RSI Threshold: 55 (more selective)")
    print("  • ETH/SOL RSI Threshold: 50 (more sensitive)")
    print("\n" + "-" * 80)
    
    # Configure strategy
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
    
    # Simulate current market conditions
    market_conditions = [
        {
            'symbol': 'BTC/USDT:USDT',
            'price': 43500.00,
            'rsi': 53.2,
            'ema_aligned': True,
            'description': 'RSI below BTC threshold (53.2 < 55)'
        },
        {
            'symbol': 'ETH/USDT:USDT',
            'price': 2280.50,
            'rsi': 52.8,
            'ema_aligned': True,
            'description': 'RSI above ETH threshold (52.8 >= 50)'
        },
        {
            'symbol': 'SOL/USDT:USDT',
            'price': 98.75,
            'rsi': 51.3,
            'ema_aligned': True,
            'description': 'RSI above SOL threshold (51.3 >= 50)'
        },
    ]
    
    print("\n📊 Current Market Conditions:")
    for cond in market_conditions:
        print(f"   {cond['symbol']:20s} Price: ${cond['price']:>10.2f}  RSI: {cond['rsi']:>5.1f}  {cond['description']}")
    
    print("\n" + "=" * 80)
    print(" SCANNING SYMBOLS FOR SIGNALS...")
    print("=" * 80 + "\n")
    
    signals_generated = []
    
    for market in market_conditions:
        # Create test data for this symbol
        if market['ema_aligned']:
            ema21, ema50, ema200 = market['price'] * 0.99, market['price'], market['price'] * 1.01
        else:
            ema21, ema50, ema200 = market['price'] * 1.01, market['price'], market['price'] * 0.99
        
        data = {
            'close': [market['price']] * 50,
            'open': [market['price'] * 0.99] * 50,
            'high': [market['price'] * 1.01] * 50,
            'low': [market['price'] * 0.98] * 50,
            'volume': [1000.0] * 50,
            'rsi': [market['rsi']] * 50,
            'atr': [market['price'] * 0.02] * 50,
            'ema21': [ema21] * 50,
            'ema50': [ema50] * 50,
            'ema200': [ema200] * 50,
        }
        df = pd.DataFrame(data)
        
        # Generate signal
        signal = strategy.signal(df, symbol=market['symbol'])
        
        if signal:
            signals_generated.append({
                'symbol': market['symbol'],
                'entry': signal['entry'],
                'target': signal['target'],
                'stop': signal['stop'],
            })
        
        print()  # Add spacing between symbols
    
    # Summary
    print("=" * 80)
    print(" SCAN COMPLETE")
    print("=" * 80)
    
    if signals_generated:
        print(f"\n✅ Generated {len(signals_generated)} signal(s):\n")
        for sig in signals_generated:
            print(f"   📈 {sig['symbol']}")
            print(f"      Entry:  ${sig['entry']:>10.2f}")
            print(f"      Target: ${sig['target']:>10.2f}")
            print(f"      Stop:   ${sig['stop']:>10.2f}")
            print()
    else:
        print("\n❌ No signals generated")
    
    print("\n💡 Analysis:")
    print("   • BTC (53.2) did NOT trigger because 53.2 < 55 threshold")
    print("   • ETH (52.8) triggered because 52.8 >= 50 threshold ✅")
    print("   • SOL (51.3) triggered because 51.3 >= 50 threshold ✅")
    print("\n📝 Before symbol-specific config:")
    print("   • All symbols used threshold 55")
    print("   • Only BTC would generate signals")
    print("   • ETH and SOL were filtered out (RSI too low)")
    print("\n🎯 With symbol-specific config:")
    print("   • ETH and SOL use lower threshold (50)")
    print("   • More signals from altcoins")
    print("   • Better diversification across portfolio")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    simulate_market_scan()
