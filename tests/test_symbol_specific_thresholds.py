#!/usr/bin/env python3
"""
Test symbol-specific RSI threshold configuration
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies.adaptive_str import AdaptiveShortTheRip
from strategies.adaptive_ob import AdaptiveOversoldBounce


def create_test_dataframe(rsi_value=60.0):
    """Create a minimal test dataframe with required columns"""
    data = {
        'close': [100.0] * 50,
        'open': [99.0] * 50,
        'high': [101.0] * 50,
        'low': [98.0] * 50,
        'volume': [1000.0] * 50,
        'rsi': [rsi_value] * 50,
        'atr': [2.0] * 50,
        'ema21': [99.0] * 50,
        'ema50': [100.0] * 50,
        'ema200': [101.0] * 50,
    }
    df = pd.DataFrame(data)
    return df


def test_symbol_specific_threshold_str():
    """Test symbol-specific threshold for ShortTheRip strategy"""
    print("\n=== Testing Symbol-Specific Thresholds for ShortTheRip ===\n")
    
    # Config with symbol-specific thresholds
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
    
    # Test 1: RSI 52 should trigger signal for ETH (threshold 50) but not BTC (threshold 55)
    df_52 = create_test_dataframe(rsi_value=52.0)
    
    print("Test 1: RSI = 52.0")
    print("-" * 50)
    
    # BTC should NOT generate signal (52 < 55)
    signal_btc = strategy.signal(df_52, symbol='BTC/USDT:USDT')
    print(f"BTC (threshold 55): {'✅ Signal' if signal_btc else '❌ No Signal'}")
    assert signal_btc is None, "BTC should NOT generate signal with RSI 52 < threshold 55"
    
    # ETH should generate signal (52 >= 50)
    signal_eth = strategy.signal(df_52, symbol='ETH/USDT:USDT')
    print(f"ETH (threshold 50): {'✅ Signal' if signal_eth else '❌ No Signal'}")
    assert signal_eth is not None, "ETH should generate signal with RSI 52 >= threshold 50"
    
    # Test 2: RSI 56 should trigger signal for both
    df_56 = create_test_dataframe(rsi_value=56.0)
    
    print("\nTest 2: RSI = 56.0")
    print("-" * 50)
    
    signal_btc_2 = strategy.signal(df_56, symbol='BTC/USDT:USDT')
    print(f"BTC (threshold 55): {'✅ Signal' if signal_btc_2 else '❌ No Signal'}")
    assert signal_btc_2 is not None, "BTC should generate signal with RSI 56 >= threshold 55"
    
    signal_eth_2 = strategy.signal(df_56, symbol='ETH/USDT:USDT')
    print(f"ETH (threshold 50): {'✅ Signal' if signal_eth_2 else '❌ No Signal'}")
    assert signal_eth_2 is not None, "ETH should generate signal with RSI 56 >= threshold 50"
    
    print("\n✅ All ShortTheRip tests passed!\n")


def test_default_threshold_fallback():
    """Test that default threshold is used when no symbol-specific config exists"""
    print("\n=== Testing Default Threshold Fallback ===\n")
    
    config = {
        'adaptive_rsi_base': 55,
        'adaptive_rsi_range': 10,
        'tp_atr_mult': 3.0,
        'sl_atr_mult': 1.5,
        'min_tp_pct': 0.01,
        'max_sl_pct': 0.02,
        'symbols': {
            'BTC/USDT:USDT': {'rsi_threshold': 55},
        }
    }
    
    strategy = AdaptiveShortTheRip(config)
    
    # Test with symbol not in config - should use default adaptive threshold
    df = create_test_dataframe(rsi_value=52.0)
    
    # Get the threshold for a symbol not in config
    threshold = strategy.get_symbol_specific_threshold('MATIC/USDT:USDT')
    print(f"MATIC (not in config): threshold = {threshold}")
    assert threshold is None, "Should return None for symbols not in config"
    
    print("✅ Default fallback test passed!\n")


def test_get_symbol_specific_threshold():
    """Test the get_symbol_specific_threshold method directly"""
    print("\n=== Testing get_symbol_specific_threshold Method ===\n")
    
    config = {
        'adaptive_rsi_base': 55,
        'symbols': {
            'BTC/USDT:USDT': {'rsi_threshold': 55},
            'ETH/USDT:USDT': {'rsi_threshold': 50},
        }
    }
    
    strategy = AdaptiveShortTheRip(config)
    
    # Test configured symbols
    btc_threshold = strategy.get_symbol_specific_threshold('BTC/USDT:USDT')
    print(f"BTC threshold: {btc_threshold}")
    assert btc_threshold == 55, "BTC threshold should be 55"
    
    eth_threshold = strategy.get_symbol_specific_threshold('ETH/USDT:USDT')
    print(f"ETH threshold: {eth_threshold}")
    assert eth_threshold == 50, "ETH threshold should be 50"
    
    # Test unconfigured symbol
    sol_threshold = strategy.get_symbol_specific_threshold('SOL/USDT:USDT')
    print(f"SOL threshold: {sol_threshold}")
    assert sol_threshold is None, "SOL should have no specific threshold"
    
    # Test None symbol
    none_threshold = strategy.get_symbol_specific_threshold(None)
    print(f"None threshold: {none_threshold}")
    assert none_threshold is None, "None symbol should return None"
    
    print("✅ Method tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Symbol-Specific Threshold Configuration Tests")
    print("=" * 60)
    
    try:
        test_get_symbol_specific_threshold()
        test_default_threshold_fallback()
        test_symbol_specific_threshold_str()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
