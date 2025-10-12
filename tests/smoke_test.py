#!/usr/bin/env python3
"""
Smoke test for Bearish Alpha Bot.
Verifies basic functionality without requiring exchange credentials.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    try:
        from core.ccxt_client import CcxtClient
        from core.multi_exchange import build_clients_from_env
        from core.indicators import add_indicators
        from core.regime import is_bearish_regime
        from core.notify import Telegram
        from core.sizing import position_size_usdt
        from core.trailing import initial_stops
        from core.state import load_state, save_state
        from core.logger import setup_logger
        from strategies.oversold_bounce import OversoldBounce
        from strategies.short_the_rip import ShortTheRip
        from universe import build_universe, pick_execution_exchange
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_position_sizing():
    """Test position sizing calculations."""
    print("Testing position sizing...")
    from core.sizing import position_size_usdt
    
    # Test LONG
    qty_long = position_size_usdt(100, 99, 10, 'long')
    if abs(qty_long - 10.0) < 0.01:
        print("  ✓ LONG position sizing correct")
    else:
        print(f"  ✗ LONG position sizing wrong: expected 10.0, got {qty_long}")
        return False
    
    # Test SHORT
    qty_short = position_size_usdt(100, 101, 10, 'short')
    if abs(qty_short - 10.0) < 0.01:
        print("  ✓ SHORT position sizing correct")
    else:
        print(f"  ✗ SHORT position sizing wrong: expected 10.0, got {qty_short}")
        return False
    
    return True

def test_indicators():
    """Test indicator calculations."""
    print("Testing indicators...")
    import pandas as pd
    from core.indicators import add_indicators
    
    # Create sample OHLCV data
    data = {
        'open': [100] * 50,
        'high': [105] * 50,
        'low': [95] * 50,
        'close': [102] * 50,
        'volume': [1000] * 50
    }
    df = pd.DataFrame(data)
    
    try:
        df_ind = add_indicators(df, {'rsi_period': 14, 'atr_period': 14})
        required_cols = ['rsi', 'atr', 'ema21', 'ema50', 'ema200']
        if all(col in df_ind.columns for col in required_cols):
            print("  ✓ Indicators calculated successfully")
            return True
        else:
            missing = [c for c in required_cols if c not in df_ind.columns]
            print(f"  ✗ Missing indicator columns: {missing}")
            return False
    except Exception as e:
        print(f"  ✗ Indicator calculation failed: {e}")
        return False

def test_strategies():
    """Test strategy signal generation."""
    print("Testing strategies...")
    import pandas as pd
    from strategies.oversold_bounce import OversoldBounce
    from strategies.short_the_rip import ShortTheRip
    
    # Create sample indicator data
    data = {
        'rsi': [20, 25, 30],
        'atr': [1.5, 1.6, 1.7],
        'ema21': [100, 101, 102],
        'ema50': [105, 106, 107],
        'ema200': [110, 111, 112]
    }
    df = pd.DataFrame(data)
    
    try:
        # Test OversoldBounce
        ob = OversoldBounce({'rsi_max': 30, 'tp_pct': 0.015})
        signal = ob.signal(df)
        if signal and signal['side'] == 'buy':
            print("  ✓ OversoldBounce strategy works")
        else:
            print("  ℹ OversoldBounce no signal (expected for test data)")
        
        # Test ShortTheRip
        str_strat = ShortTheRip({'rsi_min': 15, 'tp_pct': 0.012})
        signal = str_strat.signal(df, df)
        print("  ✓ ShortTheRip strategy works")
        
        return True
    except Exception as e:
        print(f"  ✗ Strategy test failed: {e}")
        return False

def test_config_loading():
    """Test config file loading."""
    print("Testing config loading...")
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.example.yaml')
    
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        if cfg and 'signals' in cfg:
            print("  ✓ Config file loaded successfully")
            return True
        else:
            print("  ✗ Config file is invalid or empty")
            return False
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Bearish Alpha Bot - Smoke Test")
    print("=" * 60)
    print()
    
    tests = [
        test_imports,
        test_config_loading,
        test_position_sizing,
        test_indicators,
        test_strategies,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All smoke tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
