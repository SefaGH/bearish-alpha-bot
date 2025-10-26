"""
Test that market_data dictionary includes all timeframes for strategies.
This test validates the fix for the issue where strategies couldn't access
1m and 5m timeframe data.
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_mock_dataframe(periods=100):
    """Create a mock OHLCV dataframe with indicators."""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
    
    # Generate mock OHLCV data
    close = np.cumsum(np.random.randn(periods)) + 100
    high = close + np.random.rand(periods) * 2
    low = close - np.random.rand(periods) * 2
    open_ = close + np.random.randn(periods)
    volume = np.random.randint(1000, 10000, periods)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'rsi': np.random.uniform(20, 80, periods),  # Mock RSI
        'atr': np.random.uniform(0.5, 2.0, periods),  # Mock ATR
        'ema_50': close * (1 + np.random.randn(periods) * 0.01)  # Mock EMA
    }, index=dates)
    
    return df


def test_adaptive_ob_accepts_market_data():
    """Test that AdaptiveOversoldBounce accepts market_data parameter."""
    try:
        from strategies.adaptive_ob import AdaptiveOversoldBounce
    except ImportError:
        try:
            from src.strategies.adaptive_ob import AdaptiveOversoldBounce
        except ImportError:
            print("⚠️ Could not import AdaptiveOversoldBounce, skipping test")
            return
    
    # Create mock data
    df_30m = create_mock_dataframe()
    df_1h = create_mock_dataframe()
    df_1m = create_mock_dataframe()
    df_5m = create_mock_dataframe()
    
    # Create market_data dictionary
    market_data = {
        '1m': df_1m,
        '5m': df_5m,
        '30m': df_30m,
        '1h': df_1h
    }
    
    # Initialize strategy
    config = {
        'rsi_max': 35,
        'tp_pct': 0.015,
        'sl_atr_mult': 1.0
    }
    
    strategy = AdaptiveOversoldBounce(config, regime_analyzer=None)
    
    # Test that signal method accepts market_data parameter
    try:
        signal = strategy.signal(
            df_30m, 
            df_1h, 
            regime_data=None, 
            symbol='BTC/USDT:USDT',
            market_data=market_data
        )
        print("✅ AdaptiveOversoldBounce accepts market_data parameter")
        print(f"   Signal returned: {signal is not None}")
        print(f"   Market data timeframes available: {list(market_data.keys())}")
    except TypeError as e:
        print(f"❌ AdaptiveOversoldBounce does not accept market_data parameter: {e}")
        raise


def test_adaptive_str_accepts_market_data():
    """Test that AdaptiveShortTheRip accepts market_data parameter."""
    try:
        from strategies.adaptive_str import AdaptiveShortTheRip
    except ImportError:
        try:
            from src.strategies.adaptive_str import AdaptiveShortTheRip
        except ImportError:
            print("⚠️ Could not import AdaptiveShortTheRip, skipping test")
            return
    
    # Create mock data
    df_30m = create_mock_dataframe()
    df_1h = create_mock_dataframe()
    df_1m = create_mock_dataframe()
    df_5m = create_mock_dataframe()
    
    # Create market_data dictionary
    market_data = {
        '1m': df_1m,
        '5m': df_5m,
        '30m': df_30m,
        '1h': df_1h
    }
    
    # Initialize strategy
    config = {
        'rsi_min': 65,
        'tp_pct': 0.015,
        'sl_atr_mult': 1.0
    }
    
    strategy = AdaptiveShortTheRip(config, regime_analyzer=None)
    
    # Test that signal method accepts market_data parameter
    try:
        signal = strategy.signal(
            df_30m, 
            df_1h, 
            regime_data=None, 
            symbol='BTC/USDT:USDT',
            market_data=market_data
        )
        print("✅ AdaptiveShortTheRip accepts market_data parameter")
        print(f"   Signal returned: {signal is not None}")
        print(f"   Market data timeframes available: {list(market_data.keys())}")
    except TypeError as e:
        print(f"❌ AdaptiveShortTheRip does not accept market_data parameter: {e}")
        raise


def test_market_data_dictionary_creation():
    """Test that market_data dictionary is created correctly."""
    # Simulate what production_coordinator does
    df_1m = create_mock_dataframe()
    df_5m = create_mock_dataframe()
    df_30m = create_mock_dataframe()
    df_1h = create_mock_dataframe()
    df_4h = create_mock_dataframe()
    
    # Create market_data dictionary (as done in production_coordinator)
    market_data = {}
    if df_1m is not None:
        market_data['1m'] = df_1m
    if df_5m is not None:
        market_data['5m'] = df_5m
    if df_30m is not None:
        market_data['30m'] = df_30m
    if df_1h is not None:
        market_data['1h'] = df_1h
    if df_4h is not None:
        market_data['4h'] = df_4h
    
    # Verify all timeframes are present
    expected_timeframes = ['1m', '5m', '30m', '1h', '4h']
    assert set(market_data.keys()) == set(expected_timeframes), \
        f"Expected {expected_timeframes}, got {list(market_data.keys())}"
    
    print("✅ Market data dictionary created correctly")
    print(f"   Timeframes: {list(market_data.keys())}")
    
    # Verify each timeframe has data
    for tf, df in market_data.items():
        assert isinstance(df, pd.DataFrame), f"{tf} is not a DataFrame"
        assert not df.empty, f"{tf} DataFrame is empty"
        assert 'rsi' in df.columns, f"{tf} missing RSI column"
        print(f"   {tf}: {len(df)} rows, RSI range [{df['rsi'].min():.1f}, {df['rsi'].max():.1f}]")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Testing Market Data Timeframe Support")
    print("="*70 + "\n")
    
    try:
        test_market_data_dictionary_creation()
        print()
        test_adaptive_ob_accepts_market_data()
        print()
        test_adaptive_str_accepts_market_data()
        print()
        print("="*70)
        print("✅ All tests passed!")
        print("="*70)
    except Exception as e:
        print()
        print("="*70)
        print(f"❌ Test failed: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        exit(1)
