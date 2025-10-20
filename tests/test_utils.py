"""
Shared utilities for testing multi-symbol trading strategies
"""

import pandas as pd

# Price adjustment constants for test data generation
PRICE_OPEN_RATIO = 0.99
PRICE_HIGH_RATIO = 1.01
PRICE_LOW_RATIO = 0.98

# EMA alignment ratios (for bearish alignment: 21 < 50 < 200)
EMA21_RATIO = 0.99
EMA50_RATIO = 1.00
EMA200_RATIO = 1.01


def create_test_dataframe(price=100.0, rsi_value=60.0, ema_aligned=True, bars=50):
    """
    Create a test dataframe with specified parameters
    
    Args:
        price: Base price for the asset
        rsi_value: RSI value to use
        ema_aligned: If True, creates bearish EMA alignment (21 < 50 < 200)
        bars: Number of bars to generate
        
    Returns:
        DataFrame with OHLCV and indicator data
    """
    if ema_aligned:
        # Bearish alignment for short: ema21 < ema50 < ema200
        ema21 = price * EMA21_RATIO
        ema50 = price * EMA50_RATIO
        ema200 = price * EMA200_RATIO
    else:
        # Not aligned (bullish or random)
        ema21 = price * EMA200_RATIO
        ema50 = price * EMA50_RATIO
        ema200 = price * EMA21_RATIO
    
    data = {
        'close': [price] * bars,
        'open': [price * PRICE_OPEN_RATIO] * bars,
        'high': [price * PRICE_HIGH_RATIO] * bars,
        'low': [price * PRICE_LOW_RATIO] * bars,
        'volume': [1000.0] * bars,
        'rsi': [rsi_value] * bars,
        'atr': [price * 0.02] * bars,  # 2% ATR
        'ema21': [ema21] * bars,
        'ema50': [ema50] * bars,
        'ema200': [ema200] * bars,
    }
    return pd.DataFrame(data)


def get_default_strategy_config():
    """
    Get default strategy configuration for testing
    
    Returns:
        Dict with standard strategy configuration
    """
    return {
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
