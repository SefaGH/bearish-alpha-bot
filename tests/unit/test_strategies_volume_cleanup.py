import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.strategies.adaptive_str import AdaptiveShortTheRip
from src.strategies.adaptive_ob import AdaptiveOversoldBounce

@pytest.fixture
def mock_config():
    return {
        'adaptive_rsi_base': 50,
        'adaptive_rsi_range': 10,
        'tp_atr_mult': 2.0,
        'sl_atr_mult': 1.0,
        'min_tp_pct': 0.01,
        'max_sl_pct': 0.02,
        'min_rr_ratio': 1.5,
        'debug': {'strategy_logging': True}
    }

@pytest.fixture
def mock_data_str():
    # Data that should trigger a SELL signal for ShortTheRip
    # RSI > threshold (e.g. 70 > 60)
    # Price > EMA50 + Rip (if applicable)
    # We need to construct a DataFrame that satisfies the conditions
    data = {
        'close': [100.0, 105.0],
        'rsi': [60.0, 70.0], # RSI 70 > 60 (min threshold)
        'atr': [1.0, 1.0],
        'ema_fast': [100.0, 100.0],
        'ema21': [100.0, 100.0],
        'ema50': [90.0, 90.0], # Price 105 > EMA50 90
        'ema200': [80.0, 80.0],
        'volume': [1000, 1000]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def mock_data_ob():
    # Data that should trigger a BUY signal for OversoldBounce
    # RSI < threshold (e.g. 25 < 32)
    # Price < EMA Fast
    data = {
        'close': [100.0, 90.0],
        'rsi': [40.0, 25.0],
        'atr': [1.0, 1.0],
        'ema_fast': [100.0, 95.0], # Price 90 < EMA Fast 95
        'ema_mid': [105.0, 100.0],
        'volume': [1000, 1000]
    }
    df = pd.DataFrame(data)
    return df

def test_adaptive_str_signal_does_not_raise(mock_config, mock_data_str):
    strategy = AdaptiveShortTheRip(mock_config)
    
    # Ensure we trigger a signal
    # RSI 40 < 50 (base)
    # Price 105 > EMA50 90 + Rip (1.0 * 1.0 = 1.0) = 91
    
    try:
        signal = strategy.signal(mock_data_str, df_1h=mock_data_str, symbol="BTC/USDT")
    except Exception as e:
        pytest.fail(f"AdaptiveShortTheRip.signal raised exception: {e}")
    
    assert signal is not None, "Signal should be generated"
    assert 'strategy_volume_decision' not in signal, "Legacy field 'strategy_volume_decision' should not be present"
    assert signal['side'] == 'sell'

def test_adaptive_ob_signal_does_not_raise(mock_config, mock_data_ob):
    strategy = AdaptiveOversoldBounce(mock_config)
    
    # Ensure we trigger a signal
    # RSI 25 < 32 (base default in code is 32, mock config is 50 but code might override or use default if not in config correctly)
    # Actually mock_config has adaptive_rsi_base=50.
    # RSI 25 < 50.
    # Price 90 < EMA Fast 95.
    
    try:
        signal = strategy.signal(mock_data_ob, df_1h=mock_data_ob, symbol="BTC/USDT")
    except Exception as e:
        pytest.fail(f"AdaptiveOversoldBounce.signal raised exception: {e}")
    
    assert signal is not None, "Signal should be generated"
    assert 'strategy_volume_decision' not in signal, "Legacy field 'strategy_volume_decision' should not be present"
    assert 'volume_boost' not in signal.get('meta', {}), "Legacy meta 'volume_boost' should not be present"
    assert signal['side'] == 'buy'
