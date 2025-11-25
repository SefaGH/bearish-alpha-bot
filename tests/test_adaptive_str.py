import logging

import pandas as pd
import pytest

from strategies.adaptive_str import AdaptiveShortTheRip


logger = logging.getLogger(__name__)


@pytest.fixture
def base_config():
    return {
        'min_rr_ratio': 1.5,
        'adaptive_rsi_base': 68,
        'adaptive_rsi_range': 8,
        'tp_atr_mult': 3.0,
        'sl_atr_mult': 1.5,
        'min_tp_pct': 0.01,
        'max_sl_pct': 0.02,
        'volatility_stop': {
            'enabled': True,
            'min_sl_pct': 0.002,
            'max_sl_pct': 0.02,
            'overrides': {
                'low': {'atr_scale': 0.6, 'min_sl_pct': 0.0005},
                'high': {'atr_scale': 1.3, 'max_sl_pct': 0.03},
            },
        },
        'symbols': {
            'BTC/USDT': {'rsi_threshold': 60},
        }
    }


def test_volatility_stop_tuning_scales_with_regime(base_config):
    strategy = AdaptiveShortTheRip(base_config)

    low_sl, low_meta = strategy._apply_volatility_stop_tuning('low', base_sl_pct=0.01, atr_pct=0.004)
    assert low_sl == pytest.approx(0.006)
    assert low_meta['volatility'] == 'low'

    high_sl, high_meta = strategy._apply_volatility_stop_tuning('high', base_sl_pct=0.01, atr_pct=0.004)
    assert high_sl == pytest.approx(0.013)
    assert high_meta['volatility'] == 'high'


def test_signal_includes_volatility_stop_metadata(base_config):
    strategy = AdaptiveShortTheRip(base_config)

    df_30m = pd.DataFrame([
        {
            'close': 87000.0,
            'rsi': 75.0,
            'atr': 80.0,
            'ema_fast': 86000.0,
            'ema21': 84000.0,
            'ema50': 85000.0,
            'ema200': 88000.0,
        }
    ])

    regime_data = {
        'trend': 'neutral',
        'momentum': 'sideways',
        'volatility': 'low',
        'micro_trend_strength': 0.5,
    }

    signal = strategy.signal(df_30m=df_30m, df_1h=df_30m, regime_data=regime_data, symbol='BTC/USDT')
    assert signal is not None
    assert 'volatility_stop_meta' in signal
    meta = signal['volatility_stop_meta']
    assert meta['volatility'] == 'low'
    assert meta['final_sl_pct'] <= meta['base_sl_pct']


def test_signal_logs_volatility_stop_metadata(base_config, caplog):
    strategy = AdaptiveShortTheRip(base_config)

    df_30m = pd.DataFrame([
        {
            'close': 87500.0,
            'rsi': 78.0,
            'atr': 140.0,
            'ema_fast': 86000.0,
            'ema21': 84000.0,
            'ema50': 85000.0,
            'ema200': 88000.0,
        }
    ])

    regime_data = {
        'trend': 'bullish',
        'momentum': 'strong',
        'volatility': 'low',
        'micro_trend_strength': 0.8,
    }

    caplog.set_level(logging.INFO, logger='strategies.adaptive_str')
    signal = strategy.signal(df_30m=df_30m, df_1h=df_30m, regime_data=regime_data, symbol='BTC/USDT')
    assert signal is not None
    meta = signal['volatility_stop_meta']
    logger.info("volatility_stop_meta=%s", meta)

    assert meta['volatility'] == 'low'
    assert meta['final_sl_pct'] < meta['base_sl_pct']
    assert any('[VolStop]' in record.message for record in caplog.records)
