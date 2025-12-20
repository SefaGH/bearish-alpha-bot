import copy
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


@pytest.fixture
def sample_df_30m():
    return pd.DataFrame([
        {
            'close': 100.0,
            'rsi': 75.0,
            'atr': 1.0,
            'ema_fast': 99.0,
            'ema21': 98.0,
            'ema50': 90.0,
            'ema200': 110.0,
            'volume': 1000.0,
        }
    ])


@pytest.fixture
def sample_df_1h_bearish():
    return pd.DataFrame([
        {
            'close': 100.0,
            'rsi': 55.0,
            'ema21': 90.0,
            'ema50': 100.0,
            'ema200': 110.0,
        }
    ])


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


def test_mtf_disabled_does_not_block(base_config, sample_df_30m, sample_df_1h_bearish):
    strategy = AdaptiveShortTheRip(base_config)

    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
    )
    assert signal is not None


def test_mtf_missing_15m_skip_allows_signal(base_config, sample_df_30m, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg['mtf_confirmation'] = {
        'enabled': True,
        'require_15m': False,
        'require_1h': False,
        'on_missing_15m': 'skip',
        'on_missing_1h': 'skip',
    }
    strategy = AdaptiveShortTheRip(cfg)

    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
    )
    assert signal is not None
    assert signal['features']['mtf_15m']['status'] == 'missing'
    assert signal['features']['mtf_15m']['action'] == 'skip'


def test_mtf_missing_15m_reject_blocks_signal(base_config, sample_df_30m, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg['mtf_confirmation'] = {
        'enabled': True,
        'require_15m': False,
        'require_1h': False,
        'on_missing_15m': 'reject',
        'on_missing_1h': 'skip',
    }
    strategy = AdaptiveShortTheRip(cfg)

    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
    )
    assert signal is None


def test_mtf_missing_1h_reject_blocks_signal(base_config, sample_df_30m):
    cfg = copy.deepcopy(base_config)
    cfg['mtf_confirmation'] = {
        'enabled': True,
        'require_15m': False,
        'require_1h': True,
        'on_missing_15m': 'skip',
        'on_missing_1h': 'reject',
    }
    strategy = AdaptiveShortTheRip(cfg)

    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=None,
        symbol='BTC/USDT',
    )
    assert signal is None


def test_mtf_insufficient_bars_1h_reject_blocks_signal(base_config, sample_df_30m):
    cfg = copy.deepcopy(base_config)
    cfg['mtf_confirmation'] = {
        'enabled': True,
        'require_15m': False,
        'require_1h': True,
        'require_1h_bearish_ema_stack': True,
        'rsi_1h_max': 60.0,
        'on_missing_15m': 'skip',
        'on_missing_1h': 'reject',
    }
    strategy = AdaptiveShortTheRip(cfg)

    df_1h_short = pd.DataFrame({'close': [100.0] * 100})
    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=df_1h_short,
        symbol='BTC/USDT',
    )
    assert signal is None
    assert strategy._mtf_telemetry['mtf_1h_fallback_attempted'] == 1
    assert strategy._mtf_telemetry['mtf_1h_fallback_skipped_insufficient_bars'] == 1


def test_mtf_insufficient_bars_15m_skip_allows_signal(base_config, sample_df_30m, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg['mtf_confirmation'] = {
        'enabled': True,
        'require_15m': False,
        'require_1h': False,
        'min_15m_close_over_ema50_pct': 0.002,
        'on_missing_15m': 'skip',
        'on_missing_1h': 'skip',
    }
    strategy = AdaptiveShortTheRip(cfg)

    df_15m_short = pd.DataFrame({'close': [100.0] * 50})
    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
        market_data={'15m': df_15m_short},
    )
    assert signal is not None
    assert strategy._mtf_telemetry['mtf_15m_fallback_attempted'] == 1
    assert strategy._mtf_telemetry['mtf_15m_fallback_skipped_insufficient_bars'] == 1


def test_mtf_fallback_cache_hit(base_config, sample_df_30m, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg['mtf_confirmation'] = {
        'enabled': True,
        'require_15m': False,
        'require_1h': False,
        'rsi_15m_min': 10.0,
        'min_15m_close_over_ema50_pct': 0.0,
        'on_missing_15m': 'skip',
        'on_missing_1h': 'skip',
    }
    strategy = AdaptiveShortTheRip(cfg)

    df_15m = pd.DataFrame({'close': [100.0 + i for i in range(30)]})
    signal_1 = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
        market_data={'15m': df_15m},
    )
    signal_2 = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
        market_data={'15m': df_15m},
    )
    assert signal_1 is not None
    assert signal_2 is not None
    assert strategy._mtf_telemetry['mtf_15m_fallback_attempted'] == 2
    assert strategy._mtf_telemetry['mtf_15m_fallback_computed'] == 1
    assert strategy._mtf_telemetry['mtf_15m_cache_hit'] == 1
