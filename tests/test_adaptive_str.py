import copy
import logging

import pandas as pd
import pytest

from strategies.adaptive_str import AdaptiveShortTheRip
from config.mtf_policy import build_str_mtf_config


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
            'ema50': 99.0,
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


def _default_mtf_cfg():
    return {
        "enabled": True,
        "15m_mode": "hard",
        "1h_mode": "hard",
        "missing_15m_is_fatal": False,
        "missing_1h_is_fatal": False,
        "on_missing_15m": "skip",
        "on_missing_1h": "skip",
        "rsi_15m_min": 62.0,
        "min_15m_close_over_ema50_pct": 0.0,
        "require_1h_bearish_ema_stack": True,
        "rsi_1h_max": 60.0,
        "min_bars_rsi": 20,
        "min_bars_ema21": 30,
        "min_bars_ema50": 100,
        "min_bars_ema200": 250,
    }


def _build_mtf_policy(**overrides):
    cfg = _default_mtf_cfg()
    cfg.update(overrides)
    return build_str_mtf_config(cfg)


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
            'close': 86650.0,
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
            'close': 86600.0,
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
    cfg = copy.deepcopy(base_config)
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(**{"15m_mode": "off", "1h_mode": "off"})
    strategy = AdaptiveShortTheRip(cfg)

    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
    )
    assert signal is not None
    assert signal["features"]["mtf_15m"]["status"] == "skipped"
    assert signal["features"]["mtf_1h"]["status"] == "skipped"


def test_mtf_15m_hard_vetoes_signal(base_config, sample_df_30m, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(
        **{
            "15m_mode": "hard",
            "1h_mode": "off",
            "rsi_15m_min": 70.0,
        }
    )
    strategy = AdaptiveShortTheRip(cfg)

    df_15m = pd.DataFrame([{"close": 100.0, "rsi": 50.0}])
    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
        market_data={"15m": df_15m},
    )
    assert signal is None


def test_mtf_15m_soft_allows_signal(base_config, sample_df_30m, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(
        **{
            "15m_mode": "soft",
            "1h_mode": "off",
            "rsi_15m_min": 70.0,
        }
    )
    strategy = AdaptiveShortTheRip(cfg)

    df_15m = pd.DataFrame([{"close": 100.0, "rsi": 50.0}])
    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
        market_data={"15m": df_15m},
    )
    assert signal is not None
    assert signal["features"]["mtf_15m"]["soft_fail"] is True


def test_mtf_1h_hard_veto_can_be_bypassed_on_extreme_move(base_config):
    cfg = copy.deepcopy(base_config)

    mtf_cfg = _default_mtf_cfg()
    mtf_cfg.update(
        {
            "15m_mode": "off",
            "1h_mode": "hard",
            "require_1h_bearish_ema_stack": True,
        }
    )
    mtf_cfg["extreme_bypass"] = {
        "enabled": True,
        "min_directional_move_pct": 0.02,
        "min_abs_move_pct": 0.0,
        "min_atr_pct": 0.005,
        "rsi_oversold_threshold": 0.0,
        "rsi_overbought_threshold": 0.0,
    }

    cfg["mtf_confirmation"] = mtf_cfg
    cfg["mtf_confirmation_effective"] = build_str_mtf_config(mtf_cfg)
    strategy = AdaptiveShortTheRip(cfg)

    df_30m = pd.DataFrame(
        [
            {
                "close": 100.0,
                "rsi": 70.0,
                "atr": 1.0,
                "ema_fast": 99.0,
                "ema21": 98.0,
                "ema50": 101.0,
                "ema200": 110.0,
                "volume": 1000.0,
            },
            {
                "close": 103.0,  # +3%: triggers directional move bypass for sell signals
                "rsi": 75.0,
                "atr": 1.0,
                "ema_fast": 99.0,
                "ema21": 98.0,
                "ema50": 101.0,
                "ema200": 110.0,
                "volume": 1000.0,
            },
        ]
    )
    df_1h_bullish = pd.DataFrame(
        [
            {
                "close": 103.0,
                "rsi": 55.0,
                "ema21": 120.0,
                "ema50": 100.0,
                "ema200": 90.0,
            }
        ]
    )

    signal = strategy.signal(df_30m=df_30m, df_1h=df_1h_bullish, symbol="BTC/USDT")
    assert signal is not None
    assert signal["features"]["mtf_1h"]["bypass"] is True


def test_mtf_1h_hard_veto_bypass_blocked_when_atr_missing(base_config):
    cfg = copy.deepcopy(base_config)

    mtf_cfg = _default_mtf_cfg()
    mtf_cfg.update(
        {
            "15m_mode": "off",
            "1h_mode": "hard",
            "require_1h_bearish_ema_stack": True,
        }
    )
    mtf_cfg["extreme_bypass"] = {
        "enabled": True,
        "min_directional_move_pct": 0.02,
        "min_abs_move_pct": 0.0,
        "min_atr_pct": 0.005,
        "rsi_oversold_threshold": 0.0,
        "rsi_overbought_threshold": 0.0,
    }

    cfg["mtf_confirmation"] = mtf_cfg
    cfg["mtf_confirmation_effective"] = build_str_mtf_config(mtf_cfg)
    strategy = AdaptiveShortTheRip(cfg)

    # Intentionally omit the 'atr' column so the strategy's upstream fallback
    # (close*0.02) is treated as unsafe for bypass decisions.
    df_30m = pd.DataFrame(
        [
            {
                "close": 100.0,
                "rsi": 70.0,
                "ema_fast": 99.0,
                "ema21": 98.0,
                "ema50": 101.0,
                "ema200": 110.0,
                "volume": 1000.0,
            },
            {
                "close": 103.0,  # +3% directional move would normally satisfy min_directional_move_pct
                "rsi": 75.0,
                "ema_fast": 99.0,
                "ema21": 98.0,
                "ema50": 101.0,
                "ema200": 110.0,
                "volume": 1000.0,
            },
        ]
    )

    df_1h_bullish = pd.DataFrame(
        [
            {
                "close": 103.0,
                "rsi": 55.0,
                "ema21": 120.0,
                "ema50": 100.0,
                "ema200": 90.0,
            }
        ]
    )

    signal = strategy.signal(df_30m=df_30m, df_1h=df_1h_bullish, symbol="BTC/USDT")
    assert signal is None


def test_rsi_rollover_guard_blocks_risky_short_when_not_rolling_over(base_config, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(**{"15m_mode": "off", "1h_mode": "off"})
    cfg["rsi_rollover_guard"] = {"enabled": True, "eps": 0.2}
    strategy = AdaptiveShortTheRip(cfg)

    df_closed = pd.DataFrame(
        [
            {
                "close": 101.0,
                "rsi": 70.0,
                "atr": 1.0,
                "ema_fast": 100.5,
                "ema21": 99.0,
                "ema50": 100.0,
                "ema200": 110.0,
                "volume": 1000.0,
            }
        ]
    )
    df_hybrid = pd.DataFrame(
        [
            {
                "close": 101.0,
                "rsi": 70.0,
                "atr": 1.0,
                "ema_fast": 100.5,
                "ema21": 99.0,
                "ema50": 100.0,
                "ema200": 110.0,
                "volume": 1000.0,
            },
            {
                "close": 101.0,
                "rsi": 70.0,  # not rolling over vs prev (within eps)
                "atr": 1.0,
                "ema_fast": 100.5,
                "ema21": 99.0,
                "ema50": 100.0,
                "ema200": 110.0,
                "volume": 1000.0,
            },
        ]
    )
    df_hybrid.attrs["includes_forming"] = True
    df_hybrid.attrs["fallback_reason"] = None
    df_hybrid.attrs["merge_action"] = "appended"

    signal = strategy.signal(
        df_30m=df_closed,
        df_1h=sample_df_1h_bearish,
        symbol="BTC/USDT",
        market_data={"30m_closed": df_closed, "30m_hybrid": df_hybrid},
    )
    assert signal is None
    assert strategy._guard_telemetry["guard_rollover_defer_count"] == 1


def test_rsi_rollover_guard_allows_risky_short_when_rsi_rolls_over(base_config, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(**{"15m_mode": "off", "1h_mode": "off"})
    cfg["rsi_rollover_guard"] = {"enabled": True, "eps": 0.2}
    strategy = AdaptiveShortTheRip(cfg)

    df_closed = pd.DataFrame(
        [
            {
                "close": 101.0,
                "rsi": 70.0,
                "atr": 1.0,
                "ema_fast": 100.5,
                "ema21": 99.0,
                "ema50": 100.0,
                "ema200": 110.0,
                "volume": 1000.0,
            }
        ]
    )
    df_hybrid = pd.DataFrame(
        [
            {
                "close": 101.0,
                "rsi": 70.0,
                "atr": 1.0,
                "ema_fast": 100.5,
                "ema21": 99.0,
                "ema50": 100.0,
                "ema200": 110.0,
                "volume": 1000.0,
            },
            {
                "close": 101.0,
                "rsi": 69.6,  # rolling over (prev - now >= eps)
                "atr": 1.0,
                "ema_fast": 100.5,
                "ema21": 99.0,
                "ema50": 100.0,
                "ema200": 110.0,
                "volume": 1000.0,
            },
        ]
    )
    df_hybrid.attrs["includes_forming"] = True
    df_hybrid.attrs["fallback_reason"] = None
    df_hybrid.attrs["merge_action"] = "appended"

    signal = strategy.signal(
        df_30m=df_closed,
        df_1h=sample_df_1h_bearish,
        symbol="BTC/USDT",
        market_data={"30m_closed": df_closed, "30m_hybrid": df_hybrid},
    )
    assert signal is not None
    assert strategy._guard_telemetry["guard_rollover_defer_count"] == 0


def test_mtf_missing_15m_skip_allows_signal(base_config, sample_df_30m, sample_df_1h_bearish):
    cfg = copy.deepcopy(base_config)
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(
        **{
            "15m_mode": "hard",
            "1h_mode": "hard",
            "missing_15m_is_fatal": False,
            "on_missing_15m": "skip",
        }
    )
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
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(
        **{
            "15m_mode": "hard",
            "1h_mode": "hard",
            "missing_15m_is_fatal": True,
            "on_missing_15m": "reject",
        }
    )
    strategy = AdaptiveShortTheRip(cfg)

    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=sample_df_1h_bearish,
        symbol='BTC/USDT',
    )
    assert signal is None


def test_mtf_missing_1h_reject_blocks_signal(base_config, sample_df_30m):
    cfg = copy.deepcopy(base_config)
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(
        **{
            "15m_mode": "hard",
            "1h_mode": "hard",
            "missing_1h_is_fatal": True,
            "on_missing_1h": "reject",
        }
    )
    strategy = AdaptiveShortTheRip(cfg)

    signal = strategy.signal(
        df_30m=sample_df_30m,
        df_1h=None,
        symbol='BTC/USDT',
    )
    assert signal is None


def test_mtf_insufficient_bars_1h_reject_blocks_signal(base_config, sample_df_30m):
    cfg = copy.deepcopy(base_config)
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(
        **{
            "15m_mode": "hard",
            "1h_mode": "hard",
            "missing_1h_is_fatal": True,
            "on_missing_1h": "reject",
        }
    )
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
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(
        **{
            "15m_mode": "hard",
            "1h_mode": "hard",
            "min_15m_close_over_ema50_pct": 0.002,
            "missing_15m_is_fatal": False,
            "on_missing_15m": "skip",
        }
    )
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
    cfg["mtf_confirmation_effective"] = _build_mtf_policy(
        **{
            "15m_mode": "hard",
            "1h_mode": "hard",
            "rsi_15m_min": 10.0,
            "min_15m_close_over_ema50_pct": 0.0,
            "missing_15m_is_fatal": False,
            "on_missing_15m": "skip",
        }
    )
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
