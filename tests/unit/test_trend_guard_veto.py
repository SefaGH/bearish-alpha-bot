import numpy as np
import pandas as pd

from src.safety.trend_guard import TrendGuard


def _build_downtrend_df() -> pd.DataFrame:
    n_flat = 80

    flat = 100 + 0.02 * np.sin(np.linspace(0, 6 * np.pi, n_flat - 1))
    close = np.concatenate([flat, [90.0]])

    open_ = close + 0.02
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    high[-1] = max(open_[-1], close[-1]) + 2.0
    low[-1] = min(open_[-1], close[-1]) - 2.0
    volume = np.full(len(close), 1000.0)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_guard() -> TrendGuard:
    return TrendGuard(
        {
            "enabled": True,
            "default_timeframe": "5m",
            "lookback_bars": 120,
            "min_history_bars": 60,
            "update_every_bars": 0,
            "update_every_seconds": 0,
            "bb_period": 20,
            "bb_std": 2.0,
            "bbw_squeeze_quantile": 0.2,
            "bbw_expand_quantile": 0.8,
            "bbw_expand_lookback": 20,
            "squeeze_lookback": 20,
            "slope_ema_period": 20,
            "slope_lookback": 5,
            "slope_quantile": 0.7,
            "slope_use_atr": True,
            "slope_atr_period": 14,
            "min_body_ratio": 0.0,
        }
    )


def test_trend_guard_veto_breakout_down_long() -> None:
    df = _build_downtrend_df()
    guard = _make_guard()

    result = guard.check_veto(
        symbol="BTC/USDT",
        side="long",
        current_candle=None,
        dataframe=df,
        timeframe="5m",
    )

    assert result.is_vetoed is True
    assert result.reason == "trend_guard_veto_long_breakout_down"
    assert result.meta_data.get("breakout_dir") == "down"


def test_trend_guard_pass_breakout_down_short() -> None:
    df = _build_downtrend_df()
    guard = _make_guard()

    result = guard.check_veto(
        symbol="BTC/USDT",
        side="short",
        current_candle=None,
        dataframe=df,
        timeframe="5m",
    )

    assert result.is_vetoed is False
    assert result.reason == "trend_guard_pass"
