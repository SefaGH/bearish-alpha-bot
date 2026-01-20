from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.core.indicators import add_indicators


VOL_COLS = [
    "vol_rs_bps",
    "vol_gk_bps",
    "vol_yz_bps",
    "vol_atr_bps",
    "vol_std_bps",
]


def _make_ohlcv_df(*, n: int = 40, base: float = 100.0, step: float = 0.25) -> pd.DataFrame:
    idx = pd.date_range(
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        periods=n,
        freq="1min",
    )

    close = base + (np.arange(n) * step)
    open_ = close - (step / 2.0)
    high = close + (step / 2.0)
    low = close - (step / 2.0)

    return pd.DataFrame(
        {
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": np.full(n, 1.0, dtype=float),
        },
        index=idx,
    )


def test_adv_vol_disabled_does_not_add_vol_columns():
    df = _make_ohlcv_df()
    df.attrs["timeframe"] = "1m"

    out = add_indicators(
        df,
        {
            "advanced_volatility": {
                "enabled": False,
                "enabled_timeframes": ["1m"],
                "window": 14,
                "ddof": 1,
            }
        },
    )

    for c in VOL_COLS:
        assert c not in out.columns


def test_adv_vol_enabled_timeframe_mismatch_skips_compute():
    df = _make_ohlcv_df()
    df.attrs["timeframe"] = "5m"

    out = add_indicators(
        df,
        {
            "advanced_volatility": {
                "enabled": True,
                "enabled_timeframes": ["1m"],
                "window": 14,
                "ddof": 1,
            }
        },
    )

    for c in VOL_COLS:
        assert c not in out.columns


def test_adv_vol_enabled_timeframe_match_adds_vol_columns():
    df = _make_ohlcv_df()
    df.attrs["timeframe"] = "1m"

    out = add_indicators(
        df,
        {
            "advanced_volatility": {
                "enabled": True,
                "enabled_timeframes": ["1m"],
                "window": 14,
                "ddof": 1,
            }
        },
    )

    for c in VOL_COLS:
        assert c in out.columns

    # sanity: by the end of the series (n >> window), we should have computed values
    tail = out[VOL_COLS].tail(1).iloc[0]
    assert tail.notna().any()


def test_adv_vol_missing_timeframe_fail_closed_by_default():
    df = _make_ohlcv_df()

    out = add_indicators(
        df,
        {
            "advanced_volatility": {
                "enabled": True,
                "allow_without_timeframe": False,
                "enabled_timeframes": ["1m"],
                "window": 14,
                "ddof": 1,
            }
        },
    )

    for c in VOL_COLS:
        assert c not in out.columns


def test_adv_vol_allow_without_timeframe_can_compute_when_enabled():
    df = _make_ohlcv_df()

    out = add_indicators(
        df,
        {
            "advanced_volatility": {
                "enabled": True,
                "allow_without_timeframe": True,
                "enabled_timeframes": ["1m"],
                "window": 14,
                "ddof": 1,
            }
        },
    )

    for c in VOL_COLS:
        assert c in out.columns


def test_adv_vol_window_guard_skips_compute():
    df = _make_ohlcv_df()
    df.attrs["timeframe"] = "1m"

    out = add_indicators(
        df,
        {
            "advanced_volatility": {
                "enabled": True,
                "enabled_timeframes": ["1m"],
                "window": 1,
                "ddof": 1,
            }
        },
    )

    for c in VOL_COLS:
        assert c not in out.columns


def test_adv_vol_ddof_guard_skips_compute():
    df = _make_ohlcv_df()
    df.attrs["timeframe"] = "1m"

    out = add_indicators(
        df,
        {
            "advanced_volatility": {
                "enabled": True,
                "enabled_timeframes": ["1m"],
                "window": 10,
                "ddof": 10,
            }
        },
    )

    for c in VOL_COLS:
        assert c not in out.columns


def test_adv_vol_invalid_prices_does_not_crash_and_yields_nan_vols():
    idx = pd.date_range(
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        periods=40,
        freq="1min",
    )
    df = pd.DataFrame(
        {
            "open": np.zeros(40, dtype=float),
            "high": np.zeros(40, dtype=float),
            "low": np.zeros(40, dtype=float),
            "close": np.zeros(40, dtype=float),
            "volume": np.ones(40, dtype=float),
        },
        index=idx,
    )
    df.attrs["timeframe"] = "1m"

    out = add_indicators(
        df,
        {
            "advanced_volatility": {
                "enabled": True,
                "enabled_timeframes": ["1m"],
                "window": 14,
                "ddof": 1,
            }
        },
    )

    for c in VOL_COLS:
        assert c in out.columns
        assert out[c].isna().all()
