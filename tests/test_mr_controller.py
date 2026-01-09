from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.core.indicators import add_indicators
from src.strategies.mean_reversion import VWAPMeanReversion
from src.strategies.mr_controller import DynamicMRController


def test_mr_controller_disabled_preserves_pipeline_bands():
    df_vwap = pd.DataFrame(
        [
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "vwap": 100.0,
                "vwap_std": 10.0,
                "vwap_lower": 80.0,   # pipeline bands assume 2.0x
                "vwap_upper": 120.0,
            },
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "vwap": 100.0,
                "vwap_std": 10.0,
                "vwap_lower": 80.0,
                "vwap_upper": 120.0,
            },
        ],
        index=[
            datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        ],
    )

    df_sig = pd.DataFrame(
        [
            {
                "open": 115.0,
                "high": 116.0,
                "low": 114.0,
                "close": 115.0,  # inside pipeline [80, 120], outside 1.0x if recomputed
                "volume": 1.0,
                "adx": 10.0,
                "atr": 1.0,
            }
        ],
        index=[datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
    )

    strategy = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "band_multiplier": 1.0,  # would generate a SELL if strategy recomputed bands locally
            "vwap_lookback": 1440,
            "adx_threshold": 25,
            "min_rows": 2,
            "min_signal_rows": 1,
            "dynamic_controller": {"enabled": False},
        }
    )

    signal = asyncio.run(strategy.generate_signal(symbol="BTC/USDT:USDT", df_vwap=df_vwap, df_sig=df_sig))
    assert signal is None


def test_mr_controller_clamps_band_multiplier():
    controller = DynamicMRController(
        {
            "enabled": True,
            "warmup_samples": 1,
            "abs_z_window": 10,
            "update_interval_sec": 0,
            "min_m_change": 0.0,
            "m_min": 1.0,
            "m_max": 2.0,
            "freeze_on_trend": False,
            "log_every_update": False,
        },
        static_band_multiplier=1.0,
        static_lookback=1440,
    )

    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    decision = controller.evaluate(
        symbol="BTC/USDT:USDT",
        ts=ts,
        price=100.0,
        vwap=0.0,
        vwap_std=1.0,
        adx=0.0,
        atr=None,
        df_vwap=None,
    )

    assert decision.band_multiplier == 2.0
    assert decision.lower == -2.0
    assert decision.upper == 2.0


def test_mr_controller_respects_update_interval():
    controller = DynamicMRController(
        {
            "enabled": True,
            "warmup_samples": 1,
            "abs_z_window": 50,
            "update_interval_sec": 300,
            "min_m_change": 0.0,
            "m_min": 1.0,
            "m_max": 3.0,
            "freeze_on_trend": False,
            "log_every_update": False,
        },
        static_band_multiplier=1.0,
        static_lookback=1440,
    )

    ts0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    d0 = controller.evaluate(
        symbol="BTC/USDT:USDT",
        ts=ts0,
        price=1.0,   # abs_z = 1.0 -> m_eff ~ 1.0
        vwap=0.0,
        vwap_std=1.0,
        adx=0.0,
        atr=None,
        df_vwap=None,
    )
    assert d0.band_multiplier == 1.0
    assert d0.updated is True

    d1 = controller.evaluate(
        symbol="BTC/USDT:USDT",
        ts=ts0 + timedelta(seconds=60),
        price=3.0,   # would push m_eff to 3.0, but update interval blocks
        vwap=0.0,
        vwap_std=1.0,
        adx=0.0,
        atr=None,
        df_vwap=None,
    )
    assert d1.updated is False
    assert d1.band_multiplier == 1.0

    d2 = controller.evaluate(
        symbol="BTC/USDT:USDT",
        ts=ts0 + timedelta(seconds=301),
        price=3.0,
        vwap=0.0,
        vwap_std=1.0,
        adx=0.0,
        atr=None,
        df_vwap=None,
    )
    assert d2.updated is True
    assert d2.band_multiplier == 3.0


def test_mr_controller_local_bands_match_pipeline_math():
    lookback = 20
    band_mult = 2.0

    idx = pd.date_range("2026-01-01", periods=60, freq="min", tz="UTC")
    close = pd.Series(range(len(idx)), index=idx, dtype=float) * 0.1 + 100.0
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10.0,
        },
        index=idx,
    )

    ind = add_indicators(df.copy(), {"vwap_lookback": lookback, "vwap_band_multiplier": band_mult})
    last = ind.iloc[-1]

    controller = DynamicMRController(
        {
            "enabled": True,
            "warmup_samples": 9999,  # keep m_eff == static
            "abs_z_window": 50,
            "update_interval_sec": 0,
            "min_m_change": 0.0,
            "m_min": band_mult,
            "m_max": band_mult,
            "freeze_on_trend": False,
            "log_every_update": False,
            "dynamic_lookback": {
                "enabled": True,
                "lookback_static": lookback,
                "lookback_min": lookback,
                "lookback_max": lookback,
            },
        },
        static_band_multiplier=band_mult,
        static_lookback=lookback,
    )

    decision = controller.evaluate(
        symbol="BTC/USDT:USDT",
        ts=idx[-1].to_pydatetime(),
        price=float(close.iloc[-1]),
        vwap=float(last["vwap"]),
        vwap_std=float(last["vwap_std"]),
        adx=0.0,
        atr=None,
        df_vwap=df,
    )

    assert decision.vwap == pytest.approx(float(last["vwap"]), abs=1e-9)
    assert decision.vwap_std == pytest.approx(float(last["vwap_std"]), abs=1e-9)
    assert decision.lower == pytest.approx(float(last["vwap_lower"]), abs=1e-9)
    assert decision.upper == pytest.approx(float(last["vwap_upper"]), abs=1e-9)


def test_mr_controller_cache_invalidates_on_last_row_change():
    lookback = 20
    band_mult = 2.0
    idx = pd.date_range("2026-01-01", periods=60, freq="min", tz="UTC")
    close = pd.Series(range(len(idx)), index=idx, dtype=float) * 0.1 + 100.0
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10.0,
        },
        index=idx,
    )

    controller = DynamicMRController(
        {
            "enabled": True,
            "warmup_samples": 9999,
            "abs_z_window": 50,
            "update_interval_sec": 0,
            "min_m_change": 0.0,
            "m_min": band_mult,
            "m_max": band_mult,
            "freeze_on_trend": False,
            "log_every_update": False,
            "dynamic_lookback": {
                "enabled": True,
                "lookback_static": lookback,
                "lookback_min": lookback,
                "lookback_max": lookback,
            },
        },
        static_band_multiplier=band_mult,
        static_lookback=lookback,
    )

    ind = add_indicators(df.copy(), {"vwap_lookback": lookback, "vwap_band_multiplier": band_mult})
    last = ind.iloc[-1]

    d1 = controller.evaluate(
        symbol="BTC/USDT:USDT",
        ts=idx[-1].to_pydatetime(),
        price=float(close.iloc[-1]),
        vwap=float(last["vwap"]),
        vwap_std=float(last["vwap_std"]),
        adx=0.0,
        atr=None,
        df_vwap=df,
        is_forming_candle=True,
    )

    df2 = df.copy()
    df2.loc[df2.index[-1], "close"] = float(df2.loc[df2.index[-1], "close"]) + 50.0
    df2.loc[df2.index[-1], "high"] = float(df2.loc[df2.index[-1], "high"]) + 50.0
    df2.loc[df2.index[-1], "low"] = float(df2.loc[df2.index[-1], "low"]) + 50.0
    df2.loc[df2.index[-1], "volume"] = float(df2.loc[df2.index[-1], "volume"]) + 1000.0

    ind2 = add_indicators(df2.copy(), {"vwap_lookback": lookback, "vwap_band_multiplier": band_mult})
    last2 = ind2.iloc[-1]

    d2 = controller.evaluate(
        symbol="BTC/USDT:USDT",
        ts=idx[-1].to_pydatetime(),
        price=float(df2["close"].iloc[-1]),
        vwap=float(last2["vwap"]),
        vwap_std=float(last2["vwap_std"]),
        adx=0.0,
        atr=None,
        df_vwap=df2,
        is_forming_candle=True,
    )

    assert d1.vwap != d2.vwap


def test_mr_controller_z_uses_effective_vwap_std():
    lookback = 20
    band_mult = 2.0
    idx = pd.date_range("2026-01-01", periods=60, freq="min", tz="UTC")
    close = pd.Series(range(len(idx)), index=idx, dtype=float) * 0.1 + 100.0
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10.0,
        },
        index=idx,
    )

    controller = DynamicMRController(
        {
            "enabled": True,
            "warmup_samples": 9999,
            "abs_z_window": 50,
            "update_interval_sec": 0,
            "min_m_change": 0.0,
            "m_min": band_mult,
            "m_max": band_mult,
            "freeze_on_trend": False,
            "log_every_update": False,
            "dynamic_lookback": {
                "enabled": True,
                "lookback_static": lookback,
                "lookback_min": lookback,
                "lookback_max": lookback,
            },
        },
        static_band_multiplier=band_mult,
        static_lookback=lookback,
    )

    # Force a mismatch between input vwap/vwap_std and locally recomputed effective values.
    price = float(df["close"].iloc[-1]) + 10.0
    decision = controller.evaluate(
        symbol="BTC/USDT:USDT",
        ts=idx[-1].to_pydatetime(),
        price=price,
        vwap=0.0,
        vwap_std=9999.0,
        adx=0.0,
        atr=None,
        df_vwap=df,
    )

    assert decision.z == pytest.approx((price - decision.vwap) / decision.vwap_std, rel=1e-12, abs=1e-12)
