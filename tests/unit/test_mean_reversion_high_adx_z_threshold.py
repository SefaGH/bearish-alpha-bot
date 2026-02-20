import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.strategies.mean_reversion import VWAPMeanReversion


@pytest.mark.asyncio
async def test_high_adx_z_threshold_controls_dynamic_z_gate(caplog):
    """Blueprint: Opportunity #5 simulation.

    Setup: ADX=30 (>=25), Z=1.95.

    - Scenario A: high_adx_z_threshold unset (defaults to 2.0) => Dynamic Z veto.
    - Scenario B: high_adx_z_threshold=1.90 => passes Z-gate (no dynamic_z_veto).

    We assert Scenario B produces a stable downstream outcome (soft_deferral_event)
    to prove execution continues past the Z-gate.
    """

    symbol = "BTC/USDT:USDT"
    idx = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
        ]
    )

    # Make vwap=100, std=1 => z = (101.95 - 100)/1 = 1.95.
    # Keep price inside [lower, upper] and near upper to trigger a soft deferral when Z-gate passes.
    df_vwap = pd.DataFrame(
        {
            "vwap": [100.0, 100.0],
            "vwap_std": [1.0, 1.0],
            "vwap_lower": [98.0, 98.0],
            "vwap_upper": [102.0, 102.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )

    df_sig = pd.DataFrame(
        {
            "close": [101.95, 101.95],
            "adx": [30.0, 30.0],
        },
        index=idx,
    )

    base_cfg = {
        "timeframe": "1m",
        "signal_timeframe": "5m",
        "min_rows": 2,
        "min_signal_rows": 2,
        # Ensure ADX filter doesn't block soft deferral.
        "adx_threshold": 40.0,
        "soft_deferral_threshold": 0.005,
        "dynamic_controller": {"enabled": False},
    }

    caplog.set_level(logging.INFO)

    # Scenario A: default high_adx_z_threshold = 2.0 => veto at z=1.95.
    strategy_default = VWAPMeanReversion(dict(base_cfg))
    result_default = await strategy_default.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig)
    assert result_default is None
    assert any("Dynamic Z veto" in rec.message for rec in caplog.records)

    # Scenario B: high_adx_z_threshold = 1.90 => pass Z-gate.
    caplog.clear()
    cfg_optimized = dict(base_cfg)
    cfg_optimized["high_adx_z_threshold"] = 1.90
    strategy_optimized = VWAPMeanReversion(cfg_optimized)
    result_optimized = await strategy_optimized.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig)

    assert isinstance(result_optimized, dict)
    assert result_optimized.get("event_type") == "soft_deferral_event"
    assert result_optimized.get("reason_code") == "strategy.mean_reversion.near_miss"
    assert result_optimized.get("symbol") == symbol

    # Core assertion: it should not have been rejected by the Dynamic Z gate.
    assert not any("Dynamic Z veto" in rec.message for rec in caplog.records)

    # Sanity: the simulated ADX/Z align with the blueprint.
    assert float(result_optimized.get("condition_data", {}).get("adx")) == pytest.approx(30.0)
    assert result_optimized.get("setup_anchor_ts_ms") == int(
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc).timestamp() * 1000
    )


@pytest.mark.asyncio
async def test_high_adx_z_threshold_floor_is_configurable(caplog):
    """When threshold is below floor, default floor still protects unless explicitly lowered."""

    symbol = "BTC/USDT:USDT"
    idx = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
        ]
    )

    # vwap=100, std=1 -> z = 1.55
    df_vwap = pd.DataFrame(
        {
            "vwap": [100.0, 100.0],
            "vwap_std": [1.0, 1.0],
            "vwap_lower": [98.0, 98.0],
            "vwap_upper": [102.0, 102.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame(
        {
            "close": [101.55, 101.55],
            "adx": [30.0, 30.0],
        },
        index=idx,
    )

    base_cfg = {
        "timeframe": "1m",
        "signal_timeframe": "5m",
        "min_rows": 2,
        "min_signal_rows": 2,
        "adx_threshold": 40.0,
        "soft_deferral_threshold": 0.005,
        "dynamic_controller": {"enabled": False},
        "high_adx_z_threshold": 1.30,
    }

    caplog.set_level(logging.INFO)

    # Default floor is 1.60 -> 1.30 gets clamped to 1.60 and z=1.55 is vetoed.
    strategy_default_floor = VWAPMeanReversion(dict(base_cfg))
    result_default_floor = await strategy_default_floor.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig)
    assert result_default_floor is None
    assert any("Dynamic Z veto" in rec.message for rec in caplog.records)

    # Lower floor explicitly -> same z can pass the dynamic Z gate.
    caplog.clear()
    cfg_lower_floor = dict(base_cfg)
    cfg_lower_floor["high_adx_z_threshold_floor"] = 1.20
    strategy_lower_floor = VWAPMeanReversion(cfg_lower_floor)
    result_lower_floor = await strategy_lower_floor.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig)

    assert isinstance(result_lower_floor, dict)
    assert result_lower_floor.get("event_type") == "soft_deferral_event"
    assert not any("Dynamic Z veto" in rec.message for rec in caplog.records)
    gate = (result_lower_floor.get("condition_data") or {}).get("gate_telemetry") or {}
    assert gate.get("dynamic_z_passed") is True
    assert gate.get("entry_blocked_by_band") is True
