from datetime import datetime, timezone

import pandas as pd
import pytest

from src.strategies.mean_reversion import VWAPMeanReversion


@pytest.mark.asyncio
async def test_soft_deferral_threshold_is_config_driven():
    symbol = "BTC/USDT:USDT"
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "vwap": [100.0, 100.0],
            "vwap_lower": [99.0, 99.0],
            "vwap_upper": [101.0, 101.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame(
        {
            "close": [99.3, 99.3],  # ~0.303% above lower band
            "adx": [10.0, 10.0],
        },
        index=idx,
    )

    strategy_tight = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "adx_threshold": 30,
            "soft_deferral_threshold": 0.002,
            "dynamic_controller": {"enabled": False},
        }
    )
    result_tight = await strategy_tight.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig)
    assert result_tight is None

    strategy_loose = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "adx_threshold": 30,
            "soft_deferral_threshold": 0.005,
            "dynamic_controller": {"enabled": False},
        }
    )
    result_loose = await strategy_loose.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig)
    assert isinstance(result_loose, dict)
    assert result_loose.get("event_type") == "soft_deferral_event"
    assert result_loose.get("reason_code") == "strategy.mean_reversion.near_miss"
    assert result_loose.get("symbol") == symbol
    assert result_loose.get("side") == "long"
    assert result_loose.get("timeframe") == "5m"
    assert result_loose.get("setup_anchor_ts_ms") == int(datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc).timestamp() * 1000)
    assert float(result_loose.get("condition_data", {}).get("threshold")) == 0.005

