import pandas as pd
import pytest

from src.strategies.mean_reversion import VWAPMeanReversion


def _build_vwap_df() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
            "vwap": [100.0, 100.0, 100.0],
            "vwap_upper": [101.0, 101.0, 101.0],
            "vwap_lower": [99.0, 99.0, 99.0],
            "vwap_std": [0.1, 0.1, 0.1],
        },
        index=idx,
    )


def _build_sig_df() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    # Last candle: red, upper touch, close above upper -> valid short context.
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 103.0],
            "high": [100.5, 100.5, 105.0],
            "low": [99.5, 99.5, 101.0],
            "close": [100.0, 100.0, 102.0],
            "adx": [10.0, 10.0, 10.0],
            "atr": [1.0, 1.0, 1.0],
        },
        index=idx,
    )


def _cfg() -> dict:
    return {
        "timeframe": "1m",
        "signal_timeframe": "5m",
        "price_source": "signal_close",
        "min_rows": 3,
        "min_signal_rows": 3,
        "vwap_lookback": 20,
        "band_multiplier": 2.0,
        "adx_threshold": 30.0,
        "dynamic_controller": {"enabled": False},
        "rsi_rebound_guard": {"enabled": False},
        "rejection_confirmation": {"enabled": True, "upper_wick_ratio_min": 0.8},
    }


@pytest.mark.asyncio
async def test_short_regime_policy_opt_in_disabled_blocks_short():
    cfg = _cfg()
    cfg["regime_policy"] = {
        "enabled": True,
        "trend_up": {
            "short_mode": "disabled",
            "adx_floor": 25,
        },
    }
    strategy = VWAPMeanReversion(cfg)

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": _build_vwap_df(), "df_sig": _build_sig_df()},
        regime_data={"trend": "bullish", "trend_strength": 30.0},
    )

    assert signal is None


@pytest.mark.asyncio
async def test_short_regime_policy_off_keeps_legacy_short_entry():
    cfg = _cfg()
    # trend_up absent -> opt-in disabled -> preserve legacy behavior
    cfg["regime_policy"] = {"enabled": True}
    strategy = VWAPMeanReversion(cfg)

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": _build_vwap_df(), "df_sig": _build_sig_df()},
        regime_data={"trend": "bullish", "trend_strength": 30.0},
    )

    assert signal is not None
    assert signal["side"] == "sell"
