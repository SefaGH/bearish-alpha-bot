import asyncio

import pandas as pd

from src.strategies.mean_reversion import VWAPMeanReversion


def _make_strategy() -> VWAPMeanReversion:
    cfg = {
        "timeframe": "1m",
        "signal_timeframe": "5m",
        "min_rows": 2,
        "min_signal_rows": 2,
        "band_multiplier": 2.0,
        "adx_threshold": 25.0,
        "soft_deferral_threshold": 0.0,
        "dynamic_controller": {"enabled": False},
        "rsi_rebound_guard": {
            "enabled": True,
            "tf": "1m",
            "use_closed_only": True,
            "activation_rsi": 25.0,
            "activation_z_score": 2.2,
            "rebound_rsi": 27.0,
            "max_wait_s": 120,
        },
    }
    return VWAPMeanReversion(cfg)


def _build_frames(*, price: float, rsi_value: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=2, freq="min")
    vwap = 100.0
    vwap_std = 1.0
    lower = 99.0
    upper = 101.0
    df_vwap = pd.DataFrame(
        {
            "close": [price, price],
            "volume": [1.0, 1.0],
            "vwap": [vwap, vwap],
            "vwap_lower": [lower, lower],
            "vwap_upper": [upper, upper],
            "vwap_std": [vwap_std, vwap_std],
            "rsi": [rsi_value, rsi_value],
        },
        index=index,
    )
    df_sig = pd.DataFrame(
        {
            "close": [price, price],
            "adx": [20.0, 20.0],
        },
        index=index,
    )
    df_vwap.attrs["includes_forming"] = False
    df_sig.attrs["includes_forming"] = False
    return df_vwap, df_sig


def _run_signal(strategy: VWAPMeanReversion, symbol: str, df_vwap: pd.DataFrame, df_sig: pd.DataFrame):
    return asyncio.run(strategy.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig))


def _guard_state(strategy: VWAPMeanReversion, symbol: str) -> str:
    return strategy._rsi_guard_state_by_symbol.get(symbol, "IDLE")


def test_mr_guard_fast_path_normal_dip():
    strategy = _make_strategy()
    symbol = "BTC/USDT:USDT"
    df_vwap, df_sig = _build_frames(price=98.5, rsi_value=24.0)

    signal = _run_signal(strategy, symbol, df_vwap, df_sig)

    assert signal is not None
    assert signal["side"] == "buy"
    assert signal.get("rsi_guard_status") == "bypassed_low_z"
    assert _guard_state(strategy, symbol) == "IDLE"


def test_mr_guard_crash_activation():
    strategy = _make_strategy()
    symbol = "BTC/USDT:USDT"
    df_vwap, df_sig = _build_frames(price=97.0, rsi_value=24.0)

    signal = _run_signal(strategy, symbol, df_vwap, df_sig)

    assert signal is None
    assert _guard_state(strategy, symbol) == "ARMED"


def test_mr_guard_wait_phase():
    strategy = _make_strategy()
    symbol = "BTC/USDT:USDT"
    df_vwap, df_sig = _build_frames(price=97.0, rsi_value=24.0)
    _run_signal(strategy, symbol, df_vwap, df_sig)

    df_vwap, df_sig = _build_frames(price=96.5, rsi_value=20.0)
    signal = _run_signal(strategy, symbol, df_vwap, df_sig)

    assert signal is None
    assert _guard_state(strategy, symbol) == "ARMED"


def test_mr_guard_successful_rebound():
    strategy = _make_strategy()
    symbol = "BTC/USDT:USDT"
    df_vwap, df_sig = _build_frames(price=97.0, rsi_value=24.0)
    _run_signal(strategy, symbol, df_vwap, df_sig)

    df_vwap, df_sig = _build_frames(price=98.0, rsi_value=28.0)
    signal = _run_signal(strategy, symbol, df_vwap, df_sig)

    assert signal is not None
    assert signal["side"] == "buy"
    assert signal.get("rsi_guard_status") == "triggered_and_valid"
    assert _guard_state(strategy, symbol) == "IDLE"


def test_mr_guard_fake_rebound_resets():
    strategy = _make_strategy()
    symbol = "BTC/USDT:USDT"
    df_vwap, df_sig = _build_frames(price=97.0, rsi_value=24.0)
    _run_signal(strategy, symbol, df_vwap, df_sig)

    df_vwap, df_sig = _build_frames(price=100.0, rsi_value=28.0)
    signal = _run_signal(strategy, symbol, df_vwap, df_sig)

    assert signal is None
    assert _guard_state(strategy, symbol) == "IDLE"
