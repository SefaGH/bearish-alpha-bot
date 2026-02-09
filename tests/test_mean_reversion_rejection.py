import pandas as pd
import pytest

from src.safety.signal_integrity_guard import SignalIntegrityGuard
from src.strategies.mean_reversion import VWAPMeanReversion


def _make_strategy(*, recheck_mode: str | None = None) -> VWAPMeanReversion:
    rejection_confirmation = {"enabled": True, "upper_wick_ratio_min": 0.8}
    if recheck_mode is not None:
        rejection_confirmation["recheck_mode"] = recheck_mode
    cfg = {
        "timeframe": "1m",
        "signal_timeframe": "5m",
        "price_source": "signal_close",
        "min_rows": 3,
        "min_signal_rows": 3,
        "vwap_lookback": 20,
        "band_multiplier": 2.0,
        "adx_threshold": 30.0,
        "rsi_rebound_guard": {"enabled": False},
        "rejection_confirmation": rejection_confirmation,
        "impulse_veto": {"enabled": True, "body_atr_mult": 1.5, "sum2_range_atr_mult": 2.5},
    }
    return VWAPMeanReversion(cfg)


def _make_vwap_df(rows: int, *, vwap: float, upper: float, lower: float, std: float) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
    data = {
        "close": [vwap] * rows,
        "volume": [1.0] * rows,
        "vwap": [vwap] * rows,
        "vwap_upper": [upper] * rows,
        "vwap_lower": [lower] * rows,
        "vwap_std": [std] * rows,
    }
    return pd.DataFrame(data, index=idx)


def _make_sig_df(rows: int, last_rows: list[dict], *, includes_forming: bool) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="5min")
    base = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adx": 20.0, "atr": 1.0}
    data = [base.copy() for _ in range(rows)]
    for offset, row in enumerate(last_rows):
        data[rows - len(last_rows) + offset].update(row)
    df = pd.DataFrame(data, index=idx)
    df.attrs["includes_forming"] = includes_forming
    return df


class DummyMarketDataPipeline:
    def __init__(self, price: float):
        self._price = float(price)

    async def get_latest_price(self, _symbol: str, timeframe: str = "1m"):
        return self._price


@pytest.mark.asyncio
async def test_episode_c_rejection_closed_only_uses_prev_candle():
    strategy = _make_strategy()
    df_vwap = _make_vwap_df(3, vwap=100.0, upper=101.0, lower=99.0, std=0.5)
    df_sig = _make_sig_df(
        3,
        [
            {"open": 100.0, "close": 101.0, "high": 102.0, "low": 99.0, "adx": 20.0, "atr": 1.0},
            {"open": 103.0, "close": 102.0, "high": 105.0, "low": 101.0, "adx": 20.0, "atr": 1.0},
        ],
        includes_forming=True,
    )

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": df_vwap, "df_sig": df_sig},
    )

    assert signal is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "last_row, expected_signal",
    [
        ({"open": 103.0, "close": 102.0, "high": 105.0, "low": 101.0, "adx": 20.0, "atr": 1.0}, True),
        ({"open": 103.0, "close": 102.0, "high": 103.1, "low": 101.0, "adx": 20.0, "atr": 1.0}, False),
        ({"open": 100.6, "close": 100.4, "high": 101.2, "low": 100.0, "adx": 20.0, "atr": 1.0}, True),
        ({"open": 100.6, "close": 100.4, "high": 100.8, "low": 100.0, "adx": 20.0, "atr": 1.0}, False),
    ],
)
async def test_episode_c_rejection_confirmation_short_cases(last_row, expected_signal):
    strategy = _make_strategy()
    df_vwap = _make_vwap_df(3, vwap=100.0, upper=101.0, lower=99.0, std=0.1)
    df_sig = _make_sig_df(3, [last_row], includes_forming=False)

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": df_vwap, "df_sig": df_sig},
    )

    if expected_signal:
        assert signal is not None
        assert signal["side"] == "sell"
    else:
        assert signal is None


@pytest.mark.asyncio
async def test_episode_c_impulse_telemetry_emitted():
    strategy = _make_strategy()
    df_vwap = _make_vwap_df(3, vwap=100.0, upper=101.0, lower=99.0, std=0.1)
    df_sig = _make_sig_df(
        3,
        [
            {"open": 102.0, "close": 101.5, "high": 103.0, "low": 101.0, "adx": 20.0, "atr": 1.0},
            {"open": 103.0, "close": 102.0, "high": 105.0, "low": 101.0, "adx": 20.0, "atr": 1.0},
        ],
        includes_forming=False,
    )

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": df_vwap, "df_sig": df_sig},
    )

    assert signal is not None
    impulse = signal["meta"]["impulse_guard"]
    assert impulse["body_atr_mult"] is not None
    assert impulse["sum2_range_atr_mult"] is not None
    assert impulse["trade_dir"] == "down"


@pytest.mark.asyncio
async def test_episode_c_impulse_veto_overrides_rejection_entry():
    strategy = _make_strategy()
    df_vwap = _make_vwap_df(3, vwap=100.0, upper=101.0, lower=99.0, std=0.1)
    df_sig = _make_sig_df(
        3,
        [
            {"open": 104.0, "close": 103.0, "high": 106.0, "low": 102.0, "adx": 20.0, "atr": 1.0},
            {"open": 104.0, "close": 102.0, "high": 106.0, "low": 101.0, "adx": 20.0, "atr": 1.0},
        ],
        includes_forming=False,
    )

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": df_vwap, "df_sig": df_sig},
    )
    assert signal is not None
    assert signal["side"] == "sell"

    impulse_meta = signal["meta"].get("impulse_guard", {})
    impulse_meta["is_shock_move"] = True
    impulse_meta["candle_dir"] = "up"
    impulse_meta["trade_dir"] = "down"
    impulse_meta["require_opposite"] = True
    signal["meta"]["impulse_guard"] = impulse_meta

    guard = SignalIntegrityGuard(
        {"signals": {"integrity_guard": {"enabled": True, "impulse_guard_enabled": True}}},
        DummyMarketDataPipeline(price=signal["entry"]),
    )
    result = await guard.validate(signal)
    assert result["valid"] is False
    assert result["reason"] == "impulse_shock"


@pytest.mark.asyncio
async def test_episode_c_integrity_guard_overrides_mr_allow():
    strategy = _make_strategy()
    df_vwap = _make_vwap_df(3, vwap=100.0, upper=101.0, lower=99.0, std=0.1)
    df_sig = _make_sig_df(
        3,
        [
            {"open": 102.0, "close": 101.5, "high": 104.0, "low": 100.5, "adx": 20.0, "atr": 5.0},
        ],
        includes_forming=False,
    )

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": df_vwap, "df_sig": df_sig},
    )
    assert signal is not None

    guard = SignalIntegrityGuard(
        {
            "signals": {
                "integrity_guard": {
                    "enabled": True,
                    "atr_guard_enabled": True,
                    "atr_guard_mult": 0.5,
                    "max_deviation_pct": 0.001,
                    "impulse_guard_enabled": False,
                }
            }
        },
        DummyMarketDataPipeline(price=signal["entry"] * 1.002),
    )
    result = await guard.validate(signal)
    assert result["valid"] is False
    assert result["reason"] == "price_moved_fast"


@pytest.mark.asyncio
async def test_recheck_rejection_confirmation_observe_mode_does_not_block_short():
    strategy = _make_strategy(recheck_mode="observe")
    df_vwap = _make_vwap_df(3, vwap=100.0, upper=101.0, lower=99.0, std=0.1)
    # Above upper band but green candle -> rejection would fail if enforced.
    df_sig = _make_sig_df(
        3,
        [{"open": 100.0, "close": 101.2, "high": 101.3, "low": 99.5, "adx": 20.0, "atr": 1.0}],
        includes_forming=False,
    )

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": df_vwap, "df_sig": df_sig},
        parent_pending_id="pending-1",
    )

    assert signal is not None
    assert signal["side"] == "sell"
    rej = signal.get("meta", {}).get("rejection_confirmation", {})
    assert rej.get("recheck_mode") == "observe"
    assert rej.get("observed_fail") is True
    assert rej.get("enforced") is False


@pytest.mark.asyncio
async def test_recheck_rejection_confirmation_enforce_mode_blocks_short():
    strategy = _make_strategy(recheck_mode="enforce")
    df_vwap = _make_vwap_df(3, vwap=100.0, upper=101.0, lower=99.0, std=0.1)
    # Same setup as observe test: should be rejected when enforce mode is active.
    df_sig = _make_sig_df(
        3,
        [{"open": 100.0, "close": 101.2, "high": 101.3, "low": 99.5, "adx": 20.0, "atr": 1.0}],
        includes_forming=False,
    )

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": df_vwap, "df_sig": df_sig},
        parent_pending_id="pending-1",
    )

    assert signal is None


@pytest.mark.asyncio
async def test_recheck_short_missing_ohlc_observe_mode_keeps_legacy_signal_and_meta():
    strategy = _make_strategy(recheck_mode="observe")
    df_vwap = _make_vwap_df(3, vwap=100.0, upper=101.0, lower=99.0, std=0.1)
    idx = pd.date_range("2024-01-01", periods=3, freq="5min")
    # Intentionally omit OHLC columns to exercise the "missing OHLC on recheck" path.
    df_sig = pd.DataFrame(
        {
            "close": [100.0, 100.5, 101.2],
            "adx": [20.0, 20.0, 20.0],
        },
        index=idx,
    )
    df_sig.attrs["includes_forming"] = False

    signal = await strategy.generate_signal(
        "BTC/USDT:USDT",
        market_data={"df_vwap": df_vwap, "df_sig": df_sig},
        parent_pending_id="pending-1",
    )

    assert signal is not None
    assert signal["side"] == "sell"
    rej = signal.get("meta", {}).get("rejection_confirmation", {})
    assert rej.get("evaluation") == "skipped_missing_ohlc"
    assert rej.get("recheck_mode") == "observe"
    assert rej.get("enforced") is False
