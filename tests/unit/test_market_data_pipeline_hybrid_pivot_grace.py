import os
import sys
from unittest.mock import patch

import pandas as pd

# Ensure src/ is importable when running via pytest.cmd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.market_data_pipeline import MarketDataPipeline


class _CollectorStub:
    def __init__(self, forming):
        self._forming = forming

    def get_forming_ohlcv(self, exchange, symbol, timeframe):
        return self._forming


class _WSStub:
    def __init__(self, forming):
        self.collector = _CollectorStub(forming)


def _make_closed_df(last_open_ms: int) -> pd.DataFrame:
    idx = pd.to_datetime([last_open_ms], unit="ms", utc=True)
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        },
        index=idx,
    )
    return df


def test_pivot_grace_prev_bucket_downgrades_closed_only_by_default():
    # 30m interval
    interval_ms = 30 * 60 * 1000
    expected_open = 2 * interval_ms
    now_ms = expected_open + 1000  # within grace
    forming_open_ms = expected_open - interval_ms  # prev bucket

    closed_df = _make_closed_df(forming_open_ms)
    forming = [forming_open_ms, 100.0, 101.0, 99.0, 100.2, 5.0]

    ws = _WSStub(forming)
    pipeline = MarketDataPipeline(
        exchanges={},
        config={"websocket": {"hybrid_pivot_grace_enabled": True, "hybrid_pivot_grace_ms": 90000}},
        websocket_manager=ws,
    )

    with patch("time.time", return_value=now_ms / 1000.0):
        out_df, merge_action, reason = pipeline._merge_forming_candle(
            closed_df,
            exchange="bingx",
            symbol="BTC/USDT:USDT",
            timeframe="30m",
            forming_last_update_ts=now_ms - 500,
        )

    assert out_df.attrs.get("includes_forming") is False
    assert out_df.attrs.get("fallback_reason") == "pivot_grace_prev_bucket"
    assert reason == "pivot_grace_prev_bucket"
    assert merge_action == "none"


def test_pivot_stale_prev_bucket_after_grace_is_fallback():
    interval_ms = 30 * 60 * 1000
    expected_open = 2 * interval_ms
    pivot_grace_ms = 90000
    now_ms = expected_open + pivot_grace_ms + 1000  # after grace
    forming_open_ms = expected_open - interval_ms

    closed_df = _make_closed_df(forming_open_ms)
    forming = [forming_open_ms, 100.0, 101.0, 99.0, 100.2, 5.0]

    ws = _WSStub(forming)
    pipeline = MarketDataPipeline(
        exchanges={},
        config={"websocket": {"hybrid_pivot_grace_enabled": True, "hybrid_pivot_grace_ms": pivot_grace_ms}},
        websocket_manager=ws,
    )

    with patch("time.time", return_value=now_ms / 1000.0):
        out_df, merge_action, reason = pipeline._merge_forming_candle(
            closed_df,
            exchange="bingx",
            symbol="BTC/USDT:USDT",
            timeframe="30m",
            forming_last_update_ts=now_ms - 500,
        )

    assert out_df.attrs.get("includes_forming") is False
    assert out_df.attrs.get("fallback_reason") == "pivot_stale_prev_bucket"
    assert reason == "pivot_stale_prev_bucket"
    assert merge_action == "none"


def test_pivot_grace_accept_prev_bucket_flag_allows_merge_when_fresh():
    interval_ms = 30 * 60 * 1000
    expected_open = 2 * interval_ms
    now_ms = expected_open + 1000  # within grace
    forming_open_ms = expected_open - interval_ms

    closed_df = _make_closed_df(forming_open_ms)
    forming = [forming_open_ms, 100.0, 101.0, 99.0, 100.2, 5.0]

    ws = _WSStub(forming)
    pipeline = MarketDataPipeline(
        exchanges={},
        config={
            "websocket": {
                "hybrid_pivot_grace_enabled": True,
                "hybrid_pivot_grace_ms": 90000,
                "pivot_grace_accept_prev_bucket": True,
                "forming_update_stale_ms": 15000,
            }
        },
        websocket_manager=ws,
    )

    with patch("time.time", return_value=now_ms / 1000.0):
        out_df, merge_action, reason = pipeline._merge_forming_candle(
            closed_df,
            exchange="bingx",
            symbol="BTC/USDT:USDT",
            timeframe="30m",
            forming_last_update_ts=now_ms - 500,  # fresh
        )

    assert out_df.attrs.get("includes_forming") is True
    assert out_df.attrs.get("fallback_reason") is None
    assert reason is None
    assert merge_action in ("replaced_last", "appended")
