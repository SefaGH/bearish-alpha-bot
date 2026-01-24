import asyncio
from unittest.mock import patch

import pandas as pd
import pytest

from src.core.volume_analyzer import VolumeAnalyzer
from src.utils.volume_utils import get_bucket_rank, VOLUME_BUCKET_ORDER


class FakeMDP:
    def __init__(self, data_map):
        self.data_map = data_map

    async def get_latest_ohlcv(self, symbol: str, timeframe: str, limit: int = None, **kwargs):
        df = self.data_map.get((symbol, timeframe))
        if df is None:
            return None
        out = df.copy()
        if limit is not None:
            return out.tail(limit)
        return out


class FakeCollector:
    def __init__(self, data_map):
        self.data_map = data_map

    def get_latest_ohlcv(self, exchange: str, symbol: str, timeframe: str, limit: int = None):
        data = self.data_map.get((exchange, symbol, timeframe))
        if not data:
            return None
        if limit is None:
            return data
        return data[-limit:]


class FakeWSManager:
    def __init__(self, collector):
        self.collector = collector


@pytest.mark.asyncio
async def test_compute_context_bucket_and_strength():
    symbol = "BTC/USDT:USDT"
    # Baseline on 1h, medium on 1h to simplify scaling; trade timeframe 5m
    baseline_df = pd.DataFrame({"volume": [100, 100, 100]})
    trade_df = pd.DataFrame({"volume": [10, 10, 10]})
    mdp = FakeMDP({(symbol, "1h"): baseline_df, (symbol, "5m"): trade_df})

    cfg = {
        "baseline_short_tf": "1h",
        "baseline_medium_tf": "1h",
        "short_lookback": 3,
        "medium_lookback": 3,
        "window_bars": 2,
        "weight_short": 0.6,
        "weight_medium": 0.4,
        "sigmoid_alpha": 1.2,
        "min_ratio": 0.1,
        "max_ratio": 10.0,
        "buckets": [
            (0.0, "LOW"),
            (0.3, "NORMAL"),
            (0.6, "HIGH"),
            (0.85, "EXTREME"),
        ],
    }

    analyzer = VolumeAnalyzer(mdp, cfg)
    ctx = await analyzer.compute_context(symbol, trade_timeframe="5m", as_of_ts=123.0)

    assert ctx is not None
    # With trade volume sum=20 over 10m, baseline scaled ~16.67 → ratios ~1.2 → sigmoid ~0.56 → NORMAL bucket
    assert ctx.bucket == "NORMAL"
    assert 0.5 <= ctx.volume_strength <= 0.7
    assert ctx.last_updated_ts == 123.0


@pytest.mark.asyncio
async def test_compute_context_prefers_stream_collector_buffers():
    symbol = "BTC/USDT:USDT"
    exchange = "bingx"
    tf = "5m"
    ts0 = 1_700_000_000_000
    candles = [[ts0 + i * 300_000, 1.0, 1.0, 1.0, 1.0, 100.0] for i in range(6)]

    collector = FakeCollector({(exchange, symbol, tf): candles})

    class MDPCollectorOnly:
        def __init__(self):
            self.websocket_manager = FakeWSManager(collector)
            self.exchanges = {exchange: object()}
            self.DEFAULT_EXCHANGE = exchange

        async def get_latest_ohlcv(self, *args, **kwargs):
            raise AssertionError("compute_context should use StreamDataCollector, not pipeline")

    cfg = {
        "baseline_short_tf": tf,
        "baseline_medium_tf": tf,
        "short_lookback": 5,
        "medium_lookback": 5,
        "window_bars": 2,
        "weight_short": 0.6,
        "weight_medium": 0.4,
        "sigmoid_alpha": 1.2,
        "min_ratio": 0.1,
        "max_ratio": 10.0,
        "buckets": [
            (0.0, "LOW"),
            (0.3, "NORMAL"),
            (0.6, "HIGH"),
            (0.85, "EXTREME"),
        ],
    }

    analyzer = VolumeAnalyzer(MDPCollectorOnly(), cfg)
    ctx = await analyzer.compute_context(symbol, trade_timeframe=tf, as_of_ts=123.0)

    assert ctx is not None
    assert ctx.volume_data_sources == {"trade": "collector", "short": "collector", "medium": "collector"}
    assert ctx.volume_data_limits == {"trade": 2, "short": 6, "medium": 6}


@pytest.mark.asyncio
async def test_compute_context_includes_forming_numerator_when_armed():
    symbol = "BTC/USDT:USDT"
    exchange = "bingx"
    tf = "5m"
    interval_ms = 300_000
    ts0 = 1_700_000_000_000

    candles = [[ts0 + i * interval_ms, 1.0, 1.0, 1.0, 1.0, 10.0] for i in range(6)]
    forming_open_ms = ts0 + 6 * interval_ms
    forming = [forming_open_ms, 1.0, 1.0, 1.0, 1.0, 100.0]
    now_ms = forming_open_ms + (interval_ms // 2)

    class CollectorWithForming(FakeCollector):
        def get_forming_ohlcv(self, exchange: str, symbol: str, timeframe: str):
            return forming

        def get_state(self, exchange: str, symbol: str, timeframe: str):
            return {"forming_last_update_ts": now_ms - 200, "last_closed_ts": candles[-1][0]}

    collector = CollectorWithForming({(exchange, symbol, tf): candles})

    class MDPCollectorOnly:
        def __init__(self):
            self.websocket_manager = FakeWSManager(collector)
            self.exchanges = {exchange: object()}
            self.DEFAULT_EXCHANGE = exchange

        async def get_latest_ohlcv(self, *args, **kwargs):
            raise AssertionError("compute_context should use StreamDataCollector, not pipeline")

    cfg = {
        "baseline_short_tf": tf,
        "baseline_medium_tf": tf,
        "short_lookback": 3,
        "medium_lookback": 3,
        "window_bars": 2,
        "weight_short": 0.6,
        "weight_medium": 0.4,
        "sigmoid_alpha": 1.2,
        "min_ratio": 0.1,
        "max_ratio": 10.0,
        "buckets": [
            (0.0, "LOW"),
            (0.3, "NORMAL"),
            (0.6, "HIGH"),
            (0.85, "EXTREME"),
        ],
        "forming_update_stale_ms": 3000,
        "forming_volume_cap_ratio": 0.6,
    }

    analyzer = VolumeAnalyzer(MDPCollectorOnly(), cfg)

    with patch("src.core.volume_analyzer.time.time", return_value=now_ms / 1000.0):
        ctx = await analyzer.compute_context(
            symbol,
            trade_timeframe=tf,
            shock_state="ARMED",
            include_forming_trade=True,
        )

    assert ctx is not None
    assert ctx.current_window_volume_closed == 20.0
    assert ctx.forming_volume_raw == 100.0
    assert 0.45 <= float(ctx.forming_elapsed_ratio) <= 0.55
    assert 49.0 <= float(ctx.forming_volume_added) <= 51.0
    assert 69.0 <= float(ctx.current_window_volume) <= 71.0
    assert ctx.current_window_volume_mode == "closed_plus_forming_numerator"


@pytest.mark.asyncio
async def test_compute_context_accepts_three_element_bucket_tuples():
    symbol = "BTC/USDT:USDT"
    baseline_df = pd.DataFrame({"volume": [100, 100, 100]})
    trade_df = pd.DataFrame({"volume": [50, 50, 50]})
    mdp = FakeMDP({(symbol, "1h"): baseline_df, (symbol, "5m"): trade_df})

    cfg = {
        "baseline_short_tf": "1h",
        "baseline_medium_tf": "1h",
        "short_lookback": 3,
        "medium_lookback": 3,
        "window_bars": 3,
        "weight_short": 0.6,
        "weight_medium": 0.4,
        "sigmoid_alpha": 1.2,
        "min_ratio": 0.1,
        "max_ratio": 10.0,
        # Real-world shape includes extra metadata (ignored)
        "buckets": [
            (0.0, "VERY_LOW", "gray"),
            (0.3, "LOW", "blue"),
            (0.6, "NORMAL", "green"),
            (0.85, "HIGH", "orange"),
            (0.95, "EXTREME", "red"),
        ],
    }

    analyzer = VolumeAnalyzer(mdp, cfg)
    ctx = await analyzer.compute_context(symbol, trade_timeframe="5m", as_of_ts=456.0)

    assert ctx is not None
    assert ctx.bucket in {"NORMAL", "HIGH", "EXTREME", "LOW", "VERY_LOW"}
    assert 0.0 <= ctx.volume_strength <= 1.0
    assert ctx.last_updated_ts == 456.0


@pytest.mark.asyncio
async def test_compute_context_accepts_dict_bucket_entries():
    symbol = "ETH/USDT:USDT"
    baseline_df = pd.DataFrame({"volume": [80, 80, 80]})
    trade_df = pd.DataFrame({"volume": [20, 25, 30]})
    mdp = FakeMDP({(symbol, "1h"): baseline_df, (symbol, "5m"): trade_df})

    cfg = {
        "baseline_short_tf": "1h",
        "baseline_medium_tf": "1h",
        "short_lookback": 3,
        "medium_lookback": 3,
        "window_bars": 3,
        "weight_short": 0.6,
        "weight_medium": 0.4,
        "sigmoid_alpha": 1.2,
        "min_ratio": 0.1,
        "max_ratio": 10.0,
        # Dict-based entries with extra metadata
        "buckets": [
            {"threshold": 0.0, "name": "VL", "color": "gray"},
            {"threshold": 0.4, "name": "L", "color": "blue"},
            {"threshold": 0.7, "name": "N", "color": "green"},
            {"threshold": 0.9, "name": "H", "color": "orange"},
        ],
    }

    analyzer = VolumeAnalyzer(mdp, cfg)
    ctx = await analyzer.compute_context(symbol, trade_timeframe="5m", as_of_ts=789.0)

    assert ctx is not None
    assert ctx.bucket in {"VL", "L", "N", "H"}
    assert 0.0 <= ctx.volume_strength <= 1.0
    assert ctx.last_updated_ts == 789.0


@pytest.mark.asyncio
async def test_compute_context_accepts_string_encoded_bucket_entries():
    symbol = "ADA/USDT:USDT"
    baseline_df = pd.DataFrame({"volume": [50, 55, 60]})
    trade_df = pd.DataFrame({"volume": [12, 13, 14]})
    mdp = FakeMDP({(symbol, "1h"): baseline_df, (symbol, "5m"): trade_df})

    cfg = {
        "baseline_short_tf": "1h",
        "baseline_medium_tf": "1h",
        "short_lookback": 3,
        "medium_lookback": 3,
        "window_bars": 2,
        "weight_short": 0.6,
        "weight_medium": 0.4,
        "sigmoid_alpha": 1.2,
        "min_ratio": 0.1,
        "max_ratio": 10.0,
        # Simulate output of minimal YAML parser (stringified entries)
        "buckets": [
            '[0.0, "LOW"]',
            '[0.3, "NORMAL"]',
            '[0.6, "HIGH"]',
            '[0.85, "EXTREME"]',
        ],
    }

    analyzer = VolumeAnalyzer(mdp, cfg)

    assert analyzer._buckets
    assert all(isinstance(b, dict) for b in analyzer._buckets)

    ctx = await analyzer.compute_context(symbol, trade_timeframe="5m", as_of_ts=111.0)
    # Context may be None if data insufficient, but should not raise and, if present, bucket name must be valid.
    assert ctx is None or ctx.bucket in {"LOW", "NORMAL", "HIGH", "EXTREME"}


def test_bucket_rank_helper_defaults_to_normal():
    assert get_bucket_rank("LOW") == VOLUME_BUCKET_ORDER["LOW"]
    assert get_bucket_rank("EXTREME") == VOLUME_BUCKET_ORDER["EXTREME"]
    assert get_bucket_rank("UNKNOWN") == VOLUME_BUCKET_ORDER["NORMAL"]
