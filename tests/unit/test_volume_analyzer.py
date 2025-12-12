import asyncio

import pandas as pd
import pytest

from src.core.volume_analyzer import VolumeAnalyzer
from src.utils.volume_utils import get_bucket_rank, VOLUME_BUCKET_ORDER


class FakeMDP:
    def __init__(self, data_map):
        self.data_map = data_map

    async def get_latest_ohlcv(self, symbol: str, timeframe: str):
        df = self.data_map.get((symbol, timeframe))
        return df.copy() if df is not None else None


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
    # With trade volume sum=20 over 5m, baseline scaled ~8.33 → ratios ~2.4 → sigmoid ~0.84 → HIGH bucket
    assert ctx.bucket == "HIGH"
    assert 0.8 <= ctx.volume_strength <= 0.9
    assert ctx.last_updated_ts == 123.0


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
