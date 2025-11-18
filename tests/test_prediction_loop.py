"""Prediction loop tests for AdvancedPricePredictionEngine."""

import asyncio
import sys
from contextlib import suppress
from pathlib import Path
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.price_predictor import AdvancedPricePredictionEngine


@pytest.fixture(autouse=True)
def force_torch_available(monkeypatch) -> Iterator[None]:
    monkeypatch.setattr("src.ml.price_predictor.TORCH_AVAILABLE", True)
    yield


@pytest.fixture(autouse=True)
def stub_manifest_loader(monkeypatch) -> Iterator[None]:
    def _load(_self, _bundle):
        return {
            "version": "test",
            "feature_count": 1,
            "feature_names_ordered": ["feature_0"],
            "price_model_path": None,
            "gemma_price_model_path": None,
            "price_scaler_path": None,
            "gemma_price_scaler_path": None,
            "selected_features_price": [0],
            "selected_features_regime": [0],
        }

    monkeypatch.setattr("src.ml.price_predictor.ManifestManager.load_manifest", _load)
    yield


@pytest.fixture
def feature_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.extract_features.return_value = pd.DataFrame([{"feature_0": 0.1}])
    pipeline.models_config = {}
    return pipeline


@pytest.fixture
def market_data_pipeline() -> MagicMock:
    candles = pd.DataFrame(
        {
            "open": np.linspace(100, 101, 20),
            "high": np.linspace(101, 102, 20),
            "low": np.linspace(99, 100, 20),
            "close": np.linspace(100.5, 101.5, 20),
            "volume": np.full(20, 1_000),
        }
    )

    pipeline = MagicMock()
    pipeline.get_latest_ohlcv = AsyncMock(return_value=candles)
    return pipeline


@pytest.fixture
def engine(market_data_pipeline: MagicMock, feature_pipeline: MagicMock) -> AdvancedPricePredictionEngine:
    instance = AdvancedPricePredictionEngine(
        market_data_pipeline=market_data_pipeline,
        feature_pipeline=feature_pipeline,
        config={
            "timeframes": ["5m"],
            "update_interval_seconds": 0.05,
            "cache_ttl_seconds": 5,
        },
    )
    instance.update_interval = 0.01
    return instance


@pytest.mark.asyncio
async def test_prediction_loop_populates_cache(engine: AdvancedPricePredictionEngine) -> None:
    symbol = "BTC/USDT:USDT"
    task = asyncio.create_task(engine.start_prediction_loop([symbol]))

    await asyncio.sleep(0.05)
    await engine.stop_prediction_loop()

    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert symbol in engine.prediction_cache
    prediction = engine.prediction_cache[symbol]
    assert prediction["aggregated"].get("forecast") is not None


@pytest.mark.asyncio
async def test_get_price_forecast_returns_cached_prediction(engine: AdvancedPricePredictionEngine) -> None:
    symbol = "BTC/USDT"
    engine.prediction_cache[symbol] = {
        "aggregated": {
            "forecast": np.array([1.0]),
            "uncertainty": np.array([0.2]),
            "consensus_strength": 0.8,
        },
        "timestamp": pd.Timestamp.utcnow(),
    }

    forecast = engine.get_price_forecast(symbol)
    assert forecast is not None
    assert forecast["aggregated"]["consensus_strength"] == 0.8


def test_generate_trading_signals_reflects_forecast(engine: AdvancedPricePredictionEngine) -> None:
    symbol = "BTC/USDT"
    engine.prediction_cache[symbol] = {
        "aggregated": {
            "forecast": np.array([3.0]),
            "uncertainty": np.array([0.1]),
            "consensus_strength": 0.9,
        },
        "timestamp": pd.Timestamp.utcnow(),
    }

    signals = engine.generate_trading_signals(symbol, current_price=100.0, threshold=0.02)
    assert signals["signal"] == "bullish"
    assert signals["forecast_price"] > 100.0