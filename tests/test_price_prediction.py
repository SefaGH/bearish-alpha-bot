"""Focused tests for the advanced price prediction stack."""

import sys
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
from src.ml.strategy_integration import AIEnhancedStrategyAdapter


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
    pipeline.extract_features.return_value = pd.DataFrame([{"feature_0": 0.2}])
    pipeline.models_config = {}
    return pipeline


@pytest.fixture
def market_data_pipeline() -> MagicMock:
    candles = pd.DataFrame(
        {
            "open": np.linspace(100, 101, 32),
            "high": np.linspace(101, 102, 32),
            "low": np.linspace(99, 100, 32),
            "close": np.linspace(100.5, 101.5, 32),
            "volume": np.full(32, 1_000),
        }
    )
    pipeline = MagicMock()
    pipeline.get_latest_ohlcv = AsyncMock(return_value=candles)
    return pipeline


@pytest.fixture
def price_engine(market_data_pipeline: MagicMock, feature_pipeline: MagicMock) -> AdvancedPricePredictionEngine:
    engine = AdvancedPricePredictionEngine(
        market_data_pipeline=market_data_pipeline,
        feature_pipeline=feature_pipeline,
        config={
            "timeframes": ["5m"],
            "update_interval_seconds": 1,
            "cache_ttl_seconds": 30,
        },
    )
    return engine


@pytest.mark.asyncio
async def test_price_engine_updates_prediction_cache(price_engine: AdvancedPricePredictionEngine) -> None:
    symbol = "BTC/USDT:USDT"
    await price_engine._update_predictions([symbol])  # noqa: SLF001 - exercising internal helper

    assert symbol in price_engine.prediction_cache
    cached = price_engine.prediction_cache[symbol]
    assert cached["aggregated"]["forecast"].shape[0] == 1


def test_price_engine_generates_neutral_signal_when_cache_empty(price_engine: AdvancedPricePredictionEngine) -> None:
    signal = price_engine.generate_trading_signals("BTC/USDT", current_price=100.0)
    assert signal["signal"] == "neutral"
    assert signal["strength"] == 0.0


def test_price_engine_generates_signal_from_cache(price_engine: AdvancedPricePredictionEngine) -> None:
    price_engine.prediction_cache["BTC/USDT"] = {
        "aggregated": {
            "forecast": np.array([2.0]),
            "uncertainty": np.array([0.1]),
            "consensus_strength": 0.9,
        },
        "timestamp": pd.Timestamp.utcnow(),
    }

    signal = price_engine.generate_trading_signals("BTC/USDT", current_price=100.0)
    assert signal["signal"] in {"bullish", "neutral"}
    assert signal["forecast_price"] >= 100.0


@pytest.mark.asyncio
async def test_adapter_enhances_signal(price_engine: AdvancedPricePredictionEngine) -> None:
    regime_predictor = MagicMock()
    regime_predictor.predict_regime_transition = AsyncMock(
        return_value={"predicted_regime": "bullish", "confidence": 0.75}
    )

    adapter = AIEnhancedStrategyAdapter(
        price_engine,
        regime_predictor,
        config={
            "prediction": {"min_confidence_threshold": 0.4, "consensus_threshold": 0.5},
            "regime": {"min_confidence_hard_reject": 0.2, "min_confidence_full_weight": 0.6},
        },
    )

    price_engine.prediction_cache["BTC/USDT"] = {
        "aggregated": {
            "forecast": np.array([1.5]),
            "uncertainty": np.array([0.2]),
            "consensus_strength": 0.8,
        },
        "timestamp": pd.Timestamp.utcnow(),
    }

    base_signal = {"signal": "bullish", "strength": 0.6}
    enhancement = await adapter.enhance_strategy_signal(
        symbol="BTC/USDT",
        base_signal=base_signal,
        current_price=101.0,
        market_data_pipeline=MagicMock(get_latest_ohlcv=AsyncMock(return_value=pd.DataFrame())),
    )

    assert enhancement["original_signal"] == "bullish"
    assert "final_signal" in enhancement
    assert "recommendations" in enhancement
