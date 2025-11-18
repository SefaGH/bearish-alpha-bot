"""Phase 2 initialization regression tests aligned with current modules."""

import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.risk_config import RiskConfiguration
from src.core.performance_monitor import RealTimePerformanceMonitor
from src.core.portfolio_manager import PortfolioManager
from src.core.production_coordinator import ProductionCoordinator
from src.core.risk_manager import RiskManager
from src.ml.price_predictor import AdvancedPricePredictionEngine


@pytest.fixture(autouse=True)
def force_torch_available(monkeypatch) -> Iterator[None]:
    """Ensure tests run with torch flagged as available."""
    monkeypatch.setattr("src.ml.price_predictor.TORCH_AVAILABLE", True)
    yield


@pytest.fixture(autouse=True)
def stub_manifest_loader(monkeypatch) -> Iterator[None]:
    """Provide a minimal manifest so initialization avoids file I/O."""

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
    pipeline = MagicMock()
    pipeline.get_latest_ohlcv = AsyncMock()
    return pipeline


def test_advanced_price_prediction_engine_initializes_with_config(
    market_data_pipeline: MagicMock,
    feature_pipeline: MagicMock,
) -> None:
    engine = AdvancedPricePredictionEngine(
        market_data_pipeline=market_data_pipeline,
        feature_pipeline=feature_pipeline,
        config={
            "timeframes": ["5m"],
            "update_interval_seconds": 3,
            "cache_ttl_seconds": 15,
            "classification_to_pct_scale": 1.2,
        },
    )

    status = engine.get_engine_status()
    assert status["timeframes"] == ["5m"]
    assert status["update_interval"] == 3
    assert status["mode"] in {"fallback", "gemma"}


def test_advanced_price_prediction_engine_requires_torch(
    market_data_pipeline: MagicMock,
    feature_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.ml.price_predictor.TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        AdvancedPricePredictionEngine(
            market_data_pipeline=market_data_pipeline,
            feature_pipeline=feature_pipeline,
            config={},
        )


def test_portfolio_manager_register_strategy_returns_status() -> None:
    risk_config = RiskConfiguration(custom_limits={"equity_usd": 1_000})
    risk_manager = RiskManager(portfolio_value=1_000, risk_config=risk_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)

    result = portfolio_manager.register_strategy(
        strategy_name="test_strategy",
        strategy_instance=MagicMock(),
        initial_allocation=0.25,
    )

    assert result["status"] == "success"
    assert "success" not in result


def test_production_coordinator_register_strategy_surface() -> None:
    coordinator = ProductionCoordinator()
    result = coordinator.register_strategy(
        strategy_name="test_strategy",
        strategy_instance=MagicMock(),
        initial_allocation=0.25,
    )

    assert result["success"] is False
    assert "reason" in result
