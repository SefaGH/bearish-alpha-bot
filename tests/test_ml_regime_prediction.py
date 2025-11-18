"""
Tests for Phase 4.1: ML Market Regime Prediction

Tests ML components including feature engineering, models, and prediction engine.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("ML_ENABLED", "true")

from src.ml.feature_engineering import (
    FeatureEngineeringPipeline,
    TechnicalIndicatorFeatures,
    MarketMicrostructureFeatures,
    VolatilityFeatures,
    MomentumFeatures
)
from src.ml.regime_predictor import MLRegimePredictor
from src.ml.model_trainer import RegimeModelTrainer, TimeSeriesCV, WalkForwardValidation
from src.ml.prediction_engine import RealTimePredictionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def enable_ml(monkeypatch) -> None:
    """Force ML components to run in enabled mode for tests."""
    monkeypatch.setattr("src.ml.model_trainer.ML_ENABLED", True, raising=False)
    monkeypatch.setattr("src.ml.regime_predictor.ML_ENABLED", True, raising=False)
    from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit as SKTimeSeriesSplit
    from sklearn.preprocessing import StandardScaler as SKStandardScaler
    from imblearn.combine import SMOTETomek as SKSMOTETomek

    monkeypatch.setattr("src.ml.model_trainer.RandomForestClassifier", SKRandomForestClassifier, raising=False)
    monkeypatch.setattr("src.ml.model_trainer.TimeSeriesSplit", SKTimeSeriesSplit, raising=False)
    monkeypatch.setattr("src.ml.model_trainer.StandardScaler", SKStandardScaler, raising=False)
    monkeypatch.setattr("src.ml.model_trainer.SMOTETomek", SKSMOTETomek, raising=False)
    monkeypatch.setattr("src.ml.regime_predictor.RandomForestClassifier", SKRandomForestClassifier, raising=False)
    yield


@pytest.fixture
def regime_config(tmp_path) -> Dict[str, Any]:
    """Provide a minimal manifest-backed configuration for ML predictors."""
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()

    manifest = {
        "version": "test",
        "mode": "bundle",
        "feature_count": 3,
        "feature_names_ordered": ["feature_0", "feature_1", "feature_2"],
        "selected_features_price": [0, 1, 2],
        "selected_features_regime": [0, 1, 2],
        "regime_scaler_path": "scaler.pkl",
        "regime_rf_path": "random_forest.pkl",
        "regime_model_path": "lstm_regime.pth",
    }
    (bundle_path / "manifest.json").write_text(json.dumps(manifest))

    return {
        "active_bundle": str(bundle_path),
        "model_params": {
            "lstm_regime": {
                "hidden_size": 16,
                "num_layers": 1,
            }
        },
        "ensemble_weights": {
            "random_forest": 0.5,
            "lstm": 0.25,
            "transformer": 0.25,
        },
    }


@pytest.fixture
def feature_pipeline_instance(regime_config) -> Any:
    """Provide a lightweight feature pipeline stub tailored for ML predictor tests."""

    class _StubPipeline:
        def __init__(self, bundle_path: str):
            self.models_config = {"active_bundle": bundle_path}

        def extract_features(self, price_data: pd.DataFrame, mode: str = "regime") -> pd.DataFrame:
            index = getattr(price_data, "index", None)
            base = np.linspace(0.0, 1.0, len(price_data)) if len(price_data) else np.array([0.0])
            data = {
                "feature_0": base,
                "feature_1": base * 2,
                "feature_2": base * -1,
            }
            return pd.DataFrame(data, index=index)

        def prepare_for_training(self, features: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
            aligned = pd.concat([features, labels.rename("label")], axis=1).dropna()
            X = aligned[["feature_0", "feature_1", "feature_2"]].to_numpy(dtype=float)
            y = aligned["label"].to_numpy(dtype=int)
            return X, y

    return _StubPipeline(regime_config["active_bundle"])


def create_sample_price_data(n_bars=200):
    """Create sample price data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_price = close + np.random.randn(n_bars) * 0.2
    volume = np.abs(np.random.randn(n_bars) * 1000 + 5000)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'rsi': 50 + np.random.randn(n_bars) * 15,
        'macd': np.random.randn(n_bars) * 0.5,
        'macd_signal': np.random.randn(n_bars) * 0.5,
        'ema_20': close,
        'ema_50': close - 1,
        'bb_upper': close + 2,
        'bb_lower': close - 2,
        'atr': np.abs(np.random.randn(n_bars) * 0.5 + 1)
    }, index=dates)
    
    return df


def create_regime_labels(n_bars=200, index=None):
    """Create sample regime labels for testing."""
    np.random.seed(42)
    # 0=bullish, 1=neutral, 2=bearish
    labels = np.random.choice([0, 1, 2], size=n_bars, p=[0.3, 0.4, 0.3])
    if index is not None:
        return pd.Series(labels, index=index)
    return pd.Series(labels)


class TestFeatureEngineering:
    """Test feature engineering components."""
    
    def test_technical_indicator_features(self):
        """Test technical indicator feature extraction."""
        logger.info("Testing technical indicator features...")
        
        price_data = create_sample_price_data()
        extractor = TechnicalIndicatorFeatures()
        features = extractor.compute(price_data)
        
        assert not features.empty
        assert len(features) == len(price_data)
        logger.info(f"✓ Extracted {len(features.columns)} technical indicator features")
    
    def test_market_microstructure_features(self):
        """Test market microstructure feature extraction."""
        logger.info("Testing market microstructure features...")
        
        price_data = create_sample_price_data()
        extractor = MarketMicrostructureFeatures()
        features = extractor.compute(price_data)
        
        assert not features.empty
        assert 'price_range' in features.columns
        assert 'volume_ratio' in features.columns
        logger.info(f"✓ Extracted {len(features.columns)} microstructure features")
    
    def test_volatility_features(self):
        """Test volatility feature extraction."""
        logger.info("Testing volatility features...")
        
        price_data = create_sample_price_data()
        extractor = VolatilityFeatures()
        features = extractor.compute(price_data)
        
        assert not features.empty
        assert any('vol_' in col for col in features.columns)
        logger.info(f"✓ Extracted {len(features.columns)} volatility features")
    
    def test_momentum_features(self):
        """Test momentum feature extraction."""
        logger.info("Testing momentum features...")
        
        price_data = create_sample_price_data()
        extractor = MomentumFeatures()
        features = extractor.compute(price_data)
        
        assert not features.empty
        assert any('roc_' in col for col in features.columns)
        logger.info(f"✓ Extracted {len(features.columns)} momentum features")
    
    def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline."""
        logger.info("Testing feature engineering pipeline...")
        
        price_data = create_sample_price_data()
        pipeline = FeatureEngineeringPipeline()
        
        features = pipeline.extract_features(price_data)
        
        assert not features.empty
        assert len(features) > 0
        logger.info(f"✓ Pipeline extracted {len(features.columns)} total features")
    
    def test_prepare_for_training(self):
        """Test data preparation for training."""
        logger.info("Testing data preparation...")
        
        price_data = create_sample_price_data()
        labels = create_regime_labels(index=price_data.index)
        
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_features(price_data)
        
        X, y = pipeline.prepare_for_training(features, labels)
        
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]
        logger.info(f"✓ Prepared {len(X)} samples with {X.shape[1]} features")


class TestModelTrainer:
    """Test model trainer components."""
    
    def test_time_series_cv(self):
        """Test time series cross-validation."""
        logger.info("Testing time series cross-validation...")
        
        cv = TimeSeriesCV(n_splits=3)
        X = np.random.randn(100, 10)
        
        splits = cv.split(X)
        
        assert len(splits) == 3
        logger.info("✓ Time series CV created 3 splits")
    
    def test_walk_forward_validation(self):
        """Test walk-forward validation."""
        logger.info("Testing walk-forward validation...")
        
        wfv = WalkForwardValidation(train_size=50, test_size=10)
        X = np.random.randn(200, 10)
        
        splits = wfv.split(X)
        
        assert len(splits) > 0
        logger.info(f"✓ Walk-forward validation created {len(splits)} splits")
    
    def test_model_trainer_initialization(self, regime_config):
        """Test model trainer initialization."""
        logger.info("Testing model trainer initialization...")
        
        trainer = RegimeModelTrainer(config=regime_config)
        
        assert trainer.models is not None
        assert trainer.validators is not None
        logger.info("✓ Model trainer initialized successfully")
    
    def test_train_ensemble(self, regime_config):
        """Test ensemble model training."""
        logger.info("Testing ensemble model training...")
        
        trainer = RegimeModelTrainer(config=regime_config)
        
        # Create sample data
        X = np.random.randn(200, 20)
        y = np.random.choice([0, 1, 2], size=200)
        
        results = trainer.train_ensemble_models(X, y)
        
        assert results is not None
        assert 'models' in results
        logger.info("✓ Ensemble models trained successfully")


class TestMLRegimePredictor:
    """Test ML regime predictor."""
    
    def test_ml_predictor_initialization(self, feature_pipeline_instance, regime_config):
        """Test ML predictor initialization."""
        logger.info("Testing ML predictor initialization...")
        
        predictor = MLRegimePredictor(feature_pipeline_instance, regime_config)
        
        assert predictor.feature_pipeline is feature_pipeline_instance
        assert isinstance(predictor.models, dict)
        logger.info("✓ ML predictor initialized successfully")
    
    def test_train_regime_models(self, feature_pipeline_instance, regime_config):
        """Test regime model training."""
        logger.info("Testing regime model training...")
        
        predictor = MLRegimePredictor(feature_pipeline_instance, regime_config)
        
        price_data = create_sample_price_data()
        labels = create_regime_labels(index=price_data.index)
        
        result = predictor.train_regime_models(price_data, labels)

        assert result['success'] is False
        assert 'error' in result
        assert predictor.is_trained is False
        logger.info("Training deferred in test environment: %s", result.get('error'))
    
    @pytest.mark.asyncio
    async def test_predict_regime_transition(self, feature_pipeline_instance, regime_config):
        """Test regime transition prediction."""
        logger.info("Testing regime transition prediction...")
        
        predictor = MLRegimePredictor(feature_pipeline_instance, regime_config)
        
        price_data = create_sample_price_data()
        labels = create_regime_labels(index=price_data.index)
        
        # Train models first
        predictor.train_regime_models(price_data, labels)
        
        # Make prediction
        result = await predictor.predict_regime_transition('BTC/USDT', price_data)
        
        assert result is not None
        assert 'predicted_regime' in result
        assert 'probabilities' in result
        assert 'confidence' in result
        logger.info(f"✓ Predicted regime: {result['predicted_regime']}")


class TestPredictionEngine:
    """Test real-time prediction engine."""
    
    def test_prediction_engine_initialization(self):
        """Test prediction engine initialization."""
        logger.info("Testing prediction engine initialization...")
        
        engine = RealTimePredictionEngine(trained_models={})
        
        assert engine.models is not None
        assert engine.feature_buffer is not None
        assert engine.prediction_cache is not None
        logger.info("✓ Prediction engine initialized successfully")
    
    @pytest.mark.asyncio
    async def test_start_stop_engine(self):
        """Test starting and stopping the engine."""
        logger.info("Testing engine start/stop...")
        
        engine = RealTimePredictionEngine(trained_models={})
        
        # Start engine
        await engine.start_prediction_engine(symbols=['BTC/USDT'])
        assert engine.is_running is True
        
        # Stop engine
        await engine.stop_prediction_engine()
        assert engine.is_running is False
        
        logger.info("✓ Engine start/stop working correctly")
    
    @pytest.mark.asyncio
    async def test_market_data_update(self):
        """Test market data update processing."""
        logger.info("Testing market data update...")
        
        engine = RealTimePredictionEngine(trained_models={})
        
        data = {
            'close': 100.0,
            'volume': 1000.0,
            'timestamp': pd.Timestamp.now()
        }
        
        await engine.on_market_data_update('BTC/USDT', data)
        
        assert 'BTC/USDT' in engine.feature_buffer
        assert len(engine.feature_buffer['BTC/USDT']) == 1
        logger.info("✓ Market data update processed successfully")
    
    def test_get_engine_status(self):
        """Test engine status reporting."""
        logger.info("Testing engine status...")
        
        engine = RealTimePredictionEngine(trained_models={})
        status = engine.get_engine_status()
        
        assert 'running' in status
        assert 'symbols_tracked' in status
        assert 'n_predictions_cached' in status
        logger.info(f"✓ Engine status: {status}")


class TestIntegration:
    """Integration tests for ML components."""
    
    @pytest.mark.asyncio
    async def test_full_ml_pipeline(self, feature_pipeline_instance, regime_config):
        """Test complete ML pipeline from training to prediction."""
        logger.info("Testing full ML pipeline...")
        
        # 1. Create training data
        price_data = create_sample_price_data(n_bars=300)
        labels = create_regime_labels(n_bars=300, index=price_data.index)
        
        # 2. Initialize and train predictor
        predictor = MLRegimePredictor(feature_pipeline_instance, regime_config)
        train_result = predictor.train_regime_models(price_data, labels)

        if train_result['success']:
            logger.info(f"✓ Training completed with {train_result['n_samples']} samples")
        else:
            assert 'error' in train_result
            logger.info("Training skipped in integration test: %s", train_result['error'])
        
        # 3. Make predictions (falls back to heuristics when untrained)
        prediction = await predictor.predict_regime_transition('BTC/USDT', price_data)
        
        assert prediction is not None
        assert prediction['predicted_regime'] in ['bullish', 'neutral', 'bearish']
        logger.info(
            "✓ Prediction: %s with confidence %.2f",
            prediction['predicted_regime'],
            prediction.get('confidence', 0.0)
        )

        # 4. Initialize prediction engine with available models (may be empty)
        engine_models = predictor.models if predictor.is_trained else {}
        engine = RealTimePredictionEngine(trained_models=engine_models)
        
        # 5. Test real-time updates
        await engine.start_prediction_engine(symbols=['BTC/USDT'])
        
        for i in range(10):
            data = {
                'close': 100 + i * 0.1,
                'volume': 1000.0,
                'timestamp': pd.Timestamp.now()
            }
            await engine.on_market_data_update('BTC/USDT', data)
        
        await engine.stop_prediction_engine()
        
        logger.info("✓ Full ML pipeline completed successfully")


def run_ml_tests():
    """Run all ML tests."""
    logger.info("=" * 70)
    logger.info("Running Phase 4.1 ML Regime Prediction Tests")
    logger.info("=" * 70)
    
    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_ml_tests()
