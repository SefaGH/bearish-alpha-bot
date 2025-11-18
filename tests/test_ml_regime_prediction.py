"""
Tests for Phase 4.1: ML Market Regime Prediction

Tests ML components including feature engineering, models, and prediction engine.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import logging
from unittest.mock import Mock, patch

from src.ml.feature_engineering import (
    FeatureEngineeringPipeline,
    TechnicalIndicatorFeatures,
    MarketMicrostructureFeatures,
    VolatilityFeatures,
    MomentumFeatures
)
from src.ml.regime_predictor import MLRegimePredictor, EnsembleRegimePredictor
from src.ml.model_trainer import RegimeModelTrainer, TimeSeriesCV, WalkForwardValidation
from src.ml.prediction_engine import RealTimePredictionEngine
from src.config.ml_config import MLConfiguration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class TestMLConfiguration:
    """Test ML configuration."""
    
    def test_model_config(self):
        """Test model configuration."""
        logger.info("Testing model configuration...")
        
        config = MLConfiguration.get_model_config()
        
        assert config.lstm is not None
        assert config.transformer is not None
        assert config.ensemble_weights is not None
        logger.info("✓ Model configuration loaded successfully")
    
    def test_training_config(self):
        """Test training configuration."""
        logger.info("Testing training configuration...")
        
        config = MLConfiguration.get_training_config()
        
        assert config.sequence_length > 0
        assert config.prediction_horizon > 0
        assert config.batch_size > 0
        logger.info("✓ Training configuration loaded successfully")
    
    def test_feature_config(self):
        """Test feature configuration."""
        logger.info("Testing feature configuration...")
        
        config = MLConfiguration.get_feature_config()
        
        assert config.technical_indicators is True
        assert config.volatility_features is True
        logger.info("✓ Feature configuration loaded successfully")


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
    
    def test_model_trainer_initialization(self):
        """Test model trainer initialization."""
        logger.info("Testing model trainer initialization...")
        
        trainer = RegimeModelTrainer()
        
        assert trainer.models is not None
        assert trainer.validators is not None
        logger.info("✓ Model trainer initialized successfully")
    
    def test_train_ensemble(self):
        """Test ensemble model training."""
        logger.info("Testing ensemble model training...")
        
        trainer = RegimeModelTrainer()
        
        # Create sample data
        X = np.random.randn(200, 20)
        y = np.random.choice([0, 1, 2], size=200)
        
        results = trainer.train_ensemble_models(X, y)
        
        assert results is not None
        assert 'models' in results
        logger.info("✓ Ensemble models trained successfully")


class TestMLRegimePredictor:
    """Test ML regime predictor."""
    
    def test_ml_predictor_initialization(self):
        """Test ML predictor initialization."""
        logger.info("Testing ML predictor initialization...")
        
        predictor = MLRegimePredictor()
        
        assert predictor.feature_engine is not None
        assert predictor.models is not None
        logger.info("✓ ML predictor initialized successfully")
    
    def test_train_regime_models(self):
        """Test regime model training."""
        logger.info("Testing regime model training...")
        
        predictor = MLRegimePredictor()
        
        price_data = create_sample_price_data()
        labels = create_regime_labels(index=price_data.index)
        
        result = predictor.train_regime_models(price_data, labels)
        
        assert result['success'] is True
        assert predictor.is_trained is True
        logger.info(f"✓ Trained models with {result['n_samples']} samples")
    
    @pytest.mark.asyncio
    async def test_predict_regime_transition(self):
        """Test regime transition prediction."""
        logger.info("Testing regime transition prediction...")
        
        predictor = MLRegimePredictor()
        
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
    async def test_full_ml_pipeline(self):
        """Test complete ML pipeline from training to prediction."""
        logger.info("Testing full ML pipeline...")
        
        # 1. Create training data
        price_data = create_sample_price_data(n_bars=300)
        labels = create_regime_labels(n_bars=300, index=price_data.index)
        
        # 2. Initialize and train predictor
        predictor = MLRegimePredictor()
        train_result = predictor.train_regime_models(price_data, labels)
        
        assert train_result['success'] is True
        logger.info(f"✓ Training completed with {train_result['n_samples']} samples")
        
        # 3. Make predictions
        prediction = await predictor.predict_regime_transition('BTC/USDT', price_data)
        
        assert prediction is not None
        assert prediction['predicted_regime'] in ['bullish', 'neutral', 'bearish']
        logger.info(f"✓ Prediction: {prediction['predicted_regime']} with confidence {prediction['confidence']:.2f}")
        
        # 4. Initialize prediction engine
        engine = RealTimePredictionEngine(trained_models=predictor.models)
        
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
