"""
Machine Learning System Configuration for Phase 4.1.

Configuration for ML models, training, and feature engineering.
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for individual ML models."""
    lstm: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'input_size': 50,
        'num_classes': 3,
    })
    
    transformer: Dict[str, Any] = field(default_factory=lambda: {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'learning_rate': 0.0001,
        'num_classes': 3,
    })
    
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
    })
    
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'lstm': 0.4,
        'transformer': 0.4,
        'random_forest': 0.2,
    })


@dataclass
class TrainingConfig:
    """Training configuration for ML models."""
    sequence_length: int = 100  # 100 time steps for prediction
    prediction_horizon: int = 12  # 12 steps ahead (1 hour if 5min data)
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    learning_rate_decay: float = 0.95
    min_learning_rate: float = 0.00001


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    technical_indicators: bool = True
    market_microstructure: bool = True
    volatility_features: bool = True
    momentum_features: bool = True
    cross_asset_correlations: bool = True
    
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    atr_period: int = 14
    
    # Volatility feature parameters
    volatility_windows: list = field(default_factory=lambda: [5, 10, 20, 50])
    
    # Momentum feature parameters
    momentum_windows: list = field(default_factory=lambda: [5, 10, 20, 50])


@dataclass
class PredictionConfig:
    """Real-time prediction configuration."""
    min_confidence_threshold: float = 0.6  # Minimum confidence for regime change signal
    prediction_update_interval: int = 60  # Update predictions every 60 seconds
    feature_buffer_size: int = 200  # Keep last 200 data points in buffer
    cache_ttl: int = 300  # Cache predictions for 5 minutes
    
    # Regime transition thresholds
    regime_change_threshold: float = 0.7  # Probability threshold for regime change
    signal_strength_multiplier: float = 1.5  # Signal strength calculation multiplier


class MLConfiguration:
    """
    Centralized machine learning system configuration.
    
    Provides configuration for models, training, features, and predictions.
    """
    
    MODEL_CONFIG = {
        'lstm': {
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'learning_rate': 0.001,
        },
        'transformer': {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'learning_rate': 0.0001,
        },
        'ensemble_weights': {
            'lstm': 0.4,
            'transformer': 0.4,
            'random_forest': 0.2,
        }
    }
    
    TRAINING_CONFIG = {
        'sequence_length': 100,  # 100 time steps for prediction
        'prediction_horizon': 12,  # 12 steps ahead (1 hour if 5min data)
        'batch_size': 32,
        'max_epochs': 100,
        'early_stopping_patience': 10,
        'validation_split': 0.2,
    }
    
    FEATURE_CONFIG = {
        'technical_indicators': True,
        'market_microstructure': True,
        'volatility_features': True,
        'momentum_features': True,
        'cross_asset_correlations': True,
    }
    
    PREDICTION_CONFIG = {
        'min_confidence_threshold': 0.6,
        'prediction_update_interval': 60,
        'regime_change_threshold': 0.7,
    }
    
    @classmethod
    def get_model_config(cls) -> ModelConfig:
        """Get model configuration as dataclass."""
        return ModelConfig()
    
    @classmethod
    def get_training_config(cls) -> TrainingConfig:
        """Get training configuration as dataclass."""
        return TrainingConfig()
    
    @classmethod
    def get_feature_config(cls) -> FeatureConfig:
        """Get feature configuration as dataclass."""
        return FeatureConfig()
    
    @classmethod
    def get_prediction_config(cls) -> PredictionConfig:
        """Get prediction configuration as dataclass."""
        return PredictionConfig()
