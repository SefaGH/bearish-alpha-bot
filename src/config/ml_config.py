"""
DEPRECATED - Machine Learning System Configuration

This file is deprecated as of v3.5 and is no longer used by the application.
All machine learning configurations have been centralized in the main `config.example.yaml` file.

This allows for easier management of parameters and enables overriding values
through environment variables or GitHub repository variables without code changes.

Please refer to the `ml:` section in your `config.example.yaml` for all ML-related settings.

This file is kept temporarily for historical reference and will be removed in a future version.
"""

import warnings

warnings.warn(
    "The 'src/config/ml_config.py' file is deprecated and no longer in use. "
    "All ML configurations are now managed via the main YAML config file.",
    DeprecationWarning,
    stacklevel=2
)

# The original classes are kept below but are not imported or used anywhere in the application.

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class _ModelConfig:
    """DEPRECATED"""
    lstm: Dict[str, Any] = field(default_factory=dict)
    transformer: Dict[str, Any] = field(default_factory=dict)
    random_forest: Dict[str, Any] = field(default_factory=dict)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class _TrainingConfig:
    """DEPRECATED"""
    sequence_length: int = 100
    prediction_horizon: int = 12
    batch_size: int = 32
    max_epochs: int = 100


@dataclass
class _FeatureConfig:
    """DEPRECATED"""
    rsi_period: int = 14
    atr_period: int = 14


@dataclass
class _PredictionConfig:
    """DEPRECATED"""
    min_confidence_threshold: float = 0.6
    prediction_update_interval: int = 60


class _MLConfiguration:
    """DEPRECATED"""
    
    @classmethod
    def get_model_config(cls) -> _ModelConfig:
        raise NotImplementedError("This class is deprecated. Use the main YAML config.")
    
    @classmethod
    def get_training_config(cls) -> _TrainingConfig:
        raise NotImplementedError("This class is deprecated. Use the main YAML config.")
    
    @classmethod
    def get_feature_config(cls) -> _FeatureConfig:
        raise NotImplementedError("This class is deprecated. Use the main YAML config.")
    
    @classmethod
    def get_prediction_config(cls) -> _PredictionConfig:
        raise NotImplementedError("This class is deprecated. Use the main YAML config.")
