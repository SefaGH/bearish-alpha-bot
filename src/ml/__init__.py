"""
Machine Learning Module for Predictive Market Regime Detection.

Phase 4.1: ML-powered predictive market regime detection system.
Phase 4.2: Adaptive learning with reinforcement learning.
Phase 4.3: AI-powered strategy optimization with genetic algorithms.
Phase 4 Final: Advanced LSTM/Transformer price prediction with multi-timeframe forecasting.
"""

from .regime_predictor import MLRegimePredictor
from .feature_engineering import FeatureEngineeringPipeline
from .model_trainer import RegimeModelTrainer
from .prediction_engine import RealTimePredictionEngine
from .reinforcement_learning import TradingRLAgent, DQNNetwork
from .experience_replay import ExperienceReplay, EpisodeBuffer
from .genetic_optimizer import GeneticOptimizer, Individual
from .multi_objective_optimizer import MultiObjectiveOptimizer, MultiObjectiveIndividual
from .neural_architecture_search import NeuralArchitectureSearch, NetworkArchitecture
from .strategy_optimizer import StrategyOptimizer, OptimizationResult
from .price_predictor import (
    LSTMPricePredictor,
    TransformerPricePredictor,
    EnsemblePricePredictor,
    MultiTimeframePricePredictor,
    AdvancedPricePredictionEngine
)
from .strategy_integration import (
    AIEnhancedStrategyAdapter,
    StrategyPerformanceTracker,
    MLStrategyIntegrationManager
)

__all__ = [
    # Phase 4.1: Regime Prediction
    'MLRegimePredictor',
    'FeatureEngineeringPipeline',
    'RegimeModelTrainer',
    'RealTimePredictionEngine',
    
    # Phase 4.2: Adaptive Learning
    'TradingRLAgent',
    'DQNNetwork',
    'ExperienceReplay',
    'EpisodeBuffer',
    
    # Phase 4.3: Strategy Optimization
    'GeneticOptimizer',
    'Individual',
    'MultiObjectiveOptimizer',
    'MultiObjectiveIndividual',
    'NeuralArchitectureSearch',
    'NetworkArchitecture',
    'StrategyOptimizer',
    'OptimizationResult',
    
    # Phase 4 Final: Advanced Price Prediction
    'LSTMPricePredictor',
    'TransformerPricePredictor',
    'EnsemblePricePredictor',
    'MultiTimeframePricePredictor',
    'AdvancedPricePredictionEngine',
    'AIEnhancedStrategyAdapter',
    'StrategyPerformanceTracker',
    'MLStrategyIntegrationManager',
]
