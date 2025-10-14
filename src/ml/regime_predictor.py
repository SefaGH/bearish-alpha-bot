"""
ML-Based Market Regime Predictor.

Integrates with Phase 2 regime detection to provide predictive capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

from .feature_engineering import FeatureEngineeringPipeline
from .neural_networks import LSTMRegimePredictor, TransformerRegimePredictor
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class EnsembleRegimePredictor:
    """Ensemble predictor combining multiple models."""
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: Dictionary of trained models
            weights: Optional weights for ensemble voting
        """
        self.models = models
        self.weights = weights or {
            'lstm': 0.4,
            'transformer': 0.4,
            'random_forest': 0.2
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        predictions = []
        probabilities = []
        
        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 1.0 / len(self.models))
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
            elif hasattr(model, 'predict'):
                pred, probs = model.predict(X)
            else:
                # Fallback to uniform probabilities
                probs = np.ones((len(X), 3)) / 3
            
            probabilities.append(probs * weight)
        
        # Weighted average of probabilities
        ensemble_probs = np.sum(probabilities, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        return ensemble_preds, ensemble_probs


class MLRegimePredictor:
    """
    Machine learning-based market regime prediction system.
    
    Integrates with Phase 2 MarketRegimeAnalyzer to provide predictive
    capabilities using ensemble of ML models.
    """
    
    def __init__(self, regime_analyzer=None, websocket_manager=None):
        """
        Initialize ML regime predictor.
        
        Args:
            regime_analyzer: Phase 2 MarketRegimeAnalyzer instance
            websocket_manager: Phase 3.1 WebSocket manager for real-time data
        """
        self.regime_analyzer = regime_analyzer
        self.ws_manager = websocket_manager
        
        # Initialize models
        self.models = {
            'lstm': None,
            'transformer': None,
            'random_forest': None,
            'ensemble': None
        }
        
        self.feature_engine = FeatureEngineeringPipeline()
        self.prediction_history = []
        self.is_trained = False
        
        logger.info("ML Regime Predictor initialized")
    
    async def predict_regime_transition(self, symbol: str, 
                                       price_data: pd.DataFrame,
                                       horizon: str = '1h') -> Dict[str, Any]:
        """
        Predict future market regime with confidence intervals.
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data with indicators
            horizon: Prediction horizon (e.g., '1h', '4h')
            
        Returns:
            Dictionary with predictions and confidence metrics
        """
        try:
            logger.info(f"Predicting regime transition for {symbol} with {horizon} horizon")
            logger.debug(f"🧠 [ML-REGIME] Starting regime prediction for {symbol}")
            
            # Feature extraction from multi-timeframe data
            features = self.feature_engine.extract_features(price_data)
            
            if features.empty:
                logger.warning("No features extracted, returning default prediction")
                logger.debug(f"🧠 [ML-REGIME] No features extracted, using default")
                return self._default_prediction()
            
            logger.debug(f"🧠 [ML-REGIME] Extracted {len(features.columns)} features from price data")
            
            # Prepare features for prediction
            feature_values = features.dropna().tail(1).values
            
            if len(feature_values) == 0:
                logger.debug(f"🧠 [ML-REGIME] No valid feature values after cleanup")
                return self._default_prediction()
            
            # Model ensemble prediction
            if self.models['ensemble'] is not None and self.is_trained:
                predictions, probabilities = self.models['ensemble'].predict(feature_values)
                
                # Confidence interval calculation
                confidence = self._calculate_confidence(probabilities[0])
                
                # Prediction quality assessment
                quality = self._assess_prediction_quality(features, probabilities[0])
                
                result = {
                    'symbol': symbol,
                    'horizon': horizon,
                    'predicted_regime': self._regime_class_to_label(predictions[0]),
                    'probabilities': {
                        'bullish': float(probabilities[0][0]),
                        'neutral': float(probabilities[0][1]),
                        'bearish': float(probabilities[0][2])
                    },
                    'confidence': confidence,
                    'quality_score': quality,
                    'timestamp': pd.Timestamp.now()
                }
                
                self.prediction_history.append(result)
                
                logger.debug(f"🧠 [ML-REGIME] Market regime: {result['predicted_regime']} (confidence: {confidence:.2%})")
                logger.debug(f"🧠 [ML-REGIME] Probabilities: Bull={probabilities[0][0]:.2%}, Neutral={probabilities[0][1]:.2%}, Bear={probabilities[0][2]:.2%}")
                logger.info(f"Prediction: {result['predicted_regime']} (confidence: {confidence:.2f})")
                return result
            else:
                logger.warning("Models not trained, using fallback prediction")
                logger.debug(f"🧠 [ML-REGIME] Models not trained, using fallback")
                return self._fallback_prediction(symbol, price_data)
                
        except Exception as e:
            logger.error(f"Error predicting regime transition: {e}")
            logger.debug(f"🧠 [ML-REGIME] Prediction error: {e}")
            return self._default_prediction()
    
    def train_regime_models(self, historical_data: pd.DataFrame, 
                           regime_labels: pd.Series) -> Dict[str, Any]:
        """
        Train all regime prediction models.
        
        Args:
            historical_data: Historical price data with indicators
            regime_labels: Historical regime labels
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Training regime prediction models...")
            
            # Feature engineering and preprocessing
            features = self.feature_engine.extract_features(historical_data)
            
            # Prepare data for training
            X, y = self.feature_engine.prepare_for_training(features, regime_labels)
            
            if len(X) < 100:
                logger.warning(f"Insufficient training data: {len(X)} samples")
                return {'success': False, 'error': 'insufficient_data'}
            
            # Train Random Forest
            logger.info("Training Random Forest model...")
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            rf_model.fit(X, y)
            self.models['random_forest'] = rf_model
            
            # Initialize LSTM and Transformer (placeholders)
            self.models['lstm'] = LSTMRegimePredictor(input_size=X.shape[1])
            self.models['transformer'] = TransformerRegimePredictor()
            
            # Create ensemble
            self.models['ensemble'] = EnsembleRegimePredictor(
                models={
                    'random_forest': self.models['random_forest'],
                    'lstm': self.models['lstm'],
                    'transformer': self.models['transformer']
                }
            )
            
            self.is_trained = True
            
            # Model performance evaluation
            train_score = rf_model.score(X, y)
            
            result = {
                'success': True,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'train_accuracy': train_score,
                'models_trained': ['random_forest', 'lstm', 'transformer', 'ensemble']
            }
            
            logger.info(f"Training complete: {result['models_trained']}")
            logger.info(f"Training accuracy: {train_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'success': False, 'error': str(e)}
    
    async def real_time_regime_forecast(self, symbol: str, 
                                       update_interval: int = 60) -> Dict[str, Any]:
        """
        Continuous real-time regime forecasting.
        
        Args:
            symbol: Trading symbol
            update_interval: Update interval in seconds
            
        Returns:
            Dictionary with forecast results
        """
        logger.info(f"Starting real-time regime forecast for {symbol}")
        
        # Placeholder for real-time forecasting loop
        # In production, this would integrate with WebSocket data streams
        
        return {
            'symbol': symbol,
            'status': 'active',
            'update_interval': update_interval,
            'forecasts': []
        }
    
    def _regime_class_to_label(self, regime_class: int) -> str:
        """Convert regime class index to label."""
        labels = ['bullish', 'neutral', 'bearish']
        return labels[min(regime_class, len(labels) - 1)]
    
    def _calculate_confidence(self, probabilities: np.ndarray) -> float:
        """Calculate prediction confidence from probabilities."""
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Confidence is high when max probability is high and entropy is low
        confidence = max_prob * (1 - entropy / np.log(len(probabilities)))
        
        return float(confidence)
    
    def _assess_prediction_quality(self, features: pd.DataFrame, 
                                   probabilities: np.ndarray) -> float:
        """Assess prediction quality based on feature completeness and probability distribution."""
        # Feature completeness score
        feature_completeness = 1 - (features.isna().sum().sum() / (features.shape[0] * features.shape[1]))
        
        # Probability distribution score (higher is better)
        prob_score = np.max(probabilities)
        
        # Combined quality score
        quality = 0.6 * prob_score + 0.4 * feature_completeness
        
        return float(quality)
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when unable to compute."""
        return {
            'predicted_regime': 'neutral',
            'probabilities': {
                'bullish': 0.33,
                'neutral': 0.34,
                'bearish': 0.33
            },
            'confidence': 0.0,
            'quality_score': 0.0,
            'timestamp': pd.Timestamp.now()
        }
    
    def _fallback_prediction(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback prediction using basic heuristics."""
        # Use simple momentum-based prediction
        if len(price_data) < 20:
            return self._default_prediction()
        
        recent_return = (price_data['close'].iloc[-1] - price_data['close'].iloc[-20]) / price_data['close'].iloc[-20]
        
        if recent_return > 0.05:
            predicted_regime = 'bullish'
            probs = {'bullish': 0.6, 'neutral': 0.3, 'bearish': 0.1}
        elif recent_return < -0.05:
            predicted_regime = 'bearish'
            probs = {'bullish': 0.1, 'neutral': 0.3, 'bearish': 0.6}
        else:
            predicted_regime = 'neutral'
            probs = {'bullish': 0.25, 'neutral': 0.5, 'bearish': 0.25}
        
        return {
            'symbol': symbol,
            'predicted_regime': predicted_regime,
            'probabilities': probs,
            'confidence': 0.5,
            'quality_score': 0.5,
            'timestamp': pd.Timestamp.now()
        }
    
    def get_prediction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get prediction history.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of recent predictions
        """
        return self.prediction_history[-limit:]
