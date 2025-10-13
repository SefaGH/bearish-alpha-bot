"""
Real-Time ML Inference Engine for Regime Prediction.

Provides real-time prediction updates and trading signal generation.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class RealTimePredictionEngine:
    """
    Real-time ML inference engine for regime prediction.
    
    Integrates with WebSocket data streams to provide continuous
    regime predictions and trading signals.
    """
    
    def __init__(self, trained_models: Dict[str, Any], websocket_manager=None):
        """
        Initialize real-time prediction engine.
        
        Args:
            trained_models: Dictionary of trained ML models
            websocket_manager: WebSocket manager for real-time data
        """
        self.models = trained_models
        self.ws_manager = websocket_manager
        self.prediction_cache = {}
        self.feature_buffer = {}
        self.is_running = False
        self.prediction_tasks = []
        
        # Configuration
        self.update_interval = 60  # Update predictions every 60 seconds
        self.buffer_size = 200  # Keep last 200 data points
        self.cache_ttl = 300  # Cache predictions for 5 minutes
        
        logger.info("Real-Time Prediction Engine initialized")
    
    async def start_prediction_engine(self, symbols: Optional[List[str]] = None):
        """
        Start real-time prediction loop.
        
        Args:
            symbols: List of symbols to track (None for all)
        """
        self.is_running = True
        logger.info("Starting real-time prediction engine...")
        
        try:
            # Initialize feature buffers for tracked symbols
            if symbols:
                for symbol in symbols:
                    self.feature_buffer[symbol] = deque(maxlen=self.buffer_size)
            
            # Start prediction loop
            prediction_task = asyncio.create_task(self._prediction_loop())
            self.prediction_tasks.append(prediction_task)
            
            logger.info("Real-time prediction engine started")
            
        except Exception as e:
            logger.error(f"Error starting prediction engine: {e}")
            self.is_running = False
    
    async def stop_prediction_engine(self):
        """Stop the real-time prediction engine."""
        logger.info("Stopping real-time prediction engine...")
        self.is_running = False
        
        # Cancel all prediction tasks
        for task in self.prediction_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.prediction_tasks, return_exceptions=True)
        
        self.prediction_tasks = []
        logger.info("Real-time prediction engine stopped")
    
    async def _prediction_loop(self):
        """Main prediction loop."""
        while self.is_running:
            try:
                # Update predictions for all tracked symbols
                for symbol in self.feature_buffer.keys():
                    await self._update_symbol_prediction(symbol)
                
                # Wait for next update interval
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_symbol_prediction(self, symbol: str):
        """Update prediction for a specific symbol."""
        try:
            if symbol not in self.feature_buffer:
                return
            
            buffer = self.feature_buffer[symbol]
            if len(buffer) < 50:  # Need minimum data for prediction
                return
            
            # Convert buffer to DataFrame
            data = pd.DataFrame(list(buffer))
            
            # Make prediction
            prediction = await self.predict_regime_probabilities(symbol, data)
            
            # Update cache
            self.prediction_cache[symbol] = {
                'prediction': prediction,
                'timestamp': pd.Timestamp.now()
            }
            
            logger.debug(f"Updated prediction for {symbol}: {prediction['regime']}")
            
        except Exception as e:
            logger.error(f"Error updating prediction for {symbol}: {e}")
    
    async def on_market_data_update(self, symbol: str, data: Dict[str, Any]):
        """
        Process new market data for predictions.
        
        Args:
            symbol: Trading symbol
            data: Market data update
        """
        try:
            # Initialize buffer if needed
            if symbol not in self.feature_buffer:
                self.feature_buffer[symbol] = deque(maxlen=self.buffer_size)
            
            # Add data to buffer
            self.feature_buffer[symbol].append(data)
            
            # Trigger prediction if enough data
            if len(self.feature_buffer[symbol]) >= 50:
                # Check if prediction is stale
                if symbol in self.prediction_cache:
                    last_update = self.prediction_cache[symbol]['timestamp']
                    time_since_update = (pd.Timestamp.now() - last_update).total_seconds()
                    
                    if time_since_update > self.cache_ttl:
                        await self._update_symbol_prediction(symbol)
            
        except Exception as e:
            logger.error(f"Error processing market data update: {e}")
    
    async def predict_regime_probabilities(self, symbol: str, 
                                          data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get current regime probability predictions.
        
        Args:
            symbol: Trading symbol
            data: Optional price data (uses buffer if not provided)
            
        Returns:
            Dictionary with regime probabilities and predictions
        """
        try:
            # Check cache first
            if symbol in self.prediction_cache:
                cached = self.prediction_cache[symbol]
                time_since_cache = (pd.Timestamp.now() - cached['timestamp']).total_seconds()
                
                if time_since_cache < self.cache_ttl:
                    return cached['prediction']
            
            # Get data from buffer if not provided
            if data is None:
                if symbol not in self.feature_buffer or len(self.feature_buffer[symbol]) < 50:
                    return self._default_probabilities()
                
                data = pd.DataFrame(list(self.feature_buffer[symbol]))
            
            # Ensemble model inference
            if self.models.get('ensemble') is not None:
                # Prepare features (simplified)
                features = self._extract_simple_features(data)
                
                if features is not None:
                    predictions, probabilities = self.models['ensemble'].predict(features)
                    
                    # Probability calibration
                    calibrated_probs = self._calibrate_probabilities(probabilities[0])
                    
                    # Confidence interval calculation
                    confidence = self._calculate_confidence_interval(calibrated_probs)
                    
                    result = {
                        'regime': self._regime_class_to_label(predictions[0]),
                        'probabilities': {
                            'bullish': float(calibrated_probs[0]),
                            'neutral': float(calibrated_probs[1]),
                            'bearish': float(calibrated_probs[2])
                        },
                        'confidence': confidence,
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    return result
            
            return self._default_probabilities()
            
        except Exception as e:
            logger.error(f"Error predicting regime probabilities: {e}")
            return self._default_probabilities()
    
    def get_regime_transition_signals(self, symbol: str, 
                                     threshold: float = 0.7) -> Dict[str, Any]:
        """
        Generate regime transition trading signals.
        
        Args:
            symbol: Trading symbol
            threshold: Probability threshold for regime change signal
            
        Returns:
            Dictionary with trading signals
        """
        try:
            if symbol not in self.prediction_cache:
                return {'signal': 'neutral', 'strength': 0.0, 'reason': 'no_prediction'}
            
            prediction = self.prediction_cache[symbol]['prediction']
            probs = prediction['probabilities']
            
            # Regime change probability thresholds
            max_prob = max(probs.values())
            regime = max(probs, key=probs.get)
            
            # Signal strength calculation
            strength = self._calculate_signal_strength(probs, max_prob)
            
            # Risk-adjusted signal sizing
            position_size = self._calculate_position_size(strength, prediction['confidence'])
            
            signal = {
                'symbol': symbol,
                'signal': regime,
                'strength': strength,
                'position_size': position_size,
                'probabilities': probs,
                'confidence': prediction['confidence'],
                'timestamp': prediction['timestamp']
            }
            
            # Only generate strong signals above threshold
            if max_prob < threshold:
                signal['signal'] = 'neutral'
                signal['strength'] = 0.0
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating regime transition signals: {e}")
            return {'signal': 'neutral', 'strength': 0.0, 'reason': 'error'}
    
    def _extract_simple_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract simple features from data."""
        try:
            if 'close' not in data.columns:
                return None
            
            # Calculate basic features
            features = []
            
            # Returns
            returns = data['close'].pct_change().fillna(0)
            features.extend(returns.tail(10).tolist())
            
            # Volatility
            vol = returns.rolling(20).std().fillna(0)
            features.append(vol.iloc[-1])
            
            # Momentum
            mom = data['close'].pct_change(20).fillna(0)
            features.append(mom.iloc[-1])
            
            # Pad or truncate to expected size
            while len(features) < 50:
                features.append(0.0)
            
            return np.array([features[:50]])
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _calibrate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Calibrate probability predictions."""
        # Simple temperature scaling
        temperature = 1.2
        calibrated = probabilities ** (1 / temperature)
        calibrated = calibrated / np.sum(calibrated)
        return calibrated
    
    def _calculate_confidence_interval(self, probabilities: np.ndarray) -> float:
        """Calculate confidence interval from probabilities."""
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = -np.log(1.0 / len(probabilities))
        
        confidence = max_prob * (1 - entropy / max_entropy)
        return float(confidence)
    
    def _calculate_signal_strength(self, probabilities: Dict[str, float], 
                                   max_prob: float) -> float:
        """Calculate trading signal strength."""
        # Signal strength based on probability concentration
        prob_values = list(probabilities.values())
        strength = (max_prob - np.mean(prob_values)) / (np.std(prob_values) + 1e-10)
        strength = max(0.0, min(1.0, strength))  # Clip to [0, 1]
        return float(strength)
    
    def _calculate_position_size(self, signal_strength: float, 
                                confidence: float) -> float:
        """Calculate risk-adjusted position size."""
        base_size = 1.0
        adjusted_size = base_size * signal_strength * confidence
        return float(adjusted_size)
    
    def _regime_class_to_label(self, regime_class: int) -> str:
        """Convert regime class to label."""
        labels = ['bullish', 'neutral', 'bearish']
        return labels[min(regime_class, len(labels) - 1)]
    
    def _default_probabilities(self) -> Dict[str, Any]:
        """Return default probabilities when unable to compute."""
        return {
            'regime': 'neutral',
            'probabilities': {
                'bullish': 0.33,
                'neutral': 0.34,
                'bearish': 0.33
            },
            'confidence': 0.0,
            'timestamp': pd.Timestamp.now()
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get engine status.
        
        Returns:
            Dictionary with engine status information
        """
        return {
            'running': self.is_running,
            'symbols_tracked': list(self.feature_buffer.keys()),
            'n_predictions_cached': len(self.prediction_cache),
            'update_interval': self.update_interval,
            'buffer_size': self.buffer_size,
            'cache_ttl': self.cache_ttl
        }
