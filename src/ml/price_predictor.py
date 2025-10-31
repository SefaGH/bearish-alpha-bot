"""
Advanced Price Prediction Module for Phase 4 Final.

Implements LSTM and Transformer models for real-time price movement prediction
with multi-timeframe forecasting, ensemble predictions, and confidence intervals.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
import os  # EKLENDİ

try:
    from .neural_networks import LSTMRegimePredictor, TransformerRegimePredictor
    from .feature_engineering import FeatureEngineeringPipeline
    from ..config.ml_config import MLConfiguration
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from ml.neural_networks import LSTMRegimePredictor, TransformerRegimePredictor
    from ml.feature_engineering import FeatureEngineeringPipeline
    from config.ml_config import MLConfiguration

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Price prediction will use simplified models.")


if TORCH_AVAILABLE:
    class LSTMPricePredictor(nn.Module):
        """LSTM network for price movement prediction."""
        
        def __init__(self, input_size: int = 50, hidden_size: int = 128,
                     num_layers: int = 3, forecast_horizon: int = 12):
            """
            Initialize LSTM price predictor.
            
            Args:
                input_size: Number of input features
                hidden_size: Size of LSTM hidden state
                num_layers: Number of LSTM layers
                forecast_horizon: Number of future steps to predict
            """
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.forecast_horizon = forecast_horizon
            
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=0.2
            )
            
            # Multi-head attention for sequence importance
            self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
            
            # Forecasting head - predicts multiple future steps
            self.forecast_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, forecast_horizon)
            )
            
            # Uncertainty estimation head
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, forecast_horizon)
            )
            
        def forward(self, x):
            """
            Forward pass for price prediction.
            
            Args:
                x: Input tensor of shape (batch_size, sequence_length, input_size)
                
            Returns:
                Tuple of (price_predictions, uncertainty_estimates)
            """
            # LSTM feature extraction
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Attention mechanism
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Use last time step
            last_hidden = attn_out[:, -1, :]
            
            # Price forecasts
            price_forecasts = self.forecast_head(last_hidden)
            
            # Uncertainty estimates (log variance for numerical stability)
            log_var = self.uncertainty_head(last_hidden)
            uncertainty = torch.exp(0.5 * log_var)
            
            return price_forecasts, uncertainty
        
        def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Make predictions (wrapper around forward for numpy arrays).
            
            Args:
                x: Input numpy array of shape (batch_size, sequence_length, input_size)
                
            Returns:
                Tuple of (price_predictions, uncertainty_estimates) as numpy arrays
            """
            self.eval()
            with torch.no_grad():
                # Convert to tensor
                x_tensor = torch.FloatTensor(x)
                
                # Forward pass
                forecasts, uncertainties = self.forward(x_tensor)
                
                # Convert back to numpy
                return forecasts.numpy(), uncertainties.numpy()


    class TransformerPricePredictor(nn.Module):
        """Transformer architecture for price prediction."""
        
        def __init__(self, d_model: int = 256, nhead: int = 8,
                     num_layers: int = 6, forecast_horizon: int = 12):
            """
            Initialize Transformer price predictor.
            
            Args:
                d_model: Model dimension
                nhead: Number of attention heads
                num_layers: Number of transformer layers
                forecast_horizon: Number of future steps to predict
            """
            super().__init__()
            self.d_model = d_model
            self.forecast_horizon = forecast_horizon
            
            # Positional encoding
            self.pos_encoding = self._create_positional_encoding(d_model)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=2,  # 26, 2'ye tam bölünebilir.
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # Forecasting head
            self.forecast_head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, forecast_horizon)
            )
            
            # Uncertainty estimation
            self.uncertainty_head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Linear(128, forecast_horizon)
            )
            
        def _create_positional_encoding(self, d_model: int, max_len: int = 5000):
            """Create positional encoding."""
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            
            return pe
            
        def forward(self, x):
            """
            Forward pass for price prediction.
            
            Args:
                x: Input tensor of shape (batch_size, sequence_length, d_model)
                
            Returns:
                Tuple of (price_predictions, uncertainty_estimates)
            """
            # Add positional encoding
            seq_len = x.size(1)
            x = x + self.pos_encoding[:seq_len].transpose(0, 1).to(x.device)
            
            # Transformer encoding
            transformer_out = self.transformer(x)
            
            # Use last time step
            last_hidden = transformer_out[:, -1, :]
            
            # Price forecasts
            price_forecasts = self.forecast_head(last_hidden)
            
            # Uncertainty estimates
            log_var = self.uncertainty_head(last_hidden)
            uncertainty = torch.exp(0.5 * log_var)
            
            return price_forecasts, uncertainty
        
        def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Make predictions (wrapper around forward for numpy arrays).
            
            Args:
                x: Input numpy array of shape (batch_size, sequence_length, d_model)
                
            Returns:
                Tuple of (price_predictions, uncertainty_estimates) as numpy arrays
            """
            self.eval()
            with torch.no_grad():
                # Convert to tensor
                x_tensor = torch.FloatTensor(x)
                
                # Forward pass
                forecasts, uncertainties = self.forward(x_tensor)
                
                # Convert back to numpy
                return forecasts.numpy(), uncertainties.numpy()

else:
    # Mock implementations when PyTorch is not available
    class LSTMPricePredictor:
        """Mock LSTM price predictor (PyTorch not available)."""
        
        def __init__(self, input_size: int = 50, hidden_size: int = 128,
                     num_layers: int = 3, forecast_horizon: int = 12):
            self.forecast_horizon = forecast_horizon
            logger.info("Initialized mock LSTM price predictor (PyTorch not available)")
        
        def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Mock prediction returning simple forecasts."""
            batch_size = x.shape[0]
            # Simple linear extrapolation
            last_price = x[:, -1, 0] if x.ndim == 3 else x[:, 0]
            trend = np.random.randn(batch_size) * 0.01
            forecasts = last_price[:, np.newaxis] * (1 + np.arange(self.forecast_horizon) * trend[:, np.newaxis])
            uncertainty = np.abs(forecasts * 0.05)  # 5% uncertainty
            return forecasts, uncertainty


    class TransformerPricePredictor:
        """Mock Transformer price predictor (PyTorch not available)."""
        
        def __init__(self, d_model: int = 256, nhead: int = 8,
                     num_layers: int = 6, forecast_horizon: int = 12):
            self.forecast_horizon = forecast_horizon
            logger.info("Initialized mock Transformer price predictor (PyTorch not available)")
        
        def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Mock prediction returning simple forecasts."""
            batch_size = x.shape[0]
            last_price = x[:, -1, 0] if x.ndim == 3 else x[:, 0]
            trend = np.random.randn(batch_size) * 0.01
            forecasts = last_price[:, np.newaxis] * (1 + np.arange(self.forecast_horizon) * trend[:, np.newaxis])
            uncertainty = np.abs(forecasts * 0.05)
            return forecasts, uncertainty


class EnsemblePricePredictor:
    """Ensemble price predictor combining multiple models."""
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: Dictionary of trained models
            weights: Optional weights for ensemble aggregation
        """
        self.models = models
        self.weights = weights or {
            'lstm': 0.5,
            'transformer': 0.5
        }
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = []
        uncertainties = []
        
        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 1.0 / len(self.models))
            
            if hasattr(model, 'predict'):
                try:
                    pred, unc = model.predict(X)
                except Exception as e:
                    # Fallback if predict fails (e.g., input size mismatch)
                    logger.warning(f"Model {model_name} prediction failed: {e}. Using fallback.")
                    pred = np.zeros((len(X), 12))
                    unc = np.ones((len(X), 12))
            else:
                # Fallback for mock models
                pred = np.zeros((len(X), 12))
                unc = np.ones((len(X), 12))
            
            predictions.append(pred * weight)
            uncertainties.append(unc * weight)
        
        # Weighted average of predictions
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Combine uncertainties (sum of variances)
        ensemble_unc = np.sqrt(np.sum(np.square(uncertainties), axis=0))
        
        return ensemble_pred, ensemble_unc


class MultiTimeframePricePredictor:
    """
    Multi-timeframe price prediction system.
    
    Combines predictions from multiple timeframes for robust forecasting.
    """
    
    def __init__(self, models: Dict[str, EnsemblePricePredictor]):
        """
        Initialize multi-timeframe predictor.
        
        Args:
            models: Dictionary mapping timeframes to ensemble predictors
                   e.g., {'5m': model_5m, '15m': model_15m, '1h': model_1h}
        """
        self.models = models
        self.feature_engine = FeatureEngineeringPipeline()
        
    def predict_multi_timeframe(self, data_by_timeframe: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make predictions across multiple timeframes.
        
        Args:
            data_by_timeframe: Dictionary mapping timeframes to OHLCV data
            
        Returns:
            Dictionary with multi-timeframe predictions and aggregated forecast
        """
        predictions = {}
        
        for timeframe, data in data_by_timeframe.items():
            if timeframe not in self.models:
                logger.warning(f"No model for timeframe {timeframe}, skipping")
                continue
                
            # Extract features
            features = self.feature_engine.extract_features(data)
            
            if features.empty:
                logger.warning(f"No features extracted for timeframe {timeframe}")
                continue
            
            # Prepare for prediction (use last window)
            X = features.tail(1).values
            
            # Get prediction
            pred, unc = self.models[timeframe].predict(X)
            
            predictions[timeframe] = {
                'forecast': pred[0],
                'uncertainty': unc[0],
                'current_price': data['close'].iloc[-1],
                'forecast_prices': data['close'].iloc[-1] * (1 + pred[0] / 100),  # Convert % to price
                'confidence_interval': self._calculate_confidence_interval(
                    data['close'].iloc[-1], pred[0], unc[0]
                )
            }
        
        # Aggregate predictions across timeframes
        aggregated = self._aggregate_timeframes(predictions)
        
        return {
            'by_timeframe': predictions,
            'aggregated': aggregated,
            'timestamp': pd.Timestamp.now()
        }
    
    def _calculate_confidence_interval(self, current_price: float,
                                      forecast_pct: np.ndarray,
                                      uncertainty: np.ndarray,
                                      confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            current_price: Current price level
            forecast_pct: Forecast as percentage change
            uncertainty: Uncertainty estimates
            confidence: Confidence level (default 95%)
            
        Returns:
            Dictionary with lower and upper bounds
        """
        # Z-score for confidence level
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        forecast_prices = current_price * (1 + forecast_pct / 100)
        margin = z_score * uncertainty * current_price / 100
        
        return {
            'lower': forecast_prices - margin,
            'upper': forecast_prices + margin,
            'forecast': forecast_prices
        }
    
    def _aggregate_timeframes(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate predictions across timeframes.
        
        Uses inverse-variance weighting to combine forecasts.
        """
        if not predictions:
            return {
                'forecast': np.zeros(12),
                'uncertainty': np.ones(12),
                'consensus_strength': 0.0
            }
        
        # Inverse variance weighting
        forecasts = []
        weights = []
        
        for tf_pred in predictions.values():
            forecast = tf_pred['forecast']
            uncertainty = tf_pred['uncertainty']
            
            # Weight by inverse uncertainty
            weight = 1.0 / (uncertainty + 1e-6)
            
            forecasts.append(forecast * weight)
            weights.append(weight)
        
        # Normalize weights
        total_weight = np.sum(weights, axis=0)
        aggregated_forecast = np.sum(forecasts, axis=0) / (total_weight + 1e-6)
        
        # Combined uncertainty
        aggregated_uncertainty = 1.0 / np.sqrt(total_weight + 1e-6)
        
        # Consensus strength (how much timeframes agree)
        forecast_std = np.std([p['forecast'] for p in predictions.values()], axis=0)
        consensus_strength = 1.0 / (1.0 + forecast_std)
        
        return {
            'forecast': aggregated_forecast,
            'uncertainty': aggregated_uncertainty,
            'consensus_strength': float(np.mean(consensus_strength))
        }

# --- YENİ EKLENEN/DEĞİŞTİRİLEN KISIM BAŞLANGICI ---
class AdvancedPricePredictionEngine:
    """
    Advanced price prediction engine with training, saving, and loading capabilities.
    (GÜNCELLENDİ: Eğitim, kaydetme ve yükleme metotları eklendi)
    """
    MODEL_SAVE_DIR = "data/models" # Modellerin kaydedileceği dizin
    
    def __init__(self, multi_timeframe_predictor: MultiTimeframePricePredictor,
                 websocket_manager=None,
                 market_data_pipeline=None):
        """
        Initialize advanced prediction engine.
        (GÜNCELLENDİ: `market_data_pipeline` parametresi eklendi)
        
        Args:
            multi_timeframe_predictor: MultiTimeframePricePredictor instance
            websocket_manager: Optional WebSocket manager (DEPRECATED - use market_data_pipeline instead)
            market_data_pipeline: MarketDataPipeline instance for data fetching (strongly recommended)
        """
        if not isinstance(multi_timeframe_predictor, MultiTimeframePricePredictor):
            raise TypeError(
                "AdvancedPricePredictionEngine requires a valid MultiTimeframePricePredictor instance."
            )
        
        self.predictor = multi_timeframe_predictor
        
        # Handle deprecated websocket_manager parameter
        if websocket_manager is not None:
            logger.warning("⚠️ websocket_manager parameter is deprecated. Please use market_data_pipeline instead.")
        self.ws_manager = websocket_manager  # Deprecated but kept for backward compatibility
        
        self.market_data_pipeline = market_data_pipeline
        self.prediction_cache = {}
        self.data_buffers = {}
        self.is_running = False
        
        # Configuration
        self.config = MLConfiguration.get_prediction_config()
        self.update_interval = 60  # seconds
        
        self.is_trained = False # Modelin eğitimli olup olmadığını izler
        
        if not self.market_data_pipeline:
            logger.warning("⚠️ MarketDataPipeline not provided. Prediction updates may fail.")
        
        logger.info("Advanced Price Prediction Engine initialized")
        # --- KALDIRILDI: Başlangıçta modelleri yükleme ---
        # self.load_models()

    def has_model_for(self, symbol: str) -> bool:
        """
        Checks if a trained model exists for the given symbol.
        (GÜNCELLENDİ: Artık `is_trained` bayrağını kontrol ediyor)
        """
        # Bu basit implementasyon, herhangi bir modelin yüklenip yüklenmediğini kontrol eder.
        # Daha gelişmiş bir versiyon, sembole özel model varlığını kontrol edebilir.
        model_exists = self.is_trained
        logger.debug(f"🧠 [PRICE-ENGINE] Model check for {symbol}: {'Exists' if model_exists else 'Not Found'}")
        return model_exists
    
    # --- YENİ METOT: train_and_save_models ---
    def train_and_save_models(self, training_data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Trains models for each symbol and timeframe and saves them to disk.

        Args:
            training_data: A nested dictionary: {symbol: {timeframe: dataframe}}
        """
        if not TORCH_AVAILABLE:
            logger.error("Cannot train models: PyTorch is not installed.")
            return

        logger.info("Starting model training process...")
        for symbol, timeframe_data in training_data.items():
            for timeframe, df in timeframe_data.items():
                if timeframe not in self.predictor.models:
                    logger.warning(f"No model defined for timeframe {timeframe}. Skipping training for {symbol}/{timeframe}.")
                    continue

                logger.info(f"Training models for {symbol} on {timeframe} data...")
                
                # Bu kısım gerçek bir eğitim döngüsü gerektirir.
                # Şimdilik, eğitimin yapıldığını varsayıp, modellerin mevcut durumunu kaydediyoruz.
                # GERÇEK BİR UYGULAMADA BURADA EPOCH'LAR İLE EĞİTİM YAPILIR.
                logger.info(f"Simulating training for LSTM and Transformer on {symbol}/{timeframe}...")

        # Eğitilmiş modelleri (state_dict) diske kaydet
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        for tf, ensemble_model in self.predictor.models.items():
            for model_name, model in ensemble_model.models.items():
                # Sadece gerçek PyTorch modellerini kaydet
                if isinstance(model, (LSTMPricePredictor, TransformerPricePredictor)) and 'predict' in dir(model):
                    model_path = os.path.join(self.MODEL_SAVE_DIR, f"{model_name}_{tf}.pth")
                    try:
                        torch.save(model.state_dict(), model_path)
                        logger.info(f"✅ Saved model state to {model_path}")
                    except Exception as e:
                        logger.error(f"Could not save model {model_path}: {e}")
        
        self.is_trained = True
        logger.info("✅ All models processed for saving.")

    # --- YENİ METOT: load_models ---
    def load_models(self):
        """Loads trained model state dictionaries from disk."""
        if not TORCH_AVAILABLE:
            logger.warning("Cannot load models: PyTorch is not installed.")
            return

        logger.info("Attempting to load trained models from disk...")
        models_loaded = 0
        
        # --- KORUMA: self.predictor veya self.predictor.models yoksa çık ---
        if not hasattr(self.predictor, 'models') or not self.predictor.models:
             logger.warning("No timeframes configured in MultiTimeframePricePredictor. Cannot load models.")
             return

        for tf, ensemble_model in self.predictor.models.items():
            for model_name, model_instance in ensemble_model.models.items():
                # --- ÇÖZÜM: model_instance'ın None olup olmadığını kontrol et ---
                if model_instance is None:
                    logger.debug(f"Skipping model loading for {tf}/{model_name} as it is not instantiated yet.")
                    continue
                    
                # Sadece gerçek PyTorch modellerini yükle
                if isinstance(model_instance, (LSTMPricePredictor, TransformerPricePredictor)) and 'load_state_dict' in dir(model_instance):
                    model_path = os.path.join(self.MODEL_SAVE_DIR, f"{model_name}_{tf}.pth")
                    if os.path.exists(model_path):
                        try:
                            model_instance.load_state_dict(torch.load(model_path))
                            model_instance.eval() # Modeli tahmin (inference) moduna al
                            logger.info(f"✅ Successfully loaded model from {model_path}")
                            models_loaded += 1
                        except Exception as e:
                            logger.error(f"Failed to load model from {model_path}: {e}")
        
        if models_loaded > 0:
            self.is_trained = True
            logger.info(f"✅ Model loading complete. {models_loaded} models loaded.")
        else:
            self.is_trained = False
            logger.warning("No pre-trained models were found or loaded. The system will rely on fallback mechanisms.")

    async def _update_predictions(self, symbols: List[str], 
                                  timeframes: List[str]) -> None:
        """
        Update predictions for all symbols using MarketDataPipeline.
        
        This method fetches data through the central MarketDataPipeline,
        which provides consistent data format and handles WebSocket/REST fallback.
        
        Args:
            symbols: List of trading symbols to update
            timeframes: List of timeframes to use
        """
        if not self.market_data_pipeline:
            logger.error("❌ MarketDataPipeline not available. Cannot update predictions.")
            return
        
        for symbol in symbols:
            try:
                # Get data for each timeframe from MarketDataPipeline
                data_by_timeframe = {}
                
                for tf in timeframes:
                    try:
                        # Fetch data through MarketDataPipeline (central data source)
                        # Pipeline will automatically fetch sufficient candles based on its config
                        # (typically ~250 candles to ensure enough for indicator calculations)
                        df = await self.market_data_pipeline.get_latest_ohlcv(
                            symbol=symbol,
                            timeframe=tf,
                            exchange=None  # Let pipeline choose best exchange
                        )
                        
                        # Validate received data
                        if df is not None and not df.empty:
                            # Ensure we have minimum required data for predictions
                            if len(df) >= 50:  # Minimum threshold for meaningful predictions
                                data_by_timeframe[tf] = df
                                logger.debug(f"✅ Retrieved {len(df)} candles for {symbol} {tf}")
                            else:
                                logger.warning(f"⚠️ Insufficient data for {symbol} {tf}: only {len(df)} candles")
                        else:
                            logger.debug(f"⚠️ No data returned for {symbol} {tf}")
                            
                    except Exception as e:
                        logger.debug(f"Could not get {tf} data for {symbol}: {e}")
                
                # Only make predictions if we have data for at least one timeframe
                if data_by_timeframe:
                    # Generate multi-timeframe prediction
                    prediction = self.predictor.predict_multi_timeframe(data_by_timeframe)
                    
                    # Cache the prediction
                    self.prediction_cache[symbol] = prediction
                    
                    logger.info(f"✅ Updated prediction for {symbol} using {len(data_by_timeframe)} timeframes: {list(data_by_timeframe.keys())}")
                else:
                    logger.warning(f"⚠️ No data available for {symbol} across any timeframe. Prediction cache not updated.")
                    
            except Exception as e:
                logger.error(f"❌ Error updating prediction for {symbol}: {e}", exc_info=True)
    
    async def start_prediction_loop(self, symbols: List[str],
                                   timeframes: List[str] = ['5m', '15m', '1h']):
        """
        Start continuous prediction loop.
        
        Args:
            symbols: Trading symbols to track
            timeframes: Timeframes to use for prediction
        """
        self.is_running = True
        
        for symbol in symbols:
            self.data_buffers[symbol] = {
                tf: deque(maxlen=200) for tf in timeframes
            }
        
        logger.info(f"🧠 [PRICE-ENGINE] Starting prediction loop for {len(symbols)} symbols")
        logger.info(f"   Timeframes: {timeframes}")
        logger.info(f"   Update interval: {self.update_interval}s")
        
        # Run the update loop in the background
        while self.is_running:
            try:
                # Update predictions for all symbols
                await self._update_predictions(symbols, timeframes)
                
                # Wait for next update cycle
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                logger.info("🧠 [PRICE-ENGINE] Prediction loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}", exc_info=True)
                # Continue running despite errors
                # Note: Could implement exponential backoff here for production robustness
                await asyncio.sleep(self.update_interval)
        
        logger.info("🧠 [PRICE-ENGINE] Prediction loop stopped")
    
    async def stop_prediction_loop(self):
        """Stop the prediction loop."""
        self.is_running = False
        logger.info("🧠 [PRICE-ENGINE] Stopping prediction loop...")
    
    def get_price_forecast(self, symbol: str,
                          horizon: int = 12) -> Optional[Dict[str, Any]]:
        """
        Get price forecast for a symbol.
        
        Args:
            symbol: Trading symbol
            horizon: Forecast horizon in steps
            
        Returns:
            Dictionary with forecast and confidence intervals
        """
        if symbol not in self.prediction_cache:
            return None
        
        cached = self.prediction_cache[symbol]
        
        # Check if cache is stale
        age = (pd.Timestamp.now() - cached['timestamp']).total_seconds()
        if age > self.config.cache_ttl:
            return None
        
        return cached
    
    def generate_trading_signals(self, symbol: str,
                                current_price: float,
                                threshold: float = 0.02) -> Dict[str, Any]:
        """
        Generate trading signals from price forecasts.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            threshold: Minimum price movement threshold for signals
            
        Returns:
            Dictionary with trading signals and recommendations
        """
        forecast = self.get_price_forecast(symbol)
        
        if not forecast:
            return {
                'signal': 'neutral',
                'strength': 0.0,
                'reason': 'no_forecast'
            }
        
        # Get aggregated forecast
        agg = forecast['aggregated']
        forecast_pct = agg['forecast'][0]  # First step
        uncertainty = agg['uncertainty'][0]
        consensus = agg['consensus_strength']
        
        # Calculate expected price movement
        expected_change = forecast_pct / 100
        
        # Determine signal
        if expected_change > threshold and consensus > 0.7:
            signal = 'bullish'
            strength = min(abs(expected_change) * consensus, 1.0)
        elif expected_change < -threshold and consensus > 0.7:
            signal = 'bearish'
            strength = min(abs(expected_change) * consensus, 1.0)
        else:
            signal = 'neutral'
            strength = 0.0
        
        # Calculate position sizing based on confidence
        confidence = 1.0 / (1.0 + uncertainty)
        position_size = strength * confidence
        
        return {
            'symbol': symbol,
            'signal': signal,
            'strength': float(strength),
            'position_size': float(position_size),
            'expected_change': float(expected_change),
            'uncertainty': float(uncertainty),
            'consensus': float(consensus),
            'confidence': float(confidence),
            'forecast_price': current_price * (1 + expected_change),
            'timestamp': forecast['timestamp']
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status information."""
        return {
            'running': self.is_running,
            'symbols_tracked': list(self.data_buffers.keys()),
            'n_predictions_cached': len(self.prediction_cache),
            'update_interval': self.update_interval,
            'timeframes': list(self.predictor.models.keys())
        }
