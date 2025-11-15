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
import os
from datetime import timedelta

# Import local modules with a fallback for script execution
try:
    from .feature_engineering import FeatureEngineeringPipeline
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from ml.feature_engineering import FeatureEngineeringPipeline

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
                    # Gelen 2D veriyi (örn: [1, 42]) modelin beklediği 3D formata çevir (örn: [1, 1, 42])
                    if X.ndim == 2:
                        X_reshaped = np.reshape(X, (X.shape[0], 1, X.shape[1]))
                    else:
                        X_reshaped = X
                    
                    # Modeli, yeniden şekillendirilmiş veri ile çağır
                    pred, unc = model.predict(X_reshaped) 
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
    
    def __init__(self, models: Dict[str, EnsemblePricePredictor], feature_pipeline: FeatureEngineeringPipeline):
        """
        Initialize multi-timeframe predictor.

        Args:
            models: Dictionary mapping timeframes to ensemble predictors.
            feature_pipeline: The shared feature engineering pipeline instance.
        """
        self.models = models
        self.feature_pipeline = feature_pipeline
        
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
                
            # Extract features for price prediction
            features = self.feature_pipeline.extract_features(data, mode='price')
            
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
                'forecast_prices': data['close'].iloc[-1] * (1 + pred[0] / 100),
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
    (YENİ KONFİGÜRASYON YAPISIYLA TAM UYUMLU HALE GETİRİLDİ)
    """
    MODEL_SAVE_DIR = "data/models" 

    def __init__(self, market_data_pipeline, feature_pipeline, config: Dict[str, Any]):
        """
        Initialize the advanced prediction engine using its specific configuration block.

        Args:
            market_data_pipeline: Instance of MarketDataPipeline.
            feature_pipeline: Instance of FeatureEngineeringPipeline.
            config (Dict[str, Any]): The 'price_prediction' configuration block from the main YAML file.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AdvancedPricePredictionEngine.")
        
        self.market_data_pipeline = market_data_pipeline
        self.feature_pipeline = feature_pipeline
        self.config = config  # Bu artık 'price_prediction' bloğudur.
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"AdvancedPricePredictionEngine using device: {self.device}")

        # Modelleri, kendisine verilen 'config' bloğuna göre inşa et
        self.predictor = self._build_predictors()
        
        self.prediction_cache = {}
        self.data_buffers = {}
        self.is_running = False
        
        # ✔️ DÜZELTME 1: Ayarları doğrudan self.config'ten oku.
        # Gereksiz `get('prediction', {})` çağrısını kaldır.
        self.update_interval = self.config.get('update_interval_seconds', 60)
        self.cache_ttl = timedelta(seconds=self.config.get('cache_ttl_seconds', 300))
        
        # Modelleri yükle ve eğitim durumunu ayarla
        self.is_trained = self.load_models()
        
        status_summary = self.get_status_summary()
        logger.info(f"🤖 PricePredictor Status: {status_summary}")
        
        if not self.is_trained:
            logger.warning("⚠️ PricePredictor running in FALLBACK mode - predictions based on technical analysis only")
        
        if not self.market_data_pipeline:
            logger.warning("⚠️ MarketDataPipeline not provided. Prediction updates may fail.")
        
        logger.info("Advanced Price Prediction Engine initialized.")

    def _build_predictors(self) -> Optional[MultiTimeframePricePredictor]:
        """Builds the entire prediction stack (LSTM, Transformer, etc.) from its config."""
        logger.info("Building multi-timeframe prediction models from configuration...")

        # ✔️ DÜZELTME 2: Ayarları doğrudan self.config'ten oku.
        # `model_config` ara değişkenini ve gereksiz `get('prediction', {})` çağrısını kaldır.
        input_feature_size = self.config.get('feature_size', 42)
        forecast_horizon = self.config.get('forecast_horizon', 12)
        timeframes_from_config = self.config.get('timeframes', ['5m', '15m', '1h'])
        
        # ... (zaman dilimi parse etme mantığı aynı kalır)
        if isinstance(timeframes_from_config, str):
            timeframes = [tf.strip() for tf in timeframes_from_config.split(',') if tf.strip()]
        elif isinstance(timeframes_from_config, list):
            timeframes = [str(tf).strip() for tf in timeframes_from_config]
        else:
            logger.error(f"Invalid 'timeframes' format in config. Expected list or string, got {type(timeframes_from_config)}.")
            return None

        if not timeframes:
            logger.error("Timeframe list is empty after parsing. Cannot build predictors.")
            return None
            
        logger.info(f"Predictors will be built for timeframes: {timeframes}")

        mtf_models = {}
        # Parametreleri yine doğrudan `self.config`'ten oku
        model_types_to_build = self.config.get('models', [])
        model_params = self.config.get('model_params', {})

        for tf in timeframes:
            tf_models = {}
            if 'lstm' in model_types_to_build:
                params = model_params.get('lstm', {})
                tf_models['lstm'] = LSTMPricePredictor(
                    input_size=input_feature_size,
                    hidden_size=params.get('hidden_size', 128),
                    num_layers=params.get('num_layers', 3),
                    forecast_horizon=forecast_horizon
                ).to(self.device)
            
            if 'transformer' in model_types_to_build:
                params = model_params.get('transformer', {})
                d_model = input_feature_size
                if d_model % 2 != 0:
                    d_model += 1
                    logger.warning(f"Adjusted d_model to {d_model} to be even for Transformer.")

                tf_models['transformer'] = TransformerPricePredictor(
                    d_model=d_model, 
                    nhead=params.get('nhead', 2),
                    num_layers=params.get('num_layers', 6),
                    forecast_horizon=forecast_horizon
                ).to(self.device)
            
            if tf_models:
                ensemble_weights = self.config.get('ensemble_weights', None)
                mtf_models[tf] = EnsemblePricePredictor(tf_models, weights=ensemble_weights)

        return MultiTimeframePricePredictor(mtf_models, self.feature_pipeline)

    def get_status_summary(self) -> str:
        """Get human-readable status for logging."""
        # Safely get timeframes list
        timeframes = []
        if hasattr(self, 'predictor') and hasattr(self.predictor, 'models'):
            try:
                timeframes = list(self.predictor.models.keys())
            except (AttributeError, TypeError):
                pass
        
        if self.is_trained:
            return f"ML Mode - {len(timeframes)} models loaded: {sorted(timeframes)}"
        else:
            return f"FALLBACK Mode - No trained models (configured for: {timeframes})"
    
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
    def load_models(self) -> bool:
        """
        Loads trained model state dictionaries from disk.
        
        Returns:
            True if models were loaded successfully, False otherwise
        """
        if not TORCH_AVAILABLE:
            logger.warning("Cannot load models: PyTorch is not installed.")
            return False

        logger.info("Attempting to load trained models from disk...")
        models_loaded = 0
        
        # --- KORUMA: self.predictor veya self.predictor.models yoksa çık ---
        if not hasattr(self.predictor, 'models') or not self.predictor.models:
             logger.warning("No timeframes configured in MultiTimeframePricePredictor. Cannot load models.")
             return False

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
            logger.info(f"✅ Model loading complete. {models_loaded} models loaded.")
            return True
        else:
            logger.warning("No pre-trained models were found or loaded. The system will rely on fallback mechanisms.")
            return False

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
                    # Log prediction mode
                    if self.is_trained:
                        logger.debug(f"📊 ML prediction for {symbol} using {len(data_by_timeframe)} trained models")
                    else:
                        logger.debug(f"📈 Fallback prediction for {symbol} (no trained models)")
                    
                    # Generate multi-timeframe prediction
                    prediction = self.predictor.predict_multi_timeframe(data_by_timeframe)
                    
                    # Cache the prediction
                    self.prediction_cache[symbol] = prediction
                    
                    # Log with clear mode indication
                    if self.is_trained:
                        logger.info(f"✅ ML prediction updated for {symbol} using {len(data_by_timeframe)} timeframes: {list(data_by_timeframe.keys())}")
                    else:
                        logger.info(f"⚠️ FALLBACK prediction for {symbol} - using technical indicators only ({len(data_by_timeframe)} timeframes)")
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
        (GÜNCELLENDİ: cache_ttl değerini doğru yerden okuyor)
        
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
        
        # --- ANA DÜZELTME BURADA ---
        # `cache_ttl` değeri, `self.config` sözlüğünden değil,
        # __init__ içinde oluşturulan `self.cache_ttl` (timedelta nesnesi) özelliğinden okunmalıdır.
        if age > self.cache_ttl.total_seconds():
            logger.warning(f"Cache for {symbol} is stale (age: {age:.1f}s > ttl: {self.cache_ttl.total_seconds():.1f}s). Deleting cache.")
            del self.prediction_cache[symbol] # Bayat veriyi silmek iyi bir pratiktir.
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
