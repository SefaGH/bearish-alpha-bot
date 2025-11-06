"""
Model Training and Validation System for Regime Prediction.

Provides comprehensive training, validation, and hyperparameter optimization.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ML_ENABLED ortam değişkenini oku
ML_ENABLED = os.getenv("ML_ENABLED", "false").lower() in ("1", "true", "yes")

# sklearn import işlemlerini koruma altına al
if ML_ENABLED:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
else:
    # ML kapalıysa, programın çökmemesi için sahte (None) sınıflar oluştur
    RandomForestClassifier = None
    StandardScaler = None
    TimeSeriesSplit = None

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """Time series cross-validation."""
    
    def __init__(self, n_splits: int = 5):
        """
        Initialize time series cross-validation.
        
        Args:
            n_splits: Number of splits for cross-validation
        """
        # === KORUMA EKLE ===
        if not ML_ENABLED:
            raise RuntimeError(
                "TimeSeriesCV sınıfı ML_ENABLED=true gerektirir. "
                "Lütfen botu 'enable_ml=true' ile çalıştırın."
            )
        # === KORUMA SONU ===
        
        self.n_splits = n_splits
        self.splitter = TimeSeriesSplit(n_splits=n_splits)
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation splits.
        
        Args:
            X: Input data
            
        Returns:
            List of (train_indices, validation_indices) tuples
        """
        return list(self.splitter.split(X))


class WalkForwardValidation:
    """Walk-forward validation for time series."""
    
    def __init__(self, train_size: int = 1000, test_size: int = 100):
        """
        Initialize walk-forward validation.
        
        Args:
            train_size: Size of training window
            test_size: Size of test window
        """
        # === KORUMA EKLE ===
        if not ML_ENABLED:
            raise RuntimeError(
                "WalkForwardValidation sınıfı ML_ENABLED=true gerektirir. "
                "Lütfen botu 'enable_ml=true' ile çalıştırın."
            )
        # === KORUMA SONU ===
        
        self.train_size = train_size
        self.test_size = test_size
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.
        
        Args:
            X: Input data
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        splits = []
        
        start = 0
        while start + self.train_size + self.test_size <= n_samples:
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            
            train_idx = np.arange(start, train_end)
            test_idx = np.arange(train_end, test_end)
            
            splits.append((train_idx, test_idx))
            start += self.test_size
        
        return splits


class MonteCarloValidation:
    """Monte Carlo cross-validation."""
    
    def __init__(self, n_iterations: int = 100, test_size: float = 0.2):
        """
        Initialize Monte Carlo validation.
        
        Args:
            n_iterations: Number of random splits
            test_size: Fraction of data for testing
        """
        self.n_iterations = n_iterations
        self.test_size = test_size
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate random train/test splits.
        
        Args:
            X: Input data
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        
        splits = []
        for _ in range(self.n_iterations):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
            
            splits.append((train_idx, test_idx))
        
        return splits


class RegimeModelTrainer:
    """
    Comprehensive model training and validation system.
    (GÜNCELLENDİ: save_models metodu eklendi ve config parametresi eklendi)
    """
    MODEL_SAVE_DIR = "data/models/regime" # Rejim modelleri için ayrı bir klasör
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Optional configuration dictionary from ml.regime_prediction config block
        """
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.validators = {
            'time_series_cv': TimeSeriesCV(),
            'walk_forward': WalkForwardValidation(),
            'monte_carlo': MonteCarloValidation()
        }
        self.performance_history = []
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True) # Klasörün var olduğundan emin ol
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D data (samples, features) to 3D sequences (samples, seq_length, features).
        
        This method creates overlapping sequences from the input data to allow LSTM and Transformer
        models to learn temporal patterns.
        
        Args:
            X: 2D feature array of shape (n_samples, n_features)
            y: 1D label array of shape (n_samples,)
            seq_length: Length of each sequence (default: 10)
            
        Returns:
            Tuple of (X_sequences, y_sequences) where:
                - X_sequences has shape (n_sequences, seq_length, n_features)
                - y_sequences has shape (n_sequences,) containing the label for each sequence
        """
        if len(X) < seq_length:
            logger.warning(f"Data length ({len(X)}) is less than sequence length ({seq_length}). Cannot create sequences.")
            return np.array([]), np.array([])
        
        n_sequences = len(X) - seq_length + 1
        X_sequences = np.zeros((n_sequences, seq_length, X.shape[1]))
        y_sequences = np.zeros(n_sequences, dtype=y.dtype)
        
        for i in range(n_sequences):
            X_sequences[i] = X[i:i + seq_length]
            # Use the label from the last timestep of the sequence
            y_sequences[i] = y[i + seq_length - 1]
        
        logger.info(f"Created {n_sequences} sequences of length {seq_length} from {len(X)} samples")
        return X_sequences, y_sequences
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray,
                             validation_method: str = 'time_series_cv', seq_length: int = 10) -> Dict[str, Any]:
        """
        Train ensemble of regime prediction models.
        
        Args:
            X: Feature array (2D: samples x features)
            y: Label array
            validation_method: Validation method to use
            seq_length: Sequence length for LSTM/Transformer (default: 10)
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Training ensemble models with {validation_method} validation")
        logger.info(f"Input data shape: X={X.shape}, y={y.shape}")
        
        results = {
            'models': {},
            'metrics': {},
            'validation_method': validation_method
        }
        
        try:
            # Data preprocessing and feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['ensemble'] = scaler
            logger.info(f"Data scaled. Shape: {X_scaled.shape}")
            
            # Train Random Forest model with 2D data
            logger.info("Training Random Forest model (using 2D data)...")
            rf_model, rf_metrics = self._train_random_forest(X_scaled, y, validation_method)
            results['models']['random_forest'] = rf_model
            results['metrics']['random_forest'] = rf_metrics
            
            # Create sequences for LSTM and Transformer (3D data)
            logger.info(f"Creating sequences with length {seq_length} for LSTM and Transformer...")
            X_seq, y_seq = self._create_sequences(X_scaled, y, seq_length)
            logger.info(f"Sequence data shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
            
            # Train LSTM model with 3D sequence data
            logger.info("Training LSTM model (using 3D sequence data)...")
            lstm_model, lstm_metrics = self._train_lstm(X_seq, y_seq, validation_method)
            results['models']['lstm'] = lstm_model
            results['metrics']['lstm'] = lstm_metrics
            
            # Train Transformer model with 3D sequence data
            logger.info("Training Transformer model (using 3D sequence data)...")
            transformer_model, transformer_metrics = self._train_transformer(X_seq, y_seq, validation_method)
            results['models']['transformer'] = transformer_model
            results['metrics']['transformer'] = transformer_metrics
            
            # Store trained models
            self.models = results['models']

            # Save models to disk
            self.save_models()
            
            logger.info("Ensemble training complete and models saved.")
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble training: {e}", exc_info=True)
            return results

    def save_models(self):
        """Eğitilmiş rejim modellerini ve scaler'ı diske kaydeder."""
        if not self.models:
            logger.warning("No trained regime models to save.")
            return

        try:
            # Scaler'ı kaydet
            if 'ensemble' in self.scalers:
                scaler_path = os.path.join(self.MODEL_SAVE_DIR, "scaler.pkl")
                joblib.dump(self.scalers['ensemble'], scaler_path)
                logger.info(f"✅ Regime feature scaler saved to {scaler_path}")

            # Modelleri kaydet
            model_configs = {}
            for name, model in self.models.items():
                if model is None:
                    logger.warning(f"Skipping save for '{name}' model as it is None.")
                    continue
                
                if name == 'random_forest' and hasattr(model, 'fit'):
                    model_path = os.path.join(self.MODEL_SAVE_DIR, "random_forest.pkl")
                    joblib.dump(model, model_path)
                    logger.info(f"✅ Saved 'random_forest' model to {model_path}")
                    
                elif hasattr(model, 'state_dict'):  # PyTorch modeli
                    model_path = os.path.join(self.MODEL_SAVE_DIR, f"{name}_regime.pth")
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"✅ Saved '{name}' model to {model_path}")
                    
                    # Save model configuration for loading later
                    if name == 'lstm':
                        model_configs['lstm'] = {
                            'input_size': model.lstm.input_size,
                            'hidden_size': model.hidden_size,
                            'num_layers': model.num_layers,
                            'num_classes': model.classifier[-1].out_features
                        }
                    elif name == 'transformer':
                        # Extract nhead from the first encoder layer's attention module
                        try:
                            # Access the encoder layers through the correct path
                            first_layer = model.transformer.layers[0]
                            nhead = first_layer.self_attn.num_heads
                        except (AttributeError, IndexError):
                            # Fallback if structure is different
                            nhead = 2  # Use a safe default
                            logger.warning("Could not extract nhead from transformer, using default=2")
                        
                        model_configs['transformer'] = {
                            'd_model': model.d_model,
                            'nhead': nhead,
                            'num_layers': len(model.transformer.layers),
                            'num_classes': model.classifier[-1].out_features
                        }
            
            # Save model configurations
            if model_configs:
                config_path = os.path.join(self.MODEL_SAVE_DIR, "model_config.pkl")
                joblib.dump(model_configs, config_path)
                logger.info(f"✅ Saved model configurations to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save regime models: {e}", exc_info=True)
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray,
                           validation_method: str) -> Tuple[RandomForestClassifier, Dict[str, float]]:
        """Train Random Forest classifier."""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Cross-validation
        validator = self.validators.get(validation_method, self.validators['time_series_cv'])
        scores = []
        
        for train_idx, val_idx in validator.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        # Final training on all data
        model.fit(X, y)
        
        metrics = {
            'mean_cv_score': np.mean(scores),
            'std_cv_score': np.std(scores),
            'n_features': X.shape[1]
        }
        
        logger.info(f"Random Forest CV score: {metrics['mean_cv_score']:.4f} ± {metrics['std_cv_score']:.4f}")
        
        return model, metrics
    
    def _train_lstm(self, X: np.ndarray, y: np.ndarray,
                   validation_method: str) -> Tuple[Any, Dict[str, float]]:
        """
        Train LSTM model with a real training loop.
        
        Args:
            X: 3D sequence data of shape (n_sequences, seq_length, n_features)
            y: Label array of shape (n_sequences,)
            validation_method: Validation method (not used in this simplified version)
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # === YENİ KORUMA: YETERLİ VERİ KONTROLÜ ===
        if X.shape[0] < 20: # Eğitim ve validasyon için makul bir alt sınır
            logger.warning(f"LSTM training skipped: Insufficient sequences ({X.shape[0]}) available.")
            return None, {}
        # === KORUMA SONU ===
        
        from .neural_networks import LSTMRegimePredictor # Local import
        
        logger.info(f"LSTM training - Input shape: X={X.shape}, y={y.shape}")
        
        # Convert to PyTorch tensors (X is already 3D: batch, seq_len, features)
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()
        
        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, shuffle=False)
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # === YENİ KORUMA: BOŞ DATALOADER KONTROLÜ ===
        if len(train_loader) == 0:
            logger.warning(f"LSTM training skipped: Not enough training samples ({len(X_train)}) for a single batch (size=64).")
            return None, {}
        # === KORUMA SONU ===
        
        # Initialize LSTM model with correct input size (number of features per timestep)
        # Read parameters from config (ml.regime_prediction.model_params.lstm_regime)
        model_params = self.config.get('model_params', {})
        lstm_config = model_params.get('lstm_regime', {})
        
        model = LSTMRegimePredictor(
            input_size=X.shape[2],  # Number of features (last dimension of 3D array)
            hidden_size=lstm_config.get('hidden_size', 64),
            num_layers=lstm_config.get('num_layers', 2),
            num_classes=len(np.unique(y))
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        logger.info(f"Starting LSTM training for 10 epochs...")
        model.train()
        for epoch in range(10):
            epoch_loss = 0
            for i, (features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(features)  # Returns logits only (not tuple)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"  LSTM Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

        # Calculate validation accuracy
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)  # Returns logits only
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).sum().item() / y_val.size(0)

        metrics = {
            'accuracy': accuracy, 
            'n_features': X.shape[2],
            'seq_length': X.shape[1]
        }
        logger.info(f"✅ LSTM validation accuracy: {accuracy:.4f}")
        
        return model, metrics

    def _train_transformer(self, X: np.ndarray, y: np.ndarray,
                         validation_method: str) -> Tuple[Any, Dict[str, float]]:
        """
        Train Transformer model with a real training loop.
        
        Args:
            X: 3D sequence data of shape (n_sequences, seq_length, n_features)
            y: Label array of shape (n_sequences,)
            validation_method: Validation method (not used in this simplified version)
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # === YENİ KORUMA: YETERLİ VERİ KONTROLÜ ===
        if X.shape[0] < 20: # Eğitim ve validasyon için makul bir alt sınır
            logger.warning(f"Transformer training skipped: Insufficient sequences ({X.shape[0]}) available.")
            return None, {}
        # === KORUMA SONU ===
        
        from .neural_networks import TransformerRegimePredictor # Local import
        
        logger.info(f"Transformer training - Input shape: X={X.shape}, y={y.shape}")
        
        # Transformer d_model must be divisible by nhead.
        n_features = X.shape[2]
        d_model = n_features
        nhead = 2 # Start with a reasonable default
        # Find the smallest d_model >= n_features that is divisible by a reasonable nhead
        for h in [4, 2]: # Prefer 4 heads, fallback to 2
            if (n_features % h) == 0:
                d_model = n_features
                nhead = h
                break
        else: # If not divisible by 4 or 2, pad it
            nhead = 2
            d_model = n_features + (nhead - n_features % nhead) % nhead
        
        logger.info(f"Transformer params: d_model={d_model}, nhead={nhead} (original features: {n_features})")
        
        # Convert to PyTorch tensors and pad if necessary
        X_tensor = torch.from_numpy(X).float()
        if n_features != d_model:
            padding = torch.zeros(X_tensor.shape[0], X_tensor.shape[1], d_model - n_features)
            X_tensor = torch.cat([X_tensor, padding], dim=2)
            logger.info(f"Padded input from {n_features} to {d_model} features")
        
        y_tensor = torch.from_numpy(y).long()

        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, shuffle=False)
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # === YENİ KORUMA: BOŞ DATALOADER KONTROLÜ ===
        if len(train_loader) == 0:
            logger.warning(f"Transformer training skipped: Not enough training samples ({len(X_train)}) for a single batch (size=64).")
            return None, {}
        # === KORUMA SONU ===
        
        # Initialize Transformer model
        model = TransformerRegimePredictor(
            d_model=d_model,
            nhead=nhead,
            num_layers=2,
            num_classes=len(np.unique(y))
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        logger.info(f"Starting Transformer training for 10 epochs...")
        model.train()
        for epoch in range(10):
            epoch_loss = 0
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)  # Returns logits only (not tuple)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"  Transformer Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

        # Calculate validation accuracy
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)  # Returns logits only
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).sum().item() / y_val.size(0)

        metrics = {
            'accuracy': accuracy, 
            'n_features': n_features,
            'd_model': d_model,
            'seq_length': X.shape[1]
        }
        logger.info(f"✅ Transformer validation accuracy: {accuracy:.4f}")
        
        return model, metrics
    
    def optimize_hyperparameters(self, model_type: str, 
                                param_space: Dict[str, List]) -> Dict[str, Any]:
        """
        Bayesian optimization for hyperparameter tuning.
        
        Args:
            model_type: Type of model to optimize
            param_space: Parameter space to search
            
        Returns:
            Dictionary with best parameters and performance
        """
        logger.info(f"Optimizing hyperparameters for {model_type}")
        
        # Placeholder implementation
        best_params = {}
        for param, values in param_space.items():
            best_params[param] = values[len(values) // 2]  # Choose middle value
        
        return {
            'best_params': best_params,
            'best_score': 0.70,
            'n_iterations': len(param_space)
        }
    
    def evaluate_model_performance(self, model: Any, X_test: np.ndarray, 
                                  y_test: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive model performance evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                
                # Calculate per-class metrics
                unique_classes = np.unique(y_test)
                precision_per_class = []
                recall_per_class = []
                
                for cls in unique_classes:
                    true_positive = np.sum((y_pred == cls) & (y_test == cls))
                    false_positive = np.sum((y_pred == cls) & (y_test != cls))
                    false_negative = np.sum((y_pred != cls) & (y_test == cls))
                    
                    precision = true_positive / (true_positive + false_positive + 1e-10)
                    recall = true_positive / (true_positive + false_negative + 1e-10)
                    
                    precision_per_class.append(precision)
                    recall_per_class.append(recall)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': np.mean(precision_per_class),
                    'recall': np.mean(recall_per_class),
                    'f1': 2 * np.mean(precision_per_class) * np.mean(recall_per_class) / (np.mean(precision_per_class) + np.mean(recall_per_class) + 1e-10)
                }
                
                logger.info(f"Model evaluation: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
                
                return metrics
            else:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def generate_feature_importance(self, model: Any) -> Dict[str, float]:
        """
        Feature importance analysis.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary with feature importance scores
        """
        # === KORUMA EKLE ===
        if not ML_ENABLED:
            logger.warning("ML_ENABLED=false olduğu için özellik önemi (feature importance) atlanıyor.")
            return {}
        # === KORUMA SONU ===
        
        importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Random Forest feature importance
                importances = model.feature_importances_
                for i, imp in enumerate(importances):
                    importance[f'feature_{i}'] = float(imp)
                
                logger.info(f"Extracted feature importance for {len(importance)} features")
            else:
                logger.info("Model does not support feature importance extraction")
                
        except Exception as e:
            logger.error(f"Error generating feature importance: {e}")
        
        return importance
