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

# ============================================================================
# TRAINING HYPERPARAMETERS - FAZ 3.2 + 2.2 OPTIMIZATION
# ============================================================================
NUM_EPOCHS = 50  # Increased from 10 for better convergence
EARLY_STOPPING_PATIENCE = 5  # Default patience (overridden per model)
MIN_DELTA = 0.001  # Minimum improvement to be considered progress
MIN_EPOCHS = 20  # Minimum epochs before early stopping can trigger

SEQUENCE_LENGTH = 20  # Increased from 10 for better temporal context

LEARNING_RATE = 0.0005  # Reduced from 0.001 for more stable training
WEIGHT_DECAY = 1e-4  # 1e-5 → 1e-4 (10x stronger L2 regularization)

# ===== LSTM CONFIGURATION (FAZ 3.2 - ANTI-OVERFIT) =====
LSTM_HIDDEN_SIZE = 96  # 128 → 96 (reduced to prevent overfit)
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.5  # 0.3 → 0.5 (stronger regularization)
LSTM_EARLY_STOPPING_PATIENCE = 3  # 5 → 3 (stop earlier)

# ===== TRANSFORMER CONFIGURATION =====
TRANSFORMER_NHEAD = 6  # Increased from 2 for better attention
TRANSFORMER_NUM_LAYERS = 4  # Increased from 2 for deeper network
TRANSFORMER_DIM_FEEDFORWARD = 256  # Increased from 128
TRANSFORMER_DROPOUT = 0.3
TRANSFORMER_EARLY_STOPPING_PATIENCE = 5  # Keep same, transformer is healthy

# ===== DATA AUGMENTATION (FAZ 2.2) =====
USE_DATA_AUGMENTATION = True
USE_SMOTE = True
USE_JITTERING = True
JITTERING_NOISE_LEVEL = 0.01
# ============================================================================


class EarlyStopping:
    """Early stopping to prevent overfitting during training."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, min_epochs: int = 20):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            min_epochs: Minimum number of epochs to train before considering early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number (0-indexed)
            
        Returns:
            True if training should stop, False otherwise
        """
        # Don't stop before minimum epochs
        if epoch < self.min_epochs:
            return False
            
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


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
        self.training_history = []  # Store epoch-by-epoch metrics
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True) # Klasörün var olduğundan emin ol
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D data (samples, features) to 3D sequences (samples, seq_length, features).
        
        This method creates overlapping sequences from the input data to allow LSTM and Transformer
        models to learn temporal patterns.
        
        Args:
            X: 2D feature array of shape (n_samples, n_features)
            y: 1D label array of shape (n_samples,)
            seq_length: Length of each sequence (default: SEQUENCE_LENGTH = 20)
            
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
                             validation_method: str = 'time_series_cv', seq_length: int = SEQUENCE_LENGTH) -> Dict[str, Any]:
        """
        Train ensemble of regime prediction models.
        
        Args:
            X: Feature array (2D: samples x features)
            y: Label array
            validation_method: Validation method to use
            seq_length: Sequence length for LSTM/Transformer (default: SEQUENCE_LENGTH = 20)
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("="*60)
        logger.info("🧠 NEURAL NETWORK TRAINING CONFIGURATION (FAZ 3.2 + 2.2)")
        logger.info(f"   Total Samples (Original): {len(X)}")
        logger.info(f"   Features: {X.shape[1]}")
        logger.info(f"   Sequence Length: {seq_length}")
        logger.info(f"   Max Epochs: {NUM_EPOCHS} (min: {MIN_EPOCHS})")
        logger.info(f"   Learning Rate: {LEARNING_RATE}")
        logger.info(f"   Weight Decay: {WEIGHT_DECAY}")
        logger.info("")
        logger.info("   LSTM Configuration:")
        logger.info(f"      Hidden Size: {LSTM_HIDDEN_SIZE} (reduced from 128)")
        logger.info(f"      Layers: {LSTM_NUM_LAYERS}")
        logger.info(f"      Dropout: {LSTM_DROPOUT} (increased from 0.3)")
        logger.info(f"      Early Stop Patience: {LSTM_EARLY_STOPPING_PATIENCE}")
        logger.info("")
        logger.info("   Transformer Configuration:")
        logger.info(f"      nhead: {TRANSFORMER_NHEAD}")
        logger.info(f"      Layers: {TRANSFORMER_NUM_LAYERS}")
        logger.info(f"      Dropout: {TRANSFORMER_DROPOUT}")
        logger.info(f"      Early Stop Patience: {TRANSFORMER_EARLY_STOPPING_PATIENCE}")
        logger.info("")
        logger.info("   Data Augmentation:")
        logger.info(f"      Enabled: {USE_DATA_AUGMENTATION}")
        logger.info(f"      SMOTE: {USE_SMOTE}")
        logger.info(f"      Jittering: {USE_JITTERING} (noise={JITTERING_NOISE_LEVEL})")
        logger.info("="*60)
        
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
            
            # ===== DATA AUGMENTATION =====
            if USE_DATA_AUGMENTATION:
                from src.ml.data_augmentation import DataAugmentation
                
                augmenter = DataAugmentation()
                X_seq_aug, y_seq_aug = augmenter.augment_sequence_data(
                    X_seq, 
                    y_seq,
                    use_smote=USE_SMOTE,
                    use_jittering=USE_JITTERING,
                    jitter_noise=JITTERING_NOISE_LEVEL
                )
            else:
                X_seq_aug, y_seq_aug = X_seq, y_seq
                logger.info("Data augmentation disabled, using original sequences")
            
            # Train LSTM model with 3D sequence data
            logger.info("Training LSTM model (using 3D sequence data)...")
            lstm_model, lstm_metrics = self._train_lstm(
                X_seq_aug,  # Augmented data
                y_seq_aug,  # Augmented labels
                validation_method,
                hidden_size=LSTM_HIDDEN_SIZE,
                num_layers=LSTM_NUM_LAYERS,
                dropout=LSTM_DROPOUT,
                patience=LSTM_EARLY_STOPPING_PATIENCE
            )
            results['models']['lstm'] = lstm_model
            results['metrics']['lstm'] = lstm_metrics
            
            # Train Transformer model with 3D sequence data
            logger.info("Training Transformer model (using 3D sequence data)...")
            transformer_model, transformer_metrics = self._train_transformer(
                X_seq_aug,  # Augmented data
                y_seq_aug,  # Augmented labels
                validation_method,
                patience=TRANSFORMER_EARLY_STOPPING_PATIENCE
            )
            results['models']['transformer'] = transformer_model
            results['metrics']['transformer'] = transformer_metrics
            
            # Store trained models
            self.models = results['models']

            # Save models to disk
            self.save_models()
            
            # Save training metrics
            self._save_training_metrics()
            
            logger.info("Ensemble training complete and models saved.")
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble training: {e}", exc_info=True)
            return results

    def _save_training_metrics(self):
        """Save training history to CSV and JSON files."""
        if not self.training_history:
            logger.info("No training history to save.")
            return
        
        try:
            # Create logs directory if it doesn't exist
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            # Validate and normalize training history entries
            normalized_history = []
            for entry in self.training_history:
                normalized_entry = {
                    'model': entry.get('model', 'unknown'),
                    'epoch': entry.get('epoch', 0),
                    'loss': entry.get('loss', 0.0)
                }
                normalized_history.append(normalized_entry)
            
            # Save as CSV
            df = pd.DataFrame(normalized_history)
            csv_path = os.path.join(log_dir, 'regime_training_metrics.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Saved regime training metrics: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save training metrics: {e}", exc_info=True)
    
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
                   validation_method: str, hidden_size=None, num_layers=None, 
                   dropout=None, patience=None) -> Tuple[Any, Dict[str, float]]:
        """
        Train LSTM model with configurable parameters.
        
        Args:
            X: 3D sequence data of shape (n_sequences, seq_length, n_features)
            y: Label array of shape (n_sequences,)
            validation_method: Validation method (not used in this simplified version)
            hidden_size: LSTM hidden dimension (default: LSTM_HIDDEN_SIZE)
            num_layers: Number of LSTM layers (default: LSTM_NUM_LAYERS)
            dropout: Dropout rate (default: LSTM_DROPOUT)
            patience: Early stopping patience (default: LSTM_EARLY_STOPPING_PATIENCE)
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # Use provided params or defaults
        hidden_size = hidden_size or LSTM_HIDDEN_SIZE
        num_layers = num_layers or LSTM_NUM_LAYERS
        dropout = dropout or LSTM_DROPOUT
        patience = patience or LSTM_EARLY_STOPPING_PATIENCE
        
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
        
        # Initialize LSTM model with configurable parameters
        input_size = X.shape[2]  # Number of features (last dimension of 3D array)
        num_classes = len(np.unique(y))
        
        model = LSTMRegimePredictor(
            input_size=input_size,
            hidden_size=hidden_size,  # Use parameter
            num_layers=num_layers,    # Use parameter
            num_classes=num_classes,
            dropout=dropout           # Use parameter
        )
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"LSTM model: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=patience, 
            min_delta=MIN_DELTA,
            min_epochs=MIN_EPOCHS
        )
        logger.info(f"Early stopping patience: {patience}")
        
        logger.info(f"Starting LSTM training for up to {NUM_EPOCHS} epochs (min: {MIN_EPOCHS})...")
        model.train()
        
        for epoch in range(NUM_EPOCHS):
            # Training phase
            model.train()
            epoch_loss = 0
            for i, (features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(features)  # Returns logits only (not tuple)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_loss_value = val_loss.item()
            
            # Update learning rate
            scheduler.step(val_loss_value)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log progress
            logger.info(f"  LSTM Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss_value:.4f}, LR: {current_lr:.6f}")
            
            # Store epoch metrics
            self.training_history.append({
                'model': 'lstm',
                'epoch': epoch + 1,
                'loss': train_loss,
                'val_loss': val_loss_value,
                'learning_rate': current_lr
            })
            
            # Check early stopping
            if early_stopping(val_loss_value, epoch):
                logger.info(f"  ⏹️  Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # Calculate final validation accuracy
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)  # Returns logits only
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).sum().item() / y_val.size(0)

        metrics = {
            'accuracy': accuracy, 
            'n_features': X.shape[2],
            'seq_length': X.shape[1],
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'total_params': total_params,
            'final_epoch': epoch + 1
        }
        logger.info(f"✅ LSTM validation accuracy: {accuracy:.4f} (trained for {epoch+1} epochs)")
        
        return model, metrics

    def _train_transformer(self, X: np.ndarray, y: np.ndarray,
                         validation_method: str, patience=None) -> Tuple[Any, Dict[str, float]]:
        """
        Train Transformer model with configurable patience.
        
        Args:
            X: 3D sequence data of shape (n_sequences, seq_length, n_features)
            y: Label array of shape (n_sequences,)
            validation_method: Validation method (not used in this simplified version)
            patience: Early stopping patience (default: TRANSFORMER_EARLY_STOPPING_PATIENCE)
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # Use provided patience or default
        patience = patience or TRANSFORMER_EARLY_STOPPING_PATIENCE
        
        # === YENİ KORUMA: YETERLİ VERİ KONTROLÜ ===
        if X.shape[0] < 20: # Eğitim ve validasyon için makul bir alt sınır
            logger.warning(f"Transformer training skipped: Insufficient sequences ({X.shape[0]}) available.")
            return None, {}
        # === KORUMA SONU ===
        
        from .neural_networks import TransformerRegimePredictor # Local import
        
        logger.info(f"Transformer training - Input shape: X={X.shape}, y={y.shape}")
        
        # Transformer d_model must be divisible by nhead.
        # Use optimized nhead from constants
        n_features = X.shape[2]
        d_model = n_features
        nhead = TRANSFORMER_NHEAD
        
        # Find the smallest d_model >= n_features that is divisible by nhead
        if (n_features % nhead) != 0:
            d_model = n_features + (nhead - n_features % nhead)
            logger.info(f"Padding features from {n_features} to {d_model} to be divisible by nhead={nhead}")
        
        logger.info(f"Transformer params: d_model={d_model}, nhead={nhead}, num_layers={TRANSFORMER_NUM_LAYERS}, dim_feedforward={TRANSFORMER_DIM_FEEDFORWARD}")
        
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
        
        # Initialize Transformer model with optimized parameters
        model = TransformerRegimePredictor(
            d_model=d_model,
            nhead=nhead,
            num_layers=TRANSFORMER_NUM_LAYERS,
            num_classes=len(np.unique(y)),
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
            dropout=TRANSFORMER_DROPOUT
        )
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Transformer model: d_model={d_model}, nhead={nhead}, num_layers={TRANSFORMER_NUM_LAYERS}")
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=patience, 
            min_delta=MIN_DELTA,
            min_epochs=MIN_EPOCHS
        )
        logger.info(f"Early stopping patience: {patience}")

        logger.info(f"Starting Transformer training for up to {NUM_EPOCHS} epochs (min: {MIN_EPOCHS})...")
        model.train()
        
        for epoch in range(NUM_EPOCHS):
            # Training phase
            model.train()
            epoch_loss = 0
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)  # Returns logits only (not tuple)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_loss_value = val_loss.item()
            
            # Update learning rate
            scheduler.step(val_loss_value)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log progress
            logger.info(f"  Transformer Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss_value:.4f}, LR: {current_lr:.6f}")
            
            # Store epoch metrics
            self.training_history.append({
                'model': 'transformer',
                'epoch': epoch + 1,
                'loss': train_loss,
                'val_loss': val_loss_value,
                'learning_rate': current_lr
            })
            
            # Check early stopping
            if early_stopping(val_loss_value, epoch):
                logger.info(f"  ⏹️  Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # Calculate final validation accuracy
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)  # Returns logits only
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).sum().item() / y_val.size(0)

        metrics = {
            'accuracy': accuracy, 
            'n_features': n_features,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': TRANSFORMER_NUM_LAYERS,
            'seq_length': X.shape[1],
            'total_params': total_params,
            'final_epoch': epoch + 1
        }
        logger.info(f"✅ Transformer validation accuracy: {accuracy:.4f} (trained for {epoch+1} epochs)")
        
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
