"""
Standalone Hyperparameter Tuning for Regime Models
Uses cached/synthetic data to avoid import dependencies.

Usage:
    python scripts/tune_regime_models_standalone.py --model lstm --trials 30

Author: SefaGH & GitHub Copilot
Date: 2025-11-08
"""

import argparse
import sys
import os
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.validation_framework import TimeSeriesValidator, ValidationReport
from scripts.utils.optuna_tuner import OptunaModelTuner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegimeModelTuner:
    """Tune regime models using pre-processed data."""
    
    def __init__(self):
        self.data_cache_dir = Path('data/cache')
    
    def load_cached_data(self, symbol='BTC-USDT'):
        """Load pre-processed training data from cache."""
        logger.info(f"Loading cached data for {symbol}...")
        
        # Try cache locations (matching prepare_training_data.py naming)
        cache_file = self.data_cache_dir / f'{symbol}_training_data.npz'
        
        if cache_file.exists():
            logger.info(f"✅ Found cached data: {cache_file}")
            data = np.load(cache_file)
            X = data['X']
            y = data['y']
            
            logger.info(f"✅ Loaded {len(X)} real samples with {X.shape[1]} features")
            
            # Show label distribution
            unique, counts = np.unique(y, return_counts=True)
            logger.info("Label distribution:")
            label_names = ['Bullish', 'Bearish', 'Neutral', 'Volatile']
            for label, count in zip(unique, counts):
                percentage = (count / len(y)) * 100
                logger.info(f"  {label_names[label]}: {count} ({percentage:.1f}%)")
            
            return X, y
        
        # ❌ NO SYNTHETIC DATA FALLBACK
        raise FileNotFoundError(
            f"Training data not found: {cache_file}\n"
            f"Please run: python scripts/prepare_training_data.py --symbol {symbol.replace('-', '/')}"
        )
    
    def _generate_synthetic_data(self, n_samples=7200, n_features=42):
        """Generate synthetic training data."""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 4, n_samples)
        logger.info(f"Generated {n_samples} synthetic samples")
        return X, y
    
    def create_lstm_model(self, params: dict):
        """Create sklearn-compatible LSTM wrapper with class weighting."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_size, num_classes)
                
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out
        
        class SklearnLSTMWrapper:
            """Sklearn-compatible wrapper with class weighting and early stopping."""
            
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, 
                        learning_rate, weight_decay, batch_size, class_weights=None):
                self.model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes, dropout)
                
                # Use class weights if provided
                if class_weights is not None:
                    self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
                else:
                    self.criterion = nn.CrossEntropyLoss()
                
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                self.batch_size = batch_size
                self.num_epochs = 30
                self.patience = 5
            
            def fit(self, X, y):
                """Sklearn-style fit with validation split and early stopping."""
                import torch
                from torch.utils.data import TensorDataset, DataLoader
                
                # Validation split (20% of training data)
                val_split = int(len(X) * 0.8)
                X_train, X_val = X[:val_split], X[val_split:]
                y_train, y_val = y[:val_split], y[val_split:]
                
                # Create data loaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.LongTensor(y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.LongTensor(y_val)
                )
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )
                
                # Early stopping variables
                best_val_loss = float('inf')
                patience_counter = 0
                best_model_state = None
                
                # Training loop with early stopping
                for epoch in range(self.num_epochs):
                    # Training phase
                    self.model.train()
                    train_loss = 0
                    for batch_X, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                        self.optimizer.step()
                        train_loss += loss.item()
                    
                    train_loss /= len(train_loader)
                    
                    # Validation phase
                    self.model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = self.model(batch_X)
                            loss = self.criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            # Restore best model
                            if best_model_state is not None:
                                self.model.load_state_dict(best_model_state)
                            break
                
                return self
            
            def predict(self, X):
                """Sklearn-style predict method."""
                import torch
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    outputs = self.model(X_tensor)
                    _, predicted = torch.max(outputs, 1)
                    return predicted.numpy()
            
            def score(self, X, y):
                """Sklearn-style score method (accuracy)."""
                predictions = self.predict(X)
                correct = (predictions == y).sum()
                return correct / len(y)
                
                # Calculate class weights from training data
                # This will be set during tune_model when we have access to y
                return SklearnLSTMWrapper(
                    input_size=params.get('input_size', 42),
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    num_classes=4,
                    dropout=params['dropout'],
                    learning_rate=params.get('learning_rate', 0.001),
                    weight_decay=params.get('weight_decay', 0.01),
                    batch_size=params.get('batch_size', 32),
                    class_weights=params.get('class_weights', None)
                )
            
            def create_rf_model(self, params: dict):
                """Create Random Forest model."""
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    random_state=42,
                    n_jobs=-1
                )
            
            def tune_model(self, model_type: str, X: np.ndarray, y: np.ndarray,
                           n_trials: int = 30, cv_splits: int = 5):
                """Run hyperparameter tuning with class weighting."""
                logger.info("="*70)
                logger.info(f"🎯 TUNING {model_type.upper()} MODEL")
                logger.info("="*70)
                
                # Calculate class weights (inverse frequency)
                from sklearn.utils.class_weight import compute_class_weight
                
                class_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(y),
                    y=y
                )
                logger.info(f"Class weights: {class_weights}")
                
                validator = TimeSeriesValidator(n_splits=cv_splits)
                X_cv, y_cv, X_test, y_test = validator.split_with_holdout(X, y)
                
                tuner = OptunaModelTuner(
                    model_type=model_type,
                    n_trials=n_trials,
                    cv_splits=cv_splits,
                    direction='maximize'
                )
                
                def model_factory(params):
                    if model_type == 'lstm':
                        params['input_size'] = X.shape[1]
                        params['class_weights'] = class_weights  # ADD CLASS WEIGHTS
                        return self.create_lstm_model(params)
                    elif model_type == 'rf':
                        return self.create_rf_model(params)
                    else:
                        raise ValueError(f"Unknown model: {model_type}")
                
                # For LSTM, use default score method (already implemented in wrapper)
                # For RF, use sklearn's score method
                best_params, best_score, study = tuner.tune(
                    X=X_cv, y=y_cv,
                    model_factory=model_factory,
                    metric_fn=None  # Use default score() method
                )
                
                logger.info("\n🔬 Validating on hold-out...")
                final_model = model_factory(best_params)
                final_model.fit(X_cv, y_cv)
                holdout_score = final_model.score(X_test, y_test)
                
                logger.info(f"Hold-out score: {holdout_score:.4f}")
                
                results = {
                    'model_type': model_type,
                    'best_params': best_params,
                    'cv_score': float(best_score),
                    'holdout_score': float(holdout_score),
                    'n_trials': n_trials,
                    'cv_splits': cv_splits,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self._save_results(results, model_type)
                return results
            
            def _save_results(self, results: dict, model_type: str):
                """Save results."""
                output_dir = Path('logs/tuning_results')
                output_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"{model_type}_tuning_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = output_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"✅ Results saved: {filepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['lstm', 'rf'])
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--symbol', default='BTC-USDT')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("🎯 REGIME MODEL TUNING (STANDALONE)")
    logger.info("="*70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Trials: {args.trials}")
    logger.info("="*70)
    
    tuner = RegimeModelTuner()
    X, y = tuner.load_cached_data(args.symbol)
    
    results = tuner.tune_model(
        model_type=args.model,
        X=X, y=y,
        n_trials=args.trials,
        cv_splits=args.cv_splits
    )
    
    logger.info("\n" + "="*70)
    logger.info("✅ TUNING COMPLETE")
    logger.info("="*70)
    logger.info(f"CV Score: {results['cv_score']:.4f}")
    logger.info(f"Hold-out: {results['holdout_score']:.4f}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
