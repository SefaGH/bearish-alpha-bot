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
        
        # Try cache locations
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
        
        # NO SYNTHETIC DATA FALLBACK
        raise FileNotFoundError(
            f"Training data not found: {cache_file}\n"
            f"Please run: python scripts/prepare_training_data.py --symbol {symbol.replace('-', '/')}"
        )
    
    def create_lstm_model(self, params: dict):
        """Create sklearn-compatible LSTM wrapper with anti-overfitting measures."""
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
            """Sklearn-compatible wrapper with early stopping and regularization."""
            
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
                self.num_epochs = 30  # Increased for early stopping
                self.patience = 5     # Early stopping patience
            
            def fit(self, X, y):
                """Sklearn-style fit with validation split and early stopping."""
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
                
                # Training loop
                for epoch in range(self.num_epochs):
                    # Training phase
                    self.model.train()
                    train_loss = 0
                    for batch_X, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                            if best_model_state is not None:
                                self.model.load_state_dict(best_model_state)
                            break
                
                return self
            
            def predict(self, X):
                """Sklearn-style predict method."""
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
        
        # Return wrapped model
        return SklearnLSTMWrapper(
            input_size=params.get('input_size', 42),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_classes=params.get('num_classes', 3),  # Default to 3 classes
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
        """Run hyperparameter tuning with balanced split."""
        logger.info("="*70)
        logger.info(f"🎯 TUNING {model_type.upper()} MODEL")
        logger.info("="*70)
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        
        num_classes = len(np.unique(y))
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        logger.info(f"Class weights: {class_weights}")
        logger.info(f"Number of classes: {num_classes}")
        
        # =====================================================================
        # STRATIFIED TIME SERIES SPLIT
        # =====================================================================
        # We want to:
        # 1. Maintain temporal order (last 20% is hold-out)
        # 2. Have balanced class distribution in both train and test
        #
        # Solution: Use last 20% as hold-out (temporal), but use class weights
        # to compensate for distribution differences during training
        # =====================================================================
        
        # Simple temporal split (last 20% as hold-out)
        split_idx = int(len(X) * 0.8)
        X_cv = X[:split_idx]
        y_cv = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        logger.info(f"\nTemporal split:")
        logger.info(f"  CV samples: {len(X_cv)} (80%)")
        logger.info(f"  Test samples: {len(X_test)} (20%)")
        
        # Show distributions
        logger.info("\n  CV distribution:")
        cv_unique, cv_counts = np.unique(y_cv, return_counts=True)
        label_names = ['Bullish', 'Bearish', 'Neutral', 'Volatile']
        for l, c in zip(cv_unique, cv_counts):
            logger.info(f"    {label_names[l]}: {c:4d} ({c/len(y_cv)*100:5.1f}%)")
        
        logger.info("\n  Test distribution:")
        test_unique, test_counts = np.unique(y_test, return_counts=True)
        for l, c in zip(test_unique, test_counts):
            logger.info(f"    {label_names[l]}: {c:4d} ({c/len(y_test)*100:5.1f}%)")
        
        # Calculate distribution shift
        logger.info("\n  Distribution shift:")
        max_shift = 0
        for l in range(num_classes):
            cv_pct = (cv_counts[l] / len(y_cv) * 100) if l < len(cv_counts) else 0
            test_pct = (test_counts[l] / len(y_test) * 100) if l < len(test_counts) else 0
            shift = abs(cv_pct - test_pct)
            max_shift = max(max_shift, shift)
            logger.info(f"    {label_names[l]}: {shift:5.1f}%")
        
        if max_shift > 10:
            logger.warning(f"\n  ⚠️  Large distribution shift ({max_shift:.1f}%)")
            logger.warning(f"  ⚠️  Using class weights to compensate")
            logger.warning(f"  ⚠️  Consider using longer training period for better balance")
        
        # =====================================================================
        # Create tuner and model factory
        # =====================================================================
        
        validator = TimeSeriesValidator(n_splits=cv_splits)
        tuner = OptunaModelTuner(
            model_type=model_type,
            n_trials=n_trials,
            cv_splits=cv_splits,
            direction='maximize'
        )
        
        def model_factory(params):
            if model_type == 'lstm':
                params['input_size'] = X.shape[1]
                params['num_classes'] = num_classes
                params['class_weights'] = class_weights
                return self.create_lstm_model(params)
            elif model_type == 'rf':
                return self.create_rf_model(params)
            else:
                raise ValueError(f"Unknown model: {model_type}")
        
        # =====================================================================
        # Run Optuna tuning on CV data
        # =====================================================================
        
        logger.info("\n🔬 Starting hyperparameter optimization...")
        best_params, best_score, study = tuner.tune(
            X=X_cv, y=y_cv,
            model_factory=model_factory,
            metric_fn=None
        )
        
        # =====================================================================
        # Validate on hold-out test set
        # =====================================================================
        
        logger.info("\n🔬 Validating on hold-out test set...")
        final_model = model_factory(best_params)
        final_model.fit(X_cv, y_cv)
        holdout_score = final_model.score(X_test, y_test)
        
        logger.info(f"Hold-out score: {holdout_score:.4f}")
        
        # Calculate gap
        gap = best_score - holdout_score
        logger.info(f"CV-Holdout gap: {gap:+.4f}")
        
        # Interpretation
        if abs(gap) > 0.15:
            logger.error(f"🔴 CRITICAL: Severe overfitting (gap: {gap:+.4f})")
        elif abs(gap) > 0.10:
            logger.warning(f"⚠️  WARNING: Overfitting detected (gap: {gap:+.4f})")
        elif abs(gap) > 0.05:
            logger.warning(f"⚠️  CAUTION: Moderate overfitting (gap: {gap:+.4f})")
        else:
            logger.info(f"✅ Good generalization (gap: {gap:+.4f})")
        
        # =====================================================================
        # Save results
        # =====================================================================
        
        results = {
            'model_type': model_type,
            'best_params': best_params,
            'cv_score': float(best_score),
            'holdout_score': float(holdout_score),
            'gap': float(gap),
            'n_trials': n_trials,
            'cv_splits': cv_splits,
            'num_classes': int(num_classes),
            'class_weights': class_weights.tolist(),
            'distribution_shift': float(max_shift),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._save_results(results, model_type)
        return results

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
    logger.info(f"Gap: {results['gap']:+.4f}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
