"""
Standalone Hyperparameter Tuning for Regime Models - FIXED VERSION
Addresses: Distribution shift, identical scores, missing method

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
        
        raise FileNotFoundError(
            f"Training data not found: {cache_file}\n"
            f"Please run: python scripts/prepare_training_data.py --symbol {symbol.replace('-', '/')}"
        )
    
    def create_lstm_model(self, params: dict):
        """Create sklearn-compatible LSTM wrapper."""
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
            """Sklearn-compatible wrapper with balanced training."""
            
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, 
                        learning_rate, weight_decay, batch_size, class_weights=None):
                self.model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes, dropout)
                
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
                self.num_epochs = 25  # Reduced for faster iteration
                self.patience = 7     # More lenient
                self.min_delta = 0.001
            
            def fit(self, X, y):
                """Sklearn-style fit with validation split."""
                val_split = int(len(X) * 0.8)
                X_train, X_val = X[:val_split], X[val_split:]
                y_train, y_val = y[:val_split], y[val_split:]
                
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.LongTensor(y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.LongTensor(y_val)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
                
                best_val_loss = float('inf')
                patience_counter = 0
                best_model_state = None
                
                for epoch in range(self.num_epochs):
                    # Training
                    self.model.train()
                    for batch_X, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    # Validation
                    self.model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = self.model(batch_X)
                            loss = self.criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    # Early stopping with min_delta
                    if val_loss < (best_val_loss - self.min_delta):
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
                """Sklearn-style score method."""
                predictions = self.predict(X)
                correct = (predictions == y).sum()
                return correct / len(y)
        
        return SklearnLSTMWrapper(
            input_size=params.get('input_size', 42),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_classes=params.get('num_classes', 3),
            dropout=params['dropout'],
            learning_rate=params.get('learning_rate', 0.001),
            weight_decay=params.get('weight_decay', 0.01),
            batch_size=params.get('batch_size', 32),
            class_weights=params.get('class_weights', None)
        )
    
    def tune_model(self, model_type: str, X: np.ndarray, y: np.ndarray,
                   n_trials: int = 30, cv_splits: int = 5):
        """Run hyperparameter tuning with balanced split."""
        logger.info("="*70)
        logger.info(f"🎯 TUNING {model_type.upper()} MODEL (BALANCED SPLIT)")
        logger.info("="*70)
        
        from sklearn.utils.class_weight import compute_class_weight
        
        num_classes = len(np.unique(y))
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        
        logger.info(f"Class weights: {class_weights}")
        logger.info(f"Number of classes: {num_classes}")
        
        # ================================================================
        # BALANCED SPLIT STRATEGY - Use middle 20% as test
        # ================================================================
        total_len = len(X)
        test_size = int(total_len * 0.2)
        test_start = int(total_len * 0.4)  # Start at 40%
        test_end = test_start + test_size   # End at 60%
        
        # Test: middle 20% (40-60%)
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train: first 40% + last 40% (skip middle)
        X_cv = np.vstack([X[:test_start], X[test_end:]])
        y_cv = np.concatenate([y[:test_start], y[test_end:]])
        
        logger.info(f"\nBalanced split (using middle 20% as test):")
        logger.info(f"  CV samples: {len(X_cv)} (80%)")
        logger.info(f"  Test samples: {len(X_test)} (20%)")
        logger.info(f"  Test period: samples {test_start}-{test_end} (middle of dataset)")
        
        # Show distributions
        logger.info("\n  CV distribution:")
        label_names = ['Bullish', 'Bearish', 'Neutral', 'Volatile']
        cv_unique, cv_counts = np.unique(y_cv, return_counts=True)
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
            logger.warning(f"\n  ⚠️  Distribution shift: {max_shift:.1f}%")
        else:
            logger.info(f"\n  ✅ Good balance: {max_shift:.1f}% max shift")
        
        # Create tuner
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
            else:
                raise ValueError(f"Unknown model: {model_type}")
        
        # Run tuning
        logger.info("\n🔬 Starting hyperparameter optimization...")
        best_params, best_score, study = tuner.tune(
            X=X_cv, y=y_cv,
            model_factory=model_factory,
            metric_fn=None
        )
        
        # Validate on hold-out
        logger.info("\n🔬 Validating on balanced hold-out test set...")
        final_model = model_factory(best_params)
        final_model.fit(X_cv, y_cv)
        holdout_score = final_model.score(X_test, y_test)
        
        gap = best_score - holdout_score
        
        logger.info(f"Hold-out score: {holdout_score:.4f}")
        logger.info(f"CV-Holdout gap: {gap:+.4f}")
        
        if abs(gap) > 0.15:
            logger.error(f"🔴 CRITICAL: Severe overfitting (gap: {gap:+.4f})")
        elif abs(gap) > 0.10:
            logger.warning(f"⚠️  WARNING: Overfitting (gap: {gap:+.4f})")
        elif abs(gap) > 0.05:
            logger.warning(f"⚠️  CAUTION: Moderate overfitting (gap: {gap:+.4f})")
        else:
            logger.info(f"✅ Good generalization (gap: {gap:+.4f})")
        
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
            'split_strategy': 'balanced_middle',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._save_results(results, model_type)
        return results
    
    def _save_results(self, results: dict, model_type: str):
        """Save tuning results to JSON file."""
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
    logger.info("🎯 REGIME MODEL TUNING (BALANCED SPLIT STRATEGY)")
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
    logger.info(f"Split strategy: {results['split_strategy']}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
