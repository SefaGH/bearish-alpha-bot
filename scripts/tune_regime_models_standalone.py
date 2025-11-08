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
        """Load pre-processed training data from cache or generate synthetic."""
        logger.info(f"Loading data for {symbol}...")
        
        # Try cache locations
        possible_files = [
            self.data_cache_dir / f'{symbol}_training_data.npz',
            self.data_cache_dir / 'regime_training_data.npz',
            Path('data') / 'regime_training_data.npz'
        ]
        
        for filepath in possible_files:
            if filepath.exists():
                logger.info(f"Found cached data: {filepath}")
                data = np.load(filepath)
                X = data['X']
                y = data['y']
                logger.info(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
                return X, y
        
        # Generate synthetic data
        logger.warning("⚠️  No cache found, generating synthetic data")
        return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples=7200, n_features=42):
        """Generate synthetic training data."""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 4, n_samples)
        logger.info(f"Generated {n_samples} synthetic samples")
        return X, y
    
    def create_lstm_model(self, params: dict):
        """Create LSTM model."""
        import torch
        import torch.nn as nn
        
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
        
        model = SimpleLSTM(
            input_size=params.get('input_size', 42),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_classes=4,
            dropout=params['dropout']
        )
        
        model.criterion = nn.CrossEntropyLoss()
        model.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.get('learning_rate', 0.001),
            weight_decay=params.get('weight_decay', 0.0001)
        )
        
        return model
    
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
        """Run hyperparameter tuning."""
        logger.info("="*70)
        logger.info(f"🎯 TUNING {model_type.upper()} MODEL")
        logger.info("="*70)
        
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
                return self.create_lstm_model(params)
            elif model_type == 'rf':
                return self.create_rf_model(params)
            else:
                raise ValueError(f"Unknown model: {model_type}")
        
        def metric_fn(model, X, y):
            if model_type == 'lstm':
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    outputs = model(X_tensor)
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted.numpy() == y).sum()
                    return correct / len(y)
            else:
                return model.score(X, y)
        
        best_params, best_score, study = tuner.tune(
            X=X_cv, y=y_cv,
            model_factory=model_factory,
            metric_fn=metric_fn
        )
        
        logger.info("\n🔬 Validating on hold-out...")
        final_model = model_factory(best_params)
        
        if model_type == 'lstm':
            self._quick_train_lstm(final_model, X_cv, y_cv, epochs=10)
            holdout_score = metric_fn(final_model, X_test, y_test)
        else:
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
    
    def _quick_train_lstm(self, model, X, y, epochs=10):
        """Quick LSTM training."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                model.optimizer.zero_grad()
                outputs = model(batch_X)
                loss = model.criterion(outputs, batch_y)
                loss.backward()
                model.optimizer.step()
    
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
