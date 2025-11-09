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
from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegimeModelTuner:
    """Tune regime models using pre-processed data."""
    
    def __init__(self):
        self.data_cache_dir = Path('data/cache')
        
    def load_cached_data(self, symbol: str) -> tuple:
        """
        Load cached training data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
        
        Returns:
            tuple: (X, y) features and labels
        
        Raises:
            FileNotFoundError: If cache file doesn't exist
            KeyError: If cache file has wrong structure
        """
        # Convert symbol to filesystem-safe filename
        symbol_safe = symbol.replace('/', '-')  # BTC/USDT → BTC-USDT
        cache_file = f"data/cache/{symbol_safe}_training_data.npz"
        
        logger.info(f"Loading cached data for {symbol}...")
        logger.info(f"Cache file: {cache_file}")
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Training data not found: {cache_file}\n"
                f"Please run: python scripts/prepare_training_data.py --symbol {symbol}"
            )
        
        logger.info(f"✅ Found cached data: {cache_file}")
        
        # Load and debug
        data = np.load(cache_file)
        
        # ✅ DEBUG: Print available keys
        logger.info(f"📋 Available keys in NPZ: {list(data.keys())}")
        
        # Try to detect correct keys automatically
        if 'X' in data and 'y' in data:
            logger.info("✅ Using keys: 'X', 'y'")
            X, y = data['X'], data['y']
        elif 'features' in data and 'labels' in data:
            logger.info("✅ Using keys: 'features', 'labels'")
            X, y = data['features'], data['labels']
        else:
            raise KeyError(
                f"Unknown NPZ structure. Available keys: {list(data.keys())}\n"
                f"Expected: ('X', 'y') or ('features', 'labels')"
            )
        
        logger.info(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y
        
        # Log label distribution
        unique, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique, counts))
        logger.info("Label distribution:")
        label_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
        for label, count in label_dist.items():
            percentage = count / len(y) * 100
            label_name = label_names.get(int(label), f'Unknown({label})')
            logger.info(f"  {label_name}: {count} ({percentage:.1f}%)")
        
        return X, y

    def prepare_data_splits(self, X, y, test_size=0.2, strategy='stratified'):
        """
        Prepare train/test splits with different strategies.
        
        Args:
            X: Features array
            y: Labels array
            test_size: Test set proportion (default: 0.2)
            strategy: Split strategy - 'stratified', 'time_last', or 'time_middle'
        
        Returns:
            tuple: (X_cv, X_test, y_cv, y_test)
        """
        if strategy == 'stratified':
            logger.info("📊 Using stratified random split (balanced distribution)")
            X_cv, X_test, y_cv, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=42,
                shuffle=True
            )
            
        elif strategy == 'time_last':
            logger.info("📊 Using time-based split (last period as test)")
            split_idx = int(len(X) * (1 - test_size))
            X_cv = X[:split_idx]
            y_cv = y[:split_idx]
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
        elif strategy == 'time_middle':
            logger.info("📊 Using time-based split (middle period as test)")
            test_start = int(len(X) * 0.4)
            test_end = int(len(X) * 0.6)
            
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            X_cv = np.vstack([X[:test_start], X[test_end:]])
            y_cv = np.concatenate([y[:test_start], y[test_end:]])
        
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
        
        # Log distributions
        from collections import Counter
        logger.info("\n📊 Data Split Summary:")
        logger.info(f"  Total: {len(X)} | CV: {len(X_cv)} ({len(X_cv)/len(X)*100:.1f}%) | Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        cv_dist = Counter(y_cv)
        test_dist = Counter(y_test)
        label_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
        
        logger.info("\n  CV Distribution:")
        for label in sorted(cv_dist.keys()):
            name = label_names.get(label, f'Class_{label}')
            pct = cv_dist[label] / len(y_cv) * 100
            logger.info(f"    {name}: {cv_dist[label]:4d} ({pct:5.1f}%)")
        
        logger.info("\n  Test Distribution:")
        for label in sorted(test_dist.keys()):
            name = label_names.get(label, f'Class_{label}')
            pct = test_dist[label] / len(y_test) * 100
            logger.info(f"    {name}: {test_dist[label]:4d} ({pct:5.1f}%)")
        
        # Check shift
        max_shift = 0
        logger.info("\n  Distribution Shift:")
        for label in sorted(set(list(cv_dist.keys()) + list(test_dist.keys()))):
            cv_pct = cv_dist.get(label, 0) / len(y_cv) * 100
            test_pct = test_dist.get(label, 0) / len(y_test) * 100
            shift = abs(cv_pct - test_pct)
            max_shift = max(max_shift, shift)
            name = label_names.get(label, f'Class_{label}')
            logger.info(f"    {name}: {shift:5.1f}%")
        
        if max_shift > 15:
            logger.error(f"\n  🔴 CRITICAL: Severe shift ({max_shift:.1f}%) - Use stratified!")
        elif max_shift > 10:
            logger.warning(f"\n  ⚠️  WARNING: Large shift ({max_shift:.1f}%)")
        elif max_shift > 5:
            logger.warning(f"\n  ⚠️  CAUTION: Moderate shift ({max_shift:.1f}%)")
        else:
            logger.info(f"\n  ✅ Excellent balance! Shift: {max_shift:.1f}%")
        
        return X_cv, X_test, y_cv, y_test
    
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
        """
        Tune on REAL data with balanced_accuracy metric.
        NO SMOTETomek in tuning (correct ML practice).
        """
        logger.info("="*70)
        logger.info(f"🎯 TUNING {model_type.upper()} MODEL (Stratified + Balanced Accuracy)")  # FIX: .upper()
        logger.info("="*70)
        
        num_classes = len(np.unique(y))
        
        # 1. Calculate class weights from REAL data
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        logger.info(f"Class weights (for imbalanced data): {class_weights}")
        logger.info(f"Number of classes: {num_classes}")
        
        # 2. Stratified split (NO SMOTETomek - keep data REAL!)
        X_cv, X_test, y_cv, y_test = self.prepare_data_splits(
            X, y, 
            test_size=0.2, 
            strategy='stratified'
        )
        
        logger.info(f"\n📊 Data Split:")
        logger.info(f"   CV samples: {len(X_cv)} (80%)")
        logger.info(f"   Test samples: {len(X_test)} (20%)")
        
        # Log REAL distribution
        unique, counts = np.unique(y_cv, return_counts=True)
        logger.info(f"\n📊 CV Distribution (REAL data, no synthetic):")
        class_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
        for cls_id, count in zip(unique, counts):
            pct = count / len(y_cv) * 100
            logger.info(f"   {class_names[cls_id]:10s}: {count:,} ({pct:.1f}%)")
        
        # 3. Define balanced_accuracy scorer (NEW!)
        logger.info("\n🎯 Optimization Metric: balanced_accuracy_score")
        logger.info("   (Treats all classes equally, not biased to majority)")
        balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
        
        # 4. Create tuner
        validator = TimeSeriesValidator(n_splits=cv_splits)
        tuner = OptunaModelTuner(
            model_type=model_type,
            n_trials=n_trials,
            cv_splits=cv_splits,
            direction='maximize'
        )
        
        # 5. Model factory WITH class weights (for imbalanced data)
        def model_factory(params):
            if model_type == 'lstm':
                params['input_size'] = X.shape[1]
                params['num_classes'] = num_classes
                params['class_weights'] = class_weights  # Help model learn minority
                return self.create_lstm_model(params)
            else:
                raise ValueError(f"Unknown model: {model_type}")
        
        # 6. Tune on REAL data with balanced_accuracy (NEW!)
        logger.info("\n🔬 Starting hyperparameter optimization on REAL data...")
        logger.info("   (NO SMOTETomek - tuning on original distribution)")
        best_params, best_score, study = tuner.tune(
            X=X_cv, y=y_cv,  # REAL data (not synthetic!)
            model_factory=model_factory,
            metric_fn=balanced_accuracy_scorer  # NEW: Balanced metric
        )
        
        logger.info(f"\n✅ Best CV Balanced Accuracy: {best_score:.4f}")
        
        # 7. Validate on hold-out with BOTH metrics (ENHANCED!)
        logger.info("\n🔬 Validating on stratified hold-out test set...")
        final_model = model_factory(best_params)
        final_model.fit(X_cv, y_cv)  # Train on REAL CV data
        
        # Calculate both metrics
        holdout_score_total_acc = final_model.score(X_test, y_test)
        y_pred_test = final_model.predict(X_test)
        holdout_score_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
        
        gap = best_score - holdout_score_balanced_acc
        
        logger.info(f"   Hold-out Total Accuracy: {holdout_score_total_acc:.4f} (Yanıltıcı metrik)")  # FIX: Typo
        logger.info(f"   Hold-out Balanced Accuracy: {holdout_score_balanced_acc:.4f} (Asıl metrik)")
        logger.info(f"   Best CV Balanced Accuracy: {best_score:.4f}")
        logger.info(f"   CV-Holdout (Balanced) Gap: {gap:+.4f}")
        
        if abs(gap) < 0.05:
            logger.info("   ✅ Excellent generalization (gap < 5%)")
        elif abs(gap) < 0.10:
            logger.info("   ✅ Good generalization (gap < 10%)")
        else:
            logger.info("   ⚠️  Warning: Large gap suggests overfitting/underfitting")
        
        # 8. Save results with BOTH metrics (NEW!)
        # Calculate distribution shift
        max_shift = 0.0  # Stratified split has 0% shift
        
        results = {
            'model_type': model_type,
            'best_params': best_params,
            
            # Old metrics (for compatibility)
            'cv_score': float(best_score),  # Now balanced (not total)
            'holdout_score': float(holdout_score_total_acc),  # Total accuracy
            
            # NEW: Balanced metrics (PRIMARY)
            'balanced_cv_score': float(best_score),
            'balanced_holdout_score': float(holdout_score_balanced_acc),
            
            'gap': float(gap),  # Now uses balanced metric
            'n_trials': n_trials,
            'cv_splits': cv_splits,
            'num_classes': int(num_classes),
            'class_weights': class_weights.tolist(),
            'distribution_shift': float(max_shift),
            'split_strategy': 'stratified_class_weights',  # Updated name
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._save_results(results, model_type)
        logger.info("\n✅ Tuning complete! Results saved with balanced metrics.")
        
        return results
    
    def _convert_numpy_to_python(self, obj):
        """
        Recursively convert numpy types to Python native types for JSON serialization.
        
        Args:
            obj: Any object that may contain numpy types
            
        Returns:
            Object with all numpy types converted to Python native types
        """
        if isinstance(obj, np.ndarray):
            # Convert numpy arrays to lists
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            # Convert numpy integers to Python int
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            # Convert numpy floats to Python float
            return float(obj)
        elif isinstance(obj, np.bool_):
            # Convert numpy bool to Python bool
            return bool(obj)
        elif isinstance(obj, dict):
            # Recursively convert dictionary values
            return {key: self._convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively convert list/tuple elements
            converted = [self._convert_numpy_to_python(item) for item in obj]
            return converted if isinstance(obj, list) else tuple(converted)
        elif obj is None:
            # Handle None explicitly
            return None
        else:
            # Return as-is for Python native types (str, int, float, bool)
            return obj
    
    def _save_results(self, results: dict, model_type: str):
        """Save tuning results to JSON file."""
        output_dir = Path('logs/tuning_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{model_type}_tuning_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        
        # Convert all numpy types to Python native types recursively
        serializable_results = self._convert_numpy_to_python(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
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
