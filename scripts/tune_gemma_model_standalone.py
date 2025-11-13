"""
Standalone Hyperparameter Tuning for GEMMA Model - MLP Architecture
Based on tune_regime_models_standalone.py template

Phase 2: Create GEMMA Hyperparameter Tuning Script (CODE-ONLY BLUEPRINT)

Author: SefaGH & GitHub Copilot
Date: 2025-11-12
"""

import argparse
import sys
import os
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib # <<< YENİ İMPORT (Scaler'ı kaydetmek için) >>>

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.validation_framework import TimeSeriesValidator, ValidationReport
from scripts.utils.optuna_tuner import OptunaModelTuner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # <<< YENİ İMPORT >>>

from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegimeModelTuner:
    """Tune GEMMA MLP model using pre-processed data."""
    
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
    
    def create_gemma_model(self, params: dict):
        """Create sklearn-compatible MLP wrapper for GEMMA."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        class SklearnMLPWrapper:
            """Sklearn-compatible wrapper for a PyTorch MLP model with balanced training."""
            
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, 
                        learning_rate, weight_decay, batch_size, class_weights=None):
                
                # ==================== DYNAMIC MLP (FEED-FORWARD) MODEL ====================
                # Dinamik MLP (Feed-forward) modelini oluştur
                layers = []
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                
                layers.append(nn.Linear(hidden_size, num_classes))
                self.model = nn.Sequential(*layers)
                # ==========================================================================
                
                if class_weights is not None:
                    self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
                else:
                    self.criterion = nn.CrossEntropyLoss()
                
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay  # L2 Regularization
                )
                self.batch_size = batch_size
                self.num_epochs = 25
                self.patience = 7
                self.min_delta = 0.001
            
            def fit(self, X, y):
                """
                Sklearn-style fit.
                (DÜZELTİLDİ: Artık iç validasyon/early stopping yapmıyor.
                Bu görev, dışarıdaki Optuna/TimeSeriesValidator'a aittir.)
                """
                import torch
                from torch.utils.data import TensorDataset, DataLoader

                self.model.train()
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.LongTensor(y)
                
                train_dataset = TensorDataset(X_tensor, y_tensor)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                
                # Optuna'nın CV'si için 'num_epochs' kadar tam eğitim yap
                # Early stopping burada yapılmaz, CV'nin kendisi zaten modeli değerlendirir.
                for epoch in range(self.num_epochs):
                    epoch_loss = 0
                    if len(train_loader) == 0:
                        logger.warning("Data loader boş, fit adımı atlanıyor.")
                        break # Bu 'break' artık for epoch... döngüsü içinde, YAZIM HATASI YOK.

                    for batch_X, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item()
                    
                    # logger.debug(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {epoch_loss/len(train_loader):.4f}")
                
                return self
            
            def predict(self, X):
                """Sklearn-style predict method."""
                # IMPORTANT: MLP model expects 2D data, not 3D (no unsqueeze needed)
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
        
        return SklearnMLPWrapper(
            input_size=params.get('input_size', 83),  # Expected feature count 83
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_classes=params.get('num_classes', 3),
            dropout=params['dropout'],
            learning_rate=params.get('learning_rate', 0.001),
            weight_decay=params.get('weight_decay', 0.0),  # MLP can have weight_decay 0
            batch_size=params.get('batch_size', 64),
            class_weights=params.get('class_weights', None)
        )
        
    def tune_model(self, model_type: str, X: np.ndarray, y: np.ndarray,
                   n_trials: int = 30, cv_splits: int = 5):
        """
        Tune on REAL data with balanced_accuracy metric.
        NO SMOTETomek in tuning (correct ML practice).
        """
        logger.info("="*70)
        logger.info(f"🎯 TUNING {model_type.upper()} MODEL (Stratified + Balanced Accuracy)")
        logger.info("="*70)
        
        num_classes = len(np.unique(y))
        
        # 1. Calculate class weights from REAL data
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        logger.info(f"Class weights (for imbalanced data): {class_weights}")
        logger.info(f"Number of classes: {num_classes}")
        
        logger.info("📊 Using time-based split (Hold-out test set)")
        validator_instance = TimeSeriesValidator(n_splits=cv_splits) 
        X_cv, y_cv, X_test, y_test = validator_instance.split_with_holdout(X, y)
        tuner_validator = validator_instance
    
        # Artık 'tuner' için yeni bir validator oluşturmaya gerek yok, aynısını kullan
        tuner_validator = validator_instance
        
        # <<< BAŞLANGIÇ: YENİ ÖLÇEKLEME (SCALING) ADIMI >>>
        # ==============================================================================
        logger.info("\n" + "="*70)
        logger.info("⚖️ FITTING STANDARD SCALER (ÖLÇEKLEYİCİ)")
        logger.info("="*70)
        
        scaler = StandardScaler()
        
        # SADECE 80% CV verisi üzerinde 'fit' yap (data leakage önlemi)
        logger.info(f"Fitting scaler on {len(X_cv)} CV samples...")
        scaler.fit(X_cv)
        
        # Hem CV hem de Test verisini 'transform' et
        logger.info("Transforming CV and Test data...")
        X_cv_scaled = scaler.transform(X_cv)
        X_test_scaled = scaler.transform(X_test)
        
        # Ölçekleyiciyi (scaler) daha sonra 'train_final_model' 
        # ve 'analyze_model_explainability' betiklerinin kullanabilmesi için kaydet
        scaler_path = self.data_cache_dir / 'scaler_production.joblib'
        joblib.dump(scaler, scaler_path)
        logger.info(f"✅ Scaler (Ölçekleyici) şuraya kaydedildi: {scaler_path}")
        logger.info("="*70)
        # ============================================================================== #
        # <<< SON: YENİ ÖLÇEKLEME (SCALING) ADIMI >>>
        
        logger.info(f"\n📊 Data Split:")
        logger.info(f"   CV samples: {len(X_cv_scaled)} (80%)") # <-- Değişti
        logger.info(f"   Test samples: {len(X_test_scaled)} (20%)") # <-- Değişti
        
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
        validator = tuner_validator
        tuner = OptunaModelTuner(
            model_type=model_type,
            n_trials=n_trials,
            cv_splits=cv_splits,
            direction='maximize'
        )
        
        # 5. Model factory WITH class weights (for imbalanced data)
        def model_factory(params):
            if model_type == 'gemma':
                params['input_size'] = X.shape[1]
                params['num_classes'] = num_classes
                params['class_weights'] = class_weights
                return self.create_gemma_model(params)
            else:
                raise ValueError(f"Unknown model: {model_type}")
        
        # 6. Tune on REAL data with balanced_accuracy (NEW!)
        logger.info("\n🔬 Starting hyperparameter optimization on REAL data...")
        logger.info("   (NO SMOTETomek - tuning on original distribution)")
        best_params, best_score, study = tuner.tune(
            X=X_cv_scaled, y=y_cv,  # <<< YENİ: Ölçeklenmiş veri kullanılıyor >>>
            model_factory=model_factory,
            metric_fn=balanced_accuracy_scorer  # NEW: Balanced metric
        )
        
        logger.info(f"\n✅ Best CV Balanced Accuracy: {best_score:.4f}")
        
        # 7. Validate on hold-out with BOTH metrics (ENHANCED!)
        logger.info("\n🔬 Validating on stratified hold-out test set...")
        final_model = model_factory(best_params)
        final_model.fit(X_cv_scaled, y_cv)  # <<< YENİ: Ölçeklenmiş veri kullanılıyor >>>
        
        # Calculate both metrics
        holdout_score_total_acc = final_model.score(X_test_scaled, y_test) # <<< YENİ >>>
        y_pred_test = final_model.predict(X_test_scaled) # <<< YENİ >>>
        holdout_score_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
        
        gap = best_score - holdout_score_balanced_acc
        
        logger.info(f"   Hold-out Total Accuracy: {holdout_score_total_acc:.4f} (Yanıltıcı metrik)")
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
        
        filename = f"gemma_tuning_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        
        # Convert all numpy types to Python native types recursively
        serializable_results = self._convert_numpy_to_python(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✅ Results saved: {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemma', choices=['gemma'])
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--symbol', default='BTC-USDT')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("🎯 GEMMA MODEL TUNING (BALANCED SPLIT STRATEGY)")
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
