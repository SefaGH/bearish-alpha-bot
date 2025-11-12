"""
Optuna-based Hyperparameter Tuning with CV Validation
Integrates Optuna optimization with time series cross-validation.

Author: SefaGH & GitHub Copilot  
Date: 2025-11-08
"""

import optuna
import numpy as np
import logging
from typing import Dict, Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# Import with fallback
try:
    from scripts.utils.validation_framework import TimeSeriesValidator
except ImportError:
    try:
        from .validation_framework import TimeSeriesValidator
    except ImportError:
        from validation_framework import TimeSeriesValidator


class OptunaModelTuner:
    """
    Hyperparameter tuning with Optuna and CV validation.
    
    Features:
    - Bayesian optimization via Optuna
    - CV-validated objective function
    - Automatic best parameter selection
    - Pruning for faster convergence
    """
    
    def __init__(
        self,
        model_type: str,
        n_trials: int = 50,
        cv_splits: int = 5,
        timeout: int = None,
        direction: str = "maximize"
    ):
        """
        Initialize Optuna tuner.
        
        Args:
            model_type: Model identifier (lstm, rf, xgboost, etc.)
            n_trials: Number of optimization trials
            cv_splits: Number of CV folds
            timeout: Optional timeout in seconds
            direction: 'maximize' or 'minimize'
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        
        # Validator
        self.validator = TimeSeriesValidator(n_splits=cv_splits)
        
        # Storage for data (set during tune())
        self.X_cv = None
        self.y_cv = None
        self.model_factory = None
        self.metric_fn = None
        
        logger.info(f"OptunaModelTuner initialized: {model_type}, {n_trials} trials, {cv_splits} CV splits")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.
        Returns mean CV score for given hyperparameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Mean CV score (to be maximized/minimized)
        """
        # Suggest hyperparameters
        params = self.suggest_params(trial)
        
        # Create model factory with suggested params
        def factory():
            return self.create_model(params)
        
        # Cross-validate
        cv_results = self.validator.cross_validate(
            model_factory=factory,
            X=self.X_cv,
            y=self.y_cv,
            metric_fn=self.metric_fn
        )
        
        mean_score = cv_results['mean']
        std_score = cv_results['std']
        
        # Log trial results
        logger.info(f"Trial {trial.number}: {mean_score:.4f} ± {std_score:.4f} | Params: {params}")
        
        # Report intermediate value for pruning
        trial.report(mean_score, step=0)
        
        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_score
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on model type.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dict of suggested parameters
        """
        if self.model_type == 'lstm':
            return self._suggest_lstm_params(trial)
        elif self.model_type == 'gemma':
            return self._suggest_gemma_params(trial)
        elif self.model_type == 'transformer':
            return self._suggest_transformer_params(trial)
        elif self.model_type == 'rf':
            return self._suggest_rf_params(trial)
        elif self.model_type == 'xgboost':
            return self._suggest_xgboost_params(trial)
        elif self.model_type == 'rl_agent':
            return self._suggest_rl_params(trial)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _suggest_lstm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LSTM hyperparameters - ANTI-OVERFITTING VERSION."""
        return {
            # REDUCED COMPLEXITY
            'hidden_size': trial.suggest_categorical('hidden_size', [32, 64]),  # Was: [64, 96, 128, 160]
            'num_layers': trial.suggest_int('num_layers', 1, 2),  # Was: 1-3
            
            # INCREASED REGULARIZATION
            'dropout': trial.suggest_float('dropout', 0.5, 0.7, step=0.1),  # Was: 0.2-0.6
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),  # Same
            'weight_decay': trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True),  # Was: 1e-6 to 1e-3
            
            # TRAINING
            'batch_size': trial.suggest_categorical('batch_size', [32, 64])  # Same
        }
    
    def _suggest_gemma_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest GEMMA MLP hyperparameters."""
        return {
            'hidden_size': trial.suggest_int('hidden_size', 32, 128, log=True),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dropout': trial.suggest_uniform('dropout', 0.2, 0.6),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
        }
    
    def _suggest_transformer_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Transformer hyperparameters."""
        return {
            'nhead': trial.suggest_categorical('nhead', [4, 6, 8]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dim_feedforward': trial.suggest_categorical('dim_feedforward', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }
    
    def _suggest_rf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Random Forest hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
    
    def _suggest_xgboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest XGBoost hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
        }
    
    def _suggest_rl_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest RL agent hyperparameters."""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'hidden_sizes': trial.suggest_categorical('hidden_sizes', 
                [[64, 32], [128, 64], [256, 128]]),
            'target_update_freq': trial.suggest_int('target_update_freq', 10, 100, step=10),
            'gradient_clip_norm': trial.suggest_float('gradient_clip_norm', 0.5, 5.0, step=0.5),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999, step=0.01),
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.985, 0.995, step=0.002)
        }
    
    def create_model(self, params: Dict[str, Any]):
        """
        Create model instance with given parameters.
        Should be overridden or provided via model_factory.
        """
        if self.model_factory is None:
            raise NotImplementedError("model_factory must be set or create_model must be overridden")
        return self.model_factory(params)
    
    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        metric_fn: Callable = None
    ) -> Tuple[Dict[str, Any], float, optuna.Study]:
        """
        Run hyperparameter tuning.
        
        Args:
            X: Feature array (CV set)
            y: Target array (CV set)
            model_factory: Function (params) -> model
            metric_fn: Optional custom metric function
            
        Returns:
            (best_params, best_score, study)
        """
        # Store data and factories
        self.X_cv = X
        self.y_cv = y
        self.model_factory = model_factory
        self.metric_fn = metric_fn
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        )
        
        # Optimize
        logger.info(f"Starting Optuna optimization: {self.n_trials} trials")
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Results
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"✅ Optimization complete!")
        logger.info(f"   Best score: {best_score:.4f}")
        logger.info(f"   Best params: {best_params}")
        
        return best_params, best_score, study
