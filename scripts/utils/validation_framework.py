"""
Time Series Cross-Validation Framework
Provides robust validation utilities for trading ML models.

Author: SefaGH & GitHub Copilot
Date: 2025-11-08
"""

import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from typing import Dict, List, Tuple, Any, Callable

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """
    Time series cross-validation with statistical validation.
    
    Features:
    - TimeSeriesSplit for temporal data
    - Confidence interval calculation
    - Statistical significance testing
    - Performance comparison utilities
    """
    
    def __init__(self, n_splits: int = 5, test_size_ratio: float = 0.2):
        """
        Initialize validator.
        
        Args:
            n_splits: Number of CV folds
            test_size_ratio: Proportion for final hold-out test
        """
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        logger.info(f"TimeSeriesValidator initialized: {n_splits} splits, {test_size_ratio:.1%} hold-out")
    
    def split_with_holdout(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into CV set and hold-out test set.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            (X_cv, y_cv, X_test, y_test)
        """
        split_idx = int(len(X) * (1 - self.test_size_ratio))
        
        X_cv, X_test = X[:split_idx], X[split_idx:]
        y_cv, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Split: CV={len(X_cv)} ({len(X_cv)/len(X):.1%}), Hold-out={len(X_test)} ({len(X_test)/len(X):.1%})")
        return X_cv, y_cv, X_test, y_test
    
    def cross_validate(
        self, 
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn: Callable = None
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            model_factory: Function that returns a new model instance
            X: Features (CV set only)
            y: Targets (CV set only)
            metric_fn: Custom metric function (default: accuracy)
            
        Returns:
            Dict with 'scores', 'mean', 'std', 'ci_95', 'folds'
        """
        if metric_fn is None:
            metric_fn = lambda model, X, y: model.score(X, y)
        
        scores = []
        fold_details = []
        
        logger.info(f"Starting {self.n_splits}-fold time series CV...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.tscv.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fresh model
            model = model_factory()
            
            # Train
            model.fit(X_train, y_train)
            
            # Validate
            score = metric_fn(model, X_val, y_val)
            scores.append(score)
            
            fold_details.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score
            })
            
            logger.info(f"  Fold {fold_idx}/{self.n_splits}: Score={score:.4f} (train={len(train_idx)}, val={len(val_idx)})")
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_95 = self._calculate_ci_95(scores)
        
        results = {
            'scores': scores,
            'mean': mean_score,
            'std': std_score,
            'ci_95': ci_95,
            'folds': fold_details
        }
        
        logger.info(f"✅ CV Complete: {mean_score:.4f} ± {std_score:.4f} (95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}])")
        return results
    
    def _calculate_ci_95(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate 95% confidence interval."""
        mean = np.mean(scores)
        std = np.std(scores)
        n = len(scores)
        
        # Use t-distribution for small samples
        if n < 30:
            t_critical = stats.t.ppf(0.975, n - 1)
            margin = t_critical * (std / np.sqrt(n))
        else:
            # Use normal distribution for large samples
            z_critical = 1.96
            margin = z_critical * (std / np.sqrt(n))
        
        return (mean - margin, mean + margin)
    
    def compare_models(
        self,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """
        Statistical comparison of two models.
        
        Args:
            results_a: CV results from model A
            results_b: CV results from model B
            model_a_name: Name for model A
            model_b_name: Name for model B
            
        Returns:
            Dict with comparison statistics
        """
        scores_a = results_a['scores']
        scores_b = results_b['scores']
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Determine significance
        is_significant = p_value < 0.05
        
        comparison = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'mean_a': results_a['mean'],
            'mean_b': results_b['mean'],
            'difference': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'cohens_d': cohens_d,
            'interpretation': self._interpret_comparison(mean_diff, p_value, cohens_d)
        }
        
        logger.info(f"Model Comparison: {model_a_name} vs {model_b_name}")
        logger.info(f"  Difference: {mean_diff:+.4f} (p={p_value:.4f})")
        logger.info(f"  Result: {comparison['interpretation']}")
        
        return comparison
    
    def _interpret_comparison(
        self,
        mean_diff: float,
        p_value: float,
        cohens_d: float
    ) -> str:
        """Interpret comparison results."""
        if p_value >= 0.05:
            return "No significant difference"
        
        if abs(cohens_d) < 0.2:
            effect = "negligible"
        elif abs(cohens_d) < 0.5:
            effect = "small"
        elif abs(cohens_d) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        better_model = "A" if mean_diff > 0 else "B"
        return f"Model {better_model} significantly better ({effect} effect, d={cohens_d:.2f})"


class ValidationReport:
    """Generate comprehensive validation reports."""
    
    @staticmethod
    def generate_report(
        model_name: str,
        cv_results: Dict[str, Any],
        holdout_score: float = None
    ) -> str:
        """
        Generate formatted validation report.
        
        Args:
            model_name: Name of the model
            cv_results: Results from cross_validate()
            holdout_score: Optional hold-out test score
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append(f"VALIDATION REPORT: {model_name}")
        report.append("=" * 70)
        
        # CV Results
        report.append("\nCross-Validation Results:")
        report.append(f"  Number of folds: {len(cv_results['scores'])}")
        report.append(f"  Mean Score:      {cv_results['mean']:.4f}")
        report.append(f"  Std Deviation:   {cv_results['std']:.4f}")
        report.append(f"  95% CI:          [{cv_results['ci_95'][0]:.4f}, {cv_results['ci_95'][1]:.4f}]")
        
        # Per-fold scores
        report.append("\n  Per-Fold Scores:")
        for fold_info in cv_results['folds']:
            report.append(f"    Fold {fold_info['fold']}: {fold_info['score']:.4f}")
        
        # Hold-out test
        if holdout_score is not None:
            report.append("\nHold-out Test:")
            report.append(f"  Score: {holdout_score:.4f}")
            
            # Check if within CI
            in_ci = cv_results['ci_95'][0] <= holdout_score <= cv_results['ci_95'][1]
            status = "✅ Within CI" if in_ci else "⚠️ Outside CI"
            report.append(f"  Status: {status}")
        
        report.append("=" * 70)
        return "\n".join(report)
