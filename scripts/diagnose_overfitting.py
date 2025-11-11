"""
Diagnose overfitting issues in LSTM tuning.
Analyzes data quality, distribution shifts, and feature problems.

Usage:
    python scripts/diagnose_overfitting.py

Author: SefaGH
Date: 2025-11-08
"""

import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def diagnose_data():
    """Comprehensive overfitting diagnosis."""
    
    logger.info("="*70)
    logger.info("🔍 OVERFITTING DIAGNOSIS")
    logger.info("="*70)
    
    # Load cached data
    cache_file = Path('data/cache/BTC-USDT_training_data.npz')
    
    if not cache_file.exists():
        logger.error(f"❌ Cache file not found: {cache_file}")
        return
    
    data = np.load(cache_file)
    X = data['X']
    y = data['y']
    
    logger.info(f"\n📊 DATA OVERVIEW")
    logger.info(f"   Total samples: {len(X)}")
    logger.info(f"   Features: {X.shape[1]}")
    logger.info(f"   Classes: {len(np.unique(y))}")
    
    # 1. Label Distribution Analysis
    logger.info(f"\n📊 LABEL DISTRIBUTION")
    unique, counts = np.unique(y, return_counts=True)
    label_names = ['Bullish', 'Bearish', 'Neutral', 'Volatile']
    
    issues = []
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        logger.info(f"   {label_names[label]}: {count:4d} ({percentage:5.1f}%)")
        
        if percentage > 60:
            issues.append(f"Class {label_names[label]} dominates ({percentage:.1f}%)")
        if percentage < 10:
            issues.append(f"Class {label_names[label]} is rare ({percentage:.1f}%)")
    
    # 2. Feature Quality Check
    logger.info(f"\n📊 FEATURE QUALITY")
    
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    
    if nan_count > 0:
        logger.error(f"   ❌ NaN values: {nan_count}")
        issues.append(f"Found {nan_count} NaN values")
    else:
        logger.info(f"   ✅ No NaN values")
    
    if inf_count > 0:
        logger.error(f"   ❌ Inf values: {inf_count}")
        issues.append(f"Found {inf_count} Inf values")
    else:
        logger.info(f"   ✅ No Inf values")
    
    # Feature variance
    feature_vars = np.var(X, axis=0)
    zero_var_count = (feature_vars == 0).sum()
    low_var_count = (feature_vars < 0.01).sum()
    
    if zero_var_count > 0:
        logger.warning(f"   ⚠️  Zero variance features: {zero_var_count}")
        issues.append(f"{zero_var_count} features have zero variance")
    
    if low_var_count > 0:
        logger.warning(f"   ⚠️  Low variance features: {low_var_count}")
    
    logger.info(f"   Mean variance: {np.mean(feature_vars):.6f}")
    logger.info(f"   Min variance:  {np.min(feature_vars):.6f}")
    logger.info(f"   Max variance:  {np.max(feature_vars):.6f}")
    
    # 3. Train/Test Split Analysis
    logger.info(f"\n📊 TRAIN/TEST SPLIT ANALYSIS")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"   Train: {len(X_train):4d} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"   Test:  {len(X_test):4d} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    logger.info(f"\n   Train Distribution:")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    for label, count in zip(train_unique, train_counts):
        percentage = (count / len(y_train)) * 100
        logger.info(f"      {label_names[label]}: {count:4d} ({percentage:5.1f}%)")
    
    logger.info(f"\n   Test Distribution:")
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    for label, count in zip(test_unique, test_counts):
        percentage = (count / len(y_test)) * 100
        logger.info(f"      {label_names[label]}: {count:4d} ({percentage:5.1f}%)")
    
    # 4. Distribution Shift Detection
    logger.info(f"\n📊 DISTRIBUTION SHIFT")
    
    max_shift = 0
    for label in range(4):
        train_pct = (train_counts[label] / len(y_train) * 100) if label < len(train_counts) else 0
        test_pct = (test_counts[label] / len(y_test) * 100) if label < len(test_counts) else 0
        shift = abs(train_pct - test_pct)
        max_shift = max(max_shift, shift)
        
        status = "⚠️" if shift > 10 else "✅"
        logger.info(f"   {status} {label_names[label]}: {shift:5.1f}% shift")
        
        if shift > 10:
            issues.append(f"{label_names[label]} has {shift:.1f}% distribution shift")
    
    # 5. Feature-Label Correlation
    logger.info(f"\n📊 FEATURE PREDICTIVE POWER")
    
    from scipy.stats import spearmanr
    
    correlations = []
    for i in range(min(X.shape[1], 100)):  # Check first 100 features
        try:
            corr, _ = spearmanr(X[:, i], y)
            correlations.append(abs(corr))
        except:
            correlations.append(0)
    
    correlations = np.array(correlations)
    
    logger.info(f"   Mean |correlation|: {np.mean(correlations):.4f}")
    logger.info(f"   Max |correlation|:  {np.max(correlations):.4f}")
    logger.info(f"   Strong features (>0.3): {(correlations > 0.3).sum()}")
    
    if np.max(correlations) < 0.1:
        logger.error(f"   ❌ Very weak predictive power!")
        issues.append("Features have very weak correlation with labels")
    
    # 6. Summary and Recommendations
    logger.info(f"\n" + "="*70)
    logger.info(f"📋 DIAGNOSIS SUMMARY")
    logger.info(f"="*70)
    
    if not issues:
        logger.info("✅ No major issues detected")
    else:
        logger.warning(f"⚠️  Found {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            logger.warning(f"   {i}. {issue}")
    
    logger.info(f"\n💡 RECOMMENDATIONS:")
    
    # Prioritized recommendations
    if max(counts) / len(y) > 0.6:
        logger.info(f"   1. 🔴 HIGH PRIORITY: Address class imbalance")
        logger.info(f"      - Use class_weight='balanced' in models")
        logger.info(f"      - Consider SMOTE or oversampling")
    
    if max_shift > 10:
        logger.info(f"   2. 🟡 MEDIUM: Address train/test distribution shift")
        logger.info(f"      - Use stratified split")
        logger.info(f"      - Consider different test period")
    
    if zero_var_count > 0 or low_var_count > 5:
        logger.info(f"   3. 🟡 MEDIUM: Remove low-variance features")
    
    if np.max(correlations) < 0.15:
        logger.info(f"   4. 🟢 LOW: Improve feature engineering")
    
    logger.info(f"\n   5. 🔴 HIGH PRIORITY: Reduce model complexity")
    logger.info(f"      - Use smaller hidden_size (32-64)")
    logger.info(f"      - Increase dropout (0.5-0.7)")
    logger.info(f"      - Add early stopping")
    logger.info(f"      - Increase weight_decay")
    
    logger.info("="*70)
    
    return issues


if __name__ == '__main__':
    diagnose_data()
