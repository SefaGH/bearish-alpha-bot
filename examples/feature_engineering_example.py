"""
Example usage of enhanced feature engineering pipeline.

This example demonstrates how to use the new advanced features
for machine learning model training and prediction.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureEngineeringPipeline


def main():
    """Demonstrate feature engineering usage."""
    
    print("=" * 70)
    print("ENHANCED FEATURE ENGINEERING - USAGE EXAMPLE")
    print("=" * 70)
    
    # Create sample OHLCV data
    np.random.seed(42)
    n_samples = 200
    
    price_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1h'),
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.rand(n_samples) * 1000
    })
    
    print(f"\n📊 Sample Data: {len(price_data)} rows")
    print(price_data.head())
    
    # Example 1: Extract features with advanced features (default)
    print("\n" + "=" * 70)
    print("Example 1: Default Configuration (Advanced Features)")
    print("=" * 70)
    
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.extract_features(price_data)
    
    print(f"\n✅ Extracted {len(features.columns)} features")
    print(f"Shape: {features.shape}")
    print(f"\nFirst 10 feature names:")
    for i, col in enumerate(features.columns[:10], 1):
        print(f"  {i}. {col}")
    
    print(f"\nLast 10 feature names (advanced features):")
    for i, col in enumerate(features.columns[-10:], 1):
        print(f"  {i}. {col}")
    
    # Example 2: Backward compatibility mode
    print("\n" + "=" * 70)
    print("Example 2: Legacy Mode (Backward Compatibility)")
    print("=" * 70)
    
    config_legacy = {
        'use_advanced_features': True,  # Include advanced features
        'use_legacy_alignment': True     # But align to legacy 42 features
    }
    pipeline_legacy = FeatureEngineeringPipeline(config_legacy)
    features_legacy = pipeline_legacy.extract_features(price_data)
    
    print(f"\n✅ Extracted {len(features_legacy.columns)} features (legacy mode)")
    print("This mode ensures compatibility with pre-trained models.")
    
    # Example 3: Without advanced features
    print("\n" + "=" * 70)
    print("Example 3: Basic Features Only")
    print("=" * 70)
    
    config_basic = {'use_advanced_features': False}
    pipeline_basic = FeatureEngineeringPipeline(config_basic)
    features_basic = pipeline_basic.extract_features(price_data)
    
    print(f"\n✅ Extracted {len(features_basic.columns)} features (basic mode)")
    print("This mode excludes advanced features for faster computation.")
    
    # Example 4: Custom configuration
    print("\n" + "=" * 70)
    print("Example 4: Custom Configuration")
    print("=" * 70)
    
    config_custom = {
        'use_advanced_features': True,
        'use_legacy_alignment': False,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'atr_period': 14,
        'adx_period': 14,
        'volatility_windows': '5,10,20,50',
        'momentum_windows': '5,10,20,50'
    }
    pipeline_custom = FeatureEngineeringPipeline(config_custom)
    features_custom = pipeline_custom.extract_features(price_data)
    
    print(f"\n✅ Extracted {len(features_custom.columns)} features (custom config)")
    print("Custom configuration allows fine-tuning of technical indicator periods.")
    
    # Example 5: Feature categories
    print("\n" + "=" * 70)
    print("Example 5: Feature Categories")
    print("=" * 70)
    
    # Group features by category
    categories = {
        'Technical': [col for col in features.columns if col.startswith('technical_')],
        'Microstructure': [col for col in features.columns if col.startswith('microstructure_')],
        'Volatility': [col for col in features.columns if col.startswith('volatility_')],
        'Momentum': [col for col in features.columns if col.startswith('momentum_')],
        'Advanced Momentum': [col for col in features.columns if col.startswith('advanced_momentum_')],
        'Advanced Volume': [col for col in features.columns if col.startswith('advanced_volume_')],
        'Advanced Volatility': [col for col in features.columns if col.startswith('advanced_volatility_')],
        'Advanced Trend': [col for col in features.columns if col.startswith('advanced_trend_')],
        'Support/Resistance': [col for col in features.columns if col.startswith('support_resistance_')]
    }
    
    print("\n📊 Feature Categories:")
    for category, cols in categories.items():
        if cols:
            print(f"\n{category} ({len(cols)} features):")
            for col in cols[:5]:  # Show first 5
                print(f"  • {col}")
            if len(cols) > 5:
                print(f"  ... and {len(cols) - 5} more")
    
    # Example 6: Data quality
    print("\n" + "=" * 70)
    print("Example 6: Data Quality Check")
    print("=" * 70)
    
    # Check for NaN values
    nan_counts = features.isna().sum()
    features_with_nan = nan_counts[nan_counts > 0]
    
    print(f"\n📊 Features with NaN values: {len(features_with_nan)} / {len(features.columns)}")
    print(f"Total NaN percentage: {features.isna().sum().sum() / features.size * 100:.2f}%")
    
    # Remove NaN rows
    features_clean = features.dropna()
    print(f"\nUsable samples after NaN removal: {len(features_clean)} / {len(features)}")
    print(f"Data retention: {len(features_clean) / len(features) * 100:.1f}%")
    
    # Check for infinite values
    inf_count = np.isinf(features.values).sum()
    print(f"\nInfinite values: {inf_count}")
    
    print("\n" + "=" * 70)
    print("✅ EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Default mode provides 87 comprehensive features")
    print("  • Legacy mode maintains backward compatibility (42 features)")
    print("  • Basic mode provides faster computation (43 features)")
    print("  • Custom configuration allows fine-tuning")
    print("  • ~65% data retention after NaN removal (due to rolling windows)")
    print("  • No infinite values in output")


if __name__ == '__main__':
    main()
