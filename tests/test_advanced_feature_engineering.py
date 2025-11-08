"""
Tests for advanced feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np
from src.ml.feature_engineering import (
    FeatureEngineeringPipeline,
    AdvancedMomentumFeatures,
    AdvancedVolumeFeatures,
    AdvancedVolatilityFeatures,
    AdvancedTrendFeatures,
    SupportResistanceFeatures
)


class TestAdvancedFeatureEngineering:
    """Test advanced feature engineering methods."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1h'),
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 102,
            'low': np.random.randn(n_samples).cumsum() + 98,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.rand(n_samples) * 1000
        })
    
    def test_advanced_momentum_features(self, sample_ohlcv_data):
        """Test advanced momentum feature extraction."""
        extractor = AdvancedMomentumFeatures()
        features = extractor.compute(sample_ohlcv_data)
        
        # Check that features are extracted
        assert not features.empty
        assert len(features) == len(sample_ohlcv_data)
        
        # Check for expected feature columns
        expected_columns = [
            'momentum_3', 'momentum_5', 'momentum_10', 'momentum_14', 
            'momentum_20', 'momentum_30', 'momentum_acceleration',
            'cumulative_momentum_10', 'cumulative_momentum_20'
        ]
        
        for col in expected_columns:
            assert col in features.columns, f"Missing feature: {col}"
    
    def test_advanced_volume_features(self, sample_ohlcv_data):
        """Test advanced volume feature extraction."""
        extractor = AdvancedVolumeFeatures()
        features = extractor.compute(sample_ohlcv_data)
        
        # Check that features are extracted
        assert not features.empty
        assert len(features) == len(sample_ohlcv_data)
        
        # Check for expected feature columns
        expected_columns = [
            'volume_momentum_5', 'volume_momentum_10',
            'volume_ma_ratio_5', 'volume_ma_ratio_20',
            'vwap', 'distance_from_vwap',
            'obv', 'obv_momentum'
        ]
        
        for col in expected_columns:
            assert col in features.columns, f"Missing feature: {col}"
    
    def test_advanced_volume_features_without_volume(self):
        """Test advanced volume features handle missing volume gracefully."""
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
        })
        
        extractor = AdvancedVolumeFeatures()
        features = extractor.compute(data)
        
        # Should return features filled with NaN
        assert not features.empty
        assert len(features) == len(data)
    
    def test_advanced_volatility_features(self, sample_ohlcv_data):
        """Test advanced volatility feature extraction."""
        extractor = AdvancedVolatilityFeatures()
        features = extractor.compute(sample_ohlcv_data)
        
        # Check that features are extracted
        assert not features.empty
        assert len(features) == len(sample_ohlcv_data)
        
        # Check for expected feature columns
        expected_columns = [
            'atr_ratio', 'atr_momentum',
            'bb_width_normalized', 'bb_width_momentum', 'bb_position_advanced',
            'hist_volatility_5', 'hist_volatility_10', 'hist_volatility_20',
            'volatility_ratio'
        ]
        
        for col in expected_columns:
            assert col in features.columns, f"Missing feature: {col}"
    
    def test_advanced_trend_features(self, sample_ohlcv_data):
        """Test advanced trend feature extraction."""
        extractor = AdvancedTrendFeatures()
        features = extractor.compute(sample_ohlcv_data)
        
        # Check that features are extracted
        assert not features.empty
        assert len(features) == len(sample_ohlcv_data)
        
        # Check for ADX-related features
        adx_columns = ['adx', 'adx_strong_trend', 'adx_momentum']
        for col in adx_columns:
            assert col in features.columns, f"Missing ADX feature: {col}"
        
        # Check for directional indicator features
        di_columns = ['plus_di', 'minus_di', 'di_difference', 'di_ratio']
        for col in di_columns:
            assert col in features.columns, f"Missing DI feature: {col}"
        
        # Check for MA features
        ma_columns = ['ma_distance_ratio_10_20', 'ma_distance_ratio_20_50', 'trend_consistency']
        for col in ma_columns:
            assert col in features.columns, f"Missing MA feature: {col}"
    
    def test_support_resistance_features(self, sample_ohlcv_data):
        """Test support/resistance feature extraction."""
        extractor = SupportResistanceFeatures()
        features = extractor.compute(sample_ohlcv_data)
        
        # Check that features are extracted
        assert not features.empty
        assert len(features) == len(sample_ohlcv_data)
        
        # Check for distance from highs features
        for period in [10, 20, 50]:
            assert f'distance_from_high_{period}' in features.columns
        
        # Check for distance from lows features
        for period in [10, 20, 50]:
            assert f'distance_from_low_{period}' in features.columns
        
        # Check for range position features
        for period in [20, 50]:
            assert f'range_position_{period}' in features.columns
    
    def test_pipeline_with_advanced_features(self, sample_ohlcv_data):
        """Test full pipeline with advanced features enabled."""
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_features(sample_ohlcv_data)
        
        # Check that we have significantly more features than the baseline
        assert len(features.columns) > 70, f"Expected >70 features, got {len(features.columns)}"
        
        # Check that advanced features are included
        advanced_prefixes = [
            'advanced_momentum_',
            'advanced_volume_',
            'advanced_volatility_',
            'advanced_trend_',
            'support_resistance_'
        ]
        
        for prefix in advanced_prefixes:
            matching_cols = [col for col in features.columns if col.startswith(prefix)]
            assert len(matching_cols) > 0, f"No features found with prefix: {prefix}"
    
    def test_pipeline_without_advanced_features(self, sample_ohlcv_data):
        """Test pipeline with advanced features disabled."""
        config = {'use_advanced_features': False}
        pipeline = FeatureEngineeringPipeline(config)
        features = pipeline.extract_features(sample_ohlcv_data)
        
        # Check that we have only basic features
        assert len(features.columns) < 50, f"Expected <50 features, got {len(features.columns)}"
        
        # Check that advanced features are NOT included
        advanced_prefixes = [
            'advanced_momentum_',
            'advanced_volume_',
            'advanced_volatility_',
            'advanced_trend_',
            'support_resistance_'
        ]
        
        for prefix in advanced_prefixes:
            matching_cols = [col for col in features.columns if col.startswith(prefix)]
            assert len(matching_cols) == 0, f"Found unexpected features with prefix: {prefix}"
    
    def test_pipeline_with_legacy_alignment(self, sample_ohlcv_data):
        """Test pipeline with legacy alignment for backward compatibility."""
        config = {'use_legacy_alignment': True}
        pipeline = FeatureEngineeringPipeline(config)
        features = pipeline.extract_features(sample_ohlcv_data)
        
        # Check that we have exactly the legacy feature count
        assert len(features.columns) == 42, f"Expected 42 features with legacy alignment, got {len(features.columns)}"
    
    def test_feature_extraction_handles_nan_gracefully(self):
        """Test that feature extraction handles NaN values gracefully."""
        # Create data with NaN values
        data = pd.DataFrame({
            'open': [100, 101, np.nan, 103, 104],
            'high': [102, 103, 104, np.nan, 106],
            'low': [98, 99, 100, 101, np.nan],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, np.nan, 1400]
        })
        
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_features(data)
        
        # Should not raise an error and should return features
        assert not features.empty
        assert len(features) == len(data)
    
    def test_feature_extraction_with_minimal_data(self):
        """Test feature extraction with minimal data."""
        # Create minimal data (20 rows)
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.randn(20).cumsum() + 100,
            'high': np.random.randn(20).cumsum() + 102,
            'low': np.random.randn(20).cumsum() + 98,
            'close': np.random.randn(20).cumsum() + 100,
            'volume': np.random.rand(20) * 1000
        })
        
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_features(data)
        
        # Should return features even with minimal data
        assert not features.empty
        assert len(features) == len(data)
    
    def test_feature_names_have_correct_prefixes(self, sample_ohlcv_data):
        """Test that all advanced features have correct prefixes."""
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_features(sample_ohlcv_data)
        
        # Define valid prefixes
        valid_prefixes = [
            'technical_', 'microstructure_', 'volatility_', 'momentum_',
            'advanced_momentum_', 'advanced_volume_', 'advanced_volatility_',
            'advanced_trend_', 'support_resistance_'
        ]
        
        # Check that all columns have a valid prefix
        for col in features.columns:
            has_valid_prefix = any(col.startswith(prefix) for prefix in valid_prefixes)
            assert has_valid_prefix, f"Column '{col}' does not have a valid prefix"
    
    def test_obv_calculation(self, sample_ohlcv_data):
        """Test OBV (On-Balance Volume) calculation."""
        extractor = AdvancedVolumeFeatures()
        features = extractor.compute(sample_ohlcv_data)
        
        # OBV should be present
        assert 'obv' in features.columns
        
        # OBV should not have all NaN values (unless volume is all NaN)
        assert not features['obv'].isna().all()
        
        # OBV momentum should be present
        assert 'obv_momentum' in features.columns


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering pipeline."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n_samples = 200
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1h'),
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 102,
            'low': np.random.randn(n_samples).cumsum() + 98,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.rand(n_samples) * 1000
        })
    
    def test_end_to_end_feature_extraction(self, sample_ohlcv_data):
        """Test complete end-to-end feature extraction."""
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_features(sample_ohlcv_data)
        
        # Verify output
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert len(features) == len(sample_ohlcv_data)
        
        # Verify we have the expected number of features
        assert len(features.columns) > 80, f"Expected >80 features, got {len(features.columns)}"
        
        # Verify no infinite values
        assert not np.isinf(features.values).any(), "Features contain infinite values"
        
        # Verify index is preserved
        assert features.index.equals(sample_ohlcv_data.index)
