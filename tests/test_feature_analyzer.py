"""
Unit tests for Feature Analysis Tool

Tests the FeatureAnalyzer class for variance analysis, correlation analysis,
feature selection, and report generation functionality.
"""

import json
import numpy as np
import pytest
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.analyze_features import FeatureAnalyzer


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data" / "cache"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_training_data(test_data_dir):
    """Create sample training data for testing."""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 20
    
    # Create features with varying characteristics
    features = np.zeros((n_samples, n_features))
    
    # Features 0-4: High variance, high correlation
    for i in range(5):
        features[:, i] = np.random.randn(n_samples) * 5 + np.linspace(-10, 10, n_samples)
    
    # Features 5-9: High variance, low correlation
    for i in range(5, 10):
        features[:, i] = np.random.randn(n_samples) * 5
    
    # Features 10-14: Low variance, any correlation
    for i in range(10, 15):
        features[:, i] = np.random.randn(n_samples) * 0.005 + 1.0
    
    # Features 15-19: Very low variance
    for i in range(15, 20):
        features[:, i] = np.random.randn(n_samples) * 0.001
    
    # Create labels correlated with first 5 features
    labels = np.zeros(n_samples)
    for i in range(5):
        labels += features[:, i] * (0.1 + i * 0.01)
    
    labels += np.random.randn(n_samples) * 2
    labels = np.digitize(labels, bins=[np.percentile(labels, 33), np.percentile(labels, 67)])
    
    # Save to file (using keys 'X' and 'y' to match production code)
    data_file = test_data_dir / "test_training_data.npz"
    feature_names = [f"feature_{i}" for i in range(n_features)]
    np.savez(data_file, X=features, y=labels, feature_names=feature_names)
    
    return data_file


class TestFeatureAnalyzer:
    """Test suite for FeatureAnalyzer class."""
    
    def test_initialization(self, sample_training_data):
        """Test analyzer initialization."""
        analyzer = FeatureAnalyzer(
            data_path=str(sample_training_data),
            variance_threshold=0.01,
            correlation_threshold=0.05
        )
        
        assert analyzer.data_path == Path(sample_training_data)
        assert analyzer.variance_threshold == 0.01
        assert analyzer.correlation_threshold == 0.05
        assert analyzer.features is None
        assert analyzer.labels is None
    
    def test_load_data_success(self, sample_training_data):
        """Test successful data loading."""
        analyzer = FeatureAnalyzer(data_path=str(sample_training_data))
        result = analyzer.load_data()
        
        assert result is True
        assert analyzer.features is not None
        assert analyzer.labels is not None
        assert analyzer.n_samples == 1000
        assert analyzer.n_features == 20
        assert len(analyzer.feature_names) == 20
    
    def test_load_data_file_not_found(self, tmp_path):
        """Test data loading with non-existent file."""
        analyzer = FeatureAnalyzer(data_path=str(tmp_path / "nonexistent.npz"))
        result = analyzer.load_data()
        
        assert result is False
        assert analyzer.features is None
    
    def test_analyze_variance(self, sample_training_data):
        """Test variance analysis."""
        analyzer = FeatureAnalyzer(data_path=str(sample_training_data))
        analyzer.load_data()
        
        result = analyzer.analyze_variance()
        
        assert 'statistics' in result
        assert 'threshold_counts' in result
        assert 'low_variance_count' in result
        
        # Check statistics keys
        stats = result['statistics']
        assert 'mean' in stats
        assert 'median' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'std' in stats
        
        # Variance should be calculated
        assert analyzer.variances is not None
        assert len(analyzer.variances) == analyzer.n_features
        
        # Check that low variance features are identified
        assert result['low_variance_count'] > 0
    
    def test_analyze_correlations(self, sample_training_data):
        """Test correlation analysis."""
        analyzer = FeatureAnalyzer(data_path=str(sample_training_data))
        analyzer.load_data()
        
        result = analyzer.analyze_correlations()
        
        assert 'statistics' in result
        assert 'threshold_counts' in result
        assert 'weak_count' in result
        assert 'strong_count' in result
        assert 'top_features' in result
        
        # Correlations should be calculated
        assert analyzer.correlations is not None
        assert analyzer.p_values is not None
        assert len(analyzer.correlations) == analyzer.n_features
        assert len(analyzer.p_values) == analyzer.n_features
        
        # Check top features
        assert len(result['top_features']) <= 15
        for feature in result['top_features']:
            assert 'name' in feature
            assert 'correlation' in feature
            assert 'p_value' in feature
    
    def test_select_features(self, sample_training_data):
        """Test feature selection."""
        analyzer = FeatureAnalyzer(
            data_path=str(sample_training_data),
            variance_threshold=0.01,
            correlation_threshold=0.05
        )
        analyzer.load_data()
        analyzer.analyze_variance()
        analyzer.analyze_correlations()
        
        result = analyzer.select_features()
        
        assert 'selected_count' in result
        assert 'rejected_count' in result
        assert 'selection_rate' in result
        assert 'rejection_breakdown' in result
        assert 'selected_features' in result
        
        # Feature mask should be created
        assert analyzer.feature_mask is not None
        assert len(analyzer.feature_mask) == analyzer.n_features
        
        # Selected + rejected should equal total
        assert result['selected_count'] + result['rejected_count'] == analyzer.n_features
        
        # Selection rate should be between 0 and 1
        assert 0 <= result['selection_rate'] <= 1
        
        # Should select high variance + high correlation features (first 5)
        # and reject low variance features
        assert result['selected_count'] > 0
        assert result['rejected_count'] > 0
    
    def test_select_features_without_analysis(self, sample_training_data):
        """Test feature selection without prior analysis."""
        analyzer = FeatureAnalyzer(data_path=str(sample_training_data))
        analyzer.load_data()
        
        result = analyzer.select_features()
        
        # Should return empty dict if analysis not performed
        assert result == {}
    
    def test_save_feature_mask(self, sample_training_data, tmp_path):
        """Test saving feature mask and metadata."""
        analyzer = FeatureAnalyzer(data_path=str(sample_training_data))
        analyzer.load_data()
        analyzer.analyze_variance()
        analyzer.analyze_correlations()
        analyzer.select_features()
        
        output_dir = tmp_path / "output"
        result = analyzer.save_feature_mask(output_dir=str(output_dir))
        
        assert result is True
        
        # Check mask file exists
        mask_file = output_dir / "feature_selection_mask.npy"
        assert mask_file.exists()
        
        # Load and verify mask
        loaded_mask = np.load(mask_file)
        assert len(loaded_mask) == analyzer.n_features
        assert np.array_equal(loaded_mask, analyzer.feature_mask)
        
        # Check metadata file exists
        metadata_file = output_dir / "feature_selection_metadata.json"
        assert metadata_file.exists()
        
        # Load and verify metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert 'n_samples' in metadata
        assert 'n_features_original' in metadata
        assert 'n_features_selected' in metadata
        assert 'variance_threshold' in metadata
        assert 'correlation_threshold' in metadata
        assert 'selected_features' in metadata
        assert 'rejected_features' in metadata
        assert 'feature_statistics' in metadata
        
        assert metadata['n_samples'] == analyzer.n_samples
        assert metadata['n_features_original'] == analyzer.n_features
    
    def test_save_feature_mask_without_selection(self, sample_training_data):
        """Test saving mask without feature selection."""
        analyzer = FeatureAnalyzer(data_path=str(sample_training_data))
        analyzer.load_data()
        
        result = analyzer.save_feature_mask()
        
        assert result is False
    
    def test_generate_report(self, sample_training_data, tmp_path):
        """Test report generation."""
        analyzer = FeatureAnalyzer(data_path=str(sample_training_data))
        analyzer.load_data()
        analyzer.analyze_variance()
        analyzer.analyze_correlations()
        analyzer.select_features()
        
        output_dir = tmp_path / "reports"
        result = analyzer.generate_report(output_dir=str(output_dir))
        
        assert result is True
        
        # Check report file exists
        report_file = output_dir / "feature_analysis_report.md"
        assert report_file.exists()
        
        # Read and verify report content
        with open(report_file) as f:
            content = f.read()
        
        # Check for key sections
        assert "# Feature Analysis Report" in content
        assert "## 1. Variance Analysis" in content
        assert "## 2. Correlation Analysis" in content
        assert "## 3. Feature Selection Results" in content
        assert "## 4. Recommendations" in content
        
        # Check for tables
        assert "| Metric | Value |" in content
        assert "| Threshold | Count | Percentage |" in content
        
        # Check for statistics
        assert "Variance Statistics" in content
        assert "Correlation Statistics" in content
        assert "Top 15 Features" in content
    
    def test_generate_report_without_analysis(self, sample_training_data):
        """Test report generation without prior analysis."""
        analyzer = FeatureAnalyzer(data_path=str(sample_training_data))
        analyzer.load_data()
        
        result = analyzer.generate_report()
        
        assert result is False
    
    def test_variance_thresholds(self, sample_training_data):
        """Test different variance thresholds."""
        analyzer = FeatureAnalyzer(
            data_path=str(sample_training_data),
            variance_threshold=0.1
        )
        analyzer.load_data()
        analyzer.analyze_variance()
        
        # Higher threshold should identify more low-variance features
        low_var_count = analyzer.analyze_variance()['low_variance_count']
        assert low_var_count > 0
    
    def test_correlation_thresholds(self, sample_training_data):
        """Test different correlation thresholds."""
        analyzer = FeatureAnalyzer(
            data_path=str(sample_training_data),
            correlation_threshold=0.1
        )
        analyzer.load_data()
        analyzer.analyze_correlations()
        
        # Higher threshold should identify more weak features
        weak_count = analyzer.analyze_correlations()['weak_count']
        assert weak_count > 0
    
    def test_full_workflow(self, sample_training_data, tmp_path):
        """Test complete analysis workflow."""
        analyzer = FeatureAnalyzer(
            data_path=str(sample_training_data),
            variance_threshold=0.01,
            correlation_threshold=0.05
        )
        
        # Load data
        assert analyzer.load_data() is True
        
        # Run analysis
        var_result = analyzer.analyze_variance()
        assert var_result is not None
        
        corr_result = analyzer.analyze_correlations()
        assert corr_result is not None
        
        # Select features
        select_result = analyzer.select_features()
        assert select_result['selected_count'] > 0
        
        # Save outputs
        output_dir = tmp_path / "output"
        assert analyzer.save_feature_mask(output_dir=str(output_dir)) is True
        
        report_dir = tmp_path / "reports"
        assert analyzer.generate_report(output_dir=str(report_dir)) is True
        
        # Verify all outputs exist
        assert (output_dir / "feature_selection_mask.npy").exists()
        assert (output_dir / "feature_selection_metadata.json").exists()
        assert (report_dir / "feature_analysis_report.md").exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
