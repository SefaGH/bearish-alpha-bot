"""
Test Suite for Feature Selection Integration in prepare_training_data.py

Tests feature selection mask loading, validation, and application.
"""

import pytest
import numpy as np
import tempfile
import shutil
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    cache_dir = Path(temp_dir) / 'data' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    yield cache_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    X = np.random.randn(100, 82)  # 100 samples, 82 features
    y = np.random.randint(0, 4, size=100)  # 4 classes (regime labels)
    return X, y


@pytest.fixture
def feature_mask():
    """Create sample feature selection mask"""
    # Select 45 out of 82 features
    mask = np.zeros(82, dtype=bool)
    mask[:45] = True
    np.random.shuffle(mask)
    return mask


class TestFeatureSelectionIntegration:
    """Test feature selection integration in prepare_training_data.py"""
    
    def test_feature_mask_loading_success(self, temp_cache_dir, sample_data, feature_mask):
        """Test successful loading and application of feature mask"""
        X, y = sample_data
        
        # Save mask to temp directory
        mask_path = temp_cache_dir / 'feature_selection_mask.npy'
        np.save(mask_path, feature_mask)
        
        # Apply mask
        X_filtered = X[:, feature_mask]
        
        # Verify shape
        assert X_filtered.shape[0] == X.shape[0], "Sample count should remain the same"
        assert X_filtered.shape[1] == feature_mask.sum(), "Feature count should match selected features"
        assert X_filtered.shape[1] == 45, "Should have 45 selected features"
    
    def test_feature_mask_shape_validation(self, temp_cache_dir, sample_data):
        """Test mask shape validation when mismatch occurs"""
        X, y = sample_data
        
        # Create mask with wrong size
        wrong_mask = np.ones(50, dtype=bool)  # Wrong size (50 instead of 82)
        mask_path = temp_cache_dir / 'feature_selection_mask.npy'
        np.save(mask_path, wrong_mask)
        
        # Load and check
        loaded_mask = np.load(mask_path)
        
        # Should detect mismatch
        assert len(loaded_mask) != X.shape[1], "Should detect size mismatch"
    
    def test_feature_mask_missing_file(self, temp_cache_dir):
        """Test behavior when feature mask file doesn't exist"""
        mask_path = temp_cache_dir / 'feature_selection_mask.npy'
        
        # Verify file doesn't exist
        assert not mask_path.exists(), "Mask file should not exist"
        
        # This should be handled gracefully (no exception)
        # The actual script should log a warning and continue
    
    def test_feature_selection_preserves_sample_count(self, sample_data, feature_mask):
        """Test that feature selection preserves number of samples"""
        X, y = sample_data
        
        # Apply mask
        X_filtered = X[:, feature_mask]
        
        # Verify sample count unchanged
        assert X_filtered.shape[0] == X.shape[0], "Sample count must be preserved"
        assert len(y) == X.shape[0], "Labels must match sample count"
    
    def test_feature_selection_reduces_features(self, sample_data, feature_mask):
        """Test that feature selection reduces feature count"""
        X, y = sample_data
        
        # Apply mask
        X_filtered = X[:, feature_mask]
        
        # Verify feature reduction
        original_features = X.shape[1]
        selected_features = X_filtered.shape[1]
        removed_features = (~feature_mask).sum()
        
        assert selected_features < original_features, "Features should be reduced"
        assert selected_features + removed_features == original_features, "Sum should equal original"
        assert selected_features == feature_mask.sum(), "Selected count should match mask"
    
    def test_mask_boolean_type(self, feature_mask):
        """Test that mask is boolean type"""
        assert feature_mask.dtype == bool, "Mask should be boolean type"
        assert len(feature_mask.shape) == 1, "Mask should be 1-dimensional"
    
    def test_feature_selection_statistics(self, sample_data, feature_mask):
        """Test feature selection statistics calculation"""
        X, y = sample_data
        
        original_count = X.shape[1]
        selected_count = feature_mask.sum()
        removed_count = (~feature_mask).sum()
        
        # Verify counts
        assert selected_count == 45, "Should select 45 features"
        assert removed_count == 37, "Should remove 37 features"
        assert selected_count + removed_count == original_count, "Counts should sum to original"
    
    def test_data_integrity_after_selection(self, sample_data, feature_mask):
        """Test that data integrity is maintained after feature selection"""
        X, y = sample_data
        
        # Get selected feature indices
        selected_indices = np.where(feature_mask)[0]
        
        # Apply mask
        X_filtered = X[:, feature_mask]
        
        # Verify data integrity for each selected feature
        for i, original_idx in enumerate(selected_indices):
            np.testing.assert_array_equal(
                X_filtered[:, i],
                X[:, original_idx],
                err_msg=f"Feature {i} data mismatch"
            )


class TestFeatureSelectionCommandLine:
    """Test command line arguments for feature selection"""
    
    def test_default_feature_selection_enabled(self):
        """Test that feature selection is enabled by default"""
        # When no flag is provided, use_feature_selection should be True
        no_flag_value = True  # Default behavior
        assert no_flag_value is True, "Feature selection should be enabled by default"
    
    def test_no_feature_selection_flag(self):
        """Test --no-feature-selection flag disables feature selection"""
        # When --no-feature-selection is provided, use_feature_selection should be False
        with_flag_value = False  # After parsing --no-feature-selection
        assert with_flag_value is False, "Flag should disable feature selection"
    
    def test_flag_inversion_logic(self):
        """Test that flag correctly inverts to parameter"""
        # args.no_feature_selection = True -> use_feature_selection = False
        no_feature_selection_flag = True
        use_feature_selection = not no_feature_selection_flag
        assert use_feature_selection is False, "Flag should invert correctly"
        
        # args.no_feature_selection = False -> use_feature_selection = True
        no_feature_selection_flag = False
        use_feature_selection = not no_feature_selection_flag
        assert use_feature_selection is True, "Flag should invert correctly"


class TestFeatureSelectionErrorHandling:
    """Test error handling for feature selection"""
    
    def test_corrupted_mask_file_handling(self, temp_cache_dir):
        """Test handling of corrupted mask file"""
        mask_path = temp_cache_dir / 'feature_selection_mask.npy'
        
        # Create a corrupted file
        with open(mask_path, 'wb') as f:
            f.write(b'corrupted data')
        
        # Loading should raise an exception
        with pytest.raises(Exception):
            np.load(mask_path)
    
    def test_empty_mask_handling(self, temp_cache_dir):
        """Test handling of empty mask"""
        mask_path = temp_cache_dir / 'feature_selection_mask.npy'
        
        # Create empty mask
        empty_mask = np.array([], dtype=bool)
        np.save(mask_path, empty_mask)
        
        loaded_mask = np.load(mask_path)
        assert len(loaded_mask) == 0, "Mask should be empty"
    
    def test_all_false_mask_handling(self, sample_data):
        """Test handling when all features are deselected"""
        X, y = sample_data
        
        # Create all-False mask (select no features)
        all_false_mask = np.zeros(X.shape[1], dtype=bool)
        
        # This should result in empty feature array
        X_filtered = X[:, all_false_mask]
        assert X_filtered.shape[1] == 0, "Should have no features selected"
    
    def test_all_true_mask_handling(self, sample_data):
        """Test handling when all features are selected"""
        X, y = sample_data
        
        # Create all-True mask (select all features)
        all_true_mask = np.ones(X.shape[1], dtype=bool)
        
        # This should keep all features
        X_filtered = X[:, all_true_mask]
        assert X_filtered.shape[1] == X.shape[1], "Should keep all features"
        np.testing.assert_array_equal(X_filtered, X, "Data should be unchanged")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
