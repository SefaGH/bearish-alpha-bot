"""
Tests for data augmentation module (FAZ 3.2 + 2.2).
"""

import pytest
import numpy as np
from src.ml.data_augmentation import DataAugmentation


class TestDataAugmentation:
    """Test suite for DataAugmentation class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.augmenter = DataAugmentation()
        # Create synthetic test data
        np.random.seed(42)
        
    def test_initialization(self):
        """Test DataAugmentation initialization"""
        assert self.augmenter is not None
        # Check if SMOTE is available (should be after installing imbalanced-learn)
        assert hasattr(self.augmenter, 'smote_available')
        
    def test_jittering_2d(self):
        """Test jittering on 2D data"""
        X = np.random.randn(100, 10)  # 100 samples, 10 features
        X_jittered = self.augmenter.add_jittering(X, noise_level=0.01)
        
        # Check shape is preserved
        assert X_jittered.shape == X.shape
        
        # Check that jittering was applied (data should be different)
        assert not np.allclose(X, X_jittered)
        
        # Check that jittering is small (within reasonable bounds)
        diff = np.abs(X - X_jittered)
        assert np.mean(diff) < 0.1  # Mean difference should be small
        
    def test_jittering_3d(self):
        """Test jittering on 3D data (sequences)"""
        X = np.random.randn(50, 20, 10)  # 50 sequences, 20 timesteps, 10 features
        X_jittered = self.augmenter.add_jittering(X, noise_level=0.01)
        
        # Check shape is preserved
        assert X_jittered.shape == X.shape
        
        # Check that jittering was applied
        assert not np.allclose(X, X_jittered)
        
    def test_smote_2d(self):
        """Test SMOTE on 2D data with imbalanced classes"""
        if not self.augmenter.smote_available:
            pytest.skip("SMOTE not available")
            
        # Create imbalanced dataset
        X = np.random.randn(100, 10)
        # Class 0: 70 samples, Class 1: 20 samples, Class 2: 10 samples
        y = np.array([0]*70 + [1]*20 + [2]*10)
        
        X_aug, y_aug = self.augmenter.augment_with_smote(X, y)
        
        # Check that augmentation increased samples
        assert len(X_aug) >= len(X)
        assert len(y_aug) == len(X_aug)
        
        # Check class distribution is more balanced
        unique, counts = np.unique(y_aug, return_counts=True)
        assert len(unique) == 3  # Still 3 classes
        # Minority classes should have more samples after SMOTE
        assert counts[1] > 20  # Class 1 should have more than 20
        assert counts[2] > 10  # Class 2 should have more than 10
        
    def test_smote_3d(self):
        """Test SMOTE on 3D sequence data"""
        if not self.augmenter.smote_available:
            pytest.skip("SMOTE not available")
            
        # Create imbalanced 3D dataset
        X = np.random.randn(100, 20, 10)  # 100 sequences, 20 timesteps, 10 features
        y = np.array([0]*70 + [1]*20 + [2]*10)
        
        X_aug, y_aug = self.augmenter.augment_with_smote(X, y)
        
        # Check shape preservation (should still be 3D)
        assert len(X_aug.shape) == 3
        assert X_aug.shape[1] == 20  # Sequence length preserved
        assert X_aug.shape[2] == 10  # Feature count preserved
        
        # Check augmentation worked
        assert len(X_aug) >= len(X)
        assert len(y_aug) == len(X_aug)
        
    def test_augment_sequence_data_full_pipeline(self):
        """Test full augmentation pipeline with SMOTE + jittering"""
        if not self.augmenter.smote_available:
            pytest.skip("SMOTE not available")
            
        # Create test data
        X = np.random.randn(100, 20, 10)
        y = np.array([0]*70 + [1]*20 + [2]*10)
        
        X_aug, y_aug = self.augmenter.augment_sequence_data(
            X, y,
            use_smote=True,
            use_jittering=True,
            jitter_noise=0.01
        )
        
        # Check that data was augmented significantly
        assert len(X_aug) > len(X)
        # With jittering, we should roughly double the SMOTE-augmented data
        assert len(X_aug) >= len(X) * 1.5
        
        # Check shape preservation
        assert X_aug.shape[1] == X.shape[1]  # Sequence length
        assert X_aug.shape[2] == X.shape[2]  # Features
        
    def test_augment_without_smote(self):
        """Test augmentation with only jittering (no SMOTE)"""
        X = np.random.randn(100, 20, 10)
        y = np.array([0]*70 + [1]*20 + [2]*10)
        
        X_aug, y_aug = self.augmenter.augment_sequence_data(
            X, y,
            use_smote=False,
            use_jittering=True,
            jitter_noise=0.01
        )
        
        # Should double the data (original + jittered)
        assert len(X_aug) == len(X) * 2
        assert len(y_aug) == len(X_aug)
        
    def test_augment_without_jittering(self):
        """Test augmentation with only SMOTE (no jittering)"""
        if not self.augmenter.smote_available:
            pytest.skip("SMOTE not available")
            
        X = np.random.randn(100, 20, 10)
        y = np.array([0]*70 + [1]*20 + [2]*10)
        
        X_aug, y_aug = self.augmenter.augment_sequence_data(
            X, y,
            use_smote=True,
            use_jittering=False,
            jitter_noise=0.01
        )
        
        # Should only apply SMOTE (balances minority classes to majority)
        # Original: 70+20+10=100, After SMOTE: ~70+70+70=210
        assert len(X_aug) >= len(X)
        # SMOTE balances classes, so we expect roughly 2x original size
        assert len(X_aug) >= len(X) * 1.5  # Allow some flexibility
        
    def test_no_augmentation(self):
        """Test when both augmentation methods are disabled"""
        X = np.random.randn(100, 20, 10)
        y = np.array([0]*70 + [1]*20 + [2]*10)
        
        X_aug, y_aug = self.augmenter.augment_sequence_data(
            X, y,
            use_smote=False,
            use_jittering=False
        )
        
        # Should return original data
        assert np.allclose(X_aug, X)
        assert np.array_equal(y_aug, y)
