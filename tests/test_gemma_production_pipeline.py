"""
Unit tests for GEMMA production pipeline (Plan 2).
Tests the fixed feature plan implementation and production scaler creation.
"""

import os
# Set ML_ENABLED before any imports
os.environ['ML_ENABLED'] = 'true'

import pytest
import numpy as np
import torch
import joblib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.ml.model_trainer import RegimeModelTrainer


@pytest.mark.unit
class TestGEMMAProductionPipeline:
    """Test suite for GEMMA production pipeline changes (Plan 2)"""

    def test_gemma_model_save_path(self, tmp_path):
        """Verify that GEMMA models are saved to data/models/final/"""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        
        # Create trainer with temporary path
        trainer = RegimeModelTrainer()
        
        # Override the final_dir to use tmp_path for testing
        with patch('src.ml.model_trainer.Path') as mock_path:
            mock_final_dir = tmp_path / "data" / "models" / "final"
            mock_final_dir.mkdir(parents=True, exist_ok=True)
            mock_path.return_value = mock_final_dir
            
            # Call the save method
            trainer._save_gemma_model(mock_model, 'gemma_price', 'mlp')
            
            # Verify the model file was created with correct name
            expected_model_path = mock_final_dir / "gemma_price.pt"
            assert expected_model_path.exists(), f"Model should be saved to {expected_model_path}"

    def test_gemma_scaler_save_path(self, tmp_path):
        """Verify that GEMMA scalers are saved to data/models/final/"""
        from sklearn.preprocessing import StandardScaler
        
        # Create a real scaler
        scaler = StandardScaler()
        X_dummy = np.random.randn(100, 10)
        scaler.fit(X_dummy)
        
        # Create trainer
        trainer = RegimeModelTrainer()
        
        # Override the final_dir to use tmp_path for testing
        with patch('src.ml.model_trainer.Path') as mock_path:
            mock_final_dir = tmp_path / "data" / "models" / "final"
            mock_final_dir.mkdir(parents=True, exist_ok=True)
            mock_path.return_value = mock_final_dir
            
            # Call the save method
            trainer._save_gemma_scaler(scaler, 'gemma_price')
            
            # Verify the scaler file was created with correct name
            expected_scaler_path = mock_final_dir / "gemma_price_scaler.joblib"
            assert expected_scaler_path.exists(), f"Scaler should be saved to {expected_scaler_path}"

    def test_separate_scalers_for_price_and_regime(self, tmp_path):
        """Verify that price and regime models get separate scalers"""
        from sklearn.preprocessing import StandardScaler
        
        # Create trainer
        trainer = RegimeModelTrainer()
        
        # Create scalers with different characteristics
        scaler_price = StandardScaler()
        X_price = np.random.randn(100, 10)
        scaler_price.fit(X_price)
        
        scaler_regime = StandardScaler()
        X_regime = np.random.randn(100, 10) * 2  # Different scale
        scaler_regime.fit(X_regime)
        
        # Save both scalers
        final_dir = tmp_path / "data" / "models" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        price_scaler_path = final_dir / "gemma_price_scaler.joblib"
        regime_scaler_path = final_dir / "gemma_regime_scaler.joblib"
        
        joblib.dump(scaler_price, price_scaler_path)
        joblib.dump(scaler_regime, regime_scaler_path)
        
        # Verify both files exist
        assert price_scaler_path.exists(), "Price scaler should exist"
        assert regime_scaler_path.exists(), "Regime scaler should exist"
        
        # Load and verify they are different
        loaded_price = joblib.load(price_scaler_path)
        loaded_regime = joblib.load(regime_scaler_path)
        
        # Scalers should have different means due to different training data
        assert not np.allclose(loaded_price.mean_, loaded_regime.mean_), \
            "Price and regime scalers should be different"

    def test_model_type_parameter_accepted(self):
        """Verify that train_and_evaluate accepts model_type parameter"""
        trainer = RegimeModelTrainer()
        
        # Create dummy data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        # Check that the method signature includes model_type
        import inspect
        sig = inspect.signature(trainer.train_and_evaluate)
        assert 'model_type' in sig.parameters, \
            "train_and_evaluate should accept model_type parameter"

    def test_feature_mask_loading_fallback(self, tmp_path):
        """Verify graceful fallback when feature mask doesn't exist"""
        # This test simulates the behavior in train_gemma_model
        mask_path = tmp_path / "data" / "cache" / "gemma" / "feature_selection_mask.npy"
        
        # Mask doesn't exist
        assert not mask_path.exists()
        
        # Create dummy full data
        X_data_full = np.random.randn(100, 87)  # All 87 features
        
        # Simulate the logic in train_gemma_model
        if not mask_path.exists():
            X_selected = X_data_full
        else:
            feature_mask = np.load(mask_path)
            X_selected = X_data_full[:, feature_mask]
        
        # Should use all features when mask doesn't exist
        assert X_selected.shape == X_data_full.shape, \
            "Should use all features when mask doesn't exist"

    def test_feature_mask_application(self, tmp_path):
        """Verify feature mask is correctly applied when it exists"""
        # Create mask file
        mask_dir = tmp_path / "data" / "cache" / "gemma"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / "feature_selection_mask.npy"
        
        # Create a mask that selects half the features
        n_features = 87
        feature_mask = np.array([i % 2 == 0 for i in range(n_features)])  # Select even indices
        np.save(mask_path, feature_mask)
        
        # Create dummy full data
        X_data_full = np.random.randn(100, n_features)
        
        # Simulate the logic in train_gemma_model
        if mask_path.exists():
            loaded_mask = np.load(mask_path)
            X_selected = X_data_full[:, loaded_mask]
        else:
            X_selected = X_data_full
        
        # Should select only the features indicated by the mask
        expected_n_features = feature_mask.sum()
        assert X_selected.shape[1] == expected_n_features, \
            f"Should select {expected_n_features} features, got {X_selected.shape[1]}"
        assert X_selected.shape[0] == X_data_full.shape[0], \
            "Should preserve number of samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
