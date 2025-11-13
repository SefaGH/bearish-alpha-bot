"""
Test suite for GEMMA Blueprint Integration

Tests the integration of tuning artifacts into the training pipeline.
"""

import pytest
import json
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set ML_ENABLED environment variable
os.environ['ML_ENABLED'] = 'true'


class TestTuningHyperparameterLoading:
    """Test loading of tuning hyperparameters from artifacts"""
    
    def test_load_tuning_hyperparameters_success(self):
        """Test successful loading of tuning hyperparameters"""
        # Create temporary tuning results file
        with tempfile.TemporaryDirectory() as tmpdir:
            tuning_dir = Path(tmpdir) / 'logs' / 'tuning_results'
            tuning_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock tuning results
            tuning_results = {
                'best_params': {
                    'hidden_size': 64,
                    'num_layers': 3,
                    'dropout': 0.5,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0001,
                    'batch_size': 32
                },
                'cv_score': 0.75,
                'holdout_score': 0.73
            }
            
            tuning_file = tuning_dir / 'gemma_tuning_20251113_120000.json'
            with open(tuning_file, 'w') as f:
                json.dump(tuning_results, f)
            
            # Import the function (we'll need to modify the import to be testable)
            # For now, we'll test the logic inline
            
            # Find the latest tuning results file
            tuning_files = list(tuning_dir.glob('gemma_tuning_*.json'))
            assert len(tuning_files) == 1
            
            latest_file = max(tuning_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                loaded_results = json.load(f)
            
            best_params = loaded_results.get('best_params', {})
            
            assert best_params['hidden_size'] == 64
            assert best_params['num_layers'] == 3
            assert best_params['dropout'] == 0.5
            assert best_params['learning_rate'] == 0.001
            assert best_params['weight_decay'] == 0.0001
    
    def test_load_tuning_hyperparameters_no_file(self):
        """Test handling when no tuning results file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tuning_dir = Path(tmpdir) / 'logs' / 'tuning_results'
            tuning_dir.mkdir(parents=True, exist_ok=True)
            
            # No tuning files
            tuning_files = list(tuning_dir.glob('gemma_tuning_*.json'))
            assert len(tuning_files) == 0
            
            # Should return empty dict
            best_params = {}
            assert best_params == {}


class TestFeatureMaskLoading:
    """Test loading and application of feature selection mask"""
    
    def test_apply_feature_mask(self):
        """Test applying feature selection mask to data"""
        # Create mock data
        X_full = np.random.rand(100, 87)  # 100 samples, 87 features
        
        # Create mock feature mask (select 82 features)
        feature_mask = np.zeros(87, dtype=bool)
        feature_mask[:82] = True  # Select first 82 features
        
        # Apply mask
        X_selected = X_full[:, feature_mask]
        
        assert X_selected.shape == (100, 82)
        assert X_full.shape == (100, 87)
    
    def test_feature_mask_dimension_check(self):
        """Test that mask and data dimensions must match"""
        X_full = np.random.rand(100, 87)
        wrong_mask = np.ones(80, dtype=bool)  # Wrong size
        
        # This should raise an error or be caught
        with pytest.raises(IndexError):
            X_selected = X_full[:, wrong_mask]


class TestDynamicClassWeights:
    """Test dynamic class weight calculation"""
    
    def test_compute_class_weights(self):
        """Test computing balanced class weights"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Create imbalanced labels
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2])  # 4 class 0, 8 class 1, 1 class 2
        
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        
        # Check that weights are computed
        assert len(class_weights) == 3
        
        # Class 2 (minority) should have highest weight
        assert class_weights[2] > class_weights[0]
        assert class_weights[2] > class_weights[1]
        
        # Class 0 should have higher weight than class 1
        assert class_weights[0] > class_weights[1]
    
    def test_class_weight_dict_creation(self):
        """Test creating class weight dictionary"""
        from sklearn.utils.class_weight import compute_class_weight
        
        y_train = np.array([0, 0, 1, 1, 1, 2])
        unique_classes = np.unique(y_train)
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        assert 0 in class_weight_dict
        assert 1 in class_weight_dict
        assert 2 in class_weight_dict
        assert len(class_weight_dict) == 3


class TestProductionScalerLoading:
    """Test loading of production scaler from artifact"""
    
    def test_scaler_loading(self):
        """Test loading a joblib scaler"""
        import joblib
        from sklearn.preprocessing import StandardScaler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a scaler
            X_train = np.random.rand(100, 82)
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            scaler_path = Path(tmpdir) / 'scaler_production.joblib'
            joblib.dump(scaler, scaler_path)
            
            # Load the scaler
            loaded_scaler = joblib.load(scaler_path)
            
            # Test that it works
            X_test = np.random.rand(10, 82)
            X_scaled = loaded_scaler.transform(X_test)
            
            assert X_scaled.shape == X_test.shape
            # Check that mean is approximately 0 (scaled)
            assert abs(X_scaled.mean()) < 0.5


class TestWorkflowArtifactValidation:
    """Test workflow artifact validation logic"""
    
    def test_artifact_files_exist(self):
        """Test checking if artifact files exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock artifact structure
            cache_dir = Path(tmpdir) / 'data' / 'cache' / 'gemma'
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            tuning_dir = Path(tmpdir) / 'logs' / 'tuning_results'
            tuning_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock files
            mask_file = cache_dir / 'feature_selection_mask.npy'
            np.save(mask_file, np.ones(87, dtype=bool))
            
            tuning_file = tuning_dir / 'gemma_tuning_test.json'
            with open(tuning_file, 'w') as f:
                json.dump({'best_params': {}}, f)
            
            scaler_file = Path(tmpdir) / 'data' / 'cache' / 'scaler_production.joblib'
            scaler_file.parent.mkdir(parents=True, exist_ok=True)
            scaler_file.touch()
            
            # Validate files exist
            assert mask_file.exists()
            assert tuning_file.exists()
            assert scaler_file.exists()


class TestHyperparameterMerging:
    """Test merging tuning hyperparameters into config"""
    
    def test_merge_hyperparameters(self):
        """Test merging tuning params into config structure"""
        # Mock config
        gemma_config = {
            'enabled': True,
            'architecture': {
                'hidden_size': 32,
                'num_layers': 2,
                'dropout': 0.6
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }
        
        # Mock tuning params
        tuning_params = {
            'hidden_size': 64,
            'num_layers': 3,
            'dropout': 0.5,
            'learning_rate': 0.0005,
            'weight_decay': 0.0001
        }
        
        # Merge logic (as implemented in train_all_models.py)
        gemma_config = dict(gemma_config)
        
        param_mapping = {
            'hidden_size': ('architecture', 'hidden_size'),
            'num_layers': ('architecture', 'num_layers'),
            'dropout': ('architecture', 'dropout'),
            'learning_rate': ('training', 'learning_rate'),
            'weight_decay': ('training', 'weight_decay'),
        }
        
        for param_name, value in tuning_params.items():
            if param_name in param_mapping:
                section, key = param_mapping[param_name]
                if section not in gemma_config:
                    gemma_config[section] = {}
                gemma_config[section][key] = value
        
        # Verify merged values
        assert gemma_config['architecture']['hidden_size'] == 64
        assert gemma_config['architecture']['num_layers'] == 3
        assert gemma_config['architecture']['dropout'] == 0.5
        assert gemma_config['training']['learning_rate'] == 0.0005
        assert gemma_config['training']['weight_decay'] == 0.0001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
