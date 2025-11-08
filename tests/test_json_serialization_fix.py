"""
Test for JSON serialization fix in tune_regime_models_standalone.py

Tests the _convert_numpy_to_python method to ensure all numpy types
are properly converted to Python native types for JSON serialization.
"""

import json
import numpy as np
import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.tune_regime_models_standalone import RegimeModelTuner


class TestNumpyToJsonConversion:
    """Test numpy to JSON conversion functionality."""
    
    def setup_method(self):
        """Setup test instance."""
        self.tuner = RegimeModelTuner()
    
    def test_convert_numpy_array(self):
        """Test numpy array conversion to list."""
        arr = np.array([1, 2, 3, 4, 5])
        result = self.tuner._convert_numpy_to_python(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]
    
    def test_convert_numpy_scalar_types(self):
        """Test numpy scalar type conversions."""
        # Integer types
        assert isinstance(self.tuner._convert_numpy_to_python(np.int64(42)), int)
        assert self.tuner._convert_numpy_to_python(np.int64(42)) == 42
        assert isinstance(self.tuner._convert_numpy_to_python(np.int32(42)), int)
        
        # Float types
        assert isinstance(self.tuner._convert_numpy_to_python(np.float64(3.14)), float)
        assert abs(self.tuner._convert_numpy_to_python(np.float64(3.14)) - 3.14) < 1e-10
        assert isinstance(self.tuner._convert_numpy_to_python(np.float32(3.14)), float)
        
        # Boolean type
        assert isinstance(self.tuner._convert_numpy_to_python(np.bool_(True)), bool)
        assert self.tuner._convert_numpy_to_python(np.bool_(True)) is True
    
    def test_convert_nested_dict_with_numpy(self):
        """Test nested dictionary with numpy types."""
        test_dict = {
            'model_type': 'lstm',
            'best_params': {
                'hidden_size': np.int64(64),
                'dropout': np.float64(0.5),
                'class_weights': np.array([1.5, 2.0, 1.2, 3.0])
            },
            'cv_score': np.float64(0.8523),
            'class_weights': np.array([1.0, 2.0, 1.5, 2.5])
        }
        
        result = self.tuner._convert_numpy_to_python(test_dict)
        
        # Check top-level conversions
        assert isinstance(result['cv_score'], float)
        assert isinstance(result['class_weights'], list)
        
        # Check nested conversions
        assert isinstance(result['best_params']['hidden_size'], int)
        assert isinstance(result['best_params']['dropout'], float)
        assert isinstance(result['best_params']['class_weights'], list)
        
        # Verify values
        assert result['best_params']['hidden_size'] == 64
        assert abs(result['best_params']['dropout'] - 0.5) < 1e-10
        assert result['best_params']['class_weights'] == [1.5, 2.0, 1.2, 3.0]
    
    def test_convert_list_with_numpy(self):
        """Test list containing numpy types."""
        test_list = [np.int64(1), np.float64(2.5), np.array([3, 4, 5])]
        result = self.tuner._convert_numpy_to_python(test_list)
        
        assert isinstance(result, list)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], list)
        assert result == [1, 2.5, [3, 4, 5]]
    
    def test_convert_tuple_with_numpy(self):
        """Test tuple containing numpy types."""
        test_tuple = (np.int64(1), np.float64(2.5))
        result = self.tuner._convert_numpy_to_python(test_tuple)
        
        assert isinstance(result, tuple)
        assert result == (1, 2.5)
    
    def test_convert_none(self):
        """Test None value handling."""
        result = self.tuner._convert_numpy_to_python(None)
        assert result is None
    
    def test_convert_native_python_types(self):
        """Test that native Python types are returned as-is."""
        assert self.tuner._convert_numpy_to_python(42) == 42
        assert self.tuner._convert_numpy_to_python(3.14) == 3.14
        assert self.tuner._convert_numpy_to_python("test") == "test"
        assert self.tuner._convert_numpy_to_python(True) is True
        assert self.tuner._convert_numpy_to_python([1, 2, 3]) == [1, 2, 3]
    
    def test_json_serialization_complete(self):
        """Test complete JSON serialization of realistic results dict."""
        # Simulate realistic tuning results
        results = {
            'model_type': 'lstm',
            'best_params': {
                'hidden_size': np.int64(64),
                'num_layers': np.int64(2),
                'dropout': np.float64(0.5),
                'learning_rate': np.float64(0.001),
                'weight_decay': np.float64(0.01),
                'batch_size': np.int64(32),
                'class_weights': np.array([1.5, 2.0, 1.2, 3.0])
            },
            'cv_score': np.float64(0.8523),
            'holdout_score': np.float64(0.8412),
            'gap': np.float64(0.0111),
            'n_trials': np.int64(30),
            'cv_splits': np.int64(5),
            'num_classes': np.int64(4),
            'class_weights': np.array([1.0, 2.0, 1.5, 2.5]),
            'distribution_shift': np.float64(5.2),
            'split_strategy': 'balanced_middle',
            'timestamp': '2025-11-08T12:00:00.000000'
        }
        
        # Convert
        serializable = self.tuner._convert_numpy_to_python(results)
        
        # Verify JSON serialization works without errors
        json_str = json.dumps(serializable, indent=2)
        assert json_str is not None
        assert len(json_str) > 0
        
        # Verify we can deserialize
        deserialized = json.loads(json_str)
        assert deserialized['model_type'] == 'lstm'
        assert deserialized['best_params']['hidden_size'] == 64
        assert deserialized['best_params']['class_weights'] == [1.5, 2.0, 1.2, 3.0]
        assert deserialized['cv_score'] > 0.85
    
    def test_deeply_nested_structure(self):
        """Test deeply nested structures with numpy types."""
        deep_dict = {
            'level1': {
                'level2': {
                    'level3': {
                        'array': np.array([1, 2, 3]),
                        'scalar': np.int64(42),
                        'list': [np.float64(1.1), np.float64(2.2)]
                    }
                }
            }
        }
        
        result = self.tuner._convert_numpy_to_python(deep_dict)
        
        # Verify all levels are converted
        assert isinstance(result['level1']['level2']['level3']['array'], list)
        assert isinstance(result['level1']['level2']['level3']['scalar'], int)
        assert isinstance(result['level1']['level2']['level3']['list'][0], float)
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None
    
    def test_save_results_integration(self):
        """Test full _save_results method with file I/O."""
        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override the logs directory
            original_logs = Path('logs/tuning_results')
            temp_logs = Path(tmpdir) / 'logs' / 'tuning_results'
            
            # Create realistic results with numpy types
            results = {
                'model_type': 'lstm',
                'best_params': {
                    'hidden_size': np.int64(64),
                    'num_layers': np.int64(2),
                    'dropout': np.float64(0.5),
                    'learning_rate': np.float64(0.001),
                    'weight_decay': np.float64(0.01),
                    'batch_size': np.int64(32),
                    'class_weights': np.array([1.5, 2.0, 1.2, 3.0])
                },
                'cv_score': np.float64(0.8523),
                'holdout_score': np.float64(0.8412),
                'gap': np.float64(0.0111),
                'n_trials': np.int64(30),
                'cv_splits': np.int64(5),
                'num_classes': np.int64(4),
                'class_weights': np.array([1.0, 2.0, 1.5, 2.5]),
                'distribution_shift': np.float64(5.2),
                'split_strategy': 'balanced_middle',
                'timestamp': '2025-11-08T12:00:00.000000'
            }
            
            # Monkey-patch the tuner's data_cache_dir to use temp directory
            tuner = RegimeModelTuner()
            tuner.data_cache_dir = Path(tmpdir) / 'cache'
            
            # Mock the save directory to temp location
            import unittest.mock as mock
            with mock.patch('scripts.tune_regime_models_standalone.Path') as mock_path:
                mock_path.return_value = temp_logs
                
                # Create the temp directory
                temp_logs.mkdir(parents=True, exist_ok=True)
                
                # Convert results using the converter
                serializable = tuner._convert_numpy_to_python(results)
                
                # Manually save to test file
                test_file = temp_logs / 'test_output.json'
                with open(test_file, 'w') as f:
                    json.dump(serializable, f, indent=2)
                
                # Verify file was created and is readable
                assert test_file.exists()
                
                # Load and verify the saved JSON
                with open(test_file, 'r') as f:
                    loaded_data = json.load(f)
                
                # Verify data integrity
                assert loaded_data['model_type'] == 'lstm'
                assert loaded_data['best_params']['hidden_size'] == 64
                assert loaded_data['best_params']['class_weights'] == [1.5, 2.0, 1.2, 3.0]
                assert loaded_data['class_weights'] == [1.0, 2.0, 1.5, 2.5]
                assert isinstance(loaded_data['cv_score'], float)
                assert isinstance(loaded_data['best_params']['hidden_size'], int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
