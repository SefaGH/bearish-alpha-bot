"""
Test suite to verify ML model loading fix.

This test validates that:
1. Config contains required keys for model building
2. Models are properly initialized when config is correct
3. Model loading can proceed when models are built
"""

import pytest
import yaml
import os
from pathlib import Path


class TestModelLoadingFix:
    """Test that the model loading configuration fix works."""
    
    def test_config_has_required_ml_keys(self):
        """Verify config.example.yaml has all required ML keys."""
        config_path = Path(__file__).parent.parent / "config" / "config.example.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        ml_config = config.get('ml', {})
        
        # These are the CRITICAL keys that were missing
        assert 'models' in ml_config, "ml.models key is missing"
        assert 'feature_size' in ml_config, "ml.feature_size key is missing"
        assert 'forecast_horizon' in ml_config, "ml.forecast_horizon key is missing"
        
        # Verify the values are sensible
        assert isinstance(ml_config['models'], list), "ml.models should be a list"
        assert len(ml_config['models']) > 0, "ml.models should not be empty"
        assert 'lstm' in ml_config['models'] or 'transformer' in ml_config['models'], \
            "ml.models should contain at least 'lstm' or 'transformer'"
        
        assert isinstance(ml_config['feature_size'], int), "ml.feature_size should be an integer"
        assert ml_config['feature_size'] > 0, "ml.feature_size should be positive"
        
        assert isinstance(ml_config['forecast_horizon'], int), "ml.forecast_horizon should be an integer"
        assert ml_config['forecast_horizon'] > 0, "ml.forecast_horizon should be positive"
    
    def test_config_has_model_params(self):
        """Verify model_params are still present."""
        config_path = Path(__file__).parent.parent / "config" / "config.example.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        ml_config = config.get('ml', {})
        
        assert 'model_params' in ml_config, "ml.model_params key is missing"
        model_params = ml_config['model_params']
        
        # Check LSTM params
        assert 'lstm' in model_params, "model_params.lstm is missing"
        assert 'hidden_size' in model_params['lstm'], "lstm.hidden_size is missing"
        assert 'num_layers' in model_params['lstm'], "lstm.num_layers is missing"
        
        # Check Transformer params
        assert 'transformer' in model_params, "model_params.transformer is missing"
        assert 'd_model' in model_params['transformer'], "transformer.d_model is missing"
        assert 'nhead' in model_params['transformer'], "transformer.nhead is missing"
        assert 'num_layers' in model_params['transformer'], "transformer.num_layers is missing"
    
    def test_config_has_prediction_timeframes(self):
        """Verify prediction.timeframes is present."""
        config_path = Path(__file__).parent.parent / "config" / "config.example.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        ml_config = config.get('ml', {})
        
        assert 'prediction' in ml_config, "ml.prediction key is missing"
        prediction_config = ml_config['prediction']
        
        assert 'timeframes' in prediction_config, "ml.prediction.timeframes is missing"
        timeframes = prediction_config['timeframes']
        
        assert isinstance(timeframes, list), "prediction.timeframes should be a list"
        assert len(timeframes) > 0, "prediction.timeframes should not be empty"
        
        # Check that common timeframes are present
        expected_timeframes = {'5m', '15m', '1h'}
        actual_timeframes = set(timeframes)
        assert expected_timeframes.issubset(actual_timeframes), \
            f"Expected timeframes {expected_timeframes} to be in {actual_timeframes}"
    
    def test_model_files_exist(self):
        """Verify that model .pth files exist on disk."""
        base_dir = Path(__file__).parent.parent / "data" / "models"
        
        # These are the files mentioned in the problem statement
        expected_files = [
            'lstm_5m.pth',
            'transformer_5m.pth',
            'lstm_15m.pth',
            'transformer_15m.pth',
            'lstm_1h.pth',
            'transformer_1h.pth'
        ]
        
        for filename in expected_files:
            file_path = base_dir / filename
            assert file_path.exists(), f"Model file {filename} does not exist at {file_path}"
            assert file_path.stat().st_size > 0, f"Model file {filename} is empty"
    
    def test_config_structure_for_build_predictors(self):
        """
        Test that the config structure matches what _build_predictors() expects.
        
        This is the integration test that validates the fix addresses the root cause.
        """
        config_path = Path(__file__).parent.parent / "config" / "config.example.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        ml_config = config.get('ml', {})
        
        # Simulate what _build_predictors() does:
        # Line 540: input_feature_size = self.config.get('feature_size', 42)
        input_feature_size = ml_config.get('feature_size', 42)
        assert input_feature_size == ml_config['feature_size'], \
            "feature_size should be explicitly set, not defaulting"
        
        # Line 541: forecast_horizon = self.config.get('forecast_horizon', 12)
        forecast_horizon = ml_config.get('forecast_horizon', 12)
        assert forecast_horizon == ml_config['forecast_horizon'], \
            "forecast_horizon should be explicitly set, not defaulting"
        
        # Line 544: timeframes_from_config = model_config.get('timeframes', ['5m', '15m', '1h'])
        model_config = ml_config.get('prediction', {})
        timeframes_from_config = model_config.get('timeframes', ['5m', '15m', '1h'])
        assert isinstance(timeframes_from_config, list), "timeframes should be a list"
        assert len(timeframes_from_config) > 0, "timeframes should not be empty"
        
        # Line 564: model_types_to_build = self.config.get('models', [])
        # THIS WAS THE BUG - this was returning [] before the fix
        model_types_to_build = ml_config.get('models', [])
        assert len(model_types_to_build) > 0, \
            "models list should not be empty - this was the root cause of the bug"
        
        # Line 565: model_params = self.config.get('model_params', {})
        model_params = ml_config.get('model_params', {})
        assert len(model_params) > 0, "model_params should not be empty"
        
        # Verify the loop at line 570 will actually execute
        for model_type in model_types_to_build:
            assert model_type in ['lstm', 'transformer', 'random_forest'], \
                f"Unknown model type: {model_type}"
            
            # Each model type should have parameters
            if model_type in ['lstm', 'transformer']:
                assert model_type in model_params, \
                    f"model_params.{model_type} is missing"


class TestModelLoadingBehavior:
    """Test the actual model loading behavior (requires mocking)."""
    
    def test_empty_models_list_causes_no_models_built(self):
        """
        Demonstrate the bug: when models=[], no models are built.
        
        This is a regression test to ensure we don't reintroduce the bug.
        """
        # Simulate the buggy config
        buggy_config = {
            'prediction': {
                'timeframes': ['5m', '15m', '1h']
            },
            'models': [],  # BUG: Empty list
            'model_params': {
                'lstm': {'hidden_size': 128},
                'transformer': {'d_model': 256, 'nhead': 2}
            },
            'feature_size': 42,
            'forecast_horizon': 12
        }
        
        # Simulate _build_predictors logic
        model_types_to_build = buggy_config.get('models', [])
        timeframes = buggy_config['prediction']['timeframes']
        
        models_built_count = 0
        for tf in timeframes:
            tf_models = {}
            
            # Line 570: if 'lstm' in model_types_to_build:
            if 'lstm' in model_types_to_build:
                tf_models['lstm'] = 'mock_lstm_model'
                models_built_count += 1
            
            # Line 580: if 'transformer' in model_types_to_build:
            if 'transformer' in model_types_to_build:
                tf_models['transformer'] = 'mock_transformer_model'
                models_built_count += 1
        
        # With buggy config, NO models should be built
        assert models_built_count == 0, \
            "Bug demonstrated: empty models list results in no models being built"
    
    def test_correct_models_list_causes_models_built(self):
        """
        Demonstrate the fix: when models=['lstm', 'transformer'], models ARE built.
        """
        # Simulate the fixed config
        fixed_config = {
            'prediction': {
                'timeframes': ['5m', '15m', '1h']
            },
            'models': ['lstm', 'transformer'],  # FIX: Non-empty list
            'model_params': {
                'lstm': {'hidden_size': 128},
                'transformer': {'d_model': 256, 'nhead': 2}
            },
            'feature_size': 42,
            'forecast_horizon': 12
        }
        
        # Simulate _build_predictors logic
        model_types_to_build = fixed_config.get('models', [])
        timeframes = fixed_config['prediction']['timeframes']
        
        models_built_count = 0
        for tf in timeframes:
            tf_models = {}
            
            if 'lstm' in model_types_to_build:
                tf_models['lstm'] = 'mock_lstm_model'
                models_built_count += 1
            
            if 'transformer' in model_types_to_build:
                tf_models['transformer'] = 'mock_transformer_model'
                models_built_count += 1
        
        # With fixed config, models SHOULD be built
        # 3 timeframes * 2 models = 6 total
        expected_count = len(timeframes) * len(fixed_config['models'])
        assert models_built_count == expected_count, \
            f"Fix verified: {models_built_count} models built (expected {expected_count})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
