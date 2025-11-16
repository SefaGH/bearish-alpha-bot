"""
Test Suite for GEMMA Model Training Pipeline Integration (Phase 3)

Tests for:
1. GEMMA configuration loading from config.example.yaml
2. train_gemma_model function existence and signature
3. Configuration validation and defaults
"""

import pytest
import yaml
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


class TestGemmaConfiguration:
    """Test GEMMA configuration in config.example.yaml"""
    
    def test_gemma_config_exists(self):
        """GEMMA configuration block should exist in config.example.yaml"""
        config_path = Path(project_root) / 'config' / 'config.example.yaml'
        assert config_path.exists(), "config.example.yaml not found"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        ml_config = config.get('ml', {})
        assert 'gemma' in ml_config, "GEMMA config block not found in ml section"
    
    def test_gemma_config_structure(self):
        """GEMMA configuration should have required keys"""
        config_path = Path(project_root) / 'config' / 'config.example.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        gemma_config = config['ml']['gemma']
        
        # Check top-level keys
        assert 'enabled' in gemma_config, "enabled key missing"
        assert 'feature_set' in gemma_config, "feature_set key missing"
        assert 'architecture' in gemma_config, "architecture key missing"
        assert 'training' in gemma_config, "training key missing"
        assert 'thresholds' in gemma_config, "thresholds key missing"
    
    def test_gemma_architecture_params(self):
        """GEMMA architecture should have correct parameters"""
        config_path = Path(project_root) / 'config' / 'config.example.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        arch = config['ml']['gemma']['architecture']
        
        # Validate architecture parameters
        assert arch['input_size'] == 82, "Input size should be 82 (Phase 2 features)"
        assert arch['hidden_size'] > 0, "Hidden size must be positive"
        assert arch['num_layers'] > 0, "Number of layers must be positive"
        assert 0 <= arch['dropout'] <= 1, "Dropout must be between 0 and 1"
        assert arch['num_classes'] == 3, "Should have 3 classes (bearish, neutral, bullish)"
    
    def test_gemma_training_params(self):
        """GEMMA training parameters should be valid"""
        config_path = Path(project_root) / 'config' / 'config.example.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        training = config['ml']['gemma']['training']
        
        assert training['epochs'] > 0, "Epochs must be positive"
        assert training['batch_size'] > 0, "Batch size must be positive"
        assert training['learning_rate'] > 0, "Learning rate must be positive"
        assert training['early_stopping_patience'] > 0, "Patience must be positive"
    
    def test_gemma_thresholds(self):
        """GEMMA thresholds should be valid"""
        config_path = Path(project_root) / 'config' / 'config.example.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        thresholds = config['ml']['gemma']['thresholds']
        
        assert 0 <= thresholds['deployment_accuracy'] <= 1, \
            "Deployment accuracy must be between 0 and 1"
        assert thresholds['min_samples'] > 0, "Min samples must be positive"


class TestGemmaTrainingFunction:
    """Test train_gemma_model function"""
    
    def test_train_gemma_model_function_exists(self):
        """train_gemma_model function should exist in train_all_models.py"""
        # Read the source file directly to avoid import issues
        train_all_models_path = Path(project_root) / 'scripts' / 'train_all_models.py'
        
        with open(train_all_models_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check function definition exists
        assert 'def train_gemma_model(' in source, \
            "train_gemma_model function not found in train_all_models.py"
    
    def test_train_gemma_model_signature(self):
        """train_gemma_model should have correct signature"""
        train_all_models_path = Path(project_root) / 'scripts' / 'train_all_models.py'
        
        with open(train_all_models_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Find function definition
        func_def_start = source.find('def train_gemma_model(')
        assert func_def_start != -1, "Function definition not found"
        
        # Extract function signature (up to closing paren and colon)
        # Find the end of the function signature
        paren_count = 0
        func_def_end = func_def_start
        for i, char in enumerate(source[func_def_start:], start=func_def_start):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    func_def_end = i
                    break
        
        func_signature = source[func_def_start:func_def_end]
        
        # Check required parameters are in signature
        assert 'X_selected' in func_signature, "X_selected parameter missing"
        assert 'y_data' in func_signature, "y_data parameter missing"
        assert 'config' in func_signature, "config parameter missing"
        # Optional parameters for model type and tuning
        assert 'model_type' in func_signature, "model_type parameter missing"
        assert 'tuning_params' in func_signature, "tuning_params parameter missing"
    
    def test_train_gemma_model_has_pytorch_import(self):
        """train_gemma_model should import PyTorch when needed"""
        train_all_models_path = Path(project_root) / 'scripts' / 'train_all_models.py'
        
        with open(train_all_models_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check for PyTorch import inside the function (lazy loading)
        func_start = source.find('def train_gemma_model(')
        next_func_start = source.find('\nasync def main(', func_start)
        func_body = source[func_start:next_func_start]
        
        # Function should lazily import the centralized trainer dependency
        assert 'from src.ml.model_trainer import RegimeModelTrainer' in func_body, \
            "RegimeModelTrainer lazy import missing from train_gemma_model"


class TestGemmaIntegration:
    """Test GEMMA integration with main training pipeline"""
    
    def test_gemma_metrics_in_training_metrics(self):
        """Training metrics should include gemma_models key"""
        # Read source file to verify integration
        train_all_models_path = Path(project_root) / 'scripts' / 'train_all_models.py'
        
        with open(train_all_models_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Verify GEMMA training is called
        assert 'train_gemma_model' in source, \
            "train_gemma_model should be called in train_all_models.py"
        assert "'gemma_models'" in source or '"gemma_models"' in source, \
            "gemma_models should be in training metrics"
        
        # Verify it's called in main function
        main_func_start = source.find('async def main():')
        assert main_func_start != -1, "main function not found"
        
        # Check that train_gemma_model is called after main function definition
        train_gemma_call = source.find('train_gemma_model(', main_func_start)
        assert train_gemma_call != -1, \
            "train_gemma_model should be called in main()"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
