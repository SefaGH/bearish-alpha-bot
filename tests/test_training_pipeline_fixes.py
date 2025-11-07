"""
Test Suite for Training Pipeline Bug Fixes (FAZ 1)

Tests for the three critical fixes:
1. Pandas FutureWarning fix (timeframe format)
2. Model Performance Tracker format error fix (metrics cleaning)
3. MarketDataPipeline warning fix (proper initialization)
"""

import pytest
import pandas as pd
import warnings
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.utils.model_performance_tracker import ModelPerformanceTracker


class TestPandasTimeframeFix:
    """Test that pandas timeframe formats use lowercase to avoid FutureWarning"""
    
    def test_lowercase_timeframes_no_warning(self):
        """Lowercase timeframes should not generate FutureWarning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test lowercase formats (correct)
            pd.Timedelta('1h')
            pd.Timedelta('4h')
            pd.Timedelta('1d')
            
            # Check for FutureWarning
            future_warnings = [warning for warning in w 
                             if issubclass(warning.category, FutureWarning)]
            
            assert len(future_warnings) == 0, \
                "Lowercase timeframes should not generate FutureWarning"
    
    def test_uppercase_timeframes_generate_warning(self):
        """Uppercase timeframes should generate FutureWarning in pandas 2.2+"""
        pandas_version = tuple(map(int, pd.__version__.split('.')[:2]))
        
        # Skip test if pandas version is too old
        if pandas_version < (2, 2):
            pytest.skip("FutureWarning only in pandas 2.2+")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test uppercase format (deprecated)
            pd.Timedelta('1H')
            
            # Check for FutureWarning
            future_warnings = [warning for warning in w 
                             if issubclass(warning.category, FutureWarning)]
            
            assert len(future_warnings) > 0, \
                "Uppercase 'H' should generate FutureWarning in pandas 2.2+"


class TestMetricsCleaningFix:
    """Test the _clean_metrics method in ModelPerformanceTracker"""
    
    @pytest.fixture
    def tracker(self):
        """Create a temporary tracker for testing"""
        temp_dir = tempfile.mkdtemp()
        tracker = ModelPerformanceTracker(performance_dir=temp_dir)
        yield tracker
        shutil.rmtree(temp_dir)
    
    def test_clean_string_numbers(self, tracker):
        """String numbers should be converted to float"""
        metrics = {
            'accuracy': '0.95',
            'loss': '0.123',
            'precision': '0.88'
        }
        
        cleaned = tracker._clean_metrics(metrics)
        
        assert isinstance(cleaned['accuracy'], float)
        assert cleaned['accuracy'] == 0.95
        assert isinstance(cleaned['loss'], float)
        assert cleaned['loss'] == 0.123
        assert isinstance(cleaned['precision'], float)
        assert cleaned['precision'] == 0.88
    
    def test_clean_mixed_types(self, tracker):
        """Mixed types should be handled correctly"""
        metrics = {
            'accuracy': 0.95,           # Already float
            'loss': '0.123',            # String to float
            'count': 100,               # Integer
            'name': 'model_test'        # Non-numeric string
        }
        
        cleaned = tracker._clean_metrics(metrics)
        
        assert isinstance(cleaned['accuracy'], float)
        assert isinstance(cleaned['loss'], float)
        assert isinstance(cleaned['count'], int)
        assert isinstance(cleaned['name'], str)
        assert cleaned['name'] == 'model_test'
    
    def test_clean_nested_dict(self, tracker):
        """Nested dictionaries should be cleaned recursively"""
        metrics = {
            'random_forest': {
                'accuracy': '0.95',
                'precision': '0.93'
            },
            'lstm': {
                'accuracy': '0.89',
                'loss': '0.234'
            }
        }
        
        cleaned = tracker._clean_metrics(metrics)
        
        assert isinstance(cleaned['random_forest']['accuracy'], float)
        assert cleaned['random_forest']['accuracy'] == 0.95
        assert isinstance(cleaned['lstm']['loss'], float)
        assert cleaned['lstm']['loss'] == 0.234
    
    def test_record_training_with_string_metrics(self, tracker):
        """record_training should handle string metrics without format errors"""
        # This simulates the bug scenario
        metrics = {
            'random_forest': {'accuracy': '0.95', 'precision': '0.93'},
            'lstm': {'accuracy': '0.89', 'loss': '0.234'}
        }
        
        # Should not raise ValueError: Unknown format code 'f' for object of type 'str'
        try:
            result = tracker.record_training(
                model_type='regime',
                model_name='BTC-USDT_ensemble',
                metrics=metrics,
                data_info={'samples': 1000},
                training_time=120.5
            )
            
            # Verify metrics were cleaned
            assert isinstance(result['metrics']['random_forest']['accuracy'], float)
            assert isinstance(result['metrics']['lstm']['loss'], float)
            
        except ValueError as e:
            if "Unknown format code 'f'" in str(e):
                pytest.fail(f"Format error not fixed: {e}")
            raise


class TestMarketDataPipelineFix:
    """Test that MarketDataPipeline can be properly initialized in training"""
    
    def test_market_data_pipeline_import(self):
        """MarketDataPipeline should be importable"""
        try:
            from src.core.market_data_pipeline import MarketDataPipeline
            assert MarketDataPipeline is not None
        except ImportError as e:
            # Skip test if dependencies are not installed
            if 'ccxt' in str(e) or 'No module' in str(e):
                pytest.skip(f"Dependencies not installed: {e}")
            pytest.fail(f"MarketDataPipeline import failed: {e}")
    
    def test_train_all_models_imports(self):
        """train_all_models.py should import MarketDataPipeline"""
        train_script_path = Path(project_root) / 'scripts' / 'train_all_models.py'
        
        if not train_script_path.exists():
            pytest.skip("train_all_models.py not found")
        
        with open(train_script_path, 'r') as f:
            content = f.read()
        
        assert 'from src.core.market_data_pipeline import MarketDataPipeline' in content, \
            "MarketDataPipeline import missing from train_all_models.py"


class TestDiagnoseScriptTimeframes:
    """Test that diagnose_training_data.py uses correct timeframe format"""
    
    def test_diagnose_script_uses_lowercase_timeframes(self):
        """diagnose_training_data.py should use lowercase timeframes"""
        diagnose_script_path = Path(project_root) / 'scripts' / 'diagnose_training_data.py'
        
        if not diagnose_script_path.exists():
            pytest.skip("diagnose_training_data.py not found")
        
        with open(diagnose_script_path, 'r') as f:
            content = f.read()
        
        # Check that uppercase timeframes are not used
        assert "'1H'" not in content, "Uppercase '1H' found - should be '1h'"
        assert "'4H'" not in content, "Uppercase '4H' found - should be '4h'"
        assert "'1D'" not in content, "Uppercase '1D' found - should be '1d'"
        
        # Check that lowercase timeframes are used
        assert "'1h'" in content, "Lowercase '1h' not found"
        assert "'4h'" in content, "Lowercase '4h' not found"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
