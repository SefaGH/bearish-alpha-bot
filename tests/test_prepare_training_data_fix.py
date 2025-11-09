"""
Test Suite for ML Training Data Preparation Variable Name Fix

This test validates that the NameError bug in prepare_training_data.py is fixed.
The bug was: generate_regime_labels() was called with undefined variable 'price_data'
The fix: Changed to use 'ohlcv_df' which is the correct variable name.

Author: GitHub Copilot
Date: 2025-11-09
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


class TestPrepareTrainingDataVariableFix:
    """Test that the variable name fix in prepare_training_data.py works correctly"""
    
    @pytest.fixture
    def mock_ohlcv_data(self):
        """Create mock OHLCV data for testing"""
        # Create 200 candles of realistic price data
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(200) * 2)
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1h'),
            'open': close_prices + np.random.randn(200) * 0.5,
            'high': close_prices + np.abs(np.random.randn(200) * 1.5),
            'low': close_prices - np.abs(np.random.randn(200) * 1.5),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 200)
        })
    
    def test_label_generator_accepts_dataframe(self, mock_ohlcv_data):
        """Test that generate_regime_labels works with DataFrame (the correct type)"""
        from src.ml.label_generator import generate_regime_labels
        
        # This should work without errors
        labels = generate_regime_labels(
            mock_ohlcv_data,
            window=20,
            threshold=0.015,
            prediction_horizon=5,
            volume_confirm=True,
            multi_timeframe=True
        )
        
        # Verify output
        assert isinstance(labels, pd.Series)
        assert len(labels) > 0
        assert labels.name == "regime_labels"
        # Labels should be 0 (Bullish), 1 (Neutral), or 2 (Bearish)
        assert set(labels.unique()).issubset({0.0, 1.0, 2.0})
    
    def test_label_generator_with_minimal_data(self):
        """Test label generator with minimal valid data"""
        from src.ml.label_generator import generate_regime_labels
        
        # Create minimal data (50 candles)
        data = pd.DataFrame({
            'close': np.linspace(100, 110, 50),
            'volume': [1000] * 50
        })
        
        labels = generate_regime_labels(
            data,
            window=10,
            threshold=0.01,
            prediction_horizon=3,
            volume_confirm=True,
            multi_timeframe=False  # Disable multi-timeframe for minimal data
        )
        
        assert isinstance(labels, pd.Series)
        assert len(labels) > 0
    
    def test_prepare_training_script_syntax(self):
        """Test that prepare_training_data.py has no syntax errors"""
        script_path = os.path.join(project_root, 'scripts', 'prepare_training_data.py')
        
        assert os.path.exists(script_path), "prepare_training_data.py not found"
        
        # Try to compile the script to check for syntax errors
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        try:
            compile(script_content, script_path, 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in prepare_training_data.py: {e}")
    
    def test_variable_name_consistency(self):
        """Test that the variable name 'ohlcv_df' is used consistently"""
        script_path = os.path.join(project_root, 'scripts', 'prepare_training_data.py')
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check that the bug (using 'price_data' in wrong context) is fixed
        # The generate_regime_labels call should use 'ohlcv_df', not 'price_data'
        
        # Look for the generate_regime_labels call
        assert 'generate_regime_labels(' in content, "generate_regime_labels call not found"
        
        # Find the section where generate_regime_labels is called
        lines = content.split('\n')
        in_label_generation = False
        found_correct_variable = False
        found_incorrect_variable = False
        
        for i, line in enumerate(lines):
            if 'generate_regime_labels(' in line:
                in_label_generation = True
            
            if in_label_generation:
                # Check if we're passing the correct variable (ohlcv_df)
                if 'ohlcv_df,' in line or 'ohlcv_df)' in line:
                    # Check if it's the first parameter (not a keyword arg)
                    if i > 0 and 'generate_regime_labels(' in lines[i-1]:
                        found_correct_variable = True
                        break
                    # Or it's on the same line as the function call
                    elif 'generate_regime_labels(' in line:
                        found_correct_variable = True
                        break
                
                # Check for the incorrect variable (price_data) in this context
                if 'price_data,' in line and in_label_generation:
                    # Make sure it's not in a comment
                    if '#' not in line or line.index('price_data') < line.index('#'):
                        found_incorrect_variable = True
                        break
                
                # Stop searching after we exit the function call
                if ')' in line and in_label_generation:
                    break
        
        assert found_correct_variable, (
            "The variable 'ohlcv_df' should be passed to generate_regime_labels()"
        )
        assert not found_incorrect_variable, (
            "The incorrect variable 'price_data' is still being used. "
            "It should be 'ohlcv_df' instead."
        )
    
    @pytest.mark.asyncio
    async def test_fetch_and_process_data_mock(self, mock_ohlcv_data):
        """Test fetch_and_process_data function with mocked dependencies"""
        # Import the function
        from scripts.prepare_training_data import fetch_and_process_data
        
        # Mock the CcxtClient and FeatureEngineeringPipeline
        with patch('scripts.prepare_training_data.CcxtClient') as MockClient, \
             patch('scripts.prepare_training_data.FeatureEngineeringPipeline') as MockPipeline:
            
            # Set up mocks
            mock_client = MockClient.return_value
            mock_client.ohlcv = AsyncMock(return_value=mock_ohlcv_data)
            
            mock_pipeline = MockPipeline.return_value
            
            # Create mock features (87 features as expected)
            mock_features = pd.DataFrame(
                np.random.randn(len(mock_ohlcv_data), 87),
                columns=[f'feature_{i}' for i in range(87)]
            )
            mock_pipeline.extract_features.return_value = mock_features
            
            # Mock prepare_for_training to return aligned data
            mock_pipeline.prepare_for_training.return_value = (
                mock_features.values[:150],  # X
                np.random.randint(0, 3, 150)  # y (labels)
            )
            
            # This should work without NameError
            try:
                X, y = await fetch_and_process_data(
                    symbol='BTC/USDT',
                    timeframes=['15m'],  # Just test one timeframe
                    use_feature_selection=False,
                    use_all_features=True
                )
                
                # Verify output
                assert X is not None
                assert y is not None
                assert len(X) == len(y)
                assert X.shape[0] > 0
                
            except NameError as e:
                if 'price_data' in str(e):
                    pytest.fail(
                        f"NameError with 'price_data' still exists: {e}. "
                        f"The variable should be 'ohlcv_df'."
                    )
                raise


class TestLabelGeneratorParameters:
    """Test that label generator receives correct parameters"""
    
    def test_label_generator_signature(self):
        """Verify generate_regime_labels has correct signature"""
        from src.ml.label_generator import generate_regime_labels
        import inspect
        
        sig = inspect.signature(generate_regime_labels)
        params = list(sig.parameters.keys())
        
        # Verify expected parameters exist
        assert 'price_data' in params, "First parameter should be 'price_data'"
        assert 'window' in params
        assert 'threshold' in params
        assert 'prediction_horizon' in params
        assert 'volume_confirm' in params
        assert 'multi_timeframe' in params
    
    def test_label_generator_type_annotation(self):
        """Verify price_data parameter expects DataFrame"""
        from src.ml.label_generator import generate_regime_labels
        import inspect
        
        sig = inspect.signature(generate_regime_labels)
        price_data_param = sig.parameters['price_data']
        
        # Check type annotation
        assert price_data_param.annotation == pd.DataFrame, (
            "price_data parameter should be annotated as pd.DataFrame"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
