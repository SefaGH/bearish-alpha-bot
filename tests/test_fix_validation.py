"""
Integration test to validate the fix for FeatureEngineeringPipeline initialization.

This test demonstrates that the bug described in the issue is fixed.
"""

import pytest
from src.ml.feature_engineering import FeatureEngineeringPipeline


def test_issue_fix_bracket_format():
    """
    Test that reproduces the original bug scenario from the issue.
    
    Original error:
    ValueError: invalid literal for int() with base 10: '[5'
    
    The environment variables from GitHub Actions were being passed 
    with array brackets like "[5,10,20,50]" instead of "5,10,20,50".
    """
    # Simulate environment variables with brackets (the problematic format)
    config = {
        'volatility_windows': '[5,10,20,50]',
        'momentum_windows': '[5,10,20,50]'
    }
    
    # This should NOT raise ValueError anymore
    pipeline = FeatureEngineeringPipeline(config)
    
    # Verify the pipeline initialized successfully
    assert pipeline is not None
    assert hasattr(pipeline, 'volatility_features')
    assert hasattr(pipeline, 'momentum_features')
    assert pipeline.volatility_features is not None
    assert pipeline.momentum_features is not None


def test_issue_fix_quoted_format():
    """
    Test that handles quoted array format like "['5','10','20','50']".
    
    This format can also come from environment variables depending on 
    how the workflow is configured.
    """
    config = {
        'volatility_windows': "['5','10','20','50']",
        'momentum_windows': "['5','10','20','50']"
    }
    
    # This should NOT raise ValueError anymore
    pipeline = FeatureEngineeringPipeline(config)
    
    # Verify the pipeline initialized successfully
    assert pipeline is not None
    assert hasattr(pipeline, 'volatility_features')
    assert hasattr(pipeline, 'momentum_features')


def test_backward_compatibility_plain_csv():
    """
    Test that the fix maintains backward compatibility with the original format.
    """
    config = {
        'volatility_windows': '5,10,20,50',
        'momentum_windows': '5,10,20,50'
    }
    
    # This should still work (backward compatibility)
    pipeline = FeatureEngineeringPipeline(config)
    
    assert pipeline is not None
    assert hasattr(pipeline, 'volatility_features')
    assert hasattr(pipeline, 'momentum_features')


def test_backward_compatibility_list_input():
    """
    Test that the fix maintains backward compatibility with list inputs.
    """
    config = {
        'volatility_windows': [5, 10, 20, 50],
        'momentum_windows': [5, 10, 20, 50]
    }
    
    # This should still work (backward compatibility)
    pipeline = FeatureEngineeringPipeline(config)
    
    assert pipeline is not None
    assert hasattr(pipeline, 'volatility_features')
    assert hasattr(pipeline, 'momentum_features')


def test_default_values_used_when_not_provided():
    """
    Test that default values are used when config doesn't specify windows.
    """
    config = {}  # No volatility_windows or momentum_windows specified
    
    # Should use default values and not crash
    pipeline = FeatureEngineeringPipeline(config)
    
    assert pipeline is not None
    assert hasattr(pipeline, 'volatility_features')
    assert hasattr(pipeline, 'momentum_features')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
