"""
Test for GEMMA CircuitBreaker 'enabled' parameter fix
This test verifies that the adapter correctly handles the 'enabled' parameter in circuit_breaker config.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import joblib
from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter


@pytest.fixture
def mock_gemma_components():
    """Mock all GEMMA components needed for initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a real TorchScript model
        model_path = tmpdir / "model.pt"
        simple_model = torch.nn.Linear(82, 3)
        traced_model = torch.jit.script(simple_model)
        torch.jit.save(traced_model, str(model_path))
        
        # Create a real scaler
        scaler_path = tmpdir / "scaler.joblib"
        from sklearn.preprocessing import StandardScaler
        real_scaler = StandardScaler()
        real_scaler.fit(np.random.rand(100, 82))
        joblib.dump(real_scaler, scaler_path)
        
        # Create features file
        features_path = tmpdir / "features.json"
        features = [f"feature_{i}" for i in range(82)]
        with open(features_path, 'w') as f:
            json.dump({'features': features}, f)
        
        yield {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'features_path': str(features_path),
            'features': features
        }


class TestCircuitBreakerEnabledParameter:
    """Test suite for circuit breaker 'enabled' parameter handling."""

    def test_circuit_breaker_enabled_true(self, mock_gemma_components):
        """Test circuit breaker is initialized when enabled=true."""
        config = {
            'model_path': mock_gemma_components['model_path'],
            'scaler_path': mock_gemma_components['scaler_path'],
            'features_path': mock_gemma_components['features_path'],
            'circuit_breaker': {
                'enabled': True,
                'failure_threshold': 5,
                'recovery_timeout': 60
            }
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Verify circuit breaker is enabled and initialized
        assert adapter.circuit_breaker_enabled is True
        assert adapter.circuit_breaker is not None
        assert adapter.circuit_breaker.state == "CLOSED"
        assert adapter.circuit_breaker.failure_threshold == 5
        assert adapter.circuit_breaker.recovery_timeout == 60

    def test_circuit_breaker_enabled_false(self, mock_gemma_components):
        """Test circuit breaker is None when enabled=false."""
        config = {
            'model_path': mock_gemma_components['model_path'],
            'scaler_path': mock_gemma_components['scaler_path'],
            'features_path': mock_gemma_components['features_path'],
            'circuit_breaker': {
                'enabled': False,
                'failure_threshold': 5,
                'recovery_timeout': 60
            }
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Verify circuit breaker is disabled
        assert adapter.circuit_breaker_enabled is False
        assert adapter.circuit_breaker is None

    def test_circuit_breaker_default_enabled_true(self, mock_gemma_components):
        """Test circuit breaker defaults to enabled when 'enabled' is not specified."""
        config = {
            'model_path': mock_gemma_components['model_path'],
            'scaler_path': mock_gemma_components['scaler_path'],
            'features_path': mock_gemma_components['features_path'],
            'circuit_breaker': {
                'failure_threshold': 5,
                'recovery_timeout': 60
            }
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Verify circuit breaker defaults to enabled
        assert adapter.circuit_breaker_enabled is True
        assert adapter.circuit_breaker is not None

    def test_circuit_breaker_no_config(self, mock_gemma_components):
        """Test circuit breaker uses defaults when no config provided."""
        config = {
            'model_path': mock_gemma_components['model_path'],
            'scaler_path': mock_gemma_components['scaler_path'],
            'features_path': mock_gemma_components['features_path']
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Verify circuit breaker defaults to enabled with default parameters
        assert adapter.circuit_breaker_enabled is True
        assert adapter.circuit_breaker is not None
        assert adapter.circuit_breaker.failure_threshold == 5
        assert adapter.circuit_breaker.recovery_timeout == 60

    def test_get_metrics_with_circuit_breaker_enabled(self, mock_gemma_components):
        """Test get_metrics returns correct circuit breaker state when enabled."""
        config = {
            'model_path': mock_gemma_components['model_path'],
            'scaler_path': mock_gemma_components['scaler_path'],
            'features_path': mock_gemma_components['features_path'],
            'circuit_breaker': {
                'enabled': True,
                'failure_threshold': 5,
                'recovery_timeout': 60
            }
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        metrics = adapter.get_metrics()
        
        assert metrics['circuit_breaker_enabled'] is True
        assert metrics['circuit_state'] == 'CLOSED'

    def test_get_metrics_with_circuit_breaker_disabled(self, mock_gemma_components):
        """Test get_metrics returns correct circuit breaker state when disabled."""
        config = {
            'model_path': mock_gemma_components['model_path'],
            'scaler_path': mock_gemma_components['scaler_path'],
            'features_path': mock_gemma_components['features_path'],
            'circuit_breaker': {
                'enabled': False,
                'failure_threshold': 5,
                'recovery_timeout': 60
            }
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        metrics = adapter.get_metrics()
        
        assert metrics['circuit_breaker_enabled'] is False
        assert metrics['circuit_state'] == 'DISABLED'

    def test_predict_with_circuit_breaker_disabled(self, mock_gemma_components):
        """Test predict works correctly when circuit breaker is disabled."""
        config = {
            'model_path': mock_gemma_components['model_path'],
            'scaler_path': mock_gemma_components['scaler_path'],
            'features_path': mock_gemma_components['features_path'],
            'circuit_breaker': {
                'enabled': False
            }
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Create feature dict
        features_dict = {f"feature_{i}": float(i) for i in range(82)}
        
        # Should work without using circuit breaker
        result = adapter.predict(features_dict)
        
        assert result is not None
        assert 'prediction' in result
        assert result['fallback'] is False

    def test_predict_with_circuit_breaker_enabled(self, mock_gemma_components):
        """Test predict uses circuit breaker when enabled."""
        config = {
            'model_path': mock_gemma_components['model_path'],
            'scaler_path': mock_gemma_components['scaler_path'],
            'features_path': mock_gemma_components['features_path'],
            'circuit_breaker': {
                'enabled': True,
                'failure_threshold': 3,
                'recovery_timeout': 60
            }
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Create feature dict
        features_dict = {f"feature_{i}": float(i) for i in range(82)}
        
        # Should work with circuit breaker
        result = adapter.predict(features_dict)
        
        assert result is not None
        assert 'prediction' in result
        assert result['fallback'] is False
        assert adapter.circuit_breaker.state == "CLOSED"
