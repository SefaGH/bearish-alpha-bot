"""
Tests for GEMMA TorchScript Adapter
Tests cover circuit breaker, caching, feature alignment, and fallback behavior.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import joblib
from src.ml.adapters.gemma.gemma_torchscript_adapter import (
    GemmaTorchScriptAdapter,
    CircuitBreaker
)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_circuit_breaker_stays_closed_on_success(self):
        """Test circuit breaker stays closed when function succeeds."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        
        def successful_func():
            return "success"
        
        result = cb.call(successful_func)
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Fail 3 times to reach threshold
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == "OPEN"
        assert cb.failure_count == 3

    def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when OPEN."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        # Should reject next call
        with pytest.raises(RuntimeError, match="Circuit is open"):
            cb.call(failing_func)

    def test_circuit_breaker_recovers_to_half_open(self):
        """Test circuit breaker transitions to HALF_OPEN after timeout."""
        import time
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)  # 1 second timeout
        
        def failing_func():
            raise ValueError("Test error")
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Next call should transition to HALF_OPEN (but still fail)
        with pytest.raises(ValueError):
            cb.call(failing_func)
        
        # After the failed call in HALF_OPEN, it should go back to OPEN
        assert cb.state == "OPEN"

    def test_circuit_breaker_closes_from_half_open_on_success(self):
        """Test circuit breaker closes from HALF_OPEN on successful call."""
        import time
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        def failing_func():
            raise ValueError("Test error")
        
        def success_func():
            return "recovered"
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Successful call should close the circuit
        result = cb.call(success_func)
        assert result == "recovered"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0


class TestGemmaTorchScriptAdapter:
    """Test suite for GemmaTorchScriptAdapter class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock TorchScript model."""
        model = MagicMock()
        model.eval = Mock()
        # Mock the model to return a tensor output
        output_tensor = torch.tensor([[0.2, 0.3, 0.5]])  # Bullish prediction
        model.return_value = output_tensor
        return model

    @pytest.fixture
    def mock_scaler(self):
        """Create a mock StandardScaler."""
        scaler = Mock()
        scaler.transform = Mock(return_value=np.random.rand(1, 82))
        return scaler

    @pytest.fixture
    def feature_list(self):
        """Create a sample feature list (82 features)."""
        return [f"feature_{i}" for i in range(82)]

    @pytest.fixture
    def temp_files(self, mock_model, mock_scaler, feature_list):
        """Create temporary files for model, scaler, and features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create model file
            model_path = tmpdir_path / "model.pt"
            # Create a simple TorchScript model
            simple_model = torch.nn.Linear(82, 3)
            traced_model = torch.jit.script(simple_model)
            torch.jit.save(traced_model, str(model_path))
            
            # Create scaler file
            scaler_path = tmpdir_path / "scaler.joblib"
            from sklearn.preprocessing import StandardScaler
            real_scaler = StandardScaler()
            real_scaler.fit(np.random.rand(100, 82))
            joblib.dump(real_scaler, scaler_path)
            
            # Create features file
            features_path = tmpdir_path / "features.json"
            features_data = {
                "repository": "test",
                "version": "1.0.0",
                "type": "price_prediction",
                "count": 82,
                "features": feature_list
            }
            with open(features_path, 'w') as f:
                json.dump(features_data, f)
            
            yield {
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'features_path': str(features_path),
                'dir': tmpdir_path
            }

    def test_adapter_initialization_success(self, temp_files):
        """Test successful adapter initialization."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path'],
            'cache_ttl': 30,
            'shadow_mode': False
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        assert adapter.model is not None
        assert adapter.scaler is not None
        assert len(adapter.features) == 82
        assert adapter.cache_ttl == 30
        assert adapter.shadow_mode is False

    def test_adapter_initialization_missing_model(self, temp_files):
        """Test adapter handles missing model file."""
        config = {
            'model_path': '/nonexistent/model.pt',
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path']
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Model should be None when loading fails
        assert adapter.model is None

    def test_prediction_with_valid_features(self, temp_files, feature_list):
        """Test prediction with valid feature dictionary."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path'],
            'cache_ttl': 30
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Create feature dictionary with all required features
        features_dict = {feat: np.random.rand() for feat in feature_list}
        
        result = adapter.predict(features_dict)
        
        assert 'price_confidence' in result
        assert 'prediction' in result
        assert 'prediction_label' in result
        assert 'probabilities' in result
        assert 'timestamp' in result
        assert 'fallback' in result
        assert 'inference_time_ms' in result
        
        assert result['prediction'] in [0, 1, 2]
        assert result['prediction_label'] in ['bearish', 'neutral', 'bullish']
        assert result['fallback'] is False

    def test_prediction_caching(self, temp_files, feature_list):
        """Test that predictions are cached."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path'],
            'cache_ttl': 30
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        features_dict = {feat: 0.5 for feat in feature_list}
        
        # First prediction
        result1 = adapter.predict(features_dict)
        
        # Second prediction with same features
        result2 = adapter.predict(features_dict)
        
        # Should be cached (same timestamp)
        assert result1['timestamp'] == result2['timestamp']
        assert len(adapter.prediction_cache) > 0

    def test_fallback_prediction_on_error(self, temp_files, feature_list):
        """Test fallback prediction is returned on error."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path']
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Set model to None to force error
        adapter.model = None
        
        features_dict = {feat: 0.5 for feat in feature_list}
        result = adapter.predict(features_dict)
        
        assert result['fallback'] is True
        assert result['prediction'] == 1  # Neutral
        assert result['prediction_label'] == 'neutral'
        assert result['price_confidence'] == 0.5

    def test_feature_alignment(self, temp_files, feature_list):
        """Test feature alignment with missing features."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path']
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        # Provide only half the features
        partial_features = {feat: np.random.rand() for feat in feature_list[:41]}
        
        aligned = adapter._align_features(partial_features)
        
        assert len(aligned) == 82
        # Missing features should be 0.0
        assert aligned[41] == 0.0

    def test_get_metrics(self, temp_files):
        """Test metrics retrieval."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path']
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        metrics = adapter.get_metrics()
        
        assert 'model_loaded' in metrics
        assert 'circuit_state' in metrics
        assert 'cache_size' in metrics
        assert 'avg_inference_time_ms' in metrics
        assert 'p95_inference_time_ms' in metrics
        assert 'shadow_log_size' in metrics
        
        assert metrics['model_loaded'] is True
        assert metrics['circuit_state'] == 'CLOSED'

    def test_shadow_mode_logging(self, temp_files, feature_list):
        """Test shadow mode prediction logging."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path'],
            'shadow_mode': True
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        features_dict = {feat: 0.5 for feat in feature_list}
        
        adapter.predict(features_dict)
        
        assert len(adapter.shadow_predictions) > 0
        shadow_pred = adapter.shadow_predictions[0]
        assert 'timestamp' in shadow_pred
        assert 'features_hash' in shadow_pred
        assert 'prediction' in shadow_pred

    def test_cache_key_generation(self, temp_files, feature_list):
        """Test cache key generation is deterministic."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path']
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        features1 = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        features2 = {'c': 3.0, 'a': 1.0, 'b': 2.0}  # Different order
        
        key1 = adapter._get_cache_key(features1)
        key2 = adapter._get_cache_key(features2)
        
        # Should be the same despite different order
        assert key1 == key2

    def test_inference_time_tracking(self, temp_files, feature_list):
        """Test inference time is tracked."""
        config = {
            'model_path': temp_files['model_path'],
            'scaler_path': temp_files['scaler_path'],
            'features_path': temp_files['features_path']
        }
        
        adapter = GemmaTorchScriptAdapter(config)
        
        features_dict = {feat: 0.5 for feat in feature_list}
        
        result = adapter.predict(features_dict)
        
        assert len(adapter.inference_times) > 0
        assert result['inference_time_ms'] > 0
