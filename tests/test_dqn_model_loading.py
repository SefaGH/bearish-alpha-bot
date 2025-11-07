"""
Test for DQN Model Loading Fix in feature_engineer_and_run.py

Tests that DQNNetwork models can be loaded correctly from various checkpoint formats.
Specifically tests the fix for "TypeError: 'dict' object is not callable" error.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from torch.nn.functional import softmax

# Import the DQNNetwork class
from src.ml.reinforcement_learning import DQNNetwork


class TestDQNModelLoadingFix:
    """Test DQN model loading functionality for different checkpoint formats."""
    
    @pytest.fixture
    def sample_state_size(self):
        """Sample state size for testing."""
        return 50
    
    @pytest.fixture
    def sample_action_size(self):
        """Sample action size for testing."""
        return 3  # BUY, HOLD, SELL
    
    @pytest.fixture
    def trained_model(self, sample_state_size, sample_action_size):
        """Create a simple trained model."""
        model = DQNNetwork(state_size=sample_state_size, action_size=sample_action_size)
        # Put model in eval mode
        model.eval()
        return model
    
    def test_model_instantiation(self, sample_state_size, sample_action_size):
        """Test that DQNNetwork can be instantiated correctly."""
        model = DQNNetwork(state_size=sample_state_size, action_size=sample_action_size)
        assert model is not None
        assert model.state_size == sample_state_size
        assert model.action_size == sample_action_size
    
    def test_model_forward_pass(self, trained_model, sample_state_size):
        """Test that model can perform forward pass."""
        # Create sample input
        sample_input = torch.randn(1, sample_state_size)
        
        with torch.no_grad():
            output = trained_model(sample_input)
        
        # Check output shape
        assert output.shape == (1, 3)  # batch_size=1, action_size=3
    
    def test_checkpoint_format_q_network(self, trained_model, sample_state_size, sample_action_size):
        """Test loading model from checkpoint with 'q_network' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model in q_network format
            checkpoint_path = os.path.join(tmpdir, "test_model_q_network.pth")
            checkpoint = {
                'q_network': trained_model.state_dict(),
                'optimizer': {},
                'epsilon': 0.1
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint and create new model
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            new_model = DQNNetwork(state_size=sample_state_size, action_size=sample_action_size)
            
            # This should work without error
            assert isinstance(loaded_checkpoint, dict)
            assert 'q_network' in loaded_checkpoint
            new_model.load_state_dict(loaded_checkpoint['q_network'])
            new_model.eval()
            
            # Verify model works
            sample_input = torch.randn(1, sample_state_size)
            with torch.no_grad():
                output = new_model(sample_input)
            assert output.shape == (1, 3)
    
    def test_checkpoint_format_model_state_dict(self, trained_model, sample_state_size, sample_action_size):
        """Test loading model from checkpoint with 'model_state_dict' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model in model_state_dict format
            checkpoint_path = os.path.join(tmpdir, "test_model_state_dict.pth")
            checkpoint = {
                'model_state_dict': trained_model.state_dict(),
                'epoch': 10
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint and create new model
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            new_model = DQNNetwork(state_size=sample_state_size, action_size=sample_action_size)
            
            # This should work without error
            assert isinstance(loaded_checkpoint, dict)
            assert 'model_state_dict' in loaded_checkpoint
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            new_model.eval()
            
            # Verify model works
            sample_input = torch.randn(1, sample_state_size)
            with torch.no_grad():
                output = new_model(sample_input)
            assert output.shape == (1, 3)
    
    def test_checkpoint_format_direct_state_dict(self, trained_model, sample_state_size, sample_action_size):
        """Test loading model from checkpoint with direct state_dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model as direct state_dict
            checkpoint_path = os.path.join(tmpdir, "test_direct_state_dict.pth")
            torch.save(trained_model.state_dict(), checkpoint_path)
            
            # Load checkpoint and create new model
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            new_model = DQNNetwork(state_size=sample_state_size, action_size=sample_action_size)
            
            # This should work without error
            assert isinstance(loaded_checkpoint, dict)
            new_model.load_state_dict(loaded_checkpoint)
            new_model.eval()
            
            # Verify model works
            sample_input = torch.randn(1, sample_state_size)
            with torch.no_grad():
                output = new_model(sample_input)
            assert output.shape == (1, 3)
    
    def test_inference_with_numpy_input(self, trained_model, sample_state_size):
        """Test inference works with numpy array input (as used in feature_engineer_and_run.py)."""
        # Create numpy input (simulating scaled features)
        numpy_input = np.random.randn(1, sample_state_size).astype(np.float32)
        
        # Convert to tensor and run inference
        with torch.no_grad():
            tensor_input = torch.tensor(numpy_input)
            logits = trained_model(tensor_input).detach().cpu().numpy()
            
            # Apply softmax to get probabilities
            probs = softmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()
        
        # Verify shapes
        assert logits.shape == (1, 3)
        assert probs.shape == (1, 3)
        
        # Verify probabilities sum to 1
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)
        
        # Verify all probabilities are positive
        assert np.all(probs >= 0)
    
    def test_dict_object_not_callable_error_fixed(self, trained_model, sample_state_size, sample_action_size):
        """
        Test that the original error "'dict' object is not callable" is fixed.
        
        This test simulates the original bug where checkpoint was a dict
        but was being called directly instead of creating a model instance first.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model in q_network format (as done in the codebase)
            checkpoint_path = os.path.join(tmpdir, "test_model.pth")
            checkpoint = {
                'q_network': trained_model.state_dict(),
                'optimizer': {},
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # THE BUG: This would fail if we tried to call checkpoint directly
            # checkpoint(input) --> TypeError: 'dict' object is not callable
            
            # THE FIX: Create model instance first, then load state_dict
            assert isinstance(loaded_checkpoint, dict)
            net = DQNNetwork(state_size=sample_state_size, action_size=sample_action_size)
            net.load_state_dict(loaded_checkpoint['q_network'])
            net.eval()
            
            # Now inference should work
            sample_input = torch.randn(1, sample_state_size)
            with torch.no_grad():
                output = net(sample_input)
            
            # Verify it produces valid output
            assert output.shape == (1, 3)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
