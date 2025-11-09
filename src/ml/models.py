"""
Shared ML Model Definitions
Production-ready model architectures
"""

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    Simple LSTM for regime classification
    
    Production-safe, self-contained model definition.
    Used by both training and inference pipelines.
    
    Args:
        input_size: Number of input features (82)
        hidden_size: LSTM hidden dimension (64)
        num_layers: Number of LSTM layers (1)
        num_classes: Number of output classes (3)
        dropout: Dropout rate (0.6)
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, dropout: float):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, features) or (batch_size, seq_len, features)
        
        Returns:
            Output logits, shape (batch_size, num_classes)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # (batch_size, features) -> (batch_size, 1, features)
            x = x.unsqueeze(1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep output
        out = self.fc(lstm_out[:, -1, :])
        
        return out
    
    def get_config(self):
        """Get model configuration for serialization"""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
        }


def create_model_from_params(params: dict) -> SimpleLSTM:
    """
    Factory function to create model from hyperparameters
    
    Args:
        params: Dictionary with model hyperparameters
    
    Returns:
        Initialized SimpleLSTM model
    """
    return SimpleLSTM(
        input_size=params['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        dropout=params.get('dropout', 0.5)
    )
