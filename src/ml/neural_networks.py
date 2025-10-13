"""
Neural Network Architectures for Market Regime Prediction.

Implements LSTM and Transformer-based models for regime forecasting.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Neural network models will use mock implementations.")


if TORCH_AVAILABLE:
    class MultiHeadAttention(nn.Module):
        """Multi-head attention mechanism for LSTM."""
        
        def __init__(self, hidden_size: int, num_heads: int = 8):
            """
            Initialize multi-head attention.
            
            Args:
                hidden_size: Size of hidden layer
                num_heads: Number of attention heads
            """
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            
        def forward(self, x):
            """Forward pass through attention mechanism."""
            attn_output, attn_weights = self.attention(x, x, x)
            return attn_output


    class LSTMRegimePredictor(nn.Module):
        """LSTM network for regime sequence prediction."""
        
        def __init__(self, input_size: int = 50, hidden_size: int = 128, 
                     num_layers: int = 3, num_classes: int = 3):
            """
            Initialize LSTM regime predictor.
            
            Args:
                input_size: Number of input features
                hidden_size: Size of LSTM hidden state
                num_layers: Number of LSTM layers
                num_classes: Number of regime classes
            """
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=0.2
            )
            self.attention = MultiHeadAttention(hidden_size, num_heads=8)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
            
        def forward(self, x):
            """
            Forward pass with attention mechanism.
            
            Args:
                x: Input tensor of shape (batch_size, sequence_length, input_size)
                
            Returns:
                Tuple of (predictions, confidence)
            """
            # LSTM feature extraction
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Attention-weighted sequence representation
            attn_out = self.attention(lstm_out)
            
            # Use last time step for classification
            last_hidden = attn_out[:, -1, :]
            
            # Classification output
            logits = self.classifier(last_hidden)
            probs = torch.softmax(logits, dim=1)
            
            return logits, probs


    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer."""
        
        def __init__(self, d_model: int, max_len: int = 5000):
            """
            Initialize positional encoding.
            
            Args:
                d_model: Model dimension
                max_len: Maximum sequence length
            """
            super().__init__()
            
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            """Add positional encoding to input."""
            return x + self.pe[:x.size(0)]


    class TransformerRegimePredictor(nn.Module):
        """Transformer architecture for regime prediction."""
        
        def __init__(self, d_model: int = 256, nhead: int = 8, 
                     num_layers: int = 6, num_classes: int = 3):
            """
            Initialize Transformer regime predictor.
            
            Args:
                d_model: Model dimension
                nhead: Number of attention heads
                num_layers: Number of transformer layers
                num_classes: Number of regime classes
            """
            super().__init__()
            self.d_model = d_model
            
            self.pos_encoding = PositionalEncoding(d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model*4,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
            
        def forward(self, x):
            """
            Transformer-based regime prediction.
            
            Args:
                x: Input tensor of shape (batch_size, sequence_length, d_model)
                
            Returns:
                Tuple of (predictions, confidence)
            """
            # Positional encoding
            x = self.pos_encoding(x)
            
            # Multi-head self-attention
            transformer_out = self.transformer(x)
            
            # Use last time step for prediction
            last_hidden = transformer_out[:, -1, :]
            
            # Feed-forward prediction
            logits = self.classifier(last_hidden)
            probs = torch.softmax(logits, dim=1)
            
            return logits, probs

else:
    # Mock implementations when PyTorch is not available
    class LSTMRegimePredictor:
        """Mock LSTM network for regime prediction (PyTorch not available)."""
        
        def __init__(self, input_size: int = 50, hidden_size: int = 128,
                     num_layers: int = 3, num_classes: int = 3):
            """Initialize mock LSTM predictor."""
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_classes = num_classes
            logger.info("Initialized mock LSTM predictor (PyTorch not available)")
        
        def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Mock prediction returning random probabilities."""
            batch_size = x.shape[0]
            probs = np.random.dirichlet(np.ones(self.num_classes), size=batch_size)
            predictions = np.argmax(probs, axis=1)
            return predictions, probs


    class TransformerRegimePredictor:
        """Mock Transformer network for regime prediction (PyTorch not available)."""
        
        def __init__(self, d_model: int = 256, nhead: int = 8,
                     num_layers: int = 6, num_classes: int = 3):
            """Initialize mock Transformer predictor."""
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.num_classes = num_classes
            logger.info("Initialized mock Transformer predictor (PyTorch not available)")
        
        def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Mock prediction returning random probabilities."""
            batch_size = x.shape[0]
            probs = np.random.dirichlet(np.ones(self.num_classes), size=batch_size)
            predictions = np.argmax(probs, axis=1)
            return predictions, probs
