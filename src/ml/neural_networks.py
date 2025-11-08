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
        """
        LSTM network for regime sequence prediction.
        
        OPTIMIZED ARCHITECTURE (FAZ 3.1):
        - input_size: 42 (number of features from feature engineering)
        - hidden_size: 128 (increased from 64 for more capacity)
        - num_layers: 3 (increased from 2 for deeper network)
        - dropout: 0.3 (increased from 0.2 for better regularization)
        - num_classes: 3 (bullish, neutral, bearish)
        
        These defaults MUST match the ML configuration file at:
          ml -> regime_prediction -> model_params -> lstm_regime
        
        NOTE: If config parameters change, this architecture must be updated
        and models must be retrained to avoid size mismatch errors.
        """
        
        def __init__(self, input_size: int = 42, hidden_size: int = 128, 
                     num_layers: int = 3, num_classes: int = 3, dropout: float = 0.3):
            """
            Initialize LSTM regime predictor.
            
            Args:
                input_size: Number of input features (default: 42)
                hidden_size: Size of LSTM hidden state (default: 128, increased from 64)
                num_layers: Number of LSTM layers (default: 3, increased from 2)
                num_classes: Number of regime classes (default: 3)
                dropout: Dropout rate (default: 0.3, increased from 0.2)
            """
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            self.attention = MultiHeadAttention(hidden_size, num_heads=8)
            
            # Enhanced classifier with batch normalization
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
            
        def forward(self, x, return_probs=False):
            """
            Forward pass with attention mechanism.
            
            Args:
                x: Input tensor of shape (batch_size, sequence_length, input_size)
                return_probs: If True, returns (logits, probs). If False, returns only logits.
                
            Returns:
                logits or Tuple of (logits, probs) depending on return_probs
            """
            # LSTM feature extraction
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Attention-weighted sequence representation
            attn_out = self.attention(lstm_out)
            
            # Use last time step for classification
            last_hidden = attn_out[:, -1, :]
            
            # Classification output
            logits = self.classifier(last_hidden)
            
            if return_probs:
                probs = torch.softmax(logits, dim=1)
                return logits, probs
            return logits


    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer."""
        
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            """
            Initialize positional encoding.
            
            Args:
                d_model: Model dimension
                dropout: Dropout rate (default: 0.1)
                max_len: Maximum sequence length
            """
            super().__init__()
            
            self.dropout = nn.Dropout(p=dropout)
            
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            """Add positional encoding to input."""
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)


    class TransformerRegimePredictor(nn.Module):
        """
        Transformer architecture for regime prediction.
        
        OPTIMIZED ARCHITECTURE (FAZ 3.1):
        - d_model: 256 (model dimension)
        - nhead: 6 (increased from 2/8 for better multi-head attention)
        - num_layers: 4 (increased from 2/6 for deeper network)
        - dim_feedforward: 256 (feedforward dimension)
        - dropout: 0.3 (for regularization)
        """
        
        def __init__(self, d_model: int = 256, nhead: int = 6, 
                     num_layers: int = 4, num_classes: int = 3,
                     dim_feedforward: int = 256, dropout: float = 0.3):
            """
            Initialize Transformer regime predictor.
            
            Args:
                d_model: Model dimension (must be divisible by nhead)
                nhead: Number of attention heads (default: 6, optimized for FAZ 3.1)
                num_layers: Number of transformer layers (default: 4, increased from 2)
                num_classes: Number of regime classes (default: 3)
                dim_feedforward: Feedforward dimension (default: 256)
                dropout: Dropout rate (default: 0.3)
            """
            super().__init__()
            self.d_model = d_model
            
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # Enhanced classifier with batch normalization
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.BatchNorm1d(d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )
            
        def forward(self, x, return_probs=False):
            """
            Transformer-based regime prediction.
            
            Args:
                x: Input tensor of shape (batch_size, sequence_length, d_model)
                return_probs: If True, returns (logits, probs). If False, returns only logits.
                
            Returns:
                logits or Tuple of (logits, probs) depending on return_probs
            """
            # Positional encoding
            x = self.pos_encoding(x)
            
            # Multi-head self-attention
            transformer_out = self.transformer(x)
            
            # Use last time step for prediction
            last_hidden = transformer_out[:, -1, :]
            
            # Feed-forward prediction
            logits = self.classifier(last_hidden)
            
            if return_probs:
                probs = torch.softmax(logits, dim=1)
                return logits, probs
            return logits

else:
    # Mock implementations when PyTorch is not available
    class LSTMRegimePredictor:
        """Mock LSTM network for regime prediction (PyTorch not available)."""
        
        def __init__(self, input_size: int = 42, hidden_size: int = 64,
                     num_layers: int = 2, num_classes: int = 3):
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
