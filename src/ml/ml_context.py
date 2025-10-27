"""
ML Context - Data Contract for ML Predictions.

This module defines the standard data structure for passing ML predictions
and insights to trading strategies. It ensures type safety and provides
a clear contract between ML components and strategy execution.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from datetime import datetime


@dataclass
class MLContext:
    """
    ML Context encapsulates all ML predictions and insights for a trading decision.
    
    This is the data contract between ML components and trading strategies.
    Strategies can use this information to enhance their decision-making.
    
    Attributes:
        is_healthy: Whether ML data is reliable and safe to use
        regime_prediction: Market regime ('bullish', 'bearish', 'neutral')
        regime_confidence: Confidence in regime prediction (0.0-1.0)
        price_forecast: Price prediction data with uncertainties
        rl_action_suggestion: RL agent's recommended action
        consensus_score: Agreement level between different ML models (0.0-1.0)
        validation_errors: List of validation issues if any
        timestamp: When this context was created
        symbol: Trading symbol this context is for
    """
    
    # Health and Validation
    is_healthy: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Regime Prediction
    regime_prediction: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    regime_confidence: float = 0.0  # 0.0 to 1.0
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Price Prediction
    price_forecast: Optional[Dict[str, Any]] = field(default_factory=dict)
    price_direction: Optional[str] = None  # 'up', 'down', 'neutral'
    price_confidence: float = 0.0
    
    # RL Agent
    rl_action_suggestion: Optional[str] = None  # 'buy', 'sell', 'hold'
    rl_confidence: float = 0.0
    
    # Consensus and Quality
    consensus_score: float = 0.0  # How much models agree (0.0-1.0)
    quality_score: float = 0.0   # Overall prediction quality
    uncertainty: float = 1.0      # Prediction uncertainty (higher = less certain)
    
    # Metadata
    timestamp: Optional[datetime] = None
    symbol: Optional[str] = None
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize values after initialization."""
        # Ensure timestamp is set
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Clamp confidence and consensus values to [0, 1]
        self.regime_confidence = max(0.0, min(1.0, self.regime_confidence))
        self.price_confidence = max(0.0, min(1.0, self.price_confidence))
        self.rl_confidence = max(0.0, min(1.0, self.rl_confidence))
        self.consensus_score = max(0.0, min(1.0, self.consensus_score))
        self.quality_score = max(0.0, min(1.0, self.quality_score))
        self.uncertainty = max(0.0, self.uncertainty)
    
    def has_regime_prediction(self) -> bool:
        """Check if regime prediction is available and confident."""
        return (
            self.is_healthy 
            and self.regime_prediction is not None 
            and self.regime_confidence > 0.5
        )
    
    def has_price_prediction(self) -> bool:
        """Check if price prediction is available and confident."""
        return (
            self.is_healthy 
            and self.price_direction is not None 
            and self.price_confidence > 0.5
        )
    
    def has_rl_suggestion(self) -> bool:
        """Check if RL suggestion is available and confident."""
        return (
            self.is_healthy 
            and self.rl_action_suggestion is not None 
            and self.rl_confidence > 0.5
        )
    
    def get_combined_signal(self) -> Optional[str]:
        """
        Get combined signal from all ML components.
        
        Returns consensus if models agree, None if conflicting or low confidence.
        """
        if not self.is_healthy or self.consensus_score < 0.6:
            return None
        
        # Map predictions to numeric values for comparison
        direction_map = {
            'bullish': 1, 'up': 1, 'buy': 1,
            'neutral': 0, 'hold': 0,
            'bearish': -1, 'down': -1, 'sell': -1
        }
        
        signals = []
        if self.regime_prediction:
            signals.append(direction_map.get(self.regime_prediction, 0))
        if self.price_direction:
            signals.append(direction_map.get(self.price_direction, 0))
        if self.rl_action_suggestion:
            signals.append(direction_map.get(self.rl_action_suggestion, 0))
        
        if not signals:
            return None
        
        # Check for consensus
        avg_signal = sum(signals) / len(signals)
        
        if avg_signal > 0.5:
            return 'bullish'
        elif avg_signal < -0.5:
            return 'bearish'
        else:
            return 'neutral'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'is_healthy': self.is_healthy,
            'validation_errors': self.validation_errors,
            'regime': {
                'prediction': self.regime_prediction,
                'confidence': self.regime_confidence,
                'probabilities': self.regime_probabilities
            },
            'price': {
                'direction': self.price_direction,
                'confidence': self.price_confidence,
                'forecast': self.price_forecast
            },
            'rl': {
                'action': self.rl_action_suggestion,
                'confidence': self.rl_confidence
            },
            'consensus_score': self.consensus_score,
            'quality_score': self.quality_score,
            'uncertainty': self.uncertainty,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "✓ HEALTHY" if self.is_healthy else "✗ UNHEALTHY"
        regime = f"{self.regime_prediction}({self.regime_confidence:.2f})" if self.regime_prediction else "N/A"
        price = f"{self.price_direction}({self.price_confidence:.2f})" if self.price_direction else "N/A"
        
        return (
            f"MLContext[{status}] "
            f"Regime:{regime} Price:{price} "
            f"Consensus:{self.consensus_score:.2f} "
            f"Quality:{self.quality_score:.2f}"
        )
