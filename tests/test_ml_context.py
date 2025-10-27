"""
Tests for MLContext data structure.
"""

import pytest
from datetime import datetime
from src.ml.ml_context import MLContext


class TestMLContext:
    """Test suite for MLContext data structure."""
    
    def test_ml_context_creation(self):
        """Test basic MLContext creation."""
        context = MLContext(
            symbol="BTC/USDT",
            is_healthy=True,
            regime_prediction="bullish",
            regime_confidence=0.85
        )
        
        assert context.symbol == "BTC/USDT"
        assert context.is_healthy is True
        assert context.regime_prediction == "bullish"
        assert context.regime_confidence == 0.85
    
    def test_ml_context_default_values(self):
        """Test MLContext default values."""
        context = MLContext()
        
        assert context.is_healthy is False
        assert context.regime_prediction is None
        assert context.regime_confidence == 0.0
        assert context.consensus_score == 0.0
        assert context.validation_errors == []
        assert isinstance(context.timestamp, datetime)
    
    def test_ml_context_confidence_clamping(self):
        """Test that confidence values are clamped to [0, 1]."""
        context = MLContext(
            regime_confidence=1.5,  # Over 1.0
            price_confidence=-0.2,  # Below 0.0
            consensus_score=0.5
        )
        
        # Should be clamped
        assert context.regime_confidence == 1.0
        assert context.price_confidence == 0.0
        assert context.consensus_score == 0.5
    
    def test_has_regime_prediction(self):
        """Test regime prediction availability check."""
        # Healthy with prediction
        context1 = MLContext(
            is_healthy=True,
            regime_prediction="bullish",
            regime_confidence=0.7
        )
        assert context1.has_regime_prediction() is True
        
        # Unhealthy
        context2 = MLContext(
            is_healthy=False,
            regime_prediction="bullish",
            regime_confidence=0.7
        )
        assert context2.has_regime_prediction() is False
        
        # Low confidence
        context3 = MLContext(
            is_healthy=True,
            regime_prediction="bullish",
            regime_confidence=0.3
        )
        assert context3.has_regime_prediction() is False
    
    def test_has_price_prediction(self):
        """Test price prediction availability check."""
        context = MLContext(
            is_healthy=True,
            price_direction="up",
            price_confidence=0.6
        )
        assert context.has_price_prediction() is True
        
        # Low confidence
        context2 = MLContext(
            is_healthy=True,
            price_direction="up",
            price_confidence=0.3
        )
        assert context2.has_price_prediction() is False
    
    def test_get_combined_signal_bullish(self):
        """Test combined signal for bullish consensus."""
        context = MLContext(
            is_healthy=True,
            regime_prediction="bullish",
            price_direction="up",
            rl_action_suggestion="buy",
            consensus_score=0.8
        )
        
        signal = context.get_combined_signal()
        assert signal == "bullish"
    
    def test_get_combined_signal_bearish(self):
        """Test combined signal for bearish consensus."""
        context = MLContext(
            is_healthy=True,
            regime_prediction="bearish",
            price_direction="down",
            rl_action_suggestion="sell",
            consensus_score=0.8
        )
        
        signal = context.get_combined_signal()
        assert signal == "bearish"
    
    def test_get_combined_signal_low_consensus(self):
        """Test that low consensus returns None."""
        context = MLContext(
            is_healthy=True,
            regime_prediction="bullish",
            price_direction="down",  # Conflicting
            consensus_score=0.3  # Low consensus
        )
        
        signal = context.get_combined_signal()
        assert signal is None
    
    def test_get_combined_signal_unhealthy(self):
        """Test that unhealthy context returns None."""
        context = MLContext(
            is_healthy=False,
            regime_prediction="bullish",
            consensus_score=0.9
        )
        
        signal = context.get_combined_signal()
        assert signal is None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = MLContext(
            symbol="ETH/USDT",
            is_healthy=True,
            regime_prediction="neutral",
            regime_confidence=0.6,
            price_direction="up",
            price_confidence=0.7,
            consensus_score=0.65
        )
        
        d = context.to_dict()
        
        assert d['is_healthy'] is True
        assert d['regime']['prediction'] == "neutral"
        assert d['regime']['confidence'] == 0.6
        assert d['price']['direction'] == "up"
        assert d['price']['confidence'] == 0.7
        assert d['consensus_score'] == 0.65
        assert d['symbol'] == "ETH/USDT"
    
    def test_repr(self):
        """Test string representation."""
        context = MLContext(
            is_healthy=True,
            regime_prediction="bullish",
            regime_confidence=0.85,
            price_direction="up",
            price_confidence=0.75,
            consensus_score=0.8,
            quality_score=0.7
        )
        
        repr_str = repr(context)
        
        assert "HEALTHY" in repr_str
        assert "bullish" in repr_str
        assert "up" in repr_str
        assert "0.80" in repr_str  # Consensus
        assert "0.70" in repr_str  # Quality
    
    def test_validation_errors(self):
        """Test validation errors are properly stored."""
        context = MLContext(
            is_healthy=False,
            validation_errors=["NaN values in close", "Insufficient data"]
        )
        
        assert len(context.validation_errors) == 2
        assert "NaN values in close" in context.validation_errors
        assert "Insufficient data" in context.validation_errors
    
    def test_metadata(self):
        """Test metadata storage."""
        context = MLContext(
            metadata={
                'model_version': 'v1.2',
                'feature_count': 50,
                'processing_time_ms': 125
            }
        )
        
        assert context.metadata['model_version'] == 'v1.2'
        assert context.metadata['feature_count'] == 50
        assert context.metadata['processing_time_ms'] == 125


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
