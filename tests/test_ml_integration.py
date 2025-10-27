"""
Integration tests for ML pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from src.ml.ml_context import MLContext
from src.ml.strategy_integration import MLStrategyIntegrationManager


class TestMLPipelineIntegration:
    """Integration tests for ML pipeline."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.rand(100) * 1000
        })
    
    @pytest.fixture
    def mock_regime_predictor(self):
        """Create mock regime predictor."""
        predictor = AsyncMock()
        predictor.predict_regime_transition = AsyncMock(return_value={
            'predicted_regime': 'bullish',
            'confidence': 0.75,
            'probabilities': {
                'bullish': 0.75,
                'neutral': 0.15,
                'bearish': 0.10
            },
            'quality_score': 0.8
        })
        return predictor
    
    @pytest.fixture
    def mock_price_engine(self):
        """Create mock price engine."""
        engine = Mock()
        engine.get_price_forecast = Mock(return_value={
            'aggregated': {
                'forecast': [0.03],  # 3% up
                'uncertainty': [0.02],
                'consensus_strength': 0.7
            }
        })
        engine.get_engine_status = Mock(return_value={'running': True})
        return engine
    
    @pytest.mark.asyncio
    async def test_ml_context_generation_healthy(
        self, 
        sample_price_data, 
        mock_regime_predictor, 
        mock_price_engine
    ):
        """Test that ML context is generated correctly with healthy data."""
        # Create integration manager
        manager = MLStrategyIntegrationManager(
            price_engine=mock_price_engine,
            regime_predictor=mock_regime_predictor
        )
        
        # Prepare market data
        market_data = {
            'price_data': sample_price_data,
            'timeframes': {'30m': sample_price_data}
        }
        
        # Generate ML context
        ml_context = await manager.get_ml_context(
            symbol='BTC/USDT',
            market_data=market_data
        )
        
        # Verify context is healthy
        assert ml_context is not None
        assert ml_context.is_healthy is True
        assert ml_context.symbol == 'BTC/USDT'
        
        # Verify regime prediction
        assert ml_context.regime_prediction == 'bullish'
        assert ml_context.regime_confidence == 0.75
        assert 'bullish' in ml_context.regime_probabilities
        
        # Verify price prediction
        assert ml_context.price_direction == 'up'
        assert ml_context.price_confidence > 0
        
        # Verify consensus
        assert ml_context.consensus_score > 0
        
    @pytest.mark.asyncio
    async def test_ml_context_with_empty_data(
        self, 
        mock_regime_predictor, 
        mock_price_engine
    ):
        """Test that ML context handles empty data gracefully."""
        manager = MLStrategyIntegrationManager(
            price_engine=mock_price_engine,
            regime_predictor=mock_regime_predictor
        )
        
        # Empty market data
        market_data = {
            'price_data': pd.DataFrame()  # Empty!
        }
        
        # Generate ML context
        ml_context = await manager.get_ml_context(
            symbol='BTC/USDT',
            market_data=market_data
        )
        
        # Should be unhealthy
        assert ml_context.is_healthy is False
        assert len(ml_context.validation_errors) > 0
        assert 'empty' in ' '.join(ml_context.validation_errors).lower()
    
    @pytest.mark.asyncio
    async def test_ml_context_with_nan_data(
        self, 
        sample_price_data,
        mock_regime_predictor, 
        mock_price_engine
    ):
        """Test that ML context detects NaN values."""
        manager = MLStrategyIntegrationManager(
            price_engine=mock_price_engine,
            regime_predictor=mock_regime_predictor
        )
        
        # Add NaN values
        bad_data = sample_price_data.copy()
        bad_data.loc[10:15, 'close'] = np.nan
        
        # Without indicator validator (fallback validation)
        market_data = {'price_data': bad_data}
        
        ml_context = await manager.get_ml_context(
            symbol='BTC/USDT',
            market_data=market_data
        )
        
        # Should detect NaN
        assert ml_context.is_healthy is False
        assert any('NaN' in error or 'nan' in error.lower() for error in ml_context.validation_errors)
    
    @pytest.mark.asyncio
    async def test_ml_context_insufficient_data(
        self,
        mock_regime_predictor, 
        mock_price_engine
    ):
        """Test that ML context detects insufficient data."""
        manager = MLStrategyIntegrationManager(
            price_engine=mock_price_engine,
            regime_predictor=mock_regime_predictor
        )
        
        # Only 10 rows (need at least 50)
        small_data = pd.DataFrame({
            'close': np.random.randn(10).cumsum() + 100,
            'volume': np.random.rand(10) * 1000,
            'high': np.random.randn(10).cumsum() + 102,
            'low': np.random.randn(10).cumsum() + 98,
            'open': np.random.randn(10).cumsum() + 100
        })
        
        market_data = {'price_data': small_data}
        
        ml_context = await manager.get_ml_context(
            symbol='BTC/USDT',
            market_data=market_data
        )
        
        # Should detect insufficient data
        assert ml_context.is_healthy is False
        assert any('Insufficient' in error for error in ml_context.validation_errors)
    
    def test_ml_context_health_methods(self):
        """Test MLContext health check methods."""
        # Healthy context with regime prediction
        context = MLContext(
            is_healthy=True,
            regime_prediction='bullish',
            regime_confidence=0.8
        )
        assert context.has_regime_prediction() is True
        
        # Healthy but low confidence
        context2 = MLContext(
            is_healthy=True,
            regime_prediction='bullish',
            regime_confidence=0.3
        )
        assert context2.has_regime_prediction() is False
        
        # Unhealthy
        context3 = MLContext(
            is_healthy=False,
            regime_prediction='bullish',
            regime_confidence=0.8
        )
        assert context3.has_regime_prediction() is False
    
    def test_ml_context_combined_signal(self):
        """Test combined signal generation."""
        # All bullish
        context = MLContext(
            is_healthy=True,
            regime_prediction='bullish',
            price_direction='up',
            rl_action_suggestion='buy',
            consensus_score=0.8
        )
        assert context.get_combined_signal() == 'bullish'
        
        # All bearish
        context2 = MLContext(
            is_healthy=True,
            regime_prediction='bearish',
            price_direction='down',
            rl_action_suggestion='sell',
            consensus_score=0.8
        )
        assert context2.get_combined_signal() == 'bearish'
        
        # Mixed signals, low consensus
        context3 = MLContext(
            is_healthy=True,
            regime_prediction='bullish',
            price_direction='down',
            consensus_score=0.3
        )
        assert context3.get_combined_signal() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
