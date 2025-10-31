"""
Test for verifying prediction loop populates the cache correctly.
"""
import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from src.ml.price_predictor import (
    AdvancedPricePredictionEngine,
    MultiTimeframePricePredictor,
    EnsemblePricePredictor,
    LSTMPricePredictor
)


@pytest.mark.asyncio
async def test_prediction_loop_populates_cache():
    """Test that the prediction loop actually populates the cache."""
    # Create a simple model
    models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
    mt_predictor = MultiTimeframePricePredictor(models)
    
    # Mock websocket manager with collector
    mock_ws_manager = Mock()
    mock_collector = Mock()
    
    # Constants for test data
    MILLIS_PER_MINUTE = 60000
    BASE_PRICE = 100.0
    PRICE_INCREMENT = 0.1
    BASE_VOLUME = 1000
    NUM_CANDLES = 100
    
    # Create sample OHLCV data
    sample_data = [
        [int(pd.Timestamp.now().timestamp() * 1000) - i * MILLIS_PER_MINUTE, 
         BASE_PRICE + i * PRICE_INCREMENT, 
         BASE_PRICE + 1 + i * PRICE_INCREMENT, 
         BASE_PRICE - 1 + i * PRICE_INCREMENT, 
         BASE_PRICE + 0.5 + i * PRICE_INCREMENT, 
         BASE_VOLUME]
        for i in range(NUM_CANDLES)
    ]
    sample_data.reverse()  # Oldest first
    
    mock_collector.get_latest_ohlcv = Mock(return_value=sample_data)
    mock_ws_manager.collector = mock_collector
    
    # Create engine with mocked websocket manager
    engine = AdvancedPricePredictionEngine(mt_predictor, websocket_manager=mock_ws_manager)
    
    # Verify cache is empty initially
    assert len(engine.prediction_cache) == 0
    
    # Start prediction loop in background
    symbols = ['BTC/USDT:USDT']
    timeframes = ['5m']
    
    # Create task and run for a short time
    task = asyncio.create_task(engine.start_prediction_loop(symbols, timeframes))
    
    # Wait for one update cycle plus a bit
    await asyncio.sleep(2)
    
    # Stop the loop
    await engine.stop_prediction_loop()
    
    # Cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Verify cache was populated
    assert len(engine.prediction_cache) > 0
    assert 'BTC/USDT:USDT' in engine.prediction_cache
    
    # Verify prediction structure
    prediction = engine.prediction_cache['BTC/USDT:USDT']
    assert 'aggregated' in prediction
    assert 'by_timeframe' in prediction
    assert 'timestamp' in prediction
    
    print("✓ Prediction loop successfully populated cache")


@pytest.mark.asyncio
async def test_get_price_forecast_returns_cached_prediction():
    """Test that get_price_forecast returns the cached prediction."""
    models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
    mt_predictor = MultiTimeframePricePredictor(models)
    engine = AdvancedPricePredictionEngine(mt_predictor)
    
    # Mock a prediction in the cache
    engine.prediction_cache['BTC/USDT'] = {
        'aggregated': {
            'forecast': np.array([1.0, 1.5, 2.0]),
            'uncertainty': np.array([0.5, 0.5, 0.5]),
            'consensus_strength': 0.8
        },
        'by_timeframe': {},
        'timestamp': pd.Timestamp.now()
    }
    
    # Get forecast
    forecast = engine.get_price_forecast('BTC/USDT')
    
    # Verify it returns the cached prediction
    assert forecast is not None
    assert 'aggregated' in forecast
    assert forecast['aggregated']['consensus_strength'] == 0.8
    
    print("✓ get_price_forecast correctly returns cached prediction")


@pytest.mark.asyncio
async def test_get_price_forecast_returns_none_when_cache_empty():
    """Test that get_price_forecast returns None when cache is empty."""
    models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
    mt_predictor = MultiTimeframePricePredictor(models)
    engine = AdvancedPricePredictionEngine(mt_predictor)
    
    # Get forecast for symbol not in cache
    forecast = engine.get_price_forecast('BTC/USDT')
    
    # Verify it returns None
    assert forecast is None
    
    print("✓ get_price_forecast correctly returns None when cache is empty")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
