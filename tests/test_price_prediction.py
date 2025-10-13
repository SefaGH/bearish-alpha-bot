"""
Tests for Phase 4 Final: Advanced Price Prediction System

Tests price prediction models, multi-timeframe forecasting, ensemble predictions,
confidence intervals, and strategy integration.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import logging
from unittest.mock import Mock, patch

from src.ml.price_predictor import (
    LSTMPricePredictor,
    TransformerPricePredictor,
    EnsemblePricePredictor,
    MultiTimeframePricePredictor,
    AdvancedPricePredictionEngine
)
from src.ml.strategy_integration import (
    AIEnhancedStrategyAdapter,
    StrategyPerformanceTracker,
    MLStrategyIntegrationManager
)
from src.ml.regime_predictor import MLRegimePredictor
from src.ml.feature_engineering import FeatureEngineeringPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_price_data(n_bars=200, trend='upward'):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    
    # Create trending data
    if trend == 'upward':
        base_trend = np.linspace(100, 110, n_bars)
    elif trend == 'downward':
        base_trend = np.linspace(110, 100, n_bars)
    else:
        base_trend = np.ones(n_bars) * 100
    
    noise = np.random.randn(n_bars) * 0.5
    close = base_trend + noise
    
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_price = close + np.random.randn(n_bars) * 0.2
    volume = np.abs(np.random.randn(n_bars) * 1000 + 5000)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


class TestLSTMPricePredictor:
    """Test LSTM price prediction model."""
    
    def test_lstm_initialization(self):
        """Test LSTM model initialization."""
        logger.info("Testing LSTM price predictor initialization...")
        
        model = LSTMPricePredictor(
            input_size=50,
            hidden_size=128,
            num_layers=3,
            forecast_horizon=12
        )
        
        assert model is not None
        assert model.forecast_horizon == 12
        logger.info("✓ LSTM price predictor initialized successfully")
    
    def test_lstm_prediction(self):
        """Test LSTM price prediction."""
        logger.info("Testing LSTM price prediction...")
        
        model = LSTMPricePredictor(forecast_horizon=12)
        
        # Create sample input
        X = np.random.randn(2, 100, 50)  # batch_size=2, seq_len=100, features=50
        
        forecasts, uncertainties = model.predict(X)
        
        assert forecasts.shape[0] == 2  # batch size
        assert forecasts.shape[1] == 12  # forecast horizon
        assert uncertainties.shape == forecasts.shape
        logger.info(f"✓ LSTM prediction successful: {forecasts.shape}")


class TestTransformerPricePredictor:
    """Test Transformer price prediction model."""
    
    def test_transformer_initialization(self):
        """Test Transformer model initialization."""
        logger.info("Testing Transformer price predictor initialization...")
        
        model = TransformerPricePredictor(
            d_model=256,
            nhead=8,
            num_layers=6,
            forecast_horizon=12
        )
        
        assert model is not None
        assert model.forecast_horizon == 12
        logger.info("✓ Transformer price predictor initialized successfully")
    
    def test_transformer_prediction(self):
        """Test Transformer price prediction."""
        logger.info("Testing Transformer price prediction...")
        
        model = TransformerPricePredictor(d_model=256, forecast_horizon=12)
        
        # Create sample input
        X = np.random.randn(2, 100, 256)  # batch_size=2, seq_len=100, d_model=256
        
        forecasts, uncertainties = model.predict(X)
        
        assert forecasts.shape[0] == 2
        assert forecasts.shape[1] == 12
        assert uncertainties.shape == forecasts.shape
        logger.info(f"✓ Transformer prediction successful: {forecasts.shape}")


class TestEnsemblePricePredictor:
    """Test ensemble price prediction."""
    
    def test_ensemble_initialization(self):
        """Test ensemble predictor initialization."""
        logger.info("Testing ensemble price predictor initialization...")
        
        models = {
            'lstm': LSTMPricePredictor(forecast_horizon=12),
            'transformer': TransformerPricePredictor(forecast_horizon=12)
        }
        
        ensemble = EnsemblePricePredictor(models)
        
        assert ensemble is not None
        assert len(ensemble.models) == 2
        logger.info("✓ Ensemble predictor initialized successfully")
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        logger.info("Testing ensemble prediction...")
        
        models = {
            'lstm': LSTMPricePredictor(forecast_horizon=12),
            'transformer': TransformerPricePredictor(forecast_horizon=12)
        }
        
        ensemble = EnsemblePricePredictor(models)
        
        X = np.random.randn(2, 100, 50)
        forecasts, uncertainties = ensemble.predict(X)
        
        assert forecasts.shape[0] == 2
        assert forecasts.shape[1] == 12
        logger.info(f"✓ Ensemble prediction successful: {forecasts.shape}")


class TestMultiTimeframePricePredictor:
    """Test multi-timeframe price prediction."""
    
    def test_multi_timeframe_initialization(self):
        """Test multi-timeframe predictor initialization."""
        logger.info("Testing multi-timeframe predictor initialization...")
        
        # Create ensemble for each timeframe
        models = {}
        for tf in ['5m', '15m', '1h']:
            tf_models = {
                'lstm': LSTMPricePredictor(forecast_horizon=12),
                'transformer': TransformerPricePredictor(forecast_horizon=12)
            }
            models[tf] = EnsemblePricePredictor(tf_models)
        
        mt_predictor = MultiTimeframePricePredictor(models)
        
        assert mt_predictor is not None
        assert len(mt_predictor.models) == 3
        logger.info("✓ Multi-timeframe predictor initialized successfully")
    
    def test_multi_timeframe_prediction(self):
        """Test multi-timeframe prediction."""
        logger.info("Testing multi-timeframe prediction...")
        
        # Create models
        models = {}
        for tf in ['5m', '15m']:
            tf_models = {
                'lstm': LSTMPricePredictor(forecast_horizon=12),
                'transformer': TransformerPricePredictor(forecast_horizon=12)
            }
            models[tf] = EnsemblePricePredictor(tf_models)
        
        mt_predictor = MultiTimeframePricePredictor(models)
        
        # Create sample data for each timeframe
        data_by_tf = {
            '5m': create_sample_price_data(n_bars=200),
            '15m': create_sample_price_data(n_bars=100)
        }
        
        result = mt_predictor.predict_multi_timeframe(data_by_tf)
        
        assert 'by_timeframe' in result
        assert 'aggregated' in result
        assert '5m' in result['by_timeframe']
        assert 'forecast' in result['aggregated']
        logger.info("✓ Multi-timeframe prediction successful")
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        logger.info("Testing confidence interval calculation...")
        
        models = {}
        for tf in ['5m']:
            tf_models = {
                'lstm': LSTMPricePredictor(forecast_horizon=12)
            }
            models[tf] = EnsemblePricePredictor(tf_models)
        
        mt_predictor = MultiTimeframePricePredictor(models)
        
        data_by_tf = {'5m': create_sample_price_data(n_bars=200)}
        result = mt_predictor.predict_multi_timeframe(data_by_tf)
        
        # Check confidence intervals exist
        tf_result = result['by_timeframe']['5m']
        assert 'confidence_interval' in tf_result
        assert 'lower' in tf_result['confidence_interval']
        assert 'upper' in tf_result['confidence_interval']
        assert 'forecast' in tf_result['confidence_interval']
        
        # Verify bounds make sense
        ci = tf_result['confidence_interval']
        assert np.all(ci['lower'] <= ci['forecast'])
        assert np.all(ci['forecast'] <= ci['upper'])
        
        logger.info("✓ Confidence intervals calculated correctly")


class TestAdvancedPricePredictionEngine:
    """Test advanced price prediction engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        logger.info("Testing prediction engine initialization...")
        
        # Create predictor
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        
        engine = AdvancedPricePredictionEngine(mt_predictor)
        
        assert engine is not None
        assert engine.predictor is not None
        logger.info("✓ Prediction engine initialized successfully")
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self):
        """Test engine start and stop."""
        logger.info("Testing engine start/stop...")
        
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        engine = AdvancedPricePredictionEngine(mt_predictor)
        
        await engine.start_prediction_loop(['BTC/USDT'], ['5m'])
        assert engine.is_running is True
        
        await engine.stop_prediction_loop()
        assert engine.is_running is False
        
        logger.info("✓ Engine start/stop working correctly")
    
    def test_trading_signals(self):
        """Test trading signal generation."""
        logger.info("Testing trading signal generation...")
        
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        engine = AdvancedPricePredictionEngine(mt_predictor)
        
        # Mock a forecast in cache
        engine.prediction_cache['BTC/USDT'] = {
            'aggregated': {
                'forecast': np.array([2.0, 1.5, 1.0]),  # 2% up
                'uncertainty': np.array([0.5, 0.6, 0.7]),
                'consensus_strength': 0.8
            },
            'timestamp': pd.Timestamp.now()
        }
        
        signals = engine.generate_trading_signals('BTC/USDT', 100.0)
        
        assert signals is not None
        assert 'signal' in signals
        assert 'strength' in signals
        assert 'position_size' in signals
        logger.info(f"✓ Generated signal: {signals['signal']} with strength {signals['strength']:.2f}")


class TestAIEnhancedStrategyAdapter:
    """Test AI-enhanced strategy adapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        logger.info("Testing strategy adapter initialization...")
        
        # Create engines
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        price_engine = AdvancedPricePredictionEngine(mt_predictor)
        regime_predictor = MLRegimePredictor()
        
        adapter = AIEnhancedStrategyAdapter(price_engine, regime_predictor)
        
        assert adapter is not None
        assert adapter.price_engine is not None
        assert adapter.regime_predictor is not None
        logger.info("✓ Strategy adapter initialized successfully")
    
    @pytest.mark.asyncio
    async def test_signal_enhancement(self):
        """Test signal enhancement."""
        logger.info("Testing signal enhancement...")
        
        # Setup
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        price_engine = AdvancedPricePredictionEngine(mt_predictor)
        regime_predictor = MLRegimePredictor()
        
        adapter = AIEnhancedStrategyAdapter(price_engine, regime_predictor)
        
        # Base signal
        base_signal = {
            'signal': 'bullish',
            'strength': 0.7
        }
        
        # Enhance signal
        enhancement = await adapter.enhance_strategy_signal(
            'BTC/USDT', base_signal, 100.0
        )
        
        assert enhancement is not None
        assert 'original_signal' in enhancement
        assert 'final_signal' in enhancement
        logger.info(f"✓ Signal enhanced: {enhancement['original_signal']} -> {enhancement['final_signal']}")
    
    def test_position_sizing(self):
        """Test AI-adjusted position sizing."""
        logger.info("Testing position sizing...")
        
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        price_engine = AdvancedPricePredictionEngine(mt_predictor)
        regime_predictor = MLRegimePredictor()
        
        adapter = AIEnhancedStrategyAdapter(price_engine, regime_predictor)
        
        enhancement = {
            'confidence_adjustment': 0.8,
            'risk_adjustment': 0.9
        }
        
        sizing = adapter.calculate_position_sizing('BTC/USDT', 1.0, enhancement)
        
        assert 'base_position' in sizing
        assert 'adjusted_position' in sizing
        assert sizing['adjusted_position'] <= sizing['base_position'] * 1.5
        logger.info(f"✓ Position sized: {sizing['base_position']} -> {sizing['adjusted_position']:.2f}")


class TestStrategyPerformanceTracker:
    """Test strategy performance tracking."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        logger.info("Testing performance tracker initialization...")
        
        tracker = StrategyPerformanceTracker()
        
        assert tracker is not None
        assert tracker.metrics['total_trades'] == 0
        logger.info("✓ Performance tracker initialized successfully")
    
    def test_record_trade(self):
        """Test trade recording."""
        logger.info("Testing trade recording...")
        
        tracker = StrategyPerformanceTracker()
        
        trade = {
            'strategy_type': 'base',
            'pnl': 100.0,
            'symbol': 'BTC/USDT'
        }
        
        tracker.record_trade(trade)
        
        assert tracker.metrics['total_trades'] == 1
        logger.info("✓ Trade recorded successfully")
    
    def test_performance_summary(self):
        """Test performance summary."""
        logger.info("Testing performance summary...")
        
        tracker = StrategyPerformanceTracker()
        
        # Record some trades
        for i in range(5):
            tracker.record_trade({
                'strategy_type': 'base',
                'pnl': i * 10,
                'symbol': 'BTC/USDT'
            })
        
        for i in range(5):
            tracker.record_trade({
                'strategy_type': 'ai_enhanced',
                'pnl': i * 15,
                'symbol': 'BTC/USDT'
            })
        
        summary = tracker.get_performance_summary()
        
        assert summary['total_trades'] == 10
        logger.info(f"✓ Performance summary: {summary['total_trades']} trades")


class TestMLStrategyIntegrationManager:
    """Test ML strategy integration manager."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        logger.info("Testing integration manager initialization...")
        
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        price_engine = AdvancedPricePredictionEngine(mt_predictor)
        regime_predictor = MLRegimePredictor()
        
        manager = MLStrategyIntegrationManager(price_engine, regime_predictor)
        
        assert manager is not None
        assert manager.adapter is not None
        assert manager.tracker is not None
        logger.info("✓ Integration manager initialized successfully")
    
    @pytest.mark.asyncio
    async def test_process_strategy_signal(self):
        """Test complete signal processing."""
        logger.info("Testing complete signal processing...")
        
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        price_engine = AdvancedPricePredictionEngine(mt_predictor)
        regime_predictor = MLRegimePredictor()
        
        manager = MLStrategyIntegrationManager(price_engine, regime_predictor)
        
        base_signal = {
            'signal': 'bullish',
            'strength': 0.7
        }
        
        result = await manager.process_strategy_signal(
            'BTC/USDT', base_signal, 100.0, 1.0
        )
        
        assert result is not None
        assert 'enhancement' in result
        assert 'position_sizing' in result
        assert 'risk_metrics' in result
        logger.info("✓ Complete signal processing successful")
    
    def test_integration_status(self):
        """Test integration status reporting."""
        logger.info("Testing integration status...")
        
        models = {'5m': EnsemblePricePredictor({'lstm': LSTMPricePredictor()})}
        mt_predictor = MultiTimeframePricePredictor(models)
        price_engine = AdvancedPricePredictionEngine(mt_predictor)
        regime_predictor = MLRegimePredictor()
        
        manager = MLStrategyIntegrationManager(price_engine, regime_predictor)
        
        status = manager.get_integration_status()
        
        assert status is not None
        assert 'price_engine' in status
        assert 'performance' in status
        assert status['active'] is True
        logger.info(f"✓ Integration status: active={status['active']}")


class TestIntegration:
    """Integration tests for complete price prediction system."""
    
    @pytest.mark.asyncio
    async def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        logger.info("Testing full prediction pipeline...")
        
        # Create models
        models = {}
        for tf in ['5m', '15m']:
            tf_models = {
                'lstm': LSTMPricePredictor(forecast_horizon=12),
                'transformer': TransformerPricePredictor(forecast_horizon=12)
            }
            models[tf] = EnsemblePricePredictor(tf_models)
        
        mt_predictor = MultiTimeframePricePredictor(models)
        
        # Create data
        data_by_tf = {
            '5m': create_sample_price_data(n_bars=200, trend='upward'),
            '15m': create_sample_price_data(n_bars=100, trend='upward')
        }
        
        # Make prediction
        result = mt_predictor.predict_multi_timeframe(data_by_tf)
        
        # Verify results
        assert result is not None
        assert 'by_timeframe' in result
        assert 'aggregated' in result
        
        # Verify confidence intervals
        for tf, pred in result['by_timeframe'].items():
            assert 'confidence_interval' in pred
            ci = pred['confidence_interval']
            assert np.all(ci['lower'] <= ci['forecast'])
            assert np.all(ci['forecast'] <= ci['upper'])
        
        # Verify aggregated forecast
        agg = result['aggregated']
        assert 'forecast' in agg
        assert 'uncertainty' in agg
        assert 'consensus_strength' in agg
        assert 0.0 <= agg['consensus_strength'] <= 1.0
        
        logger.info("✓ Full prediction pipeline completed successfully")
        logger.info(f"   Consensus: {agg['consensus_strength']:.2f}")
        logger.info(f"   Forecast steps: {len(agg['forecast'])}")


def run_price_prediction_tests():
    """Run all price prediction tests."""
    logger.info("=" * 70)
    logger.info("Running Phase 4 Final: Advanced Price Prediction Tests")
    logger.info("=" * 70)
    
    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_price_prediction_tests()
