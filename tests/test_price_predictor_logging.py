"""
Test suite for PricePredictor logging improvements.

This test validates that:
1. Logs clearly indicate ML vs FALLBACK mode
2. Status summary shows correct model status
3. No misleading logs in fallback mode
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from src.ml.price_predictor import AdvancedPricePredictionEngine


class TestPricePredictorLogging:
    """Test logging improvements in PricePredictor."""
    
    @pytest.fixture
    def mock_market_data_pipeline(self):
        """Create mock market data pipeline."""
        pipeline = Mock()
        pipeline.get_latest_ohlcv = Mock(return_value=None)
        return pipeline
    
    @pytest.fixture
    def mock_feature_pipeline(self):
        """Create mock feature engineering pipeline."""
        pipeline = Mock()
        return pipeline
    
    @pytest.fixture
    def basic_config(self):
        """Create basic ML configuration."""
        return {
            'prediction': {
                'timeframes': ['5m', '15m', '1h'],
                'update_interval_seconds': 60,
                'cache_ttl_seconds': 300
            },
            'models': ['lstm'],
            'model_params': {
                'lstm': {
                    'hidden_size': 64,
                    'num_layers': 2
                }
            },
            'feature_size': 42,
            'forecast_horizon': 12
        }
    
    def test_status_summary_with_no_models(self, mock_market_data_pipeline, 
                                           mock_feature_pipeline, basic_config, caplog):
        """Test status summary when no models are loaded (FALLBACK mode)."""
        with caplog.at_level(logging.INFO):
            engine = AdvancedPricePredictionEngine(
                market_data_pipeline=mock_market_data_pipeline,
                feature_pipeline=mock_feature_pipeline,
                config=basic_config
            )
            
            # Check status summary
            status = engine.get_status_summary()
            assert 'FALLBACK Mode' in status
            assert 'No trained models' in status
            
            # Check that warning log was issued
            assert any('FALLBACK mode' in record.message for record in caplog.records)
    
    def test_status_summary_shows_timeframes(self, mock_market_data_pipeline, 
                                             mock_feature_pipeline, basic_config):
        """Test that status summary shows configured timeframes."""
        engine = AdvancedPricePredictionEngine(
            market_data_pipeline=mock_market_data_pipeline,
            feature_pipeline=mock_feature_pipeline,
            config=basic_config
        )
        
        status = engine.get_status_summary()
        # Should mention the configured timeframes
        assert any(tf in status for tf in ['5m', '15m', '1h'])
    
    def test_initialization_logging_shows_mode(self, mock_market_data_pipeline, 
                                               mock_feature_pipeline, basic_config, caplog):
        """Test that initialization logs clearly show the operation mode."""
        with caplog.at_level(logging.INFO):
            engine = AdvancedPricePredictionEngine(
                market_data_pipeline=mock_market_data_pipeline,
                feature_pipeline=mock_feature_pipeline,
                config=basic_config
            )
            
            # Check that status is logged
            log_messages = [record.message for record in caplog.records]
            
            # Should have a log with PricePredictor Status
            assert any('PricePredictor Status' in msg for msg in log_messages)
            
            # In FALLBACK mode, should have warning
            if not engine.is_trained:
                assert any('FALLBACK mode' in msg for msg in log_messages)
    
    @pytest.mark.asyncio
    async def test_update_prediction_logging_fallback_mode(self, mock_market_data_pipeline, 
                                                           mock_feature_pipeline, 
                                                           basic_config, caplog):
        """Test that prediction updates log FALLBACK mode clearly."""
        import pandas as pd
        import numpy as np
        
        # Create realistic mock data with sufficient rows
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        mock_df = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.rand(100) * 1000,
            'high': 50500 + np.random.rand(100) * 1000,
            'low': 49500 + np.random.rand(100) * 1000,
            'close': 50000 + np.random.rand(100) * 1000,
            'volume': np.random.rand(100) * 1000
        })
        
        # Use async mock for get_latest_ohlcv
        async def mock_get_ohlcv(*args, **kwargs):
            return mock_df
        
        mock_market_data_pipeline.get_latest_ohlcv = mock_get_ohlcv
        
        # Mock feature pipeline - return features with proper shape
        mock_features = pd.DataFrame(np.random.rand(100, 42))
        mock_feature_pipeline.extract_features = Mock(return_value=mock_features)
        
        engine = AdvancedPricePredictionEngine(
            market_data_pipeline=mock_market_data_pipeline,
            feature_pipeline=mock_feature_pipeline,
            config=basic_config
        )
        
        with caplog.at_level(logging.DEBUG):
            # Trigger prediction update
            await engine._update_predictions(['BTC/USDT'], ['5m'])
            
            # Check logs
            log_messages = [record.message for record in caplog.records]
            
            # In fallback mode, either we get FALLBACK prediction or no data warning
            if not engine.is_trained:
                # Should have some indication of fallback mode or data issues
                has_fallback_msg = any('FALLBACK prediction' in msg or 'Fallback prediction' in msg for msg in log_messages)
                has_no_data_msg = any('No data available' in msg for msg in log_messages)
                assert has_fallback_msg or has_no_data_msg, f"Expected fallback or no-data message in logs: {log_messages}"
    
    def test_get_status_summary_format(self, mock_market_data_pipeline, 
                                       mock_feature_pipeline, basic_config):
        """Test that status summary has expected format."""
        engine = AdvancedPricePredictionEngine(
            market_data_pipeline=mock_market_data_pipeline,
            feature_pipeline=mock_feature_pipeline,
            config=basic_config
        )
        
        status = engine.get_status_summary()
        
        # Should be a non-empty string
        assert isinstance(status, str)
        assert len(status) > 0
        
        # Should contain either "ML Mode" or "FALLBACK Mode"
        assert 'ML Mode' in status or 'FALLBACK Mode' in status
    
    def test_has_model_logging(self, mock_market_data_pipeline, 
                               mock_feature_pipeline, basic_config, caplog):
        """Test that has_model_for logs appropriately."""
        engine = AdvancedPricePredictionEngine(
            market_data_pipeline=mock_market_data_pipeline,
            feature_pipeline=mock_feature_pipeline,
            config=basic_config
        )
        
        with caplog.at_level(logging.DEBUG):
            result = engine.has_model_for('BTC/USDT')
            
            # Should log model check
            log_messages = [record.message for record in caplog.records]
            assert any('Model check' in msg for msg in log_messages)
            
            # Result should match is_trained status
            assert result == engine.is_trained


class TestPricePredictorLoggingWithMockedModels:
    """Test logging when models are mocked as loaded."""
    
    @pytest.fixture
    def mock_market_data_pipeline(self):
        """Create mock market data pipeline."""
        pipeline = Mock()
        return pipeline
    
    @pytest.fixture
    def mock_feature_pipeline(self):
        """Create mock feature engineering pipeline."""
        pipeline = Mock()
        return pipeline
    
    @pytest.fixture
    def basic_config(self):
        """Create basic ML configuration."""
        return {
            'prediction': {
                'timeframes': ['5m', '15m'],
                'update_interval_seconds': 60,
                'cache_ttl_seconds': 300
            },
            'models': ['lstm'],
            'model_params': {
                'lstm': {
                    'hidden_size': 64,
                    'num_layers': 2
                }
            },
            'feature_size': 42,
            'forecast_horizon': 12
        }
    
    def test_status_summary_with_loaded_models(self, mock_market_data_pipeline, 
                                               mock_feature_pipeline, basic_config, caplog):
        """Test status summary when models are loaded (ML mode)."""
        with patch.object(AdvancedPricePredictionEngine, 'load_models', return_value=True):
            with caplog.at_level(logging.INFO):
                engine = AdvancedPricePredictionEngine(
                    market_data_pipeline=mock_market_data_pipeline,
                    feature_pipeline=mock_feature_pipeline,
                    config=basic_config
                )
                
                # Manually set is_trained to True for this test
                engine.is_trained = True
                
                # Check status summary
                status = engine.get_status_summary()
                assert 'ML Mode' in status
                assert 'models loaded' in status
                
                # Should NOT have FALLBACK warning
                log_messages = [record.message for record in caplog.records]
                # In ML mode, we shouldn't have fallback warnings
                if engine.is_trained:
                    fallback_logs = [msg for msg in log_messages if 'FALLBACK mode' in msg]
                    assert len(fallback_logs) == 0, f"Expected no FALLBACK logs in ML mode, but found: {fallback_logs}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
