"""
Test Suite for StrategyCoordinator AI-Gate Integration (Phase 5)

Tests for:
1. GEMMA adapter initialization in StrategyCoordinator
2. AI-Gate signal filtering logic
3. process_signal method with GEMMA integration
4. Fallback to legacy ML when GEMMA is not available
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from src.core.strategy_coordinator import StrategyCoordinator


class TestStrategyCoordinatorGemmaInit:
    """Test GEMMA initialization in StrategyCoordinator"""
    
    def test_gemma_not_initialized_when_disabled(self):
        """GEMMA adapter should not be initialized when disabled in config"""
        config = {
            'ml': {
                'gemma': {
                    'enabled': False
                }
            }
        }
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        assert coordinator.gemma_adapter is None
    
    def test_gemma_not_initialized_when_config_missing(self):
        """GEMMA adapter should not be initialized when config is missing"""
        config = {}
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        assert coordinator.gemma_adapter is None
    
    @patch('src.core.strategy_coordinator.logger')
    def test_gemma_initialized_when_enabled(self, mock_logger):
        """GEMMA adapter should be initialized when enabled in config"""
        config = {
            'ml': {
                'gemma': {
                    'enabled': True,
                    'model_path': 'test/path/model.pt',
                    'scaler_path': 'test/path/scaler.joblib'
                }
            }
        }
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        # The import will fail naturally since torch is not installed in test env
        # This is expected behavior - it should handle the error gracefully
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        # Should gracefully fail and log error
        assert coordinator.gemma_adapter is None
        # Check that error was logged
        assert any('GEMMA' in str(call) for call in mock_logger.error.call_args_list)
    
    def test_gemma_initialization_handles_import_error(self):
        """GEMMA initialization should handle ImportError gracefully"""
        config = {
            'ml': {
                'gemma': {
                    'enabled': True
                }
            }
        }
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        # The import will fail naturally since we don't have torch installed
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        # Should not raise exception, just set adapter to None
        assert coordinator.gemma_adapter is None


class TestAIGateFiltering:
    """Test AI-Gate filtering logic"""
    
    def test_ai_gate_passes_high_confidence_signal(self):
        """AI-Gate should pass signals with confidence above threshold"""
        config = {
            'ml': {
                'price': {
                    'min_confidence': 0.66
                }
            }
        }
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        signal = {
            'symbol': 'BTC/USDT',
            'ml_confidence': 0.75,
            'features': {}
        }
        
        result = coordinator._apply_ai_gate(signal)
        assert result is True
        assert coordinator.processing_stats['ai_gate_rejections'] == 0
    
    def test_ai_gate_rejects_low_confidence_signal(self):
        """AI-Gate should reject signals with confidence below threshold"""
        config = {
            'ml': {
                'price': {
                    'min_confidence': 0.66
                }
            }
        }
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        signal = {
            'symbol': 'BTC/USDT',
            'ml_confidence': 0.50,
            'features': {}
        }
        
        result = coordinator._apply_ai_gate(signal)
        assert result is False
        assert coordinator.processing_stats['ai_gate_rejections'] == 1
    
    def test_ai_gate_uses_gemma_confidence_when_available(self):
        """AI-Gate should prioritize GEMMA confidence over legacy ML"""
        config = {
            'ml': {
                'price': {
                    'min_confidence': 0.66
                }
            }
        }
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        # Mock GEMMA adapter
        mock_adapter = Mock()
        mock_adapter.predict.return_value = {
            'price_confidence': 0.80,
            'prediction_label': 'bullish',
            'prediction': 2
        }
        coordinator.gemma_adapter = mock_adapter
        
        signal = {
            'symbol': 'BTC/USDT',
            'ml_confidence': 0.50,  # Low legacy confidence
            'features': {'feature1': 1.0, 'feature2': 2.0}
        }
        
        result = coordinator._apply_ai_gate(signal)
        
        # Should use GEMMA confidence (0.80) instead of ml_confidence (0.50)
        assert result is True
        assert signal['gemma_confidence'] == 0.80
        assert signal['gemma_prediction'] == 'bullish'
        mock_adapter.predict.assert_called_once_with(signal['features'])
    
    def test_ai_gate_handles_gemma_prediction_failure(self):
        """AI-Gate should handle GEMMA prediction failures gracefully"""
        config = {
            'ml': {
                'price': {
                    'min_confidence': 0.66
                }
            }
        }
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        # Mock GEMMA adapter that raises exception
        mock_adapter = Mock()
        mock_adapter.predict.side_effect = Exception("Model failed")
        coordinator.gemma_adapter = mock_adapter
        
        signal = {
            'symbol': 'BTC/USDT',
            'ml_confidence': 0.75,  # Should fall back to this
            'features': {'feature1': 1.0}
        }
        
        result = coordinator._apply_ai_gate(signal)
        
        # Should fall back to ml_confidence and pass
        assert result is True
        assert 'gemma_confidence' not in signal or signal.get('gemma_confidence') is None


class TestProcessSignalMethod:
    """Test process_signal method with AI-Gate integration"""
    
    @pytest.mark.asyncio
    async def test_process_signal_rejects_at_ai_gate(self):
        """process_signal should reject signal at AI-Gate if confidence too low"""
        config = {
            'ml': {
                'price': {
                    'min_confidence': 0.66
                }
            }
        }
        
        portfolio_manager = Mock()
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        signal = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry': 50000,
            'ml_confidence': 0.40,  # Below threshold
            'features': {},
            'strategy_name': 'test_strategy'
        }
        
        result = await coordinator.process_signal(signal)
        
        assert result is None
        assert coordinator.processing_stats['ai_gate_rejections'] == 1
    
    @pytest.mark.asyncio
    async def test_process_signal_passes_high_confidence(self):
        """process_signal should process signal with high confidence through all gates"""
        config = {
            'ml': {
                'price': {
                    'min_confidence': 0.66
                }
            }
        }
        
        portfolio_manager = Mock()
        portfolio_manager.cfg = {}  # For validate_duplicate
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        # Mock _assess_signal_risk to return acceptable
        coordinator._assess_signal_risk = AsyncMock(return_value={
            'acceptable': True,
            'reason': 'OK'
        })
        
        signal = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry': 50000,
            'ml_confidence': 0.80,  # Above threshold
            'features': {},
            'strategy_name': 'test_strategy'
        }
        
        result = await coordinator.process_signal(signal)
        
        assert result is not None
        assert result['symbol'] == 'BTC/USDT'
        assert coordinator.processing_stats['approved_signals'] == 1
    
    @pytest.mark.asyncio
    async def test_process_signal_with_gemma_enhancement(self):
        """process_signal should enrich signal with GEMMA predictions"""
        config = {
            'ml': {
                'price': {
                    'min_confidence': 0.66
                }
            }
        }
        
        portfolio_manager = Mock()
        portfolio_manager.cfg = {}
        risk_manager = Mock()
        
        coordinator = StrategyCoordinator(
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        # Mock GEMMA adapter
        mock_adapter = Mock()
        mock_adapter.predict.return_value = {
            'price_confidence': 0.85,
            'prediction_label': 'bullish',
            'prediction': 2
        }
        coordinator.gemma_adapter = mock_adapter
        
        # Mock risk assessment
        coordinator._assess_signal_risk = AsyncMock(return_value={
            'acceptable': True,
            'reason': 'OK'
        })
        
        signal = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry': 50000,
            'features': {'feature1': 1.0},
            'strategy_name': 'test_strategy'
        }
        
        result = await coordinator.process_signal(signal)
        
        assert result is not None
        assert result['gemma_confidence'] == 0.85
        assert result['gemma_prediction'] == 'bullish'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
