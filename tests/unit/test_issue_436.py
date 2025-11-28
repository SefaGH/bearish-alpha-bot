import pytest
import logging
from unittest.mock import MagicMock, patch
from src.quality.quality_calculator import compute_quality
from src.core.strategy_coordinator import StrategyCoordinator
from src.core.position_manager import AdvancedPositionManager

class TestIssue436:
    """Tests for Issue #436: Log & Signal Explainability."""

    def test_compute_quality_logic(self):
        """Test the quality score calculation logic."""
        # Case 1: All components perfect
        features = {
            "ml_component": 1.0,
            "volume_component": 2.0,  # Normalized to 1.0
            "momentum_component": 5.0, # Normalized to 1.0
            "spread_component": 1.0   # Normalized to 1.0
        }
        result = compute_quality(features)
        # Weights: ML(0.6) + Vol(0.2) + Mom(0.15) + Spread(0.05) = 1.0
        assert result["value"] == 1.0
        assert not result["reason"]

        # Case 2: All components zero/missing (fallback)
        features = {}
        result = compute_quality(features)
        # Fallbacks: ML(0.1), Vol(0.05), Mom(0.05), Spread(0.05)
        # Score: 0.1*0.6 + 0.05*0.2 + 0.05*0.15 + 0.05*0.05 = 0.06 + 0.01 + 0.0075 + 0.0025 = 0.08
        assert 0.07 <= result["value"] <= 0.09
        assert "ml_component_missing_fallback_used" in result["reason"]

        # Case 3: Zero quality (forced low values)
        features = {
            "ml_component": 0.0,
            "volume_component": 0.0,
            "momentum_component": -5.0, # Normalized to 0.0
            "spread_component": 0.0
        }
        result = compute_quality(features)
        assert result["value"] == 0.0
        assert "ml_component_below_threshold" in result["reason"]

    @patch("src.core.strategy_coordinator.logger")
    def test_emit_signal_breakdown(self, mock_logger):
        """Test that emit_signal_breakdown logs correct JSON and alerts on zero quality."""
        coordinator = StrategyCoordinator(MagicMock(), MagicMock())
        
        signal = {
            "signal_id": "test_sig_1",
            "symbol": "BTC/USDT",
            "strategy_name": "test_strat",
            "side": "buy",
            "ml_confidence": 0.85,
            "predicted_regime": "bullish"
        }
        
        quality_result = {
            "value": 0.0,
            "components": {},
            "reason": ["test_reason"]
        }
        
        coordinator.emit_signal_breakdown(signal, quality_result)
        
        # Check for alert warning
        mock_logger.warning.assert_called()
        assert "Signal quality is 0.0" in mock_logger.warning.call_args[0][0]
        
        # Check for breakdown info log
        mock_logger.info.assert_called()
        log_call = [c for c in mock_logger.info.call_args_list if "SIGNAL_BREAKDOWN" in c[0][0]]
        assert log_call
        assert "quality_score" in log_call[0][0][0]

    def test_position_manager_metadata_extraction(self):
        """Test that PositionManager extracts quality score correctly."""
        pm = AdvancedPositionManager(MagicMock(), MagicMock())
        
        signal = {
            "symbol": "BTC/USDT",
            "quality_score": 0.95,
            "quality_breakdown": {"value": 0.95},
            "metadata": {},
            "ml_metadata": {"consensus": 0.8}
        }
        
        meta = pm._extract_entry_metadata(signal)
        
        assert meta["quality_score"] == 0.95
        assert meta["quality_breakdown"] == {"value": 0.95}
        assert meta["ml_price_score"] == 0.8
