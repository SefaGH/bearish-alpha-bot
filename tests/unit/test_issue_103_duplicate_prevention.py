import pytest
import time
from unittest.mock import MagicMock, patch
from src.core.strategy_coordinator import StrategyCoordinator

class TestDuplicatePrevention:
    @pytest.fixture
    def coordinator(self):
        # Mock dependencies
        portfolio_manager = MagicMock()
        risk_manager = MagicMock()
        
        # Setup default config
        portfolio_manager.cfg = {
            'signals': {
                'duplicate_prevention': {
                    'enabled': True,
                    'cooldown_seconds': 60,
                    'min_price_change_pct': 0.005,  # 0.5%
                    'price_delta_bypass_enabled': True
                }
            }
        }
        
        coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
        return coordinator

    def test_basic_cooldown(self, coordinator):
        """Test that signals are rejected during cooldown period."""
        signal = {'symbol': 'BTC/USDT', 'entry': 50000}
        strategy = 'test_strategy'
        
        # First signal - should be accepted
        is_valid, reason = coordinator.validate_duplicate(signal, strategy)
        assert is_valid is True
        assert reason == "OK"
        
        # Immediate second signal - should be rejected (no price change)
        is_valid, reason = coordinator.validate_duplicate(signal, strategy)
        assert is_valid is False
        assert "cooldown" in reason

    def test_price_delta_bypass_success(self, coordinator):
        """Test that cooldown is bypassed when price change is sufficient."""
        symbol = 'BTC/USDT'
        strategy = 'test_strategy'
        
        # First signal at 50000
        signal1 = {'symbol': symbol, 'entry': 50000}
        coordinator.validate_duplicate(signal1, strategy)
        
        # Second signal at 51000 (2% change > 0.5% threshold)
        # Should be accepted despite being immediate
        signal2 = {'symbol': symbol, 'entry': 51000}
        is_valid, reason = coordinator.validate_duplicate(signal2, strategy)
        
        assert is_valid is True
        assert "price delta bypass" in reason

    def test_price_delta_bypass_failure(self, coordinator):
        """Test that cooldown is enforced when price change is insufficient."""
        symbol = 'BTC/USDT'
        strategy = 'test_strategy'
        
        # First signal at 50000
        signal1 = {'symbol': symbol, 'entry': 50000}
        coordinator.validate_duplicate(signal1, strategy)
        
        # Second signal at 50010 (0.02% change < 0.5% threshold)
        # Should be rejected
        signal2 = {'symbol': symbol, 'entry': 50010}
        is_valid, reason = coordinator.validate_duplicate(signal2, strategy)
        
        assert is_valid is False
        assert "price change" in reason
        assert "< threshold" in reason

    def test_different_strategies_independent(self, coordinator):
        """Test that cooldowns are independent per strategy."""
        symbol = 'BTC/USDT'
        
        # Strategy A signal
        signal1 = {'symbol': symbol, 'entry': 50000}
        is_valid, _ = coordinator.validate_duplicate(signal1, 'StrategyA')
        assert is_valid is True
        
        # Strategy B signal (same symbol, immediate)
        # Should be accepted as it's a different strategy
        signal2 = {'symbol': symbol, 'entry': 50000}
        is_valid, _ = coordinator.validate_duplicate(signal2, 'StrategyB')
        assert is_valid is True

    def test_legacy_config_fallback(self, coordinator):
        """Test fallback to legacy monitoring config."""
        # Setup legacy config structure
        coordinator.portfolio_manager.cfg = {
            'monitoring': {
                'duplicate_prevention': {
                    'enabled': True,
                    'same_symbol_cooldown': 60,
                    'price_delta_bypass_threshold': 0.005,
                    'price_delta_bypass_enabled': True
                }
            }
        }
        
        signal = {'symbol': 'ETH/USDT', 'entry': 2000}
        strategy = 'legacy_test'
        
        # First signal accepted
        assert coordinator.validate_duplicate(signal, strategy)[0] is True
        
        # Second signal rejected (cooldown)
        assert coordinator.validate_duplicate(signal, strategy)[0] is False

