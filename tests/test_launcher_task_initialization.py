"""
Test launcher task attribute initialization.

This test validates that all task attributes are properly initialized
in the LiveTradingLauncher to prevent AttributeError during cleanup.

Related Issue: SefaGH/bearish-alpha-bot#278
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestLauncherTaskInitialization:
    """Test that all task attributes are properly initialized."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Clear the singleton cache
        import config.live_trading_config
        config.live_trading_config._config_instance = None
        
        # Set minimal required env vars
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT'
        os.environ['CAPITAL_USDT'] = '100'
    
    def teardown_method(self):
        """Cleanup after tests."""
        # Clear test env vars
        for var in ['TRADING_SYMBOLS', 'CAPITAL_USDT']:
            if var in os.environ:
                del os.environ[var]
        
        # Clear singleton cache
        import config.live_trading_config
        config.live_trading_config._config_instance = None
    
    def test_launcher_initializes_task_attributes(self):
        """Test that LiveTradingLauncher initializes all task attributes."""
        from live_trading_launcher import LiveTradingLauncher
        
        launcher = LiveTradingLauncher(
            mode='paper',
            dry_run=True,
            infinite=False,
            auto_restart=False,
            max_restarts=3,
            restart_delay=60,
            debug_mode=False
        )
        
        # Verify all task attributes are initialized
        assert hasattr(launcher, '_main_trading_task'), "Missing _main_trading_task attribute"
        assert hasattr(launcher, '_prediction_loop_task'), "Missing _prediction_loop_task attribute"
        assert hasattr(launcher, '_websocket_task'), "Missing _websocket_task attribute"
        assert hasattr(launcher, '_heartbeat_task'), "Missing _heartbeat_task attribute"
        assert hasattr(launcher, '_monitoring_task'), "Missing _monitoring_task attribute"
        
        # Verify they are initialized to None
        assert launcher._main_trading_task is None
        assert launcher._prediction_loop_task is None
        assert launcher._websocket_task is None
        assert launcher._heartbeat_task is None
        assert launcher._monitoring_task is None
    
    def test_trading_pairs_is_list(self):
        """Test that TRADING_PAIRS is always a list after initialization."""
        from live_trading_launcher import LiveTradingLauncher
        
        # Test with single symbol
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT'
        launcher = LiveTradingLauncher(
            mode='paper',
            dry_run=True,
            infinite=False,
            auto_restart=False,
            max_restarts=3,
            restart_delay=60,
            debug_mode=False
        )
        
        assert isinstance(launcher.TRADING_PAIRS, list), \
            f"TRADING_PAIRS should be list, got {type(launcher.TRADING_PAIRS)}"
        assert len(launcher.TRADING_PAIRS) > 0, "TRADING_PAIRS should not be empty"
        assert 'BTC/USDT' in launcher.TRADING_PAIRS or 'BTC/USDT:USDT' in launcher.TRADING_PAIRS
    
    def test_trading_pairs_with_multiple_symbols(self):
        """Test TRADING_PAIRS with multiple symbols."""
        from live_trading_launcher import LiveTradingLauncher
        
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT,ETH/USDT,SOL/USDT'
        
        # Clear singleton for fresh config load
        import config.live_trading_config
        config.live_trading_config._config_instance = None
        
        launcher = LiveTradingLauncher(
            mode='paper',
            dry_run=True,
            infinite=False,
            auto_restart=False,
            max_restarts=3,
            restart_delay=60,
            debug_mode=False
        )
        
        assert isinstance(launcher.TRADING_PAIRS, list)
        assert len(launcher.TRADING_PAIRS) >= 3, f"Expected at least 3 symbols, got {len(launcher.TRADING_PAIRS)}"
    
    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_prediction_task(self):
        """Test that cleanup gracefully handles missing _prediction_loop_task."""
        from live_trading_launcher import LiveTradingLauncher
        
        launcher = LiveTradingLauncher(
            mode='paper',
            dry_run=True,
            infinite=False,
            auto_restart=False,
            max_restarts=3,
            restart_delay=60,
            debug_mode=False
        )
        
        # Simulate the old bug by deleting the attribute
        if hasattr(launcher, '_prediction_loop_task'):
            delattr(launcher, '_prediction_loop_task')
        
        # This should not raise AttributeError thanks to hasattr check
        try:
            await launcher.cleanup()
            # If we get here, the defensive check worked
            assert True
        except AttributeError as e:
            pytest.fail(f"cleanup() should handle missing attributes gracefully, but got: {e}")
    
    @pytest.mark.asyncio
    async def test_cleanup_with_initialized_attributes(self):
        """Test that cleanup works correctly with initialized attributes."""
        from live_trading_launcher import LiveTradingLauncher
        
        launcher = LiveTradingLauncher(
            mode='paper',
            dry_run=True,
            infinite=False,
            auto_restart=False,
            max_restarts=3,
            restart_delay=60,
            debug_mode=False
        )
        
        # All attributes should be initialized
        assert hasattr(launcher, '_prediction_loop_task')
        assert launcher._prediction_loop_task is None
        
        # Cleanup should work without errors
        try:
            await launcher.cleanup()
            assert True
        except AttributeError as e:
            pytest.fail(f"cleanup() failed with AttributeError: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
