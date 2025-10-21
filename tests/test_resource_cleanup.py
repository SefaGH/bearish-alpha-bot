#!/usr/bin/env python3
"""
Tests for Resource Cleanup and Shutdown Logic.

Validates proper resource cleanup to prevent resource leaks.
Issue: https://github.com/SefaGH/bearish-alpha-bot/issues/XXX
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Set env var to skip Python version check for testing
os.environ['SKIP_PYTHON_VERSION_CHECK'] = '1'

from live_trading_launcher import LiveTradingLauncher, OptimizedWebSocketManager


class TestCcxtClientCleanup:
    """Test CcxtClient close method."""
    
    @pytest.mark.asyncio
    async def test_ccxt_client_close_method_exists(self):
        """Test that CcxtClient has close method."""
        from core.ccxt_client import CcxtClient
        
        # Check method exists
        assert hasattr(CcxtClient, 'close')
        
        # Check it's async
        import inspect
        assert inspect.iscoroutinefunction(CcxtClient.close)
    
    @pytest.mark.asyncio
    async def test_ccxt_client_close_calls_exchange_close(self):
        """Test that CcxtClient.close() calls exchange.close()."""
        from core.ccxt_client import CcxtClient
        
        with patch('core.ccxt_client.ccxt') as mock_ccxt:
            # Mock exchange class
            mock_exchange = MagicMock()
            mock_exchange.close = AsyncMock()
            
            # Mock ccxt module
            mock_ccxt.bingx = MagicMock(return_value=mock_exchange)
            mock_ccxt.__dict__['bingx'] = mock_ccxt.bingx
            
            # Create client
            client = CcxtClient('bingx', {'apiKey': 'test', 'secret': 'test'})
            
            # Close client
            await client.close()
            
            # Verify exchange.close() was called
            mock_exchange.close.assert_called_once()


class TestOptimizedWebSocketManagerCleanup:
    """Test OptimizedWebSocketManager cleanup methods."""
    
    @pytest.mark.asyncio
    async def test_stop_streaming_method_exists(self):
        """Test that OptimizedWebSocketManager has stop_streaming method."""
        manager = OptimizedWebSocketManager()
        
        # Check method exists
        assert hasattr(manager, 'stop_streaming')
        
        # Check it's async
        import inspect
        assert inspect.iscoroutinefunction(manager.stop_streaming)
    
    @pytest.mark.asyncio
    async def test_stop_streaming_handles_no_manager(self):
        """Test stop_streaming handles case when ws_manager is None."""
        manager = OptimizedWebSocketManager()
        manager.ws_manager = None
        
        # Should not raise error
        await manager.stop_streaming()
    
    @pytest.mark.asyncio
    async def test_stop_streaming_calls_ws_manager_close(self):
        """Test stop_streaming calls ws_manager.close()."""
        manager = OptimizedWebSocketManager()
        
        # Mock ws_manager
        mock_ws_manager = MagicMock()
        mock_ws_manager.close = AsyncMock()
        manager.ws_manager = mock_ws_manager
        manager.is_initialized = True
        
        # Call stop_streaming
        await manager.stop_streaming()
        
        # Verify ws_manager.close() was called
        mock_ws_manager.close.assert_called_once()
        
        # Verify is_initialized is reset
        assert manager.is_initialized == False


class TestLiveTradingLauncherCleanup:
    """Test LiveTradingLauncher cleanup functionality."""
    
    def test_cleanup_tracking_variables_initialized(self):
        """Test that cleanup tracking variables are initialized."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Check cleanup tracking variables exist
        assert hasattr(launcher, '_cleanup_done')
        assert hasattr(launcher, '_shutdown_event')
        
        # Check initial values
        assert launcher._cleanup_done == False
        assert isinstance(launcher._shutdown_event, asyncio.Event)
    
    @pytest.mark.asyncio
    async def test_cleanup_method_exists(self):
        """Test that cleanup method exists and is async."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Check method exists
        assert hasattr(launcher, 'cleanup')
        
        # Check it's async
        import inspect
        assert inspect.iscoroutinefunction(launcher.cleanup)
    
    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent(self):
        """Test that cleanup can be called multiple times safely."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Call cleanup multiple times
        await launcher.cleanup()
        await launcher.cleanup()
        await launcher.cleanup()
        
        # Should set cleanup flag
        assert launcher._cleanup_done == True
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_cleanup_stops_websocket(self):
        """Test that cleanup stops WebSocket streams."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Mock ws_optimizer with stop_streaming
        mock_ws_optimizer = MagicMock()
        mock_ws_optimizer.stop_streaming = AsyncMock()
        launcher.ws_optimizer = mock_ws_optimizer
        
        # Call cleanup
        await launcher.cleanup()
        
        # Verify stop_streaming was called
        mock_ws_optimizer.stop_streaming.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_cleanup_closes_exchange_clients(self):
        """Test that cleanup closes exchange clients."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Mock exchange client with close method
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        launcher.exchange_clients = {'bingx': mock_client}
        
        # Call cleanup
        await launcher.cleanup()
        
        # Verify client.close() was called
        mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_cleanup_stops_production_system(self):
        """Test that cleanup stops production coordinator."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Mock coordinator with stop_system
        mock_coordinator = MagicMock()
        mock_coordinator.stop_system = AsyncMock()
        launcher.coordinator = mock_coordinator
        
        # Call cleanup
        await launcher.cleanup()
        
        # Verify stop_system was called
        mock_coordinator.stop_system.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_handles_errors_gracefully(self):
        """Test that cleanup continues even if some steps fail."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Mock ws_optimizer that raises error
        mock_ws_optimizer = MagicMock()
        mock_ws_optimizer.stop_streaming = AsyncMock(side_effect=Exception("Test error"))
        launcher.ws_optimizer = mock_ws_optimizer
        
        # Mock coordinator (should still be called even after ws error)
        mock_coordinator = MagicMock()
        mock_coordinator.stop_system = AsyncMock()
        launcher.coordinator = mock_coordinator
        
        # Call cleanup - should not raise
        await launcher.cleanup()
        
        # Verify cleanup completed despite error
        assert launcher._cleanup_done == True
        
        # Verify coordinator was still called
        mock_coordinator.stop_system.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_cancels_pending_tasks(self):
        """Test that cleanup cancels pending async tasks."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Create a dummy pending task
        async def dummy_task():
            await asyncio.sleep(100)
        
        task = asyncio.create_task(dummy_task())
        
        # Call cleanup
        await launcher.cleanup()
        
        # Wait a bit for cancellation to propagate
        await asyncio.sleep(0.1)
        
        # Task should be cancelled or done
        assert task.done() or task.cancelled()
        
        # Clean up
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestExitCodes:
    """Test proper exit codes."""
    
    @pytest.mark.asyncio
    async def test_exit_code_success(self):
        """Test exit code 0 on success."""
        with patch.dict(os.environ, {
            'BINGX_KEY': 'test_key',
            'BINGX_SECRET': 'test_secret'
        }):
            launcher = LiveTradingLauncher(mode='paper', dry_run=True)
            
            # Mock all initialization steps
            launcher._load_environment = lambda: True
            launcher._initialize_exchange_connection = lambda: True
            launcher._initialize_risk_management = lambda: True
            launcher._initialize_ai_components = AsyncMock(return_value=True)
            launcher._initialize_strategies = AsyncMock(return_value=True)
            launcher._initialize_production_system = AsyncMock(return_value=True)
            launcher._register_strategies = AsyncMock(return_value=True)
            launcher._perform_preflight_checks = AsyncMock(return_value=True)
            launcher._print_configuration_summary = lambda: None
            
            # Run once (dry-run mode should return 0 after checks)
            exit_code = await launcher._run_once()
            
            assert exit_code == 0
    
    @pytest.mark.asyncio
    async def test_exit_code_keyboard_interrupt(self):
        """Test exit code 130 on keyboard interrupt."""
        with patch.dict(os.environ, {
            'BINGX_KEY': 'test_key',
            'BINGX_SECRET': 'test_secret'
        }):
            launcher = LiveTradingLauncher(mode='paper', dry_run=True)
            
            # Mock to raise KeyboardInterrupt
            async def raise_interrupt():
                raise KeyboardInterrupt()
            
            launcher._load_environment = raise_interrupt
            
            # Run once - should catch KeyboardInterrupt
            exit_code = await launcher._run_once()
            
            assert exit_code == 130


class TestCleanupIntegration:
    """Integration tests for cleanup workflow."""
    
    @pytest.mark.asyncio
    async def test_cleanup_called_in_finally_block(self):
        """Test that cleanup is called even when errors occur."""
        with patch.dict(os.environ, {
            'BINGX_KEY': 'test_key',
            'BINGX_SECRET': 'test_secret'
        }):
            launcher = LiveTradingLauncher(mode='paper', dry_run=True)
            
            # Track if cleanup was called
            cleanup_called = False
            original_cleanup = launcher.cleanup
            
            async def tracked_cleanup():
                nonlocal cleanup_called
                cleanup_called = True
                await original_cleanup()
            
            launcher.cleanup = tracked_cleanup
            
            # Mock to raise error
            async def raise_error():
                raise RuntimeError("Test error")
            
            launcher._load_environment = raise_error
            
            # Run once - should call cleanup even on error
            exit_code = await launcher._run_once()
            
            # Verify cleanup was called
            assert cleanup_called == True
            
            # Verify error exit code
            assert exit_code == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
