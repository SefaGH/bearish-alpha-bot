#!/usr/bin/env python3
"""
Tests for Live Trading Launcher.

Validates the launcher initialization and configuration.
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Set env var to skip Python version check for testing
os.environ['SKIP_PYTHON_VERSION_CHECK'] = '1'

from live_trading_launcher import LiveTradingLauncher


class TestLiveTradingLauncher:
    """Test suite for LiveTradingLauncher."""
    
    def test_launcher_initialization(self):
        """Test launcher can be initialized."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        assert launcher.mode == 'paper'
        assert launcher.dry_run == True
        assert launcher.CAPITAL_USDT == 100.0
        assert len(launcher.TRADING_PAIRS) == 8
        
    def test_trading_pairs_configuration(self):
        """Test trading pairs are correctly configured."""
        launcher = LiveTradingLauncher(mode='paper')
        
        expected_pairs = [
            'BTC/USDT:USDT',
            'ETH/USDT:USDT',
            'SOL/USDT:USDT',
            'BNB/USDT:USDT',
            'ADA/USDT:USDT',
            'DOT/USDT:USDT',
            'LTC/USDT:USDT',
            'AVAX/USDT:USDT'
        ]
        
        assert launcher.TRADING_PAIRS == expected_pairs
    
    def test_risk_parameters(self):
        """Test risk parameters are correctly configured."""
        launcher = LiveTradingLauncher(mode='paper')
        
        assert launcher.RISK_PARAMS['max_position_size'] == 0.15  # 15%
        assert launcher.RISK_PARAMS['stop_loss_pct'] == 0.05  # 5%
        assert launcher.RISK_PARAMS['take_profit_pct'] == 0.10  # 10%
        assert launcher.RISK_PARAMS['max_drawdown'] == 0.15  # 15%
        assert launcher.RISK_PARAMS['max_portfolio_risk'] == 0.02  # 2%
    
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    def test_environment_loading_with_creds(self):
        """Test environment loading with credentials."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        result = launcher._load_environment()
        
        assert result == True
    
    def test_environment_loading_without_creds(self):
        """Test environment loading fails without credentials."""
        # Clear environment
        for key in ['BINGX_KEY', 'BINGX_SECRET']:
            os.environ.pop(key, None)
        
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        result = launcher._load_environment()
        
        assert result == False
    
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret',
        'TELEGRAM_BOT_TOKEN': 'test_token',
        'TELEGRAM_CHAT_ID': 'test_chat'
    })
    def test_telegram_initialization(self):
        """Test Telegram is initialized when credentials present."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        launcher._load_environment()
        
        assert launcher.telegram is not None
    
    def test_capital_configuration(self):
        """Test capital is configured to 100 USDT."""
        launcher = LiveTradingLauncher(mode='paper')
        
        assert launcher.CAPITAL_USDT == 100.0


class TestLauncherIntegration:
    """Integration tests for launcher components."""
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    @patch('live_trading_launcher.CcxtClient')
    async def test_dry_run_workflow(self, mock_ccxt):
        """Test dry-run workflow completes successfully."""
        # Mock the CCXT client with proper methods
        mock_client = MagicMock()
        
        # Mock fetch_ticker to return a dictionary with 'last' key
        mock_client.fetch_ticker.return_value = {'last': 50000.0}
        
        # Mock get_bingx_balance (authentication test)
        mock_client.get_bingx_balance.return_value = {'USDT': {'free': 100.0, 'used': 0.0, 'total': 100.0}}
        
        # Set the mock client as return value for CcxtClient constructor
        mock_ccxt.return_value = mock_client
        
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Should complete environment and exchange initialization
        assert launcher._load_environment() == True
        assert launcher._initialize_exchange_connection() == True
        assert launcher._initialize_risk_management() == True
    
    @pytest.mark.asyncio
    async def test_auto_restart_guard_clause(self):
        """Test that _run_with_auto_restart handles None restart_manager gracefully."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True, auto_restart=False)
        
        # Ensure restart_manager is None (auto_restart=False)
        assert launcher.restart_manager is None
        
        # Mock _run_once to avoid actual execution
        async def mock_run_once(duration):
            return 0
        
        launcher._run_once = mock_run_once
        
        # Call _run_with_auto_restart directly - should fallback to _run_once
        exit_code = await launcher._run_with_auto_restart()
        
        # Should return success and not crash with AttributeError
        assert exit_code == 0


class TestWebSocketConnectionLogic:
    """Test suite for WebSocket connection timeout and retry logic."""
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_wait_for_connection_success(self):
        """Test _wait_for_websocket_connection returns True when connected."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        launcher._load_environment()
        
        # Mock ws_optimizer with connection status
        launcher.ws_optimizer = MagicMock()
        launcher.ws_optimizer.get_connection_status.return_value = {
            'connected': True,
            'error': None,
            'exchanges': {'bingx': {'connected': True}}
        }
        
        # Should return True immediately
        result = await launcher._wait_for_websocket_connection(timeout=5)
        assert result == True
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_wait_for_connection_timeout(self):
        """Test _wait_for_websocket_connection returns False on timeout."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        launcher._load_environment()
        
        # Mock ws_optimizer with never-connecting status
        launcher.ws_optimizer = MagicMock()
        launcher.ws_optimizer.get_connection_status.return_value = {
            'connected': False,
            'error': None,
            'exchanges': {}
        }
        
        # Should timeout after 2 seconds
        result = await launcher._wait_for_websocket_connection(timeout=2, check_interval=1)
        assert result == False
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_wait_for_connection_with_error(self):
        """Test _wait_for_websocket_connection returns False on error."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        launcher._load_environment()
        
        # Mock ws_optimizer with error status
        launcher.ws_optimizer = MagicMock()
        launcher.ws_optimizer.get_connection_status.return_value = {
            'connected': False,
            'error': 'Connection refused',
            'exchanges': {}
        }
        
        # Should return False immediately due to error
        result = await launcher._wait_for_websocket_connection(timeout=5)
        assert result == False
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_establish_connection_success_first_attempt(self):
        """Test _establish_websocket_connection succeeds on first attempt."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        launcher._load_environment()
        
        # Mock ws_optimizer
        launcher.ws_optimizer = MagicMock()
        launcher.ws_optimizer.is_initialized = True
        launcher.ws_optimizer._connection_status = {'connecting': False, 'error': None}
        launcher.ws_optimizer.initialize_websockets = asyncio.coroutine(
            lambda clients: [MagicMock()]  # Return mock tasks
        )
        launcher.ws_optimizer.get_connection_status.return_value = {
            'connected': True,
            'error': None
        }
        
        # Mock exchange clients
        launcher.exchange_clients = {'bingx': MagicMock()}
        
        # Should succeed on first attempt
        result = await launcher._establish_websocket_connection(max_retries=3, timeout=5)
        assert result == True
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_establish_connection_retry_logic(self):
        """Test _establish_websocket_connection retries with exponential backoff."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        launcher._load_environment()
        
        # Mock ws_optimizer
        launcher.ws_optimizer = MagicMock()
        launcher.ws_optimizer.is_initialized = True
        launcher.ws_optimizer._connection_status = {'connecting': False, 'error': None}
        
        # Mock initialize_websockets to return tasks
        launcher.ws_optimizer.initialize_websockets = asyncio.coroutine(
            lambda clients: [MagicMock()]
        )
        
        # Mock stop_streaming
        launcher.ws_optimizer.stop_streaming = asyncio.coroutine(lambda: None)
        
        # Mock exchange clients
        launcher.exchange_clients = {'bingx': MagicMock()}
        
        # First two attempts fail (timeout), third succeeds
        call_count = {'count': 0}
        
        def get_status_side_effect():
            call_count['count'] += 1
            if call_count['count'] <= 4:  # First 2 attempts timeout (2 checks each)
                return {'connected': False, 'error': None}
            else:  # Third attempt succeeds
                return {'connected': True, 'error': None}
        
        launcher.ws_optimizer.get_connection_status.side_effect = get_status_side_effect
        
        # Should succeed on third attempt after retries
        result = await launcher._establish_websocket_connection(max_retries=3, timeout=2)
        assert result == True
        
        # Verify retries occurred (at least 2 stops should have been called)
        assert launcher.ws_optimizer.stop_streaming.call_count >= 2
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_establish_connection_all_retries_fail(self):
        """Test _establish_websocket_connection returns False after all retries fail."""
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        launcher._load_environment()
        
        # Mock ws_optimizer
        launcher.ws_optimizer = MagicMock()
        launcher.ws_optimizer.is_initialized = True
        launcher.ws_optimizer._connection_status = {'connecting': False, 'error': None}
        
        # Mock initialize_websockets to return tasks
        launcher.ws_optimizer.initialize_websockets = asyncio.coroutine(
            lambda clients: [MagicMock()]
        )
        
        # Mock stop_streaming
        launcher.ws_optimizer.stop_streaming = asyncio.coroutine(lambda: None)
        
        # Mock exchange clients
        launcher.exchange_clients = {'bingx': MagicMock()}
        
        # All attempts fail (never connects)
        launcher.ws_optimizer.get_connection_status.return_value = {
            'connected': False,
            'error': None
        }
        
        # Should fail after all retries
        result = await launcher._establish_websocket_connection(max_retries=2, timeout=1)
        assert result == False
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'BINGX_KEY': 'test_key',
        'BINGX_SECRET': 'test_secret'
    })
    async def test_connection_status_tracking(self):
        """Test OptimizedWebSocketManager tracks connection status."""
        from live_trading_launcher import OptimizedWebSocketManager
        
        ws_manager = OptimizedWebSocketManager()
        
        # Initial status should show not connected
        status = ws_manager.get_connection_status()
        assert status['connected'] == False
        assert status['connecting'] == False
        assert 'last_check' in status
        assert 'exchanges' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
