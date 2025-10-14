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
        # Mock the CCXT client
        mock_client = Mock()
        mock_client.markets.return_value = {
            'BTC/USDT:USDT': {},
            'ETH/USDT:USDT': {},
            'SOL/USDT:USDT': {},
            'BNB/USDT:USDT': {},
            'ADA/USDT:USDT': {},
            'DOT/USDT:USDT': {},
            'LTC/USDT:USDT': {},
            'AVAX/USDT:USDT': {}
        }
        mock_client.ticker.return_value = {'last': 50000.0}
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
