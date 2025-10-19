#!/usr/bin/env python3
"""
Test System Information Module

Tests for system_info.py module including SystemInfoCollector
and format_startup_header function.

Created for Issue #119: Enhanced log header with complete system information.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.system_info import SystemInfoCollector, format_startup_header


class TestSystemInfoCollector:
    """Test suite for SystemInfoCollector class."""
    
    def test_get_system_info_returns_dict(self):
        """Test that get_system_info returns a dictionary with required keys."""
        info = SystemInfoCollector.get_system_info()
        
        # Check it's a dictionary
        assert isinstance(info, dict), "get_system_info should return a dictionary"
        
        # Check all required keys are present
        required_keys = [
            'user', 'timestamp', 'python_version', 
            'os_name', 'os_release', 'machine'
        ]
        for key in required_keys:
            assert key in info, f"Missing required key: {key}"
            assert info[key] is not None, f"Key {key} should not be None"
    
    def test_timestamp_format(self):
        """Test that timestamp is in correct format (YYYY-MM-DD HH:MM:SS)."""
        info = SystemInfoCollector.get_system_info()
        timestamp = info['timestamp']
        
        # Check length (19 characters: YYYY-MM-DD HH:MM:SS)
        assert len(timestamp) == 19, f"Timestamp should be 19 chars, got {len(timestamp)}"
        
        # Verify format by parsing
        try:
            datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp}' is not in format YYYY-MM-DD HH:MM:SS")
    
    def test_format_os_string_windows(self):
        """Test Windows OS string formatting."""
        # Test Windows 11 detection (build >= 22000)
        info_win11 = {
            'os_name': 'Windows',
            'os_release': '10',
            'os_version': '10.0.22621'
        }
        result = SystemInfoCollector.format_os_string(info_win11)
        assert result == 'Windows 11', f"Expected 'Windows 11', got '{result}'"
        
        # Test Windows 10 detection (build < 22000)
        info_win10 = {
            'os_name': 'Windows',
            'os_release': '10',
            'os_version': '10.0.19045'
        }
        result = SystemInfoCollector.format_os_string(info_win10)
        assert result == 'Windows 10', f"Expected 'Windows 10', got '{result}'"
    
    def test_format_os_string_linux(self):
        """Test Linux OS string formatting."""
        # Mock /etc/os-release file
        mock_os_release = 'NAME="Ubuntu"\nVERSION_ID="22.04"\n'
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.readlines.return_value = mock_os_release.split('\n')
            
            info_linux = {
                'os_name': 'Linux',
                'os_release': '5.15.0',
                'os_version': '#1 SMP'
            }
            result = SystemInfoCollector.format_os_string(info_linux)
            
            # Should contain Linux or Ubuntu
            assert 'Linux' in result or 'Ubuntu' in result, f"Expected Linux/Ubuntu in result, got '{result}'"
    
    def test_get_exchange_status_no_clients(self):
        """Test exchange status with empty clients dict."""
        result = SystemInfoCollector.get_exchange_status({})
        
        assert result['connected'] is False, "Should not be connected with empty clients"
        assert result['status_emoji'] == '❌', "Should show error emoji"
        assert result['status_text'] == 'NO EXCHANGE CLIENT', "Should show no client message"
        assert result['latency_ms'] is None, "Latency should be None"
        assert result['error'] is not None, "Should have error message"
    
    def test_get_exchange_status_with_mock_client(self):
        """Test exchange status with mock client."""
        # Create mock exchange client
        mock_exchange = Mock()
        mock_exchange.fetch_time = Mock(return_value=1234567890000)
        
        clients = {'bingx': mock_exchange}
        
        result = SystemInfoCollector.get_exchange_status(clients)
        
        assert result['connected'] is True, "Should be connected with working client"
        assert result['status_emoji'] == '✅', "Should show success emoji"
        assert result['status_text'] == 'CONNECTED', "Should show connected status"
        assert result['latency_ms'] is not None, "Latency should be measured"
        assert isinstance(result['latency_ms'], int), "Latency should be integer"
        assert result['error'] is None, "Should have no error"
        
        # Verify fetch_time was called
        mock_exchange.fetch_time.assert_called_once()
    
    def test_get_websocket_status_no_manager(self):
        """Test WebSocket status with None manager."""
        result = SystemInfoCollector.get_websocket_status(None)
        
        assert result['enabled'] is False, "Should not be enabled with None manager"
        assert result['status_emoji'] == '⚠️', "Should show warning emoji"
        assert result['status_text'] == 'REST MODE', "Should show REST MODE"
        assert result['stream_count'] == 0, "Should have 0 streams"
        assert result['mode'] == 'rest', "Should be in rest mode"
    
    def test_get_websocket_status_with_manager(self):
        """Test WebSocket status with connected manager."""
        # Create mock WebSocket manager
        mock_ws = Mock()
        mock_ws.is_connected = Mock(return_value=True)
        mock_ws.streams = {
            'BTC/USDT': Mock(),
            'ETH/USDT': Mock()
        }
        
        result = SystemInfoCollector.get_websocket_status(mock_ws)
        
        assert result['enabled'] is True, "Should be enabled with connected manager"
        assert result['status_emoji'] == '✅', "Should show success emoji"
        assert result['status_text'] == 'OPTIMIZED', "Should show OPTIMIZED status"
        assert result['stream_count'] == 2, "Should have 2 streams"
        assert result['mode'] == 'websocket', "Should be in websocket mode"


class TestStartupHeaderFormatting:
    """Test suite for format_startup_header function."""
    
    @pytest.fixture
    def mock_components(self):
        """Fixture providing mock components for header formatting."""
        system_info = {
            'user': 'TestUser',
            'timestamp': '2025-10-19 18:52:16',
            'python_version': '3.11.5',
            'os_name': 'Windows',
            'os_release': '10',
            'os_version': '10.0.22621',
            'machine': 'x86_64',
            'processor': 'Intel64'
        }
        
        # Mock exchange client
        mock_exchange = Mock()
        mock_exchange.fetch_time = Mock(return_value=1234567890000)
        exchange_clients = {'bingx': mock_exchange}
        
        # Mock WebSocket manager
        mock_ws = Mock()
        mock_ws.is_connected = Mock(return_value=True)
        mock_ws.streams = {'BTC/USDT': Mock(), 'ETH/USDT': Mock()}
        
        trading_pairs = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        
        # Mock strategies
        strategy1 = Mock()
        strategy1.allocation = 0.5
        strategy2 = Mock()
        strategy2.allocation = 0.5
        strategies = {
            'Adaptive Oversold Bounce': strategy1,
            'Adaptive Short The Rip': strategy2
        }
        
        risk_params = {
            'max_position_size': 0.20,
            'risk_per_trade': 0.05,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.015,
            'max_drawdown': 0.05,
            'max_portfolio_risk': 0.05,
            'max_correlation': 0.70,
            'daily_loss_limit': 0.02
        }
        
        return {
            'system_info': system_info,
            'mode': 'paper',
            'dry_run': False,
            'debug_mode': False,
            'exchange_clients': exchange_clients,
            'ws_manager': mock_ws,
            'capital': 100.0,
            'trading_pairs': trading_pairs,
            'strategies': strategies,
            'risk_params': risk_params,
            'risk_manager': None
        }
    
    def test_format_startup_header_returns_string(self, mock_components):
        """Test that format_startup_header returns a non-empty string."""
        header = format_startup_header(**mock_components)
        
        assert isinstance(header, str), "Should return a string"
        assert len(header) > 0, "Should not be empty"
        assert '\n' in header, "Should be multi-line"
    
    def test_header_contains_system_info(self, mock_components):
        """Test that header contains system information."""
        header = format_startup_header(**mock_components)
        
        # Check for user
        assert 'TestUser' in header, "Should contain username"
        
        # Check for timestamp
        assert '2025-10-19 18:52:16' in header, "Should contain timestamp"
        
        # Check for Python version
        assert '3.11.5' in header, "Should contain Python version"
        
        # Check for OS (should detect Windows 11)
        assert 'Windows' in header, "Should contain OS name"
    
    def test_header_contains_exchange_info(self, mock_components):
        """Test that header contains exchange information."""
        header = format_startup_header(**mock_components)
        
        # Check for exchange name
        assert 'bingx' in header, "Should contain exchange name"
        
        # Check for connected status
        assert 'CONNECTED' in header or '✅' in header, "Should show connected status"
    
    def test_header_contains_capital_info(self, mock_components):
        """Test that header contains capital and risk information."""
        header = format_startup_header(**mock_components)
        
        # Check for capital
        assert '100.00 USDT' in header, "Should contain capital amount"
        
        # Check for risk percentages
        assert '20.0%' in header, "Should contain max position size"
        assert '5.0%' in header, "Should contain risk percentage"
    
    def test_header_contains_trading_pairs(self, mock_components):
        """Test that header contains trading pairs information."""
        header = format_startup_header(**mock_components)
        
        # Check for trading pairs
        assert 'BTC/USDT:USDT' in header, "Should contain BTC pair"
        assert 'ETH/USDT:USDT' in header, "Should contain ETH pair"
        assert '2 active symbols' in header, "Should show pair count"
    
    def test_header_contains_strategies(self, mock_components):
        """Test that header contains strategy information."""
        header = format_startup_header(**mock_components)
        
        # Check for strategy names
        assert 'Adaptive Oversold Bounce' in header, "Should contain strategy name"
        
        # Check for emoji
        assert '✅' in header, "Should contain success emoji"
        
        # Check for allocation
        assert '50%' in header or 'allocation' in header, "Should show allocation"
    
    def test_header_with_active_positions(self, mock_components):
        """Test header with active positions from risk manager."""
        # Create mock risk manager
        mock_risk_manager = Mock()
        mock_risk_manager.get_portfolio_summary = Mock(return_value={
            'current_exposure': 45.0,
            'available_capital': 55.0,
            'capital_utilization': 45.0,
            'active_positions': 3
        })
        
        mock_components['risk_manager'] = mock_risk_manager
        
        header = format_startup_header(**mock_components)
        
        # Check for position count
        assert '3 positions open' in header, "Should show active positions"
        
        # Check for utilization
        assert '45.0%' in header, "Should show capital utilization"
        
        # Check for available capital
        assert '55.00 USDT' in header, "Should show available capital"
        
        # Verify get_portfolio_summary was called
        mock_risk_manager.get_portfolio_summary.assert_called_once()
    
    def test_header_formatting_lines(self, mock_components):
        """Test that header has proper section formatting."""
        header = format_startup_header(**mock_components)
        
        # Check for section headers
        assert '[SYSTEM INFORMATION]' in header, "Should have system info section"
        assert '[EXCHANGE CONFIGURATION]' in header, "Should have exchange section"
        assert '[CAPITAL & RISK MANAGEMENT]' in header, "Should have capital section"
        assert '[TRADING STRATEGIES]' in header, "Should have strategies section"
        assert '[RISK LIMITS]' in header, "Should have risk limits section"
        
        # Check for decorative lines
        assert '=' * 80 in header, "Should have 80-char border lines"
        
        # Check for title
        assert 'BEARISH ALPHA BOT' in header, "Should have bot title"
        assert 'SYSTEM READY' in header, "Should have ready message"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
