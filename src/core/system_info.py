#!/usr/bin/env python3
"""
System Information Collector Module

This module provides utilities to collect and format system information,
exchange status, and generate formatted startup headers for the trading bot.

Created for Issue #119: Enhanced log header with complete system information.
"""

import sys
import os
import platform
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


class SystemInfoCollector:
    """
    Collector for system information, exchange status, and WebSocket status.
    All methods are static for easy use without instantiation.
    """
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Collect comprehensive system information.
        
        Returns:
            Dictionary containing:
            - user: Username from environment or git config
            - timestamp: Current UTC timestamp (YYYY-MM-DD HH:MM:SS)
            - python_version: Python version string
            - os_name: Operating system name
            - os_release: OS release version
            - os_version: OS version details
            - machine: Machine type (x86_64, etc.)
            - processor: Processor name
        """
        try:
            # Get user from environment
            user = os.environ.get('USER') or os.environ.get('USERNAME') or 'Unknown'
            
            # Try to get from git config if available
            if user == 'Unknown':
                try:
                    import subprocess
                    result = subprocess.run(
                        ['git', 'config', 'user.name'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        user = result.stdout.strip()
                except Exception:
                    pass
            
            # Get timestamp in UTC format
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get Python version
            python_version = sys.version.split()[0]
            
            # Get OS information
            os_name = platform.system()
            os_release = platform.release()
            os_version = platform.version()
            machine = platform.machine()
            
            # Get processor info
            try:
                processor = platform.processor()
                if not processor:
                    processor = machine
            except Exception:
                processor = machine
            
            return {
                'user': user,
                'timestamp': timestamp,
                'python_version': python_version,
                'os_name': os_name,
                'os_release': os_release,
                'os_version': os_version,
                'machine': machine,
                'processor': processor
            }
            
        except Exception as e:
            # Return defaults on error
            return {
                'user': 'Unknown',
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'python_version': sys.version.split()[0],
                'os_name': platform.system(),
                'os_release': 'Unknown',
                'os_version': 'Unknown',
                'machine': 'Unknown',
                'processor': 'Unknown'
            }
    
    @staticmethod
    def format_os_string(info: Dict[str, Any]) -> str:
        """
        Format operating system string with proper detection.
        
        Args:
            info: System info dictionary from get_system_info()
        
        Returns:
            Formatted OS string like "Windows 11", "Ubuntu 22.04", "macOS 14.0"
        """
        try:
            os_name = info.get('os_name', 'Unknown')
            os_release = info.get('os_release', '')
            os_version = info.get('os_version', '')
            
            if os_name == 'Windows':
                # Detect Windows 10 vs 11 from build number
                # Windows 11 has build number >= 22000
                try:
                    build_number = int(os_version.split('.')[-1]) if '.' in os_version else 0
                    if build_number >= 22000:
                        return 'Windows 11'
                    else:
                        return 'Windows 10'
                except (ValueError, IndexError):
                    return f'Windows {os_release}'
            
            elif os_name == 'Linux':
                # Try to get distribution name
                distro_name = None
                distro_version = None
                
                # Try using distro module if available
                try:
                    import distro
                    distro_name = distro.name()
                    distro_version = distro.version()
                except ImportError:
                    pass
                
                # Try /etc/os-release if distro module not available
                if not distro_name:
                    try:
                        with open('/etc/os-release', 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.startswith('NAME='):
                                    distro_name = line.split('=')[1].strip().strip('"')
                                elif line.startswith('VERSION_ID='):
                                    distro_version = line.split('=')[1].strip().strip('"')
                    except Exception:
                        pass
                
                if distro_name and distro_version:
                    return f'{distro_name} {distro_version}'
                elif distro_name:
                    return distro_name
                else:
                    return f'Linux {os_release}'
            
            elif os_name == 'Darwin':
                # macOS
                try:
                    mac_ver = platform.mac_ver()[0]
                    if mac_ver:
                        return f'macOS {mac_ver}'
                except Exception:
                    pass
                return f'macOS {os_release}'
            
            else:
                return f'{os_name} {os_release}'
                
        except Exception:
            return info.get('os_name', 'Unknown')
    
    @staticmethod
    def get_exchange_status(exchange_clients: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check exchange connectivity and status.
        
        Args:
            exchange_clients: Dictionary of exchange client instances
        
        Returns:
            Dictionary containing:
            - connected: bool - Whether exchange is connected
            - status_emoji: str - Status emoji (✅/❌)
            - status_text: str - Status text (CONNECTED/FAILED/NO EXCHANGE CLIENT)
            - latency_ms: int or None - Latency in milliseconds
            - error: str or None - Error message if failed
        """
        try:
            # Check if exchange_clients is empty
            if not exchange_clients or len(exchange_clients) == 0:
                return {
                    'connected': False,
                    'status_emoji': '❌',
                    'status_text': 'NO EXCHANGE CLIENT',
                    'latency_ms': None,
                    'error': 'No exchange clients configured'
                }
            
            # Test first exchange with ping
            exchange_name = list(exchange_clients.keys())[0]
            exchange = exchange_clients[exchange_name]
            
            start_time = time.time()
            
            try:
                # Try different methods to test connection
                if hasattr(exchange, 'fetch_time'):
                    exchange.fetch_time()
                elif hasattr(exchange, 'fetch_status'):
                    exchange.fetch_status()
                elif hasattr(exchange, 'fetch_ticker'):
                    # Try to fetch a common ticker
                    exchange.fetch_ticker('BTC/USDT')
                else:
                    # No suitable method found
                    return {
                        'connected': False,
                        'status_emoji': '⚠️',
                        'status_text': 'UNKNOWN',
                        'latency_ms': None,
                        'error': 'Cannot test connection'
                    }
                
                # Calculate latency
                latency_ms = int((time.time() - start_time) * 1000)
                
                return {
                    'connected': True,
                    'status_emoji': '✅',
                    'status_text': 'CONNECTED',
                    'latency_ms': latency_ms,
                    'error': None
                }
                
            except Exception as e:
                return {
                    'connected': False,
                    'status_emoji': '❌',
                    'status_text': 'FAILED',
                    'latency_ms': None,
                    'error': str(e)
                }
                
        except Exception as e:
            return {
                'connected': False,
                'status_emoji': '❌',
                'status_text': 'ERROR',
                'latency_ms': None,
                'error': str(e)
            }
    
    @staticmethod
    def get_websocket_status(ws_manager: Any) -> Dict[str, Any]:
        """
        Check WebSocket status and active streams.
        
        Args:
            ws_manager: WebSocket manager instance or None
        
        Returns:
            Dictionary containing:
            - enabled: bool - Whether WebSocket is enabled
            - status_emoji: str - Status emoji (✅/⚠️)
            - status_text: str - Status text (CONNECTED/STREAMING/INITIALIZED/DISCONNECTED/REST MODE)
            - stream_count: int - Number of active streams
            - mode: str - Mode (websocket/rest/rest_fallback)
        """
        try:
            # Check if ws_manager is None
            if ws_manager is None:
                return {
                    'enabled': False,
                    'status_emoji': '⚠️',
                    'status_text': 'REST MODE',
                    'stream_count': 0,
                    'mode': 'rest'
                }
            
            # If OptimizedWebSocketManager wrapper verilmişse bağlanma bayrağını al
            connecting_flag = False
            try:
                if hasattr(ws_manager, '_connection_status') and isinstance(getattr(ws_manager, '_connection_status'), dict):
                    connecting_flag = bool(ws_manager._connection_status.get('connecting', False))
            except Exception:
                connecting_flag = False
            
            # Check if this is OptimizedWebSocketManager with ws_manager attribute
            actual_ws_manager = ws_manager
            if hasattr(ws_manager, 'ws_manager') and ws_manager.ws_manager:
                actual_ws_manager = ws_manager.ws_manager
            
            # Check actual connection state using is_connected() on clients
            connected_clients = []
            streaming_clients = []
            
            if hasattr(actual_ws_manager, 'clients'):
                try:
                    clients = actual_ws_manager.clients
                    for client in clients.values():
                        if hasattr(client, 'is_connected'):
                            try:
                                if client.is_connected():
                                    connected_clients.append(client)
                                    # Check if actually streaming (received messages)
                                    if hasattr(client, '_first_message_received') and client._first_message_received:
                                        streaming_clients.append(client)
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # Count active streams
            stream_count = 0
            if hasattr(actual_ws_manager, '_tasks'):
                try:
                    # Count running tasks
                    tasks = actual_ws_manager._tasks
                    stream_count = sum(1 for t in tasks if not t.done())
                except Exception:
                    pass
            
            # Determine status based on actual connection state
            if streaming_clients and stream_count > 0:
                return {
                    'enabled': True,
                    'status_emoji': '✅',
                    'status_text': 'CONNECTED and STREAMING',
                    'stream_count': stream_count,
                    'mode': 'websocket'
                }
            elif connected_clients and stream_count > 0:
                # “connecting...” yalnızca gerçekten bağlantı kurulurken gösterilsin
                status_text = 'STREAMING (connecting...)' if connecting_flag else 'STREAMING'
                return {
                    'enabled': True,
                    'status_emoji': '✅',
                    'status_text': status_text,
                    'stream_count': stream_count,
                    'mode': 'websocket'
                }
            elif stream_count > 0:
                return {
                    'enabled': True,
                    'status_emoji': '⚠️',
                    'status_text': 'INITIALIZED (not streaming)',
                    'stream_count': stream_count,
                    'mode': 'rest_fallback'
                }
            elif hasattr(ws_manager, 'is_initialized') and ws_manager.is_initialized:
                return {
                    'enabled': True,
                    'status_emoji': '⚠️',
                    'status_text': 'INITIALIZED (no streams)',
                    'stream_count': 0,
                    'mode': 'rest_fallback'
                }
            else:
                return {
                    'enabled': False,
                    'status_emoji': '⚠️',
                    'status_text': 'DISCONNECTED',
                    'stream_count': 0,
                    'mode': 'rest_fallback'
                }
                
        except Exception:
            return {
                'enabled': False,
                'status_emoji': '⚠️',
                'status_text': 'UNKNOWN',
                'stream_count': 0,
                'mode': 'rest'
            }


def format_startup_header(
    system_info: Dict[str, Any],
    mode: str,
    dry_run: bool,
    debug_mode: bool,
    exchange_clients: Dict[str, Any],
    ws_manager: Any,
    capital: float,
    trading_pairs: List[str],
    strategies: Dict[str, Any],
    risk_params: Dict[str, Any],
    risk_manager: Any = None
) -> str:
    """
    Format comprehensive startup header with system information.
    
    Args:
        system_info: System information dictionary
        mode: Trading mode (live/paper)
        dry_run: Whether in dry run mode
        debug_mode: Whether debug mode is enabled
        exchange_clients: Dictionary of exchange clients
        ws_manager: WebSocket manager instance
        capital: Total capital in USDT
        trading_pairs: List of trading pairs
        strategies: Dictionary of strategies
        risk_params: Risk parameters dictionary
        risk_manager: Risk manager instance (optional)
    
    Returns:
        Formatted multi-line header string
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("BEARISH ALPHA BOT - LIVE TRADING SYSTEM".center(80))
    lines.append("=" * 80)
    lines.append("")
    
    # System Information
    lines.append("[SYSTEM INFORMATION]")
    lines.append(f"User:              {system_info.get('user', 'Unknown')}")
    lines.append(f"Timestamp (UTC):   {system_info.get('timestamp', 'Unknown')}")
    lines.append(f"Python Version:    {system_info.get('python_version', 'Unknown')}")
    
    # Format OS string
    os_string = SystemInfoCollector.format_os_string(system_info)
    lines.append(f"Operating System:  {os_string}")
    
    lines.append(f"Mode:              {mode.upper()}")
    lines.append(f"Dry Run:           {'YES' if dry_run else 'NO'}")
    lines.append(f"Debug Mode:        {'ENABLED' if debug_mode else 'DISABLED'}")
    lines.append("")
    
    # Exchange Configuration
    lines.append("[EXCHANGE CONFIGURATION]")
    
    # Exchange names
    if exchange_clients:
        exchange_names = ', '.join(exchange_clients.keys())
        lines.append(f"Exchange:          {exchange_names}")
    else:
        lines.append("Exchange:          None")
    
    # Exchange status
    exchange_status = SystemInfoCollector.get_exchange_status(exchange_clients)
    if exchange_status['latency_ms'] is not None:
        lines.append(f"API Status:        {exchange_status['status_emoji']} {exchange_status['status_text']} (Latency: {exchange_status['latency_ms']}ms)")
    else:
        lines.append(f"API Status:        {exchange_status['status_emoji']} {exchange_status['status_text']}")
    
    # WebSocket status
    ws_status = SystemInfoCollector.get_websocket_status(ws_manager)
    if ws_status['stream_count'] > 0:
        lines.append(f"WebSocket:         {ws_status['status_emoji']} {ws_status['status_text']} ({ws_status['stream_count']} streams active)")
    else:
        lines.append(f"WebSocket:         {ws_status['status_emoji']} {ws_status['status_text']}")
    
    # Trading pairs
    lines.append(f"Trading Pairs:     {len(trading_pairs)} active symbols")
    lines.append("")
    
    # List all trading pairs
    for i, pair in enumerate(trading_pairs, 1):
        lines.append(f"  {i}. {pair}")
    lines.append("")
    
    # Capital & Risk Management
    lines.append("[CAPITAL & RISK MANAGEMENT]")
    lines.append(f"Total Capital:          {capital:.2f} USDT")
    
    # Get portfolio summary if risk manager available
    available_capital = capital
    current_exposure = 0.0
    capital_utilization = 0.0
    active_positions = 0
    
    if risk_manager and hasattr(risk_manager, 'get_portfolio_summary'):
        try:
            portfolio_summary = risk_manager.get_portfolio_summary()
            if isinstance(portfolio_summary, dict):
                current_exposure = portfolio_summary.get('current_exposure', 0.0)
                available_capital = portfolio_summary.get('available_capital', capital)
                capital_utilization = portfolio_summary.get('capital_utilization', 0.0)
                active_positions = portfolio_summary.get('active_positions', 0)
        except Exception:
            pass
    
    # Available balance
    if active_positions > 0:
        lines.append(f"Available Balance:      {available_capital:.2f} USDT ({active_positions} positions open)")
    else:
        lines.append(f"Available Balance:      {available_capital:.2f} USDT (0 positions open)")
    
    # Current exposure
    lines.append(f"Current Exposure:       {current_exposure:.2f} USDT ({capital_utilization:.1f}% utilization)")
    
    # Max position size
    max_position_size = risk_params.get('max_position_size', 0.2)
    max_position_usdt = capital * max_position_size
    lines.append(f"Max Position Size:      {max_position_size * 100:.1f}% ({max_position_usdt:.2f} USDT per trade)")
    
    # Risk per trade (use max_position_size if risk_per_trade not specified)
    risk_per_trade = risk_params.get('risk_per_trade', max_position_size * 0.25)
    risk_per_trade_usdt = capital * risk_per_trade
    lines.append(f"Risk Per Trade:         {risk_per_trade * 100:.1f}% ({risk_per_trade_usdt:.2f} USDT max risk)")
    
    # Stop loss
    stop_loss = risk_params.get('stop_loss_pct', 0.02)
    lines.append(f"Stop Loss:              {stop_loss * 100:.1f}%")
    
    # Take profit
    take_profit = risk_params.get('take_profit_pct', 0.015)
    lines.append(f"Take Profit:            {take_profit * 100:.1f}%")
    
    # Max drawdown
    max_drawdown = risk_params.get('max_drawdown', 0.05)
    max_drawdown_usdt = capital * max_drawdown
    lines.append(f"Max Drawdown:           {max_drawdown * 100:.1f}% ({max_drawdown_usdt:.2f} USDT)")
    lines.append("")
    
    # Trading Strategies
    lines.append("[TRADING STRATEGIES]")
    if strategies:
        for strategy_name, strategy_obj in strategies.items():
            # Try to get allocation if available
            allocation = None
            if hasattr(strategy_obj, 'allocation'):
                allocation = strategy_obj.allocation
            elif isinstance(strategy_obj, dict) and 'allocation' in strategy_obj:
                allocation = strategy_obj['allocation']
            
            if allocation is not None:
                lines.append(f"  ✅ {strategy_name} (allocation: {allocation * 100:.0f}%)")
            else:
                lines.append(f"  ✅ {strategy_name}")
    else:
        lines.append("  No strategies configured")
    lines.append("")
    
    # Risk Limits
    lines.append("[RISK LIMITS]")
    max_portfolio_risk = risk_params.get('max_portfolio_risk', 0.05)
    lines.append(f"Max Portfolio Risk:     {max_portfolio_risk * 100:.1f}%")
    
    max_correlation = risk_params.get('max_correlation', 0.7)
    lines.append(f"Max Correlation:        {max_correlation * 100:.1f}%")
    
    daily_loss_limit = risk_params.get('daily_loss_limit', 0.02)
    lines.append(f"Daily Loss Limit:       {daily_loss_limit * 100:.1f}%")
    lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append("SYSTEM READY - STARTING TRADING".center(80))
    lines.append("=" * 80)
    
    return "\n".join(lines)
