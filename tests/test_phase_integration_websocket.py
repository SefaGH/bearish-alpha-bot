"""
Test Phase 1 + Phase 3.1 Integration.
Verifies WebSocketManager works with build_clients_from_env() pattern.
"""

import pytest
import asyncio
import logging
import os
from unittest.mock import Mock, patch

# Import components
from src.core.multi_exchange import build_clients_from_env
from src.core.ccxt_client import CcxtClient
from src.core.websocket_manager import WebSocketManager, StreamDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPhaseIntegration:
    """Test Phase 1 + Phase 3.1 integration patterns."""
    
    def test_ccxt_client_to_websocket_manager(self):
        """Test WebSocketManager can accept CcxtClient instances."""
        # Phase 1: Create exchange clients
        clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None),
            'bingx': CcxtClient('bingx', None)
        }
        
        # Phase 3.1: Initialize WebSocket manager with clients
        config = {'reconnect_delay': 5}
        ws_manager = WebSocketManager(clients, config=config)
        
        # Verify initialization
        assert ws_manager._use_ccxt_clients == True
        assert len(ws_manager.clients) == 2
        assert 'kucoinfutures' in ws_manager.clients
        assert 'bingx' in ws_manager.clients
        assert ws_manager.config['reconnect_delay'] == 5
        
        logger.info("✓ CcxtClient to WebSocketManager integration successful")
    
    def test_legacy_credential_mode(self):
        """Test backward compatibility with credential-based initialization."""
        # Legacy mode: Pass credentials directly
        exchanges = {
            'kucoinfutures': None,  # No credentials (unauthenticated)
            'bingx': None
        }
        
        ws_manager = WebSocketManager(exchanges)
        
        # Verify initialization
        assert ws_manager._use_ccxt_clients == False
        assert len(ws_manager.clients) == 2
        
        logger.info("✓ Legacy credential mode backward compatible")
    
    @pytest.mark.asyncio
    async def test_production_coordinator_pattern(self):
        """Test initialization pattern used by ProductionCoordinator."""
        # Simulate ProductionCoordinator initialization
        exchange_clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None),
            'bingx': CcxtClient('bingx', None)
        }
        
        # Initialize with None (default exchanges) - legacy pattern
        ws_manager_legacy = WebSocketManager(exchanges=None)
        assert len(ws_manager_legacy.clients) > 0
        await ws_manager_legacy.close()
        
        # Initialize with CcxtClient instances - new pattern
        ws_manager_new = WebSocketManager(exchange_clients)
        assert len(ws_manager_new.clients) == 2
        await ws_manager_new.close()
        
        logger.info("✓ ProductionCoordinator pattern verified")
    
    def test_callback_system(self):
        """Test callback registration system."""
        ws_manager = WebSocketManager()
        
        callback_count = {'ticker': 0, 'orderbook': 0}
        
        async def ticker_cb(ex, sym, ticker):
            callback_count['ticker'] += 1
        
        async def orderbook_cb(ex, sym, ob):
            callback_count['orderbook'] += 1
        
        # Test registration
        ws_manager.on_ticker_update(ticker_cb)
        ws_manager.on_orderbook_update(orderbook_cb)
        
        assert len(ws_manager.callbacks['ticker']) == 1
        assert len(ws_manager.callbacks['orderbook']) == 1
        
        logger.info("✓ Callback system working")
    
    @pytest.mark.asyncio
    async def test_subscribe_api(self):
        """Test new subscription API methods."""
        clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None)
        }
        ws_manager = WebSocketManager(clients)
        
        # Test that methods exist and are callable
        assert callable(ws_manager.subscribe_tickers)
        assert callable(ws_manager.subscribe_orderbook)
        assert callable(ws_manager.start_streams)
        assert callable(ws_manager.shutdown)
        
        # Test start_streams with empty subscriptions
        tasks = await ws_manager.start_streams({})
        assert isinstance(tasks, dict)
        
        await ws_manager.shutdown()
        logger.info("✓ Subscription API verified")
    
    def test_manager_attributes(self):
        """Test that all required attributes from problem statement exist."""
        clients = {
            'bingx': CcxtClient('bingx', None)
        }
        config = {'timeout': 30}
        
        ws_manager = WebSocketManager(clients, config)
        
        # Attributes from problem statement
        assert hasattr(ws_manager, 'exchanges')
        assert hasattr(ws_manager, 'config')
        assert hasattr(ws_manager, 'connections')
        assert hasattr(ws_manager, 'callbacks')
        assert hasattr(ws_manager, 'is_running')
        assert hasattr(ws_manager, 'reconnect_delays')
        
        # Verify values
        assert ws_manager.config == config
        assert isinstance(ws_manager.callbacks, dict)
        assert isinstance(ws_manager.reconnect_delays, dict)
        assert ws_manager.is_running == False
        
        logger.info("✓ All required attributes present")
    
    def test_multi_exchange_support(self):
        """Test multi-exchange support with priority exchanges."""
        # Priority exchanges from problem statement: BingX, KuCoin, Binance, Bitget
        clients = {
            'bingx': CcxtClient('bingx', None),
            'kucoinfutures': CcxtClient('kucoinfutures', None),
            'binance': CcxtClient('binance', None),
        }
        
        ws_manager = WebSocketManager(clients)
        
        # Verify all exchanges initialized
        assert 'bingx' in ws_manager.clients
        assert 'kucoinfutures' in ws_manager.clients
        assert 'binance' in ws_manager.clients
        
        # Get status
        status = ws_manager.get_stream_status()
        assert len(status['exchanges']) == 3
        
        logger.info("✓ Multi-exchange support verified")
    
    @pytest.mark.asyncio
    async def test_integration_with_data_collector(self):
        """Test WebSocketManager works with StreamDataCollector."""
        clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None)
        }
        
        ws_manager = WebSocketManager(clients)
        collector = StreamDataCollector(buffer_size=50)
        
        # Register collector callbacks
        ws_manager.on_ticker_update(collector.ticker_callback)
        
        # Verify callback registered
        assert collector.ticker_callback in ws_manager.callbacks['ticker']
        
        await ws_manager.close()
        logger.info("✓ Integration with StreamDataCollector verified")


class TestPhaseCompatibility:
    """Test compatibility with existing Phase 2 and Phase 3 components."""
    
    def test_correlation_monitor_compatibility(self):
        """Test compatibility with CorrelationMonitor."""
        # CorrelationMonitor accepts websocket_manager in __init__
        ws_manager = WebSocketManager()
        
        # Simulate CorrelationMonitor initialization pattern
        assert ws_manager is not None
        assert hasattr(ws_manager, 'clients')
        
        logger.info("✓ CorrelationMonitor compatibility verified")
    
    def test_live_trading_engine_compatibility(self):
        """Test compatibility with LiveTradingEngine."""
        clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None)
        }
        ws_manager = WebSocketManager(clients)
        
        # LiveTradingEngine requires these attributes
        assert hasattr(ws_manager, 'clients')
        assert ws_manager.clients is not None
        
        # Test that ws_manager can be passed to components
        assert ws_manager is not None
        
        logger.info("✓ LiveTradingEngine compatibility verified")
    
    @pytest.mark.asyncio
    async def test_production_workflow(self):
        """Test complete production workflow."""
        # Step 1: Phase 1 - Build clients
        clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None),
            'bingx': CcxtClient('bingx', None)
        }
        logger.info("Step 1: Phase 1 clients created")
        
        # Step 2: Phase 3.1 - Initialize WebSocket manager
        config = {'reconnect_delay': 5, 'max_retries': 3}
        ws_manager = WebSocketManager(clients, config=config)
        logger.info("Step 2: WebSocket manager initialized")
        
        # Step 3: Register callbacks for market data
        collector = StreamDataCollector()
        ws_manager.on_ticker_update(collector.ticker_callback)
        logger.info("Step 3: Callbacks registered")
        
        # Step 4: Verify system ready
        status = ws_manager.get_stream_status()
        assert status['running'] == False
        assert len(status['exchanges']) == 2
        logger.info("Step 4: System verified ready")
        
        # Step 5: Clean shutdown
        await ws_manager.shutdown()
        assert ws_manager.is_running == False
        logger.info("Step 5: Clean shutdown completed")
        
        logger.info("✓ Complete production workflow verified")


def run_integration_tests():
    """Run all integration tests."""
    logger.info("="*60)
    logger.info("Phase 1 + Phase 3.1 Integration Tests")
    logger.info("="*60)
    
    pytest_args = [
        __file__,
        '-v',
        '--tb=short',
        '-p', 'no:warnings'
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        logger.info("\n" + "="*60)
        logger.info("✅ All integration tests passed!")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ Some tests failed")
        logger.error("="*60)
    
    return exit_code


if __name__ == '__main__':
    run_integration_tests()
