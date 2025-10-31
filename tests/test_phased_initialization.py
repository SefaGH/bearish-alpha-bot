"""
Test phased initialization flow for ProductionCoordinator.

Tests the new phased initialization:
- Phase 1: Core Systems
- Phase 1.5: Data Layer Health Check
- Phase 2: ML Systems
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.production_coordinator import ProductionCoordinator


@pytest.fixture
def mock_exchange_clients():
    """Create mock exchange clients."""
    mock_client = Mock()
    mock_client.ex = Mock()
    mock_client.ex.apiKey = None
    return {'bingx': mock_client}


@pytest.fixture
def mock_websocket_manager():
    """Create mock WebSocket manager."""
    mock_ws = Mock()
    mock_ws.is_any_client_connected = Mock(return_value=True)
    mock_ws.get_active_stream_count = Mock(return_value=3)
    mock_ws.is_collector_ready = Mock(return_value=True)
    mock_ws.collector = Mock()
    mock_ws.get_latest_data = Mock(return_value={
        'ohlcv': [[1000, 100, 101, 99, 100.5, 1000]],
        'symbol': 'BTC/USDT:USDT',
        'timeframe': '1m'
    })
    return mock_ws


@pytest.fixture
def coordinator():
    """Create ProductionCoordinator instance."""
    return ProductionCoordinator()


@pytest.mark.asyncio
async def test_initialize_core_systems(coordinator, mock_exchange_clients, mock_websocket_manager):
    """Test Phase 1: Core systems initialization."""
    
    portfolio_config = {
        'equity_usd': 100.0,
        'max_portfolio_risk': 0.05,
        'max_position_size': 0.20,
        'max_drawdown': 0.10
    }
    
    trading_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    
    result = await coordinator.initialize_core_systems(
        exchange_clients=mock_exchange_clients,
        portfolio_config=portfolio_config,
        mode='paper',
        trading_symbols=trading_symbols,
        websocket_manager=mock_websocket_manager
    )
    
    # Verify core systems initialized
    assert result['success'] is True
    assert 'components' in result
    assert len(result['components']) > 0
    assert coordinator.is_initialized == False  # Not fully initialized yet (no ML)
    assert coordinator.websocket_manager is mock_websocket_manager
    assert coordinator.active_symbols == trading_symbols
    
    # Verify core components exist
    assert coordinator.market_data_pipeline is not None
    assert coordinator.risk_manager is not None
    assert coordinator.portfolio_manager is not None
    assert coordinator.trading_engine is not None


@pytest.mark.asyncio
async def test_data_layer_health_check_healthy(coordinator, mock_exchange_clients, mock_websocket_manager):
    """Test Phase 1.5: Data layer health check - healthy case."""
    
    # Initialize core systems first
    portfolio_config = {'equity_usd': 100.0}
    trading_symbols = ['BTC/USDT:USDT']
    
    await coordinator.initialize_core_systems(
        exchange_clients=mock_exchange_clients,
        portfolio_config=portfolio_config,
        mode='paper',
        trading_symbols=trading_symbols,
        websocket_manager=mock_websocket_manager
    )
    
    # Perform health check
    health_result = await coordinator.is_data_layer_healthy()
    
    # Verify health check results
    assert 'healthy' in health_result
    assert 'checks' in health_result
    assert 'websocket_connection' in health_result['checks']
    assert 'subscriptions' in health_result['checks']
    assert 'data_flow' in health_result['checks']
    
    # Should be healthy with mocked data
    assert health_result['healthy'] is True
    assert health_result['checks']['websocket_connection']['status'] == 'healthy'
    assert health_result['checks']['subscriptions']['status'] == 'healthy'


@pytest.mark.asyncio
async def test_data_layer_health_check_no_websocket(coordinator):
    """Test Phase 1.5: Data layer health check - no WebSocket."""
    
    # Don't initialize core systems - no WebSocket manager
    coordinator.websocket_manager = None
    coordinator.active_symbols = ['BTC/USDT:USDT']
    
    # Perform health check
    health_result = await coordinator.is_data_layer_healthy()
    
    # Should report unhealthy due to missing WebSocket
    assert health_result['healthy'] is False
    assert health_result['checks']['websocket_connection']['status'] == 'not_available'


@pytest.mark.asyncio
async def test_initialize_ml_systems(coordinator, mock_exchange_clients, mock_websocket_manager):
    """Test Phase 2: ML systems initialization."""
    
    # Initialize core systems first
    portfolio_config = {'equity_usd': 100.0}
    trading_symbols = ['BTC/USDT:USDT']
    
    await coordinator.initialize_core_systems(
        exchange_clients=mock_exchange_clients,
        portfolio_config=portfolio_config,
        mode='paper',
        trading_symbols=trading_symbols,
        websocket_manager=mock_websocket_manager
    )
    
    # Mock ML components
    mock_price_engine = Mock()
    mock_regime_predictor = Mock()
    
    # Initialize ML systems
    result = await coordinator.initialize_ml_systems(
        price_engine=mock_price_engine,
        regime_predictor=mock_regime_predictor
    )
    
    # Verify ML initialization attempted
    assert 'success' in result
    assert 'components' in result


@pytest.mark.asyncio
async def test_full_phased_initialization(coordinator, mock_exchange_clients, mock_websocket_manager):
    """Test complete phased initialization flow."""
    
    portfolio_config = {'equity_usd': 100.0}
    trading_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    
    # Phase 1: Core Systems
    core_result = await coordinator.initialize_core_systems(
        exchange_clients=mock_exchange_clients,
        portfolio_config=portfolio_config,
        mode='paper',
        trading_symbols=trading_symbols,
        websocket_manager=mock_websocket_manager
    )
    assert core_result['success'] is True
    
    # Phase 1.5: Health Check
    health_result = await coordinator.is_data_layer_healthy()
    assert 'healthy' in health_result
    
    # Phase 2: ML Systems (only if healthy)
    # Note: In production, system proceeds even if unhealthy (REST fallback)
    # For testing, we always proceed to test ML initialization
    mock_price_engine = Mock()
    mock_regime_predictor = Mock()
    
    ml_result = await coordinator.initialize_ml_systems(
        price_engine=mock_price_engine,
        regime_predictor=mock_regime_predictor
    )
    assert 'success' in ml_result


@pytest.mark.asyncio
async def test_orchestrated_initialization(coordinator, mock_exchange_clients, mock_websocket_manager):
    """Test orchestrated initialization using the main method."""
    
    portfolio_config = {'equity_usd': 100.0}
    trading_symbols = ['BTC/USDT:USDT']
    mock_price_engine = Mock()
    mock_regime_predictor = Mock()
    
    # Call orchestrated method
    result = await coordinator.initialize_production_system(
        exchange_clients=mock_exchange_clients,
        portfolio_config=portfolio_config,
        mode='paper',
        trading_symbols=trading_symbols,
        websocket_manager=mock_websocket_manager,
        price_engine=mock_price_engine,
        regime_predictor=mock_regime_predictor
    )
    
    # Verify successful initialization
    assert result['success'] is True
    assert 'components' in result
    assert 'health_check' in result
    assert coordinator.is_initialized is True


def test_coordinator_has_new_methods(coordinator):
    """Test that coordinator has the new phased methods."""
    assert hasattr(coordinator, 'initialize_core_systems')
    assert hasattr(coordinator, 'is_data_layer_healthy')
    assert hasattr(coordinator, 'initialize_ml_systems')
    assert callable(coordinator.initialize_core_systems)
    assert callable(coordinator.is_data_layer_healthy)
    assert callable(coordinator.initialize_ml_systems)
