import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.core.position_manager import AdvancedPositionManager

class TestPositionSync:
    @pytest.fixture
    def manager(self):
        risk_manager = MagicMock()
        order_manager = MagicMock()
        manager = AdvancedPositionManager(risk_manager, order_manager)
        return manager

    @pytest.mark.asyncio
    async def test_sync_positions_bingx(self, manager):
        """Test syncing positions from BingX."""
        # Mock BingX client
        bingx_client = MagicMock()
        bingx_client.get_bingx_positions.return_value = {
            'code': 0,
            'data': [
                {
                    'symbol': 'BTC-USDT',
                    'positionAmt': '0.001',
                    'avgPrice': '50000',
                    'unrealizedProfit': '10'
                }
            ]
        }
        
        exchange_clients = {'bingx': bingx_client}
        
        await manager.sync_positions(exchange_clients)
        
        # Verify position imported
        assert len(manager.positions) == 1
        pos = list(manager.positions.values())[0]
        assert pos['symbol'] == 'BTC/USDT:USDT'
        assert pos['amount'] == 0.001
        assert pos['entry_price'] == 50000.0
        assert pos['exchange'] == 'bingx'
        assert pos['imported'] is True

    @pytest.mark.asyncio
    async def test_sync_positions_ccxt(self, manager):
        """Test syncing positions from generic CCXT."""
        # Mock CCXT client
        ccxt_client = MagicMock()
        ccxt_client.fetch_positions = MagicMock(return_value=[
            {
                'symbol': 'ETH/USDT:USDT',
                'contracts': 1.5,
                'entryPrice': 3000,
                'unrealizedPnl': 50,
                'side': 'long'
            }
        ])
        
        exchange_clients = {'binance': ccxt_client}
        
        await manager.sync_positions(exchange_clients)
        
        # Verify position imported
        assert len(manager.positions) == 1
        pos = list(manager.positions.values())[0]
        assert pos['symbol'] == 'ETH/USDT:USDT'
        assert pos['amount'] == 1.5
        assert pos['entry_price'] == 3000.0
        assert pos['exchange'] == 'binance'
        assert pos['imported'] is True

    @pytest.mark.asyncio
    async def test_sync_positions_empty(self, manager):
        """Test syncing with no positions."""
        bingx_client = MagicMock()
        bingx_client.get_bingx_positions.return_value = {'code': 0, 'data': []}
        
        exchange_clients = {'bingx': bingx_client}
        
        await manager.sync_positions(exchange_clients)
        
        assert len(manager.positions) == 0
