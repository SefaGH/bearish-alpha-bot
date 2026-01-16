import os

import pytest

from src.core.position_manager import AdvancedPositionManager


class _StubOrderManager:
    def __init__(self):
        self.last_order_request = None
        self.calls = 0

    async def place_order(self, order_request, execution_algo="market", exchange_clients=None):
        self.calls += 1
        self.last_order_request = order_request
        return {"success": True, "avg_price": 123.45, "order_id": "oid-1"}


@pytest.mark.asyncio
async def test_shutdown_close_all_positions_sets_bingx_reduce_only_and_position_side(monkeypatch):
    # Force "real execution" branch used in VST demo (live+ccxt)
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_BACKEND", "ccxt")
    monkeypatch.setenv("BINGX_ENV", "vst")

    pm = AdvancedPositionManager.__new__(AdvancedPositionManager)
    pm.order_manager = _StubOrderManager()
    pm.positions = {
        "pos1": {
            "symbol": "BTC/USDT:USDT",
            "side": "short",
            "amount": 0.0025,
            "exchange": "bingx",
        }
    }

    async def _close_position_stub(*_args, **_kwargs):
        return {"success": True}

    pm.close_position = _close_position_stub

    result = await pm.close_all_positions(exchange_clients={"bingx": object()}, reason="shutdown")
    assert result["success"] is True
    assert pm.order_manager.calls == 1

    req = pm.order_manager.last_order_request
    assert req["side"] == "buy"  # close short
    assert req["exchange"] == "bingx"

    params = req.get("execution_params")
    assert isinstance(params, dict)
    assert params.get("reduceOnly") is True
    assert params.get("positionSide") == "SHORT"
