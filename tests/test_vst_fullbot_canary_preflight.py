import asyncio
import os


class _DummyBingxEx:
    def __init__(self, state):
        self._state = state
        self.sandbox = True
        self.urls = {"api": {"swap": "https://open-api-vst.bingx.com/swap"}}

    def fetch_open_orders(self, symbol):
        return list(self._state.get("open_orders", []))


class _DummyBingxClient:
    def __init__(self, state):
        self._state = state
        self.ex = _DummyBingxEx(state)
        self._bingx_rest_base_url = "https://open-api-vst.bingx.com"
        self._bingx_is_hedged = True

    def _get_bingx_native_symbol(self, symbol: str) -> str:
        return symbol.split(":")[0].replace("/", "-")

    def ensure_bingx_hedge_mode(self, symbol: str, require_hedged: bool = False):
        return True

    def get_bingx_positions(self, symbol: str = None):
        return {"code": 0, "data": list(self._state.get("positions", []))}

    def cancel_order(self, order_id: str, symbol: str = None, params=None):
        self._state["open_orders"] = [o for o in self._state.get("open_orders", []) if o.get("id") != order_id]
        return {"id": order_id, "status": "canceled"}

    def create_order(self, symbol: str, side: str, type_: str, amount: float, price=None, params=None):
        position_side = (params or {}).get("positionSide")
        if position_side in ("LONG", "SHORT"):
            for item in self._state.get("positions", []):
                if item.get("positionSide") == position_side:
                    item["positionAmt"] = "0"
        return {"id": "oid-close", "status": "closed"}


def test_vst_fullbot_canary_preflight_fails_dirty_state_without_cleanup(clean_env):
    from core.production_coordinator import ProductionCoordinator

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    state = {
        "open_orders": [{"id": "oid-1", "symbol": "BTC-USDT", "status": "open"}],
        "positions": [{"symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "0"}],
    }

    coord = ProductionCoordinator.__new__(ProductionCoordinator)
    coord.exchange_clients = {"bingx": _DummyBingxClient(state)}

    result = asyncio.run(coord._vst_fullbot_canary_preflight("BTC/USDT:USDT", allow_cleanup=False))
    assert result["ok"] is False
    assert any("dirty_state" in str(err) for err in (result.get("errors") or []))


def test_vst_fullbot_canary_preflight_cleans_dirty_state_with_cleanup(clean_env):
    from core.production_coordinator import ProductionCoordinator

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    state = {
        "open_orders": [{"id": "oid-1", "symbol": "BTC-USDT", "status": "open"}],
        "positions": [{"symbol": "BTC-USDT", "positionSide": "SHORT", "positionAmt": "0.01"}],
    }

    coord = ProductionCoordinator.__new__(ProductionCoordinator)
    coord.exchange_clients = {"bingx": _DummyBingxClient(state)}

    result = asyncio.run(coord._vst_fullbot_canary_preflight("BTC/USDT:USDT", allow_cleanup=True))
    assert result["ok"] is True
    assert state["open_orders"] == []
    assert state["positions"][0]["positionAmt"] == "0"
