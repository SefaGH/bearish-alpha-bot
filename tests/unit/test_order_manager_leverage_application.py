import pytest

from src.core.order_manager import SmartOrderManager
from src.core.market_data_pipeline import MarketDataPipeline


class _StubMarketDataPipeline(MarketDataPipeline):
    def __init__(self):
        # OrderManager's market execution path doesn't use the pipeline.
        pass

    async def get_market_metadata(self, *_args, **_kwargs):
        return {}


class _StubBingxClient:
    name = "bingx"

    def __init__(self):
        self.set_leverage_calls = []
        self.create_order_calls = []
        self.ensure_hedge_calls = 0
        self.loaded_markets = False

    def ticker(self, _symbol):
        return {"last": 100.0}

    def load_markets(self):
        self.loaded_markets = True

    def ensure_bingx_hedge_mode(self, _symbol, require_hedged=False):
        self.ensure_hedge_calls += 1
        if require_hedged:
            return True
        return None

    def set_leverage(self, symbol, leverage):
        self.set_leverage_calls.append((symbol, leverage))
        return {"ok": True}

    def create_order(self, *, symbol, side, type_, amount, price, params):
        self.create_order_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "type_": type_,
                "amount": amount,
                "price": price,
                "params": params,
            }
        )
        return {
            "id": "oid-1",
            "status": "closed",
            "average": 100.0,
            "filled": amount,
            "amount": amount,
            "price": 100.0,
        }


@pytest.mark.asyncio
async def test_market_entry_sets_exchange_leverage(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_BACKEND", "ccxt")
    monkeypatch.setenv("BINGX_ENV", "vst")

    client = _StubBingxClient()
    om = SmartOrderManager(
        market_data_pipeline=_StubMarketDataPipeline(),
        exchange_clients={"bingx": client},
    )

    order_request = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.01,
        "exchange": "bingx",
        "signal": {"leverage": 7},
        "execution_params": {},
    }

    result = await om.place_order(order_request, execution_algo="market")
    assert result["success"] is True
    assert client.set_leverage_calls == [("BTC/USDT:USDT", 7)]


@pytest.mark.asyncio
async def test_reduce_only_market_does_not_set_leverage(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_BACKEND", "ccxt")
    monkeypatch.setenv("BINGX_ENV", "vst")

    client = _StubBingxClient()
    om = SmartOrderManager(
        market_data_pipeline=_StubMarketDataPipeline(),
        exchange_clients={"bingx": client},
    )

    order_request = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "amount": 0.01,
        "exchange": "bingx",
        "signal": {"leverage": 10},
        "execution_params": {"reduceOnly": True, "positionSide": "LONG"},
    }

    result = await om.place_order(order_request, execution_algo="market")
    assert result["success"] is True
    assert client.set_leverage_calls == []
