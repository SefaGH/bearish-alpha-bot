import os
import asyncio


class FakeCcxtClient:
    def __init__(self):
        self.name = "bingx"
        self.create_order_calls = []
        self.cancel_order_calls = []
        self.load_markets_calls = 0
        self.hedge_mode_calls = []

    def ticker(self, symbol: str):
        return {"last": 100.0}

    def load_markets(self, *args, **kwargs):
        self.load_markets_calls += 1
        return {}

    def ensure_bingx_hedge_mode(self, symbol: str, require_hedged: bool = False):
        self.hedge_mode_calls.append((symbol, require_hedged))
        return True

    def create_order(self, symbol: str, side: str, type_: str, amount: float, price=None, params=None):
        self.create_order_calls.append(
            {"symbol": symbol, "side": side, "type": type_, "amount": amount, "price": price, "params": params or {}}
        )
        return {"id": "exch-1", "average": 100.5, "filled": amount, "status": "closed"}

    def fetch_order(self, order_id: str, symbol: str = None, params=None):
        # Default: assume already closed/filled.
        return {"id": order_id, "average": 100.5, "filled": 0.01, "status": "closed"}

    def cancel_order(self, order_id: str, symbol: str = None, params=None):
        self.cancel_order_calls.append({"order_id": order_id, "symbol": symbol, "params": params or {}})
        raise Exception("Order not found")


def test_market_order_simulated_by_default(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ.pop("EXECUTION_BACKEND", None)
    os.environ.pop("BINGX_ENV", None)

    client = FakeCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})
    result = asyncio.run(
        om.place_order(
            {"symbol": "BTC/USDT:USDT", "side": "buy", "amount": 0.01, "exchange": "bingx"},
            execution_algo="market",
        )
    )

    assert result["success"] is True
    assert str(result["order_id"]).startswith("order_")
    assert client.create_order_calls == []


def test_market_order_real_execution_calls_ccxt(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "amount": 0.01,
                "exchange": "bingx",
                "params": {"reduceOnly": False, "foo": 1},
                "execution_params": {"reduceOnly": True},
            },
            execution_algo="market",
        )
    )

    assert result["success"] is True
    assert result["order_id"] == "exch-1"
    assert client.create_order_calls, "Expected create_order to be called in real execution mode"
    assert client.create_order_calls[-1]["side"] == "buy"
    assert client.create_order_calls[-1]["type"] == "market"
    assert client.create_order_calls[-1]["params"]["reduceOnly"] is True


def test_limit_order_real_execution_calls_ccxt(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "amount": 0.01,
                "exchange": "bingx",
                "signal": {"entry": 100.0, "stop": 90.0},
                "limit_price": 99.5,
                "execution_params": {
                    "timeout_seconds": 0,
                    "max_chase_bps": 12.0,
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is True
    assert result["order_id"] == "exch-1"
    assert client.create_order_calls, "Expected create_order to be called"
    assert client.create_order_calls[-1]["type"] == "limit"
    assert client.create_order_calls[-1]["side"] == "buy"
    assert client.create_order_calls[-1]["price"] == 99.5


def test_real_cancel_is_idempotent(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {"symbol": "BTC/USDT:USDT", "side": "buy", "amount": 0.01, "exchange": "bingx"},
            execution_algo="market",
        )
    )
    assert result["success"] is True
    order_id = result["order_id"]
    assert order_id in om.active_orders

    cancel = asyncio.run(om.cancel_order(order_id, "bingx"))
    assert cancel["success"] is True
    assert order_id not in om.active_orders
    assert client.cancel_order_calls


def test_real_execution_requires_explicit_bingx_env(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ.pop("BINGX_ENV", None)

    client = FakeCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {"symbol": "BTC/USDT:USDT", "side": "buy", "amount": 0.01, "exchange": "bingx"},
            execution_algo="market",
        )
    )

    assert result["success"] is False
    assert "BINGX_ENV" in (result.get("reason") or "")


def test_vst_fullbot_canary_forces_market_execution(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"
    os.environ["VST_FULLBOT_CANARY"] = "true"

    client = FakeCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {"symbol": "BTC/USDT:USDT", "side": "buy", "amount": 0.01, "exchange": "bingx"},
            execution_algo="limit",
        )
    )

    assert result["success"] is True
    assert client.create_order_calls, "Expected create_order to be called in canary real execution mode"
    assert client.create_order_calls[-1]["type"] == "market"
