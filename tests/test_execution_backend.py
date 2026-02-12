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


class FakeLimitTimeoutCcxtClient(FakeCcxtClient):
    def create_order(self, symbol: str, side: str, type_: str, amount: float, price=None, params=None):
        self.create_order_calls.append(
            {"symbol": symbol, "side": side, "type": type_, "amount": amount, "price": price, "params": params or {}}
        )
        if type_ == "limit":
            return {"id": "exch-limit-1", "status": "open", "filled": 0.0, "amount": amount, "price": price}
        return {"id": "exch-market-1", "average": 100.5, "filled": amount, "status": "closed"}

    def fetch_order(self, order_id: str, symbol: str = None, params=None):
        return {"id": order_id, "status": "open", "filled": 0.0, "amount": 0.01, "price": 100.0}

    def cancel_order(self, order_id: str, symbol: str = None, params=None):
        self.cancel_order_calls.append({"order_id": order_id, "symbol": symbol, "params": params or {}})
        return {"id": order_id, "status": "canceled"}

    def fetch_positions(self, symbols=None, params=None):
        # No position delta by default.
        return [{"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.0}]


class FakeLimitTimeoutChaseGateCcxtClient(FakeLimitTimeoutCcxtClient):
    def ticker(self, symbol: str):
        # Adverse move after timeout to trigger chase gate abort for LONG entries.
        return {"last": 101.0}


class FakeLimitTimeoutRaceFillCcxtClient(FakeLimitTimeoutCcxtClient):
    def __init__(self, filled_qty: float = 0.01):
        super().__init__()
        self._pos_fetch_count = 0
        self._filled_qty = filled_qty

    def fetch_order(self, order_id: str, symbol: str = None, params=None):
        # Simulate stale order status response during cancel race.
        return {"id": order_id, "status": "open", "filled": 0.0, "amount": 0.01, "price": 100.0}

    def fetch_positions(self, symbols=None, params=None):
        self._pos_fetch_count += 1
        qty = 0.0 if self._pos_fetch_count == 1 else self._filled_qty
        return [{"symbol": "BTC/USDT:USDT", "side": "short", "contracts": qty}]


class FakeLimitTimeoutNoVerificationCcxtClient(FakeLimitTimeoutCcxtClient):
    def fetch_positions(self, symbols=None, params=None):
        raise Exception("positions unavailable")


class FakeLimitTimeoutSoftGateBlockCcxtClient(FakeLimitTimeoutCcxtClient):
    def ticker(self, symbol: str):
        # Wide spread + adverse move to force soft-gate block.
        return {"last": 99.0, "bid": 98.5, "ask": 100.5}


class FakeLimitTimeoutSoftGateAllowCcxtClient(FakeLimitTimeoutCcxtClient):
    def ticker(self, symbol: str):
        # Tight spread + no adverse drift => soft-gate allows fallback.
        return {"last": 100.0, "bid": 99.99, "ask": 100.01}


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


def test_limit_timeout_market_fallback_emits_reason(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutCcxtClient()
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
                "execution_params": {"timeout_seconds": 0},
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is True
    assert result.get("fallback_reason") == "limit_timeout_market_fallback"
    assert result.get("requested_order_type") == "limit"
    assert result.get("effective_order_type") == "market"
    assert [c["type"] for c in client.create_order_calls] == ["limit", "market"]


def test_limit_timeout_market_fallback_can_be_disabled_by_flag(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutCcxtClient()
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
                    "market_fallback_on_timeout_enabled": False,
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is False
    assert result.get("reason") == "ABORT:NO_FILL_TIMEOUT"
    assert result.get("fallback_reason") == "limit_timeout_market_fallback_disabled:flag"
    assert result.get("requested_order_type") == "limit"
    assert result.get("effective_order_type") == "limit"
    assert [c["type"] for c in client.create_order_calls] == ["limit"]


def test_limit_timeout_market_fallback_disabled_on_extreme_bucket(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "amount": 0.01,
                "exchange": "bingx",
                "signal": {"entry": 100.0, "stop": 90.0, "volume_bucket": "EXTREME"},
                "limit_price": 99.5,
                "execution_params": {
                    "timeout_seconds": 0,
                    "market_fallback_on_timeout_enabled": True,
                    "disable_market_fallback_on_extreme_bucket": True,
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is False
    assert result.get("reason") == "ABORT:NO_FILL_TIMEOUT"
    assert result.get("fallback_reason") == "limit_timeout_market_fallback_disabled:extreme_bucket"
    assert result.get("requested_order_type") == "limit"
    assert result.get("effective_order_type") == "limit"
    assert [c["type"] for c in client.create_order_calls] == ["limit"]


def test_limit_timeout_aborts_on_chase_gate_before_market_fallback(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutChaseGateCcxtClient()
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
                    "max_chase_bps": 5.0,
                    "market_fallback_on_timeout_enabled": True,
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is False
    assert str(result.get("reason") or "").startswith("ABORT:CHASE_GATE:")
    assert [c["type"] for c in client.create_order_calls] == ["limit"]


def test_limit_timeout_skip_market_when_position_delta_indicates_fill(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutRaceFillCcxtClient(filled_qty=0.01)
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "short",
                "amount": 0.01,
                "exchange": "bingx",
                "signal": {"entry": 100.0, "stop": 110.0},
                "limit_price": 100.5,
                "execution_params": {
                    "timeout_seconds": 0,
                    "market_fallback_on_timeout_enabled": True,
                    "fallback_settle_checks": 1,
                    "fallback_cancel_settle_ms": 0,
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is True
    assert result.get("fallback_reason") == "limit_timeout_skip_market_position_delta"
    assert result.get("effective_order_type") == "limit"
    assert result.get("filled_amount") == 0.01
    assert [c["type"] for c in client.create_order_calls] == ["limit"]


def test_limit_timeout_market_fallback_uses_only_residual_amount(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutRaceFillCcxtClient(filled_qty=0.004)
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "short",
                "amount": 0.01,
                "exchange": "bingx",
                "signal": {"entry": 100.0, "stop": 110.0},
                "limit_price": 100.5,
                "execution_params": {
                    "timeout_seconds": 0,
                    "market_fallback_on_timeout_enabled": True,
                    "fallback_settle_checks": 1,
                    "fallback_cancel_settle_ms": 0,
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is True
    assert result.get("effective_order_type") == "market"
    assert abs(float(result.get("fallback_residual_qty") or 0.0) - 0.006) < 1e-9
    assert [c["type"] for c in client.create_order_calls] == ["limit", "market"]
    assert abs(float(client.create_order_calls[-1]["amount"]) - 0.006) < 1e-9


def test_limit_timeout_market_fallback_aborts_when_position_delta_unverified(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutNoVerificationCcxtClient()
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
                    "market_fallback_on_timeout_enabled": True,
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is False
    assert result.get("reason") == "ABORT:NO_FILL_TIMEOUT_UNVERIFIED"
    assert result.get("fallback_reason") == "limit_timeout_market_fallback_unverified:position_delta"
    assert result.get("requested_order_type") == "limit"
    assert result.get("effective_order_type") == "limit"
    assert [c["type"] for c in client.create_order_calls] == ["limit"]


def test_limit_timeout_market_fallback_soft_gate_blocks_when_score_below_threshold(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutSoftGateBlockCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "amount": 0.01,
                "exchange": "bingx",
                "signal": {"entry": 100.0, "stop": 90.0, "target": 100.5},
                "limit_price": 99.5,
                "execution_params": {
                    "timeout_seconds": 0,
                    "market_fallback_on_timeout_enabled": True,
                },
                "_internal": {
                    "fallback_soft_gate": {
                        "enabled": True,
                        "min_passes": 2,
                        "rr_min": 1.2,
                        "max_adverse_bps": 50.0,
                        "max_spread_bps": 8.0,
                        "fail_closed_on_insufficient_context": False,
                    }
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is False
    assert str(result.get("reason") or "").startswith("ABORT:FALLBACK_SOFT_GATE:")
    assert result.get("fallback_reason") == "limit_timeout_market_fallback_soft_gate_blocked"
    assert result.get("effective_order_type") == "limit"
    assert [c["type"] for c in client.create_order_calls] == ["limit"]


def test_limit_timeout_market_fallback_soft_gate_allows_when_min_passes_met(clean_env):
    from src.core.order_manager import SmartOrderManager

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    client = FakeLimitTimeoutSoftGateAllowCcxtClient()
    om = SmartOrderManager(market_data_pipeline=None, exchange_clients={"bingx": client})

    result = asyncio.run(
        om.place_order(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "amount": 0.01,
                "exchange": "bingx",
                "signal": {"entry": 100.0, "stop": 90.0},  # edge_preserved intentionally NA (no target)
                "limit_price": 99.5,
                "execution_params": {
                    "timeout_seconds": 0,
                    "market_fallback_on_timeout_enabled": True,
                },
                "_internal": {
                    "fallback_soft_gate": {
                        "enabled": True,
                        "min_passes": 2,
                        "rr_min": 1.2,
                        "max_adverse_bps": 50.0,
                        "max_spread_bps": 8.0,
                        "fail_closed_on_insufficient_context": False,
                    }
                },
            },
            execution_algo="limit",
        )
    )

    assert result["success"] is True
    assert result.get("fallback_reason") == "limit_timeout_market_fallback"
    assert result.get("effective_order_type") == "market"
    assert isinstance(result.get("fallback_soft_gate"), dict)
    assert result["fallback_soft_gate"].get("reason") == "fallback_soft_gate_pass"
    assert [c["type"] for c in client.create_order_calls] == ["limit", "market"]


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
    assert result.get("env_forced_order_type") == "market"
