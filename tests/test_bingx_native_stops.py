import asyncio
import os
import time


class _DummyOrderManager:
    def __init__(self):
        self.exchange_clients = {}


class _DummyPortfolioManager:
    def __init__(self):
        self.exchange_clients = {}
        self.cfg = {}


class _DummyRiskManager:
    pass


def _make_pm():
    from core.position_manager import AdvancedPositionManager

    return AdvancedPositionManager(
        risk_manager=_DummyRiskManager(),
        order_manager=_DummyOrderManager(),
        websocket_manager=None,
        portfolio_manager=_DummyPortfolioManager(),
    )


def test_trailing_activation_threshold_is_honored(clean_env):
    pm = _make_pm()
    pos = {"trailing_stop_activation_threshold": 0.003, "trailing_stop_activated": False}

    active = pm._is_trailing_stop_active(pos, current_price=100.2, entry_price=100.0, is_long=True)
    assert active is False
    assert pos["trailing_stop_activated"] is False

    active = pm._is_trailing_stop_active(pos, current_price=100.3, entry_price=100.0, is_long=True)
    assert active is True
    assert pos["trailing_stop_activated"] is True


def test_native_hard_stop_suppresses_stop_loss_exit(clean_env):
    from core.position_manager import ExitReason

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"
    os.environ["BINGX_NATIVE_HARD_STOP_ENABLED"] = "true"

    pm = _make_pm()
    position_id = "pos_test_hard"
    pm.positions[position_id] = {
        "position_id": position_id,
        "exchange": "bingx",
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry_price": 100.0,
        "current_price": 98.0,
        "exit_price": 98.0,
        "amount": 1.0,
        "stop_loss": 99.0,
        "take_profit": 0.0,
        "trailing_stop_enabled": False,
        "trailing_stop_activated": False,
        "native_hard_stop_order_id": "oid-1",
        "native_trailing_stop_order_id": None,
        "open_timestamp": time.time() - 60.0,
        "native_exit_reconcile_last_ts": 0.0,
        "native_order_sync_last_ts": 0.0,
        "highest_price": 100.0,
        "lowest_price": 100.0,
    }

    result = asyncio.run(pm.manage_position_exits(position_id))
    assert result["should_exit"] is False
    assert result.get("reason") == "native_hard_stop_active"
    assert result.get("exit_reason") != ExitReason.STOP_LOSS.value


def test_native_trailing_suppresses_trailing_exit(clean_env):
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"
    os.environ["BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED"] = "true"

    pm = _make_pm()
    position_id = "pos_test_trailing"
    pm.positions[position_id] = {
        "position_id": position_id,
        "exchange": "bingx",
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry_price": 100.0,
        "current_price": 108.5,
        "exit_price": 108.5,
        "amount": 1.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "trailing_stop_enabled": True,
        "trailing_stop_distance": 0.01,
        "trailing_stop_activation_threshold": 0.0,
        "trailing_stop_activated": True,
        "native_hard_stop_order_id": None,
        "native_trailing_stop_order_id": "tid-1",
        "open_timestamp": time.time() - 60.0,
        "native_exit_reconcile_last_ts": 0.0,
        "native_order_sync_last_ts": 0.0,
        "highest_price": 110.0,
        "lowest_price": 100.0,
    }

    result = asyncio.run(pm.manage_position_exits(position_id))
    assert result["should_exit"] is False
    assert result.get("reason") == "native_trailing_active"


def test_native_exit_detection_sets_skip_market_exit(clean_env, monkeypatch):
    from core.position_manager import ExitReason

    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"
    os.environ["BINGX_NATIVE_HARD_STOP_ENABLED"] = "true"

    pm = _make_pm()
    position_id = "pos_test_exchange_closed"
    pm.positions[position_id] = {
        "position_id": position_id,
        "exchange": "bingx",
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry_price": 100.0,
        "current_price": 100.0,
        "exit_price": 100.0,
        "amount": 1.0,
        "stop_loss": 99.0,
        "take_profit": 0.0,
        "trailing_stop_enabled": False,
        "native_hard_stop_order_id": "oid-2",
        "open_timestamp": time.time() - 60.0,
        "native_exit_reconcile_last_ts": 0.0,
        "native_order_sync_last_ts": 0.0,
        "highest_price": 100.0,
        "lowest_price": 100.0,
    }

    async def _fake_is_open(_position):
        return False

    monkeypatch.setattr(pm, "_bingx_is_position_open_on_exchange", _fake_is_open)
    result = asyncio.run(pm.manage_position_exits(position_id))

    assert result["should_exit"] is True
    assert result.get("skip_market_exit") is True
    assert result.get("native_exit_detected") is True
    assert result.get("exit_reason") == ExitReason.STOP_LOSS.value


def test_bingx_exchange_size_readback_updates_local_amount(clean_env):
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    pm = _make_pm()

    class _DummyBingxClient:
        def _get_bingx_native_symbol(self, symbol: str) -> str:
            return "BTC-USDT"

        def get_bingx_positions(self, symbol: str = None):
            return {
                "code": 0,
                "data": [
                    {"symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "0.5"},
                ],
            }

    pm.portfolio_manager.exchange_clients = {"bingx": _DummyBingxClient()}

    position_id = "pos_size_readback"
    pm.positions[position_id] = {
        "position_id": position_id,
        "exchange": "bingx",
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "amount": 1.0,
        "native_order_sync_last_ts": time.time(),
    }

    is_open = asyncio.run(pm._bingx_is_position_open_on_exchange(pm.positions[position_id]))
    assert is_open is True
    assert pm.positions[position_id]["amount"] == 0.5
    assert pm.positions[position_id].get("native_order_sync_last_ts") == 0.0


def test_live_engine_preflight_skips_market_exit_when_exchange_flat(clean_env):
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    from core.live_trading_engine import LiveTradingEngine

    called = {"place_order": 0}

    class _DummyOrderManager:
        async def place_order(self, order_request, execution_algo="market"):
            called["place_order"] += 1
            return {"success": True, "order_id": "oid-exit", "avg_price": 100.0}

    class _DummyPositionManager:
        async def _bingx_is_position_open_on_exchange(self, position):
            return False

        async def close_position(self, position_id, exit_price, exit_reason):
            return {"success": True, "exit_price": exit_price, "exit_reason": exit_reason}

    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.order_manager = _DummyOrderManager()
    engine.position_manager = _DummyPositionManager()
    engine.active_positions = {
        "pos-1": {
            "position_id": "pos-1",
            "exchange": "bingx",
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "amount": 1.0,
            "entry_price": 100.0,
            "current_price": 99.0,
            "native_hard_stop_order_id": "oid-stop",
        }
    }

    result = asyncio.run(engine._execute_position_exit("pos-1", {"exit_reason": "manual"}))
    assert result["success"] is True
    assert result.get("skip_market_exit") is True
    assert result.get("preflight_skip_market_exit") is True
    assert called["place_order"] == 0
    assert "pos-1" not in engine.active_positions


def test_live_engine_exit_includes_reduceonly_and_positionside_for_bingx(clean_env):
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    from core.live_trading_engine import LiveTradingEngine

    captured = {"order": None}

    class _DummyOrderManager:
        async def place_order(self, order_request, execution_algo="market"):
            captured["order"] = order_request
            return {"success": True, "order_id": "oid-exit", "avg_price": 100.0}

    class _DummyPositionManager:
        async def _bingx_is_position_open_on_exchange(self, position):
            return True

        async def close_position(self, position_id, exit_price, exit_reason, **kwargs):
            return {"success": True, "exit_price": exit_price, "exit_reason": exit_reason}

    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.order_manager = _DummyOrderManager()
    engine.position_manager = _DummyPositionManager()
    engine.active_positions = {
        "pos-2": {
            "position_id": "pos-2",
            "exchange": "bingx",
            "symbol": "BTC/USDT:USDT",
            "side": "short",
            "amount": 1.0,
            "entry_price": 100.0,
            "current_price": 101.0,
            "native_trailing_stop_order_id": "oid-trail",
        }
    }

    result = asyncio.run(engine._execute_position_exit("pos-2", {"exit_reason": "manual"}))
    assert result["success"] is True
    assert captured["order"] is not None
    assert captured["order"]["side"] == "buy"
    assert captured["order"].get("execution_params") == {"reduceOnly": True, "positionSide": "SHORT"}
