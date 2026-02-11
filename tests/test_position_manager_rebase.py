from types import SimpleNamespace

import pytest

from src.core.position_manager import AdvancedPositionManager


class DummyRiskManager:
    def __init__(self, min_stop_pct=None):
        self.registered = {}
        self.risk_limits_dataclass = SimpleNamespace(
            stop_loss_pct=0.02,
            take_profit_ratio=2.0,
            min_stop_pct=min_stop_pct,
        )

    def register_position(self, position_id, position_data):
        self.registered[position_id] = position_data

    def close_position(self, position_id, exit_price, realized_pnl):
        self.registered.pop(position_id, None)


class DummyPortfolioManager:
    def __init__(self, cfg=None):
        self.trade_count = 0
        self.registered = {}
        self.cfg = cfg or {}

    def increment_trade_count(self):
        self.trade_count += 1

    def register_position(self, position_id, position_data):
        self.registered[position_id] = position_data


class DummyOrderManager:
    def __init__(self, exchange_clients):
        self.exchange_clients = exchange_clients


class OrderNotFound(Exception):
    pass


class DummyExchangeClient:
    def __init__(self):
        self.open_orders = [{"id": "o1"}, {"id": "o2"}]
        self.calls = []

    def get_open_orders(self, symbol):
        self.calls.append(("get_open_orders", symbol))
        return list(self.open_orders)

    def cancel_order(self, order_id, symbol=None, params=None):
        self.calls.append(("cancel_order", order_id))
        if order_id == "o2":
            raise OrderNotFound("OrderNotFound")
        return {"id": order_id, "status": "canceled"}

    def create_order(self, **kwargs):
        self.calls.append(("create_order", kwargs))
        return {"id": f"new_{len(self.calls)}"}


def make_manager(min_stop_pct=None, cfg=None):
    risk_manager = DummyRiskManager(min_stop_pct=min_stop_pct)
    portfolio_manager = DummyPortfolioManager(cfg=cfg)
    manager = AdvancedPositionManager(
        risk_manager=risk_manager,
        order_manager=object(),
        portfolio_manager=portfolio_manager,
    )
    return manager, risk_manager


def make_manager_with_exchange(exchange_key="test"):
    risk_manager = DummyRiskManager()
    client = DummyExchangeClient()
    order_manager = DummyOrderManager({exchange_key: client})
    manager = AdvancedPositionManager(
        risk_manager=risk_manager,
        order_manager=order_manager,
        portfolio_manager=DummyPortfolioManager(),
    )
    return manager, client


@pytest.mark.asyncio
async def test_episode_c_rebase_preserves_ratios_short():
    manager, _ = make_manager()
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "entry": 100.0,
        "stop": 102.0,
        "target": 98.0,
    }
    execution_result = {
        "success": True,
        "avg_price": 101.0,
        "filled_amount": 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position = result["position"]
    fill = execution_result["avg_price"]

    stop_ratio = abs(position["stop_loss"] - fill) / fill
    tp_ratio = abs(position["take_profit"] - fill) / fill

    assert position["stop_loss"] > fill
    assert position["take_profit"] < fill
    assert stop_ratio == pytest.approx(0.02, rel=1e-6)
    assert tp_ratio == pytest.approx(0.02, rel=1e-6)
    assert position["rebase_meta"]["target_stop_ratio"] == pytest.approx(0.02, rel=1e-6)
    assert position["rebase_meta"]["target_tp_ratio"] == pytest.approx(0.02, rel=1e-6)


@pytest.mark.asyncio
async def test_episode_c_rebase_min_stop_clamp_short():
    manager, _ = make_manager(min_stop_pct=0.01)
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "entry": 100.0,
        "stop": 100.2,
        "target": 98.0,
    }
    execution_result = {
        "success": True,
        "avg_price": 100.0,
        "filled_amount": 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position = result["position"]
    meta = position["rebase_meta"]

    assert meta["min_stop_applied"] is True
    assert meta["applied_stop_ratio"] == pytest.approx(0.01, rel=1e-6)
    assert meta["final_stop_ratio"] == pytest.approx(0.01, rel=1e-6)
    assert meta["floor_selected"] == "min_stop_pct"
    assert meta["floor_candidates"]["calculated"] == pytest.approx(0.002, rel=1e-6)
    assert meta["floor_candidates"]["min_stop_pct"] == pytest.approx(0.01, rel=1e-6)
    assert position["stop_loss"] == pytest.approx(101.0, rel=1e-6)


@pytest.mark.asyncio
async def test_episode_c_rebase_rr_effective_computed():
    manager, _ = make_manager()
    signal = {
        "symbol": "ETH/USDT:USDT",
        "side": "buy",
        "entry": 100.0,
        "stop": 99.0,
        "target": 102.0,
    }
    execution_result = {
        "success": True,
        "avg_price": 100.0,
        "filled_amount": 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    meta = result["position"]["rebase_meta"]

    assert meta["rr_effective"] == pytest.approx(2.0, rel=1e-6)


@pytest.mark.asyncio
async def test_episode_c_rebase_atr_floor_clamp_short():
    manager, _ = make_manager(
        min_stop_pct=None,
        cfg={
            "risk": {
                "min_stop_atr_floor": {
                    "enabled": True,
                    "atr_mult": 1.5,
                    "canary_symbols": ["*"],
                }
            }
        },
    )
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "entry": 100.0,
        "stop": 100.2,  # 0.2%
        "target": 98.0,
        "atr": 2.0,  # atr_pct=2%, floor=3%
    }
    execution_result = {
        "success": True,
        "avg_price": 100.0,
        "filled_amount": 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position = result["position"]
    meta = position["rebase_meta"]

    assert meta["atr_floor_active"] is True
    assert meta["atr_floor"] == pytest.approx(0.03, rel=1e-6)
    assert meta["min_stop_applied"] is True
    assert meta["applied_stop_ratio"] == pytest.approx(0.03, rel=1e-6)
    assert meta["floor_selected"] == "atr_floor"
    assert position["stop_loss"] == pytest.approx(103.0, rel=1e-6)


@pytest.mark.asyncio
async def test_episode_c_rebase_atr_floor_canary_miss_keeps_raw_stop():
    manager, _ = make_manager(
        min_stop_pct=None,
        cfg={
            "risk": {
                "min_stop_atr_floor": {
                    "enabled": True,
                    "atr_mult": 2.0,
                    "canary_symbols": ["ETH/USDT:USDT"],
                }
            }
        },
    )
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "entry": 100.0,
        "stop": 100.2,  # 0.2%
        "target": 98.0,
        "atr": 5.0,  # would imply huge floor if canary matched
    }
    execution_result = {
        "success": True,
        "avg_price": 100.0,
        "filled_amount": 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    meta = result["position"]["rebase_meta"]

    assert meta["atr_floor_active"] is False
    assert meta["atr_floor_canary_match"] is False
    assert meta["atr_floor"] is None
    assert meta["applied_stop_ratio"] == pytest.approx(0.002, rel=1e-6)
    assert meta["floor_selected"] == "calculated"


@pytest.mark.asyncio
async def test_episode_c_postfill_rr_action_triggers_early_exit():
    manager, _ = make_manager()
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "entry": 100.0,
        "stop": 99.8,
        "target": 100.1,
        "target_stop_ratio": 0.0020,
        "target_tp_ratio": 0.0010,
    }
    execution_result = {
        "success": True,
        "avg_price": 100.0,
        "filled_amount": 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position = result["position"]

    assert position["rr_after_fill"] == pytest.approx(0.5, rel=1e-3)
    assert position["postfill_action"] != "keep"
    assert position["postfill_reason_code"] == "rr_below_1"


@pytest.mark.asyncio
async def test_episode_c_postfill_rr_action_rr_below_required_triggers_early_exit():
    manager, _ = make_manager()
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "entry": 100.0,
        "stop": 99.0,
        "target": 101.2,
        "dynamic_rr_target": 1.58,
    }
    execution_result = {
        "success": True,
        "avg_price": 100.0,
        "filled_amount": 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position = result["position"]

    assert position["rr_after_fill"] == pytest.approx(1.2, rel=1e-3)
    assert position["rr_required"] == pytest.approx(1.58, rel=1e-6)
    assert position["postfill_action"] == "early_exit"
    assert position["postfill_reason_code"] == "rr_below_required"


@pytest.mark.asyncio
async def test_episode_c_risk_order_cancel_replace_reduce_only():
    manager, client = make_manager_with_exchange()
    position = {
        "symbol": "BTC/USDT:USDT",
        "exchange": "test",
        "side": "long",
        "amount": 1.0,
    }

    result = await manager._refresh_exchange_risk_orders(
        position,
        stop_loss=99.0,
        take_profit=101.0,
        reduce_only=True,
    )

    assert result["success"] is True
    cancel_calls = [c for c in client.calls if c[0] == "cancel_order"]
    create_calls = [c for c in client.calls if c[0] == "create_order"]
    assert len(cancel_calls) >= 2
    assert len(create_calls) == 2

    cancel_indices = [i for i, c in enumerate(client.calls) if c[0] == "cancel_order"]
    first_create_index = next(i for i, c in enumerate(client.calls) if c[0] == "create_order")
    assert first_create_index > max(cancel_indices)

    params_list = [c[1].get("params", {}) for c in create_calls]
    assert any(p.get("stopLossPrice") == pytest.approx(99.0, rel=1e-6) for p in params_list)
    assert any(p.get("takeProfitPrice") == pytest.approx(101.0, rel=1e-6) for p in params_list)
    assert all(p.get("reduceOnly") is True for p in params_list)
