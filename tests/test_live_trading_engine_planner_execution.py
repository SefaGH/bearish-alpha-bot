import pytest
from types import SimpleNamespace

from core.live_trading_engine import LiveTradingEngine, TradingMode


class DummyPortfolioManager:
    def __init__(self, equity: float):
        self._equity = equity

    def get_total_equity(self):
        return self._equity

    def get_total_exposure(self):
        return 0.0


class DummyRiskManager:
    def __init__(self):
        self.calculate_calls = 0
        self.validated_signals = []

    async def validate_new_position(self, signal, portfolio_manager):
        self.validated_signals.append(signal.copy())
        notional = (signal.get("position_size", 0) or 0) * (signal.get("entry", 0) or 0)
        equity = portfolio_manager.get_total_equity() if portfolio_manager else 0.0
        risk_metrics = {
            "new_position_value": notional,
            "position_size_pct": (notional / equity) if equity else 0.0,
        }
        return True, "ok", risk_metrics

    async def calculate_position_size(self, signal):
        self.calculate_calls += 1
        return 42.0


class DummyOrderManager:
    def __init__(self):
        self.last_request = None

    async def place_order(self, order_request, execution_algo):
        self.last_request = order_request
        return {"success": True, "order_id": "order-1"}


class DummyPositionManager:
    async def open_position(self, signal, execution_result):
        return {"success": True, "position_id": "pos-1", "position": {}}


@pytest.mark.asyncio
async def test_execute_signal_uses_planner_size_and_skips_multipliers():
    portfolio_manager = DummyPortfolioManager(equity=100.0)
    risk_manager = DummyRiskManager()
    order_manager = DummyOrderManager()
    position_manager = DummyPositionManager()

    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        exchange_clients={"binance": object()},
    )

    engine.config = {"trading": {"order_type": "limit"}}
    engine.execution_analytics = SimpleNamespace(get_best_execution_algorithm=lambda notional, urgency: "limit")

    signal = {
        "signal_id": "sig-1",
        "symbol": "BTC/USDT",
        "side": "buy",
        "entry": 10.0,
        "planner_active": True,
        "planner_planned_notional": 10.0,
        "planner_raw_notional": 15.0,
        "planner_cap_flags": {},
        "position_size": 1.0,
        "notional": 10.0,
        "sizing_meta": {"ppo_position_multiplier": 1.5},
        "position_multiplier": 1.2,
        "risk_assessment": {"metrics": {"final_position_size": 1.0, "final_notional": 10.0}},
    }

    result = await engine.execute_signal(signal)

    assert result["success"] is True
    expected_qty = float(signal["notional"]) / float(signal["entry"])
    assert order_manager.last_request["amount"] == pytest.approx(expected_qty, rel=1e-6)
    assert risk_manager.calculate_calls == 0
    assert risk_manager.validated_signals, "validate_new_position should be called"
    validated = risk_manager.validated_signals[-1]
    assert validated.get("notional") == pytest.approx(10.0)
    assert validated.get("position_size") == pytest.approx(expected_qty, rel=1e-6)
    assert engine.trade_history[-1]["risk_metrics"]["new_position_value"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_execute_signal_postfill_rr_early_exit_closes_position(monkeypatch):
    portfolio_manager = DummyPortfolioManager(equity=100.0)
    risk_manager = DummyRiskManager()
    order_manager = DummyOrderManager()

    class DummyPositionManager:
        async def open_position(self, signal, execution_result):
            return {
                "success": True,
                "position_id": "pos-1",
                "position": {
                    "position_id": "pos-1",
                    "symbol": signal.get("symbol"),
                    "side": signal.get("side"),
                    "exchange": "binance",
                    "amount": 1.0,
                    "entry_price": signal.get("entry"),
                    "current_price": signal.get("entry"),
                    "rr_required": 1.58,
                    "rr_after_fill": 1.2,
                    "postfill_action": "early_exit",
                    "postfill_reason_code": "rr_below_required",
                },
            }

    position_manager = DummyPositionManager()

    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        exchange_clients={"binance": object()},
    )

    engine.config = {"trading": {"order_type": "limit"}}
    engine.execution_analytics = SimpleNamespace(get_best_execution_algorithm=lambda notional, urgency: "limit")

    called = {"exit": 0, "exit_reason": None}

    async def fake_exit(position_id, exit_signal):
        called["exit"] += 1
        called["exit_reason"] = (exit_signal or {}).get("exit_reason")
        engine.active_positions.pop(position_id, None)
        return {"success": True, "exit_reason": called["exit_reason"]}

    monkeypatch.setattr(engine, "_execute_position_exit", fake_exit)

    signal = {
        "signal_id": "sig-rr-1",
        "symbol": "BTC/USDT",
        "side": "buy",
        "entry": 10.0,
        "planner_active": True,
        "planner_planned_notional": 10.0,
        "planner_raw_notional": 15.0,
        "planner_cap_flags": {},
        "position_size": 1.0,
        "notional": 10.0,
        "risk_assessment": {"metrics": {"final_position_size": 1.0, "final_notional": 10.0}},
    }

    result = await engine.execute_signal(signal)

    assert result["stage"] == "postfill_rr_early_exit"
    assert called["exit"] == 1
    assert called["exit_reason"] == "postfill_rr_below_required"
    assert result["close_result"]["success"] is True
    assert "pos-1" not in engine.active_positions
