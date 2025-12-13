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
    assert order_manager.last_request["amount"] == pytest.approx(1.0)
    assert risk_manager.calculate_calls == 0
    assert risk_manager.validated_signals, "validate_new_position should be called"
    validated = risk_manager.validated_signals[-1]
    assert validated.get("notional") == pytest.approx(10.0)
    assert validated.get("position_size") == pytest.approx(1.0)
    assert engine.trade_history[-1]["risk_metrics"]["new_position_value"] == pytest.approx(10.0)
