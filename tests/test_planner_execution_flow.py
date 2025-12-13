import math
import pytest
from types import SimpleNamespace

from config.risk_config import RiskConfiguration
from core.live_trading_engine import LiveTradingEngine, TradingMode
from core.risk_manager import RiskManager


class StubPortfolioManager:
    def __init__(self, equity: float):
        self._equity = equity

    def get_total_equity(self):
        return self._equity

    def get_total_exposure(self):
        return 0.0

    def get_current_drawdown(self):
        return 0.0

    def get_open_positions(self):
        return {}


class DummyOrderManager:
    async def place_order(self, order_request, execution_algo):
        return {"success": True, "order_id": "order-123"}


class DummyPositionManager:
    async def open_position(self, signal, execution_result):
        return {"success": True, "position_id": "pos-123", "position": {}}


@pytest.mark.asyncio
async def test_execution_path_uses_planner_notional_and_skips_reinflation():
    cfg = RiskConfiguration(
        custom_limits={
            "max_position_size": 0.10,
            "position_size_policy": "clip",
            "min_notional_threshold": 5.0,
            "size_planner_enabled": True,
            "max_portfolio_risk": 1.0,
        },
        initial_capital=100,
    )
    pm = StubPortfolioManager(equity=100)
    rm = RiskManager(portfolio_value=100, risk_config=cfg)

    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=pm,
        risk_manager=rm,
        order_manager=DummyOrderManager(),
        position_manager=DummyPositionManager(),
        exchange_clients={"binance": object()},
    )
    engine.config = {"trading": {"order_type": "limit"}}
    engine.execution_analytics = SimpleNamespace(get_best_execution_algorithm=lambda notional, urgency: "limit")

    entry_price = 60000.0
    planner_raw_notional = 333.33
    planner_planned_notional = 10.0
    planned_qty = planner_planned_notional / entry_price

    signal = {
        "signal_id": "sig-azure",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "entry": entry_price,
        "stop": entry_price * 0.99,
        "target": entry_price * 1.02,  # R/R >= 2 so RR rule passes
        "planner_active": True,
        "planner_planned_notional": planner_planned_notional,
        "planner_raw_notional": planner_raw_notional,
        "planner_cap_flags": {"capped_by_size_pct": True},
        "position_size": planned_qty,
        "notional": planner_planned_notional,
        "position_multiplier": 0.75,  # should be ignored when planner_active
        "risk_assessment": {
            "metrics": {
                "final_position_size": planned_qty,
                "final_notional": planner_planned_notional,
                "sizing_meta": {"ppo_position_multiplier": 0.75},
            }
        },
    }

    result = await engine.execute_signal(signal)
    assert result["success"] is True

    # The notional seen by risk validation and stored in trade history must equal the planner cap
    last_trade = engine.trade_history[-1]
    risk_metrics = last_trade["risk_metrics"]
    assert math.isclose(risk_metrics.get("new_position_value", 0), planner_planned_notional, rel_tol=1e-6)
    # Ensure PositionSizeRule did not see an inflated notional
    assert risk_metrics.get("position_size_pct", 0) <= rm.risk_limits["max_position_size"] + 1e-6


@pytest.mark.asyncio
async def test_anomaly_logging_emitted_on_position_size_rule_reject(caplog):
    cfg = RiskConfiguration(
        custom_limits={
            "max_position_size": 0.10,
            "position_size_policy": "clip",
            "min_notional_threshold": 5.0,
            "size_planner_enabled": True,
            "max_portfolio_risk": 1.0,
        },
        initial_capital=100,
    )
    pm = StubPortfolioManager(equity=100)
    rm = RiskManager(portfolio_value=100, risk_config=cfg)

    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "entry": 60000.0,
        "planner_active": True,
        "planner_raw_notional": 333.33,
        "planner_planned_notional": 10.0,
        "planner_cap_flags": {"capped_by_size_pct": True},
        # Force an inconsistent notional larger than the cap to trigger PositionSizeRule rejection
        "notional": 12.0,
        "position_size": 12.0 / 60000.0,
    }

    with caplog.at_level("WARNING"):
        ok, reason, _ = await rm.validate_new_position(signal, pm)

    assert ok is False
    assert "Position size" in reason
    anomaly_logs = [rec for rec in caplog.records if "[RISK-PLANNER] anomaly_position_size_rule" in rec.getMessage()]
    assert anomaly_logs, "anomaly log should be emitted when planner-active rule rejects"
    extra = anomaly_logs[-1].__dict__
    assert extra.get("planned_notional") == pytest.approx(10.0)
    assert extra.get("position_value_seen_by_rule") == pytest.approx(12.0)
