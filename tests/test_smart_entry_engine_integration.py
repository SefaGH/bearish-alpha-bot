import pytest

from src.config.risk_config import RiskConfiguration
from src.core.live_trading_engine import LiveTradingEngine, TradingMode
from src.core.risk_manager import RiskManager


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


class CapturingOrderManager:
    def __init__(self):
        self.last_order_request = None
        self.last_execution_algo = None

    async def place_order(self, order_request, execution_algo):
        self.last_order_request = order_request
        self.last_execution_algo = execution_algo
        return {"success": True, "order_id": "order-123"}


class DummyPositionManager:
    async def open_position(self, signal, execution_result):
        return {"success": True, "position_id": "pos-123", "position": {}}


@pytest.mark.asyncio
async def test_smart_entry_forces_market_even_when_global_limit():
    # Global config is LIMIT, but Smart Entry is enabled and should FORCE MARKET for low volatility.
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

    om = CapturingOrderManager()
    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=pm,
        risk_manager=rm,
        order_manager=om,
        position_manager=DummyPositionManager(),
        exchange_clients={"binance": object()},
    )

    engine.config = {
        "trading": {"order_type": "limit"},
        "smart_entry_policy": {
            "enabled": True,
            "force_override": False,
            "volatility_threshold_bps": 5.0,
            "params": {
                "LONG": {"atr_multiplier": 0.90, "timeout_seconds": 300, "gate_bps": 5.0},
                "SHORT": {"atr_multiplier": 0.85, "timeout_seconds": 240, "gate_bps": 12.0},
            },
        },
    }

    entry = 10000.0
    # vol_bps = (atr/entry)*10000 = 4.0 bps (< 5.0) => must force MARKET
    atr = 4.0
    notional = 10.0
    qty = notional / entry

    signal = {
        "signal_id": "sig-smart-entry-lowvol",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "entry": entry,
        "atr": atr,
        "stop": entry * 0.99,
        "target": entry * 1.02,  # RR >= 2
        "position_size": qty,
        "notional": notional,
    }

    result = await engine.execute_signal(signal)
    assert result["success"] is True

    assert om.last_execution_algo is not None
    assert str(om.last_execution_algo).lower() == "market"

    order_request = om.last_order_request
    assert isinstance(order_request, dict)

    # Expectation: the outgoing order packet must be MARKET and must not carry limit_price.
    assert "limit_price" not in order_request

    exec_params = order_request.get("execution_params")
    assert isinstance(exec_params, dict)
    assert exec_params.get("order_type") == "market"

    packaged_signal = order_request.get("signal")
    assert isinstance(packaged_signal, dict)
    assert "limit_price" not in packaged_signal
    assert "execution_price" not in packaged_signal
