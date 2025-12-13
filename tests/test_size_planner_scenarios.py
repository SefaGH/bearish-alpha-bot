import math
import pytest

from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager
from core.risk_rules import compute_max_affordable_notional


class FakePortfolioManager:
    def __init__(self, equity, open_positions=None, exposure=0.0):
        self._equity = equity
        self._open_positions = open_positions or {}
        self._exposure = exposure

    def get_current_equity(self):
        return self._equity

    def get_total_exposure(self):
        return self._exposure

    def get_open_positions(self):
        return self._open_positions


@pytest.fixture
def risk_manager_base():
    cfg = RiskConfiguration(
        custom_limits={
            'max_position_size': 0.1,
            'max_position_notional_usd': None,
            'position_size_policy': 'clip',
            'min_notional_threshold': 5.0,
        },
        initial_capital=100,
    )
    return RiskManager(portfolio_value=100, risk_config=cfg, rules=[])


def test_clip_size_cap(risk_manager_base):
    rm = risk_manager_base
    res = rm.plan_position_size(
        raw_notional=200,
        symbol="BTC/USDT",
        equity=100,
        price=1,
        available_balance=100,
        leverage=1,
        risk_limits=rm.risk_limits,
        min_notional_threshold=5.0,
        max_portfolio_risk_usd=None,
        current_open_risk_usd=0.0,
        position_size_policy="clip",
    )
    assert math.isclose(res.planned_notional, 10.0)
    assert res.below_min_notional is False
    assert res.capped_by_size_pct is True


def test_explicit_max_notional_cap(risk_manager_base):
    rm = risk_manager_base
    limits = dict(rm.risk_limits)
    limits['max_position_notional_usd'] = 8
    res = rm.plan_position_size(
        raw_notional=200,
        symbol="BTC/USDT",
        equity=100,
        price=1,
        available_balance=100,
        leverage=1,
        risk_limits=limits,
        min_notional_threshold=5.0,
        max_portfolio_risk_usd=None,
        current_open_risk_usd=0.0,
        position_size_policy="clip",
    )
    assert math.isclose(res.planned_notional, 8.0)
    assert res.capped_by_max_notional is True


def test_min_notional_reject_after_caps(risk_manager_base):
    rm = risk_manager_base
    res = rm.plan_position_size(
        raw_notional=4,
        symbol="BTC/USDT",
        equity=20,
        price=1,
        available_balance=20,
        leverage=1,
        risk_limits=rm.risk_limits,
        min_notional_threshold=5.0,
        max_portfolio_risk_usd=None,
        current_open_risk_usd=0.0,
        position_size_policy="clip",
    )
    assert res.below_min_notional is True
    assert res.reason == "REJECT_TOO_SMALL_AFTER_CAP"


def test_position_size_policy_reject_on_size_cap(risk_manager_base):
    rm = risk_manager_base
    limits = dict(rm.risk_limits)
    limits['max_position_size'] = 0.05
    res = rm.plan_position_size(
        raw_notional=200,
        symbol="BTC/USDT",
        equity=100,
        price=1,
        available_balance=100,
        leverage=1,
        risk_limits=limits,
        min_notional_threshold=5.0,
        max_portfolio_risk_usd=None,
        current_open_risk_usd=0.0,
        position_size_policy="reject",
    )
    assert res.reason == "REJECT_SIZE_CAP"
    assert res.planned_notional == 0.0


def test_capital_cap_binds():
    cfg = RiskConfiguration(
        custom_limits={
            'max_position_size': 1.0,
            'position_size_policy': 'clip',
            'min_notional_threshold': 5.0,
            'max_portfolio_risk': 1.0,
        },
        initial_capital=100,
    )
    rm = RiskManager(portfolio_value=100, risk_config=cfg, rules=[])
    res = rm.plan_position_size(
        raw_notional=200,
        symbol="BTC/USDT",
        equity=100,
        price=1,
        available_balance=10,
        leverage=2,
        risk_limits=rm.risk_limits,
        min_notional_threshold=5.0,
        max_portfolio_risk_usd=None,
        current_open_risk_usd=0.0,
        position_size_policy="clip",
    )
    assert math.isclose(res.planned_notional, compute_max_affordable_notional(10, 2))
    assert res.capped_by_capital is True


def test_heat_cap_exhausted():
    cfg = RiskConfiguration(
        custom_limits={
            'max_position_size': 1.0,
            'position_size_policy': 'clip',
            'min_notional_threshold': 5.0,
        },
        initial_capital=100,
    )
    rm = RiskManager(portfolio_value=100, risk_config=cfg, rules=[])
    res = rm.plan_position_size(
        raw_notional=10,
        symbol="BTC/USDT",
        equity=100,
        price=1,
        available_balance=100,
        leverage=1,
        risk_limits=rm.risk_limits,
        min_notional_threshold=5.0,
        max_portfolio_risk_usd=10.0,
        current_open_risk_usd=9.0,
        position_size_policy="clip",
    )
    assert res.capped_by_heat is True
    assert res.below_min_notional is True
    assert res.reason == "portfolio_heat_exhausted"


def test_planner_flag_reads_risk_config(monkeypatch):
    monkeypatch.delenv("RISK_SIZE_PLANNER_ENABLED", raising=False)
    cfg = RiskConfiguration(custom_limits={"size_planner_enabled": True}, initial_capital=100)
    rm = RiskManager(portfolio_value=100, risk_config=cfg, rules=[])

    assert rm._is_size_planner_enabled() is True
    assert getattr(rm, "_planner_flag_source", None) == "risk_config"


@pytest.mark.asyncio
async def test_planner_caps_before_position_size_rule(monkeypatch):
    monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "true")

    class StrategyFakePM:
        def __init__(self, equity: float):
            self._equity = equity

        def get_total_equity(self):
            return self._equity

        def get_total_exposure(self):
            return 0.0

        def get_open_positions(self):
            return {}

        def get_current_drawdown(self):
            return 0.0

        def get_available_capital(self):
            return self._equity

    cfg = RiskConfiguration(
        custom_limits={
            'max_position_size': 0.1,
            'position_size_policy': 'clip',
            'min_notional_threshold': 5.0,
            'size_planner_enabled': True,
            'max_portfolio_risk': 1.0,
        },
        initial_capital=100,
    )
    rm = RiskManager(portfolio_value=100, risk_config=cfg)
    pm = StrategyFakePM(equity=100)

    signal = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'entry': 33.3,
        'stop': 30.0,
        'target': 40.0,
        'notional': 333.0,
        'position_size': 10.0,
    }

    allowed, final_size, meta = await rm.size_and_validate_position(signal, pm)

    assert allowed is True, f"planner_result={meta}"
    assert meta.get('planner') is not None
    assert math.isclose(signal['notional'], 10.0, rel_tol=1e-3)
    assert math.isclose(final_size * signal['entry'], 10.0, rel_tol=1e-3)
    assert meta['planner'].capped_by_size_pct is True
    risk_metrics = meta.get('risk_metrics', {})
    assert risk_metrics.get('position_size_pct', 0) <= rm.risk_limits['max_position_size'] + 1e-6
