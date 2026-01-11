import pytest

from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager
from core.risk_rules import compute_max_affordable_notional
from core.strategy_coordinator import StrategyCoordinator


class DummyPM:
    def __init__(self, equity, exposure=0.0, open_positions=None):
        self._equity = equity
        self._exposure = exposure
        self._open_positions = open_positions or {}

    def get_current_equity(self):
        return self._equity

    def get_total_equity(self):
        return self._equity

    def get_total_exposure(self):
        return self._exposure

    def get_open_positions(self):
        return self._open_positions

    def get_available_balance(self):
        return self._equity - self._exposure


@pytest.fixture(autouse=True)
def clear_flag(monkeypatch):
    monkeypatch.delenv("RISK_SIZE_PLANNER_ENABLED", raising=False)


def make_rm(custom_limits, initial_capital):
    cfg = RiskConfiguration(custom_limits=custom_limits, initial_capital=initial_capital)
    return RiskManager(portfolio_value=initial_capital, risk_config=cfg, rules=[])


@pytest.mark.asyncio
async def test_planner_active_accepts_capped_small_account(monkeypatch):
    monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "true")
    rm = make_rm(
        custom_limits={
            "max_position_size": 0.1,
            "max_position_notional_usd": None,
            "min_notional_threshold": 5.0,
            "position_size_policy": "clip",
            "max_portfolio_risk": 1.0,
        },
        initial_capital=100,
    )
    pm = DummyPM(equity=100)
    signal = {
        "symbol": "BTC/USDT",
        "entry": 1.0,
        "stop": 0.995,  # tight stop → APS equivalent raw ~200 if it had sized
        "notional": 200.0,  # simulate APS raw_notional
        "position_size": 200.0,  # raw qty; will be overwritten
    }

    ok, final_size, meta = await rm.size_and_validate_position(signal, pm, sizing_engine=None)
    planner = meta.get("planner")

    assert ok is True
    assert planner is not None
    assert planner.get("reason") is None
    assert planner.get("capped_by_size_pct") is True
    assert planner.get("capped_by_max_notional") is False
    assert planner.get("capped_by_capital") is False
    assert planner.get("capped_by_heat") is False
    assert meta.get("planner_raw_notional") == 200.0
    assert meta.get("planner_delta_abs") > 0
    assert meta.get("planner_delta_ratio") < 1.0
    assert meta.get("validation_reason") == "All risk rules passed"

    assert pytest.approx(planner.get("planned_notional"), rel=1e-6) == 10.0
    assert pytest.approx(signal["notional"], rel=1e-6) == pytest.approx(planner.get("planned_notional"), rel=1e-6)
    assert pytest.approx(final_size, rel=1e-6) == pytest.approx(planner.get("planned_qty"), rel=1e-6)
    assert meta.get("risk_metrics", {}).get("position_size_pct") <= 0.10


@pytest.mark.asyncio
async def test_planner_active_rejects_too_small(monkeypatch):
    monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "true")
    rm = make_rm(
        custom_limits={
            "max_position_size": 0.2,
            "min_notional_threshold": 5.0,
            "position_size_policy": "clip",
            "max_portfolio_risk": 1.0,
        },
        initial_capital=20,
    )
    pm = DummyPM(equity=20)
    signal = {
        "symbol": "BTC/USDT",
        "entry": 1.0,
        "stop": 0.99,
        "notional": 4.0,
        "position_size": 4.0,
    }

    ok, final_size, meta = await rm.size_and_validate_position(signal, pm, sizing_engine=None)
    planner = meta.get("planner")

    assert ok is False
    assert planner is not None
    assert planner.get("below_min_notional") is True
    assert planner.get("reason") == "REJECT_TOO_SMALL_AFTER_CAP"
    assert meta.get("blocked_by") == "SizePlanner"
    assert meta.get("planner_reason") == "REJECT_TOO_SMALL_AFTER_CAP"
    assert planner.get("planned_notional", 0) < 5.0


@pytest.mark.asyncio
async def test_flag_gates_legacy_vs_planner(monkeypatch):
    # Legacy path should respect SSOT limits but does not apply the planner’s capital cap;
    # planner path should cap by capital affordability with the active flag.
    base_limits = {
        "max_position_size": 1.0,  # no pct cap
        "max_position_notional_usd": None,
        "min_notional_threshold": 1.0,
        "position_size_policy": "clip",
        "max_portfolio_risk": 1.0,
    }

    # Legacy (flag off)
    monkeypatch.delenv("RISK_SIZE_PLANNER_ENABLED", raising=False)
    rm_legacy = make_rm(custom_limits=base_limits, initial_capital=100)
    pm = DummyPM(equity=100, exposure=90)  # available balance = 10
    signal = {
        "symbol": "BTC/USDT",
        "entry": 1.0,
        "stop": 0.99,
        "notional": 50.0,  # raw notional from sizing
        "position_size": 50.0,
    }

    ok_legacy, final_size_legacy, meta_legacy = await rm_legacy.size_and_validate_position(signal.copy(), pm, sizing_engine=None)
    planner_shadow = meta_legacy.get("planner")
    assert ok_legacy is True
    assert planner_shadow is not None  # shadow planner still computed
    assert pytest.approx(planner_shadow.get("planned_notional"), rel=1e-6) == compute_max_affordable_notional(10, 1)
    # Legacy path output stays on raw notional (no capital cap applied)
    legacy_final_notional = meta_legacy['limit_meta']['final_notional']
    assert pytest.approx(legacy_final_notional, rel=1e-6) == 50.0
    assert pytest.approx(final_size_legacy, rel=1e-6) == 50.0
    assert meta_legacy.get("planner_delta_abs") > 0

    # Planner (flag on) should cap by capital (available 10 → 9.5 after safety factor)
    monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "true")
    rm_planner = make_rm(custom_limits=base_limits, initial_capital=100)
    ok_plan, final_size_plan, meta_plan = await rm_planner.size_and_validate_position(signal.copy(), pm, sizing_engine=None)
    planner = meta_plan.get("planner")

    assert ok_plan is True
    assert planner is not None
    expected_cap = compute_max_affordable_notional(10, 1)
    assert pytest.approx(planner.get("planned_notional"), rel=1e-6) == expected_cap
    assert meta_plan.get("planner_raw_notional") == 50.0
    assert meta_plan.get("planner_delta_abs") == pytest.approx(50.0 - expected_cap)
    assert planner.get("capped_by_capital") is True
    assert planner.get("capped_by_size_pct") is False
    assert planner.get("capped_by_max_notional") is False
    assert planner.get("capped_by_heat") is False
    assert planner.get("reason") is None
    assert planner.get("below_min_notional") is False
    assert pytest.approx(final_size_plan, rel=1e-6) == pytest.approx(planner.get("planned_qty"), rel=1e-6)


@pytest.mark.asyncio
async def test_planner_notional_used_for_enqueue_display(monkeypatch):
    monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "true")

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

    class PM:
        def get_total_equity(self):
            return 100
        def get_total_exposure(self):
            return 0.0
        def get_open_positions(self):
            return {}
        def get_current_drawdown(self):
            return 0.0
        def get_available_capital(self):
            return 100

    pm = PM()
    sc = StrategyCoordinator(portfolio_manager=pm, risk_manager=rm, market_data_pipeline=None, config={'strategies': {}})

    signal = {
        'strategy_name': 'test_strategy',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'entry': 33.3,
        # Tighten stop so APS risk-based notional clears the $5.00 min_notional guard
        'stop': 32.7,
        'target': 40.0,
        'reason': 'test',
        'notional': 333.0,
        'position_size': 10.0,
    }

    risk_assessment = await sc._assess_signal_risk(signal, signal.get('strategy_name', 'test_strategy'))

    assert risk_assessment['acceptable'] is True
    planner_notional = risk_assessment['metrics']['planner']['planned_notional']
    assert risk_assessment['notional'] == pytest.approx(planner_notional, rel=1e-6)
    assert risk_assessment['notional'] > 0
    assert risk_assessment['position_size'] == pytest.approx(risk_assessment['notional'] / signal['entry'], rel=1e-3)


@pytest.mark.asyncio
async def test_legacy_notional_unchanged_when_planner_off(monkeypatch):
    monkeypatch.delenv("RISK_SIZE_PLANNER_ENABLED", raising=False)

    cfg = RiskConfiguration(
        custom_limits={
            'max_position_size': 0.2,
            'position_size_policy': 'clip',
            'min_notional_threshold': 5.0,
            'size_planner_enabled': False,
            'max_portfolio_risk': 1.0,
        },
        initial_capital=200,
    )
    rm = RiskManager(portfolio_value=200, risk_config=cfg)

    class PM:
        def get_total_equity(self):
            return 200
        def get_total_exposure(self):
            return 0.0
        def get_open_positions(self):
            return {}
        def get_current_drawdown(self):
            return 0.0
        def get_available_capital(self):
            return 200

    pm = PM()
    sc = StrategyCoordinator(portfolio_manager=pm, risk_manager=rm, market_data_pipeline=None, config={'strategies': {}})

    signal = {
        'strategy_name': 'test_strategy',
        'symbol': 'ETH/USDT',
        'side': 'buy',
        'entry': 20.0,
        'stop': 19.0,
        'target': 22.0,
        'reason': 'test',
        'notional': 40.0,
        'position_size': 2.0,
    }

    risk_assessment = await sc._assess_signal_risk(signal, signal.get('strategy_name', 'test_strategy'))

    assert risk_assessment['acceptable'] is True
    final_notional = risk_assessment['metrics'].get('final_notional')
    assert final_notional is None or final_notional == pytest.approx(risk_assessment['notional'], rel=1e-6)
    assert risk_assessment['notional'] > 0
    assert risk_assessment['position_size'] == pytest.approx(risk_assessment['notional'] / signal['entry'], rel=1e-3)
