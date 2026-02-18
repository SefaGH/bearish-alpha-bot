import json

import pytest

from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager


class DummyPortfolioManager:
    def __init__(self, equity: float = 1000.0, exposure: float = 0.0, drawdown: float = 0.0, positions=None):
        self.equity = equity
        self.exposure = exposure
        self.drawdown = drawdown
        self._positions = positions or {}

    def get_total_equity(self):
        return self.equity

    def get_total_exposure(self):
        return self.exposure

    def get_open_positions(self):
        return self._positions

    def get_current_drawdown(self):
        return self.drawdown

    def get_available_balance(self):
        return self.equity - self.exposure


def _base_signal() -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "entry": 100.0,
        "stop": 90.0,
        "side": "long",
        "target": 120.0,
        "position_size": 1.0,
    }


class DummySizingEngine:
    async def calculate_optimal_size(self, *args, **kwargs):
        raise ValueError("sizing rejected")


class RejectRule:
    enabled = True
    rule_name = "RejectRule"

    def validate(self, signal, portfolio_manager):
        return False, "reject"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario, expected_reason_code",
    [
        ("missing_portfolio_manager", "risk.internal.missing_portfolio_manager"),
        ("sizing_engine_error", "risk.sizing.rejected"),
        ("planner_reject_size_cap", "risk.planner.reject_size_cap"),
        ("planner_heat_exhausted", "risk.planner.heat_exhausted"),
        ("planner_too_small", "risk.planner.reject_too_small_after_cap"),
        ("legacy_position_limits", "risk.legacy.position_limits.reject"),
        ("rule_rejection", "risk.rule.RejectRule"),
        ("catch_all_exception", "risk.internal.size_and_validate_exception"),
    ],
)
async def test_risk_manager_rejection_reason_codes(monkeypatch, scenario: str, expected_reason_code: str):
    monkeypatch.delenv("RISK_SIZE_PLANNER_ENABLED", raising=False)

    limits = {"equity_usd": 1000.0}
    rules = []
    sizing_engine = None
    pm = DummyPortfolioManager(equity=1000.0)
    signal = _base_signal()

    if scenario == "missing_portfolio_manager":
        pm = None
    elif scenario == "sizing_engine_error":
        sizing_engine = DummySizingEngine()
    elif scenario == "planner_reject_size_cap":
        monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "1")
        limits.update({"max_position_size": 0.1, "position_size_policy": "reject"})
        signal["position_size"] = 200.0  # ensure raw notional blows past caps
    elif scenario == "planner_heat_exhausted":
        monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "1")
        limits.update({"max_portfolio_risk": 0.01, "min_notional_threshold": 5.0, "position_size_policy": "clip"})
        pm = DummyPortfolioManager(equity=1000.0, positions={"pos1": {"risk_amount": 100.0}})
    elif scenario == "planner_too_small":
        monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "1")
        limits.update({"min_notional_threshold": 5.0, "position_size_policy": "clip"})
        signal["notional"] = 1.0
    elif scenario == "legacy_position_limits":
        monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "0")
        limits.update(
            {"max_position_size": 1.0, "max_position_notional_usd": 10.0, "position_size_policy": "reject"}
        )
    elif scenario == "rule_rejection":
        monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "0")
        rules = [RejectRule()]
    elif scenario == "catch_all_exception":
        monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "0")
        limits.update({"max_position_notional_usd": 1_000_000.0, "position_size_policy": "clip"})
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unknown scenario: {scenario}")

    cfg = RiskConfiguration(custom_limits=limits)
    rm = RiskManager(portfolio_value=1000.0, risk_config=cfg, rules=rules)

    if scenario == "catch_all_exception":

        async def boom(signal, portfolio_manager):
            raise RuntimeError("boom")

        monkeypatch.setattr(rm, "validate_new_position", boom)

    ok, _, meta = await rm.size_and_validate_position(signal, pm, sizing_engine=sizing_engine)

    assert ok is False
    assert meta.get("reason_code") == expected_reason_code
    assert meta.get("blocked_by")
    assert meta.get("validation_reason") or meta.get("planner_reason")
    json.dumps(meta)


@pytest.mark.asyncio
async def test_rr_rule_rejection_uses_prefill_reason_code_and_meta(monkeypatch):
    monkeypatch.setenv("RISK_SIZE_PLANNER_ENABLED", "0")
    cfg = RiskConfiguration(
        custom_limits={
            "equity_usd": 1000.0,
            "rr_dynamic": {
                "enabled": True,
                "base_target_rr": 0.8,
                "lower_bound_rr": 0.8,
                "upper_bound_rr": 2.0,
                "weights": {
                    "ml_confidence": 0.0,
                    "rl_agreement": 0.0,
                    "regime_clarity": 0.0,
                    "volume_strength": 0.0,
                    "momentum_strength": 0.0,
                },
                "fallback": {
                    "missing_ml_default": 0.5,
                    "missing_rl_default": 0.5,
                    "missing_regime_default": 0.3,
                },
                "regime_multipliers": {
                    "neutral": 1.0,
                },
            },
        }
    )
    rm = RiskManager(portfolio_value=1000.0, risk_config=cfg)
    pm = DummyPortfolioManager(equity=1000.0)
    signal = _base_signal()
    signal.update(
        {
            "stop": 99.0,
            "target": 100.9,  # R/R = 0.9, should fail pre-fill hard floor 1.0
            "regime_name": "neutral",
            "strategy_min_rr": 0.5,
        }
    )

    ok, _, meta = await rm.size_and_validate_position(signal, pm)
    assert ok is False
    assert meta.get("reason_code") == "risk.rr.pre_fill.rr_below_1"
    risk_metrics = meta.get("risk_metrics", {})
    assert isinstance(risk_metrics, dict)
    rr_meta = risk_metrics.get("rr_gate_prefill", {})
    assert rr_meta.get("reason_code") == "rr_below_1"
    assert rr_meta.get("action") == "reject"
