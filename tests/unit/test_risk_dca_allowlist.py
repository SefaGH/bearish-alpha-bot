import types
from types import SimpleNamespace

from src.core.risk_manager import RiskManager


def _make_risk_manager_with_dca_cfg(dca_cfg):
    rm = RiskManager.__new__(RiskManager)
    rm._get_dca_cfg = types.MethodType(lambda self, _pm: dca_cfg, rm)
    return rm


def test_risk_manager_rejects_dca_when_strategy_name_not_in_allowlist():
    dca_cfg = {
        "enabled": True,
        "allowed_base_strategies": ["adaptive_str", "adaptive_ob"],
        "strategy": {"max_layers": 3},
        "risk_limits": {},
    }
    rm = _make_risk_manager_with_dca_cfg(dca_cfg)

    signal = {
        "symbol": "BTC/USDT:USDT",
        "intent": "scale_in",
        "scale_profile": "dca",
        "strategy_name": "mean_reversion",
    }

    ok, reason = rm._check_dca_limits(
        signal=signal,
        portfolio_manager=SimpleNamespace(cfg={"dca": dca_cfg}),
        active_positions={},
        pyramiding_cfg={},
        concurrent_limits=SimpleNamespace(max_positions_per_symbol=1),
    )

    assert ok is False
    assert reason == "dca_strategy_not_allowed"


def test_risk_manager_rejects_dca_when_meta_base_strategy_not_in_allowlist():
    dca_cfg = {
        "enabled": True,
        "allowed_base_strategies": ["adaptive_str", "adaptive_ob"],
        "strategy": {"max_layers": 3},
        "risk_limits": {},
    }
    rm = _make_risk_manager_with_dca_cfg(dca_cfg)

    signal = {
        "symbol": "BTC/USDT:USDT",
        "intent": "scale_in",
        "scale_profile": "dca",
        "meta": {"base_strategy": "mean_reversion"},
    }

    ok, reason = rm._check_dca_limits(
        signal=signal,
        portfolio_manager=SimpleNamespace(cfg={"dca": dca_cfg}),
        active_positions={},
        pyramiding_cfg={},
        concurrent_limits=SimpleNamespace(max_positions_per_symbol=1),
    )

    assert ok is False
    assert reason == "dca_strategy_not_allowed"
