import pytest

from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager


def _build_risk_manager(**overrides):
    base_limits = {
        'equity_usd': 100.0,
        'per_trade_risk_pct': 0.01,
        'daily_loss_limit_pct': 0.02,
        'max_drawdown': 0.10,
        'min_stop_pct': 0.005,
        'min_notional_threshold': 5.0,
    }
    base_limits.update(overrides)
    cfg = RiskConfiguration(custom_limits=base_limits)
    return RiskManager(portfolio_value=base_limits['equity_usd'], risk_config=cfg)


def test_health_check_healthy_with_clamp_and_fraction_inputs():
    rm = _build_risk_manager(computed_max_notional_usd=75.0)
    health = rm.run_health_check()
    assert health['status'] == 'HEALTHY'
    for key, check in health['checks'].items():
        if key.startswith('config_'):
            assert check['ok'] is True


def test_health_check_allows_no_clamp_when_none():
    rm = _build_risk_manager()  # no computed or explicit max notional -> None clamp
    health = rm.run_health_check()
    assert health['status'] == 'HEALTHY'
    assert health['checks']['config_max_position_notional_usd']['ok'] is True


def test_health_check_flags_invalid_value():
    rm = _build_risk_manager()
    # force an invalid value into risk_limits
    rm.risk_limits['min_stop_pct'] = 0
    health = rm.run_health_check()
    assert health['status'] == 'UNHEALTHY'
    assert health['checks']['config_min_stop_pct']['ok'] is False
