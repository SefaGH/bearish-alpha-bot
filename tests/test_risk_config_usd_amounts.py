import pytest

from config.risk_config import RiskConfiguration


def test_usd_amounts_respect_fraction_inputs():
    cfg = RiskConfiguration(custom_limits={
        'equity_usd': 100.0,
        'per_trade_risk_pct': 0.01,      # 1%
        'daily_loss_limit_pct': 0.02,    # 2%
        'max_drawdown': 0.10,            # 10%
    })

    assert cfg.max_risk_per_trade_usd == pytest.approx(1.0)
    assert cfg.daily_loss_limit_usd == pytest.approx(2.0)
    assert cfg.max_drawdown_usd == pytest.approx(10.0)


def test_usd_amounts_accept_percent_style_inputs():
    # Percent-style inputs should normalize to fractions internally.
    cfg = RiskConfiguration(custom_limits={
        'equity_usd': 100.0,
        'per_trade_risk_pct': 1,       # 1% (percent-style)
        'daily_loss_limit_pct': 2,     # 2% (percent-style)
        'max_drawdown': 10,            # 10% (percent-style)
    })

    assert cfg.max_risk_per_trade_usd == pytest.approx(1.0)
    assert cfg.daily_loss_limit_usd == pytest.approx(2.0)
    assert cfg.max_drawdown_usd == pytest.approx(10.0)


def test_max_notional_precedence_computed_then_explicit():
    # No explicit max_position_notional_usd -> use computed_max_notional_usd
    cfg_computed = RiskConfiguration(custom_limits={
        'equity_usd': 100.0,
        'per_trade_risk_pct': 0.01,
        'daily_loss_limit_pct': 0.02,
        'max_drawdown': 0.10,
        'computed_max_notional_usd': 75.0,
    })
    assert cfg_computed.get_risk_limits().max_position_notional_usd == pytest.approx(75.0)

    # Explicit USD clamp overrides computed value
    cfg_explicit = RiskConfiguration(custom_limits={
        'equity_usd': 100.0,
        'per_trade_risk_pct': 0.01,
        'daily_loss_limit_pct': 0.02,
        'max_drawdown': 0.10,
        'computed_max_notional_usd': 75.0,
        'max_position_notional_usd': 50.0,
    })
    assert cfg_explicit.get_risk_limits().max_position_notional_usd == pytest.approx(50.0)


def test_balanced_portfolio_heat_uses_six_percent_default():
    cfg = RiskConfiguration()

    assert cfg.initial_capital == pytest.approx(500.0)
    assert cfg.risk_limits.max_portfolio_risk == pytest.approx(0.06, rel=1e-9)
    assert cfg.max_portfolio_risk_usd == pytest.approx(30.0, rel=1e-6)
    assert cfg.max_risk_per_trade_usd == pytest.approx(1.5, rel=1e-6)


def test_min_stop_pct_string_coerces_to_float():
    cfg = RiskConfiguration(custom_limits={'min_stop_pct': '0.5'})

    assert cfg.risk_limits.min_stop_pct == pytest.approx(0.5)
