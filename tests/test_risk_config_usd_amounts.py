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


def test_usd_amounts_accept_percent_style_env(monkeypatch):
    # Env set as percent-style; should normalize to fraction internally.
    monkeypatch.setenv('PER_TRADE_RISK_PCT', '1')   # 1%
    monkeypatch.setenv('DAILY_LOSS_LIMIT_PCT', '2') # 2%

    cfg = RiskConfiguration(custom_limits={
        'equity_usd': 100.0,
        'per_trade_risk_pct': 0.5,      # config value should be overridden by env
        'daily_loss_limit_pct': 0.5,
        'max_drawdown': 0.10,
    })

    assert cfg.max_risk_per_trade_usd == pytest.approx(1.0)
    assert cfg.daily_loss_limit_usd == pytest.approx(2.0)
    assert cfg.max_drawdown_usd == pytest.approx(10.0)
