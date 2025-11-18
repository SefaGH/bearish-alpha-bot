import pytest

from src.config.live_trading_config import LiveTradingConfiguration


def test_format_risk_summary_prefers_computed_max_risk():
    summary = LiveTradingConfiguration._format_risk_summary(
        {
            'computed_max_risk_usd': 25.0,
            'per_trade_risk_pct': 1.0,
        },
        capital_val=2500.0,
    )
    assert "1.00%" in summary
    assert "25.00 USDT" in summary


def test_format_risk_summary_falls_back_to_env(monkeypatch):
    monkeypatch.setenv('PER_TRADE_RISK_PCT', '2')
    summary = LiveTradingConfiguration._format_risk_summary({}, capital_val=1000.0)
    assert "2.00%" in summary
    assert "20.00 USDT" in summary
    monkeypatch.delenv('PER_TRADE_RISK_PCT', raising=False)


def test_normalize_risk_config_converts_percent_value(monkeypatch):
    monkeypatch.delenv('PER_TRADE_RISK_PCT', raising=False)
    cfg = {'risk': {'per_trade_risk_pct': 1.0, 'equity_usd': 100.0}}
    LiveTradingConfiguration()._normalize_risk_config(cfg)
    assert cfg['risk']['per_trade_risk_pct'] == pytest.approx(0.01, rel=1e-4)
    assert cfg['risk']['computed_max_risk_usd'] == pytest.approx(1.0, rel=1e-4)


def test_normalize_risk_config_uses_env_default(monkeypatch):
    monkeypatch.setenv('PER_TRADE_RISK_PCT', '2')
    cfg = {'risk': {'equity_usd': 50.0}}
    LiveTradingConfiguration()._normalize_risk_config(cfg)
    assert cfg['risk']['per_trade_risk_pct'] == pytest.approx(0.02, rel=1e-4)
    assert cfg['risk']['computed_max_risk_usd'] == pytest.approx(1.0, rel=1e-4)
    monkeypatch.delenv('PER_TRADE_RISK_PCT', raising=False)
