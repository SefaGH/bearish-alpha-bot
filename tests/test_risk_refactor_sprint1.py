import pytest

from config.risk_config import RiskConfiguration
from core.position_sizing import AdvancedPositionSizing
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


@pytest.mark.asyncio
async def test_stop_floor_applies_min_stop_pct():
    cfg = RiskConfiguration(custom_limits={'equity_usd': 100.0, 'min_stop_pct': 0.005, 'max_portfolio_risk': 0.01})
    risk_manager = RiskManager(portfolio_value=100.0, risk_config=cfg)
    sizing = AdvancedPositionSizing(risk_manager)

    signal = {
        'symbol': 'BTC/USDT:USDT',
        'entry': 92277.40,
        'stop': 92216.38,
        'side': 'long',
        'target': 93100.0,
    }

    sized_signal = await sizing.calculate_optimal_size(signal, return_signal=True)

    assert 150 < sized_signal['notional'] < 250  # ~200 USD after stop floor
    meta = sized_signal.get('sizing_meta', {})
    assert meta.get('floor_triggered') is True
    assert meta.get('effective_stop_pct') == pytest.approx(0.005, rel=1e-6)


@pytest.mark.asyncio
async def test_clip_applied_before_rules_validation():
    cfg = RiskConfiguration(custom_limits={
        'equity_usd': 1000.0,
        'max_position_notional_usd': 75.0,
        'max_position_size': 0.2,
        'max_portfolio_risk': 0.02,
    })
    risk_manager = RiskManager(portfolio_value=1000.0, risk_config=cfg)
    portfolio_manager = DummyPortfolioManager(equity=1000.0, exposure=0.0)

    signal = {
        'symbol': 'BTC/USDT:USDT',
        'entry': 100.0,
        'stop': 90.0,
        'side': 'long',
        'target': 120.0,
        'position_size': 20.0,  # $2000 notional before limits
    }

    approved, final_size, meta = await risk_manager.size_and_validate_position(signal, portfolio_manager)

    assert approved is True
    limit_meta = meta.get('limit_meta', {})
    assert limit_meta.get('action') == 'clip'
    assert limit_meta.get('final_notional') == pytest.approx(75.0, rel=1e-3)
    assert final_size == pytest.approx(0.75, rel=1e-3)


@pytest.mark.asyncio
async def test_auto_resize_on_margin_error_clamps_to_affordable():
    cfg = RiskConfiguration(custom_limits={
        'equity_usd': 1000.0,
        'max_position_size': 1.0,
        'max_position_notional_usd': 1000.0,
        'max_portfolio_risk': 0.02,
    })
    risk_manager = RiskManager(portfolio_value=1000.0, risk_config=cfg)
    portfolio_manager = DummyPortfolioManager(equity=1000.0, exposure=980.0)

    signal = {
        'symbol': 'ETH/USDT:USDT',
        'entry': 100.0,
        'stop': 90.0,
        'side': 'long',
        'position_size': 10.0,  # $1000 notional, fails margin with only $20 available
        'leverage': 10,
        'target': 120.0,
    }

    approved, final_size, meta = await risk_manager.size_and_validate_position(signal, portfolio_manager)

    assert approved is True
    resize_meta = meta.get('resize_meta')
    assert resize_meta is not None
    assert resize_meta.get('used_notional') == pytest.approx(190.0, rel=1e-3)
    assert final_size == pytest.approx(resize_meta['used_notional'] / 100.0, rel=1e-3)


def test_health_check_reports_min_stop_pct():
    cfg = RiskConfiguration(custom_limits={'equity_usd': 100.0, 'min_stop_pct': 0.01})
    risk_manager = RiskManager(portfolio_value=100.0, risk_config=cfg)

    health = risk_manager.run_health_check()

    assert health['status'] == 'HEALTHY'
    assert health['checks']['config_min_stop_pct']['ok'] is True


@pytest.mark.asyncio
async def test_min_notional_rejection_from_sizing():
    cfg = RiskConfiguration(custom_limits={'equity_usd': 100.0, 'min_notional_threshold': 50.0, 'min_stop_pct': 0.01})
    risk_manager = RiskManager(portfolio_value=100.0, risk_config=cfg)
    sizing = AdvancedPositionSizing(risk_manager)

    signal = {
        'symbol': 'BTC/USDT:USDT',
        'entry': 100.0,
        'stop': 80.0,  # large distance -> notional shrinks below min_notional
        'side': 'long',
        'target': 101.0,
    }

    with pytest.raises(ValueError):
        await sizing.calculate_optimal_size(signal, return_signal=True)


@pytest.mark.asyncio
async def test_auto_resize_failure_when_balance_too_low():
    cfg = RiskConfiguration(custom_limits={
        'equity_usd': 1000.0,
        'max_position_size': 1.0,
        'max_position_notional_usd': 1000.0,
        'min_notional_threshold': 50.0,
        'max_portfolio_risk': 0.02,
    })
    risk_manager = RiskManager(portfolio_value=1000.0, risk_config=cfg)
    portfolio_manager = DummyPortfolioManager(equity=1000.0, exposure=990.0)

    signal = {
        'symbol': 'ETH/USDT:USDT',
        'entry': 100.0,
        'stop': 90.0,
        'side': 'long',
        'position_size': 10.0,  # $1000 notional, margin will exceed available
        'leverage': 1,
        'target': 120.0,
    }

    approved, final_size, meta = await risk_manager.size_and_validate_position(signal, portfolio_manager)

    assert approved is False
    assert meta.get('resize_failed') is True
    assert meta.get('sizing_error') is None
    assert final_size == 0.0


@pytest.mark.asyncio
async def test_end_to_end_tight_stop_scenario_passes_with_stop_floor():
    cfg = RiskConfiguration(custom_limits={
        'equity_usd': 1000.0,
        'min_stop_pct': 0.005,
        'max_position_size': 0.5,  # allow notional up to 50% of equity
        'max_portfolio_risk': 0.001,  # 0.1% risk per trade
    })
    risk_manager = RiskManager(portfolio_value=1000.0, risk_config=cfg)
    sizing = AdvancedPositionSizing(risk_manager)
    portfolio_manager = DummyPortfolioManager(equity=1000.0, exposure=0.0)

    signal = {
        'symbol': 'BTC/USDT:USDT',
        'entry': 92277.40,
        'stop': 92216.38,
        'side': 'long',
        'target': 93000.0,
    }

    sized_signal = await sizing.calculate_optimal_size(signal, return_signal=True)
    assert 150 < sized_signal['notional'] < 300
    assert sized_signal.get('sizing_meta', {}).get('floor_triggered') is True

    approved, _, meta = await risk_manager.size_and_validate_position(sized_signal, portfolio_manager)
    assert approved is True
    assert meta.get('risk_metrics', {}).get('risk_amount', 0) > 0


def test_min_stop_pct_priority_chain_env_over_yaml(monkeypatch):
    monkeypatch.setenv('RISK_MIN_STOP_PCT', '0.02')
    cfg = RiskConfiguration(custom_limits={'equity_usd': 100.0, 'min_stop_pct': 0.005})

    limits = cfg.get_risk_limits()
    assert limits.min_stop_pct == pytest.approx(0.02)

    # Remove ENV, YAML should win
    monkeypatch.delenv('RISK_MIN_STOP_PCT')
    cfg_yaml = RiskConfiguration(custom_limits={'equity_usd': 100.0, 'min_stop_pct': 0.007})
    assert cfg_yaml.get_risk_limits().min_stop_pct == pytest.approx(0.007)

    # Neither ENV nor YAML: default
    cfg_default = RiskConfiguration(custom_limits={'equity_usd': 100.0})
    assert cfg_default.get_risk_limits().min_stop_pct == pytest.approx(0.005)


def test_min_stop_pct_normalization(monkeypatch):
    # ENV path: values > 1 treated as percentages (divide by 100)
    monkeypatch.setenv('RISK_MIN_STOP_PCT', '50')
    cfg_env_percent = RiskConfiguration(custom_limits={'equity_usd': 100.0})
    assert cfg_env_percent.get_risk_limits().min_stop_pct == pytest.approx(0.5)

    monkeypatch.setenv('RISK_MIN_STOP_PCT', '2')
    cfg_env_two = RiskConfiguration(custom_limits={'equity_usd': 100.0})
    assert cfg_env_two.get_risk_limits().min_stop_pct == pytest.approx(0.02)

    monkeypatch.setenv('RISK_MIN_STOP_PCT', '0.5')
    cfg_env_decimal = RiskConfiguration(custom_limits={'equity_usd': 100.0})
    assert cfg_env_decimal.get_risk_limits().min_stop_pct == pytest.approx(0.5)

    # YAML path: >1 also normalized
    monkeypatch.delenv('RISK_MIN_STOP_PCT')
    cfg_yaml_percent = RiskConfiguration(custom_limits={'equity_usd': 100.0, 'min_stop_pct': 50})
    assert cfg_yaml_percent.get_risk_limits().min_stop_pct == pytest.approx(0.5)
