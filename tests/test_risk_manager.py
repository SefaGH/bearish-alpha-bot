import pytest

from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager


def make_risk_manager(portfolio_value=10000, custom_limits=None):
    limits = dict(custom_limits or {})
    limits.setdefault('equity_usd', portfolio_value)
    config = RiskConfiguration(custom_limits=limits)
    return RiskManager(portfolio_value=portfolio_value, risk_config=config)


@pytest.mark.asyncio
async def test_volatility_sizing_low_volatility_boosts_size():
    custom_limits = {
        'max_position_size': 1.0,
        'volatility_sizing': {
            'enabled': True,
            'atr_floor_pct': 0.005,
            'atr_ceiling_pct': 0.02,
            'low_vol_multiplier': 1.5,
            'baseline_multiplier': 1.0,
            'high_vol_multiplier': 0.5,
        }
    }
    risk_manager = make_risk_manager(custom_limits=custom_limits)
    signal = {
        'entry': 100,
        'stop': 95,
        'side': 'long',
        'atr': 0.3,
    }

    position_size = await risk_manager.calculate_position_size(signal)

    assert position_size == pytest.approx(60, rel=1e-3)
    meta = signal.get('sizing_meta', {})
    assert meta.get('volatility_bucket') == 'low'
    assert meta.get('volatility_multiplier') == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_volatility_sizing_high_volatility_reduces_size():
    custom_limits = {
        'max_position_size': 1.0,
        'volatility_sizing': {
            'enabled': True,
            'atr_floor_pct': 0.005,
            'atr_ceiling_pct': 0.02,
            'low_vol_multiplier': 1.5,
            'baseline_multiplier': 1.0,
            'high_vol_multiplier': 0.5,
        }
    }
    risk_manager = make_risk_manager(custom_limits=custom_limits)
    signal = {
        'entry': 100,
        'stop': 95,
        'side': 'long',
        'atr': 3.0,
    }

    position_size = await risk_manager.calculate_position_size(signal)

    assert position_size == pytest.approx(20, rel=1e-3)
    meta = signal.get('sizing_meta', {})
    assert meta.get('volatility_bucket') == 'high'
    assert meta.get('volatility_multiplier') == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_volatility_sizing_applies_minimum_position_floor():
    custom_limits = {
        'max_portfolio_risk': 0.001,
        'max_position_size': 1.0,
        'volatility_sizing': {
            'enabled': True,
            'atr_floor_pct': 0.005,
            'atr_ceiling_pct': 0.02,
            'low_vol_multiplier': 1.2,
            'baseline_multiplier': 0.8,
            'high_vol_multiplier': 0.1,
            'min_position_size_pct': 0.02,
        }
    }
    risk_manager = make_risk_manager(custom_limits=custom_limits)
    signal = {
        'entry': 100,
        'stop': 99,
        'side': 'long',
        'atr': 3.0,
    }

    position_size = await risk_manager.calculate_position_size(signal)

    assert position_size == pytest.approx(2.0, rel=1e-3)
    meta = signal.get('sizing_meta', {})
    assert meta.get('volatility_bucket') == 'high'
    assert meta.get('volatility_multiplier') == pytest.approx(0.1)
    assert meta.get('min_position_units') == pytest.approx(2.0, rel=1e-3)


class DummyPortfolioManager:
    """Minimal PortfolioManager stub for concurrent limit tests."""

    def __init__(self, equity=10000, exposure=0.0, drawdown=0.0, open_positions=None):
        self.equity = equity
        self.exposure = exposure
        self.drawdown = drawdown
        self._open_positions = open_positions or {}

    def get_current_equity(self):
        return self.equity

    def get_total_exposure(self):
        return self.exposure

    def get_open_positions(self):
        return self._open_positions

    def get_current_drawdown(self):
        return self.drawdown

    def count_open_positions(self, symbol=None):
        if symbol:
            return sum(1 for pos in self._open_positions.values() if pos.get('symbol') == symbol)
        return len(self._open_positions)


@pytest.mark.asyncio
async def test_concurrent_limits_block_when_max_open_positions_hit():
    custom_limits = {
        'max_position_size': 1.0,
        'concurrent_limits': {
            'max_open_positions': 1,
            'max_positions_per_symbol': 5,
            'max_total_risk_pct': 0.5,
        }
    }
    risk_manager = make_risk_manager(custom_limits=custom_limits)
    portfolio_mgr = DummyPortfolioManager(
        equity=10000,
        exposure=2000,
        open_positions={
            'pos_1': {
                'symbol': 'BTC/USDT:USDT',
                'entry_price': 100,
                'size': 5,
                'risk_amount': 250,
            }
        }
    )
    signal = {
        'symbol': 'ETH/USDT:USDT',
        'entry': 100,
        'stop': 95,
        'target': 110,
        'side': 'long',
        'position_size': 5,
    }

    is_valid, reason, _ = await risk_manager.validate_new_position(signal, portfolio_mgr)

    assert is_valid is False
    assert 'Max open positions' in reason


@pytest.mark.asyncio
async def test_concurrent_limits_block_symbol_cap():
    custom_limits = {
        'max_position_size': 1.0,
        'concurrent_limits': {
            'max_open_positions': 5,
            'max_positions_per_symbol': 1,
            'max_total_risk_pct': 1.0,
        }
    }
    risk_manager = make_risk_manager(custom_limits=custom_limits)
    portfolio_mgr = DummyPortfolioManager(
        equity=10000,
        exposure=4000,
        open_positions={
            'pos_1': {
                'symbol': 'BTC/USDT:USDT',
                'entry_price': 100,
                'size': 2,
                'risk_amount': 200,
            }
        }
    )
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'entry': 120,
        'stop': 110,
        'target': 140,
        'side': 'long',
        'position_size': 1,
    }

    is_valid, reason, _ = await risk_manager.validate_new_position(signal, portfolio_mgr)

    assert is_valid is False
    assert reason
    assert ('Max positions for BTC/USDT:USDT' in reason) or str(reason).startswith('scale_in_')


@pytest.mark.asyncio
async def test_concurrent_limits_block_portfolio_heat():
    custom_limits = {
        'max_position_size': 1.0,
        'concurrent_limits': {
            'max_open_positions': 10,
            'max_positions_per_symbol': 5,
            'max_total_risk_pct': 0.05,
        }
    }
    risk_manager = make_risk_manager(custom_limits=custom_limits)
    portfolio_mgr = DummyPortfolioManager(
        equity=10000,
        exposure=0,
        open_positions={
            'pos_1': {
                'symbol': 'ETH/USDT:USDT',
                'entry_price': 100,
                'size': 10,
                'risk_amount': 400,
            }
        }
    )
    signal = {
        'symbol': 'SOL/USDT:USDT',
        'entry': 100,
        'stop': 90,
        'target': 130,
        'side': 'long',
        'position_size': 15,
    }

    is_valid, reason, metrics = await risk_manager.validate_new_position(signal, portfolio_mgr)

    assert is_valid is False
    assert 'Portfolio heat' in reason
    assert metrics['portfolio_heat'] > custom_limits['concurrent_limits']['max_total_risk_pct']


def test_can_open_new_position_blocks_when_heat_too_high():
    custom_limits = {
        'max_position_size': 1.0,
        'concurrent_limits': {
            'max_open_positions': 10,
            'max_positions_per_symbol': 5,
            'max_total_risk_pct': 0.05,
        }
    }
    risk_manager = make_risk_manager(custom_limits=custom_limits)
    portfolio_mgr = DummyPortfolioManager(
        equity=10000,
        exposure=0,
        open_positions={
            'pos_1': {
                'symbol': 'ETH/USDT:USDT',
                'entry_price': 100,
                'size': 10,
                'risk_amount': 400,
            }
        }
    )
    signal = {
        'symbol': 'SOL/USDT:USDT',
        'entry': 100,
        'stop': 90,
        'target': 130,
        'side': 'long',
        'position_size': 15,
    }

    allowed, reason, metrics = risk_manager.can_open_new_position(signal, portfolio_mgr)

    assert allowed is False
    assert 'Portfolio heat' in reason
    assert metrics['portfolio_heat'] > custom_limits['concurrent_limits']['max_total_risk_pct']


def test_can_open_new_position_merges_cached_metrics():
    risk_manager = make_risk_manager()
    portfolio_mgr = DummyPortfolioManager(
        equity=10000,
        exposure=0,
        open_positions={}
    )
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'entry': 100,
        'stop': 95,
        'target': 120,
        'side': 'long',
        'position_size': 1,
    }
    cached = {'sizing_meta': {'source': 'test'}}

    allowed, reason, metrics = risk_manager.can_open_new_position(signal, portfolio_mgr, cached_metrics=cached)

    assert allowed is True
    assert reason == 'OK'
    assert metrics is not cached
    assert metrics['sizing_meta']['source'] == 'test'
