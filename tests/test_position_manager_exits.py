import asyncio
import asyncio
import pytest
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta

from core.position_manager import AdvancedPositionManager, ExitReason


class DummyRiskManager:
    def __init__(self):
        self.registered = {}
        self.risk_limits_dataclass = SimpleNamespace(
            stop_loss_pct=0.02,
            take_profit_ratio=2.0,
        )

    def register_position(self, position_id, position_data):
        self.registered[position_id] = position_data

    def close_position(self, position_id, exit_price, realized_pnl):
        self.registered.pop(position_id, None)

    def _calculate_stop_loss_from_signal(self, signal, entry_price):
        side = (signal.get('side') or 'buy').lower()
        explicit = (
            signal.get('stop')
            or signal.get('stop_loss')
            or signal.get('stop_price')
        )
        if explicit:
            return float(explicit)

        sl_pct = signal.get('sl_pct') or signal.get('stop_loss_pct')
        if sl_pct:
            pct = float(sl_pct)
            if pct > 1:
                pct /= 100.0
            return entry_price * (1 - pct) if side in ('buy', 'long') else entry_price * (1 + pct)

        atr = signal.get('atr') or signal.get('atr_value')
        mult = signal.get('sl_atr_mult')
        if atr and mult:
            distance = float(atr) * float(mult)
            return entry_price - distance if side in ('buy', 'long') else entry_price + distance

        fallback = getattr(self.risk_limits_dataclass, 'stop_loss_pct', 0.02)
        return entry_price * (1 - fallback) if side in ('buy', 'long') else entry_price * (1 + fallback)


class DummyPortfolioManager:
    def __init__(self, cfg=None):
        self.trade_count = 0
        self.registered = {}
        self.closed = []
        self.cfg = cfg or {}

    def increment_trade_count(self):
        self.trade_count += 1

    def register_position(self, position_id, position_data):
        self.registered[position_id] = position_data

    def close_position(self, position_id, exit_price, realized_pnl):
        self.closed.append({
            'position_id': position_id,
            'exit_price': exit_price,
            'realized_pnl': realized_pnl,
        })

    def get_open_positions(self):
        return dict(self.registered)


def make_manager(cfg=None):
    risk_manager = DummyRiskManager()
    portfolio_manager = DummyPortfolioManager(cfg=cfg)
    manager = AdvancedPositionManager(
        risk_manager=risk_manager,
        order_manager=object(),
        portfolio_manager=portfolio_manager,
    )
    return manager, risk_manager, portfolio_manager


@pytest.mark.asyncio
async def test_long_signal_applies_percentage_exits():
    manager, risk_manager, portfolio = make_manager()
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'tp_pct': 0.03,
        'sl_pct': 0.01,
    }
    execution_result = {
        'success': True,
        'avg_price': 100.0,
        'filled_amount': 1.5,
    }

    result = await manager.open_position(signal, execution_result)
    assert result['success'] is True

    position = result['position']
    assert position['stop_loss'] == pytest.approx(99.0, rel=1e-6)
    assert position['take_profit'] == pytest.approx(103.0, rel=1e-6)
    assert position['risk_amount'] == pytest.approx(1.5, rel=1e-6)
    assert portfolio.trade_count == 1
    assert len(risk_manager.registered) == 1


@pytest.mark.asyncio
async def test_short_signal_builds_directional_exits():
    manager, risk_manager, _ = make_manager()
    signal = {
        'symbol': 'ETH/USDT:USDT',
        'side': 'sell',
        'tp_pct': 0.015,
        'sl_pct': 0.01,
    }
    execution_result = {
        'success': True,
        'avg_price': 200.0,
        'filled_amount': 0.5,
    }

    result = await manager.open_position(signal, execution_result)
    position = result['position']

    assert position['stop_loss'] == pytest.approx(202.0, rel=1e-6)
    assert position['take_profit'] == pytest.approx(197.0, rel=1e-6)
    assert position['risk_amount'] == pytest.approx(1.0, rel=1e-6)
    assert signal['stop'] == position['stop_loss']
    assert signal['target'] == position['take_profit']


@pytest.mark.asyncio
async def test_atr_based_guidance_used_when_percent_missing():
    manager, _, _ = make_manager()
    signal = {
        'symbol': 'SOL/USDT:USDT',
        'side': 'sell',
        'atr': 5.0,
        'sl_atr_mult': 1.4,
        'tp_atr_mult': 2.0,
    }
    execution_result = {
        'success': True,
        'avg_price': 300.0,
        'filled_amount': 2.0,
    }

    result = await manager.open_position(signal, execution_result)
    position = result['position']

    assert position['stop_loss'] == pytest.approx(307.0, rel=1e-6)
    assert position['take_profit'] == pytest.approx(290.0, rel=1e-6)
    assert position['risk_amount'] == pytest.approx(14.0, rel=1e-6)


@pytest.mark.asyncio
async def test_open_position_registers_portfolio_manager_state():
    manager, _, portfolio = make_manager()
    signal = {
        'symbol': 'XRP/USDT:USDT',
        'side': 'buy',
        'tp_pct': 0.02,
        'sl_pct': 0.01,
    }
    execution_result = {
        'success': True,
        'avg_price': 0.5,
        'filled_amount': 100.0,
    }

    result = await manager.open_position(signal, execution_result)

    assert result['success'] is True
    assert result['position_id'] in portfolio.registered
    stored = portfolio.registered[result['position_id']]
    assert stored['symbol'] == 'XRP/USDT:USDT'
    assert stored['amount'] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_close_position_updates_portfolio_manager():
    manager, _, portfolio = make_manager()
    signal = {
        'symbol': 'ADA/USDT:USDT',
        'side': 'buy',
        'tp_pct': 0.03,
        'sl_pct': 0.01,
    }
    execution_result = {
        'success': True,
        'avg_price': 1.0,
        'filled_amount': 50.0,
    }

    result = await manager.open_position(signal, execution_result)
    position_id = result['position_id']

    close_payload = await manager.close_position(position_id, exit_price=1.05)

    assert close_payload['success'] is True
    assert any(entry['position_id'] == position_id for entry in portfolio.closed)


@pytest.mark.asyncio
async def test_close_position_notifies_dispatcher():
    manager, _, _ = make_manager()
    signal = {
        'symbol': 'DOGE/USDT:USDT',
        'side': 'buy',
        'tp_pct': 0.02,
        'sl_pct': 0.01,
    }
    execution_result = {
        'success': True,
        'avg_price': 0.1,
        'filled_amount': 1000.0,
    }

    await manager.open_position(signal, execution_result)
    position_id = next(iter(manager.positions))

    calls: list[str] = []

    async def fake_notifier():
        calls.append('ping')

    manager.set_dispatch_notifier(fake_notifier)

    await manager.close_position(position_id, exit_price=0.12)
    # Allow the notifier task to run
    await asyncio.sleep(0)

    assert calls, "Dispatcher notifier should be invoked after closing a position"


def _exit_settings_cfg(*, max_hold_seconds=900, trigger_pct=0.0014, offset_pct=0.0012):
    return {
        "strategies": {
            "mean_reversion": {
                "exit_settings": {
                    "max_hold_seconds": max_hold_seconds,
                    "fee_lock_trigger_pct": trigger_pct,
                    "fee_lock_offset_pct": offset_pct,
                }
            }
        },
        "position_management": {
            "exit_monitoring": {"enabled": True, "check_frequency": 0.05}
        },
    }


@pytest.mark.asyncio
async def test_fee_lock_updates_stop_loss():
    cfg = _exit_settings_cfg(trigger_pct=0.0010, offset_pct=0.0012)
    manager, _, _ = make_manager(cfg=cfg)
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'tp_pct': 0.03,
        'sl_pct': 0.01,
        'strategy_name': 'mean_reversion',
    }
    execution_result = {
        'success': True,
        'avg_price': 100.0,
        'filled_amount': 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position_id = result['position_id']

    await manager.monitor_position_pnl(position_id, current_price=100.2)
    position = manager.positions[position_id]

    assert position['fee_lock_armed'] is True
    assert position['stop_loss'] == pytest.approx(100.12, rel=1e-6)


@pytest.mark.asyncio
async def test_fee_lock_allows_trailing_to_improve_stop():
    cfg = _exit_settings_cfg(trigger_pct=0.0010, offset_pct=0.0012)
    manager, _, _ = make_manager(cfg=cfg)
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'tp_pct': 0.03,
        'sl_pct': 0.01,
        'strategy_name': 'mean_reversion',
    }
    execution_result = {
        'success': True,
        'avg_price': 100.0,
        'filled_amount': 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position_id = result['position_id']
    position = result['position']
    position['trailing_stop_enabled'] = True
    position['trailing_stop_distance'] = 0.005
    position['trailing_stop_activation_threshold'] = 0.0

    await manager.monitor_position_pnl(position_id, current_price=100.2)
    await manager.monitor_position_pnl(position_id, current_price=101.0)

    position = manager.positions[position_id]
    assert position['fee_lock_armed'] is True
    assert position['stop_loss'] == pytest.approx(100.495, rel=1e-6)


@pytest.mark.asyncio
async def test_dynamic_trailing_tightens_long():
    manager, _, _ = make_manager()
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'tp_pct': 0.03,
        'sl_pct': 0.01,
    }
    execution_result = {
        'success': True,
        'avg_price': 100.0,
        'filled_amount': 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position_id = result['position_id']
    position = result['position']
    position['trailing_stop_enabled'] = True
    position['trailing_stop_distance'] = 0.002
    position['trailing_stop_activation_threshold'] = 0.0
    position['execution'] = {
        'trailing_stop': {
            'dynamic_steps': [
                {'activation_pnl': 0.0060, 'new_delta_pct': 0.0015},
                {'activation_pnl': 0.0120, 'new_delta_pct': 0.0010},
            ]
        }
    }

    await manager.monitor_position_pnl(position_id, current_price=100.61)
    position = manager.positions[position_id]
    assert position['trailing_stop_distance'] == pytest.approx(0.0015, rel=1e-12)
    assert position['stop_loss'] == pytest.approx(100.61 * (1 - 0.0015), rel=1e-6)

    await manager.monitor_position_pnl(position_id, current_price=101.20)
    position = manager.positions[position_id]
    assert position['trailing_stop_distance'] == pytest.approx(0.0010, rel=1e-12)
    assert position['stop_loss'] == pytest.approx(101.20 * (1 - 0.0010), rel=1e-6)


@pytest.mark.asyncio
async def test_dynamic_trailing_tightens_short():
    manager, _, _ = make_manager()
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'tp_pct': 0.03,
        'sl_pct': 0.01,
    }
    execution_result = {
        'success': True,
        'avg_price': 100.0,
        'filled_amount': 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position_id = result['position_id']
    position = result['position']
    position['trailing_stop_enabled'] = True
    position['trailing_stop_distance'] = 0.002
    position['trailing_stop_activation_threshold'] = 0.0
    position['execution'] = {
        'trailing_stop': {
            'dynamic_steps': [
                {'activation_pnl': 0.0060, 'new_delta_pct': 0.0015},
                {'activation_pnl': 0.0120, 'new_delta_pct': 0.0010},
            ]
        }
    }

    await manager.monitor_position_pnl(position_id, current_price=99.39)
    position = manager.positions[position_id]
    assert position['trailing_stop_distance'] == pytest.approx(0.0015, rel=1e-12)
    assert position['stop_loss'] == pytest.approx(99.39 * (1 + 0.0015), rel=1e-6)

    await manager.monitor_position_pnl(position_id, current_price=98.80)
    position = manager.positions[position_id]
    assert position['trailing_stop_distance'] == pytest.approx(0.0010, rel=1e-12)
    assert position['stop_loss'] == pytest.approx(98.80 * (1 + 0.0010), rel=1e-6)


@pytest.mark.asyncio
async def test_time_stop_triggers_execute_close_position():
    cfg = _exit_settings_cfg(max_hold_seconds=1, trigger_pct=0.0014, offset_pct=0.0012)
    manager, _, _ = make_manager(cfg=cfg)
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'tp_pct': 0.03,
        'sl_pct': 0.01,
        'strategy_name': 'mean_reversion',
    }
    execution_result = {
        'success': True,
        'avg_price': 100.0,
        'filled_amount': 1.0,
    }

    result = await manager.open_position(signal, execution_result)
    position_id = result['position_id']
    position = manager.positions[position_id]
    position['opened_at'] = datetime.now(timezone.utc) - timedelta(seconds=10)
    position['current_price'] = position['entry_price']

    called = asyncio.Event()

    async def fake_execute(pid, reason):
        if pid == position_id and reason == ExitReason.TIME_EXIT.value:
            called.set()
        return {'success': True}

    manager.execute_close_position = fake_execute

    await manager.start_exit_monitoring()
    await asyncio.wait_for(called.wait(), timeout=1.0)
    await manager.stop_exit_monitoring()

    assert called.is_set()
