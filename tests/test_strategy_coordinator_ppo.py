from typing import Any, Dict, cast

import pytest

from core.strategy_coordinator import StrategyCoordinator  # type: ignore


class _DummyPortfolioManager:
    def __init__(self, equity: float = 10_000.0):
        self._equity = equity
        self.cfg = {}

    def get_current_equity(self):
        return self._equity

    def get_open_positions(self):
        return {}


class _DummyRiskManager:
    async def validate_new_position(self, *args, **kwargs):  # pragma: no cover - unused in these tests
        return True, "ok", {}


@pytest.mark.asyncio
async def test_strategy_coordinator_initializes_ppo_adapter_when_ml_ready(monkeypatch):
    created = {}

    class _StubAdapter:
        def __init__(self, rl_cfg, *, market_data_pipeline, feature_pipeline):
            created['cfg'] = rl_cfg
            created['market'] = market_data_pipeline
            created['features'] = feature_pipeline

        async def get_long_score(self, *args, **kwargs):  # pragma: no cover - not invoked in this test
            return 1.0, {'reason': 'stub'}

    monkeypatch.setattr('core.strategy_coordinator.PPOTradingAdapter', _StubAdapter, raising=False)

    cfg = {
        'ml': {
            'reinforcement_learning': {
                'ppo_enabled': True,
            }
        }
    }
    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=object(),
        config=cfg,
    )

    coordinator.feature_pipeline = object()
    coordinator.on_ml_components_connected()

    assert isinstance(coordinator.ppo_adapter, _StubAdapter)
    assert created['cfg'] == cfg['ml']['reinforcement_learning']
    assert created['market']
    assert created['features']


@pytest.mark.asyncio
async def test_apply_ppo_long_filter_triggers_lazy_initialization(monkeypatch):
    class _StubAdapter:
        def __init__(self, *_args, **_kwargs):
            self.call_count = 0

        async def get_long_score(self, symbol, **_kwargs):
            self.call_count += 1
            return 0.9, {
                'reason': 'ok',
                'symbol': symbol,
                'lookback': {
                    'overall': {'price_change_pct': 0.01},
                    'bars_available': 42,
                },
            }

    stub = _StubAdapter(None)
    monkeypatch.setattr('core.strategy_coordinator.PPOTradingAdapter', lambda *a, **k: stub, raising=False)

    cfg = {
        'ml': {
            'reinforcement_learning': {
                'ppo_enabled': True,
            }
        }
    }

    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=object(),
        config=cfg,
    )
    coordinator.feature_pipeline = object()

    signal = {'side': 'buy', 'symbol': 'BTC/USDT'}
    await coordinator._apply_ppo_long_filter(signal)

    assert signal['ppo_long_score'] == pytest.approx(0.9)
    assert stub.call_count == 1
    assert 'ppo_meta' in signal
    assert 'ppo_lookback_meta' in signal
    lookback_meta = cast(Dict[str, Any], signal['ppo_lookback_meta'])
    assert lookback_meta['overall']['price_change_pct'] == pytest.approx(0.01)
