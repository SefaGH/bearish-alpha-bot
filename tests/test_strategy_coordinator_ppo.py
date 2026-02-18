from typing import Any, Dict, cast

import pytest
from unittest.mock import AsyncMock, MagicMock

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
    assert created['cfg']['ppo_enabled'] is True
    assert created['cfg']['ppo_mode'] == 'apply'
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


@pytest.mark.asyncio
async def test_apply_ppo_long_filter_shadow_mode_neutralizes_decision(monkeypatch):
    class _StubAdapter:
        async def get_long_score(self, symbol, **_kwargs):
            return 0.9, {'reason': 'ok', 'symbol': symbol, 'confidence': 0.8}

    stub = _StubAdapter()
    monkeypatch.setattr('core.strategy_coordinator.PPOTradingAdapter', lambda *a, **k: stub, raising=False)

    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=object(),
        config={
            "ml": {
                "governance": {"ppo_mode": "shadow"},
                "reinforcement_learning": {"ppo_enabled": True},
            }
        },
    )
    coordinator.feature_pipeline = object()

    signal = {'side': 'buy', 'symbol': 'BTC/USDT'}
    await coordinator._apply_ppo_long_filter(signal)

    assert signal['ppo_long_score'] == pytest.approx(0.9)
    assert signal['ppo_meta']['governance_mode'] == 'shadow'
    assert signal['ppo_decision_effective'] is False
    assert signal['ppo_shadow_action'] == 'buy'
    assert 'rl_recommendation' not in signal
    assert not hasattr(coordinator, '_last_rl_decision')


@pytest.mark.asyncio
async def test_enrich_signal_for_dynamic_rr_extracts_regime_fields_from_dict():
    cfg = {
        'ml': {
            'reinforcement_learning': {
                'ppo_enabled': True,
            },
            'regime_prediction': {
                'min_confidence_hard_reject': 0.30,
                'min_confidence_full_weight': 0.60,
            },
        }
    }
    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=None,
        config=cfg,
    )
    coordinator.ml_integration = MagicMock()
    coordinator.ml_integration.get_ml_context = AsyncMock(
        return_value={
            "consensus_score": 0.5,
            "regime": {"predicted_regime": "bearish", "confidence": 0.186},
            "quality_score": 0.4,
        }
    )

    enriched = await coordinator._enrich_signal_for_dynamic_rr(
        {"side": "buy", "symbol": "BTC/USDT:USDT"}
    )

    assert enriched["regime_name"] == "bearish"
    assert enriched["regime_confidence"] == pytest.approx(0.186)
    assert enriched["regime_weight"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_enrich_signal_for_dynamic_rr_skips_multiplier_when_ppo_inactive():
    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=None,
        config={"ml": {"reinforcement_learning": {"ppo_enabled": True}}},
    )

    enriched = await coordinator._enrich_signal_for_dynamic_rr(
        {
            "side": "buy",
            "symbol": "BTC/USDT:USDT",
            "ppo_long_score": 0.0,
            "ppo_meta": {"reason": "health_guard_fast", "guard_active": True},
        }
    )

    assert enriched["ppo_rr_multiplier"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_enrich_signal_for_dynamic_rr_shadow_mode_neutralizes_ppo_effects():
    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=None,
        config={
            "ml": {
                "governance": {"ppo_mode": "shadow"},
                "reinforcement_learning": {"ppo_enabled": True},
            }
        },
    )

    enriched = await coordinator._enrich_signal_for_dynamic_rr(
        {
            "side": "buy",
            "symbol": "BTC/USDT:USDT",
            "ppo_long_score": 0.0,
            "ppo_meta": {"reason": "ok", "guard_active": False},
        }
    )

    assert enriched["rl_is_agree"] is False
    assert enriched["rl_action_prob"] == pytest.approx(0.5)
    assert enriched["ppo_rr_multiplier"] == pytest.approx(1.0)
    assert enriched["ppo_rr_reason_code"] == "ml.governance.ppo.shadow.rr_neutralized"


def test_compute_ppo_position_multiplier_shadow_mode_returns_neutral():
    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=None,
        config={
            "ml": {
                "governance": {"ppo_mode": "shadow"},
                "reinforcement_learning": {"ppo_enabled": True},
            }
        },
    )

    signal = {"side": "buy", "ppo_long_score": 0.95}
    multiplier = coordinator._compute_ppo_position_multiplier(signal)

    assert multiplier == pytest.approx(1.0)
    assert signal["ppo_position_reason_code"] == "ml.governance.ppo.shadow.size_neutralized"


@pytest.mark.asyncio
async def test_enrich_signal_for_dynamic_rr_suppresses_legacy_rl_when_ppo_enabled():
    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=None,
        config={
            "ml": {
                "reinforcement_learning": {
                    "legacy_dqn_enabled": True,
                    "ppo_enabled": True,
                    "disable_legacy_rl_when_ppo_enabled": True,
                }
            }
        },
    )
    coordinator._last_rl_decision = {"action": "BUY", "confidence": 0.99}

    enriched = await coordinator._enrich_signal_for_dynamic_rr(
        {"side": "buy", "symbol": "BTC/USDT:USDT"}
    )

    assert enriched["rl_is_agree"] is False
    assert enriched["rl_action_prob"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_enrich_signal_for_dynamic_rr_uses_legacy_rl_when_guard_disabled():
    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=None,
        config={
            "ml": {
                "reinforcement_learning": {
                    "legacy_dqn_enabled": True,
                    "ppo_enabled": True,
                    "disable_legacy_rl_when_ppo_enabled": False,
                }
            }
        },
    )
    coordinator._last_rl_decision = {"action": "BUY", "confidence": 0.88}

    enriched = await coordinator._enrich_signal_for_dynamic_rr(
        {"side": "buy", "symbol": "BTC/USDT:USDT"}
    )

    assert enriched["rl_is_agree"] is True
    assert enriched["rl_action_prob"] == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_enrich_signal_for_dynamic_rr_keeps_existing_regime_signal_and_skips_refetch():
    coordinator = StrategyCoordinator(
        portfolio_manager=_DummyPortfolioManager(),
        risk_manager=_DummyRiskManager(),
        market_data_pipeline=None,
        config={"ml": {"reinforcement_learning": {"ppo_enabled": True}}},
    )
    coordinator.ml_integration = MagicMock()
    coordinator.ml_integration.get_ml_context = AsyncMock(
        return_value={
            "consensus_score": 0.1,
            "regime": {"predicted_regime": "bearish", "confidence": 0.2},
            "quality_score": 0.1,
        }
    )

    enriched = await coordinator._enrich_signal_for_dynamic_rr(
        {
            "side": "buy",
            "symbol": "BTC/USDT:USDT",
            "ml_confidence": 0.95,
            "regime_name": "bullish",
            "regime_confidence": 0.90,
            "regime_weight": 1.0,
        }
    )

    coordinator.ml_integration.get_ml_context.assert_not_called()
    assert enriched["regime_name"] == "bullish"
    assert enriched["regime_confidence"] == pytest.approx(0.90)
    assert enriched["regime_weight"] == pytest.approx(1.0)
    assert enriched["regime_context_source"] == "signal"
