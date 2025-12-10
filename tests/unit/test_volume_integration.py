import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.strategy_coordinator import StrategyCoordinator
from src.core.risk_rules import VolumeAwarePositionSizingRule


class StubPositionSizing:
    async def calculate_optimal_size(self, signal, method: str = "fixed_risk_capped", return_signal: bool = True, **kwargs):
        sized = dict(signal)
        sized.setdefault('amount', 1.0)
        sized.setdefault('notional', sized.get('entry', 0) * sized['amount'])
        return sized


class StubVolumeAnalyzer:
    def __init__(self, bucket="NORMAL", strength=1.0):
        self.bucket = bucket
        self.strength = strength
        self.calls = 0

    async def compute_context(self, symbol=None, trade_timeframe=None, as_of_ts=None):
        self.calls += 1
        return SimpleNamespace(
            volume_strength=self.strength,
            bucket=self.bucket,
            ratio_short=1.0,
            ratio_medium=1.0,
            ratio_combined=1.0,
        )


class StubRiskManager:
    def __init__(self):
        self.calls = []

    async def validate_new_position(self, signal, portfolio_manager=None):
        self.calls.append(dict(signal))
        return True, "ok", {}


def _make_coordinator(config=None, volume_analyzer=None, risk_manager=None):
    cfg = config or {}
    portfolio_manager = MagicMock()
    portfolio_manager.get_current_equity.return_value = 1000
    rm = risk_manager or StubRiskManager()
    coord = StrategyCoordinator(
        portfolio_manager,
        rm,
        market_data_pipeline=None,
        config=cfg,
        volume_analyzer=volume_analyzer,
    )
    coord.position_sizing = StubPositionSizing()
    return coord


@pytest.mark.asyncio
async def test_bucket_gating_rejects_low_volume():
    cfg = {
        'volume_analyzer': {'enabled': True},
        'strategies': {
            'adaptive_ob': {
                'volume_filters': {
                    'enabled': True,
                    'min_bucket': 'NORMAL',
                    'high_volume_min_bucket': 'HIGH',
                }
            }
        },
    }
    volume = StubVolumeAnalyzer(bucket="LOW", strength=0.2)
    coord = _make_coordinator(cfg, volume_analyzer=volume)

    signal = {
        'strategy_name': 'adaptive_ob',
        'strategy_volume_decision': 'accepted',
        'symbol': 'BTC/USDT',
        'timeframe': '5m',
        'entry': 100,
        'stop': 95,
        'target': 110,
        'side': 'buy',
        'quality_score': 0.5,
    }

    assessment = await coord._assess_signal_risk(signal)
    assert assessment['acceptable'] is False
    assert assessment['metrics']['volume_bucket'] == 'LOW'


@pytest.mark.asyncio
async def test_bucket_high_boosts_quality_score():
    cfg = {
        'volume_analyzer': {'enabled': True},
        'strategies': {
            'adaptive_short_the_rip': {
                'volume_filters': {
                    'enabled': True,
                    'min_bucket': 'NORMAL',
                    'high_volume_min_bucket': 'HIGH',
                    'use_volume_strength_in_score': True,
                    'volume_score_weight': 0.2,
                }
            }
        },
    }
    volume = StubVolumeAnalyzer(bucket="HIGH", strength=1.2)
    rm = StubRiskManager()
    coord = _make_coordinator(cfg, volume_analyzer=volume, risk_manager=rm)

    signal = {
        'strategy_name': 'adaptive_short_the_rip',
        'strategy_volume_decision': 'accepted',
        'symbol': 'ETH/USDT',
        'timeframe': '5m',
        'entry': 2000,
        'stop': 1900,
        'target': 2200,
        'side': 'sell',
        'quality_score': 0.1,
    }

    assessment = await coord._assess_signal_risk(signal)
    assert assessment['acceptable'] is True
    boosted = rm.calls[0]['quality_score']
    # base quality defaults to 0.0; weight 0.2 * strength 1.2 ≈ 0.24 boost
    assert boosted == pytest.approx(0.24, rel=1e-3)


@pytest.mark.parametrize(
    "bucket,expected_ps,expected_sl,expected_tp",
    [
        ("LOW", 0.8, 1.1, 0.9),
        ("NORMAL", 1.0, 1.0, 1.0),
        ("HIGH", 1.2, 0.9, 1.05),
        ("EXTREME", 1.5, 0.8, 1.1),
    ],
)
def test_volume_risk_matrix_multipliers(bucket, expected_ps, expected_sl, expected_tp):
    matrix = {
        'LOW': {
            'position_size_multiplier': 0.8,
            'stop_loss_multiplier': 1.1,
            'take_profit_multiplier': 0.9,
        },
        'NORMAL': {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
        },
        'HIGH': {
            'position_size_multiplier': 1.2,
            'stop_loss_multiplier': 0.9,
            'take_profit_multiplier': 1.05,
        },
        'EXTREME': {
            'position_size_multiplier': 1.5,
            'stop_loss_multiplier': 0.8,
            'take_profit_multiplier': 1.1,
        },
    }
    rule = VolumeAwarePositionSizingRule(matrix)
    signal = {
        'volume_bucket': bucket,
        'volume_ctx_source': 'analyzer',
        'position_size': 1.0,
        'stop_loss_dist': 10,
        'take_profit_dist': 20,
        'symbol': 'BTC/USDT',
    }

    ok, _ = rule.validate(signal, portfolio_manager={})
    assert ok is True
    assert signal['position_size'] == pytest.approx(expected_ps)
    assert signal['stop_loss_dist'] == pytest.approx(10 * expected_sl)
    assert signal['take_profit_dist'] == pytest.approx(20 * expected_tp)


def test_volume_risk_rule_skips_non_analyzer_context():
    matrix = {
        'HIGH': {
            'position_size_multiplier': 1.2,
            'stop_loss_multiplier': 0.9,
            'take_profit_multiplier': 1.1,
        }
    }
    rule = VolumeAwarePositionSizingRule(matrix)
    signal = {
        'volume_bucket': 'HIGH',
        'volume_ctx_source': 'fallback',
        'position_size': 5.0,
        'stop_loss_dist': 2.0,
        'take_profit_dist': 4.0,
        'symbol': 'BTC/USDT',
    }

    ok, reason = rule.validate(signal, portfolio_manager={})
    assert ok is True
    assert 'skipping' in reason.lower()
    assert signal['position_size'] == 5.0
    assert signal['stop_loss_dist'] == 2.0
    assert signal['take_profit_dist'] == 4.0


@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.parametrize("volume_cfg,inject_analyzer", [({'enabled': False}, True), (None, False)])
async def test_fallback_path_when_analyzer_disabled_or_missing(volume_cfg, inject_analyzer):
    cfg = {
        'strategies': {
            'adaptive_ob': {
                'volume_filters': {
                    'enabled': True,
                    'min_bucket': 'HIGH',
                }
            }
        },
    }
    if volume_cfg is not None:
        cfg['volume_analyzer'] = volume_cfg

    rm = StubRiskManager()
    volume = StubVolumeAnalyzer(bucket="EXTREME", strength=1.5)
    coord = _make_coordinator(cfg, volume_analyzer=volume if inject_analyzer else None, risk_manager=rm)
    # Even if a volume analyzer instance is injected later, the disabled flag should prevent usage
    if inject_analyzer:
        coord.volume_analyzer = volume

    signal = {
        'strategy_name': 'adaptive_ob',
        'strategy_volume_decision': 'accepted',
        'symbol': 'BTC/USDT',
        'timeframe': '5m',
        'entry': 100,
        'stop': 95,
        'target': 110,
        'side': 'buy',
        'quality_score': 0.25,
    }

    assessment = await coord._assess_signal_risk(signal)

    assert assessment['acceptable'] is True
    assert rm.calls[0].get('volume_ctx_source') != 'analyzer'
    assert rm.calls[0].get('volume_bucket') == 'NORMAL'
    assert rm.calls[0].get('quality_score', 0.0) == pytest.approx(0.0)
    assert volume.calls == 0
