import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.core.strategy_coordinator import StrategyCoordinator
from src.utils.volume_utils import get_bucket_rank

@pytest.fixture
def mock_deps():
    portfolio_manager = MagicMock()
    portfolio_manager.exchange_clients = {}
    portfolio_manager.get_strategy_allocation.return_value = 1000.0
    portfolio_manager.performance_monitor = MagicMock()
    portfolio_manager.performance_monitor.get_strategy_summary.return_value = {'metrics': {}}
    
    risk_manager = AsyncMock()
    risk_manager.assess_signal.return_value = {'acceptable': True, 'position_size': 100.0}
    
    market_data_pipeline = MagicMock()
    
    volume_analyzer = AsyncMock()
    
    return portfolio_manager, risk_manager, market_data_pipeline, volume_analyzer

@pytest.mark.asyncio
async def test_volume_score_calculation(mock_deps):
    pm, rm, mdp, va = mock_deps
    
    config = {
        'strategies': {
            'test_strat': {
                'volume_filters': {
                    'enabled': True,
                    'use_volume_strength_in_score': True,
                    'volume_score_weight': 0.2
                }
            }
        },
        'volume_analyzer': {'enabled': True}
    }
    
    coordinator = StrategyCoordinator(pm, rm, mdp, config=config, volume_analyzer=va)
    
    # Mock volume context
    mock_ctx = MagicMock()
    mock_ctx.bucket = 'NORMAL'
    mock_ctx.volume_strength = 1.5
    va.compute_context.return_value = mock_ctx
    
    signal = {'symbol': 'BTC/USDT', 'score': 0.5, 'entry': 100, 'stop': 90, 'target': 110, 'side': 'buy'}
    
    enriched = await coordinator._enrich_signal('test_strat', signal)
    
    # Expected: strength 1.5 -> raw score 0.75 -> weighted 0.15 -> total 0.65
    assert enriched['volume_score'] == pytest.approx(0.15)
    assert enriched['score'] == pytest.approx(0.65)
    assert enriched['volume_bucket'] == 'NORMAL'

@pytest.mark.asyncio
async def test_volume_gating_min_bucket(mock_deps):
    pm, rm, mdp, va = mock_deps
    
    config = {
        'strategies': {
            'test_strat': {
                'volume_filters': {
                    'enabled': True,
                    'min_bucket': 'NORMAL'
                },
                'allow_low_volume': False
            }
        },
        'volume_analyzer': {'enabled': True}
    }
    
    coordinator = StrategyCoordinator(pm, rm, mdp, config=config, volume_analyzer=va)
    
    # Mock internal methods
    coordinator._validate_signal_format = MagicMock(return_value={'valid': True})
    coordinator._enrich_signal = AsyncMock()
    coordinator.validate_duplicate = MagicMock(return_value=(True, "OK"))
    coordinator._check_signal_conflicts = AsyncMock(return_value={'has_conflict': False})
    coordinator._assess_signal_risk = AsyncMock(return_value={'acceptable': True, 'position_size': 10.0})
    coordinator._route_signal = MagicMock(return_value={})
    coordinator._generate_signal_id = MagicMock(return_value="sig_1")
    coordinator.signal_queue = AsyncMock()
    coordinator.signal_queue.put.return_value = (True, None)

    base_signal = {'symbol': 'BTC/USDT', 'entry': 100.0, 'stop': 90.0, 'target': 110.0, 'side': 'buy'}

    # Case 1: LOW bucket (should reject)
    coordinator._enrich_signal.return_value = {
        **base_signal,
        'volume_bucket': 'LOW', 
        'volume_strength': 0.2
    }
    
    result = await coordinator.process_strategy_signal('test_strat', base_signal)
    assert result['status'] == 'rejected'
    assert 'rejected_low_bucket' in result.get('reason', '') or "Volume bucket 'LOW' < min 'NORMAL'" in result.get('reason', '')

    # Case 2: NORMAL bucket (should accept)
    coordinator._enrich_signal.return_value = {
        **base_signal,
        'volume_bucket': 'NORMAL', 
        'volume_strength': 0.5
    }
    
    result = await coordinator.process_strategy_signal('test_strat', base_signal)
    assert result['status'] == 'accepted'

@pytest.mark.asyncio
async def test_volume_gating_allow_low_override(mock_deps):
    pm, rm, mdp, va = mock_deps
    
    config = {
        'strategies': {
            'test_strat': {
                'volume_filters': {
                    'enabled': True,
                    'min_bucket': 'NORMAL'
                },
                'allow_low_volume': False
            }
        },
        'volume_analyzer': {'enabled': True}
    }
    
    coordinator = StrategyCoordinator(pm, rm, mdp, config=config, volume_analyzer=va)
    
    # Mock internal methods
    coordinator._validate_signal_format = MagicMock(return_value={'valid': True})
    coordinator._enrich_signal = AsyncMock()
    coordinator.validate_duplicate = MagicMock(return_value=(True, "OK"))
    coordinator._check_signal_conflicts = AsyncMock(return_value={'has_conflict': False})
    coordinator._assess_signal_risk = AsyncMock(return_value={'acceptable': True, 'position_size': 10.0})
    coordinator._route_signal = MagicMock(return_value={})
    coordinator._generate_signal_id = MagicMock(return_value="sig_1")
    coordinator.signal_queue = AsyncMock()
    coordinator.signal_queue.put.return_value = (True, None)

    base_signal = {'symbol': 'BTC/USDT', 'entry': 100.0, 'stop': 90.0, 'target': 110.0, 'side': 'buy'}

    # Case: Bucket is LOW. min_bucket is NORMAL. Should reject.
    
    coordinator._enrich_signal.return_value = {
        **base_signal,
        'volume_bucket': 'LOW', 
        'volume_strength': 0.2
    }
    
    result = await coordinator.process_strategy_signal('test_strat', base_signal)
    assert result['status'] == 'rejected'
    
    # Now enable allow_low_volume override
    coordinator.config['strategies']['test_strat']['allow_low_volume'] = True
    
    # Bucket LOW, min_bucket NORMAL -> Rank check fails.
    # But allow_low_volume is True -> Should Accept.
    result = await coordinator.process_strategy_signal('test_strat', base_signal)
    assert result['status'] == 'accepted'

