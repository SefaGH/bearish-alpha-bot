import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.strategy_coordinator import StrategyCoordinator


def _df_from_rows(rows):
    return pd.DataFrame(rows)


def _make_coordinator(*, market_data_pipeline, config):
    pm = MagicMock()
    pm.cfg = config or {}
    pm.performance_monitor = None
    pm.exchange_clients = {}
    pm.get_strategy_allocation.return_value = 0.0
    pm.get_open_positions_for_symbol.return_value = []
    pm.get_open_positions.return_value = {}

    rm = MagicMock()
    coordinator = StrategyCoordinator(pm, rm, market_data_pipeline=market_data_pipeline, config=config)

    # Keep the pipeline minimal for unit tests.
    coordinator._validate_signal_format = MagicMock(return_value={"valid": True})
    coordinator.validate_duplicate = MagicMock(return_value=(True, "OK"))
    coordinator._check_signal_conflicts = AsyncMock(return_value={"has_conflict": False})
    coordinator._assess_signal_risk = AsyncMock(return_value={"acceptable": True, "position_size": 10.0, "metrics": {}})
    coordinator._route_signal = MagicMock(return_value={})
    coordinator._generate_signal_id = MagicMock(return_value="sig_1")
    coordinator.signal_queue = AsyncMock()
    coordinator.signal_queue.put.return_value = (True, None)

    return coordinator


@pytest.mark.asyncio
async def test_crash_guard_high_volume_pump_is_not_panic_and_allows_signal():
    mdp = MagicMock()
    mdp.get_latest_ohlcv = AsyncMock(
        return_value=_df_from_rows(
            [
                {"open": 100, "high": 106, "low": 99, "close": 104, "atr": 1.2, "ema_fast": 101},
                {"open": 104, "high": 112, "low": 103, "close": 110, "atr": 1.5, "ema_fast": 102},
            ]
        )
    )

    config = {
        "volume_analyzer": {"enabled": False},
        "risk": {"queue": {"max_queue_depth": 5}},
        "strategies": {
            "adaptive_ob": {
                "volume_filters": {"enabled": False},
                "crash_guard": {
                    "enabled": True,
                    "panic_volume_buckets": ["HIGH", "EXTREME"],
                    "panic_tf": "5m",
                    "panic_fast_drop_pct": 0.008,
                    "panic_atr_pct": 0.006,
                    "panic_bear_body_ratio": 0.60,
                    "tp_rr_fix_mode": "off",
                },
            }
        },
    }

    coordinator = _make_coordinator(market_data_pipeline=mdp, config=config)

    enriched = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry": 100.0,
        "stop": 99.0,
        "target": 105.0,
        "rr_ratio": 5.0,
        "volume_bucket": "EXTREME",
        "meta": {"rsi_hook": False, "bull_candle": False, "reclaim": False},
    }
    coordinator._enrich_signal = AsyncMock(return_value=enriched)

    result = await coordinator.process_strategy_signal("adaptive_ob", {"symbol": "BTC/USDT:USDT", "side": "long"})
    assert result["status"] == "accepted"
    assert result["enriched_signal"]["meta"]["panic_guard"]["is_panic_state"] is False


@pytest.mark.asyncio
async def test_crash_guard_high_volume_crash_blocks_without_reversal():
    mdp = MagicMock()
    mdp.get_latest_ohlcv = AsyncMock(
        return_value=_df_from_rows(
            [
                {"open": 100, "high": 101, "low": 95, "close": 100, "atr": 2.0, "ema_fast": 102},
                {"open": 100, "high": 100, "low": 85, "close": 90, "atr": 5.0, "ema_fast": 101},
            ]
        )
    )

    config = {
        "volume_analyzer": {"enabled": False},
        "risk": {"queue": {"max_queue_depth": 5}},
        "strategies": {
            "adaptive_ob": {
                "volume_filters": {"enabled": False},
                "crash_guard": {
                    "enabled": True,
                    "panic_volume_buckets": ["HIGH", "EXTREME"],
                    "panic_tf": "5m",
                    "panic_fast_drop_pct": 0.008,
                    "panic_atr_pct": 0.006,
                    "panic_bear_body_ratio": 0.60,
                    "tp_rr_fix_mode": "off",
                },
            }
        },
    }

    coordinator = _make_coordinator(market_data_pipeline=mdp, config=config)
    coordinator._enrich_signal = AsyncMock(
        return_value={
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "entry": 100.0,
            "stop": 99.0,
            "target": 105.0,
            "rr_ratio": 5.0,
            "volume_bucket": "EXTREME",
            "meta": {"rsi_hook": False, "bull_candle": False, "reclaim": False},
        }
    )

    result = await coordinator.process_strategy_signal("adaptive_ob", {"symbol": "BTC/USDT:USDT", "side": "long"})
    assert result["status"] == "rejected"
    assert result["reason_code"] == "panic_veto_no_reversal"
    assert result["stage"] == "panic_guard"


@pytest.mark.asyncio
async def test_crash_guard_extreme_bucket_requires_reclaim_even_if_bull_candle():
    mdp = MagicMock()
    mdp.get_latest_ohlcv = AsyncMock(
        return_value=_df_from_rows(
            [
                {"open": 100, "high": 101, "low": 95, "close": 100, "atr": 2.0, "ema_fast": 102},
                {"open": 100, "high": 100, "low": 85, "close": 90, "atr": 5.0, "ema_fast": 101},
            ]
        )
    )

    config = {
        "volume_analyzer": {"enabled": False},
        "risk": {"queue": {"max_queue_depth": 5}},
        "strategies": {
            "adaptive_ob": {
                "volume_filters": {"enabled": False},
                "crash_guard": {
                    "enabled": True,
                    "panic_volume_buckets": ["HIGH", "EXTREME"],
                    "panic_tf": "5m",
                    "panic_fast_drop_pct": 0.008,
                    "panic_atr_pct": 0.006,
                    "panic_bear_body_ratio": 0.60,
                    "panic_ema_gap_atr_threshold": 3.0,
                    "extreme_gap_atr_threshold": 5.0,
                    "tp_rr_fix_mode": "off",
                },
            }
        },
    }

    coordinator = _make_coordinator(market_data_pipeline=mdp, config=config)
    coordinator._enrich_signal = AsyncMock(
        return_value={
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "entry": 100.0,
            "stop": 99.0,
            "target": 105.0,
            "rr_ratio": 5.0,
            "volume_bucket": "EXTREME",
            # rsi_hook True, bull_candle True, but reclaim False => must be rejected under EXTREME rules
            "meta": {"rsi_hook": True, "bull_candle": True, "reclaim": False},
        }
    )

    result = await coordinator.process_strategy_signal("adaptive_ob", {"symbol": "BTC/USDT:USDT", "side": "long"})
    assert result["status"] == "rejected"
    assert result["reason_code"] == "panic_veto_no_reversal"
    assert result["stage"] == "panic_guard"


@pytest.mark.asyncio
async def test_crash_guard_high_volume_false_positive_control_allows_pump_even_with_high_atr():
    mdp = MagicMock()
    mdp.get_latest_ohlcv = AsyncMock(
        return_value=_df_from_rows(
            [
                {"open": 100, "high": 105, "low": 95, "close": 101, "atr": 10.0, "ema_fast": 99},
                {"open": 101, "high": 120, "low": 100, "close": 115, "atr": 12.0, "ema_fast": 100},
            ]
        )
    )

    config = {
        "volume_analyzer": {"enabled": False},
        "risk": {"queue": {"max_queue_depth": 5}},
        "strategies": {
            "adaptive_ob": {
                "volume_filters": {"enabled": False},
                "crash_guard": {
                    "enabled": True,
                    "panic_volume_buckets": ["HIGH", "EXTREME"],
                    "panic_tf": "5m",
                    "panic_fast_drop_pct": 0.008,
                    "panic_atr_pct": 0.006,
                    "panic_bear_body_ratio": 0.60,
                    "tp_rr_fix_mode": "off",
                },
            }
        },
    }

    coordinator = _make_coordinator(market_data_pipeline=mdp, config=config)
    coordinator._enrich_signal = AsyncMock(
        return_value={
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "entry": 100.0,
            "stop": 99.0,
            "target": 105.0,
            "rr_ratio": 5.0,
            "volume_bucket": "EXTREME",
            "meta": {"rsi_hook": False, "bull_candle": True, "reclaim": True},
        }
    )

    result = await coordinator.process_strategy_signal("adaptive_ob", {"symbol": "BTC/USDT:USDT", "side": "long"})
    assert result["status"] == "accepted"
    assert result["enriched_signal"]["meta"]["panic_guard"]["is_panic_state"] is False


@pytest.mark.asyncio
async def test_stop_loss_cooldown_blocks_reentry_when_panic_only_enabled():
    mdp = MagicMock()
    mdp.get_latest_ohlcv = AsyncMock(
        return_value=_df_from_rows(
            [
                {"open": 100, "high": 101, "low": 95, "close": 100, "atr": 2.0, "ema_fast": 102},
                {"open": 100, "high": 100, "low": 85, "close": 90, "atr": 5.0, "ema_fast": 101},
            ]
        )
    )

    config = {
        "volume_analyzer": {"enabled": False},
        "risk": {"queue": {"max_queue_depth": 5}},
        "strategies": {
            "adaptive_ob": {
                "volume_filters": {"enabled": False},
                "crash_guard": {
                    "enabled": True,
                    "panic_volume_buckets": ["HIGH", "EXTREME"],
                    "panic_tf": "5m",
                    "panic_fast_drop_pct": 0.008,
                    "panic_atr_pct": 0.006,
                    "panic_bear_body_ratio": 0.60,
                    "cooldown_mode": "panic_only",
                    "cooldown_seconds": 30,
                    "tp_rr_fix_mode": "off",
                },
            }
        },
    }

    coordinator = _make_coordinator(market_data_pipeline=mdp, config=config)

    await coordinator.handle_trade_closed(
        {
            "event": "TRADE_CLOSED",
            "strategy_name": "adaptive_ob",
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "exit_reason": "stop_loss",
            "volume_bucket_at_entry": "EXTREME",
        }
    )

    result = await coordinator.process_strategy_signal("adaptive_ob", {"symbol": "BTC/USDT:USDT", "side": "long"})
    assert result["status"] == "dropped"
    assert result["reason"] == "stop_loss_cooldown_active"
    assert result["stage"] == "cooldown"


@pytest.mark.asyncio
async def test_stop_loss_reentry_requires_reversal_when_reversal_only_enabled():
    mdp = MagicMock()
    mdp.get_latest_ohlcv = AsyncMock(return_value=_df_from_rows([]))

    config = {
        "volume_analyzer": {"enabled": False},
        "risk": {"queue": {"max_queue_depth": 5}},
        "strategies": {
            "adaptive_ob": {
                "volume_filters": {"enabled": False},
                "crash_guard": {
                    "enabled": True,
                    "panic_volume_buckets": ["HIGH", "EXTREME"],
                    "panic_tf": "5m",
                    "panic_fast_drop_pct": 0.008,
                    "panic_atr_pct": 0.006,
                    "panic_bear_body_ratio": 0.60,
                    "cooldown_mode": "reversal_only",
                    "tp_rr_fix_mode": "off",
                },
            }
        },
    }

    coordinator = _make_coordinator(market_data_pipeline=mdp, config=config)

    await coordinator.handle_trade_closed(
        {
            "event": "TRADE_CLOSED",
            "strategy_name": "adaptive_ob",
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "exit_reason": "stop_loss",
            "volume_bucket_at_entry": "NORMAL",
        }
    )

    assert coordinator._is_strategy_in_cooldown("adaptive_ob", "BTC/USDT:USDT", side="long") is False

    coordinator._enrich_signal = AsyncMock(
        return_value={
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "entry": 100.0,
            "stop": 99.0,
            "target": 105.0,
            "rr_ratio": 5.0,
            "volume_bucket": "NORMAL",
            "meta": {"rsi_hook": False, "bull_candle": False, "reclaim": False},
        }
    )
    result = await coordinator.process_strategy_signal("adaptive_ob", {"symbol": "BTC/USDT:USDT", "side": "long"})
    assert result["status"] == "rejected"
    assert result["reason_code"] == "stop_loss_reversal_required"
    assert result["stage"] == "cooldown"

    coordinator._enrich_signal = AsyncMock(
        return_value={
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "entry": 100.0,
            "stop": 99.0,
            "target": 105.0,
            "rr_ratio": 5.0,
            "volume_bucket": "NORMAL",
            "meta": {"rsi_hook": True, "bull_candle": True, "reclaim": False},
        }
    )
    result2 = await coordinator.process_strategy_signal("adaptive_ob", {"symbol": "BTC/USDT:USDT", "side": "long"})
    assert result2["status"] == "accepted"
