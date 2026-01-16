import pytest

from src.core.dca_watcher import DCAWatcher
from src.core.signal_intents import INTENT_SCALE_IN


class _DummyPositionManager:
    def __init__(self, positions_dict):
        # DCAWatcher expects .positions to be a dict of position_id -> position
        self.positions = positions_dict


def _cfg_with_dca(
    *,
    enabled: bool = True,
    allowed_base_strategies=None,
    default_timeframe: str = "5m",
    max_layers: int = 3,
    step_pct: float = 0.015,
):
    if allowed_base_strategies is None:
        allowed_base_strategies = ["adaptive_str", "adaptive_ob"]
    return {
        "dca": {
            "enabled": enabled,
            "allowed_base_strategies": allowed_base_strategies,
            "default_timeframe": default_timeframe,
            "strategy": {
                "max_layers": max_layers,
                "step_pct": step_pct,
                "position_weights": [1.0, 0.7, 0.5],
                "min_volume_usdt": 10.0,
                "cooldown_seconds": 0,
            },
        },
        "strategies": {"adaptive_str": {}},
    }


@pytest.mark.asyncio
async def test_dca_does_not_trigger_when_base_position_profitable_long():
    cfg = _cfg_with_dca(step_pct=0.01)

    pm = _DummyPositionManager(
        {
            "pos1": {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "entry_price": 100.0,
                "amount": 1.0,
                "entry_time": 1,
                "strategy_name": "adaptive_str",
                "timeframe": None,
            }
        }
    )

    async def price_fetcher(_symbol):
        return 101.0  # profitable long

    watcher = DCAWatcher(cfg=cfg, position_manager=pm, price_fetcher=price_fetcher)
    signals = await watcher.run_once()
    assert signals == []


@pytest.mark.asyncio
async def test_dca_triggers_when_base_position_losing_short_and_move_large_enough():
    cfg = _cfg_with_dca(step_pct=0.01)

    pm = _DummyPositionManager(
        {
            "pos1": {
                "symbol": "BTC/USDT:USDT",
                "side": "short",
                "entry_price": 100.0,
                "amount": 1.0,
                "entry_time": 1,
                "strategy_name": "adaptive_str",
                # Intentionally missing timeframe to verify default fallback
                "timeframe": None,
            }
        }
    )

    async def price_fetcher(_symbol):
        # For short, adverse move is price rising above anchor.
        # 102.0 => +2% adverse > step_pct(1%) so layer 1 should trigger.
        return 102.0

    watcher = DCAWatcher(cfg=cfg, position_manager=pm, price_fetcher=price_fetcher)
    signals = await watcher.run_once()

    assert len(signals) == 1
    sig = signals[0]
    assert sig.get("intent") == INTENT_SCALE_IN
    assert sig.get("scale_profile") == "dca"
    assert sig.get("strategy_name") == "adaptive_str"
    assert sig.get("timeframe") == "5m"


@pytest.mark.asyncio
async def test_dca_does_not_trigger_when_base_strategy_not_allowed():
    cfg = _cfg_with_dca(step_pct=0.01, allowed_base_strategies=["adaptive_str"])  # mean_reversion excluded
    cfg["strategies"] = {"mean_reversion": {}, "adaptive_str": {}}

    pm = _DummyPositionManager(
        {
            "pos1": {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "entry_price": 100.0,
                "amount": 1.0,
                "entry_time": 1,
                "strategy_name": "mean_reversion",
                "timeframe": None,
            }
        }
    )

    async def price_fetcher(_symbol):
        # Losing long + adverse move large enough would normally trigger.
        return 98.0

    watcher = DCAWatcher(cfg=cfg, position_manager=pm, price_fetcher=price_fetcher)
    signals = await watcher.run_once()
    assert signals == []


@pytest.mark.asyncio
async def test_dca_does_not_trigger_when_base_position_profitable_short():
    cfg = _cfg_with_dca(step_pct=0.01)

    pm = _DummyPositionManager(
        {
            "pos1": {
                "symbol": "BTC/USDT:USDT",
                "side": "short",
                "entry_price": 100.0,
                "amount": 1.0,
                "entry_time": 1,
                "strategy_name": "adaptive_str",
                "timeframe": None,
            }
        }
    )

    async def price_fetcher(_symbol):
        return 99.0  # profitable short

    watcher = DCAWatcher(cfg=cfg, position_manager=pm, price_fetcher=price_fetcher)
    signals = await watcher.run_once()
    assert signals == []


@pytest.mark.asyncio
async def test_dca_watcher_emits_signal():
    cfg = _cfg_with_dca(step_pct=0.015, allowed_base_strategies=None)
    pm = _DummyPositionManager(
        {
            "pos1": {
                "symbol": "BTC/USDT:USDT",
                "entry_price": 100.0,
                "amount": 1.0,
                "side": "long",
                "stop_loss": 90,
            }
        }
    )

    async def price_fetcher(_symbol):
        return 98.0

    watcher = DCAWatcher(cfg, position_manager=pm, price_fetcher=price_fetcher)
    signals = await watcher.run_once()
    assert len(signals) == 1
    sig = signals[0]
    assert sig["intent"] == INTENT_SCALE_IN
    assert sig["scale_profile"] == "dca"
    assert sig["dca_metadata"]["layer_index"] == 1
    assert pytest.approx(sig["notional"], rel=1e-3) == 100.0


@pytest.mark.asyncio
async def test_dca_watcher_disabled_no_signal():
    cfg = _cfg_with_dca(enabled=False)
    pm = _DummyPositionManager({})
    watcher = DCAWatcher(cfg, position_manager=pm, price_fetcher=lambda _symbol: None)
    signals = await watcher.run_once()
    assert signals == []


@pytest.mark.asyncio
async def test_dca_watcher_respects_layer_cap():
    cfg = _cfg_with_dca(max_layers=2, step_pct=0.015, allowed_base_strategies=None)
    pm = _DummyPositionManager(
        {
            "pos1": {"symbol": "BTC/USDT:USDT", "entry_price": 100.0, "amount": 1.0, "side": "long"},
            "pos2": {
                "symbol": "BTC/USDT:USDT",
                "entry_price": 98.0,
                "amount": 0.7,
                "side": "long",
                "scale_profile": "dca",
                "dca_metadata": {"profile": "dca", "layer_index": 1},
            },
        }
    )

    async def price_fetcher(_symbol):
        return 96.0

    watcher = DCAWatcher(cfg, position_manager=pm, price_fetcher=price_fetcher)
    signals = await watcher.run_once()
    assert signals == []


@pytest.mark.asyncio
async def test_dca_watcher_requires_adverse_move():
    cfg = _cfg_with_dca(step_pct=0.015, allowed_base_strategies=None)
    pm = _DummyPositionManager(
        {"pos1": {"symbol": "BTC/USDT:USDT", "entry_price": 100.0, "amount": 1.0, "side": "long"}}
    )

    async def price_fetcher(_symbol):
        return 99.5

    watcher = DCAWatcher(cfg, position_manager=pm, price_fetcher=price_fetcher)
    signals = await watcher.run_once()
    assert signals == []
