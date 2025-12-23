import pytest

from src.core.dca_watcher import DCAWatcher
from src.core.signal_intents import INTENT_SCALE_IN


class StubPositionManager:
    def __init__(self, positions):
        self.positions = positions


@pytest.mark.asyncio
async def test_dca_watcher_emits_signal():
    cfg = {
        "dca": {
            "enabled": True,
            "strategy": {
                "max_layers": 3,
                "step_pct": 0.015,
                "position_weights": [1.0, 0.7, 0.5],
                "min_volume_usdt": 10,
                "cooldown_seconds": 0,
            },
        }
    }
    positions = {
        "pos1": {"symbol": "BTC/USDT:USDT", "entry_price": 100.0, "amount": 1.0, "side": "long", "stop_loss": 90},
    }
    pm = StubPositionManager(positions)
    watcher = DCAWatcher(cfg, position_manager=pm, price_fetcher=lambda symbol: 98.0)

    signals = await watcher.run_once()
    assert len(signals) == 1
    sig = signals[0]
    assert sig["intent"] == INTENT_SCALE_IN
    assert sig["scale_profile"] == "dca"
    assert sig["dca_metadata"]["layer_index"] == 1
    assert pytest.approx(sig["notional"], rel=1e-3) == 100.0


@pytest.mark.asyncio
async def test_dca_watcher_disabled_no_signal():
    cfg = {"dca": {"enabled": False}}
    pm = StubPositionManager({})
    watcher = DCAWatcher(cfg, position_manager=pm, price_fetcher=lambda symbol: 98.0)
    signals = await watcher.run_once()
    assert signals == []


@pytest.mark.asyncio
async def test_dca_watcher_respects_layer_cap():
    cfg = {
        "dca": {
            "enabled": True,
            "strategy": {"max_layers": 2, "step_pct": 0.015, "min_volume_usdt": 10, "cooldown_seconds": 0},
        }
    }
    positions = {
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
    pm = StubPositionManager(positions)
    watcher = DCAWatcher(cfg, position_manager=pm, price_fetcher=lambda symbol: 96.0)
    signals = await watcher.run_once()
    assert signals == []


@pytest.mark.asyncio
async def test_dca_watcher_requires_adverse_move():
    cfg = {
        "dca": {
            "enabled": True,
            "strategy": {"max_layers": 3, "step_pct": 0.015, "min_volume_usdt": 10, "cooldown_seconds": 0},
        }
    }
    positions = {"pos1": {"symbol": "BTC/USDT:USDT", "entry_price": 100.0, "amount": 1.0, "side": "long"}}
    pm = StubPositionManager(positions)
    watcher = DCAWatcher(cfg, position_manager=pm, price_fetcher=lambda symbol: 99.5)
    signals = await watcher.run_once()
    assert signals == []
