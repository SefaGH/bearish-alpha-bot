import pytest

from src.safety.signal_integrity_guard import SignalIntegrityGuard


class DummyMarketDataPipeline:
    def __init__(self, price: float):
        self._price = float(price)

    async def get_latest_price(self, _symbol: str, timeframe: str = "1m"):
        return self._price


@pytest.mark.asyncio
async def test_episode_c_price_gap_abort_with_atr_guard():
    config = {
        "signals": {
            "integrity_guard": {
                "enabled": True,
                "max_deviation_pct": 0.001,
                "atr_guard_enabled": True,
                "atr_guard_mult": 0.5,
                "spread_buffer_bps": 0.0,
            }
        }
    }
    guard = SignalIntegrityGuard(config, DummyMarketDataPipeline(price=101.0))
    signal = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "5m",
        "side": "short",
        "entry": 100.0,
        "atr": 0.2,
        "meta": {"price_meta": {"price_used": 100.0}},
    }

    result = await guard.validate(signal)
    assert result["valid"] is False
    assert result["action"] == "reject"
    assert result["reason"] == "price_moved_fast"
    meta = result.get("metadata", {})
    assert meta.get("reason_code") == "price_moved_fast"
    assert meta.get("gap_bps") > meta.get("threshold_bps")


@pytest.mark.asyncio
async def test_episode_c_atr_missing_fallback_min_gap_bps():
    config = {
        "signals": {
            "integrity_guard": {
                "enabled": True,
                "max_deviation_pct": 0.0001,
                "atr_guard_enabled": True,
                "atr_guard_mult": 0.5,
                "min_gap_bps_fallback": 8.0,
            }
        }
    }
    guard = SignalIntegrityGuard(config, DummyMarketDataPipeline(price=100.09))
    signal = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "5m",
        "side": "short",
        "entry": 100.0,
        "meta": {"price_meta": {"price_used": 100.0}},
    }

    result = await guard.validate(signal)
    assert result["valid"] is False
    assert result["reason"] == "price_moved_fast"
    meta = result.get("metadata", {})
    assert meta.get("min_gap_bps_fallback") == pytest.approx(8.0, rel=1e-6)
    assert meta.get("gap_bps") > meta.get("threshold_bps")


@pytest.mark.asyncio
async def test_episode_c_impulse_veto_metadata_fields():
    config = {
        "signals": {
            "integrity_guard": {
                "enabled": True,
                "impulse_guard_enabled": True,
            }
        }
    }
    guard = SignalIntegrityGuard(config, DummyMarketDataPipeline(price=100.0))
    signal = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "5m",
        "side": "short",
        "entry": 100.0,
        "meta": {
            "impulse_guard": {
                "enabled": True,
                "is_shock_move": True,
                "body_atr_mult": 1.6,
                "sum2_range_atr_mult": 2.7,
                "candle_dir": "up",
                "trade_dir": "down",
                "require_opposite": True,
            }
        },
    }

    result = await guard.validate(signal)
    assert result["valid"] is False
    assert result["reason"] == "impulse_shock"
    meta = result.get("metadata", {})
    assert meta.get("reason_code") == "impulse_shock"
    assert meta.get("body_atr_mult") is not None
    assert meta.get("sum2_range_atr_mult") is not None
    assert meta.get("candle_dir") == "up"
    assert meta.get("trade_dir") == "down"
