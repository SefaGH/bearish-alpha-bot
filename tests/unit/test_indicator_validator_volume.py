import pandas as pd
import pytest

from core.indicator_validator import IndicatorValidator
from core.stream_data_collector import StreamDataCollector


SYMBOL = "BTC/USDT:USDT"
EXCHANGE = "bingx"


def _make_df(rows: int) -> pd.DataFrame:
    base = 1_700_000_000_000
    data = {
        "timestamp": [base + i * 60_000 for i in range(rows)],
        "open": [1.0] * rows,
        "high": [1.0] * rows,
        "low": [1.0] * rows,
        "close": [1.0] * rows,
        "volume": [10.0] * rows,
    }
    return pd.DataFrame(data)


@pytest.mark.asyncio
async def test_volume_validation_ready_when_bars_available():
    collector = StreamDataCollector()

    config = {
        "validator": {"volume_analyzer_required": True},
        "volume_analyzer": {
            "enabled": True,
            "baseline_short_tf": "1h",
            "baseline_medium_tf": "4h",
            "short_lookback": 2,
            "medium_lookback": 2,
            "window_bars": 2,
            "trade_timeframe": "5m",
        },
    }

    # Prime indicator and volume buffers with enough candles
    collector.prime_buffer_with_dataframe(EXCHANGE, SYMBOL, "1m", _make_df(IndicatorValidator.REQUIRED_CANDLES))
    collector.prime_buffer_with_dataframe(EXCHANGE, SYMBOL, "5m", _make_df(3))
    collector.prime_buffer_with_dataframe(EXCHANGE, SYMBOL, "1h", _make_df(3))
    collector.prime_buffer_with_dataframe(EXCHANGE, SYMBOL, "4h", _make_df(3))

    validator = IndicatorValidator(collector, config=config)

    results = await validator.validate_all(symbols=[SYMBOL], timeframes=["1m", "5m", "1h", "4h"])
    result = results[SYMBOL]

    assert result["status"] == "OK"
    assert result.get("volume_ready") is True
    assert result.get("volume_validation", {}).get("ready") is True


@pytest.mark.asyncio
async def test_volume_validation_fails_when_required_bars_missing():
    collector = StreamDataCollector()

    config = {
        "validator": {"volume_analyzer_required": True},
        "volume_analyzer": {
            "enabled": True,
            "baseline_short_tf": "1h",
            "baseline_medium_tf": "4h",
            "short_lookback": 3,
            "medium_lookback": 3,
            "window_bars": 2,
            "trade_timeframe": "5m",
        },
    }

    # Indicator data is sufficient
    collector.prime_buffer_with_dataframe(EXCHANGE, SYMBOL, "1m", _make_df(IndicatorValidator.REQUIRED_CANDLES))
    collector.prime_buffer_with_dataframe(EXCHANGE, SYMBOL, "5m", _make_df(2))
    collector.prime_buffer_with_dataframe(EXCHANGE, SYMBOL, "1h", _make_df(2))
    collector.prime_buffer_with_dataframe(EXCHANGE, SYMBOL, "4h", _make_df(1))  # Insufficient medium lookback

    validator = IndicatorValidator(collector, config=config)

    results = await validator.validate_all(symbols=[SYMBOL], timeframes=["1m", "5m", "1h", "4h"])
    result = results[SYMBOL]

    assert result.get("volume_ready") is False
    assert result.get("status") == "FAIL"
    assert "VolumeAnalyzer not ready" in result.get("reason", "")
