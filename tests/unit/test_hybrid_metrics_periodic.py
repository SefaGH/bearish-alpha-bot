import os
import sys
import logging
from unittest.mock import patch

# Ensure src/ is importable when running via pytest.cmd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.market_data_pipeline import MarketDataPipeline


def test_hybrid_metrics_cumulative_and_periodic(caplog):
    pipeline = MarketDataPipeline(
        exchanges={},
        config={"websocket": {"hybrid_fallback_metrics_interval_sec": 1}},
        websocket_manager=None,
    )

    caplog.set_level(logging.INFO)

    state_key = pipeline._hybrid_state_key("bingx", "BTC/USDT:USDT", "30m")

    # Prevent an immediate metrics log on the first call by setting last-log time.
    pipeline._hybrid_metrics_last_log_ts[state_key] = 1000.0

    with patch("time.time", return_value=1000.1):
        pipeline._record_hybrid_metrics(
            state_key=state_key,
            fallback_reason=None,
            timeframe="30m",
            symbol="BTC/USDT:USDT",
            inject_ts_ms=1700000000000,
        )

    with patch("time.time", return_value=1002.2):
        pipeline._record_hybrid_metrics(
            state_key=state_key,
            fallback_reason=None,
            timeframe="30m",
            symbol="BTC/USDT:USDT",
            inject_ts_ms=1700000001000,
        )

    # We expect counts to be >1 because two calls occurred within a single process lifetime.
    assert "[HYBRID-METRICS]" in caplog.text
    assert "total_calls=2" in caplog.text
    assert "'none': 2" in caplog.text or '"none": 2' in caplog.text
