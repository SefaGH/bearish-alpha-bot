import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from src.core.stream_data_collector import StreamDataCollector
from src.core.market_data_pipeline import MarketDataPipeline


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


class DummyCcxtClient:
    """Minimal async stub for ccxt_client.ohlcv returning a DataFrame."""

    def __init__(self, candles: List[List[int | float]]):
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(candles, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        self.df = df.set_index("timestamp")

    async def ohlcv(self, symbol: str, timeframe: str, limit: int, add_indicators: bool = False):
        return self.df.copy()


async def scenario_pivot():
    print("\n== Scenario 1: Pivot (Normal Flow) ==")
    collector = StreamDataCollector(buffer_size=10)
    exchange = "bingx"
    symbol = "BTC/USDT"
    timeframe = "1m"
    t0 = 0
    t1 = 60_000  # next minute

    # Step A
    await collector.ohlcv_callback(exchange, symbol, timeframe, [t0, 1, 1, 1, 1.1, 10])
    forming = collector.get_forming_ohlcv(exchange, symbol, timeframe)
    closed = collector.get_latest_ohlcv(exchange, symbol, timeframe)
    assert forming and forming[0] == t0, "Forming candle not set at Step A"
    assert closed is None, "Closed data should be empty at Step A"
    print("Step A passed: forming set, closed empty")

    # Step B
    await collector.ohlcv_callback(exchange, symbol, timeframe, [t0, 1, 1.2, 0.9, 1.2, 12])
    forming = collector.get_forming_ohlcv(exchange, symbol, timeframe)
    closed = collector.get_latest_ohlcv(exchange, symbol, timeframe)
    assert forming and forming[4] == 1.2, "Forming candle did not update at Step B"
    assert closed is None, "Closed data should remain empty at Step B"
    print("Step B passed: forming updated, closed still empty")

    # Step C (pivot)
    await collector.ohlcv_callback(exchange, symbol, timeframe, [t1, 2, 2.1, 1.9, 2.0, 8])
    forming = collector.get_forming_ohlcv(exchange, symbol, timeframe)
    closed = collector.get_latest_ohlcv(exchange, symbol, timeframe)
    assert forming and forming[0] == t1, "Forming candle not advanced at Step C"
    assert closed and len(closed) == 1 and closed[0][0] == t0, "Closed data did not commit previous candle at Step C"
    print("Step C passed: pivot committed previous candle and advanced forming")


async def scenario_gap_detection():
    print("\n== Scenario 3: Gap Detection ==")
    collector = StreamDataCollector(buffer_size=10)
    exchange = "bingx"
    symbol = "BTC/USDT"
    timeframe = "1m"
    t0 = 0
    t2 = 120_000  # skip one interval (no update at t=60_000 -> real gap)

    await collector.ohlcv_callback(exchange, symbol, timeframe, [t0, 1, 1, 1, 1, 5])
    await collector.ohlcv_callback(exchange, symbol, timeframe, [t2, 2, 2.1, 1.9, 2, 5])

    state = collector.get_state(exchange, symbol, timeframe)
    assert state["gap_count"] == 1, f"Expected gap_count=1, got {state['gap_count']}"
    print("Gap detection passed: gap_count incremented to 1")


async def scenario_startup_hygiene():
    print("\n== Scenario 2: Startup Hygiene (MarketDataPipeline) ==")
    now_ms = int(time.time() * 1000)
    candles = [
        [now_ms - 120_000, 1, 1, 1, 1, 5],
        [now_ms - 60_000, 2, 2, 2, 2, 5],
        [now_ms, 3, 3, 3, 3, 5],  # forming (should be dropped)
    ]
    dummy_client = DummyCcxtClient(candles)
    pipeline = MarketDataPipeline({"dummy": dummy_client}, websocket_manager=None)

    filtered = pipeline._filter_closed_dataframe(dummy_client.df, "1m", context="[TEST-STARTUP]")
    assert len(filtered) == 2, f"Expected 2 closed candles, got {len(filtered)}"
    assert int(filtered.index[-1].timestamp() * 1000) <= now_ms - 60_000, "Trailing forming candle was not dropped"
    print("Startup hygiene passed: forming candle dropped before use")


async def main():
    await scenario_pivot()
    await scenario_startup_hygiene()
    await scenario_gap_detection()
    print("\nAll scenarios passed.")


if __name__ == "__main__":
    asyncio.run(main())
