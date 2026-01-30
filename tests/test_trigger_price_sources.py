import sys
import os
from collections import deque
from datetime import datetime, timezone

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.market_data_pipeline import MarketDataPipeline
from core.ccxt_client import CcxtClient
from core.websocket_manager import StreamDataCollector
from unittest.mock import Mock


def _inject_ticker(collector, exchange, symbol, *, bid=None, ask=None, last=None, ts=None):
    exchange_bucket = collector.ticker_data.setdefault(exchange, {})
    normalized_symbol = collector._normalize_symbol(symbol)
    if normalized_symbol not in exchange_bucket:
        exchange_bucket[normalized_symbol] = deque(maxlen=collector.buffer_size)
    payload = {}
    if bid is not None:
        payload["bid"] = bid
    if ask is not None:
        payload["ask"] = ask
    if last is not None:
        payload["last"] = last
    exchange_bucket[normalized_symbol].append(
        {
            "timestamp": ts or datetime.now(timezone.utc),
            "data": payload,
        }
    )


@pytest.fixture
def mock_websocket_manager():
    collector = StreamDataCollector(buffer_size=1000)

    class MockWebSocketManager:
        def __init__(self):
            self.collector = collector
            self._data_collector = collector

        def is_collector_ready(self):
            return True

        def get_latest_data(self, symbol, timeframe, exchange=None):
            return None

    return MockWebSocketManager()


@pytest.fixture
def pipeline_with_mock(mock_websocket_manager):
    client = Mock(spec=CcxtClient)
    client.name = 'mock1'
    exchanges = {'mock1': client}
    return MarketDataPipeline(exchanges, websocket_manager=mock_websocket_manager)


def test_trigger_price_bid_ask_sources(pipeline_with_mock):
    collector = pipeline_with_mock.websocket_manager.collector
    symbol = 'BTC/USDT:USDT'
    _inject_ticker(collector, 'mock1', symbol, bid=99.0, ask=101.0)

    price, source, fallback = pipeline_with_mock.get_live_trigger_price(
        symbol,
        timeframe='1m',
        source='bid',
        exchange='mock1',
    )
    assert price == pytest.approx(99.0, rel=1e-6)
    assert source == 'bid'
    assert fallback == 'none'

    price, source, fallback = pipeline_with_mock.get_live_trigger_price(
        symbol,
        timeframe='1m',
        source='ask',
        exchange='mock1',
    )
    assert price == pytest.approx(101.0, rel=1e-6)
    assert source == 'ask'
    assert fallback == 'none'


def test_trigger_price_opposite_side_source(pipeline_with_mock):
    collector = pipeline_with_mock.websocket_manager.collector
    symbol = 'BTC/USDT:USDT'
    _inject_ticker(collector, 'mock1', symbol, bid=99.5, ask=100.5)

    price, source, _ = pipeline_with_mock.get_live_trigger_price(
        symbol,
        timeframe='1m',
        source='opposite_side',
        exchange='mock1',
        side='sell',
    )
    assert price == pytest.approx(100.5, rel=1e-6)
    assert source == 'ask'

    price, source, _ = pipeline_with_mock.get_live_trigger_price(
        symbol,
        timeframe='1m',
        source='opposite_side',
        exchange='mock1',
        side='buy',
    )
    assert price == pytest.approx(99.5, rel=1e-6)
    assert source == 'bid'
