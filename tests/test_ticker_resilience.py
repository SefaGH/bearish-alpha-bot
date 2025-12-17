import logging
import ccxt
import pytest

from src.core.ccxt_client import CcxtClient


class _FakeExchange:
    """Mock CCXT exchange to simulate transient ticker failures then success."""

    def __init__(self, params):
        self.params = params
        self.session = None
        self.fetch_calls = 0

    def fetch_ticker(self, symbol: str):
        self.fetch_calls += 1
        # First call raises, second succeeds (matches 2-attempt budget)
        if self.fetch_calls == 1:
            raise ccxt.RequestTimeout("simulated timeout")
        return {"symbol": symbol, "last": 123.45}


def test_ticker_retry_and_cache(monkeypatch, caplog):
    """Ensure ticker retries once on timeout, then caches the successful response."""
    monkeypatch.setenv("CCXT_TIMEOUT_MS", "5000")
    monkeypatch.setenv("TICKER_CACHE_TTL_S", "2")
    monkeypatch.setenv("TICKER_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("TICKER_RETRY_BASE_DELAY_S", "0")

    # Register fake exchange on ccxt module
    monkeypatch.setattr(ccxt, "mockx", _FakeExchange)

    client = CcxtClient("mockx", {"apiKey": "k", "secret": "s"})

    with caplog.at_level(logging.WARNING):
        ticker = client.ticker("BTC/USDT:USDT")

    assert ticker["last"] == 123.45
    assert client.ex.fetch_calls == 2, "Should retry once then succeed"

    # Retry warning should be emitted
    retry_logs = [
        line for line in caplog.text.splitlines()
        if "[CCXT-TICKER-RETRY/mockx/BTC/USDT:USDT]" in line
    ]
    assert retry_logs, "Expected retry warning log entry"

    caplog.clear()
    # Second call within TTL should hit cache (no additional fetch)
    ticker2 = client.ticker("BTC/USDT:USDT")
    assert ticker2["last"] == 123.45
    assert client.ex.fetch_calls == 2, "Cache hit should avoid extra fetch"
