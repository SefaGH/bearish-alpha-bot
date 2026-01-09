import pytest

from core.ccxt_client import CcxtClient


def test_bingx_minimal_market_precision_supports_amount_to_precision(monkeypatch):
    # Keep this test fully offline: we rely on the injected minimal market structure
    # (no exchange market load/network calls).
    monkeypatch.setenv("BINGX_ENV", "prod")

    client = CcxtClient("bingx", {"apiKey": "k", "secret": "s"})
    client.set_required_symbols(["BTC/USDT:USDT"])

    markets = client.load_markets(reload=False)
    assert "BTC/USDT:USDT" in markets

    market = client.ex.market("BTC/USDT:USDT")
    assert float(market["precision"]["amount"]) < 1.0

    # Regression: if precision.amount is incorrectly set to an int (e.g. 8) while CCXT is in tick-size mode,
    # ccxt.amount_to_precision() truncates to '0' and raises InvalidOrder.
    amount_str = client.ex.amount_to_precision("BTC/USDT:USDT", 0.00137845)
    assert amount_str != "0"

