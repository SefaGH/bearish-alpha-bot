from src.core.ccxt_client import CcxtClient


def test_cancel_all_bingx_open_orders_uses_swap_v2_endpoint():
    client = CcxtClient.__new__(CcxtClient)
    client.name = "bingx"
    client._get_bingx_native_symbol = lambda s: "BTC-USDT" if s else s

    captured = {}

    def _fake_request(endpoint, params=None, method="GET"):
        captured["endpoint"] = endpoint
        captured["params"] = params or {}
        captured["method"] = method
        return {"code": 0}

    client._make_authenticated_bingx_request = _fake_request

    response = client.cancel_all_bingx_open_orders(symbol="BTC/USDT:USDT", order_type="limit")
    assert response["code"] == 0
    assert captured["endpoint"] == "/openApi/swap/v2/trade/allOpenOrders"
    assert captured["method"] == "DELETE"
    assert captured["params"]["symbol"] == "BTC-USDT"
    assert captured["params"]["type"] == "LIMIT"


def test_close_all_bingx_positions_uses_swap_v2_endpoint():
    client = CcxtClient.__new__(CcxtClient)
    client.name = "bingx"
    client._get_bingx_native_symbol = lambda s: "BTC-USDT" if s else s

    captured = {}

    def _fake_request(endpoint, params=None, method="GET"):
        captured["endpoint"] = endpoint
        captured["params"] = params or {}
        captured["method"] = method
        return {"code": 0}

    client._make_authenticated_bingx_request = _fake_request

    response = client.close_all_bingx_positions(symbol="BTC/USDT:USDT")
    assert response["code"] == 0
    assert captured["endpoint"] == "/openApi/swap/v2/trade/closeAllPositions"
    assert captured["method"] == "POST"
    assert captured["params"]["symbol"] == "BTC-USDT"
