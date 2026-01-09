from core.ccxt_client import CcxtClient


def test_extract_bingx_usdt_available_from_list_shape():
    resp = {
        "code": 0,
        "data": [
            {"asset": "BTC", "availableBalance": "0.01"},
            {"asset": "USDT", "availableBalance": "123.45"},
        ],
    }
    assert CcxtClient.extract_bingx_usdt_available(resp) == 123.45


def test_extract_bingx_usdt_available_from_dict_shape():
    resp = {"code": 0, "data": {"asset": "USDT", "available": "999"}}
    assert CcxtClient.extract_bingx_usdt_available(resp) == 999.0


def test_extract_bingx_usdt_available_returns_none_when_missing():
    assert CcxtClient.extract_bingx_usdt_available({"code": 0, "data": []}) is None
    assert CcxtClient.extract_bingx_usdt_available("not-a-dict") is None

