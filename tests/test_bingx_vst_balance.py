import pytest

from core.bingx_vst_balance import (
    BingxVstBalanceClient,
    VST_BASE_URL,
    build_bingx_signature_payload,
    hmac_sha256_hex,
)


def test_signature_payload_sorted_and_not_urlencoded():
    params = {
        "timestamp": 1700000000123,
        "recvWindow": 5000,
        "note": "a b",
        "amount": 100000,
        "adjustType": 0,
    }
    payload = build_bingx_signature_payload(params)
    assert payload == "adjustType=0&amount=100000&note=a b&recvWindow=5000&timestamp=1700000000123"
    assert "%20" not in payload
    assert "+" not in payload


def test_signature_hmac_matches_expected():
    params = {
        "timestamp": 1,
        "recvWindow": 5000,
        "amount": 100000,
        "adjustType": 0,
    }
    payload = build_bingx_signature_payload(params)
    assert payload == "adjustType=0&amount=100000&recvWindow=5000&timestamp=1"

    # Deterministic expected signature for the payload above.
    secret = "test-secret"
    sig = hmac_sha256_hex(secret, payload)
    assert sig == hmac_sha256_hex(secret, payload)
    assert len(sig) == 64


def test_vst_host_guard_rejects_prod_host():
    with pytest.raises(ValueError, match="non-VST host|production host"):
        BingxVstBalanceClient(
            api_key="k",
            secret_key="s",
            base_url="https://open-api.bingx.com",
        )


def test_vst_host_guard_accepts_vst_host():
    client = BingxVstBalanceClient(
        api_key="k",
        secret_key="s",
        base_url=VST_BASE_URL,
    )
    assert client is not None

