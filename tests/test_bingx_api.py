#!/usr/bin/env python3
"""
BingX API smoke tests - Public and private endpoints.

This test module attempts to load BingX secrets from multiple locations if
they are not already present in the environment:

1. Environment variables BINGX_KEY and BINGX_SECRET (existing behavior).
2. A JSON file specified by environment variable BINGX_SECRETS_FILE or
   SECRETS_FILE, or a default .bingx_secrets.json file at the repository root.
   The JSON should contain keys "BINGX_KEY" and "BINGX_SECRET".
3. A dotenv-like file (e.g. ".env") specified by the same env vars or found at
   repo root, with lines like BINGX_KEY=... and BINGX_SECRET=...

This lets CI or local runs provide secrets via a file if they are not present
as env vars.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core.bingx_authenticator import BingXAuthenticator

BASE_URL = "https://open-api.bingx.com"


def _load_bingx_secrets_from_file(path: str) -> bool:
    """Attempt to load secrets from a JSON or dotenv-like file."""
    if not os.path.exists(path):
        return False

    try:
        key: str | None
        secret: str | None
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
            key = data.get("BINGX_KEY") or data.get("bingx_key")
            secret = data.get("BINGX_SECRET") or data.get("bingx_secret")
        else:
            data: dict[str, str] = {}
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip().strip('"').strip("'")
            key = data.get("BINGX_KEY") or data.get("bingx_key")
            secret = data.get("BINGX_SECRET") or data.get("bingx_secret")

        if key:
            os.environ.setdefault("BINGX_KEY", key)
        if secret:
            os.environ.setdefault("BINGX_SECRET", secret)
        return bool(os.getenv("BINGX_KEY"))
    except Exception:
        return False


def _ensure_bingx_secrets_loaded() -> bool:
    if os.getenv("BINGX_KEY"):
        return True

    candidates = [
        os.getenv("BINGX_SECRETS_FILE"),
        os.getenv("SECRETS_FILE"),
        os.path.join(os.path.dirname(__file__), "..", ".bingx_secrets.json"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        os.path.join(os.path.dirname(__file__), "..", "secrets.json"),
    ]

    base_dir = os.path.dirname(__file__)
    for candidate in candidates:
        if not candidate:
            continue
        if not os.path.isabs(candidate):
            candidate = os.path.normpath(os.path.join(base_dir, "..", candidate))
        if _load_bingx_secrets_from_file(candidate):
            return True
    return False


_ensure_bingx_secrets_loaded()

pytestmark = pytest.mark.skipif(
    not os.getenv("BINGX_KEY"),
    reason="BingX credentials not set",
)


@pytest.fixture
def auth() -> BingXAuthenticator:
    return BingXAuthenticator(
        os.getenv("BINGX_KEY", ""),
        os.getenv("BINGX_SECRET", ""),
    )


def test_public_price() -> None:
    response = requests.get(
        f"{BASE_URL}/openApi/swap/v2/quote/price",
        params={"symbol": "BTC-USDT"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("code") == 0
    assert "lastPrice" in data.get("data", {})


def test_public_depth() -> None:
    response = requests.get(
        f"{BASE_URL}/openApi/swap/v2/quote/depth",
        params={"symbol": "BTC-USDT", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("code") == 0
    assert "bids" in data.get("data", {})
    assert "asks" in data.get("data", {})


def test_private_balance(auth: BingXAuthenticator) -> None:
    request_data = auth.prepare_authenticated_request({})
    response = requests.get(
        f"{BASE_URL}/openApi/swap/v2/user/balance",
        params=request_data["params"],
        headers=request_data["headers"],
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("code") == 0


def test_private_positions(auth: BingXAuthenticator) -> None:
    request_data = auth.prepare_authenticated_request({})
    response = requests.get(
        f"{BASE_URL}/openApi/swap/v2/user/positions",
        params=request_data["params"],
        headers=request_data["headers"],
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("code") == 0
