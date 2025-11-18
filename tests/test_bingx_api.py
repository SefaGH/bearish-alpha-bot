#!/usr/bin/env python3
"""
BingX API smoke tests - Public and private endpoints.

This test module will attempt to load BingX secrets from multiple locations if they
are not already present in the environment:

1. Environment variables BINGX_KEY and BINGX_SECRET (existing behavior).
2. A JSON file specified by environment variable BINGX_SECRETS_FILE or SECRETS_FILE,
   or a default .bingx_secrets.json file at the repository root. The JSON should
   contain keys "BINGX_KEY" and "BINGX_SECRET".
3. A dotenv-like file (e.g. ".env") specified by the same env vars or found at repo root,
   with lines like BINGX_KEY=... and BINGX_SECRET=...

This lets CI or local runs provide secrets via a file if they are not present as env vars.
"""
import os
import pytest
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.bingx_authenticator import BingXAuthenticator
import requests


def _load_bingx_secrets_from_file(path):
    """Try to load secrets from the given path. Supports JSON and dotenv-like files."""
    if not os.path.exists(path):
        return False
    try:
        if path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            key = data.get('BINGX_KEY') or data.get('bingx_key')
            secret = data.get('BINGX_SECRET') or data.get('bingx_secret')
        else:
            # dotenv-like parsing
            data = {}
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    data[k.strip()] = v.strip().strip('"').strip("'")
            key = data.get('BINGX_KEY') or data.get('bingx_key')
            secret = data.get('BINGX_SECRET') or data.get('bingx_secret')
        if key:
            os.environ.setdefault('BINGX_KEY', key)
        if secret:
            os.environ.setdefault('BINGX_SECRET', secret)
        return bool(os.getenv('BINGX_KEY'))
    except Exception:
        return False


def _ensure_bingx_secrets_loaded():
    """
    Ensure BINGX_KEY is available for the pytest skip check. This function attempts:
    - env vars BINGX_KEY/BINGX_SECRET (already present)
    - JSON file pointed to by BINGX_SECRETS_FILE or SECRETS_FILE
    - default files at repo root: .bingx_secrets.json or .env
    """
    if os.getenv('BINGX_KEY'):
        return True

    # Candidate file paths (env var values may be relative or absolute)
    candidates = [
        os.getenv('BINGX_SECRETS_FILE'),
        os.getenv('SECRETS_FILE'),
        os.path.join(os.path.dirname(__file__), '..', '.bingx_secrets.json'),
        os.path.join(os.path.dirname(__file__), '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', 'secrets.json'),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        # If candidate is relative, normalize it relative to repo/tests dir
        if not os.path.isabs(candidate):
            candidate = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', candidate))
        if _load_bingx_secrets_from_file(candidate):
            return True

    return False


# Try to load secrets before evaluating skip condition
_ensure_bingx_secrets_loaded()

# Skip all tests if no BingX credentials
pytestmark = pytest.mark.skipif(
    not os.getenv('BINGX_KEY'),
    reason="BingX credentials not set"
)

BASE_URL = "https://open-api.bingx.com"

@pytest.fixture
def auth():
    """BingX authenticator fixture."""
    return BingXAuthenticator(
        os.getenv('BINGX_KEY', ''),
        os.getenv('BINGX_SECRET', '')
    )

def test_public_price():
    """Test public price endpoint."""
    response = requests.get(
        f"{BASE_URL}/openApi/swap/v2/quote/price",
        params={'symbol': 'BTC-USDT'}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get('code') == 0
    assert 'lastPrice' in data.get('data', {})

def test_public_depth():
    """Test public orderbook endpoint."""
    response = requests.get(
        f"{BASE_URL}/openApi/swap/v2/quote/depth",
        params={'symbol': 'BTC-USDT', 'limit': 10}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get('code') == 0
    assert 'bids' in data.get('data', {})
    assert 'asks' in data.get('data', {})

def test_private_balance(auth):
    """Test private balance endpoint."""
    request_data = auth.prepare_authenticated_request({})
    
    response = requests.get(
        f"{BASE_URL}/openApi/swap/v2/user/balance",
        params=request_data['params'],
        headers=request_data['headers']
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data.get('code') == 0

def test_private_positions(auth):
    """Test private positions endpoint."""
    request_data = auth.prepare_authenticated_request({})
    
    response = requests.get(
        f"{BASE_URL}/openApi/swap/v2/user/positions",
        params=request_data['params'],
        headers=request_data['headers']
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data.get('code') == 0
