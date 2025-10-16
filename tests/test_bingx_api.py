#!/usr/bin/env python3
"""
BingX API smoke tests - Public and private endpoints.
"""
import os
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.bingx_authenticator import BingXAuthenticator
import requests

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
