#!/usr/bin/env python3
"""
Unit tests for BingX WebSocket response parsing.

Tests the actual response formats captured from real BingX WebSocket data.
"""
import os
import sys
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.bingx_websocket import BingXWebSocket


class TestBingXWebSocketParsing:
    """Test BingX WebSocket message parsing with real response formats."""
    
    @pytest.fixture
    def ws_client(self):
        """Create a BingX WebSocket client for testing."""
        return BingXWebSocket(futures=True)
    
    @pytest.mark.asyncio
    async def test_parse_kline_response(self, ws_client):
        """
        Test parsing of real BingX kline response.
        
        This is the CRITICAL test - kline data is always an ARRAY, not a dict.
        The bug was: code tried to do kline_data.get('t') on a list.
        """
        # Real BingX kline response format (captured from actual WebSocket)
        real_response = {
            "code": 0,
            "dataType": "BTC-USDT@kline_1m",
            "s": "BTC-USDT",
            "data": [  # ALWAYS an array!
                {
                    "c": "110267.6",
                    "o": "110298.6",
                    "h": "110298.6",
                    "l": "110265.1",
                    "v": "2.0741",
                    "T": 1761327420000
                }
            ]
        }
        
        # Mock callback to capture parsed data
        kline_callback = AsyncMock()
        ws_client.on_kline(kline_callback)
        
        # Process the message
        await ws_client._process_message(real_response)
        
        # Verify callback was called
        assert kline_callback.called, "Kline callback should be called"
        
        # Verify correct arguments
        call_args = kline_callback.call_args[0]
        symbol = call_args[0]
        timeframe = call_args[1]
        klines = call_args[2]
        
        # Check symbol conversion
        assert symbol == "BTC/USDT:USDT", f"Expected BTC/USDT:USDT, got {symbol}"
        
        # Check timeframe
        assert timeframe == "1m", f"Expected 1m, got {timeframe}"
        
        # Check kline data
        assert len(klines) == 1, f"Expected 1 kline, got {len(klines)}"
        kline = klines[0]
        
        # Verify kline format: [timestamp, open, high, low, close, volume]
        assert kline[0] == 1761327420000, "Timestamp mismatch"
        assert kline[1] == 110298.6, "Open price mismatch"
        assert kline[2] == 110298.6, "High price mismatch"
        assert kline[3] == 110265.1, "Low price mismatch"
        assert kline[4] == 110267.6, "Close price mismatch"
        assert kline[5] == 2.0741, "Volume mismatch"
        
        # Verify data is stored
        stored_klines = ws_client.get_klines("BTC/USDT:USDT", "1m")
        assert stored_klines is not None, "Klines should be stored"
        assert len(stored_klines) >= 1, "At least 1 kline should be stored"
    
    @pytest.mark.asyncio
    async def test_parse_multiple_klines(self, ws_client):
        """Test parsing when BingX sends multiple klines in one message."""
        response = {
            "code": 0,
            "dataType": "ETH-USDT@kline_5m",
            "s": "ETH-USDT",
            "data": [
                {
                    "c": "3892.76",
                    "o": "3894.15",
                    "h": "3894.15",
                    "l": "3892.53",
                    "v": "176.52",
                    "T": 1761327420000
                },
                {
                    "c": "3893.50",
                    "o": "3892.76",
                    "h": "3895.00",
                    "l": "3890.00",
                    "v": "200.00",
                    "T": 1761327720000
                }
            ]
        }
        
        kline_callback = AsyncMock()
        ws_client.on_kline(kline_callback)
        
        await ws_client._process_message(response)
        
        assert kline_callback.called
        call_args = kline_callback.call_args[0]
        klines = call_args[2]
        
        # Should have 2 klines
        assert len(klines) == 2, f"Expected 2 klines, got {len(klines)}"
        
        # Verify first kline
        assert klines[0][4] == 3892.76, "First kline close price mismatch"
        
        # Verify second kline
        assert klines[1][4] == 3893.50, "Second kline close price mismatch"
    
    @pytest.mark.asyncio
    async def test_parse_ticker_response(self, ws_client):
        """Test parsing of real BingX ticker response."""
        real_response = {
            "code": 0,
            "dataType": "BTC-USDT@ticker",
            "data": {
                "e": "24hTicker",
                "E": 1761327444754,
                "s": "BTC-USDT",
                "p": "-436.0",
                "P": "-0.39",
                "c": "110267.8",
                "L": "0.0006",
                "h": "112080.0",
                "l": "109283.7",
                "v": "15204.7267",
                "q": "171854.69",
                "o": "110703.8",
                "O": 1761327293627,
                "C": 1761327444478,
                "A": "110267.8",  # Ask price
                "a": "2.5786",    # Ask volume
                "B": "110267.6",  # Bid price
                "b": "5.4305"     # Bid volume
            }
        }
        
        ticker_callback = AsyncMock()
        ws_client.on_ticker(ticker_callback)
        
        await ws_client._process_message(real_response)
        
        assert ticker_callback.called
        call_args = ticker_callback.call_args[0]
        symbol = call_args[0]
        ticker = call_args[1]
        
        # Check symbol
        assert symbol == "BTC/USDT:USDT"
        
        # Check ticker fields (mapped from real response)
        assert ticker['last'] == 110267.8, "Last price mismatch"
        assert ticker['bid'] == 110267.6, "Bid price should come from 'B' field"
        assert ticker['ask'] == 110267.8, "Ask price should come from 'A' field"
        assert ticker['high'] == 112080.0, "High price mismatch"
        assert ticker['low'] == 109283.7, "Low price mismatch"
        assert ticker['volume'] == 15204.7267, "Volume mismatch"
        assert ticker['open'] == 110703.8, "Open price mismatch"
        assert ticker['change'] == -436.0, "Price change mismatch"
        assert ticker['percentage'] == -0.39, "Percentage change mismatch"
        assert ticker['timestamp'] == 1761327444754, "Timestamp mismatch"
    
    @pytest.mark.asyncio
    async def test_parse_depth_response(self, ws_client):
        """Test parsing of real BingX depth/orderbook response."""
        real_response = {
            "code": 0,
            "dataType": "BTC-USDT@depth20@500ms",
            "ts": 1761327444754,
            "data": {
                "bids": [
                    ["110267.6", "5.4305"],
                    ["110267.5", "0.0434"],
                    ["110267.2", "0.0818"]
                ],
                "asks": [
                    ["110275.2", "0.2302"],
                    ["110274.1", "0.2171"],
                    ["110273.0", "0.1786"]
                ]
            }
        }
        
        orderbook_callback = AsyncMock()
        ws_client.on_orderbook(orderbook_callback)
        
        await ws_client._process_message(real_response)
        
        assert orderbook_callback.called
        call_args = orderbook_callback.call_args[0]
        symbol = call_args[0]
        orderbook = call_args[1]
        
        # Check symbol
        assert symbol == "BTC/USDT:USDT"
        
        # Check orderbook structure
        assert 'bids' in orderbook
        assert 'asks' in orderbook
        
        # Check bids (should be converted to float)
        assert len(orderbook['bids']) == 3
        assert orderbook['bids'][0] == [110267.6, 5.4305]
        
        # Check asks
        assert len(orderbook['asks']) == 3
        assert orderbook['asks'][0] == [110275.2, 0.2302]
    
    @pytest.mark.asyncio
    async def test_subscription_confirmation(self, ws_client):
        """Test handling of subscription confirmation."""
        confirmation = {
            "id": "test_ticker_btc",
            "code": 0,
            "msg": "",
            "dataType": "",
            "data": None
        }
        
        # Add to pending subscriptions
        ws_client.pending_subscriptions["test_ticker_btc"] = {
            "id": "test_ticker_btc",
            "reqType": "sub",
            "dataType": "BTC-USDT@ticker"
        }
        
        await ws_client._process_message(confirmation)
        
        # Should move from pending to confirmed
        assert "test_ticker_btc" not in ws_client.pending_subscriptions
        assert "test_ticker_btc" in ws_client.subscriptions
    
    @pytest.mark.asyncio
    async def test_error_response(self, ws_client):
        """Test handling of error responses."""
        error_response = {
            "id": "test_invalid",
            "code": 100001,
            "msg": "Invalid dataType",
            "dataType": "",
            "data": None
        }
        
        ws_client.pending_subscriptions["test_invalid"] = {
            "id": "test_invalid",
            "reqType": "sub",
            "dataType": "INVALID@ticker"
        }
        
        await ws_client._process_message(error_response)
        
        # Should remove from pending subscriptions on error
        assert "test_invalid" not in ws_client.pending_subscriptions
        assert "test_invalid" not in ws_client.subscriptions
    
    def test_symbol_conversion(self, ws_client):
        """Test symbol format conversion between CCXT and BingX."""
        # CCXT to BingX
        assert ws_client._convert_symbol_to_bingx("BTC/USDT:USDT") == "BTC-USDT"
        assert ws_client._convert_symbol_to_bingx("BTC/USDT") == "BTC-USDT"
        assert ws_client._convert_symbol_to_bingx("ETH/USDT:USDT") == "ETH-USDT"
        
        # BingX to CCXT (futures)
        assert ws_client._convert_symbol_from_bingx("BTC-USDT") == "BTC/USDT:USDT"
        assert ws_client._convert_symbol_from_bingx("ETH-USDT") == "ETH/USDT:USDT"
    
    def test_timeframe_conversion(self, ws_client):
        """Test timeframe conversion."""
        assert ws_client._convert_timeframe("1m") == "1m"
        assert ws_client._convert_timeframe("5m") == "5m"
        assert ws_client._convert_timeframe("1h") == "1h"
        assert ws_client._convert_timeframe("1d") == "1d"
        
        assert ws_client._convert_timeframe_from_bingx("1m") == "1m"
        assert ws_client._convert_timeframe_from_bingx("5m") == "5m"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
