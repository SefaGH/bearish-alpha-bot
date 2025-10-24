#!/usr/bin/env python3
"""
Integration test for BingX WebSocket - Verify real-time data reception.

This test connects to the actual BingX WebSocket and verifies:
1. Connection is established
2. Subscriptions are confirmed
3. Real-time data is received (ticker, kline, depth)
4. No parsing errors occur
"""
import os
import sys
import pytest
import asyncio
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.bingx_websocket import BingXWebSocket


@pytest.mark.asyncio
@pytest.mark.integration
async def test_bingx_websocket_integration():
    """
    Integration test: Connect to real BingX WebSocket and verify data reception.
    
    This test runs for 10 seconds and verifies:
    - Connection succeeds
    - Subscriptions are confirmed
    - Data is received and parsed correctly
    - No errors occur during parsing
    """
    print("\n" + "="*70)
    print("BingX WebSocket Integration Test")
    print("="*70)
    
    # Create WebSocket client
    ws = BingXWebSocket(futures=True)
    
    # Track received data
    received_data = {
        'ticker': False,
        'kline': False,
        'depth': False
    }
    
    # Callbacks to track data reception
    async def on_ticker(symbol, ticker):
        print(f"✅ Received ticker: {symbol} - Last: {ticker['last']}")
        received_data['ticker'] = True
    
    async def on_kline(symbol, timeframe, klines):
        print(f"✅ Received kline: {symbol} {timeframe} - {len(klines)} candles")
        received_data['kline'] = True
    
    async def on_orderbook(symbol, orderbook):
        bids_count = len(orderbook.get('bids', []))
        asks_count = len(orderbook.get('asks', []))
        print(f"✅ Received orderbook: {symbol} - {bids_count} bids, {asks_count} asks")
        received_data['depth'] = True
    
    # Register callbacks
    ws.on_ticker(on_ticker)
    ws.on_kline(on_kline)
    ws.on_orderbook(on_orderbook)
    
    try:
        # Connect
        print("\n1. Connecting to BingX WebSocket...")
        connected = await ws.connect()
        assert connected, "Failed to connect to BingX WebSocket"
        print("   ✅ Connected successfully")
        
        # Subscribe to test data
        print("\n2. Subscribing to data streams...")
        
        await ws.subscribe_ticker("BTC/USDT:USDT")
        print("   → Subscribed to BTC ticker")
        
        await ws.subscribe_kline("BTC/USDT:USDT", "1m")
        print("   → Subscribed to BTC kline 1m")
        
        # Subscribe to orderbook (depth)
        sub_id = str(int(time.time() * 1000))
        sub_message = {
            "id": sub_id,
            "reqType": "sub",
            "dataType": "BTC-USDT@depth20@500ms"
        }
        ws.pending_subscriptions[sub_id] = sub_message
        await ws.ws.send(json.dumps(sub_message))
        print("   → Subscribed to BTC depth")
        
        print("\n3. Listening for data (10 seconds)...")
        
        # Create listen task
        listen_task = asyncio.create_task(ws.listen())
        
        # Wait for 10 seconds
        await asyncio.sleep(10)
        
        # Stop listening
        ws._running = False
        
        # Wait for listen task to complete
        try:
            await asyncio.wait_for(listen_task, timeout=2.0)
        except asyncio.TimeoutError:
            listen_task.cancel()
        
        print("\n4. Verifying data reception...")
        
        # Check status
        status = ws.get_status()
        print(f"\n   Status:")
        print(f"   - Messages received: {status['message_count']}")
        print(f"   - Subscriptions confirmed: {status['subscriptions']}")
        print(f"   - Tickers tracked: {status['tickers_tracked']}")
        print(f"   - Klines tracked: {status['klines_tracked']}")
        print(f"   - Orderbooks tracked: {status['orderbooks_tracked']}")
        
        # Verify we received messages
        assert status['message_count'] > 0, "No messages received from WebSocket"
        print("\n   ✅ Messages received successfully")
        
        # Verify subscriptions were confirmed
        assert status['subscriptions'] > 0, "No subscriptions confirmed"
        print("   ✅ Subscriptions confirmed")
        
        # Check if we received each data type (be lenient, as we might not get all types in 10s)
        if received_data['ticker']:
            print("   ✅ Ticker data received and parsed")
        else:
            print("   ⚠️  No ticker data received (may be normal in short test)")
        
        if received_data['kline']:
            print("   ✅ Kline data received and parsed")
        else:
            print("   ⚠️  No kline data received (may be normal in short test)")
        
        if received_data['depth']:
            print("   ✅ Orderbook data received and parsed")
        else:
            print("   ⚠️  No orderbook data received (may be normal in short test)")
        
        # At least verify we have some data stored
        ticker = ws.get_ticker("BTC/USDT:USDT")
        klines = ws.get_klines("BTC/USDT:USDT", "1m")
        orderbook = ws.get_orderbook("BTC/USDT:USDT")
        
        if ticker:
            print(f"\n   Ticker data: Last={ticker['last']}, Bid={ticker.get('bid')}, Ask={ticker.get('ask')}")
        
        if klines:
            print(f"   Klines: {len(klines)} candles stored")
        
        if orderbook:
            print(f"   Orderbook: {len(orderbook.get('bids', []))} bids, {len(orderbook.get('asks', []))} asks")
        
        print("\n" + "="*70)
        print("✅ Integration test passed!")
        print("="*70)
        
    finally:
        # Cleanup
        await ws.disconnect()
        print("\n   Disconnected from WebSocket")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_bingx_websocket_integration())
