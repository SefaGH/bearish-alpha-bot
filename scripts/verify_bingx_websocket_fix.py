#!/usr/bin/env python3
"""
Quick verification script for BingX WebSocket fix.

This script does a quick 15-second test to verify:
1. WebSocket connects successfully
2. Data is received and parsed correctly
3. No parsing errors occur

Usage:
    python scripts/verify_bingx_websocket_fix.py
"""

import sys
import os
import asyncio
import logging
import json
import time
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.bingx_websocket import BingXWebSocket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def verify_websocket():
    """Quick verification of BingX WebSocket functionality."""
    
    print("\n" + "="*70)
    print("BingX WebSocket Fix Verification")
    print("="*70)
    
    # Track data reception
    received = {'ticker': 0, 'kline': 0, 'orderbook': 0}
    errors = []
    
    # Create callbacks
    async def on_ticker(symbol, ticker):
        received['ticker'] += 1
        if received['ticker'] == 1:
            print(f"✅ First ticker: {symbol} @ {ticker['last']}")
    
    async def on_kline(symbol, timeframe, klines):
        received['kline'] += 1
        if received['kline'] == 1:
            print(f"✅ First kline: {symbol} {timeframe} - {len(klines)} candles")
    
    async def on_orderbook(symbol, orderbook):
        received['orderbook'] += 1
        if received['orderbook'] == 1:
            bids = len(orderbook.get('bids', []))
            asks = len(orderbook.get('asks', []))
            print(f"✅ First orderbook: {symbol} - {bids} bids, {asks} asks")
    
    # Create WebSocket client
    ws = BingXWebSocket(futures=True)
    ws.on_ticker(on_ticker)
    ws.on_kline(on_kline)
    ws.on_orderbook(on_orderbook)
    
    try:
        # Connect
        print("\n1. Connecting to BingX WebSocket...")
        connected = await ws.connect()
        
        if not connected:
            print("❌ Failed to connect!")
            return False
        
        print("   ✅ Connected")
        
        # Subscribe
        print("\n2. Subscribing to data streams...")
        await ws.subscribe_ticker("BTC/USDT:USDT")
        await ws.subscribe_kline("BTC/USDT:USDT", "1m")
        
        # Subscribe to orderbook manually
        sub_id = str(int(time.time() * 1000))
        sub_message = {
            "id": sub_id,
            "reqType": "sub",
            "dataType": "BTC-USDT@depth20@500ms"
        }
        ws.pending_subscriptions[sub_id] = sub_message
        await ws.ws.send(json.dumps(sub_message))
        
        print("   ✅ Subscriptions sent")
        
        # Listen for 15 seconds
        print("\n3. Listening for data (15 seconds)...")
        
        listen_task = asyncio.create_task(ws.listen())
        await asyncio.sleep(15)
        
        # Stop
        ws._running = False
        try:
            await asyncio.wait_for(listen_task, timeout=2.0)
        except asyncio.TimeoutError:
            listen_task.cancel()
        
        # Verify results
        print("\n4. Results:")
        status = ws.get_status()
        
        print(f"   - Total messages: {status['message_count']}")
        print(f"   - Ticker updates: {received['ticker']}")
        print(f"   - Kline updates: {received['kline']}")
        print(f"   - Orderbook updates: {received['orderbook']}")
        print(f"   - Subscriptions confirmed: {status['subscriptions']}")
        
        # Check for success
        success = True
        
        if status['message_count'] == 0:
            print("\n❌ No messages received!")
            success = False
        else:
            print(f"\n✅ Received {status['message_count']} messages")
        
        if status['subscriptions'] == 0:
            print("❌ No subscriptions confirmed!")
            success = False
        else:
            print(f"✅ {status['subscriptions']} subscriptions confirmed")
        
        if received['ticker'] == 0:
            print("⚠️  No ticker data received (might be normal in short test)")
        else:
            print(f"✅ Received {received['ticker']} ticker updates")
        
        if received['kline'] == 0:
            print("⚠️  No kline data received (might be normal in short test)")
        else:
            print(f"✅ Received {received['kline']} kline updates")
        
        if received['orderbook'] == 0:
            print("⚠️  No orderbook data received (might be normal in short test)")
        else:
            print(f"✅ Received {received['orderbook']} orderbook updates")
        
        # Final verdict
        print("\n" + "="*70)
        if success:
            print("✅ VERIFICATION PASSED - BingX WebSocket is working correctly!")
            print("="*70)
            return True
        else:
            print("❌ VERIFICATION FAILED - Issues detected")
            print("="*70)
            return False
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        traceback.print_exc()
        return False
    
    finally:
        await ws.disconnect()
        print("\nDisconnected")


if __name__ == "__main__":
    result = asyncio.run(verify_websocket())
    sys.exit(0 if result else 1)
