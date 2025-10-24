#!/usr/bin/env python3
"""
BingX WebSocket Response Capture Script
=========================================

This script connects to BingX WebSocket, subscribes to different data types,
and logs the raw responses to help understand the exact response formats.

Purpose:
- Capture real BingX WebSocket responses
- Document exact JSON structures
- Help fix parsing issues in src/core/bingx_websocket.py

Usage:
    python scripts/test_bingx_ws_responses.py [--duration SECONDS]
"""

import asyncio
import json
import gzip
import io
import time
import argparse
from datetime import datetime
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/bingx_ws_test.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# BingX WebSocket URL
WS_URL = "wss://open-api-swap.bingx.com/swap-market"

# Test subscriptions
TEST_SUBSCRIPTIONS = [
    {"id": "test_ticker_btc", "reqType": "sub", "dataType": "BTC-USDT@ticker"},
    {"id": "test_kline_btc", "reqType": "sub", "dataType": "BTC-USDT@kline_1m"},
    {"id": "test_depth_btc", "reqType": "sub", "dataType": "BTC-USDT@depth20@500ms"},
    {"id": "test_ticker_eth", "reqType": "sub", "dataType": "ETH-USDT@ticker"},
    {"id": "test_kline_eth", "reqType": "sub", "dataType": "ETH-USDT@kline_1m"},
]

# Storage for captured responses
captured_responses = {
    "ticker": [],
    "kline": [],
    "depth": [],
    "subscription": [],
    "ping_pong": [],
    "other": []
}

# Statistics
stats = {
    "total_messages": 0,
    "ticker_count": 0,
    "kline_count": 0,
    "depth_count": 0,
    "subscription_count": 0,
    "ping_pong_count": 0,
    "errors": 0,
    "start_time": None,
    "end_time": None
}


def decompress_message(message):
    """Decompress GZIP message from BingX WebSocket."""
    try:
        if isinstance(message, bytes):
            # Use GzipFile with BytesIO (recommended method)
            compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode='rb')
            decompressed_data = compressed_data.read()
            return decompressed_data.decode('utf-8')
        return message
    except Exception as e:
        logger.error(f"Error decompressing message: {e}")
        return None


def categorize_message(message_str, data):
    """Categorize message by type."""
    try:
        # Check for subscription response
        if isinstance(data, dict) and "id" in data and "code" in data:
            return "subscription"
        
        # Check for data type
        if isinstance(data, dict) and "dataType" in data:
            data_type = data["dataType"]
            if "@ticker" in data_type:
                return "ticker"
            elif "@kline" in data_type:
                return "kline"
            elif "@depth" in data_type:
                return "depth"
        
        # Ping/Pong
        if message_str.strip() in ["Ping", "Pong"]:
            return "ping_pong"
        
        return "other"
    except Exception as e:
        logger.error(f"Error categorizing message: {e}")
        return "other"


def save_response(category, data, message_str):
    """Save response to appropriate category."""
    timestamp = datetime.now().isoformat()
    
    response_entry = {
        "timestamp": timestamp,
        "category": category,
        "raw_message": message_str[:500] if len(message_str) > 500 else message_str,
        "parsed_data": data if isinstance(data, dict) else None,
        "message_length": len(message_str)
    }
    
    captured_responses[category].append(response_entry)
    
    # Update stats
    stats[f"{category}_count"] = stats.get(f"{category}_count", 0) + 1


async def test_bingx_websocket(duration=60):
    """
    Connect to BingX WebSocket and capture responses.
    
    Args:
        duration: How long to run the test (seconds)
    """
    import websockets
    
    logger.info(f"Starting BingX WebSocket test for {duration} seconds...")
    logger.info(f"Connecting to: {WS_URL}")
    
    stats["start_time"] = datetime.now().isoformat()
    
    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=10,
            max_size=10 * 1024 * 1024
        ) as ws:
            logger.info("✅ Connected to BingX WebSocket")
            
            # Send subscriptions
            logger.info(f"Sending {len(TEST_SUBSCRIPTIONS)} subscriptions...")
            for sub in TEST_SUBSCRIPTIONS:
                await ws.send(json.dumps(sub))
                logger.info(f"  → Subscribed to: {sub['dataType']}")
                await asyncio.sleep(0.1)
            
            logger.info("\n" + "="*70)
            logger.info("Listening for messages...")
            logger.info("="*70 + "\n")
            
            # Listen for messages
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < duration:
                try:
                    # Set timeout to check duration periodically
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    stats["total_messages"] += 1
                    message_count += 1
                    
                    # Decompress message
                    message_str = decompress_message(message)
                    
                    if not message_str:
                        continue
                    
                    # Handle Ping
                    if message_str.strip() == "Ping":
                        await ws.send("Pong")
                        logger.debug("Received Ping → Sent Pong")
                        save_response("ping_pong", None, message_str)
                        continue
                    
                    # Skip empty messages
                    if not message_str.strip():
                        continue
                    
                    # Try to parse as JSON
                    try:
                        data = json.loads(message_str)
                        
                        # Categorize and save
                        category = categorize_message(message_str, data)
                        save_response(category, data, message_str)
                        
                        # Log interesting messages
                        if category == "subscription":
                            code = data.get("code", -1)
                            msg = data.get("msg", "")
                            sub_id = data.get("id", "")
                            if code == 0:
                                logger.info(f"✅ Subscription confirmed: {sub_id}")
                            else:
                                logger.error(f"❌ Subscription failed: {sub_id} - {msg}")
                        
                        elif category == "ticker":
                            data_type = data.get("dataType", "")
                            ticker_data = data.get("data", {})
                            logger.info(f"📊 TICKER [{data_type}]: {json.dumps(ticker_data, indent=2)[:200]}")
                        
                        elif category == "kline":
                            data_type = data.get("dataType", "")
                            kline_data = data.get("data", {})
                            logger.info(f"📈 KLINE [{data_type}]: {json.dumps(kline_data, indent=2)[:200]}")
                        
                        elif category == "depth":
                            data_type = data.get("dataType", "")
                            depth_data = data.get("data", {})
                            logger.info(f"📉 DEPTH [{data_type}]: bids={len(depth_data.get('bids', []))}, asks={len(depth_data.get('asks', []))}")
                        
                        else:
                            logger.info(f"🔍 OTHER: {message_str[:200]}")
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON message: {message_str[:100]}")
                        save_response("other", None, message_str)
                
                except asyncio.TimeoutError:
                    # No message received in timeout period, continue
                    continue
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    stats["errors"] += 1
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Test completed after {duration} seconds")
            logger.info(f"Total messages received: {message_count}")
            logger.info(f"{'='*70}\n")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        stats["errors"] += 1
    
    finally:
        stats["end_time"] = datetime.now().isoformat()


def save_results():
    """Save captured responses to JSON file."""
    output_file = Path("logs/bingx_ws_responses.json")
    output_file.parent.mkdir(exist_ok=True)
    
    results = {
        "test_info": {
            "ws_url": WS_URL,
            "subscriptions": TEST_SUBSCRIPTIONS,
            "test_duration": stats.get("end_time", "N/A")
        },
        "statistics": stats,
        "responses": captured_responses
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    logger.info(f"Total messages: {stats['total_messages']}")
    logger.info(f"  - Ticker messages: {stats.get('ticker_count', 0)}")
    logger.info(f"  - Kline messages: {stats.get('kline_count', 0)}")
    logger.info(f"  - Depth messages: {stats.get('depth_count', 0)}")
    logger.info(f"  - Subscription responses: {stats.get('subscription_count', 0)}")
    logger.info(f"  - Ping/Pong: {stats.get('ping_pong_count', 0)}")
    logger.info(f"  - Other messages: {len(captured_responses.get('other', []))}")
    logger.info(f"  - Errors: {stats.get('errors', 0)}")
    logger.info("="*70)
    
    # Print sample responses
    logger.info("\nSAMPLE RESPONSES:")
    for category in ["subscription", "ticker", "kline", "depth"]:
        responses = captured_responses.get(category, [])
        if responses:
            logger.info(f"\n--- {category.upper()} (Sample) ---")
            sample = responses[0]
            if sample.get("parsed_data"):
                logger.info(json.dumps(sample["parsed_data"], indent=2))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test BingX WebSocket and capture responses")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds (default: 60)")
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("BingX WebSocket Response Capture Test")
    logger.info("="*70)
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the test
    asyncio.run(test_bingx_websocket(args.duration))
    
    # Save results
    save_results()
    
    logger.info("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()
