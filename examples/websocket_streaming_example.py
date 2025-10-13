#!/usr/bin/env python3
"""
Example: Real-Time Market Data Streaming with WebSocket
Demonstrates Phase 3.1 WebSocket infrastructure for live data streaming.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.websocket_manager import WebSocketManager, StreamDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_1_basic_ohlcv_streaming():
    """Example 1: Basic OHLCV streaming from single exchange."""
    logger.info("="*60)
    logger.info("Example 1: Basic OHLCV Streaming")
    logger.info("="*60)
    
    # Initialize manager
    manager = WebSocketManager()
    collector = StreamDataCollector(buffer_size=100)
    
    # Define symbols to stream
    symbols_per_exchange = {
        'kucoinfutures': ['BTC/USDT:USDT']
    }
    
    # Start streaming
    logger.info("Starting OHLCV stream for BTC/USDT on KuCoin Futures (1m timeframe)")
    logger.info("Will stream for 10 seconds...")
    
    try:
        tasks = await manager.stream_ohlcv(
            symbols_per_exchange,
            timeframe='1m',
            callback=collector.ohlcv_callback,
            max_iterations=3  # Stop after 3 updates
        )
        
        # Run streams
        await manager.run_streams(duration=10)
        
        # Display collected data
        logger.info("\n" + "="*60)
        logger.info("Collected Data Summary:")
        latest = collector.get_latest_ohlcv('kucoinfutures', 'BTC/USDT:USDT', '1m')
        if latest:
            logger.info(f"Latest BTC/USDT candle: {len(latest)} data points")
            if latest:
                logger.info(f"  Most recent: {latest[-1]}")
        else:
            logger.info("No data collected (WebSocket may need actual connection)")
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
    finally:
        await manager.close()
        logger.info("Example 1 complete\n")


async def example_2_multi_symbol_streaming():
    """Example 2: Stream multiple symbols simultaneously."""
    logger.info("="*60)
    logger.info("Example 2: Multi-Symbol Streaming")
    logger.info("="*60)
    
    # Initialize manager
    manager = WebSocketManager()
    collector = StreamDataCollector(buffer_size=50)
    
    # Define multiple symbols
    symbols_per_exchange = {
        'kucoinfutures': ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    }
    
    logger.info("Streaming BTC and ETH from KuCoin Futures")
    logger.info("Will collect 2 updates per symbol...")
    
    try:
        tasks = await manager.stream_ohlcv(
            symbols_per_exchange,
            timeframe='1m',
            callback=collector.ohlcv_callback,
            max_iterations=2
        )
        
        await manager.run_streams(duration=10)
        
        # Show status
        status = manager.get_stream_status()
        logger.info("\n" + "="*60)
        logger.info("Stream Status:")
        logger.info(f"  Total streams: {status['total_streams']}")
        logger.info(f"  Completed: {status['completed_streams']}")
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
    finally:
        await manager.close()
        logger.info("Example 2 complete\n")


async def example_3_multi_exchange_streaming():
    """Example 3: Stream from multiple exchanges simultaneously."""
    logger.info("="*60)
    logger.info("Example 3: Multi-Exchange Streaming")
    logger.info("="*60)
    
    # Initialize manager with multiple exchanges
    exchanges = {
        'kucoinfutures': None,  # Unauthenticated
        'bingx': None
    }
    manager = WebSocketManager(exchanges)
    collector = StreamDataCollector(buffer_size=50)
    
    # Define symbols per exchange
    symbols_per_exchange = {
        'kucoinfutures': ['BTC/USDT:USDT'],
        'bingx': ['BTC/USDT:USDT']
    }
    
    logger.info("Streaming BTC from both KuCoin and BingX")
    logger.info("Will collect 2 updates per exchange...")
    
    try:
        tasks = await manager.stream_ohlcv(
            symbols_per_exchange,
            timeframe='1m',
            callback=collector.ohlcv_callback,
            max_iterations=2
        )
        
        await manager.run_streams(duration=10)
        
        # Show status
        status = manager.get_stream_status()
        logger.info("\n" + "="*60)
        logger.info("Multi-Exchange Stream Status:")
        logger.info(f"  Exchanges: {', '.join(status['exchanges'])}")
        logger.info(f"  Total streams: {status['total_streams']}")
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
    finally:
        await manager.close()
        logger.info("Example 3 complete\n")


async def example_4_ticker_streaming():
    """Example 4: Real-time ticker streaming."""
    logger.info("="*60)
    logger.info("Example 4: Real-Time Ticker Streaming")
    logger.info("="*60)
    
    # Initialize manager
    manager = WebSocketManager()
    collector = StreamDataCollector(buffer_size=50)
    
    # Define symbols
    symbols_per_exchange = {
        'kucoinfutures': ['BTC/USDT:USDT']
    }
    
    logger.info("Streaming real-time ticker for BTC/USDT")
    logger.info("Will collect 3 ticker updates...")
    
    try:
        tasks = await manager.stream_tickers(
            symbols_per_exchange,
            callback=collector.ticker_callback,
            max_iterations=3
        )
        
        await manager.run_streams(duration=10)
        
        # Display latest ticker
        latest = collector.get_latest_ticker('kucoinfutures', 'BTC/USDT:USDT')
        logger.info("\n" + "="*60)
        if latest:
            logger.info("Latest Ticker Data:")
            logger.info(f"  Symbol: {latest.get('symbol')}")
            logger.info(f"  Last: {latest.get('last')}")
            logger.info(f"  Bid: {latest.get('bid')}")
            logger.info(f"  Ask: {latest.get('ask')}")
        else:
            logger.info("No ticker data collected")
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
    finally:
        await manager.close()
        logger.info("Example 4 complete\n")


async def main():
    """Run all examples."""
    logger.info("\n" + "="*60)
    logger.info("WebSocket Infrastructure Examples - Phase 3.1")
    logger.info("="*60 + "\n")
    
    logger.info("NOTE: These examples demonstrate the WebSocket API.")
    logger.info("      Actual data streaming requires live exchange connections.")
    logger.info("      Some examples may timeout if exchange doesn't respond.\n")
    
    examples = [
        ("Basic OHLCV Streaming", example_1_basic_ohlcv_streaming),
        ("Multi-Symbol Streaming", example_2_multi_symbol_streaming),
        ("Multi-Exchange Streaming", example_3_multi_exchange_streaming),
        ("Ticker Streaming", example_4_ticker_streaming),
    ]
    
    for i, (name, example_func) in enumerate(examples, 1):
        logger.info(f"\nRunning Example {i}: {name}")
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")
        
        if i < len(examples):
            logger.info("\nWaiting 2 seconds before next example...")
            await asyncio.sleep(2)
    
    logger.info("\n" + "="*60)
    logger.info("All examples completed!")
    logger.info("="*60)


if __name__ == '__main__':
    asyncio.run(main())
