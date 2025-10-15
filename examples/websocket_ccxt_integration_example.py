#!/usr/bin/env python3
"""
Example: WebSocket Manager Integration with Phase 1 Multi-Exchange Framework
Demonstrates how to use WebSocketManager with CcxtClient instances from build_clients_from_env()
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.multi_exchange import build_clients_from_env
from core.websocket_manager import WebSocketManager, StreamDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_1_ccxt_client_integration():
    """Example 1: Initialize WebSocketManager with CcxtClient instances."""
    logger.info("="*60)
    logger.info("Example 1: CcxtClient Integration")
    logger.info("="*60)
    
    try:
        # Build clients from environment (Phase 1 pattern)
        # Note: This requires EXCHANGES environment variable to be set
        logger.info("Building exchange clients from environment...")
        
        # For demo purposes, we'll create clients manually
        # In production: clients = build_clients_from_env()
        from core.ccxt_client import CcxtClient
        clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None),
            'bingx': CcxtClient('bingx', None)
        }
        logger.info(f"Created {len(clients)} exchange clients: {list(clients.keys())}")
        
        # Initialize WebSocketManager with CcxtClient instances
        config = {
            'reconnect_delay': 5,
            'max_retries': 3
        }
        ws_manager = WebSocketManager(clients, config=config)
        
        # Verify initialization
        status = ws_manager.get_stream_status()
        logger.info(f"WebSocket Manager Status:")
        logger.info(f"  Exchanges: {status['exchanges']}")
        logger.info(f"  Running: {status['running']}")
        logger.info(f"  Total streams: {status['total_streams']}")
        
        await ws_manager.close()
        logger.info("✓ Example 1 complete\n")
        
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
        logger.info("Note: This example requires EXCHANGES env var or manual client creation")


async def example_2_new_subscription_api():
    """Example 2: Using the new subscription API methods."""
    logger.info("="*60)
    logger.info("Example 2: New Subscription API")
    logger.info("="*60)
    
    # Initialize with default exchanges
    ws_manager = WebSocketManager()
    
    # Register callbacks using the new API
    ticker_updates = []
    
    async def on_ticker(exchange, symbol, ticker):
        ticker_updates.append({
            'exchange': exchange,
            'symbol': symbol,
            'price': ticker.get('last'),
            'timestamp': ticker.get('timestamp')
        })
        logger.info(f"Ticker update: {exchange} {symbol} @ ${ticker.get('last')}")
    
    # Register callback
    ws_manager.on_ticker_update(on_ticker)
    logger.info("✓ Registered ticker callback")
    
    # Note: To actually subscribe and start streams, you would call:
    # await ws_manager.subscribe_tickers(['BTC/USDT:USDT', 'ETH/USDT:USDT'])
    # await ws_manager.run_streams(duration=10)
    
    await ws_manager.close()
    logger.info("✓ Example 2 complete\n")


async def example_3_start_streams_api():
    """Example 3: Using start_streams with subscription dict."""
    logger.info("="*60)
    logger.info("Example 3: Start Streams API")
    logger.info("="*60)
    
    ws_manager = WebSocketManager()
    
    # Define subscriptions
    subscriptions = {
        'tickers': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
        'ohlcv': ['BTC/USDT:USDT'],
        'timeframe': '1m'
    }
    
    logger.info("Subscription configuration:")
    logger.info(f"  Tickers: {subscriptions['tickers']}")
    logger.info(f"  OHLCV: {subscriptions['ohlcv']} @ {subscriptions['timeframe']}")
    
    # Note: To actually start streams:
    # tasks = await ws_manager.start_streams(subscriptions)
    # await ws_manager.run_streams(duration=10)
    # await ws_manager.shutdown()
    
    logger.info("✓ Subscription configuration ready")
    await ws_manager.close()
    logger.info("✓ Example 3 complete\n")


async def example_4_method_chaining():
    """Example 4: Method chaining for callback registration."""
    logger.info("="*60)
    logger.info("Example 4: Method Chaining")
    logger.info("="*60)
    
    ws_manager = WebSocketManager()
    
    async def ticker_handler(exchange, symbol, ticker):
        logger.info(f"Ticker: {symbol} @ {ticker.get('last')}")
    
    async def orderbook_handler(exchange, symbol, orderbook):
        logger.info(f"Orderbook update: {symbol}")
    
    # Chain multiple callback registrations
    (ws_manager
        .on_ticker_update(ticker_handler)
        .on_orderbook_update(orderbook_handler))
    
    logger.info("✓ Registered callbacks using method chaining")
    logger.info(f"  Ticker callbacks: {len(ws_manager.callbacks['ticker'])}")
    logger.info(f"  Orderbook callbacks: {len(ws_manager.callbacks['orderbook'])}")
    
    await ws_manager.close()
    logger.info("✓ Example 4 complete\n")


async def example_5_production_pattern():
    """Example 5: Production integration pattern with MarketDataPipeline."""
    logger.info("="*60)
    logger.info("Example 5: Production Integration Pattern")
    logger.info("="*60)
    
    try:
        # This simulates the production pattern from the problem statement
        from core.ccxt_client import CcxtClient
        
        # Phase 1: Build clients from environment
        # In production: clients = build_clients_from_env()
        clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None),
            'bingx': CcxtClient('bingx', None)
        }
        logger.info(f"✓ Phase 1: Exchange clients initialized: {list(clients.keys())}")
        
        # Phase 2: Initialize market intelligence (would be done in production)
        # from core.market_regime import MarketRegimeAnalyzer
        # analyzer = MarketRegimeAnalyzer()
        logger.info("✓ Phase 2: Market intelligence ready (simulated)")
        
        # Phase 3.1: WebSocket streaming with CcxtClient integration
        config = {
            'reconnect_delay': 5,
            'max_retries': 3,
            'timeout': 30
        }
        ws_manager = WebSocketManager(clients, config=config)
        logger.info("✓ Phase 3.1: WebSocket manager initialized with CcxtClient instances")
        
        # Data collector for analysis
        collector = StreamDataCollector(buffer_size=100)
        logger.info("✓ Phase 3.1: Data collector initialized")
        
        # Register callbacks for market intelligence
        ws_manager.on_ticker_update(collector.ticker_callback)
        logger.info("✓ Callbacks registered for market analysis")
        
        # Show configuration
        logger.info("\nWebSocket Configuration:")
        for key, value in ws_manager.config.items():
            logger.info(f"  {key}: {value}")
        
        # Clean up
        await ws_manager.shutdown()
        logger.info("✓ Example 5 complete - Production pattern verified\n")
        
    except Exception as e:
        logger.error(f"Example 5 failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all examples."""
    logger.info("\n" + "="*60)
    logger.info("WebSocket Manager + CcxtClient Integration Examples")
    logger.info("Demonstrates Phase 2.1 Integration Requirements")
    logger.info("="*60 + "\n")
    
    examples = [
        ("CcxtClient Integration", example_1_ccxt_client_integration),
        ("New Subscription API", example_2_new_subscription_api),
        ("Start Streams API", example_3_start_streams_api),
        ("Method Chaining", example_4_method_chaining),
        ("Production Pattern", example_5_production_pattern),
    ]
    
    for i, (name, example_func) in enumerate(examples, 1):
        logger.info(f"\nRunning Example {i}: {name}")
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")
        
        if i < len(examples):
            await asyncio.sleep(1)
    
    logger.info("\n" + "="*60)
    logger.info("All examples completed!")
    logger.info("="*60)


if __name__ == '__main__':
    asyncio.run(main())
