#!/usr/bin/env python3
"""
Test WebSocket recovery from parse_frame errors
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.multi_exchange import build_clients_from_env
from core.websocket_manager_fix import RobustWebSocketManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_websocket_recovery():
    """Test WebSocket with parse_frame recovery"""
    
    # Test configuration
    test_symbols = {
        'bingx': ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    }
    test_duration = 300  # 5 minutes
    
    logger.info("=" * 60)
    logger.info("WebSocket Parse Frame Recovery Test")
    logger.info("=" * 60)
    
    # Build clients
    clients = build_clients_from_env()
    
    # Create robust WebSocket manager
    ws_manager = RobustWebSocketManager(clients)
    
    # Start health monitor
    monitor_task = asyncio.create_task(ws_manager.monitor_health(interval=30))
    
    # Start streams
    stream_tasks = await ws_manager.start_all_streams(
        symbols_per_exchange=test_symbols,
        timeframe='1m',
        max_iterations=test_duration  # Run for 5 minutes
    )
    
    logger.info(f"Test running for {test_duration} seconds...")
    
    # Periodic status reports
    for i in range(test_duration // 30):
        await asyncio.sleep(30)
        
        status = ws_manager.get_stream_status()
        logger.info("=" * 60)
        logger.info(f"Status Report #{i+1}")
        logger.info(f"Active Streams: {status['active_streams']}/{status['total_streams']}")
        logger.info(f"Failed Streams: {status['failed_streams']}")
        logger.info(f"Parse Frame Errors: {status['parse_frame_errors']}")
        logger.info(f"Global Errors: {status['global_errors']}")
        
        # Check client health
        for exchange, health in status['client_health'].items():
            logger.info(f"{exchange}: {health['health']} "
                       f"(errors: {health['total_errors']}, "
                       f"parse_frame: {health['parse_frame_errors']})")
    
    # Wait for streams to complete
    await asyncio.gather(*stream_tasks, return_exceptions=True)
    
    # Cancel monitor
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    
    # Shutdown
    await ws_manager.shutdown()
    
    # Final report
    final_status = ws_manager.get_stream_status()
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Errors: {final_status['global_errors']}")
    logger.info(f"Parse Frame Errors: {final_status['parse_frame_errors']}")
    logger.info(f"Failed Streams: {final_status['failed_streams']}")
    
    # Determine success
    success = final_status['failed_streams'] == 0
    
    if success:
        logger.info("✅ TEST PASSED - WebSocket recovered from all errors")
    else:
        logger.error("❌ TEST FAILED - Some streams failed permanently")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(test_websocket_recovery())
    sys.exit(0 if success else 1)
