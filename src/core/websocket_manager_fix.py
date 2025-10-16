"""
WebSocket Manager with enhanced error recovery
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from .websocket_client_enhanced import EnhancedWebSocketClient

logger = logging.getLogger(__name__)

class RobustWebSocketManager:
    def __init__(self, exchanges: Dict[str, Any]):
        """
        Initialize WebSocket manager with robust error handling.
        
        Args:
            exchanges: Dictionary of exchange clients
        """
        self.exchanges = exchanges
        self.ws_clients = {}
        self.active_streams = {}
        self.stream_health = {}
        self.global_error_count = 0
        self.is_running = True
        
        # Initialize WebSocket clients
        for exchange_name in exchanges.keys():
            try:
                self.ws_clients[exchange_name] = EnhancedWebSocketClient(exchange_name)
                logger.info(f"✅ WebSocket client created for {exchange_name}")
            except Exception as e:
                logger.error(f"Failed to create WebSocket client for {exchange_name}: {e}")
    
    async def stream_ohlcv_with_recovery(self, symbol: str, exchange: str, 
                                        timeframe: str = '1m', max_iterations: int = None):
        """
        Stream OHLCV data with automatic recovery from parse_frame errors.
        """
        if exchange not in self.ws_clients:
            logger.error(f"No WebSocket client for {exchange}")
            return
        
        client = self.ws_clients[exchange]
        iteration = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        stream_key = f"{exchange}:{symbol}:{timeframe}"
        self.active_streams[stream_key] = {
            'status': 'starting',
            'iterations': 0,
            'errors': 0,
            'last_update': None
        }
        
        while self.is_running:
            try:
                # Check iteration limit
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Stream {stream_key} reached max iterations ({max_iterations})")
                    break
                
                iteration += 1
                
                # Fetch OHLCV data with error recovery
                ohlcv = await client.watch_ohlcv_safe(symbol, timeframe)
                
                if ohlcv:
                    # Success - update stream status
                    self.active_streams[stream_key].update({
                        'status': 'active',
                        'iterations': iteration,
                        'last_update': datetime.now(),
                        'errors': 0  # Reset error count on success
                    })
                    
                    consecutive_errors = 0
                    
                    # Process data (you can emit events here)
                    logger.debug(f"Stream {stream_key}: received {len(ohlcv)} candles")
                    
                else:
                    # No data received (but not an exception)
                    consecutive_errors += 1
                    self.active_streams[stream_key]['errors'] += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Stream {stream_key}: too many consecutive errors, stopping")
                        self.active_streams[stream_key]['status'] = 'failed'
                        break
                
                # Rate limiting
                await asyncio.sleep(1)  # Adjust as needed
                
            except asyncio.CancelledError:
                logger.info(f"Stream {stream_key} cancelled")
                break
                
            except Exception as e:
                consecutive_errors += 1
                self.global_error_count += 1
                
                logger.error(f"Stream {stream_key} error: {e}")
                
                self.active_streams[stream_key]['errors'] += 1
                self.active_streams[stream_key]['status'] = 'error'
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Stream {stream_key}: max errors reached, stopping")
                    self.active_streams[stream_key]['status'] = 'failed'
                    break
                
                # Exponential backoff
                wait_time = min(60, 2 ** consecutive_errors)
                logger.info(f"Stream {stream_key}: waiting {wait_time}s before retry")
                await asyncio.sleep(wait_time)
        
        # Cleanup
        self.active_streams[stream_key]['status'] = 'stopped'
        logger.info(f"Stream {stream_key} stopped after {iteration} iterations")
    
    async def start_all_streams(self, symbols_per_exchange: Dict[str, List[str]], 
                               timeframe: str = '1m', max_iterations: int = None):
        """
        Start WebSocket streams for multiple symbols across exchanges.
        """
        tasks = []
        
        for exchange, symbols in symbols_per_exchange.items():
            if exchange not in self.ws_clients:
                logger.warning(f"Skipping {exchange}: no WebSocket client")
                continue
            
            for symbol in symbols:
                task = asyncio.create_task(
                    self.stream_ohlcv_with_recovery(symbol, exchange, timeframe, max_iterations)
                )
                tasks.append(task)
                
                # Small delay between starting streams
                await asyncio.sleep(0.1)
        
        logger.info(f"Started {len(tasks)} WebSocket streams")
        return tasks
    
    async def monitor_health(self, interval: int = 60):
        """
        Monitor WebSocket health and auto-recover if needed.
        """
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                
                # Get health status from all clients
                health_report = {}
                critical_count = 0
                
                for exchange, client in self.ws_clients.items():
                    health = client.get_health_status()
                    health_report[exchange] = health
                    
                    if health['health'] == 'critical':
                        critical_count += 1
                        logger.warning(f"⚠️ {exchange} WebSocket is critical: {health}")
                
                # Check active streams
                active_count = sum(1 for s in self.active_streams.values() if s['status'] == 'active')
                failed_count = sum(1 for s in self.active_streams.values() if s['status'] == 'failed')
                
                logger.info(f"WebSocket Health: Active={active_count}, Failed={failed_count}, "
                          f"Critical={critical_count}, Global Errors={self.global_error_count}")
                
                # Auto-recovery logic
                if critical_count > len(self.ws_clients) / 2:
                    logger.error("🚨 Majority of WebSockets are critical, triggering recovery...")
                    await self._recover_all_clients()
                
                # Clean up old error history
                self._cleanup_stream_history()
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _recover_all_clients(self):
        """Recover all WebSocket clients"""
        logger.info("Starting global WebSocket recovery...")
        
        for exchange, client in self.ws_clients.items():
            try:
                await client._reconnect()
                await asyncio.sleep(2)  # Delay between reconnections
            except Exception as e:
                logger.error(f"Failed to recover {exchange}: {e}")
        
        logger.info("Global WebSocket recovery complete")
    
    def _cleanup_stream_history(self):
        """Clean up old stream entries"""
        cutoff = datetime.now() - timedelta(hours=1)
        
        for key in list(self.active_streams.keys()):
            stream = self.active_streams[key]
            if stream['status'] in ['stopped', 'failed']:
                last_update = stream.get('last_update')
                if last_update and last_update < cutoff:
                    del self.active_streams[key]
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get overall stream status"""
        active = sum(1 for s in self.active_streams.values() if s['status'] == 'active')
        failed = sum(1 for s in self.active_streams.values() if s['status'] == 'failed')
        error = sum(1 for s in self.active_streams.values() if s['status'] == 'error')
        
        # Get client health
        client_health = {}
        for exchange, client in self.ws_clients.items():
            client_health[exchange] = client.get_health_status()
        
        return {
            'running': self.is_running,
            'total_streams': len(self.active_streams),
            'active_streams': active,
            'failed_streams': failed,
            'error_streams': error,
            'global_errors': self.global_error_count,
            'client_health': client_health,
            'parse_frame_errors': sum(
                c['parse_frame_errors'] for c in client_health.values()
            )
        }
    
    async def shutdown(self):
        """Gracefully shutdown all WebSocket connections"""
        logger.info("Shutting down WebSocket manager...")
        self.is_running = False
        
        # Close all clients
        for exchange, client in self.ws_clients.items():
            try:
                await client.close()
            except Exception as e:
                logger.error(f"Error closing {exchange}: {e}")
        
        logger.info("WebSocket manager shutdown complete")
