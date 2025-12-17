"""
Smart Order Management System.
Advanced order management with execution optimization.
"""

import asyncio
import logging
import os
import time
import ccxt
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum

if TYPE_CHECKING:
    from .market_data_pipeline import MarketDataPipeline

logger = logging.getLogger(__name__)
_REST_DEBUG_ENABLED = os.getenv("BINGX_REST_DEBUG", "").strip().lower() in {"1", "true", "yes", "debug"}


def _log_rest_debug(context: str, exc: Exception) -> None:
    """Emit detailed REST diagnostics when debug flag is enabled."""
    if not _REST_DEBUG_ENABLED:
        return

    debug_payload: Dict[str, Any] = {"context": context}
    status = getattr(exc, "http_status", None) or getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status:
        debug_payload["status"] = status

    url = getattr(exc, "url", None) or getattr(exc, "request_url", None)
    if url:
        debug_payload["url"] = url

    headers = getattr(exc, "response_headers", None) or getattr(exc, "headers", None)
    if headers:
        if isinstance(headers, dict):
            limited = list(headers.items())[:5]
            debug_payload["headers"] = {k: v for k, v in limited}
        else:
            debug_payload["headers"] = str(headers)

    raw_body = getattr(exc, "response", None) or getattr(exc, "body", None)
    if raw_body:
        if hasattr(raw_body, "text"):
            raw_body = getattr(raw_body, "text")
        if isinstance(raw_body, bytes):
            try:
                raw_body = raw_body.decode("utf-8", errors="ignore")
            except Exception:
                raw_body = repr(raw_body[:256])
        if isinstance(raw_body, str):
            debug_payload["response"] = raw_body[:512]
        else:
            debug_payload["response_type"] = str(type(raw_body))

    logger.error("REST DEBUG :: %s", debug_payload)


def _is_transient_ccxt_error(exc: Exception) -> bool:
    """Classify transient CCXT/network errors safely across ccxt versions."""
    try:
        from ccxt.base import errors as ccxt_errors
    except Exception:
        ccxt_errors = None

    transient_names = [
        "RequestTimeout",
        "NetworkError",
        "ExchangeNotAvailable",
        "DDoSProtection",
        "BadGateway",
        "ServiceUnavailable",
    ]

    transient_types = tuple(
        err_type for err_type in (
            getattr(ccxt_errors, name, None) for name in transient_names
        ) if err_type
    ) if ccxt_errors else ()

    if transient_types and isinstance(exc, transient_types):
        return True

    fallback_types = tuple(
        err_type for err_type in (
            getattr(ccxt, "RequestTimeout", None),
            getattr(ccxt, "NetworkError", None),
        ) if err_type
    )
    return isinstance(exc, fallback_types) if fallback_types else False


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'


class SmartOrderManager:
    """Advanced order management with execution optimization."""
    
    def __init__(self, market_data_pipeline: 'MarketDataPipeline', risk_manager=None, exchange_clients: Optional[Dict] = None):
        """
        Initialize smart order manager.
        
        Args:
            market_data_pipeline: The central market data provider.
            risk_manager: RiskManager instance.
            exchange_clients: Dictionary of exchange client instances.
        """
        # FIX: Initialize logger for instance use
        self.logger = logging.getLogger(__name__)
        
        # Core dependencies
        self.market_data_pipeline = market_data_pipeline
        self.risk_manager = risk_manager
        self.exchange_clients = exchange_clients if exchange_clients is not None else {}
        
        # Order management
        self.active_orders = {}  # order_id -> order_data
        self.order_queue = asyncio.Queue()
        self.order_history = []
        
        # Execution algorithms
        self.execution_algorithms = {
            'market': self._market_order_execution,
            'limit': self._limit_order_execution,
            'iceberg': self._iceberg_order_execution,
            'twap': self._twap_order_execution
        }
        
        # Execution statistics
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0,
            'avg_execution_time': 0.0,
            'total_slippage': 0.0,
        }
        
        self.logger.info("SmartOrderManager initialized successfully")

    def set_dependencies(self, risk_manager: Any, exchange_clients: Dict):
        """
        Set dependencies after initialization. This allows for flexible setup.
        
        DEPRECATED: market_data_pipeline should now be provided in __init__.
        This method is kept for backward compatibility and will be removed in v2.0.0.
        
        Deprecation Timeline:
        - Deprecated: v1.1.0 (current)
        - Removal: v2.0.0 (planned)
        """
        import warnings
        warnings.warn(
            "set_dependencies() is deprecated. Pass market_data_pipeline to __init__ instead. "
            "This method will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2
        )
        self.risk_manager = risk_manager
        self.exchange_clients = exchange_clients
        logger.info(f"OrderManager dependencies set. {len(exchange_clients)} exchange client(s) registered.")
    
    async def place_order(self, order_request: Dict, execution_algo: str = 'limit', 
                         exchange_clients: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Place order with specified execution algorithm.
        
        Args:
            order_request: Order request dictionary with symbol, side, amount, etc.
            execution_algo: Execution algorithm to use ('market', 'limit', 'iceberg', 'twap')
            exchange_clients: Optional live exchange clients (overrides self.exchange_clients if provided)
            
        Returns:
            Order execution result
        """
        try:
            start_time = time.time()
            
            logger.info(f"Placing order: {order_request.get('symbol')} {order_request.get('side')} "
                       f"{order_request.get('amount')} using {execution_algo} algorithm")
            logger.debug(f"🎪 [ORDER-MGR] Signal received: {order_request}")
            
            # CRITICAL: Use injected exchange_clients if provided (e.g., during shutdown)
            clients_to_use = exchange_clients if exchange_clients is not None else self.exchange_clients
            
            # Validate order request with active clients
            validation = self._validate_order_request(order_request, clients_to_use)
            logger.debug(f"🎪 [ORDER-MGR] Pre-execution checks: {validation}")
            
            if not validation['valid']:
                logger.error(f"Order validation failed: {validation['reason']}")
                logger.debug(f"🎪 [ORDER-MGR] Execution result: REJECTED - {validation['reason']}")
                self.execution_stats['failed_orders'] += 1
                return {
                    'success': False,
                    'reason': validation['reason'],
                    'order_id': None
                }
            
            # Select execution algorithm
            exec_func = self.execution_algorithms.get(execution_algo, self._limit_order_execution)
            
            logger.debug(f"🎪 [ORDER-MGR] Order parameters: algo={execution_algo}, symbol={order_request.get('symbol')}, "
                        f"side={order_request.get('side')}, amount={order_request.get('amount')}")
            
            # Execute order with active clients (with retry for transient failures)
            max_retries = 3
            result = None
            
            for attempt in range(max_retries):
                result = await exec_func(order_request, clients_to_use)
                
                if result.get('success'):
                    break
                
                # Check if we should retry based on reason
                reason = result.get('reason', '').lower()
                is_transient = any(x in reason for x in ['timeout', 'connection', 'rate limit', '500', '502', '503', '504', 'network', 'reset'])
                
                if not is_transient or attempt == max_retries - 1:
                    break
                
                wait_time = (2 ** attempt) * 0.5
                logger.warning(f"Transient failure in order execution: {reason}. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            
            logger.debug(f"🎪 [ORDER-MGR] Execution result: {'SUCCESS' if result and result.get('success') else 'FAILED'}")
            
            # Update statistics
            self.execution_stats['total_orders'] += 1
            if result.get('success'):
                self.execution_stats['successful_orders'] += 1
                
                # Calculate execution time
                execution_time = time.time() - start_time
                current_avg = self.execution_stats['avg_execution_time']
                total = self.execution_stats['successful_orders']
                self.execution_stats['avg_execution_time'] = (
                    (current_avg * (total - 1) + execution_time) / total
                )
                
                logger.debug(f"🎪 [ORDER-MGR] Post-execution state: order_id={result.get('order_id')}, "
                            f"executed_price={result.get('executed_price')}, execution_time={execution_time:.3f}s")
                
                # Store in history
                self.order_history.append({
                    **result,
                    'execution_time': execution_time,
                    'timestamp': datetime.now(timezone.utc)
                })
            else:
                self.execution_stats['failed_orders'] += 1
                logger.debug(f"🎪 [ORDER-MGR] Post-execution state: FAILED - {result.get('reason')}")
            
            return result
            
        except Exception as e:
            _log_rest_debug("place_order", e)
            logger.error(f"Error placing order: {e}")
            self.execution_stats['failed_orders'] += 1
            return {
                'success': False,
                'reason': f"Execution error: {str(e)}",
                'order_id': None
            }
    
    async def cancel_order(self, order_id: str, exchange: str) -> Dict[str, Any]:
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID to cancel
            exchange: Exchange where order was placed
            
        Returns:
            Cancellation result
        """
        try:
            if order_id not in self.active_orders:
                return {'success': False, 'reason': 'Order not found'}
            
            order = self.active_orders[order_id]
            client = self.exchange_clients.get(exchange)
            
            if not client:
                return {'success': False, 'reason': f'Exchange client not found: {exchange}'}
            
            # Cancel order on exchange
            logger.info(f"Cancelling order {order_id} on {exchange}")
            
            # Note: Actual cancellation would call client.cancel_order()
            # For now, we mark it as cancelled
            order['status'] = OrderStatus.CANCELLED.value
            order['cancelled_at'] = datetime.now(timezone.utc)
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            self.execution_stats['cancelled_orders'] += 1
            
            logger.info(f"Order {order_id} cancelled successfully")
            return {'success': True, 'order_id': order_id}
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status information or None if not found
        """
        return self.active_orders.get(order_id)
    
    def _validate_order_request(self, order_request: Dict, clients_to_use: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate order request format and parameters.
        
        Args:
            order_request: Order request dictionary
            clients_to_use: Exchange clients to validate against (defaults to self.exchange_clients)
            
        Returns:
            Validation result
        """
        required_fields = ['symbol', 'side', 'amount', 'exchange']
        
        for field in required_fields:
            if field not in order_request:
                return {'valid': False, 'reason': f'Missing required field: {field}'}
        
        # Validate side
        if order_request['side'] not in ['buy', 'sell', 'long', 'short']:
            return {'valid': False, 'reason': f"Invalid side: {order_request['side']}"}
        
        # Validate amount
        if order_request['amount'] <= 0:
            return {'valid': False, 'reason': 'Amount must be positive'}
        
        # Validate exchange (use clients_to_use if provided, else self.exchange_clients)
        clients = clients_to_use if clients_to_use is not None else self.exchange_clients
        if order_request['exchange'] not in clients:
            return {'valid': False, 'reason': f"Exchange not available: {order_request['exchange']}"}
        
        return {'valid': True, 'reason': ''}
    
    async def _market_order_execution(self, order_request: Dict, clients_to_use: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute market order with slippage control.
        
        Args:
            order_request: Order request dictionary
            clients_to_use: Exchange clients to use (defaults to self.exchange_clients)
            
        Returns:
            Execution result
        """
        try:
            symbol = order_request['symbol']
            side = order_request['side']
            amount = order_request['amount']
            exchange = order_request['exchange']
            
            # CRITICAL: Use clients_to_use if provided
            clients = clients_to_use if clients_to_use is not None else self.exchange_clients
            client = clients[exchange]
            
            # Get current market price for slippage monitoring
            ticker = client.ticker(symbol)
            expected_price = float(ticker.get('last', 0))
            
            logger.info(f"Executing market order: {symbol} {side} {amount} @ ~{expected_price}")
            
            # Generate order ID
            order_id = f"order_{int(time.time() * 1000)}"
            
            # Create order record
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'type': 'market',
                'exchange': exchange,
                'expected_price': expected_price,
                'status': OrderStatus.SUBMITTED.value,
                'created_at': datetime.now(timezone.utc),
                'fills': []
            }
            
            # In real implementation, would call:
            # result = client.create_order(symbol, side=side, type_='market', amount=amount)
            
            # Simulate execution
            execution_price = expected_price * (1.0001 if side == 'buy' else 0.9999)
            
            order['status'] = OrderStatus.FILLED.value
            order['filled_amount'] = amount
            order['avg_fill_price'] = execution_price
            order['filled_at'] = datetime.now(timezone.utc)
            
            # Calculate slippage
            slippage = abs(execution_price - expected_price) / expected_price
            order['slippage'] = slippage
            
            self.execution_stats['total_slippage'] += slippage
            
            # Store order
            self.active_orders[order_id] = order
            
            logger.info(f"Market order filled: {order_id} @ {execution_price:.4f} (slippage: {slippage*100:.3f}%)")
            
            return {
                'success': True,
                'order_id': order_id,
                'filled_amount': amount,
                'avg_price': execution_price,
                'slippage': slippage,
                'order': order
            }
            
        except Exception as e:
            context = f"MARKET/{order_request.get('exchange')}/{order_request.get('symbol')}"
            _log_rest_debug(context, e)
            logger.error(f"Market order execution failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'order_id': None
            }
    
    async def _limit_order_execution(self, order_request: Dict, clients_to_use: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute limit order with smart pricing.
        (UPDATED: Now uses MarketDataPipeline for market metadata retrieval)
        """
        symbol = order_request.get('symbol')
        side = order_request.get('side')
        amount = order_request.get('amount')
        exchange = order_request.get('exchange')
        log_prefix = f"[ORDER-EXEC/{exchange}/{symbol}]"

        try:
            clients = clients_to_use if clients_to_use is not None else self.exchange_clients
            client = clients[exchange]
            
            # Get market metadata from pipeline (proper architecture)
            try:
                market = await self.market_data_pipeline.get_market_metadata(symbol, exchange)
            except ValueError as e:
                # Sanitize error message to avoid exposing internal details
                error_msg = f"Market metadata unavailable for {symbol} on {exchange}"
                self.logger.error(f"🛡️  {log_prefix} REJECTED (MarketMetadata): {e}")
                return {'success': False, 'reason': f"REJECT:MARKET_METADATA - {error_msg}", 'order_id': None}
            
            # Get current price from exchange
            try:
                ticker = client.ticker(symbol)
            except Exception as e:
                if _is_transient_ccxt_error(e):
                    self.logger.warning(f"[WARN] {log_prefix} Ticker fetch failed (transient): {e}")
                    return {'success': False, 'reason': str(e), 'order_id': None}
                raise

            market_price = float(ticker.get('last', 0))
            
            # Calculate optimal limit price
            price_offset = 0.001
            limit_price = market_price * (1 - price_offset) if side in ['buy', 'long'] else market_price * (1 + price_offset)
            
            # --- 🔥 YENİ EKLENEN TELEMETRİ VE ÖN KONTROL ADIMI 🔥 ---
            notional_value = amount * limit_price
            min_notional = market.get('limits', {}).get('cost', {}).get('min', 0)

            self.logger.info(f"➡️  {log_prefix} Processing LIMIT order. Side: {side}, Amount: {amount}, Limit Price: ${limit_price:.4f}, Notional: ${notional_value:.2f}")

            # 1. Borsa Limit Kontrolü (minNotional)
            if min_notional and notional_value < min_notional:
                reason = f"Order notional value (${notional_value:.2f}) is below exchange minimum (${min_notional:.2f})."
                self.logger.error(f"🛡️  {log_prefix} REJECTED (MinNotional): {reason}")
                return {'success': False, 'reason': f"REJECT:MIN_NOTIONAL - {reason}", 'order_id': None}
            
            # Diğer borsa-spesifik doğrulamalar buraya eklenebilir (örn: min_amount, price precision vs.)
            # try:
            #     client.some_pre_check_function(symbol, amount, limit_price)
            # except Exception as e:
            #     self.logger.error(f"🛡️  {log_prefix} REJECTED (Pre-check): {e}")
            #     return {'success': False, 'reason': f"REJECT:EXCHANGE_VALIDATION - {e}", 'order_id': None}
            # --- TELEMETRİ VE ÖN KONTROL SONU ---

            logger.info(f"✅ {log_prefix} Pre-flight checks passed. Submitting to exchange...")
            
            # Emir gönderme simülasyonu
            order_id = f"order_{int(time.time() * 1000)}"
            order = {
                'order_id': order_id, 'symbol': symbol, 'side': side, 'amount': amount,
                'type': 'limit', 'limit_price': limit_price, 'exchange': exchange,
                'status': OrderStatus.SUBMITTED.value, 'created_at': datetime.now(timezone.utc), 'fills': []
            }
            
            # Simülasyon: Emirin dolduğunu varsay
            order['status'] = OrderStatus.FILLED.value
            order['filled_amount'] = amount
            order['avg_fill_price'] = limit_price
            order['filled_at'] = datetime.now(timezone.utc)
            
            slippage = abs(limit_price - market_price) / market_price
            order['slippage'] = slippage
            
            self.active_orders[order_id] = order
            
            self.logger.info(f"🎉 {log_prefix} Order filled (simulated): {order_id} @ ${limit_price:.4f}")
            
            return {
                'success': True, 'order_id': order_id, 'filled_amount': amount,
                'avg_price': limit_price, 'slippage': slippage, 'order': order
            }
            
        except Exception as e:
            _log_rest_debug(f"LIMIT/{exchange}/{symbol}", e)
            self.logger.error(f"💥 {log_prefix} Limit order execution failed critically: {e}", exc_info=True)
            return {'success': False, 'reason': str(e), 'order_id': None}
    
    async def _iceberg_order_execution(self, order_request: Dict, clients_to_use: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute iceberg order (large orders split into smaller slices).
        
        Args:
            order_request: Order request dictionary
            clients_to_use: Exchange clients to use (defaults to self.exchange_clients)
            
        Returns:
            Execution result
        """
        try:
            total_amount = order_request['amount']
            slice_size = total_amount * 0.10  # 10% slices
            num_slices = int(total_amount / slice_size)
            
            logger.info(f"Executing iceberg order: {total_amount} in {num_slices} slices of {slice_size}")
            
            fills = []
            total_filled = 0.0
            
            for i in range(num_slices):
                slice_request = {**order_request, 'amount': slice_size}
                result = await self._limit_order_execution(slice_request, clients_to_use)
                
                if result['success']:
                    fills.append(result)
                    total_filled += result['filled_amount']
                    
                    # Wait between slices
                    if i < num_slices - 1:
                        await asyncio.sleep(30)
                else:
                    logger.warning(f"Slice {i+1} failed: {result.get('reason')}")
            
            # Calculate average fill price
            if fills:
                avg_price = sum(f['avg_price'] * f['filled_amount'] for f in fills) / total_filled
            else:
                avg_price = 0.0
            
            order_id = f"iceberg_{int(time.time() * 1000)}"
            
            return {
                'success': total_filled > 0,
                'order_id': order_id,
                'filled_amount': total_filled,
                'avg_price': avg_price,
                'num_slices': len(fills),
                'fills': fills
            }
            
        except Exception as e:
            logger.error(f"Iceberg order execution failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'order_id': None
            }
    
    async def _twap_order_execution(self, order_request: Dict, clients_to_use: Optional[Dict] = None, time_window: int = 300) -> Dict[str, Any]:
        """
        Time-Weighted Average Price execution.
        
        Args:
            order_request: Order request dictionary
            clients_to_use: Exchange clients to use (defaults to self.exchange_clients)
            time_window: Execution time window in seconds (default 5 minutes)
            
        Returns:
            Execution result
        """
        try:
            total_amount = order_request['amount']
            num_slices = 10
            slice_size = total_amount / num_slices
            interval = time_window / num_slices
            
            logger.info(f"Executing TWAP order: {total_amount} over {time_window}s in {num_slices} slices")
            
            fills = []
            total_filled = 0.0
            
            for i in range(num_slices):
                slice_request = {**order_request, 'amount': slice_size}
                result = await self._market_order_execution(slice_request, clients_to_use)
                
                if result['success']:
                    fills.append(result)
                    total_filled += result['filled_amount']
                    
                    # Wait for next interval
                    if i < num_slices - 1:
                        await asyncio.sleep(interval)
                else:
                    logger.warning(f"TWAP slice {i+1} failed: {result.get('reason')}")
            
            # Calculate TWAP
            if fills:
                twap = sum(f['avg_price'] * f['filled_amount'] for f in fills) / total_filled
            else:
                twap = 0.0
            
            order_id = f"twap_{int(time.time() * 1000)}"
            
            return {
                'success': total_filled > 0,
                'order_id': order_id,
                'filled_amount': total_filled,
                'avg_price': twap,
                'twap': twap,
                'num_slices': len(fills),
                'fills': fills
            }
            
        except Exception as e:
            logger.error(f"TWAP order execution failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'order_id': None
            }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics."""
        return {
            **self.execution_stats,
            'active_orders': len(self.active_orders),
            'order_history_size': len(self.order_history),
            'success_rate': (
                self.execution_stats['successful_orders'] / self.execution_stats['total_orders']
                if self.execution_stats['total_orders'] > 0 else 0.0
            )
        }
