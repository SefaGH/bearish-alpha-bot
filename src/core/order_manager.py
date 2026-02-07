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

from .execution_env import (
    get_bingx_env,
    get_execution_backend,
    get_trading_mode,
    is_bingx_leverage_fail_fast,
    is_vst_fullbot_canary_enabled,
    is_vst_fullbot_canary_force_market,
    is_real_execution_enabled,
    require_explicit_bingx_env_if_real_execution,
)

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


def _sanitize_ccxt_order(order: Any) -> Dict[str, Any]:
    if not isinstance(order, dict):
        return {"type": str(type(order))}

    keep_keys = (
        "id",
        "clientOrderId",
        "symbol",
        "type",
        "side",
        "status",
        "timestamp",
        "datetime",
        "amount",
        "filled",
        "remaining",
        "price",
        "average",
        "cost",
        "reduceOnly",
    )
    out: Dict[str, Any] = {k: order.get(k) for k in keep_keys if k in order}

    info = order.get("info")
    if isinstance(info, dict):
        items = list(info.items())[:25]
        out["info"] = {k: v for k, v in items}
    elif info is not None:
        out["info"] = str(info)[:500]

    return out


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


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return None


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
        self._fallback_locks: Dict[str, asyncio.Lock] = {}
        
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
            env_forced_order_type = None
            requested_execution_algo = str(execution_algo or "").lower().strip()
            
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
            
            # VST full-bot canary safety: enforce market-only execution.
            # This avoids accidental LIMIT placement (and Stage-1 rejects LIMIT in real execution anyway).
            if is_vst_fullbot_canary_enabled() and is_real_execution_enabled():
                requested = str(execution_algo or "").lower().strip()
                if requested != "market" and is_vst_fullbot_canary_force_market():
                    logger.warning("[VST-FULLBOT-CANARY] Forcing MARKET execution (requested=%s)", execution_algo)
                    execution_algo = "market"
                    env_forced_order_type = "market"

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
                is_abort = reason.startswith("abort:")
                is_transient = (not is_abort) and any(
                    x in reason for x in ['timeout', 'connection', 'rate limit', '500', '502', '503', '504', 'network', 'reset']
                )
                
                if not is_transient or attempt == max_retries - 1:
                    break
                
                wait_time = (2 ** attempt) * 0.5
                logger.warning(f"Transient failure in order execution: {reason}. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            
            logger.debug(f"🎪 [ORDER-MGR] Execution result: {'SUCCESS' if result and result.get('success') else 'FAILED'}")

            if isinstance(result, dict):
                execution_time_ms = int(max((time.time() - start_time) * 1000.0, 0.0))
                result.setdefault("requested_execution_algo", requested_execution_algo)
                result.setdefault("effective_execution_algo", str(execution_algo or "").lower().strip())
                result.setdefault("time_to_fill_ms", execution_time_ms)
                if env_forced_order_type:
                    result.setdefault("env_forced_order_type", env_forced_order_type)

                logger.info(
                    "order_manager_decision %s",
                    {
                        "event": "order_manager_decision",
                        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "symbol": order_request.get("symbol"),
                        "side": order_request.get("side"),
                        "requested_execution_algo": requested_execution_algo,
                        "effective_execution_algo": result.get("effective_execution_algo"),
                        "env_forced_order_type": result.get("env_forced_order_type"),
                        "fallback_reason": result.get("fallback_reason"),
                        "time_to_fill_ms": result.get("time_to_fill_ms"),
                        "success": bool(result.get("success")),
                        "reason": result.get("reason"),
                    },
                )
            
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

            if is_real_execution_enabled():
                symbol = order.get("symbol")
                try:
                    resp = client.cancel_order(order_id, symbol, params={})
                    logger.info(f"🟢 [REAL EXECUTION] Cancel response: {resp}")
                except Exception as exc:
                    msg = str(exc).lower()
                    idempotent = any(x in msg for x in ["already canceled", "already cancelled", "already closed", "not found", "does not exist"])
                    order_not_found_types = tuple(
                        t for t in (getattr(ccxt, "OrderNotFound", None), getattr(ccxt, "InvalidOrder", None))
                        if isinstance(t, type)
                    )
                    order_not_found = bool(order_not_found_types) and isinstance(exc, order_not_found_types)
                    if not (idempotent or order_not_found):
                        return {'success': False, 'reason': str(exc)}

            # Update local state
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

    def _get_fallback_lock(self, key: str) -> asyncio.Lock:
        lock = self._fallback_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._fallback_locks[key] = lock
        return lock

    def _fallback_lock_key(self, *, exchange: str, symbol: str, side: str, ccxt_params: Dict[str, Any]) -> str:
        pos_side = ccxt_params.get("positionSide") or ccxt_params.get("position_side") or ""
        return f"{str(exchange).lower()}::{str(symbol).upper()}::{str(side).lower()}::{str(pos_side).upper()}"

    def _extract_position_qty_for_side(self, *, position: Dict[str, Any], symbol: str, normalized_side: str, pos_side_hint: str) -> Optional[float]:
        if not isinstance(position, dict):
            return None
        pos_symbol = str(position.get("symbol") or position.get("market") or "").upper().strip()
        if pos_symbol and str(symbol).upper().strip() not in {pos_symbol, pos_symbol.replace(":", "/")}:
            return None

        raw_side = (
            position.get("side")
            or position.get("positionSide")
            or position.get("position_side")
            or (position.get("info") or {}).get("positionSide")
        )
        side_norm = str(raw_side or "").lower().strip()
        if side_norm == "long":
            side_norm = "buy"
        elif side_norm == "short":
            side_norm = "sell"

        qty = None
        for key in ("contracts", "positionAmt", "position_amt", "amount", "size", "qty"):
            if key in position:
                try:
                    qty = float(position.get(key) or 0.0)
                    break
                except Exception:
                    qty = None
        if qty is None:
            info = position.get("info")
            if isinstance(info, dict):
                for key in ("positionAmt", "position_amt", "amount", "size", "qty"):
                    if key in info:
                        try:
                            qty = float(info.get(key) or 0.0)
                            break
                        except Exception:
                            qty = None

        if qty is None:
            return None

        qty_abs = abs(float(qty))
        target_pos_side = "long" if normalized_side == "buy" else "short"
        target_bingx_side = "LONG" if normalized_side == "buy" else "SHORT"

        if pos_side_hint:
            hint = str(pos_side_hint).upper().strip()
            if hint in {"LONG", "SHORT"}:
                return qty_abs if hint == target_bingx_side else 0.0

        if side_norm in {"buy", "sell"}:
            return qty_abs if side_norm == normalized_side else 0.0
        if side_norm in {"long", "short"}:
            return qty_abs if side_norm == target_pos_side else 0.0

        # One-way/net mode fallback when explicit side is unavailable.
        if qty >= 0 and normalized_side == "buy":
            return qty_abs
        if qty <= 0 and normalized_side == "sell":
            return qty_abs
        return 0.0

    def _fetch_position_qty_for_side(self, *, client: Any, symbol: str, normalized_side: str, ccxt_params: Dict[str, Any]) -> Optional[float]:
        pos_side_hint = str(ccxt_params.get("positionSide") or ccxt_params.get("position_side") or "").strip()

        entries: List[Dict[str, Any]] = []
        if callable(getattr(client, "fetch_positions", None)):
            try:
                fetched = client.fetch_positions([symbol], params={})
                if isinstance(fetched, list):
                    entries.extend([p for p in fetched if isinstance(p, dict)])
            except Exception:
                pass

        if not entries and callable(getattr(client, "fetch_position", None)):
            try:
                one = client.fetch_position(symbol, params={})
                if isinstance(one, dict):
                    entries.append(one)
                elif isinstance(one, list):
                    entries.extend([p for p in one if isinstance(p, dict)])
            except Exception:
                pass

        if not entries:
            return None

        total = 0.0
        any_data = False
        for p in entries:
            q = self._extract_position_qty_for_side(
                position=p,
                symbol=symbol,
                normalized_side=normalized_side,
                pos_side_hint=pos_side_hint,
            )
            if q is None:
                continue
            any_data = True
            total += max(float(q), 0.0)
        if not any_data:
            return None
        return float(total)
    
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

            normalized_side = str(side).lower().strip()
            if normalized_side == "long":
                normalized_side = "buy"
            elif normalized_side == "short":
                normalized_side = "sell"

            ccxt_params = {}
            if isinstance(order_request.get("params"), dict):
                ccxt_params.update(order_request["params"])
            if isinstance(order_request.get("execution_params"), dict):
                ccxt_params.update(order_request["execution_params"])
             
            # CRITICAL: Use clients_to_use if provided
            clients = clients_to_use if clients_to_use is not None else self.exchange_clients
            client = clients[exchange]
            
            # Get current market price for slippage monitoring
            ticker = client.ticker(symbol)
            expected_price = float(ticker.get('last', 0))
            
            trading_mode = get_trading_mode()
            execution_backend = get_execution_backend()

            logger.info(
                f"Executing market order: {symbol} {normalized_side} {amount} @ ~{expected_price} "
                f"(TRADING_MODE={trading_mode}, EXECUTION_BACKEND={execution_backend})"
            )

            if is_real_execution_enabled():
                require_explicit_bingx_env_if_real_execution()
                bingx_env = get_bingx_env()

                try:
                    client.load_markets()
                except Exception as exc:
                    logger.warning(f"[REAL EXECUTION] load_markets failed (continuing): {exc}")

                if getattr(client, "name", None) == "bingx" and bingx_env == "vst":
                    client.ensure_bingx_hedge_mode(symbol, require_hedged=True)

                # Best-effort: apply signal leverage to the exchange BEFORE entry.
                # BingX/CCXT leverage setting is per-symbol (and sometimes cached exchange-side),
                # so we do it right before creating the order.
                try:
                    signal = order_request.get('signal') if isinstance(order_request.get('signal'), dict) else {}
                    leverage = signal.get('leverage') if isinstance(signal, dict) else None
                    reduce_only = bool(ccxt_params.get('reduceOnly') or ccxt_params.get('reduce_only'))
                    if leverage and not reduce_only and callable(getattr(client, 'set_leverage', None)):
                        side_hint = None
                        raw_side = ccxt_params.get('positionSide') or ccxt_params.get('position_side')
                        if raw_side:
                            side_hint = str(raw_side).strip().upper()
                        elif normalized_side == "buy":
                            side_hint = "LONG"
                        elif normalized_side == "sell":
                            side_hint = "SHORT"
                        strict = is_bingx_leverage_fail_fast()
                        client.set_leverage(symbol, leverage, side=side_hint, strict=strict)
                except Exception as exc:
                    logger.warning("[REAL EXECUTION] set_leverage failed (continuing): %s", exc)
                    if is_bingx_leverage_fail_fast():
                        raise

                logger.warning(f"🟢 [REAL EXECUTION] Submitting MARKET order via CCXT ({exchange})")

                exchange_order = client.create_order(
                    symbol=symbol,
                    side=normalized_side,
                    type_="market",
                    amount=amount,
                    price=None,
                    params=ccxt_params or {},
                )
                logger.info("🟢 [REAL EXECUTION] Exchange order (sanitized): %s", _sanitize_ccxt_order(exchange_order))

                exchange_order_id = exchange_order.get("id") or exchange_order.get("orderId")
                avg_fill_price = exchange_order.get("average") or exchange_order.get("price")
                filled_amount = exchange_order.get("filled") or exchange_order.get("amount") or amount

                try:
                    avg_fill_price = float(avg_fill_price or 0)
                except Exception:
                    avg_fill_price = 0.0
                try:
                    filled_amount = float(filled_amount or 0)
                except Exception:
                    filled_amount = float(amount or 0)

                slippage = 0.0
                if expected_price and avg_fill_price:
                    slippage = abs(avg_fill_price - expected_price) / expected_price

                self.execution_stats['total_slippage'] += slippage

                order = {
                    'order_id': exchange_order_id,
                    'symbol': symbol,
                    'side': normalized_side,
                    'amount': amount,
                    'type': 'market',
                    'exchange': exchange,
                    'expected_price': expected_price,
                    'status': (exchange_order.get("status") or OrderStatus.FILLED.value),
                    'created_at': datetime.now(timezone.utc),
                    'filled_amount': filled_amount,
                    'avg_fill_price': avg_fill_price or expected_price,
                    'filled_at': datetime.now(timezone.utc),
                    'slippage': slippage,
                    'ccxt_params': ccxt_params,
                    'exchange_order': exchange_order,
                }

                # Store for cancellation/audit (even if filled)
                if exchange_order_id:
                    self.active_orders[exchange_order_id] = order

                logger.info(
                    f"🟢 [REAL EXECUTION] Market order result: id={exchange_order_id} "
                    f"avg={order['avg_fill_price']:.4f} filled={filled_amount} slippage={slippage*100:.3f}%"
                )

                return {
                    'success': bool(exchange_order_id),
                    'order_id': exchange_order_id,
                    'filled_amount': filled_amount,
                    'avg_price': order['avg_fill_price'],
                    'slippage': slippage,
                    'order': order,
                }
             
            # Generate order ID
            order_id = f"order_{int(time.time() * 1000)}"
            
            # Create order record
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'side': normalized_side,
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
            execution_price = expected_price * (1.0001 if normalized_side == 'buy' else 0.9999)
            
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
            
            logger.info(f"🟡 [SIMULATED] Market order filled: {order_id} @ {execution_price:.4f} (slippage: {slippage*100:.3f}%)")
             
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
    
    async def _limit_order_execution(self, order_request: Dict, clients_to_use: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute limit order with smart pricing.
        (UPDATED: Now uses MarketDataPipeline for market metadata retrieval)
        """
        symbol = order_request.get('symbol')
        side = order_request.get('side')
        amount = order_request.get('amount')
        exchange = order_request.get('exchange')
        log_prefix = f"[ORDER-EXEC/{exchange}/{symbol}]"
        execution_params = {}
        if kwargs:
            execution_params = kwargs.get('execution_params') or {}
        if not execution_params:
            execution_params = order_request.get('execution_params') or {}
        if execution_params.get('post_only') or execution_params.get('postOnly'):
            self.logger.info("[PAPER] Processing POST_ONLY Limit Order request")

        def _normalize_side(raw: str) -> str:
            s = str(raw or "").lower().strip()
            if s == "long":
                return "buy"
            if s == "short":
                return "sell"
            return s

        def _is_filled(exchange_order: dict) -> bool:
            status = str((exchange_order or {}).get("status") or "").lower().strip()
            if status in {"closed", "filled"}:
                return True
            filled = (exchange_order or {}).get("filled")
            amount_ = (exchange_order or {}).get("amount")
            try:
                filled_f = float(filled) if filled is not None else 0.0
                amount_f = float(amount_) if amount_ is not None else 0.0
            except Exception:
                return False
            return amount_f > 0 and filled_f >= max(amount_f * 0.999, amount_f - 1e-12)

        try:
            clients = clients_to_use if clients_to_use is not None else self.exchange_clients
            client = clients[exchange]
            
            # Get market metadata from pipeline (paper/sim only). Real execution should not depend on it.
            market = {}
            if not is_real_execution_enabled():
                try:
                    market = await self.market_data_pipeline.get_market_metadata(symbol, exchange)
                except ValueError as e:
                    # Sanitize error message to avoid exposing internal details
                    error_msg = f"Market metadata unavailable for {symbol} on {exchange}"
                    self.logger.error(f"🛡️  {log_prefix} REJECTED (MarketMetadata): {e}")
                    return {'success': False, 'reason': f"REJECT:MARKET_METADATA - {error_msg}", 'order_id': None}
            
            # SSOT pricing: limit price must be provided by the caller (signal/engine).
            # OrderManager must NOT fetch ticker prices to determine limit price (prevents risk/execution drift).
            signal = order_request.get('signal') if isinstance(order_request.get('signal'), dict) else {}
            raw_limit = (
                order_request.get('limit_price')
                or signal.get('limit_price')
                or signal.get('execution_price')
                or signal.get('entry')
            )
            try:
                limit_price = float(raw_limit or 0.0)
            except (TypeError, ValueError):
                limit_price = 0.0
            if limit_price <= 0:
                return {'success': False, 'reason': 'REJECT:MISSING_LIMIT_PRICE', 'order_id': None}

            ref_price = signal.get('entry_raw') or signal.get('entry')
            try:
                reference_price = float(ref_price or 0.0)
            except (TypeError, ValueError):
                reference_price = 0.0
            
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

            # Real execution path: place LIMIT and wait up to timeout for fill, then fallback.
            if is_real_execution_enabled():
                require_explicit_bingx_env_if_real_execution()
                bingx_env = get_bingx_env()

                # Merge ccxt params from request
                ccxt_params = {}
                if isinstance(order_request.get("params"), dict):
                    ccxt_params.update(order_request["params"])
                if isinstance(order_request.get("execution_params"), dict):
                    ccxt_params.update(order_request["execution_params"])

                try:
                    client.load_markets()
                except Exception as exc:
                    logger.warning(f"[REAL EXECUTION] load_markets failed (continuing): {exc}")

                if getattr(client, "name", None) == "bingx" and bingx_env == "vst":
                    client.ensure_bingx_hedge_mode(symbol, require_hedged=True)

                # Best-effort leverage application before entry (same semantics as market).
                try:
                    leverage = signal.get('leverage') if isinstance(signal, dict) else None
                    reduce_only = bool(ccxt_params.get('reduceOnly') or ccxt_params.get('reduce_only'))
                    if leverage and not reduce_only and callable(getattr(client, 'set_leverage', None)):
                        side_hint = None
                        raw_side = ccxt_params.get('positionSide') or ccxt_params.get('position_side')
                        if raw_side:
                            side_hint = str(raw_side).strip().upper()
                        else:
                            side_hint = "LONG" if _normalize_side(side) == "buy" else "SHORT"
                        strict = is_bingx_leverage_fail_fast()
                        client.set_leverage(symbol, leverage, side=side_hint, strict=strict)
                except Exception as exc:
                    logger.warning("[REAL EXECUTION] set_leverage failed (continuing): %s", exc)
                    if is_bingx_leverage_fail_fast():
                        raise

                normalized_side = _normalize_side(side)
                if normalized_side not in {"buy", "sell"}:
                    return {'success': False, 'reason': f"REJECT:INVALID_SIDE:{side}", 'order_id': None}

                preflight_enabled = _coerce_bool(execution_params.get("fallback_preflight_position_check_enabled"))
                if preflight_enabled is None:
                    preflight_enabled = True
                reduce_only = bool(ccxt_params.get("reduceOnly") or ccxt_params.get("reduce_only"))
                baseline_position_qty = None
                if preflight_enabled and (not reduce_only):
                    baseline_position_qty = self._fetch_position_qty_for_side(
                        client=client,
                        symbol=symbol,
                        normalized_side=normalized_side,
                        ccxt_params=ccxt_params,
                    )

                # Parameters: allow optional timeInForce in ccxt_params if provided by caller.
                logger.warning(f"🟢 [REAL EXECUTION] Submitting LIMIT order via CCXT ({exchange})")

                exchange_order = client.create_order(
                    symbol=symbol,
                    side=normalized_side,
                    type_="limit",
                    amount=amount,
                    price=limit_price,
                    params=ccxt_params or {},
                )
                logger.info("🟢 [REAL EXECUTION] Exchange order (sanitized): %s", _sanitize_ccxt_order(exchange_order))

                exchange_order_id = exchange_order.get("id") or exchange_order.get("orderId")
                if not exchange_order_id:
                    return {'success': False, 'reason': 'REJECT:EXCHANGE_ORDER_ID_MISSING', 'order_id': None}

                # If filled immediately, return as executed.
                if _is_filled(exchange_order):
                    avg_fill_price = exchange_order.get("average") or exchange_order.get("price") or limit_price
                    filled_amount = exchange_order.get("filled") or exchange_order.get("amount") or amount
                    try:
                        avg_fill_price = float(avg_fill_price or 0)
                    except Exception:
                        avg_fill_price = float(limit_price)
                    try:
                        filled_amount = float(filled_amount or 0)
                    except Exception:
                        filled_amount = float(amount or 0)

                    slippage = 0.0
                    if reference_price > 0 and avg_fill_price > 0:
                        slippage = abs(avg_fill_price - reference_price) / reference_price
                    self.execution_stats['total_slippage'] += slippage

                    order = {
                        'order_id': exchange_order_id,
                        'symbol': symbol,
                        'side': normalized_side,
                        'amount': amount,
                        'type': 'limit',
                        'limit_price': limit_price,
                        'exchange': exchange,
                        'expected_price': reference_price if reference_price > 0 else limit_price,
                        'status': (exchange_order.get("status") or OrderStatus.FILLED.value),
                        'created_at': datetime.now(timezone.utc),
                        'filled_amount': filled_amount,
                        'avg_fill_price': avg_fill_price,
                        'filled_at': datetime.now(timezone.utc),
                        'slippage': slippage,
                        'ccxt_params': ccxt_params,
                        'exchange_order': exchange_order,
                    }
                    self.active_orders[exchange_order_id] = order

                    return {
                        'success': True,
                        'order_id': exchange_order_id,
                        'filled_amount': filled_amount,
                        'avg_price': avg_fill_price,
                        'slippage': slippage,
                        'order': order,
                    }

                # Otherwise, wait up to timeout and optionally fallback to market.
                poll_s = float(execution_params.get("poll_interval_s") or execution_params.get("poll_interval") or 1.0)
                if poll_s <= 0:
                    poll_s = 1.0

                timeout_s = execution_params.get("timeout_seconds") or execution_params.get("timeout_s")
                if timeout_s is None:
                    timeout_min = execution_params.get("timeout_minutes") or execution_params.get("timeout_min")
                    try:
                        timeout_s = float(timeout_min) * 60.0 if timeout_min is not None else 0.0
                    except Exception:
                        timeout_s = 0.0
                try:
                    timeout_s = float(timeout_s or 0.0)
                except Exception:
                    timeout_s = 0.0
                if timeout_s <= 0:
                    # Safe default: 0 means do not wait; caller should set this for Smart Entry.
                    timeout_s = 0.0

                stop_price = None
                for k in ("stop", "stop_loss", "stopLoss"):
                    if k in signal:
                        try:
                            stop_price = float(signal.get(k) or 0.0)
                        except Exception:
                            stop_price = None
                        break

                start_ts = time.time()
                last_seen_order = exchange_order
                while timeout_s > 0 and (time.time() - start_ts) < timeout_s:
                    # 1) stop-hit-before-entry check
                    if stop_price and stop_price > 0:
                        try:
                            tick = client.ticker(symbol)
                            last_px = float((tick or {}).get("last") or 0.0)
                        except Exception:
                            last_px = 0.0

                        if last_px > 0:
                            if normalized_side == "buy" and last_px <= stop_price:
                                try:
                                    client.cancel_order(exchange_order_id, symbol, params={})
                                except Exception:
                                    pass
                                return {'success': False, 'reason': 'ABORT:STOP_HIT_BEFORE_ENTRY', 'order_id': exchange_order_id}
                            if normalized_side == "sell" and last_px >= stop_price:
                                try:
                                    client.cancel_order(exchange_order_id, symbol, params={})
                                except Exception:
                                    pass
                                return {'success': False, 'reason': 'ABORT:STOP_HIT_BEFORE_ENTRY', 'order_id': exchange_order_id}

                    # 2) order fill check
                    fetched = None
                    if callable(getattr(client, "fetch_order", None)):
                        try:
                            fetched = client.fetch_order(exchange_order_id, symbol, params={})
                        except Exception:
                            fetched = None
                    last_seen_order = fetched or last_seen_order
                    if last_seen_order and _is_filled(last_seen_order):
                        avg_fill_price = last_seen_order.get("average") or last_seen_order.get("price") or limit_price
                        filled_amount = last_seen_order.get("filled") or last_seen_order.get("amount") or amount
                        try:
                            avg_fill_price = float(avg_fill_price or 0)
                        except Exception:
                            avg_fill_price = float(limit_price)
                        try:
                            filled_amount = float(filled_amount or 0)
                        except Exception:
                            filled_amount = float(amount or 0)

                        slippage = 0.0
                        if reference_price > 0 and avg_fill_price > 0:
                            slippage = abs(avg_fill_price - reference_price) / reference_price
                        self.execution_stats['total_slippage'] += slippage

                        order = {
                            'order_id': exchange_order_id,
                            'symbol': symbol,
                            'side': normalized_side,
                            'amount': amount,
                            'type': 'limit',
                            'limit_price': limit_price,
                            'exchange': exchange,
                            'expected_price': reference_price if reference_price > 0 else limit_price,
                            'status': (last_seen_order.get("status") or OrderStatus.FILLED.value),
                            'created_at': datetime.now(timezone.utc),
                            'filled_amount': filled_amount,
                            'avg_fill_price': avg_fill_price,
                            'filled_at': datetime.now(timezone.utc),
                            'slippage': slippage,
                            'ccxt_params': ccxt_params,
                            'exchange_order': last_seen_order,
                        }
                        self.active_orders[exchange_order_id] = order
                        return {
                            'success': True,
                            'order_id': exchange_order_id,
                            'filled_amount': filled_amount,
                            'avg_price': avg_fill_price,
                            'slippage': slippage,
                            'order': order,
                        }

                    await asyncio.sleep(poll_s)

                # Timeout reached: apply chase gate (directional bps) and optionally market-fallback.
                try:
                    tick = client.ticker(symbol)
                    current_px = float((tick or {}).get("last") or 0.0)
                except Exception:
                    current_px = 0.0

                ref_px = None
                for k in ("entry_raw", "entry", "execution_price"):
                    if k in signal:
                        try:
                            ref_px = float(signal.get(k) or 0.0)
                        except Exception:
                            ref_px = None
                        break

                max_chase_bps = execution_params.get("max_chase_bps")
                if max_chase_bps is None:
                    max_chase_bps = execution_params.get("max_chase_bps_long") if normalized_side == "buy" else execution_params.get("max_chase_bps_short")
                try:
                    max_chase_bps = None if max_chase_bps is None else float(max_chase_bps)
                except Exception:
                    max_chase_bps = None

                deviation_bps = None
                if current_px > 0 and ref_px and ref_px > 0:
                    if normalized_side == "buy":
                        deviation_bps = (current_px / ref_px - 1.0) * 10000.0
                    else:
                        deviation_bps = (1.0 - current_px / ref_px) * 10000.0

                lock_key = self._fallback_lock_key(
                    exchange=exchange,
                    symbol=symbol,
                    side=normalized_side,
                    ccxt_params=ccxt_params,
                )
                fallback_lock = self._get_fallback_lock(lock_key)
                async with fallback_lock:
                    # Cancel the resting limit order (best-effort) before any fallback.
                    try:
                        client.cancel_order(exchange_order_id, symbol, params={})
                    except Exception:
                        pass

                    post_cancel = None
                    post_cancel_filled_qty = 0.0

                    # If we can confirm the order filled during cancel race, do not market-fallback.
                    if callable(getattr(client, "fetch_order", None)):
                        try:
                            post_cancel = client.fetch_order(exchange_order_id, symbol, params={})
                        except Exception:
                            post_cancel = None
                        if post_cancel and _is_filled(post_cancel):
                            avg_fill_price = post_cancel.get("average") or post_cancel.get("price") or limit_price
                            filled_amount = post_cancel.get("filled") or post_cancel.get("amount") or amount
                            try:
                                avg_fill_price = float(avg_fill_price or 0)
                            except Exception:
                                avg_fill_price = float(limit_price)
                            try:
                                filled_amount = float(filled_amount or 0)
                            except Exception:
                                filled_amount = float(amount or 0)
                            slippage = 0.0
                            if reference_price > 0 and avg_fill_price > 0:
                                slippage = abs(avg_fill_price - reference_price) / reference_price
                            self.execution_stats['total_slippage'] += slippage
                            order = {
                                'order_id': exchange_order_id,
                                'symbol': symbol,
                                'side': normalized_side,
                                'amount': amount,
                                'type': 'limit',
                                'limit_price': limit_price,
                                'exchange': exchange,
                                'expected_price': reference_price if reference_price > 0 else limit_price,
                                'status': (post_cancel.get("status") or OrderStatus.FILLED.value),
                                'created_at': datetime.now(timezone.utc),
                                'filled_amount': filled_amount,
                                'avg_fill_price': avg_fill_price,
                                'filled_at': datetime.now(timezone.utc),
                                'slippage': slippage,
                                'ccxt_params': ccxt_params,
                                'exchange_order': post_cancel,
                            }
                            self.active_orders[exchange_order_id] = order
                            return {
                                'success': True,
                                'order_id': exchange_order_id,
                                'filled_amount': filled_amount,
                                'avg_price': avg_fill_price,
                                'slippage': slippage,
                                'order': order,
                            }

                        if isinstance(post_cancel, dict):
                            try:
                                post_cancel_filled_qty = max(float(post_cancel.get("filled") or 0.0), 0.0)
                            except Exception:
                                post_cancel_filled_qty = 0.0

                # Chase gate: negative deviation means price improved vs reference => always allow.
                if deviation_bps is not None and deviation_bps <= 0:
                    max_chase_bps = max_chase_bps if max_chase_bps is not None else 0.0

                if max_chase_bps is not None and deviation_bps is not None and deviation_bps > max_chase_bps:
                    return {
                        'success': False,
                        'reason': f"ABORT:CHASE_GATE:{deviation_bps:.2f}>{max_chase_bps:.2f}",
                        'order_id': exchange_order_id,
                    }

                # Market fallback (timeout): explicit flags first, then optional risk-regime overrides.
                fallback_enabled = True
                fallback_source = "default_true"
                for key in (
                    "market_fallback_on_timeout_enabled",
                    "market_fallback_on_timeout",
                    "market_fallback",
                    "fallback_market",
                ):
                    parsed = _coerce_bool(execution_params.get(key))
                    if parsed is not None:
                        fallback_enabled = bool(parsed)
                        fallback_source = key
                        break

                bucket = str(signal.get("volume_bucket") or "").upper().strip()
                disable_on_extreme = _coerce_bool(execution_params.get("disable_market_fallback_on_extreme_bucket"))
                disable_on_fast_move = _coerce_bool(execution_params.get("disable_market_fallback_on_fast_move"))

                meta = signal.get("meta") if isinstance(signal.get("meta"), dict) else {}
                reason_code = str(meta.get("reason_code") or "").lower().strip()
                price_moved_fast = bool(
                    _coerce_bool(execution_params.get("price_moved_fast"))
                    or _coerce_bool(signal.get("price_moved_fast"))
                    or reason_code == "price_moved_fast"
                )

                fallback_block_reason = None
                if bool(disable_on_extreme) and bucket == "EXTREME":
                    fallback_enabled = False
                    fallback_block_reason = "extreme_bucket"
                elif bool(disable_on_fast_move) and price_moved_fast:
                    fallback_enabled = False
                    fallback_block_reason = "price_moved_fast"
                current_position_qty = None
                position_delta_qty = None
                if preflight_enabled and baseline_position_qty is not None and (not reduce_only):
                    settle_ms = execution_params.get("fallback_cancel_settle_ms")
                    checks = execution_params.get("fallback_settle_checks")
                    try:
                        settle_ms = int(settle_ms if settle_ms is not None else 1000)
                    except Exception:
                        settle_ms = 1000
                    try:
                        checks = int(checks if checks is not None else 2)
                    except Exception:
                        checks = 2
                    if checks <= 0:
                        checks = 1
                    if settle_ms < 0:
                        settle_ms = 0
                    delay_s = (float(settle_ms) / 1000.0) / float(checks)
                    observed_qty = None
                    for i in range(checks):
                        q = self._fetch_position_qty_for_side(
                            client=client,
                            symbol=symbol,
                            normalized_side=normalized_side,
                            ccxt_params=ccxt_params,
                        )
                        if q is not None:
                            observed_qty = float(q)
                        if i < checks - 1 and delay_s > 0:
                            await asyncio.sleep(delay_s)
                    current_position_qty = observed_qty
                    if current_position_qty is not None:
                        position_delta_qty = max(float(current_position_qty) - float(baseline_position_qty), 0.0)

                observed_filled_qty = max(float(post_cancel_filled_qty or 0.0), float(position_delta_qty or 0.0))
                try:
                    requested_amount = max(float(amount or 0.0), 0.0)
                except Exception:
                    requested_amount = 0.0

                min_residual_qty = execution_params.get("fallback_min_residual_qty")
                if min_residual_qty is None:
                    min_residual_qty = 1e-8
                try:
                    min_residual_qty = max(float(min_residual_qty), 0.0)
                except Exception:
                    min_residual_qty = 1e-8

                remaining_qty = max(requested_amount - observed_filled_qty, 0.0)

                if observed_filled_qty > 0.0 and remaining_qty <= min_residual_qty:
                    logger.info(
                        "[ORDER-FALLBACK] skip market due to observed fill symbol=%s side=%s baseline_qty=%s current_qty=%s observed_filled=%s",
                        symbol,
                        side,
                        baseline_position_qty,
                        current_position_qty,
                        observed_filled_qty,
                    )
                    synthetic_avg = float(limit_price)
                    synthetic_slippage = 0.0
                    if reference_price > 0 and synthetic_avg > 0:
                        synthetic_slippage = abs(synthetic_avg - reference_price) / reference_price
                    return {
                        'success': True,
                        'order_id': exchange_order_id,
                        'filled_amount': float(observed_filled_qty),
                        'avg_price': synthetic_avg,
                        'slippage': synthetic_slippage,
                        'fallback_reason': 'limit_timeout_skip_market_position_delta',
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                        'baseline_position_qty': baseline_position_qty,
                        'current_position_qty': current_position_qty,
                    }

                if not fallback_enabled:
                    logger.info(
                        "[ORDER-FALLBACK] skipped symbol=%s side=%s source=%s block_reason=%s bucket=%s",
                        symbol,
                        side,
                        fallback_source,
                        fallback_block_reason or "disabled_by_flag",
                        bucket or "n/a",
                    )
                    return {
                        'success': False,
                        'reason': 'ABORT:NO_FILL_TIMEOUT',
                        'order_id': exchange_order_id,
                        'fallback_reason': (
                            f"limit_timeout_market_fallback_disabled:{fallback_block_reason}"
                            if fallback_block_reason
                            else "limit_timeout_market_fallback_disabled:flag"
                        ),
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                    }

                if remaining_qty <= min_residual_qty:
                    return {
                        'success': False,
                        'reason': 'ABORT:NO_FILL_TIMEOUT',
                        'order_id': exchange_order_id,
                        'fallback_reason': 'limit_timeout_market_fallback_skipped_no_residual',
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                    }

                # Place market order as fallback only for residual amount.
                logger.info(
                    "[ORDER-FALLBACK] limit_timeout_market_fallback symbol=%s side=%s deviation_bps=%s max_chase_bps=%s residual_qty=%.8f observed_filled=%.8f",
                    symbol,
                    side,
                    f"{deviation_bps:.2f}" if deviation_bps is not None else "n/a",
                    f"{max_chase_bps:.2f}" if max_chase_bps is not None else "n/a",
                    float(remaining_qty),
                    float(observed_filled_qty),
                )
                fallback_order_request = dict(order_request)
                fallback_order_request['amount'] = float(remaining_qty)
                fallback_result = await self._market_order_execution(fallback_order_request, clients_to_use)
                if isinstance(fallback_result, dict):
                    fallback_result.setdefault("fallback_reason", "limit_timeout_market_fallback")
                    fallback_result.setdefault("requested_order_type", "limit")
                    fallback_result.setdefault("effective_order_type", "market")
                    fallback_result.setdefault("fallback_residual_qty", float(remaining_qty))
                    fallback_result.setdefault("observed_filled_qty", float(observed_filled_qty))
                    fallback_result.setdefault("baseline_position_qty", baseline_position_qty)
                    fallback_result.setdefault("current_position_qty", current_position_qty)
                    if deviation_bps is not None:
                        fallback_result.setdefault("deviation_bps", float(deviation_bps))
                    if max_chase_bps is not None:
                        fallback_result.setdefault("max_chase_bps", float(max_chase_bps))
                return fallback_result
            
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
