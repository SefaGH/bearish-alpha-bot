"""
Smart Order Management System.
Advanced order management with execution optimization.
"""

import asyncio
import logging
import os
import time
import ccxt
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING, Set
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
                        "reason_code": result.get("reason_code"),
                        "fallback_hard_chase_reason": (
                            (result.get("fallback_hard_chase") or {}).get("reason")
                            if isinstance(result.get("fallback_hard_chase"), dict)
                            else None
                        ),
                        "fallback_soft_gate_reason": (
                            (result.get("fallback_soft_gate") or {}).get("reason")
                            if isinstance(result.get("fallback_soft_gate"), dict)
                            else None
                        ),
                        "fallback_slippage_guard_reason": (
                            (result.get("fallback_slippage_guard") or {}).get("reason")
                            if isinstance(result.get("fallback_slippage_guard"), dict)
                            else None
                        ),
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

    @staticmethod
    def _symbol_match_variants(symbol: str) -> Set[str]:
        raw = str(symbol or "").strip().upper()
        if not raw:
            return set()
        variants: Set[str] = {raw}
        slash = raw.replace("-", "/")
        variants.add(slash)
        if ":" in slash:
            variants.add(slash.split(":", 1)[0])
        if "/" in slash and ":" not in slash and slash.endswith("/USDT"):
            variants.add(f"{slash}:USDT")
        compact = slash.replace("/", "")
        if compact:
            variants.add(compact)
        return {v for v in variants if v}

    @classmethod
    def _symbols_match(cls, expected_symbol: str, position_symbol: str) -> bool:
        expected = cls._symbol_match_variants(expected_symbol)
        actual = cls._symbol_match_variants(position_symbol)
        if not expected or not actual:
            return False
        return not expected.isdisjoint(actual)

    @staticmethod
    def _is_terminal_order_status(status: Any) -> bool:
        text = str(status or "").strip().lower()
        return text in {"closed", "filled", "canceled", "cancelled", "rejected", "expired"}

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            out = float(value)
            if out != out:  # NaN
                return None
            return out
        except Exception:
            return None

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _extract_signal_atr_bps(self, *, signal: Dict[str, Any], ref_px: Optional[float]) -> Optional[float]:
        meta = signal.get("meta") if isinstance(signal.get("meta"), dict) else {}
        vol_tel = meta.get("vol_telemetry") if isinstance(meta.get("vol_telemetry"), dict) else {}
        volatility = signal.get("volatility") if isinstance(signal.get("volatility"), dict) else {}

        for candidate in (
            vol_tel.get("atr_bps"),
            volatility.get("vol_atr_bps"),
            volatility.get("atr_bps"),
            signal.get("atr_bps"),
        ):
            out = self._safe_float(candidate)
            if out is not None and out >= 0:
                return float(out)

        atr_abs = self._safe_float(signal.get("atr"))
        if atr_abs is not None and atr_abs > 0 and ref_px is not None and ref_px > 0:
            return float((float(atr_abs) / float(ref_px)) * 10000.0)
        return None

    def _extract_tick_spread_bps(self, *, tick: Dict[str, Any], signal: Dict[str, Any]) -> Optional[float]:
        bid = self._safe_float((tick or {}).get("bid"))
        ask = self._safe_float((tick or {}).get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask >= bid:
            mid = (bid + ask) / 2.0
            if mid > 0:
                return float(((ask - bid) / mid) * 10000.0)

        meta = signal.get("meta") if isinstance(signal.get("meta"), dict) else {}
        spread_meta = self._safe_float(meta.get("spread_bps"))
        if spread_meta is not None and spread_meta >= 0:
            return float(spread_meta)
        return None

    def _collect_fallback_hard_chase_cfg(
        self,
        *,
        order_request: Dict[str, Any],
        execution_params: Dict[str, Any],
        legacy_gate_bps: Optional[float],
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}

        exec_block = execution_params.get("fallback_hard_chase") if isinstance(execution_params, dict) else None
        if isinstance(exec_block, dict):
            cfg.update(exec_block)

        if isinstance(execution_params, dict):
            for k in (
                "fallback_hard_chase_enabled",
                "fallback_hard_chase_floor_bps",
                "fallback_hard_chase_min_bps",
                "fallback_hard_chase_max_bps",
                "fallback_hard_chase_atr_k",
                "fallback_hard_chase_spread_m",
            ):
                if execution_params.get(k) is not None:
                    cfg[k] = execution_params.get(k)

        internal = order_request.get("_internal") if isinstance(order_request.get("_internal"), dict) else {}
        internal_hard = (
            internal.get("fallback_hard_chase")
            if isinstance(internal.get("fallback_hard_chase"), dict)
            else None
        )
        if isinstance(internal_hard, dict):
            cfg.update(internal_hard)

        enabled = _coerce_bool(cfg.get("enabled"))
        if enabled is None:
            enabled = _coerce_bool(cfg.get("fallback_hard_chase_enabled"))
        if enabled is None:
            enabled = True

        default_floor_bps = 25.0
        floor_bps = self._safe_float(cfg.get("floor_bps", cfg.get("fallback_hard_chase_floor_bps")))
        if floor_bps is None:
            floor_bps = float(legacy_gate_bps) if legacy_gate_bps is not None else default_floor_bps
            floor_bps = max(float(floor_bps), default_floor_bps)

        min_bps = self._safe_float(cfg.get("min_bps", cfg.get("fallback_hard_chase_min_bps")))
        if min_bps is None:
            min_bps = 20.0

        max_bps = self._safe_float(cfg.get("max_bps", cfg.get("fallback_hard_chase_max_bps")))
        if max_bps is None:
            max_bps = 60.0

        atr_k = self._safe_float(cfg.get("atr_k", cfg.get("fallback_hard_chase_atr_k")))
        if atr_k is None:
            atr_k = 1.5

        spread_m = self._safe_float(cfg.get("spread_m", cfg.get("fallback_hard_chase_spread_m")))
        if spread_m is None:
            spread_m = 2.0

        min_bps = max(float(min_bps), 0.0)
        max_bps = max(float(max_bps), float(min_bps))
        floor_bps = max(float(floor_bps), 0.0)
        atr_k = max(float(atr_k), 0.0)
        spread_m = max(float(spread_m), 0.0)

        return {
            "enabled": bool(enabled),
            "floor_bps": float(floor_bps),
            "min_bps": float(min_bps),
            "max_bps": float(max_bps),
            "atr_k": float(atr_k),
            "spread_m": float(spread_m),
            "legacy_gate_bps": float(legacy_gate_bps) if legacy_gate_bps is not None else None,
        }

    def _evaluate_fallback_hard_chase(
        self,
        *,
        normalized_side: str,
        current_px: float,
        ref_px: Optional[float],
        tick: Dict[str, Any],
        signal: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            return {"enabled": False, "allow": True, "reason": "disabled"}

        floor_bps = float(cfg.get("floor_bps", 25.0) or 25.0)
        min_bps = float(cfg.get("min_bps", 20.0) or 20.0)
        max_bps = float(cfg.get("max_bps", 60.0) or 60.0)
        atr_k = float(cfg.get("atr_k", 1.5) or 1.5)
        spread_m = float(cfg.get("spread_m", 2.0) or 2.0)

        chase_bps = None
        if current_px > 0 and ref_px is not None and ref_px > 0:
            if normalized_side == "buy":
                chase_bps = max((float(current_px) - float(ref_px)) / float(ref_px) * 10000.0, 0.0)
            elif normalized_side == "sell":
                chase_bps = max((float(ref_px) - float(current_px)) / float(ref_px) * 10000.0, 0.0)

        atr_bps = self._extract_signal_atr_bps(signal=(signal if isinstance(signal, dict) else {}), ref_px=ref_px)
        spread_bps = self._extract_tick_spread_bps(
            tick=(tick if isinstance(tick, dict) else {}),
            signal=(signal if isinstance(signal, dict) else {}),
        )

        dynamic_component = (float(atr_bps or 0.0) * atr_k) + (float(spread_bps or 0.0) * spread_m)
        kill_bps_raw = max(float(floor_bps), float(dynamic_component))
        kill_bps = min(max(float(kill_bps_raw), float(min_bps)), float(max_bps))

        if chase_bps is None:
            return {
                "enabled": True,
                "allow": True,
                "reason": "fallback_hard_chase_no_context",
                "chase_bps": None,
                "kill_bps": float(kill_bps),
                "atr_bps": float(atr_bps) if atr_bps is not None else None,
                "spread_bps": float(spread_bps) if spread_bps is not None else None,
                "dynamic_component_bps": float(dynamic_component),
                "config": {
                    "floor_bps": float(floor_bps),
                    "min_bps": float(min_bps),
                    "max_bps": float(max_bps),
                    "atr_k": float(atr_k),
                    "spread_m": float(spread_m),
                },
            }

        allow = bool(float(chase_bps) <= float(kill_bps))
        return {
            "enabled": True,
            "allow": bool(allow),
            "reason": "fallback_hard_chase_pass" if allow else "fallback_hard_chase_kill",
            "chase_bps": float(chase_bps),
            "kill_bps": float(kill_bps),
            "atr_bps": float(atr_bps) if atr_bps is not None else None,
            "spread_bps": float(spread_bps) if spread_bps is not None else None,
            "dynamic_component_bps": float(dynamic_component),
            "config": {
                "floor_bps": float(floor_bps),
                "min_bps": float(min_bps),
                "max_bps": float(max_bps),
                "atr_k": float(atr_k),
                "spread_m": float(spread_m),
            },
        }

    def _collect_fallback_soft_gate_cfg(self, *, order_request: Dict[str, Any], execution_params: Dict[str, Any]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}

        exec_block = execution_params.get("fallback_soft_gate") if isinstance(execution_params, dict) else None
        if isinstance(exec_block, dict):
            cfg.update(exec_block)

        if isinstance(execution_params, dict):
            for k in (
                "fallback_soft_gate_enabled",
                "fallback_soft_gate_min_passes",
                "fallback_soft_gate_rr_min",
                "fallback_soft_gate_max_adverse_bps",
                "fallback_soft_gate_max_spread_bps",
                "fallback_soft_gate_max_peak_distance_bps",
                "fallback_soft_gate_fail_closed_on_insufficient_context",
            ):
                if execution_params.get(k) is not None:
                    cfg[k] = execution_params.get(k)

        internal = order_request.get("_internal") if isinstance(order_request.get("_internal"), dict) else {}
        internal_soft = internal.get("fallback_soft_gate") if isinstance(internal.get("fallback_soft_gate"), dict) else None
        if isinstance(internal_soft, dict):
            cfg.update(internal_soft)

        enabled = _coerce_bool(cfg.get("enabled"))
        if enabled is None:
            enabled = _coerce_bool(cfg.get("fallback_soft_gate_enabled"))
        if enabled is None:
            enabled = False

        min_passes = self._safe_int(cfg.get("min_passes", cfg.get("fallback_soft_gate_min_passes", 3)), 3)
        min_passes = max(1, min(min_passes, 4))

        rr_min = self._safe_float(cfg.get("rr_min", cfg.get("fallback_soft_gate_rr_min")))
        if rr_min is None:
            rr_min = 1.2

        max_adverse_bps = self._safe_float(
            cfg.get("max_adverse_bps", cfg.get("fallback_soft_gate_max_adverse_bps"))
        )
        if max_adverse_bps is None:
            max_adverse_bps = 15.0

        max_spread_bps = self._safe_float(cfg.get("max_spread_bps", cfg.get("fallback_soft_gate_max_spread_bps")))
        if max_spread_bps is None:
            max_spread_bps = 8.0

        max_peak_distance_bps = self._safe_float(
            cfg.get("max_peak_distance_bps", cfg.get("fallback_soft_gate_max_peak_distance_bps"))
        )
        if max_peak_distance_bps is None:
            max_peak_distance_bps = 20.0

        fail_closed = _coerce_bool(
            cfg.get(
                "fail_closed_on_insufficient_context",
                cfg.get("fallback_soft_gate_fail_closed_on_insufficient_context"),
            )
        )
        if fail_closed is None:
            fail_closed = False

        return {
            "enabled": bool(enabled),
            "min_passes": int(min_passes),
            "rr_min": float(rr_min),
            "max_adverse_bps": float(max_adverse_bps),
            "max_spread_bps": float(max_spread_bps),
            "max_peak_distance_bps": float(max_peak_distance_bps),
            "fail_closed_on_insufficient_context": bool(fail_closed),
        }

    def _evaluate_fallback_soft_gate(
        self,
        *,
        normalized_side: str,
        signal: Dict[str, Any],
        current_px: float,
        ref_px: Optional[float],
        tick: Dict[str, Any],
        wait_window_max_px: Optional[float],
        wait_window_min_px: Optional[float],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return {"enabled": False, "allow": True, "reason": "disabled"}

        min_passes = int(cfg.get("min_passes", 3) or 3)
        rr_min = float(cfg.get("rr_min", 1.2) or 1.2)
        max_adverse_bps = float(cfg.get("max_adverse_bps", 15.0) or 15.0)
        max_spread_bps = float(cfg.get("max_spread_bps", 8.0) or 8.0)
        max_peak_distance_bps = float(cfg.get("max_peak_distance_bps", 20.0) or 20.0)
        fail_closed = bool(cfg.get("fail_closed_on_insufficient_context", False))

        passes: List[str] = []
        fails: List[str] = []
        na: List[str] = []
        gates: Dict[str, Dict[str, Any]] = {}

        # Gate 1: edge_preserved (fallback fill still keeps acceptable RR)
        stop_px = self._safe_float(signal.get("stop") or signal.get("stop_loss"))
        target_px = self._safe_float(signal.get("target") or signal.get("take_profit"))
        if current_px > 0 and stop_px is not None and target_px is not None:
            risk = None
            reward = None
            if normalized_side == "buy":
                risk = current_px - stop_px
                reward = target_px - current_px
            elif normalized_side == "sell":
                risk = stop_px - current_px
                reward = current_px - target_px

            rr_now = None
            if risk is not None and reward is not None and risk > 0 and reward > 0:
                rr_now = float(reward / risk)
                ok = bool(rr_now >= rr_min)
                gates["edge_preserved"] = {"pass": ok, "na": False, "rr_now": rr_now, "rr_min": rr_min}
                (passes if ok else fails).append("edge_preserved")
            else:
                gates["edge_preserved"] = {"pass": False, "na": False, "rr_now": rr_now, "rr_min": rr_min}
                fails.append("edge_preserved")
        else:
            gates["edge_preserved"] = {"pass": False, "na": True}
            na.append("edge_preserved")

        # Gate 2: direction_continuity (avoid forcing market into strong adverse move)
        if current_px > 0 and ref_px is not None and ref_px > 0:
            adverse_bps = None
            if normalized_side == "buy":
                adverse_bps = max((float(ref_px) - float(current_px)) / float(ref_px) * 10000.0, 0.0)
            elif normalized_side == "sell":
                adverse_bps = max((float(current_px) - float(ref_px)) / float(ref_px) * 10000.0, 0.0)
            if adverse_bps is None:
                gates["direction_continuity"] = {"pass": False, "na": True}
                na.append("direction_continuity")
            else:
                ok = bool(float(adverse_bps) <= float(max_adverse_bps))
                gates["direction_continuity"] = {
                    "pass": ok,
                    "na": False,
                    "adverse_bps": float(adverse_bps),
                    "max_adverse_bps": float(max_adverse_bps),
                }
                (passes if ok else fails).append("direction_continuity")
        else:
            gates["direction_continuity"] = {"pass": False, "na": True}
            na.append("direction_continuity")

        # Gate 3: execution_quality (spread guard for market fallback)
        bid = self._safe_float((tick or {}).get("bid"))
        ask = self._safe_float((tick or {}).get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask >= bid:
            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / mid) * 10000.0 if mid > 0 else None
            if spread_bps is None:
                gates["execution_quality"] = {"pass": False, "na": True}
                na.append("execution_quality")
            else:
                ok = bool(float(spread_bps) <= float(max_spread_bps))
                gates["execution_quality"] = {
                    "pass": ok,
                    "na": False,
                    "spread_bps": float(spread_bps),
                    "max_spread_bps": float(max_spread_bps),
                }
                (passes if ok else fails).append("execution_quality")
        else:
            gates["execution_quality"] = {"pass": False, "na": True}
            na.append("execution_quality")

        # Gate 4: peak_distance (distance from wait-window extreme to current at timeout)
        peak_distance_bps = None
        if normalized_side == "sell":
            if wait_window_max_px is not None and wait_window_max_px > 0 and current_px > 0:
                peak_distance_bps = max(((float(wait_window_max_px) - float(current_px)) / float(wait_window_max_px)) * 10000.0, 0.0)
        elif normalized_side == "buy":
            if wait_window_min_px is not None and wait_window_min_px > 0 and current_px > 0:
                peak_distance_bps = max(((float(current_px) - float(wait_window_min_px)) / float(wait_window_min_px)) * 10000.0, 0.0)

        if peak_distance_bps is None:
            gates["peak_distance"] = {"pass": False, "na": True}
            na.append("peak_distance")
        else:
            ok = bool(float(peak_distance_bps) <= float(max_peak_distance_bps))
            gates["peak_distance"] = {
                "pass": ok,
                "na": False,
                "peak_distance_bps": float(peak_distance_bps),
                "max_peak_distance_bps": float(max_peak_distance_bps),
                "wait_window_max_px": float(wait_window_max_px) if wait_window_max_px is not None else None,
                "wait_window_min_px": float(wait_window_min_px) if wait_window_min_px is not None else None,
            }
            (passes if ok else fails).append("peak_distance")

        applicable = 4 - len(na)
        if applicable < min_passes:
            allow = not fail_closed
            reason = (
                "fallback_soft_gate_insufficient_context_allow"
                if allow
                else "fallback_soft_gate_insufficient_context_block"
            )
        else:
            allow = len(passes) >= min_passes
            reason = "fallback_soft_gate_pass" if allow else "fallback_soft_gate_block"

        return {
            "enabled": True,
            "allow": bool(allow),
            "reason": reason,
            "score": f"{len(passes)}/{applicable}",
            "required": int(min_passes),
            "passes": passes,
            "fails": fails,
            "na": na,
            "gates": gates,
            "config": {
                "min_passes": int(min_passes),
                "rr_min": float(rr_min),
                "max_adverse_bps": float(max_adverse_bps),
                "max_spread_bps": float(max_spread_bps),
                "max_peak_distance_bps": float(max_peak_distance_bps),
                "fail_closed_on_insufficient_context": bool(fail_closed),
            },
        }

    def _collect_fallback_slippage_guard_cfg(
        self,
        *,
        order_request: Dict[str, Any],
        execution_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}

        exec_block = execution_params.get("fallback_slippage_guard") if isinstance(execution_params, dict) else None
        if isinstance(exec_block, dict):
            cfg.update(exec_block)

        if isinstance(execution_params, dict):
            for k in (
                "fallback_slippage_guard_enabled",
                "fallback_slippage_guard_floor_bps",
                "fallback_slippage_guard_min_bps",
                "fallback_slippage_guard_max_bps",
                "fallback_slippage_guard_atr_k",
                "fallback_slippage_guard_spread_m",
                "fallback_slippage_guard_fail_closed_on_insufficient_context",
                "fallback_slippage_guard_reference",
            ):
                if execution_params.get(k) is not None:
                    cfg[k] = execution_params.get(k)

        internal = order_request.get("_internal") if isinstance(order_request.get("_internal"), dict) else {}
        internal_guard = (
            internal.get("fallback_slippage_guard")
            if isinstance(internal.get("fallback_slippage_guard"), dict)
            else None
        )
        if isinstance(internal_guard, dict):
            cfg.update(internal_guard)

        enabled = _coerce_bool(cfg.get("enabled"))
        if enabled is None:
            enabled = _coerce_bool(cfg.get("fallback_slippage_guard_enabled"))
        if enabled is None:
            enabled = False

        floor_bps = self._safe_float(cfg.get("floor_bps", cfg.get("fallback_slippage_guard_floor_bps")))
        if floor_bps is None:
            floor_bps = 8.0
        min_bps = self._safe_float(cfg.get("min_bps", cfg.get("fallback_slippage_guard_min_bps")))
        if min_bps is None:
            min_bps = 5.0
        max_bps = self._safe_float(cfg.get("max_bps", cfg.get("fallback_slippage_guard_max_bps")))
        if max_bps is None:
            max_bps = 35.0
        atr_k = self._safe_float(cfg.get("atr_k", cfg.get("fallback_slippage_guard_atr_k")))
        if atr_k is None:
            atr_k = 0.5
        spread_m = self._safe_float(cfg.get("spread_m", cfg.get("fallback_slippage_guard_spread_m")))
        if spread_m is None:
            spread_m = 1.0

        fail_closed = _coerce_bool(
            cfg.get(
                "fail_closed_on_insufficient_context",
                cfg.get("fallback_slippage_guard_fail_closed_on_insufficient_context"),
            )
        )
        if fail_closed is None:
            fail_closed = False

        reference_mode_raw = cfg.get(
            "reference_mode",
            cfg.get("reference", cfg.get("fallback_slippage_guard_reference")),
        )
        reference_mode = str(reference_mode_raw or "entry").strip().lower()
        if reference_mode not in {"entry", "limit"}:
            reference_mode = "entry"

        min_bps = max(float(min_bps), 0.0)
        max_bps = max(float(max_bps), float(min_bps))
        floor_bps = max(float(floor_bps), 0.0)
        atr_k = max(float(atr_k), 0.0)
        spread_m = max(float(spread_m), 0.0)

        return {
            "enabled": bool(enabled),
            "floor_bps": float(floor_bps),
            "min_bps": float(min_bps),
            "max_bps": float(max_bps),
            "atr_k": float(atr_k),
            "spread_m": float(spread_m),
            "fail_closed_on_insufficient_context": bool(fail_closed),
            "reference_mode": reference_mode,
        }

    def _evaluate_fallback_slippage_guard(
        self,
        *,
        normalized_side: str,
        signal: Dict[str, Any],
        tick: Dict[str, Any],
        current_px: float,
        entry_reference_px: Optional[float],
        limit_price: Optional[float],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return {"enabled": False, "allow": True, "reason": "disabled"}

        fail_closed = bool(cfg.get("fail_closed_on_insufficient_context", False))
        reference_mode = str(cfg.get("reference_mode") or "entry").strip().lower()
        if reference_mode not in {"entry", "limit"}:
            reference_mode = "entry"

        ref_px = None
        if reference_mode == "limit":
            ref_px = self._safe_float(limit_price)
        if ref_px is None or ref_px <= 0:
            ref_px = self._safe_float(entry_reference_px)

        bid = self._safe_float((tick or {}).get("bid"))
        ask = self._safe_float((tick or {}).get("ask"))
        expected_fill_px = None
        quote_source = None
        if normalized_side == "buy":
            if ask is not None and ask > 0:
                expected_fill_px = float(ask)
                quote_source = "ask"
            elif current_px and current_px > 0:
                expected_fill_px = float(current_px)
                quote_source = "last"
        elif normalized_side == "sell":
            if bid is not None and bid > 0:
                expected_fill_px = float(bid)
                quote_source = "bid"
            elif current_px and current_px > 0:
                expected_fill_px = float(current_px)
                quote_source = "last"

        if ref_px is None or ref_px <= 0 or expected_fill_px is None or expected_fill_px <= 0:
            allow = not fail_closed
            return {
                "enabled": True,
                "allow": bool(allow),
                "reason": (
                    "fallback_slippage_guard_insufficient_context_allow"
                    if allow
                    else "fallback_slippage_guard_insufficient_context_block"
                ),
                "reference_mode": reference_mode,
                "reference_price": ref_px,
                "expected_fill_price": expected_fill_px,
                "quote_source": quote_source,
                "adverse_bps": None,
                "kill_bps": None,
                "atr_bps": None,
                "spread_bps": None,
                "dynamic_component_bps": None,
                "config": {
                    "floor_bps": float(cfg.get("floor_bps", 8.0) or 8.0),
                    "min_bps": float(cfg.get("min_bps", 5.0) or 5.0),
                    "max_bps": float(cfg.get("max_bps", 35.0) or 35.0),
                    "atr_k": float(cfg.get("atr_k", 0.5) or 0.5),
                    "spread_m": float(cfg.get("spread_m", 1.0) or 1.0),
                    "fail_closed_on_insufficient_context": bool(fail_closed),
                },
            }

        floor_bps = float(cfg.get("floor_bps", 8.0) or 8.0)
        min_bps = float(cfg.get("min_bps", 5.0) or 5.0)
        max_bps = float(cfg.get("max_bps", 35.0) or 35.0)
        atr_k = float(cfg.get("atr_k", 0.5) or 0.5)
        spread_m = float(cfg.get("spread_m", 1.0) or 1.0)

        adverse_bps = None
        if normalized_side == "buy":
            adverse_bps = max(((float(expected_fill_px) - float(ref_px)) / float(ref_px)) * 10000.0, 0.0)
        elif normalized_side == "sell":
            adverse_bps = max(((float(ref_px) - float(expected_fill_px)) / float(ref_px)) * 10000.0, 0.0)

        atr_bps = self._extract_signal_atr_bps(signal=(signal if isinstance(signal, dict) else {}), ref_px=float(ref_px))
        spread_bps = self._extract_tick_spread_bps(
            tick=(tick if isinstance(tick, dict) else {}),
            signal=(signal if isinstance(signal, dict) else {}),
        )
        dynamic_component = (float(atr_bps or 0.0) * float(atr_k)) + (float(spread_bps or 0.0) * float(spread_m))
        kill_bps_raw = max(float(floor_bps), float(dynamic_component))
        kill_bps = min(max(float(kill_bps_raw), float(min_bps)), float(max_bps))

        allow = bool(float(adverse_bps or 0.0) <= float(kill_bps))
        return {
            "enabled": True,
            "allow": bool(allow),
            "reason": "fallback_slippage_guard_pass" if allow else "fallback_slippage_guard_kill",
            "reference_mode": reference_mode,
            "reference_price": float(ref_px),
            "expected_fill_price": float(expected_fill_px),
            "quote_source": quote_source,
            "adverse_bps": float(adverse_bps) if adverse_bps is not None else None,
            "kill_bps": float(kill_bps),
            "atr_bps": float(atr_bps) if atr_bps is not None else None,
            "spread_bps": float(spread_bps) if spread_bps is not None else None,
            "dynamic_component_bps": float(dynamic_component),
            "config": {
                "floor_bps": float(floor_bps),
                "min_bps": float(min_bps),
                "max_bps": float(max_bps),
                "atr_k": float(atr_k),
                "spread_m": float(spread_m),
                "fail_closed_on_insufficient_context": bool(fail_closed),
            },
        }

    def _extract_position_qty_for_side(self, *, position: Dict[str, Any], symbol: str, normalized_side: str, pos_side_hint: str) -> Optional[float]:
        if not isinstance(position, dict):
            return None
        pos_symbol = str(position.get("symbol") or position.get("market") or "").strip()
        if pos_symbol and not self._symbols_match(symbol, pos_symbol):
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

        def _extract_filled_qty(exchange_order: dict) -> float:
            if not isinstance(exchange_order, dict):
                return 0.0
            candidates: List[Any] = [
                exchange_order.get("filled"),
                exchange_order.get("executedQty"),
                exchange_order.get("executed_qty"),
            ]
            info = exchange_order.get("info")
            if isinstance(info, dict):
                candidates.extend(
                    [
                        info.get("executedQty"),
                        info.get("executed_qty"),
                        info.get("cumQty"),
                        info.get("dealQty"),
                    ]
                )
            for value in candidates:
                try:
                    qty = float(value or 0.0)
                except Exception:
                    continue
                if qty > 0:
                    return qty
            return 0.0

        def _is_cancelled_or_closed_status(exchange_order: dict) -> bool:
            status = str((exchange_order or {}).get("status") or "").lower().strip()
            return status in {"canceled", "cancelled", "closed", "expired", "rejected"}

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
                require_position_delta_verification = _coerce_bool(
                    execution_params.get("fallback_require_position_delta_verification")
                )
                if require_position_delta_verification is None:
                    require_position_delta_verification = True
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
                wait_window_max_px: Optional[float] = None
                wait_window_min_px: Optional[float] = None

                async def _cancel_with_stop_abort_verification() -> Dict[str, Any]:
                    try:
                        cancel_attempts = int(execution_params.get("stop_abort_cancel_attempts", 3))
                    except Exception:
                        cancel_attempts = 3
                    cancel_attempts = max(1, min(cancel_attempts, 10))

                    try:
                        fetch_attempts = int(execution_params.get("stop_abort_fetch_attempts", 3))
                    except Exception:
                        fetch_attempts = 3
                    fetch_attempts = max(1, min(fetch_attempts, 10))

                    try:
                        cancel_retry_delay_s = float(execution_params.get("stop_abort_cancel_retry_delay_s", 0.35))
                    except Exception:
                        cancel_retry_delay_s = 0.35
                    if cancel_retry_delay_s < 0:
                        cancel_retry_delay_s = 0.0

                    try:
                        fetch_retry_delay_s = float(execution_params.get("stop_abort_fetch_retry_delay_s", 0.20))
                    except Exception:
                        fetch_retry_delay_s = 0.20
                    if fetch_retry_delay_s < 0:
                        fetch_retry_delay_s = 0.0

                    cancel_errors: List[str] = []
                    last_snapshot: Optional[Dict[str, Any]] = None

                    for cancel_attempt in range(1, cancel_attempts + 1):
                        try:
                            cancel_resp = client.cancel_order(exchange_order_id, symbol, params={})
                            logger.warning(
                                "[STOP-ABORT/%s/%s] cancel attempt %s/%s order_id=%s resp=%s",
                                exchange,
                                symbol,
                                cancel_attempt,
                                cancel_attempts,
                                exchange_order_id,
                                _sanitize_ccxt_order(cancel_resp) if isinstance(cancel_resp, dict) else str(cancel_resp),
                            )
                        except Exception as exc:
                            cancel_errors.append(str(exc))
                            _log_rest_debug("limit_stop_abort_cancel", exc)
                            logger.warning(
                                "[STOP-ABORT/%s/%s] cancel attempt %s/%s failed order_id=%s err=%s",
                                exchange,
                                symbol,
                                cancel_attempt,
                                cancel_attempts,
                                exchange_order_id,
                                exc,
                            )

                        if callable(getattr(client, "fetch_order", None)):
                            for fetch_attempt in range(1, fetch_attempts + 1):
                                snapshot = None
                                try:
                                    snapshot = client.fetch_order(exchange_order_id, symbol, params={})
                                except Exception as exc:
                                    _log_rest_debug("limit_stop_abort_fetch_order", exc)
                                    logger.warning(
                                        "[STOP-ABORT/%s/%s] fetch attempt %s/%s failed order_id=%s err=%s",
                                        exchange,
                                        symbol,
                                        fetch_attempt,
                                        fetch_attempts,
                                        exchange_order_id,
                                        exc,
                                    )

                                if isinstance(snapshot, dict):
                                    last_snapshot = snapshot
                                    filled_qty = _extract_filled_qty(snapshot)
                                    if _is_filled(snapshot) or filled_qty > 0:
                                        return {
                                            "state": "filled",
                                            "order": snapshot,
                                            "filled_qty": filled_qty,
                                            "cancel_errors": cancel_errors,
                                        }
                                    if _is_cancelled_or_closed_status(snapshot):
                                        return {
                                            "state": "canceled",
                                            "order": snapshot,
                                            "filled_qty": filled_qty,
                                            "cancel_errors": cancel_errors,
                                        }

                                if fetch_attempt < fetch_attempts:
                                    await asyncio.sleep(fetch_retry_delay_s)

                        if cancel_attempt < cancel_attempts:
                            await asyncio.sleep(cancel_retry_delay_s)

                    final_filled_qty = _extract_filled_qty(last_snapshot or {})
                    if last_snapshot and (_is_filled(last_snapshot) or final_filled_qty > 0):
                        state = "filled"
                    elif last_snapshot and _is_cancelled_or_closed_status(last_snapshot):
                        state = "canceled"
                    elif last_snapshot:
                        state = "open"
                    else:
                        state = "unknown"

                    return {
                        "state": state,
                        "order": last_snapshot,
                        "filled_qty": final_filled_qty,
                        "cancel_errors": cancel_errors,
                    }

                while timeout_s > 0 and (time.time() - start_ts) < timeout_s:
                    # 1) stop-hit-before-entry check
                    try:
                        tick = client.ticker(symbol)
                        last_px = float((tick or {}).get("last") or 0.0)
                    except Exception:
                        last_px = 0.0

                    if last_px > 0:
                        if wait_window_max_px is None:
                            wait_window_max_px = float(last_px)
                        else:
                            wait_window_max_px = max(float(wait_window_max_px), float(last_px))
                        if wait_window_min_px is None:
                            wait_window_min_px = float(last_px)
                        else:
                            wait_window_min_px = min(float(wait_window_min_px), float(last_px))

                    if stop_price and stop_price > 0 and last_px > 0:
                        stop_hit_before_entry = (
                            (normalized_side == "buy" and last_px <= stop_price)
                            or (normalized_side == "sell" and last_px >= stop_price)
                        )
                        if stop_hit_before_entry:
                            stop_abort_diag = await _cancel_with_stop_abort_verification()
                            stop_abort_state = str(stop_abort_diag.get("state") or "unknown")
                            stop_abort_order = stop_abort_diag.get("order") or {}
                            stop_abort_filled_qty = float(stop_abort_diag.get("filled_qty") or 0.0)

                            if stop_abort_state == "filled":
                                avg_fill_price = stop_abort_order.get("average") or stop_abort_order.get("price") or limit_price
                                filled_amount = stop_abort_order.get("filled") or stop_abort_filled_qty or stop_abort_order.get("amount") or amount
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
                                    'status': (stop_abort_order.get("status") or OrderStatus.FILLED.value),
                                    'created_at': datetime.now(timezone.utc),
                                    'filled_amount': filled_amount,
                                    'avg_fill_price': avg_fill_price,
                                    'filled_at': datetime.now(timezone.utc),
                                    'slippage': slippage,
                                    'ccxt_params': ccxt_params,
                                    'exchange_order': stop_abort_order if isinstance(stop_abort_order, dict) else {},
                                }
                                self.active_orders[exchange_order_id] = order
                                return {
                                    'success': True,
                                    'reason': 'FILLED_DURING_STOP_ABORT',
                                    'order_id': exchange_order_id,
                                    'filled_amount': filled_amount,
                                    'avg_price': avg_fill_price,
                                    'slippage': slippage,
                                    'order': order,
                                }

                            if stop_abort_state == "canceled":
                                return {
                                    'success': False,
                                    'reason': 'ABORT:STOP_HIT_BEFORE_ENTRY',
                                    'order_id': exchange_order_id,
                                    'stop_abort_cancel_state': stop_abort_state,
                                }

                            return {
                                'success': False,
                                'reason': 'ABORT:STOP_HIT_BEFORE_ENTRY_CANCEL_UNCONFIRMED',
                                'order_id': exchange_order_id,
                                'stop_abort_cancel_state': stop_abort_state,
                            }

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
                tick = {}
                try:
                    tick = client.ticker(symbol)
                    current_px = float((tick or {}).get("last") or 0.0)
                except Exception:
                    current_px = 0.0
                if current_px > 0:
                    if wait_window_max_px is None:
                        wait_window_max_px = float(current_px)
                    else:
                        wait_window_max_px = max(float(wait_window_max_px), float(current_px))
                    if wait_window_min_px is None:
                        wait_window_min_px = float(current_px)
                    else:
                        wait_window_min_px = min(float(wait_window_min_px), float(current_px))

                ref_px = None
                for k in ("entry_raw", "entry", "execution_price"):
                    if k in signal:
                        try:
                            ref_px = float(signal.get(k) or 0.0)
                        except Exception:
                            ref_px = None
                        break

                legacy_gate_bps = execution_params.get("max_chase_bps")
                if legacy_gate_bps is None:
                    legacy_gate_bps = (
                        execution_params.get("max_chase_bps_long")
                        if normalized_side == "buy"
                        else execution_params.get("max_chase_bps_short")
                    )
                try:
                    legacy_gate_bps = None if legacy_gate_bps is None else float(legacy_gate_bps)
                except Exception:
                    legacy_gate_bps = None

                lock_key = self._fallback_lock_key(
                    exchange=exchange,
                    symbol=symbol,
                    side=normalized_side,
                    ccxt_params=ccxt_params,
                )
                fallback_lock = self._get_fallback_lock(lock_key)
                async with fallback_lock:
                    # Cancel the resting limit order (best-effort) before any fallback.
                    cancel_status = None
                    cancel_err = None
                    try:
                        cancel_result = client.cancel_order(exchange_order_id, symbol, params={})
                        if isinstance(cancel_result, dict):
                            cancel_status = cancel_result.get("status")
                    except Exception as exc:
                        cancel_err = str(exc)

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

                position_delta_verified = (
                    baseline_position_qty is not None
                    and current_position_qty is not None
                )
                cancel_terminal = self._is_terminal_order_status(cancel_status)
                post_cancel_terminal = self._is_terminal_order_status((post_cancel or {}).get("status"))
                verification_sources: List[str] = []
                if position_delta_verified:
                    verification_sources.append("position_delta")
                if post_cancel_filled_qty > 0.0:
                    verification_sources.append("post_cancel_filled")
                if cancel_terminal:
                    verification_sources.append("cancel_terminal")
                if post_cancel_terminal:
                    verification_sources.append("post_cancel_terminal")

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
                        'reason_code': 'execution.fallback.limit_timeout.skip_market_observed_fill',
                        'fallback_reason': 'limit_timeout_skip_market_position_delta',
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                        'baseline_position_qty': baseline_position_qty,
                        'current_position_qty': current_position_qty,
                        'verification_sources': verification_sources,
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
                        'reason_code': 'execution.fallback.limit_timeout.disabled',
                        'order_id': exchange_order_id,
                        'fallback_reason': (
                            f"limit_timeout_market_fallback_disabled:{fallback_block_reason}"
                            if fallback_block_reason
                            else "limit_timeout_market_fallback_disabled:flag"
                        ),
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                        'verification_sources': verification_sources,
                    }

                if (
                    bool(require_position_delta_verification)
                    and (not reduce_only)
                    and not position_delta_verified
                    and post_cancel_filled_qty <= 0.0
                ):
                    logger.warning(
                        "[ORDER-FALLBACK] unverified timeout fallback aborted symbol=%s side=%s cancel_status=%s cancel_err=%s post_cancel_status=%s baseline_qty=%s current_qty=%s",
                        symbol,
                        side,
                        cancel_status,
                        cancel_err or "n/a",
                        (post_cancel or {}).get("status"),
                        baseline_position_qty,
                        current_position_qty,
                    )
                    return {
                        'success': False,
                        'reason': 'ABORT:NO_FILL_TIMEOUT_UNVERIFIED',
                        'reason_code': 'execution.fallback.limit_timeout.unverified_position_delta',
                        'order_id': exchange_order_id,
                        'fallback_reason': 'limit_timeout_market_fallback_unverified:position_delta',
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                        'baseline_position_qty': baseline_position_qty,
                        'current_position_qty': current_position_qty,
                        'verification_sources': verification_sources,
                    }

                if remaining_qty <= min_residual_qty:
                    return {
                        'success': False,
                        'reason': 'ABORT:NO_FILL_TIMEOUT',
                        'reason_code': 'execution.fallback.limit_timeout.no_residual_qty',
                        'order_id': exchange_order_id,
                        'fallback_reason': 'limit_timeout_market_fallback_skipped_no_residual',
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                        'verification_sources': verification_sources,
                    }

                hard_chase_cfg = self._collect_fallback_hard_chase_cfg(
                    order_request=order_request,
                    execution_params=(execution_params if isinstance(execution_params, dict) else {}),
                    legacy_gate_bps=legacy_gate_bps,
                )
                hard_chase_eval = self._evaluate_fallback_hard_chase(
                    normalized_side=normalized_side,
                    current_px=float(current_px or 0.0),
                    ref_px=ref_px,
                    tick=(tick if isinstance(tick, dict) else {}),
                    signal=(signal if isinstance(signal, dict) else {}),
                    cfg=hard_chase_cfg,
                )
                if bool(hard_chase_eval.get("enabled")) and not bool(hard_chase_eval.get("allow", True)):
                    chase_bps = self._safe_float(hard_chase_eval.get("chase_bps"))
                    kill_bps = self._safe_float(hard_chase_eval.get("kill_bps"))
                    logger.info(
                        "[ORDER-FALLBACK] hard-chase killed symbol=%s side=%s chase_bps=%s kill_bps=%s cfg=%s",
                        symbol,
                        side,
                        f"{chase_bps:.2f}" if chase_bps is not None else "n/a",
                        f"{kill_bps:.2f}" if kill_bps is not None else "n/a",
                        hard_chase_eval.get("config"),
                    )
                    return {
                        'success': False,
                        'reason': (
                            f"ABORT:HARD_CHASE_KILL:{chase_bps:.2f}>{kill_bps:.2f}"
                            if chase_bps is not None and kill_bps is not None
                            else "ABORT:HARD_CHASE_KILL"
                        ),
                        'reason_code': 'execution.fallback.limit_timeout.hard_chase_killed',
                        'order_id': exchange_order_id,
                        'fallback_reason': 'limit_timeout_market_fallback_hard_chase_killed',
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                        'verification_sources': verification_sources,
                        'fallback_hard_chase': hard_chase_eval,
                    }

                soft_gate_cfg = self._collect_fallback_soft_gate_cfg(
                    order_request=order_request,
                    execution_params=(execution_params if isinstance(execution_params, dict) else {}),
                )
                soft_gate_eval = self._evaluate_fallback_soft_gate(
                    normalized_side=normalized_side,
                    signal=(signal if isinstance(signal, dict) else {}),
                    current_px=float(current_px or 0.0),
                    ref_px=ref_px,
                    tick=(tick if isinstance(tick, dict) else {}),
                    wait_window_max_px=wait_window_max_px,
                    wait_window_min_px=wait_window_min_px,
                    cfg=soft_gate_cfg,
                )
                if bool(soft_gate_eval.get("enabled")) and not bool(soft_gate_eval.get("allow", True)):
                    logger.info(
                        "[ORDER-FALLBACK] soft-gate blocked symbol=%s side=%s score=%s reason=%s passes=%s fails=%s na=%s",
                        symbol,
                        side,
                        soft_gate_eval.get("score"),
                        soft_gate_eval.get("reason"),
                        soft_gate_eval.get("passes"),
                        soft_gate_eval.get("fails"),
                        soft_gate_eval.get("na"),
                    )
                    return {
                        'success': False,
                        'reason': f"ABORT:FALLBACK_SOFT_GATE:{soft_gate_eval.get('reason', 'blocked')}",
                        'reason_code': 'execution.fallback.limit_timeout.soft_gate_blocked',
                        'order_id': exchange_order_id,
                        'fallback_reason': 'limit_timeout_market_fallback_soft_gate_blocked',
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                        'verification_sources': verification_sources,
                        'fallback_soft_gate': soft_gate_eval,
                    }

                slippage_guard_cfg = self._collect_fallback_slippage_guard_cfg(
                    order_request=order_request,
                    execution_params=(execution_params if isinstance(execution_params, dict) else {}),
                )
                slippage_guard_eval = self._evaluate_fallback_slippage_guard(
                    normalized_side=normalized_side,
                    signal=(signal if isinstance(signal, dict) else {}),
                    tick=(tick if isinstance(tick, dict) else {}),
                    current_px=float(current_px or 0.0),
                    entry_reference_px=ref_px,
                    limit_price=limit_price,
                    cfg=slippage_guard_cfg,
                )
                if bool(slippage_guard_eval.get("enabled")) and not bool(slippage_guard_eval.get("allow", True)):
                    adverse_bps = self._safe_float(slippage_guard_eval.get("adverse_bps"))
                    kill_bps = self._safe_float(slippage_guard_eval.get("kill_bps"))
                    logger.info(
                        "[ORDER-FALLBACK] slippage-guard blocked symbol=%s side=%s adverse_bps=%s kill_bps=%s reason=%s quote=%s",
                        symbol,
                        side,
                        f"{adverse_bps:.2f}" if adverse_bps is not None else "n/a",
                        f"{kill_bps:.2f}" if kill_bps is not None else "n/a",
                        slippage_guard_eval.get("reason"),
                        slippage_guard_eval.get("quote_source") or "n/a",
                    )
                    return {
                        'success': False,
                        'reason': f"ABORT:FALLBACK_SLIPPAGE_GUARD:{slippage_guard_eval.get('reason', 'blocked')}",
                        'reason_code': 'execution.fallback.limit_timeout.slippage_guard_blocked',
                        'order_id': exchange_order_id,
                        'fallback_reason': 'limit_timeout_market_fallback_slippage_guard_blocked',
                        'requested_order_type': 'limit',
                        'effective_order_type': 'limit',
                        'verification_sources': verification_sources,
                        'fallback_slippage_guard': slippage_guard_eval,
                        'fallback_hard_chase': hard_chase_eval if bool(hard_chase_eval.get("enabled")) else None,
                        'fallback_soft_gate': soft_gate_eval if bool(soft_gate_eval.get("enabled")) else None,
                    }

                hard_chase_bps = self._safe_float(hard_chase_eval.get("chase_bps")) if isinstance(hard_chase_eval, dict) else None
                hard_chase_kill_bps = self._safe_float(hard_chase_eval.get("kill_bps")) if isinstance(hard_chase_eval, dict) else None
                # Place market order as fallback only for residual amount.
                logger.info(
                    "[ORDER-FALLBACK] limit_timeout_market_fallback symbol=%s side=%s chase_bps=%s hard_kill_bps=%s residual_qty=%.8f observed_filled=%.8f",
                    symbol,
                    side,
                    f"{hard_chase_bps:.2f}" if hard_chase_bps is not None else "n/a",
                    f"{hard_chase_kill_bps:.2f}" if hard_chase_kill_bps is not None else "n/a",
                    float(remaining_qty),
                    float(observed_filled_qty),
                )
                fallback_order_request = dict(order_request)
                fallback_order_request['amount'] = float(remaining_qty)
                fallback_result = await self._market_order_execution(fallback_order_request, clients_to_use)
                if isinstance(fallback_result, dict):
                    fallback_result.setdefault("fallback_reason", "limit_timeout_market_fallback")
                    if bool(fallback_result.get("success")):
                        fallback_result.setdefault("reason_code", "execution.fallback.limit_timeout.market_fallback")
                    else:
                        fallback_result.setdefault("reason_code", "execution.fallback.limit_timeout.market_fallback_failed")
                    fallback_result.setdefault("requested_order_type", "limit")
                    fallback_result.setdefault("effective_order_type", "market")
                    fallback_result.setdefault("fallback_residual_qty", float(remaining_qty))
                    fallback_result.setdefault("observed_filled_qty", float(observed_filled_qty))
                    fallback_result.setdefault("baseline_position_qty", baseline_position_qty)
                    fallback_result.setdefault("current_position_qty", current_position_qty)
                    fallback_result.setdefault("verification_sources", verification_sources)
                    if hard_chase_bps is not None:
                        fallback_result.setdefault("deviation_bps", float(hard_chase_bps))
                        fallback_result.setdefault("chase_bps", float(hard_chase_bps))
                    if hard_chase_kill_bps is not None:
                        fallback_result.setdefault("max_chase_bps", float(hard_chase_kill_bps))
                        fallback_result.setdefault("hard_chase_kill_bps", float(hard_chase_kill_bps))
                    if bool(hard_chase_eval.get("enabled")):
                        fallback_result.setdefault("fallback_hard_chase", hard_chase_eval)
                    if bool(soft_gate_eval.get("enabled")):
                        fallback_result.setdefault("fallback_soft_gate", soft_gate_eval)
                    if bool(slippage_guard_eval.get("enabled")):
                        fallback_result.setdefault("fallback_slippage_guard", slippage_guard_eval)
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
