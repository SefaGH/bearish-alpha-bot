"""
Strategy Coordination Engine.
Coordinates signals and positions across multiple strategies.
"""

import asyncio
import heapq
import itertools
import logging
import math
import os
import re
import threading
import time
import json
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timezone, timedelta
from collections import defaultdict, OrderedDict
from enum import Enum
from pathlib import Path

import numpy as np
from copy import deepcopy

from src.quality.quality_calculator import compute_quality
from core.volume_analyzer import VolumeAnalyzer
from src.safety.trend_guard import TrendGuard
from src.safety.safety_override import SafetyOverride
from src.safety.signal_integrity_guard import SignalIntegrityGuard
from src.safety.regime_filter import RegimeFilter
from src.core.transition_policy import PositionTransitionPolicy
from src.core.interfaces import PositionSizingProtocol
from src.utils.volume_utils import get_bucket_rank
from core.directional_bias import compute_directional_bias_adjustment
from core.logger import get_current_run_id
from src.core.signal_intents import (
    INTENT_ENTRY,
    INTENT_REENTRY,
    INTENT_SCALE_IN,
    INTENT_CLOSE,
    MAINTENANCE_INTENTS,
    INTENT_FORCE_SWAP,
    INTENT_REVERSE,
)
INTENT_HOLD = "hold"

try:  # Optional dependency; lazily initialized when available
    from ml.adapters.ppo_trading_adapter import PPOTradingAdapter
except Exception:  # pragma: no cover - optional runtime dependency
    PPOTradingAdapter = None

logger = logging.getLogger(__name__)

# Constants for signal enrichment
DEFAULT_RL_CONFIDENCE = 0.7  # Default confidence when RL agent doesn't provide it
VOLUME_NORMALIZATION_MAX = 2.0  # Maximum volume ratio before normalization
VOLUME_NORMALIZATION_DIVISOR = 2.0  # Divisor for volume strength normalization
MOMENTUM_PRICE_CHANGE_OFFSET = 0.1  # Offset for momentum calculation
MOMENTUM_PRICE_CHANGE_RANGE = 0.2  # Range for momentum normalization
LOW_VOL_TIGHT_STOP_THRESHOLD = 0.0015  # 0.15%
LOW_VOL_WIDE_STOP_THRESHOLD = 0.005  # 0.50%
LOW_VOL_MICRO_GATE_MARGIN_BPS = 5.0


class SignalPriority(Enum):
    """Signal priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    HIGHEST_PRIORITY = 'highest_priority'
    BEST_RISK_REWARD = 'best_risk_reward'
    PERFORMANCE_WEIGHTED = 'performance_weighted'
    FIRST_IN_FIRST_OUT = 'fifo'


class PrioritySignalQueue:
    """Priority-based signal queue with TTL and per-symbol throttling."""

    def __init__(self, queue_config: Dict[str, Any], logger: logging.Logger):
        self._ttl = max(int(queue_config.get('ttl_seconds', 60)), 5)
        self._max_depth = max(int(queue_config.get('max_queue_depth', 50)), 1)
        self._max_pending_per_symbol = max(int(queue_config.get('max_pending_per_symbol', 1)), 0)
        self._max_pending_scale_in_per_symbol = max(int(queue_config.get('max_pending_scale_in_per_symbol', 0)), 0)
        self._pyramiding_enabled = bool(queue_config.get('pyramiding_enabled', False))
        default_weights = {
            'explicit_priority': 0.35,
            'risk_reward': 0.25,
            'ml_confidence': 0.2,
            'urgency': 0.1,
            'regime_alignment': 0.05,
            'strategy_urgency': 0.05,
        }
        provided_weights = queue_config.get('priority_weights') or {}
        self._weights = {**default_weights, **provided_weights}

        self._condition = asyncio.Condition()
        self._queue: List[Tuple[float, float, int, Dict[str, Any]]] = []
        self._waiting_room: List[Tuple[float, Tuple[float, float, int, Dict[str, Any]]]] = []
        self._sequence = itertools.count()
        self._pending_by_symbol: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "scale_in": 0})
        self._logger = logger
        self._stats = {
            'accepted': 0,
            'dequeued': 0,
            'expired': 0,
            'rejected_capacity': 0,
            'rejected_symbol_limit': 0,
            'requeued': 0,
        }
        self.on_expire: Optional[Callable[[Dict[str, Any]], None]] = None

    async def put(
        self, payload: Dict[str, Any], process_after: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        symbol = self._extract_symbol(payload)
        intent = self._extract_intent(payload)
        expired_payloads: List[Dict[str, Any]] = []
        ok = False
        reason: Optional[str] = None
        reason_code: Optional[str] = None
        async with self._condition:
            now = time.time()
            expired_payloads = self._purge_expired_locked()
            if symbol:
                pending_totals = self._pending_by_symbol[symbol]
                pending_total = pending_totals["total"]
                pending_scale = pending_totals["scale_in"]

                if not self._pyramiding_enabled:
                    if self._max_pending_per_symbol and pending_total >= self._max_pending_per_symbol:
                        self._stats['rejected_symbol_limit'] += 1
                        reason = f"Queue limit reached for {symbol}"
                        reason_code = "queue.symbol_pending_limit"
                        self._logger.warning(f"🚫 [QUEUE] {reason}")
                else:
                    if intent == INTENT_SCALE_IN:
                        max_allowed = self._max_pending_per_symbol + self._max_pending_scale_in_per_symbol
                        if self._max_pending_scale_in_per_symbol <= 0:
                            max_allowed = self._max_pending_per_symbol
                        if pending_total >= max_allowed or pending_scale >= self._max_pending_scale_in_per_symbol > 0:
                            self._stats['rejected_symbol_limit'] += 1
                            reason = "Scale-in queue limit reached"
                            reason_code = "queue.symbol_pending_limit"
                            self._logger.info(
                                "[PYRAMID-QUEUE] scale-in rejected at enqueue | sym=%s | pending_total=%d | pending_scale_in=%d | max_entry=%d | max_scale_in=%d",
                                symbol,
                                pending_total,
                                pending_scale,
                                self._max_pending_per_symbol,
                                self._max_pending_scale_in_per_symbol,
                            )
                    else:
                        if self._max_pending_per_symbol and pending_total >= self._max_pending_per_symbol:
                            self._stats['rejected_symbol_limit'] += 1
                            reason = f"Queue limit reached for {symbol}"
                            reason_code = "queue.symbol_pending_limit"
                            self._logger.warning(f"🚫 [QUEUE] {reason}")

            if reason is None:
                meta = payload.setdefault('queue_meta', {})
                meta['enqueued_at'] = now
                meta['expiration'] = now + self._ttl
                if process_after is not None:
                    meta['process_after'] = process_after

                priority_score = self._compute_priority(payload, now)
                entry = (-priority_score, meta['enqueued_at'], next(self._sequence), payload)
                current_depth = len(self._queue) + len(self._waiting_room)
                defer_signal = process_after is not None and process_after > now

                if current_depth >= self._max_depth:
                    replaced = self._maybe_replace_lowest(entry, priority_score, process_after)
                    if not replaced:
                        self._stats['rejected_capacity'] += 1
                        reason = "Signal queue at capacity"
                        reason_code = "queue.capacity"
                        self._logger.warning(f"🚫 [QUEUE] {reason} (score={priority_score:.3f})")
                    else:
                        ok = True
                else:
                    if defer_signal:
                        self._waiting_room.append((process_after, entry))
                    else:
                        heapq.heappush(self._queue, entry)
                    ok = True

                if ok:
                    if symbol:
                        self._pending_by_symbol[symbol]["total"] += 1
                        if intent == INTENT_SCALE_IN:
                            self._pending_by_symbol[symbol]["scale_in"] += 1
                    self._stats['accepted'] += 1
                    self._condition.notify()
                    self._logger.info(
                        f"📥 [QUEUE] Signal enqueued: symbol={symbol}, score={priority_score:.3f}, depth={len(self._queue) + len(self._waiting_room)}"
                    )

        self._notify_expired(expired_payloads)
        return ok, reason, reason_code

    async def can_accept(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """Cheap acceptance check (no mutation) for incubator gating."""
        symbol = self._extract_symbol(payload)
        intent = self._extract_intent(payload)
        expired_payloads: List[Dict[str, Any]] = []
        ok = True
        reason: Optional[str] = None
        reason_code: Optional[str] = None
        async with self._condition:
            now = time.time()
            expired_payloads = self._purge_expired_locked()
            self._check_waiting_room_locked()

            if symbol:
                pending_totals = self._pending_by_symbol[symbol]
                pending_total = pending_totals["total"]
                pending_scale = pending_totals["scale_in"]

                if not self._pyramiding_enabled:
                    if self._max_pending_per_symbol and pending_total >= self._max_pending_per_symbol:
                        ok = False
                        reason = f"Queue limit reached for {symbol}"
                        reason_code = "queue.symbol_pending_limit"
                else:
                    if intent == INTENT_SCALE_IN:
                        max_allowed = self._max_pending_per_symbol + self._max_pending_scale_in_per_symbol
                        if self._max_pending_scale_in_per_symbol <= 0:
                            max_allowed = self._max_pending_per_symbol
                        if pending_total >= max_allowed or pending_scale >= self._max_pending_scale_in_per_symbol > 0:
                            ok = False
                            reason = "Scale-in queue limit reached"
                            reason_code = "queue.symbol_pending_limit"
                    else:
                        if self._max_pending_per_symbol and pending_total >= self._max_pending_per_symbol:
                            ok = False
                            reason = f"Queue limit reached for {symbol}"
                            reason_code = "queue.symbol_pending_limit"

            if ok:
                current_depth = len(self._queue) + len(self._waiting_room)
                if current_depth >= self._max_depth:
                    ok = False
                    reason = "Signal queue at capacity"
                    reason_code = "queue.capacity"

        self._notify_expired(expired_payloads)
        return ok, reason, reason_code

    async def get(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        expired_payloads: List[Dict[str, Any]] = []
        result: Optional[Dict[str, Any]] = None
        timed_out = False
        async with self._condition:
            deadline = None
            loop = None
            if timeout is not None:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + timeout

            while True:
                expired_payloads.extend(self._purge_expired_locked())
                self._check_waiting_room_locked()
                self._refresh_priorities_locked()

                if self._queue:
                    entry = heapq.heappop(self._queue)
                    payload = entry[3]
                    symbol = self._extract_symbol(payload)
                    if symbol:
                        counts = self._pending_by_symbol.get(symbol, {"total": 0, "scale_in": 0})
                        if counts.get("total", 0) > 0:
                            counts["total"] = max(0, counts.get("total", 0) - 1)
                            if self._extract_intent(payload) == INTENT_SCALE_IN:
                                counts["scale_in"] = max(0, counts.get("scale_in", 0) - 1)
                            self._pending_by_symbol[symbol] = counts
                    payload.setdefault('queue_meta', {})['dequeued_at'] = time.time()
                    self._stats['dequeued'] += 1
                    result = payload
                    break

                if timeout is None:
                    await self._condition.wait()
                else:
                    loop = loop or asyncio.get_running_loop()
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        timed_out = True
                        break

        self._notify_expired(expired_payloads)
        if timed_out:
            raise asyncio.TimeoutError()
        if result is None:  # pragma: no cover - defensive
            raise RuntimeError("PrioritySignalQueue.get exited without a payload")
        return result

    def qsize(self) -> int:
        return len(self._queue) + len(self._waiting_room)

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    async def requeue(self, payload: Dict[str, Any]) -> None:
        symbol = self._extract_symbol(payload)
        expired_payloads: List[Dict[str, Any]] = []
        async with self._condition:
            now = time.time()
            expired_payloads = self._purge_expired_locked()
            meta = payload.setdefault('queue_meta', {})
            meta.setdefault('enqueued_at', now)
            meta.setdefault('expiration', now + self._ttl)

            priority_score = self._compute_priority(payload, now)
            entry = (-priority_score, meta['enqueued_at'], next(self._sequence), payload)
            heapq.heappush(self._queue, entry)

            if symbol:
                self._pending_by_symbol[symbol]["total"] += 1
                if self._extract_intent(payload) == INTENT_SCALE_IN:
                    self._pending_by_symbol[symbol]["scale_in"] += 1

            self._stats['requeued'] = self._stats.get('requeued', 0) + 1
            self._condition.notify()
        self._notify_expired(expired_payloads)

    def _maybe_replace_lowest(self, entry, new_score: float, process_after: Optional[float]) -> bool:
        if not self._queue and not self._waiting_room:
            if process_after is not None and process_after > time.time():
                self._waiting_room.append((process_after, entry))
            else:
                heapq.heappush(self._queue, entry)
            return True

        lowest_location = None
        lowest_index = None
        lowest_entry: Optional[Tuple[float, float, int, Dict[str, Any]]] = None
        lowest_score: Optional[float] = None

        if self._queue:
            queue_lowest_index = max(range(len(self._queue)), key=lambda idx: self._queue[idx][0])
            queue_lowest_entry = self._queue[queue_lowest_index]
            queue_lowest_score = -queue_lowest_entry[0]
            lowest_location = "queue"
            lowest_index = queue_lowest_index
            lowest_entry = queue_lowest_entry
            lowest_score = queue_lowest_score

        if self._waiting_room:
            waiting_lowest_index = max(range(len(self._waiting_room)), key=lambda idx: self._waiting_room[idx][1][0])
            waiting_lowest_entry = self._waiting_room[waiting_lowest_index][1]
            waiting_lowest_score = -waiting_lowest_entry[0]
            if lowest_score is None or waiting_lowest_score < lowest_score:
                lowest_location = "waiting"
                lowest_index = waiting_lowest_index
                lowest_entry = waiting_lowest_entry
                lowest_score = waiting_lowest_score

        if lowest_score is None:
            return False

        if new_score <= lowest_score:
            return False

        removed_payload = lowest_entry[3]
        removed_symbol = self._extract_symbol(removed_payload)
        if removed_symbol:
            pending_counts = self._pending_by_symbol[removed_symbol]
            if pending_counts["total"] > 0:
                pending_counts["total"] -= 1
            if self._extract_intent(removed_payload) == INTENT_SCALE_IN and pending_counts["scale_in"] > 0:
                pending_counts["scale_in"] -= 1

        if lowest_location == "queue" and lowest_index is not None:
            self._queue.pop(lowest_index)
            heapq.heapify(self._queue)
        elif lowest_location == "waiting" and lowest_index is not None:
            self._waiting_room.pop(lowest_index)

        if process_after is not None and process_after > time.time():
            self._waiting_room.append((process_after, entry))
        else:
            heapq.heappush(self._queue, entry)

        self._logger.info(
            f"♻️ [QUEUE] Replaced low-priority signal (score={lowest_score:.3f}) with higher score {new_score:.3f}"
        )
        return True

    def _purge_expired_locked(self) -> List[Dict[str, Any]]:
        now = time.time()
        expired = 0
        expired_payloads: List[Dict[str, Any]] = []

        if self._queue:
            kept: List[Tuple[float, float, int, Dict[str, Any]]] = []
            while self._queue:
                entry = heapq.heappop(self._queue)
                payload = entry[3]
                expiration = payload.get('queue_meta', {}).get('expiration', now + 1)
                if expiration < now:
                    expired += 1
                    expired_payloads.append(payload)
                    symbol = self._extract_symbol(payload)
                    if symbol:
                        counts = self._pending_by_symbol[symbol]
                        if counts["total"] > 0:
                            counts["total"] -= 1
                        if self._extract_intent(payload) == INTENT_SCALE_IN and counts["scale_in"] > 0:
                            counts["scale_in"] -= 1
                else:
                    kept.append(entry)

            for entry in kept:
                heapq.heappush(self._queue, entry)

        if self._waiting_room:
            kept_waiting: List[Tuple[float, Tuple[float, float, int, Dict[str, Any]]]] = []
            for process_after, entry in self._waiting_room:
                payload = entry[3]
                expiration = payload.get('queue_meta', {}).get('expiration', now + 1)
                if expiration < now:
                    expired += 1
                    expired_payloads.append(payload)
                    symbol = self._extract_symbol(payload)
                    if symbol:
                        counts = self._pending_by_symbol[symbol]
                        if counts["total"] > 0:
                            counts["total"] -= 1
                        if self._extract_intent(payload) == INTENT_SCALE_IN and counts["scale_in"] > 0:
                            counts["scale_in"] -= 1
                else:
                    kept_waiting.append((process_after, entry))

            self._waiting_room = kept_waiting

        if expired:
            self._stats['expired'] += expired
            self._logger.warning(f"⏳ [QUEUE] Dropped {expired} expired signals")

        return expired_payloads

    def _notify_expired(self, expired_payloads: List[Dict[str, Any]]) -> None:
        if not expired_payloads or not self.on_expire:
            return
        on_expire = self.on_expire
        for payload in expired_payloads:
            try:
                on_expire(payload)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error("[QUEUE] on_expire callback failed: %s", exc, exc_info=True)

    def _refresh_priorities_locked(self) -> None:
        if not self._queue:
            return

        now = time.time()
        refreshed = [
            (-self._compute_priority(entry[3], now), entry[1], entry[2], entry[3])
            for entry in self._queue
        ]
        heapq.heapify(refreshed)
        self._queue = refreshed

    def _check_waiting_room_locked(self) -> None:
        if not self._waiting_room:
            return

        now = time.time()
        remaining: List[Tuple[float, Tuple[float, float, int, Dict[str, Any]]]] = []

        for process_after, entry in self._waiting_room:
            if now >= process_after:
                payload = entry[3]
                payload.setdefault('queue_meta', {})['is_deferred'] = True
                heapq.heappush(self._queue, entry)
            else:
                remaining.append((process_after, entry))

        self._waiting_room = remaining

    def _compute_priority(self, payload: Dict[str, Any], current_ts: float) -> float:
        signal = payload.get('signal', {}) or {}
        risk_assessment = payload.get('risk_assessment', {}) or {}
        metrics = risk_assessment.get('metrics', {}) or {}

        priority_value = signal.get('priority', SignalPriority.MEDIUM)
        if isinstance(priority_value, SignalPriority):
            priority_raw = priority_value.value
        else:
            try:
                priority_raw = float(priority_value)
            except (TypeError, ValueError):
                priority_raw = SignalPriority.MEDIUM.value
        priority_component = min(1.0, max(0.0, (priority_raw - 1) / 3))

        rr_ratio = signal.get('rr_ratio') or metrics.get('risk_reward_ratio') or 1.0
        try:
            rr_component = min(1.5, float(rr_ratio)) / 1.5
        except (TypeError, ValueError):
            rr_component = 0.5

        ml_conf = signal.get('ml_confidence') or signal.get('ml_price_confidence') or 0.5
        try:
            ml_component = min(max(float(ml_conf), 0.0), 1.0)
        except (TypeError, ValueError):
            ml_component = 0.5

        regime_value = signal.get('regime_weight')
        if regime_value is None:
            regime_value = signal.get('regime_confidence')
        try:
            regime_component = min(max(float(regime_value), 0.0), 1.0)
        except (TypeError, ValueError):
            regime_component = 0.5

        strategy_component_raw = signal.get('strategy_urgency')
        if strategy_component_raw is None:
            route_hint = signal.get('regime_route_hint')
            if isinstance(route_hint, dict):
                strategy_component_raw = route_hint.get('queue_priority_boost')
        try:
            strategy_component = min(max(float(strategy_component_raw), 0.0), 1.0)
        except (TypeError, ValueError):
            strategy_component = 0.5

        enqueued_at = payload.get('queue_meta', {}).get('enqueued_at', current_ts)
        age = max(0.0, current_ts - enqueued_at)
        urgency_component = min(1.0, age / self._ttl)

        score = (
            self._weights.get('explicit_priority', 0.4) * priority_component +
            self._weights.get('risk_reward', 0.3) * rr_component +
            self._weights.get('ml_confidence', 0.2) * ml_component +
            self._weights.get('urgency', 0.1) * urgency_component +
            self._weights.get('regime_alignment', 0.0) * regime_component +
            self._weights.get('strategy_urgency', 0.0) * strategy_component
        )
        return score

    @staticmethod
    def _extract_symbol(payload: Dict[str, Any]) -> Optional[str]:
        signal = payload.get('signal') or {}
        return signal.get('symbol')

    @staticmethod
    def _extract_intent(payload: Dict[str, Any]) -> Optional[str]:
        signal = payload.get('signal') or {}
        return signal.get('intent')


class StrategyCoordinator:
    """
    Coordinate signals and positions across multiple strategies.
    Enhanced with GEMMA AI-Gate (Phase 5).
    """
    
    # __init__ metodundan 'indicator_manager' kaldırıldı.
    def __init__(self, portfolio_manager, risk_manager, market_data_pipeline=None, config=None, **kwargs):
        """
        Initialize strategy coordinator.
    
        Args:
            portfolio_manager: PortfolioManager instance
            risk_manager: RiskManager instance
            market_data_pipeline: Optional MarketDataPipeline instance (injected by production coordinator)
            config: Optional configuration dictionary
            **kwargs: Catches any other future arguments for forward compatibility.
        """
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.market_data_pipeline = market_data_pipeline
        self.config = config or {}
        callback = kwargs.get("recheck_ready_callback")
        self._recheck_ready_callback = callback if callable(callback) else None
        self._initial_equity = self._derive_initial_equity()

        va_cfg = self.config.get('volume_analyzer') if isinstance(self.config, dict) else {}
        self._volume_analyzer_enabled = bool(va_cfg.get('enabled', True) if isinstance(va_cfg, dict) else True)
        # Volume analyzer (async) for dynamic volume context
        self.volume_analyzer = kwargs.get('volume_analyzer') if self._volume_analyzer_enabled else None
        if not self.volume_analyzer and self.market_data_pipeline and self._volume_analyzer_enabled:
            self.volume_analyzer = VolumeAnalyzer(self.market_data_pipeline, va_cfg or {})

        hb_cfg: Dict[str, Any] = {}
        try:
            hb_cfg = (va_cfg or {}).get("heartbeat", {}) if isinstance(va_cfg, dict) else {}
            if not isinstance(hb_cfg, dict):
                hb_cfg = {}
        except Exception:
            hb_cfg = {}

        self._volume_heartbeat_enabled = bool(hb_cfg.get("enabled", False))
        try:
            interval_sec = int(hb_cfg.get("interval_sec", 300) or 300)
        except Exception:
            interval_sec = 300
        self._volume_heartbeat_interval_sec = max(30, interval_sec)
        self._volume_heartbeat_trade_tf = str(
            hb_cfg.get("trade_tf")
            or hb_cfg.get("trade_timeframe")
            or hb_cfg.get("timeframe")
            or "5m"
        )
        try:
            self._volume_heartbeat_min_strength_delta = float(hb_cfg.get("min_strength_delta", 0.05) or 0.05)
        except Exception:
            self._volume_heartbeat_min_strength_delta = 0.05
        if not math.isfinite(self._volume_heartbeat_min_strength_delta) or self._volume_heartbeat_min_strength_delta < 0:
            self._volume_heartbeat_min_strength_delta = 0.05
        self._volume_heartbeat_include_debug_fields = bool(hb_cfg.get("include_debug_fields", False))
        self._volume_heartbeat_last_emit_ts: Dict[str, float] = {}
        self._volume_heartbeat_last_ctx: Dict[str, Dict[str, Any]] = {}
        self._volume_heartbeat_task: Optional[asyncio.Task] = None

        tg_cfg = self.config.get("trend_guard", {}) if isinstance(self.config, dict) else {}
        tg_enabled = bool(tg_cfg.get("enabled", False)) if isinstance(tg_cfg, dict) else False
        self.trend_guard = TrendGuard(tg_cfg) if tg_enabled else None
        self._trend_guard_veto_diag_enabled = bool(tg_cfg.get("veto_diag_enabled", False)) if isinstance(tg_cfg, dict) else False
        self._trend_guard_veto_diag_log_level = str(tg_cfg.get("veto_diag_log_level", "WARNING") or "WARNING").upper()
        try:
            self._trend_guard_veto_diag_throttle_s = float(tg_cfg.get("veto_diag_throttle_seconds", 60) or 60)
        except Exception:
            self._trend_guard_veto_diag_throttle_s = 60.0
        default_key_fields = ["symbol", "timeframe", "side", "reason"]
        key_fields = tg_cfg.get("veto_diag_key_fields") if isinstance(tg_cfg, dict) else None
        self._trend_guard_veto_diag_key_fields = list(key_fields) if isinstance(key_fields, (list, tuple)) else default_key_fields
        self._trend_guard_veto_diag_last_log: Dict[Tuple[Any, ...], float] = {}

        so_cfg = self.config.get("safety_override", {}) if isinstance(self.config, dict) else {}
        so_enabled = bool(so_cfg.get("enabled", False)) if isinstance(so_cfg, dict) else False
        self.safety_override = SafetyOverride(so_cfg) if so_enabled else None
        if self.safety_override:
            try:
                logger.info(
                    "[SAFETY-OVERRIDE] enabled=true apply_to_strategies=%s apply_to_sides=%s min_passes=%s delta_min=%s",
                    (so_cfg.get("apply_to_strategies") or []),
                    (so_cfg.get("apply_to_sides") or []),
                    so_cfg.get("min_passes"),
                    so_cfg.get("aggressive_threshold_delta_min"),
                )
            except Exception:
                logger.info("[SAFETY-OVERRIDE] enabled=true")

        self.integrity_guard = SignalIntegrityGuard(self.config, self.market_data_pipeline)
        self.regime_filter = RegimeFilter(self.config, self.market_data_pipeline)
        self.transition_policy = PositionTransitionPolicy(self.config)

        strategies_cfg = self.config.get('strategies', {}) or {}
        self.regime_routing_rules = strategies_cfg.get('regime_routing', {}) or {}
        self.regime_route_stats = {
            'evaluated': 0,
            'matched': 0,
            'unmatched': 0,
            'by_regime': defaultdict(int)
        }
        self.rr_telemetry = {
            'samples': 0,
            'avg_actual_rr': 0.0,
            'avg_target_rr': 0.0,
            'by_strategy': {}
        }
    
        # Signal management
        self.active_signals = {}  # signal_id -> signal_data
        queue_cfg = ((self.config.get('risk') or {}).get('queue')) or {}
        # Pass pyramiding flag to queue for intent-aware pending limits
        try:
            pyramiding_cfg = self.config.get("pyramiding", {}) if isinstance(self.config, dict) else {}
            queue_cfg = {**queue_cfg, "pyramiding_enabled": bool(pyramiding_cfg.get("enabled", False))}
        except Exception:
            pass
        self.signal_queue = PrioritySignalQueue(queue_cfg, logger)
        self.signal_queue.on_expire = self._on_signal_queue_expire
        self.strategy_recheck_queue: asyncio.Queue = asyncio.Queue()
        self.signal_history = []
        self._signal_history_lookup: Dict[str, Dict[str, Any]] = {}
        self._dispatch_lock = asyncio.Lock()
        self._dispatch_retry_delay = float(queue_cfg.get('dispatch_retry_delay', 0.2) or 0.2)
    
        # Conflict tracking
        self.conflict_history = []
    
        # Duplicate prevention tracking
        self.last_signal_time = {}  # "symbol:strategy" -> timestamp
        self.last_signal_rsi = {}  # "symbol:strategy" -> last observed RSI
        self.signal_price_history = defaultdict(list)  # symbol -> [(timestamp, price), ...]
        self._dca_last_signal_time = defaultdict(float)  # symbol -> ts
        self._dca_recent_layers = defaultdict(dict)  # symbol -> {layer_index: ts}
        self.rsi_session_state = {}  # symbol -> {'active': True, 'anchor_price': float, 'side': 'long'}

        # Strategy+symbol cooldowns (non-blocking deferral replacement)
        self._strategy_cooldowns: Dict[str, datetime] = {}
        self._strategy_cooldowns_lock = threading.Lock()

        # Stop-loss driven entry guards (Crash Guard cooldown/reversal-only modes)
        self._stop_loss_cooldown_streak: Dict[str, Dict[str, Any]] = {}
        self._stop_loss_reversal_required: Dict[str, float] = {}
         
        # Signal processing stats
        self.processing_stats = {
            'total_signals': 0,
            'accepted_signals': 0,
            'rejected_signals': 0,
            'conflicted_signals': 0,
            'duplicate_rejections': 0,
            'last_signal_time': None,
            'cooldown_bypasses': 0,
            'bypass_success_rate': 0.0,
            'avg_bypass_price_delta': 0.0,
            'last_bypass_time': None,
            'rejected_cooldown': 0,
            'rejected_price_delta': 0,
            'ai_gate_rejections': 0,  # Phase 5: GEMMA AI-Gate rejections
            'approved_signals': 0,  # Phase 5: Signals approved for execution
            'rl_veto_count': 0,
            'rl_skipped_signals': 0,
            'bypass_approvals': 0,
            'queue_rejections': 0
        }
        self.rl_telemetry = {
            'total_decisions': 0,
            'veto_count': 0,
            'bias_applied_count': 0,
            'bias_skipped_count': 0,
            'veto_by_regime': {
                'bullish': 0,
                'bearish': 0,
                'neutral': 0,
                'volatile': 0
            },
            'q_std_values': [],
            'q_range_values': [],
            'bypass_count': 0,
            'ppo': {
                'samples': 0,
                'long_votes': 0,
                'flat_votes': 0,
                'score_sum': 0.0
            }
        }
        
        # ML integration placeholders
        self.ml_integration = None
        self.feature_pipeline = None
        self.rl_agent = None
        rl_cfg = (self.config.get('ml', {}) or {}).get('reinforcement_learning', {})
        self._rl_config = rl_cfg
        self.legacy_rl_enabled = bool(rl_cfg.get('legacy_dqn_enabled', False))
        self.ppo_multipliers = {
            'rr_up_mult': float(rl_cfg.get('ppo_rr_up_mult', 1.3)),
            'rr_down_mult': float(rl_cfg.get('ppo_rr_down_mult', 0.9)),
            'position_base': float(rl_cfg.get('ppo_position_base', 0.5)),
            'position_bonus': float(rl_cfg.get('ppo_position_bonus', 0.5)),
        }
        self.ppo_fallback_score = float(rl_cfg.get('ppo_fallback_score', 0.5))
        self.ppo_adapter = None
        self._ppo_adapter_failed = False
        self._ppo_missing_dependency_logged = False
        
        # Track ML/RL rejection context for better telemetry
        self._last_ml_rejection_reason: Optional[str] = None

        # GEMMA Adapter initialization (Phase 5)
        self.gemma_adapter = None
        if self.config.get('ml', {}).get('gemma', {}).get('enabled', False):
            self._initialize_gemma()

        incubator_cfg = {}
        try:
            signals_cfg = self.config.get("signals", {}) if isinstance(self.config, dict) else {}
            incubator_cfg = signals_cfg.get("incubator", {}) if isinstance(signals_cfg, dict) else {}
            if not incubator_cfg and isinstance(self.config, dict):
                incubator_cfg = self.config.get("incubator", {}) or {}
        except Exception:
            incubator_cfg = {}

        self._incubator_enabled = bool(incubator_cfg.get("enabled", True))
        self._incubator_tick_max_items = int(incubator_cfg.get("tick_max_items", 25) or 25)
        self._incubator_tick_time_budget_ms = int(incubator_cfg.get("tick_time_budget_ms", 75) or 75)
        self._incubator_lock = asyncio.Lock()
        self._incubator_items: Dict[str, Dict[str, Any]] = {}
        self._incubator_policies = self._build_incubator_policies(incubator_cfg)

        fast_watch_cfg = self.config.get("fast_watch", {}) if isinstance(self.config, dict) else {}
        self._fast_watch_enabled = bool(fast_watch_cfg.get("enabled", False))
        try:
            self._fast_watch_interval_ms = int(fast_watch_cfg.get("interval_ms", 3000) or 3000)
        except Exception:
            self._fast_watch_interval_ms = 3000
        self._fast_watch_interval_ms = max(250, self._fast_watch_interval_ms)
        try:
            self._fast_watch_max_items_per_tick = int(fast_watch_cfg.get("max_items_per_tick", 50) or 50)
        except Exception:
            self._fast_watch_max_items_per_tick = 50
        self._fast_watch_max_items_per_tick = max(1, self._fast_watch_max_items_per_tick)
        try:
            self._fast_watch_time_budget_ms = int(fast_watch_cfg.get("time_budget_ms", 75) or 75)
        except Exception:
            self._fast_watch_time_budget_ms = 75
        self._fast_watch_time_budget_ms = max(10, self._fast_watch_time_budget_ms)
        try:
            self._fast_watch_max_checks_default = int(fast_watch_cfg.get("max_checks_default", 9) or 9)
        except Exception:
            self._fast_watch_max_checks_default = 9
        self._fast_watch_max_checks_default = max(1, self._fast_watch_max_checks_default)
        try:
            self._fast_watch_ttl_ms_default = int(fast_watch_cfg.get("ttl_ms_default", 30000) or 30000)
        except Exception:
            self._fast_watch_ttl_ms_default = 30000
        self._fast_watch_ttl_ms_default = max(1000, self._fast_watch_ttl_ms_default)
        try:
            self._fast_watch_check_sample_rate = int(fast_watch_cfg.get("check_sample_rate", 20) or 20)
        except Exception:
            self._fast_watch_check_sample_rate = 20
        self._fast_watch_check_sample_rate = max(0, self._fast_watch_check_sample_rate)
        try:
            self._fast_watch_max_rearms = int(fast_watch_cfg.get("max_rearms", 1) or 1)
        except Exception:
            self._fast_watch_max_rearms = 1
        self._fast_watch_max_rearms = max(0, self._fast_watch_max_rearms)
        try:
            self._fast_watch_rearm_backoff_mult = float(fast_watch_cfg.get("rearm_backoff_mult", 2.0) or 2.0)
        except Exception:
            self._fast_watch_rearm_backoff_mult = 2.0
        if not math.isfinite(self._fast_watch_rearm_backoff_mult) or self._fast_watch_rearm_backoff_mult <= 0:
            self._fast_watch_rearm_backoff_mult = 2.0
        try:
            self._fast_watch_rearm_max_interval_ms = int(
                fast_watch_cfg.get("rearm_max_interval_ms", self._fast_watch_interval_ms * 4)
                or self._fast_watch_interval_ms * 4
            )
        except Exception:
            self._fast_watch_rearm_max_interval_ms = self._fast_watch_interval_ms * 4
        self._fast_watch_rearm_max_interval_ms = max(self._fast_watch_interval_ms, self._fast_watch_rearm_max_interval_ms)
        self._fast_watch_check_counter = 0
        self._fast_watch_task: Optional[asyncio.Task] = None

        micro_watch_cfg = self.config.get("micro_gate_watch", {}) if isinstance(self.config, dict) else {}
        self._micro_gate_watch_enabled = bool(micro_watch_cfg.get("enabled", True))
        try:
            self._micro_gate_watch_loop_interval_ms = int(micro_watch_cfg.get("loop_interval_ms", 1000) or 1000)
        except Exception:
            self._micro_gate_watch_loop_interval_ms = 1000
        self._micro_gate_watch_loop_interval_ms = max(250, self._micro_gate_watch_loop_interval_ms)
        try:
            self._micro_gate_watch_max_items_per_tick = int(micro_watch_cfg.get("max_items_per_tick", 50) or 50)
        except Exception:
            self._micro_gate_watch_max_items_per_tick = 50
        self._micro_gate_watch_max_items_per_tick = max(1, self._micro_gate_watch_max_items_per_tick)
        try:
            self._micro_gate_watch_time_budget_ms = int(micro_watch_cfg.get("time_budget_ms", 75) or 75)
        except Exception:
            self._micro_gate_watch_time_budget_ms = 75
        self._micro_gate_watch_time_budget_ms = max(10, self._micro_gate_watch_time_budget_ms)
        try:
            self._micro_gate_watch_interval_ms_default = int(
                micro_watch_cfg.get("watch_interval_ms_default", 10_000) or 10_000
            )
        except Exception:
            self._micro_gate_watch_interval_ms_default = 10_000
        self._micro_gate_watch_interval_ms_default = max(250, self._micro_gate_watch_interval_ms_default)
        try:
            self._micro_gate_watch_max_checks_default = int(micro_watch_cfg.get("max_checks_default", 2) or 2)
        except Exception:
            self._micro_gate_watch_max_checks_default = 2
        self._micro_gate_watch_max_checks_default = max(1, self._micro_gate_watch_max_checks_default)
        try:
            self._micro_gate_watch_ttl_ms_default = int(micro_watch_cfg.get("ttl_ms_default", 25_000) or 25_000)
        except Exception:
            self._micro_gate_watch_ttl_ms_default = 25_000
        self._micro_gate_watch_ttl_ms_default = max(1_000, self._micro_gate_watch_ttl_ms_default)
        self._micro_gate_watch_task: Optional[asyncio.Task] = None

        # Dedupe registry: prevents duplicate orders/setup collisions between incubator + active pipeline
        self._active_dedupe_by_key: Dict[str, str] = {}
        self._active_dedupe_by_signal_id: Dict[str, str] = {}

        logger.info(
            "StrategyCoordinator initialized (market_data_pipeline=%s incubator=%s)",
            bool(self.market_data_pipeline),
            self._incubator_enabled,
        )

    def _set_strategy_cooldown(
        self,
        strategy_name: str,
        symbol: str,
        duration_seconds: float,
        side: Optional[str] = None,
    ) -> None:
        """Activate a cooldown for a specific strategy+symbol(+side) key."""
        if not strategy_name or not symbol:
            return
        try:
            duration = max(0.0, float(duration_seconds))
        except (TypeError, ValueError):
            duration = 0.0
        expiry_time = datetime.now(timezone.utc) + timedelta(seconds=duration)
        side_norm = self._normalize_side(side) if side else None
        key = f"{strategy_name}:{symbol}:{side_norm}" if side_norm else f"{strategy_name}:{symbol}"
        with self._strategy_cooldowns_lock:
            self._strategy_cooldowns[key] = expiry_time

    def _is_strategy_in_cooldown(
        self,
        strategy_name: str,
        symbol: str,
        side: Optional[str] = None,
        return_expiry: bool = False,
    ):
        """
        Return True if strategy+symbol is in cooldown; auto-cleans expired keys.
        
        Args:
            strategy_name: Strategy identifier
            symbol: Trading symbol
            side: Optional side (long/short/buy/sell) for side-scoped cooldowns
            return_expiry: When True, also returns the expiry datetime for logging/telemetry
        """
        if not strategy_name or not symbol:
            return (False, None) if return_expiry else False

        now = datetime.now(timezone.utc)
        side_norm = self._normalize_side(side) if side else None
        key_side = f"{strategy_name}:{symbol}:{side_norm}" if side_norm else None
        key_plain = f"{strategy_name}:{symbol}"
        cooldown_hit = False
        expiry_time = None
        used_key = None

        with self._strategy_cooldowns_lock:
            expiry_time = self._strategy_cooldowns.get(key_side) if key_side else None
            used_key = key_side if expiry_time is not None else None
            if expiry_time is None:
                expiry_time = self._strategy_cooldowns.get(key_plain)
                used_key = key_plain if expiry_time is not None else None
            if expiry_time is None:
                cooldown_hit = False
            elif now < expiry_time:
                cooldown_hit = True
            else:
                if used_key:
                    self._strategy_cooldowns.pop(used_key, None)
                expiry_time = None

        if return_expiry:
            return cooldown_hit, expiry_time
        return cooldown_hit

    def _get_crash_guard_cfg(self, strategy_name: str) -> Dict[str, Any]:
        try:
            strategies_cfg = self.config.get("strategies", {}) if isinstance(self.config, dict) else {}
            strat_cfg = strategies_cfg.get(strategy_name, {}) if isinstance(strategies_cfg, dict) else {}
            crash_cfg = strat_cfg.get("crash_guard", {}) if isinstance(strat_cfg, dict) else {}
            return crash_cfg if isinstance(crash_cfg, dict) else {}
        except Exception:
            return {}

    async def _compute_panic_state(
        self,
        *,
        symbol: str,
        volume_bucket: Optional[str],
        crash_cfg: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "enabled": bool(crash_cfg.get("enabled", False)),
            "volume_bucket": volume_bucket,
            "tf": None,
            "close": None,
            "atr": None,
            "ema_fast": None,
            "ema_fast_gap_atr": None,
            "fast_drop_pct": None,
            "atr_pct": None,
            "bearish_body_ratio": None,
            "fast_drop": False,
            "high_atr": False,
            "bearish_body": False,
            "is_panic_state": False,
        }

        bucket = str(volume_bucket or "").upper().strip()
        if not bucket:
            return False, meta

        allowed_buckets = crash_cfg.get("panic_volume_buckets") or ["HIGH", "EXTREME"]
        try:
            allowed = {str(b).upper().strip() for b in allowed_buckets if b}
        except Exception:
            allowed = {"HIGH", "EXTREME"}

        if bucket not in allowed:
            return False, meta

        tf = str(crash_cfg.get("panic_tf", "5m") or "5m").strip()
        meta["tf"] = tf

        if not self.market_data_pipeline:
            return False, meta

        try:
            limit = int(crash_cfg.get("panic_lookback_bars", 3) or 3)
        except Exception:
            limit = 3
        limit = max(3, limit)

        try:
            df = await self.market_data_pipeline.get_latest_ohlcv(
                symbol=symbol,
                timeframe=tf,
                limit=limit,
                include_forming=True,
            )
        except Exception:
            df = None

        if df is None or not hasattr(df, "empty") or df.empty or len(df) < 2:
            return False, meta

        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
        except Exception:
            return False, meta

        close_now = None
        prev_close = None
        try:
            close_now = float(last.get("close"))
        except Exception:
            close_now = None
        try:
            prev_close = float(prev.get("close"))
        except Exception:
            prev_close = None

        meta["close"] = close_now
        try:
            meta["atr"] = float(last.get("atr")) if "atr" in df.columns else None
        except Exception:
            meta["atr"] = None
        try:
            meta["ema_fast"] = float(last.get("ema_fast")) if "ema_fast" in df.columns else None
        except Exception:
            meta["ema_fast"] = None
        try:
            if (
                meta.get("ema_fast") is not None
                and meta.get("atr") is not None
                and close_now is not None
                and float(meta.get("atr") or 0.0) > 0
            ):
                meta["ema_fast_gap_atr"] = (float(meta["ema_fast"]) - float(close_now)) / float(meta["atr"])
        except Exception:
            meta["ema_fast_gap_atr"] = None

        # fast_drop: short-TF close-to-close drop
        try:
            fast_drop_th = float(crash_cfg.get("panic_fast_drop_pct", 0.0) or 0.0)
        except Exception:
            fast_drop_th = 0.0
        fast_drop = False
        fast_drop_pct = None
        try:
            if prev_close and prev_close > 0 and close_now is not None:
                fast_drop_pct = max(0.0, (prev_close - close_now) / prev_close)
                if fast_drop_th > 0:
                    fast_drop = fast_drop_pct >= fast_drop_th
        except Exception:
            fast_drop = False
        meta["fast_drop_pct"] = fast_drop_pct
        meta["fast_drop"] = bool(fast_drop)

        # high_atr: ATR/price high AND price is down (avoid killing pumps on volatility alone)
        try:
            atr_th = float(crash_cfg.get("panic_atr_pct", 0.0) or 0.0)
        except Exception:
            atr_th = 0.0
        atr_pct = None
        high_atr = False
        try:
            atr_val = float(last.get("atr")) if "atr" in df.columns else None
            if close_now and close_now > 0 and atr_val is not None:
                atr_pct = atr_val / close_now
                if atr_th > 0 and atr_pct >= atr_th and prev_close is not None and close_now < prev_close:
                    high_atr = True
        except Exception:
            high_atr = False
        meta["atr_pct"] = atr_pct
        meta["high_atr"] = bool(high_atr)

        # bearish_body: down candle with large body / range
        try:
            bear_body_th = float(crash_cfg.get("panic_bear_body_ratio", 0.0) or 0.0)
        except Exception:
            bear_body_th = 0.0
        bearish_body = False
        body_ratio = None
        try:
            if {"open", "high", "low", "close"}.issubset(set(df.columns)):
                o = float(last.get("open"))
                h = float(last.get("high"))
                l = float(last.get("low"))
                c = float(last.get("close"))
                rng = max(0.0, h - l)
                if rng > 0:
                    body_ratio = abs(c - o) / rng
                    if bear_body_th > 0 and c < o and body_ratio >= bear_body_th:
                        bearish_body = True
        except Exception:
            bearish_body = False
        meta["bearish_body_ratio"] = body_ratio
        meta["bearish_body"] = bool(bearish_body)

        # ema_fast_gap panic: when price is far below EMA on high volume, treat as panic even if drop/atr/body
        # thresholds do not trip yet (catches early waterfall phase).
        try:
            gap_th = float(crash_cfg.get("panic_ema_gap_atr_threshold", 0.0) or 0.0)
        except Exception:
            gap_th = 0.0
        ema_gap_panic = False
        try:
            gap_val = meta.get("ema_fast_gap_atr")
            if gap_th > 0 and gap_val is not None and float(gap_val) >= gap_th:
                ema_gap_panic = True
        except Exception:
            ema_gap_panic = False
        meta["ema_gap_panic"] = bool(ema_gap_panic)

        is_panic_state = bool(fast_drop or high_atr or bearish_body or ema_gap_panic)
        meta["is_panic_state"] = is_panic_state
        return is_panic_state, meta

    async def handle_trade_closed(self, payload: Dict[str, Any]) -> None:
        """Receive TRADE_CLOSED events and apply stop-loss-driven cooldown policies (Crash Guard)."""
        try:
            if not isinstance(payload, dict):
                return

            exit_reason = str(payload.get("exit_reason") or "").strip().lower()
            if exit_reason != "stop_loss":
                return

            strategy_name = payload.get("strategy_name") or payload.get("strategy")
            symbol = payload.get("symbol")
            side = self._normalize_side(payload.get("side"))
            if not strategy_name or not symbol or not side:
                return

            # MeanReversion reentry guard (long/short stop-loss reentry protection)
            try:
                if str(strategy_name).strip().lower() == "mean_reversion" and side in {"long", "short"}:
                    pm = getattr(self, "portfolio_manager", None)
                    strategies = getattr(pm, "strategies", None) if pm is not None else None
                    if isinstance(strategies, dict):
                        mr_instance = None
                        for name, instance in strategies.items():
                            if (str(name).strip().lower() == "mean_reversion") or (
                                getattr(instance, "strategy_name", "") == "mean_reversion"
                            ):
                                mr_instance = instance
                                break
                        if mr_instance and hasattr(mr_instance, "arm_reentry_guard"):
                            mr_instance.arm_reentry_guard(str(symbol), side=side)
                            logger.info(
                                "[MR-GUARD] Armed reentry guard | strategy=%s symbol=%s side=%s",
                                strategy_name,
                                symbol,
                                side,
                            )
            except Exception:
                pass

            crash_cfg = self._get_crash_guard_cfg(str(strategy_name))
            if not bool(crash_cfg.get("enabled", False)):
                return

            mode = str(crash_cfg.get("cooldown_mode", "off") or "off").strip().lower()
            if mode == "off":
                return

            volume_bucket = payload.get("volume_bucket_at_entry") or payload.get("volume_bucket")
            is_panic_state, panic_meta = await self._compute_panic_state(
                symbol=str(symbol),
                volume_bucket=str(volume_bucket) if volume_bucket is not None else None,
                crash_cfg=crash_cfg,
            )

            if mode in {"panic_only", "escalating"} and not is_panic_state:
                return

            key = f"{strategy_name}:{symbol}:{side}"
            now_ts = time.time()

            if mode == "reversal_only":
                with self._strategy_cooldowns_lock:
                    self._stop_loss_reversal_required[key] = float(now_ts)
                logger.info(
                    "[CRASH-GUARD] Armed reversal-only reentry guard | strategy=%s symbol=%s side=%s panic=%s",
                    strategy_name,
                    symbol,
                    side,
                    bool(is_panic_state),
                )
                return

            try:
                base_seconds = float(crash_cfg.get("cooldown_seconds", 30.0) or 30.0)
            except Exception:
                base_seconds = 30.0
            cooldown_seconds = max(0.0, base_seconds)

            if mode == "escalating":
                steps = crash_cfg.get("cooldown_escalation_steps") or [30, 90, 180]
                try:
                    window_s = float(crash_cfg.get("cooldown_escalation_window_seconds", 600.0) or 600.0)
                except Exception:
                    window_s = 600.0
                window_s = max(0.0, window_s)

                with self._strategy_cooldowns_lock:
                    state = self._stop_loss_cooldown_streak.get(key) or {}
                    last_ts = float(state.get("last_ts", 0.0) or 0.0)
                    count = int(state.get("count", 0) or 0)
                    if last_ts and window_s and (now_ts - last_ts) > window_s:
                        count = 0
                    count += 1
                    self._stop_loss_cooldown_streak[key] = {"count": count, "last_ts": float(now_ts)}

                try:
                    steps_f = [float(x) for x in steps if x is not None]
                except Exception:
                    steps_f = []
                if steps_f:
                    idx = min(max(count - 1, 0), len(steps_f) - 1)
                    cooldown_seconds = max(0.0, float(steps_f[idx]))

            self._set_strategy_cooldown(str(strategy_name), str(symbol), float(cooldown_seconds), side=side)
            logger.info(
                "[CRASH-GUARD] Stop-loss cooldown set | strategy=%s symbol=%s side=%s seconds=%.1f panic=%s bucket=%s tf=%s",
                strategy_name,
                symbol,
                side,
                float(cooldown_seconds),
                bool(is_panic_state),
                volume_bucket,
                panic_meta.get("tf"),
            )
        except Exception as exc:
            logger.debug("[CRASH-GUARD] handle_trade_closed failed: %s", exc)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _normalize_side(side: Any) -> Optional[str]:
        if side is None:
            return None
        try:
            raw = str(side).strip().lower()
        except Exception:
            return None
        mapping = {"buy": "long", "long": "long", "sell": "short", "short": "short"}
        return mapping.get(raw)

    def _normalize_signal_side(self, signal: Dict[str, Any]) -> None:
        if not isinstance(signal, dict):
            return
        normalized = self._normalize_side(signal.get("side"))
        if normalized:
            signal["side"] = normalized

    def _ensure_signal_id(self, strategy_name: str, signal: Dict[str, Any]) -> str:
        if not isinstance(signal, dict):
            return ""
        existing = signal.get("signal_id") or signal.get("id")
        if existing:
            signal_id = str(existing)
        else:
            signal_id = self._generate_signal_id(strategy_name or "unknown", signal)
        signal["signal_id"] = signal_id
        return signal_id

    @staticmethod
    def _parse_timeframe_ms(timeframe: Any) -> Optional[int]:
        if not timeframe:
            return None
        tf = str(timeframe).strip().lower()
        match = re.match(r"^(\d+)\s*([smhd])$", tf)
        if not match:
            return None
        value = int(match.group(1))
        unit = match.group(2)
        if value <= 0:
            return None
        mult = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}.get(unit)
        return None if mult is None else value * mult

    @staticmethod
    def _coerce_ts_ms(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                numeric = float(value)
            except Exception:
                return None
            if numeric <= 0:
                return None
            # Heuristic: epoch ms ~ 1e12, epoch sec ~ 1e9
            if numeric >= 1e12:
                return int(numeric)
            if numeric >= 1e9:
                return int(numeric * 1000)
            return int(numeric)
        if isinstance(value, datetime):
            try:
                return int(value.timestamp() * 1000)
            except Exception:
                return None
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except Exception:
                return None
        return None

    @classmethod
    def _derive_setup_anchor_ts_ms(cls, signal: Dict[str, Any]) -> int:
        timeframe = signal.get("timeframe") or signal.get("tf") or "5m"
        timeframe_ms = cls._parse_timeframe_ms(timeframe) or cls._parse_timeframe_ms("5m") or 300_000

        anchor = cls._coerce_ts_ms(signal.get("setup_anchor_ts_ms"))
        if anchor is not None:
            return int(anchor)

        candidate_ts = (
            signal.get("timestamp")
            or signal.get("ts")
            or signal.get("signal_timestamp")
            or signal.get("created_at")
            or signal.get("createdAt")
        )
        ts_ms = cls._coerce_ts_ms(candidate_ts) or cls._now_ms()
        anchor_ms = int(ts_ms - (ts_ms % int(timeframe_ms)))
        signal["setup_anchor_ts_ms"] = anchor_ms
        return anchor_ms

    @classmethod
    def _derive_dedupe_key(cls, strategy_name: str, signal: Dict[str, Any]) -> Optional[str]:
        if not isinstance(signal, dict):
            return None
        strategy = (strategy_name or signal.get("strategy_name") or signal.get("strategy") or "").strip()
        symbol = (signal.get("symbol") or "").strip()
        intent = str(signal.get("intent") or INTENT_ENTRY).strip().lower()
        if not strategy or not symbol or not intent:
            return None
        if intent == "soft_deferral":
            side = cls._normalize_side(signal.get("side")) or str(signal.get("side") or "").strip().lower()
            if side not in ("long", "short"):
                return None
            timeframe = str(signal.get("timeframe") or signal.get("tf") or "5m").strip().lower() or "5m"
            return f"{strategy}:{symbol}:{side}:soft_deferral:{timeframe}"

        side = str(signal.get("side") or "").strip().lower()
        timeframe = str(signal.get("timeframe") or "5m").strip().lower()
        if not side:
            return None
        if intent == str(INTENT_SCALE_IN).strip().lower():
            return f"{strategy}:{symbol}:{side}:{intent}"
        anchor_ts = cls._derive_setup_anchor_ts_ms(signal)
        return f"{strategy}:{symbol}:{side}:{intent}:{timeframe}:{anchor_ts}"

    @staticmethod
    def _json_sanitize(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (list, tuple, set)):
            return [StrategyCoordinator._json_sanitize(v) for v in value]
        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            for k, v in value.items():
                try:
                    key_str = str(k)
                except Exception:
                    key_str = "key"
                sanitized[key_str] = StrategyCoordinator._json_sanitize(v)
            return sanitized
        try:
            as_dict = value.to_dict()  # type: ignore[attr-defined]
        except Exception:
            as_dict = None
        if isinstance(as_dict, dict):
            return StrategyCoordinator._json_sanitize(as_dict)
        return str(value)

    @staticmethod
    def _build_incubator_policies(incubator_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        defaults: Dict[str, Dict[str, Any]] = {
            "queue.capacity": {
                "ttl_seconds": 120,
                "max_attempts": 60,
                "base_delay_ms": 500,
                "max_delay_ms": 10_000,
                "refresh_policy": "NONE",
            },
            "queue.symbol_pending_limit": {
                "ttl_seconds": 120,
                "max_attempts": 60,
                "base_delay_ms": 500,
                "max_delay_ms": 10_000,
                "refresh_policy": "NONE",
            },
            "risk.concurrent.max_open_positions": {
                "ttl_seconds": 1800,
                "max_attempts": 180,
                "base_delay_ms": 2000,
                "max_delay_ms": 60_000,
                "refresh_policy": "NONE",
            },
            "risk.concurrent.max_positions_per_symbol": {
                "ttl_seconds": 1800,
                "max_attempts": 180,
                "base_delay_ms": 2000,
                "max_delay_ms": 60_000,
                "refresh_policy": "NONE",
            },
            "risk.planner.heat_exhausted": {
                "ttl_seconds": 900,
                "max_attempts": 120,
                "base_delay_ms": 15_000,
                "max_delay_ms": 15_000,
                "refresh_policy": "REPRICE_AND_RESIZE",
            },
            "risk.concurrent.portfolio_heat": {
                "ttl_seconds": 900,
                "max_attempts": 120,
                "base_delay_ms": 15_000,
                "max_delay_ms": 15_000,
                "refresh_policy": "REPRICE_AND_RESIZE",
            },
            "risk.concurrent.portfolio_heat_exceeded": {
                "ttl_seconds": 900,
                "max_attempts": 120,
                "base_delay_ms": 15_000,
                "max_delay_ms": 15_000,
                "refresh_policy": "REPRICE_AND_RESIZE",
            },
            "strategy.soft_deferral": {
                "ttl_seconds": 300,
                "max_attempts": 30,
                "base_delay_ms": 15_000,
                "max_delay_ms": 15_000,
                "refresh_policy": "STRATEGY_RECHECK",
            },
            "strategy.fast_watch": {
                "ttl_seconds": 30,
                "max_attempts": 9,
                "base_delay_ms": 3000,
                "max_delay_ms": 3000,
                "refresh_policy": "FAST_PRICE_WATCH",
            },
            "volume.low_vol_tight_stop": {
                "ttl_seconds": 25,
                "max_attempts": 2,
                "base_delay_ms": 10_000,
                "max_delay_ms": 10_000,
                "refresh_policy": "REPRICE_AND_RESIZE",
            },
            "volume.shock_low_bucket": {
                "ttl_seconds": 30,
                "max_attempts": 3,
                "base_delay_ms": 5000,
                "max_delay_ms": 10_000,
                "refresh_policy": "REPRICE_AND_RESIZE",
            },
        }

        overrides = incubator_cfg.get("policies", {}) if isinstance(incubator_cfg, dict) else {}
        if isinstance(overrides, dict):
            for key, override in overrides.items():
                if not isinstance(override, dict):
                    continue
                base = defaults.get(str(key), {})
                defaults[str(key)] = {**base, **override}
        return defaults

    def _emit_waiting_room_event(self, event: str, item: Dict[str, Any], **extra_fields: Any) -> None:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        safe_extra = self._json_sanitize(extra_fields) if extra_fields else {}
        refresh_policy = item.get("refresh_policy") if isinstance(item, dict) else None
        if refresh_policy and event in ("waiting_room_add", "waiting_room_drop"):
            refresh_policy_str = str(refresh_policy)
            if not isinstance(safe_extra, dict):
                safe_extra = {}
            safe_extra.setdefault("refresh_policy", refresh_policy_str)
            if refresh_policy_str.upper() == "FAST_PRICE_WATCH" and isinstance(payload, dict):
                condition_data = payload.get("condition_data")
                if isinstance(condition_data, dict):
                    near_val = condition_data.get("near")
                    try:
                        near_str = str(near_val) if near_val is not None else "unknown"
                    except Exception:
                        near_str = "unknown"
                    if not near_str.strip():
                        near_str = "unknown"
                    safe_extra.setdefault("near", near_str)
                    if event == "waiting_room_add":
                        for key in ("watch_interval_ms", "max_checks", "ttl_ms", "trigger_price"):
                            if condition_data.get(key) is not None:
                                safe_extra.setdefault(key, condition_data.get(key))
        ts_ms = self._now_ms()
        signal_id_for_log = None
        parent_pending_id = None
        if isinstance(payload, dict):
            strat = payload.get("strategy_name") or payload.get("strategy") or "unknown"
            self._ensure_signal_id(str(strat), payload)
            if not payload.get("signal_id"):
                logger.error("[INCUBATOR] Missing signal_id in waiting_room event=%s payload=%s", event, payload)
            signal_id_for_log = payload.get("signal_id")
            try:
                parent_pending_id = payload.get("parent_pending_id")
                meta = payload.get("meta")
                if isinstance(meta, dict) and meta.get("parent_pending_id"):
                    parent_pending_id = meta.get("parent_pending_id")
            except Exception:
                parent_pending_id = None
        pending_id = item.get("pending_id") if isinstance(item, dict) else None
        if pending_id and signal_id_for_log and str(signal_id_for_log) == str(pending_id):
            # Telemetry hardening: pending_id is the stable incubator identity; payload signal_id should remain distinct.
            # Do not mutate the payload in-place; only adjust the emitted telemetry fields.
            signal_id_for_log = uuid.uuid4().hex
        out = {
            "event": event,
            "ts_ms": ts_ms,
            "run_id": get_current_run_id(),
            "pending_id": pending_id,
            "parent_pending_id": parent_pending_id,
            "signal_id": signal_id_for_log,
            "strategy": payload.get("strategy_name") or payload.get("strategy"),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "reason_code": item.get("reason_code"),
            "attempts": item.get("attempts", 0),
            "ttl_seconds": item.get("ttl_seconds"),
            "elapsed_ms": ts_ms - int(item.get("first_seen_ts_ms", ts_ms) or ts_ms),
            "ctx_hash": item.get("ctx_hash"),
            "dedupe_key": item.get("dedupe_key"),
            **(safe_extra if isinstance(safe_extra, dict) else {}),
        }
        try:
            logger.info("%s %s", event, json.dumps(out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("%s %s", event, out)

    @staticmethod
    def _normalize_salvage_final_status(status: Any) -> str:
        status_norm = str(status or "").strip().lower()
        if status_norm in ("queued", "enqueued"):
            return "queued"
        if status_norm in ("accepted", "active", "executing", "executed"):
            return "accepted"
        return status_norm or "unknown"

    def _soft_deferral_salvage_cache_max(self) -> int:
        cfg = self.config if isinstance(self.config, dict) else {}
        soft_cfg = cfg.get("soft_deferral", {}) if isinstance(cfg, dict) else {}
        raw = soft_cfg.get("salvage_idempotency_max_items") if isinstance(soft_cfg, dict) else None
        if raw is None:
            return 1000
        try:
            max_items = int(raw)
        except Exception:
            return 1000
        return max(1, max_items)

    def _reset_soft_deferral_roi_caches_if_needed(self) -> None:
        try:
            run_id = get_current_run_id()
        except Exception:
            run_id = None
        last_run_id = getattr(self, "_soft_deferral_roi_cache_run_id", None)
        if last_run_id == run_id:
            return
        setattr(self, "_soft_deferral_roi_cache_run_id", run_id)
        setattr(self, "_salvaged_parent_ids", OrderedDict())
        setattr(self, "_soft_deferral_pending_reason_by_parent_id", OrderedDict())

    def _remember_soft_deferral_pending_reason(self, parent_pending_id: str, pending_reason_code: str) -> None:
        if not parent_pending_id or not pending_reason_code:
            return
        try:
            pending_reason_code = str(pending_reason_code).strip()
        except Exception:
            return
        if not pending_reason_code or pending_reason_code.lower() == "unknown":
            return
        self._reset_soft_deferral_roi_caches_if_needed()
        cache = getattr(self, "_soft_deferral_pending_reason_by_parent_id", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            setattr(self, "_soft_deferral_pending_reason_by_parent_id", cache)
        if parent_pending_id in cache:
            try:
                existing = str(cache.get(parent_pending_id) or "").strip()
            except Exception:
                existing = ""
            if existing and existing.lower() != "unknown":
                cache.move_to_end(parent_pending_id)
                return
        cache[parent_pending_id] = pending_reason_code
        cache.move_to_end(parent_pending_id)
        max_size = self._soft_deferral_salvage_cache_max()
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _mark_soft_deferral_salvaged(self, parent_pending_id: str) -> bool:
        if not parent_pending_id:
            return False
        self._reset_soft_deferral_roi_caches_if_needed()
        cache = getattr(self, "_salvaged_parent_ids", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            setattr(self, "_salvaged_parent_ids", cache)
        if parent_pending_id in cache:
            cache.move_to_end(parent_pending_id)
            return False
        cache[parent_pending_id] = self._now_ms()
        cache.move_to_end(parent_pending_id)
        max_size = self._soft_deferral_salvage_cache_max()
        while len(cache) > max_size:
            cache.popitem(last=False)
        return True

    def _emit_soft_deferral_salvaged(
        self,
        *,
        parent_pending_id: Any,
        signal_id: Any,
        final_status: Any,
        signal_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if parent_pending_id is None:
            return
        try:
            parent_pending_id_str = str(parent_pending_id)
        except Exception:
            return
        if not parent_pending_id_str:
            return

        if not self._mark_soft_deferral_salvaged(parent_pending_id_str):
            return

        payload = signal_payload if isinstance(signal_payload, dict) else {}
        pending_reason_code = None
        cache = getattr(self, "_soft_deferral_pending_reason_by_parent_id", None)
        if isinstance(cache, OrderedDict):
            pending_reason_code = cache.get(parent_pending_id_str)
        if pending_reason_code is not None:
            try:
                pending_reason_code = str(pending_reason_code).strip()
            except Exception:
                pending_reason_code = None
        if not pending_reason_code or str(pending_reason_code).strip().lower() == "unknown":
            pending_reason_code = None
        if not pending_reason_code:
            pending_reason_code = payload.get("pending_reason_code")
            if pending_reason_code is not None:
                try:
                    pending_reason_code = str(pending_reason_code).strip()
                except Exception:
                    pending_reason_code = None
            if not pending_reason_code or str(pending_reason_code).strip().lower() == "unknown":
                pending_reason_code = None
        if not pending_reason_code:
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            pending_reason_code = meta.get("pending_reason_code")
            if pending_reason_code is not None:
                try:
                    pending_reason_code = str(pending_reason_code).strip()
                except Exception:
                    pending_reason_code = None
            if not pending_reason_code or str(pending_reason_code).strip().lower() == "unknown":
                pending_reason_code = None
        if not pending_reason_code:
            pending_reason_code = "unknown"
        reason_code = pending_reason_code

        out = {
            "event": "soft_deferral_salvaged",
            "ts_ms": self._now_ms(),
            "run_id": get_current_run_id(),
            "parent_pending_id": parent_pending_id_str,
            "signal_id": signal_id,
            "final_status": self._normalize_salvage_final_status(final_status),
            "reason_code": reason_code,
            "pending_reason_code": pending_reason_code,
            "strategy": payload.get("strategy_name") or payload.get("strategy"),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "timeframe": payload.get("timeframe") or payload.get("tf"),
        }
        safe_out = self._json_sanitize(out)
        try:
            logger.info("soft_deferral_salvaged %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("soft_deferral_salvaged %s", safe_out)

    def _emit_strategy_recheck_request(self, item: Dict[str, Any], *, check_detail: Optional[Dict[str, Any]] = None) -> None:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        ts_ms = self._now_ms()
        parent_pending_id = item.get("pending_id") if isinstance(item, dict) else None
        pending_reason_code = None
        if isinstance(item, dict):
            pending_reason_code = item.get("pending_reason_code") or item.get("reason_code")
        refresh_policy = item.get("refresh_policy") if isinstance(item, dict) else None
        if parent_pending_id is not None and pending_reason_code is not None:
            try:
                self._remember_soft_deferral_pending_reason(str(parent_pending_id), str(pending_reason_code))
            except Exception:
                pass
        detail = check_detail or {}
        payload_volume_strength = payload.get("volume_strength") if isinstance(payload, dict) else None
        payload_volume_bucket = payload.get("volume_bucket") if isinstance(payload, dict) else None
        payload_volume_source = None
        if isinstance(payload, dict):
            payload_volume_source = payload.get("volume_ctx_source") or payload.get("volume_source")
        if (
            isinstance(payload_volume_strength, (int, float))
            or payload_volume_bucket is not None
            or payload_volume_source is not None
        ):
            detail = dict(detail) if isinstance(detail, dict) else {}
            detail.setdefault("volume_strength", payload_volume_strength)
            detail.setdefault("volume_bucket", payload_volume_bucket)
            detail.setdefault("volume_source", payload_volume_source)
            vol_detail = detail.get("volume") if isinstance(detail.get("volume"), dict) else {}
            vol_detail.setdefault("volume_strength", payload_volume_strength)
            vol_detail.setdefault("volume_bucket", payload_volume_bucket)
            vol_detail.setdefault("source", payload_volume_source)
            detail["volume"] = vol_detail
        if str(refresh_policy or "").upper() == "FAST_PRICE_WATCH" and isinstance(item, dict):
            fast_meta = {
                "rearm_count": item.get("rearm_count", 0),
                "max_rearms": item.get("max_rearms", self._fast_watch_max_rearms),
                "expires_at_ms": item.get("expires_at_ms") or item.get("fast_expires_at_ms"),
                "created_ts_ms": item.get("fast_created_ts_ms") or item.get("first_seen_ts_ms"),
                "watch_interval_ms": item.get("fast_watch_interval_ms"),
            }
            detail = dict(detail) if isinstance(detail, dict) else {}
            detail.setdefault("fast_watch_meta", fast_meta)
        out = {
            "event": "strategy_recheck_request",
            "ts_ms": ts_ms,
            "run_id": get_current_run_id(),
            "pending_id": parent_pending_id,
            "parent_pending_id": parent_pending_id,
            "signal_id": payload.get("signal_id") if isinstance(payload, dict) else None,
            "strategy": (payload.get("strategy_name") or payload.get("strategy")) if isinstance(payload, dict) else None,
            "symbol": payload.get("symbol") if isinstance(payload, dict) else None,
            "side": payload.get("side") if isinstance(payload, dict) else None,
            "timeframe": payload.get("timeframe") if isinstance(payload, dict) else None,
            "intent": payload.get("intent") if isinstance(payload, dict) else None,
            "setup_anchor_ts_ms": payload.get("setup_anchor_ts_ms") if isinstance(payload, dict) else None,
            "reason": payload.get("reason") if isinstance(payload, dict) else None,
            "reason_code": item.get("reason_code") if isinstance(item, dict) else None,
            "pending_reason_code": pending_reason_code,
            "dedupe_key": item.get("dedupe_key") if isinstance(item, dict) else None,
            "condition_data": payload.get("condition_data") if isinstance(payload, dict) else None,
            "refresh_policy": refresh_policy,
            "check_detail": detail,
            "volume_strength": payload_volume_strength,
            "volume_bucket": payload_volume_bucket,
            "volume_source": payload_volume_source,
        }
        safe_out = self._json_sanitize(out)
        try:
            logger.info("strategy_recheck_request %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("strategy_recheck_request %s", safe_out)
        queue = getattr(self, "strategy_recheck_queue", None)
        if queue is not None and hasattr(queue, "put_nowait"):
            try:
                queue.put_nowait(safe_out if isinstance(safe_out, dict) else out)
            except Exception:
                pass

    async def incubate_signal(
        self,
        strategy_name: str,
        signal: Dict[str, Any],
        reason_code: str,
        refresh_policy: str,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._incubator_enabled:
            return {"status": "rejected", "reason": "incubator_disabled", "stage": stage or "incubator"}

        if not isinstance(signal, dict):
            return {"status": "rejected", "reason": "invalid_signal", "stage": stage or "incubator"}

        self._normalize_signal_side(signal)
        self._ensure_signal_id(strategy_name, signal)
        if not signal.get("signal_id"):
            logger.error("[INCUBATOR] Refusing to incubate without signal_id | strat=%s signal=%s", strategy_name, signal)
            return {"status": "rejected", "reason": "missing_signal_id", "stage": stage or "incubator"}

        dedupe_key = self._derive_dedupe_key(strategy_name, signal)
        if not dedupe_key:
            return {"status": "rejected", "reason": "missing_dedupe_key", "stage": stage or "incubator"}

        now_ms = self._now_ms()
        policy = dict(self._incubator_policies.get(reason_code, {}) or {})
        if str(refresh_policy).upper() == "STRATEGY_RECHECK":
            fallback_policy = self._incubator_policies.get("strategy.soft_deferral", {})
            if isinstance(fallback_policy, dict):
                policy = {**fallback_policy, **policy}
        if str(refresh_policy).upper() == "FAST_PRICE_WATCH":
            fallback_policy = self._incubator_policies.get("strategy.fast_watch", {})
            if isinstance(fallback_policy, dict):
                policy = {**fallback_policy, **policy}
        ttl_seconds = int(policy.get("ttl_seconds", 300) or 300)
        max_attempts = int(policy.get("max_attempts", 20) or 20)
        base_delay_ms = int(policy.get("base_delay_ms", 1000) or 1000)
        micro_watch: Optional[Dict[str, Any]] = None

        safe_payload = self._json_sanitize(signal)
        if isinstance(safe_payload, dict):
            safe_payload.setdefault("strategy_name", strategy_name)
            if not safe_payload.get("signal_id"):
                self._ensure_signal_id(strategy_name, safe_payload)
            safe_payload["dedupe_key"] = dedupe_key
            safe_payload["setup_anchor_ts_ms"] = int(self._derive_setup_anchor_ts_ms(safe_payload))

            if str(refresh_policy).upper() == "FAST_PRICE_WATCH":
                condition_data = safe_payload.get("condition_data") if isinstance(safe_payload.get("condition_data"), dict) else {}
                if isinstance(condition_data, dict):
                    ttl_ms = condition_data.get("ttl_ms") or condition_data.get("watch_ttl_ms")
                    if ttl_ms is not None:
                        try:
                            ttl_ms = int(ttl_ms)
                        except Exception:
                            ttl_ms = None
                        if ttl_ms and ttl_ms > 0:
                            ttl_seconds = max(1, int(ttl_ms / 1000))
                    max_checks = condition_data.get("max_checks")
                    if max_checks is not None:
                        try:
                            max_checks = int(max_checks)
                        except Exception:
                            max_checks = None
                        if max_checks and max_checks > 0:
                            max_attempts = max_checks
            micro_watch = self._resolve_micro_gate_watch_settings(safe_payload)
            if micro_watch:
                ttl_seconds = max(1, int(int(micro_watch["ttl_ms"]) / 1000))
                max_attempts = max(1, int(micro_watch["max_checks"]))

        parent_pending_id = None
        pending_reason_code = None
        if isinstance(safe_payload, dict):
            try:
                meta = safe_payload.get("meta")
                if isinstance(meta, dict):
                    parent_pending_id = meta.get("parent_pending_id")
                    pending_reason_code = meta.get("pending_reason_code")
            except Exception:
                parent_pending_id = None
                pending_reason_code = None
        if str(refresh_policy).upper() == "STRATEGY_RECHECK" and not pending_reason_code:
            pending_reason_code = str(reason_code)

        parent_pending_id_str = None
        if parent_pending_id is not None:
            try:
                parent_pending_id_str = str(parent_pending_id)
            except Exception:
                parent_pending_id_str = None
        pending_reason_code_str = None
        if pending_reason_code is not None:
            try:
                pending_reason_code_str = str(pending_reason_code)
            except Exception:
                pending_reason_code_str = None

        if isinstance(safe_payload, dict):
            if pending_reason_code_str:
                safe_payload.setdefault("pending_reason_code", pending_reason_code_str)
                meta = safe_payload.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                    safe_payload["meta"] = meta
                meta.setdefault("pending_reason_code", pending_reason_code_str)
            if parent_pending_id_str:
                meta = safe_payload.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                    safe_payload["meta"] = meta
                meta.setdefault("parent_pending_id", parent_pending_id_str)

        pending_id = uuid.uuid4().hex
        if isinstance(safe_payload, dict) and safe_payload.get("signal_id") and str(safe_payload.get("signal_id")) == str(pending_id):
            safe_payload["signal_id"] = uuid.uuid4().hex

        new_item = {
            "pending_id": pending_id,
            "payload": safe_payload,
            "first_seen_ts_ms": now_ms,
            "next_check_at_ms": now_ms + max(250, base_delay_ms),
            "attempts": 0,
            "reason_code": str(reason_code),
            "pending_reason_code": pending_reason_code_str,
            "parent_pending_id": parent_pending_id_str,
            "refresh_policy": str(refresh_policy),
            "dedupe_key": dedupe_key,
            "ttl_seconds": ttl_seconds,
            "max_attempts": max_attempts,
            "expires_at_ms": now_ms + ttl_seconds * 1000,
            "ctx_hash": None,
            "stage": stage,
        }
        if str(refresh_policy).upper() == "FAST_PRICE_WATCH":
            condition_data = safe_payload.get("condition_data") if isinstance(safe_payload.get("condition_data"), dict) else {}
            watch_interval_ms = self._resolve_fast_watch_interval_ms(condition_data)
            max_rearms = condition_data.get("max_rearms")
            try:
                max_rearms = int(max_rearms)
            except Exception:
                max_rearms = self._fast_watch_max_rearms
            max_rearms = max(0, int(max_rearms))
            new_item.setdefault("state", "watching")
            new_item.setdefault("rearm_count", 0)
            new_item.setdefault("max_rearms", max_rearms)
            new_item.setdefault("fast_watch_interval_ms", watch_interval_ms)
        if micro_watch:
            new_item["watch_kind"] = "micro_gate_watch"
            new_item.setdefault("watch_created_ts_ms", now_ms)
            new_item.setdefault("watch_interval_ms", micro_watch["interval_ms"])
            new_item.setdefault("max_checks", micro_watch["max_checks"])
            new_item.setdefault("watch_ttl_ms", micro_watch["ttl_ms"])
            new_item.setdefault("checks_done", 0)
            new_item["next_check_at_ms"] = now_ms + int(micro_watch["interval_ms"])
            new_item["expires_at_ms"] = int(new_item["watch_created_ts_ms"] + int(micro_watch["ttl_ms"]))
            new_item["ttl_seconds"] = max(1, int(int(micro_watch["ttl_ms"]) / 1000))
            new_item["max_attempts"] = max(1, int(micro_watch["max_checks"]))

        emit_add: Optional[Dict[str, Any]] = None
        emit_drop: Optional[Dict[str, Any]] = None
        emit_drop_reason: Optional[str] = None
        emit_drop_extra: Dict[str, Any] = {}
        emit_micro_gate_dedupe_drop: Optional[Dict[str, Any]] = None
        incoming_signal_id = safe_payload.get("signal_id") if isinstance(safe_payload, dict) else None

        async with self._incubator_lock:
            active_signal_id = self._active_dedupe_by_key.get(dedupe_key)
            if active_signal_id:
                emit_drop = {
                    **new_item,
                    "first_seen_ts_ms": now_ms,
                    "attempts": 0,
                }
                emit_drop_reason = "incubator.dedupe.active_exists"
                emit_drop_extra = {"active_signal_id": active_signal_id}
            else:
                existing = self._incubator_items.get(dedupe_key)
                if existing:
                    if micro_watch and str(existing.get("watch_kind") or "").lower() == "micro_gate_watch":
                        emit_drop = {
                            **new_item,
                            "first_seen_ts_ms": now_ms,
                            "attempts": 0,
                        }
                        emit_drop_reason = "micro_watch_active"
                        emit_drop_extra = {
                            "existing_pending_id": existing.get("pending_id"),
                            "incoming_signal_id": incoming_signal_id,
                        }
                        emit_micro_gate_dedupe_drop = {
                            "dedupe_key": dedupe_key,
                            "existing_pending_id": existing.get("pending_id"),
                            "incoming_signal_id": incoming_signal_id,
                            "reason": "micro_watch_active",
                        }
                    else:
                        if not existing.get("pending_id"):
                            existing["pending_id"] = uuid.uuid4().hex

                        existing_parent_pending_id = existing.get("parent_pending_id")
                        if existing_parent_pending_id:
                            try:
                                existing_parent_pending_id = str(existing_parent_pending_id)
                            except Exception:
                                existing_parent_pending_id = None
                        existing_pending_reason = existing.get("pending_reason_code")
                        if existing_pending_reason:
                            try:
                                existing_pending_reason = str(existing_pending_reason)
                            except Exception:
                                existing_pending_reason = None

                        pending_id_existing = existing.get("pending_id")
                        if isinstance(safe_payload, dict):
                            existing_payload = existing.get("payload") if isinstance(existing.get("payload"), dict) else None
                            if isinstance(existing_payload, dict):
                                existing_condition = existing_payload.get("condition_data")
                                incoming_condition = safe_payload.get("condition_data")
                                if isinstance(existing_condition, dict):
                                    if not isinstance(incoming_condition, dict) or not incoming_condition:
                                        safe_payload["condition_data"] = dict(existing_condition)
                                    else:
                                        merged_condition = dict(existing_condition)
                                        merged_condition.update(incoming_condition)
                                        safe_payload["condition_data"] = merged_condition

                            incoming_signal_id = safe_payload.get("signal_id")
                            if not incoming_signal_id or (
                                pending_id_existing and str(incoming_signal_id) == str(pending_id_existing)
                            ):
                                safe_payload["signal_id"] = uuid.uuid4().hex

                            # Preserve immutable parent correlation data across dedupe updates.
                            if existing_pending_reason:
                                safe_payload["pending_reason_code"] = existing_pending_reason
                                meta = safe_payload.get("meta")
                                if not isinstance(meta, dict):
                                    meta = {}
                                    safe_payload["meta"] = meta
                                meta["pending_reason_code"] = existing_pending_reason
                            if existing_parent_pending_id:
                                meta = safe_payload.get("meta")
                                if not isinstance(meta, dict):
                                    meta = {}
                                    safe_payload["meta"] = meta
                                meta["parent_pending_id"] = existing_parent_pending_id

                        existing["payload"] = safe_payload
                        existing["ctx_hash"] = None
                        if not existing.get("pending_reason_code") and pending_reason_code_str:
                            existing["pending_reason_code"] = pending_reason_code_str
                        if not existing.get("parent_pending_id") and parent_pending_id_str:
                            existing["parent_pending_id"] = parent_pending_id_str
                        if str(refresh_policy).upper() == "FAST_PRICE_WATCH":
                            condition_data = (
                                safe_payload.get("condition_data") if isinstance(safe_payload.get("condition_data"), dict) else {}
                            )
                            watch_interval_ms = self._resolve_fast_watch_interval_ms(condition_data)
                            max_rearms = condition_data.get("max_rearms")
                            try:
                                max_rearms = int(max_rearms)
                            except Exception:
                                max_rearms = self._fast_watch_max_rearms
                            max_rearms = max(0, int(max_rearms))
                            existing.setdefault("state", "watching")
                            existing.setdefault("rearm_count", 0)
                            existing.setdefault("max_rearms", max_rearms)
                            existing.setdefault("fast_watch_interval_ms", watch_interval_ms)
                        if micro_watch:
                            existing.setdefault("watch_kind", "micro_gate_watch")
                            existing.setdefault("watch_interval_ms", micro_watch["interval_ms"])
                            existing.setdefault("max_checks", micro_watch["max_checks"])
                            existing.setdefault("watch_ttl_ms", micro_watch["ttl_ms"])
                            existing.setdefault("watch_created_ts_ms", existing.get("first_seen_ts_ms") or now_ms)
                            existing.setdefault("checks_done", int(existing.get("checks_done", 0) or 0))
                            existing.setdefault("ttl_seconds", max(1, int(int(micro_watch["ttl_ms"]) / 1000)))
                            existing.setdefault("max_attempts", max(1, int(micro_watch["max_checks"])))
                            if not existing.get("expires_at_ms"):
                                existing["expires_at_ms"] = int(existing["watch_created_ts_ms"] + int(micro_watch["ttl_ms"]))
                            if not existing.get("next_check_at_ms"):
                                existing["next_check_at_ms"] = self._now_ms() + int(
                                    existing.get("watch_interval_ms") or micro_watch["interval_ms"]
                                )

                        try:
                            cache_key = existing_parent_pending_id or (
                                str(pending_id_existing) if str(refresh_policy).upper() == "STRATEGY_RECHECK" else None
                            )
                            cache_reason = existing_pending_reason or pending_reason_code_str
                            if cache_key and cache_reason:
                                self._remember_soft_deferral_pending_reason(str(cache_key), str(cache_reason))
                        except Exception:
                            pass
                else:
                    self._incubator_items[dedupe_key] = new_item
                    emit_add = new_item
                    try:
                        cache_key = parent_pending_id_str or (pending_id if str(refresh_policy).upper() == "STRATEGY_RECHECK" else None)
                        if cache_key and pending_reason_code_str:
                            self._remember_soft_deferral_pending_reason(str(cache_key), str(pending_reason_code_str))
                    except Exception:
                        pass

        if emit_drop is not None:
            if not emit_drop_reason:
                emit_drop_reason = "incubator.dedupe.active_exists"
            if emit_micro_gate_dedupe_drop:
                self._emit_micro_gate_watch_dedupe_drop_incoming(**emit_micro_gate_dedupe_drop)
            self._emit_waiting_room_event(
                "waiting_room_drop",
                emit_drop,
                drop_reason=emit_drop_reason,
                **emit_drop_extra,
            )
            return {
                "status": "dropped",
                "reason": emit_drop_reason,
                "stage": stage or "incubator",
                "dedupe_key": dedupe_key,
            }

        if emit_add is not None:
            self._emit_waiting_room_event("waiting_room_add", emit_add)
            if str(emit_add.get("watch_kind") or "").lower() == "micro_gate_watch":
                self._emit_micro_gate_watch_add(emit_add)

        return {"status": "incubated", "reason_code": reason_code, "stage": stage or "incubator", "dedupe_key": dedupe_key}

    async def handle_soft_deferral(self, event: Dict[str, Any]) -> Dict[str, Any]:
        def _emit_reject(
            *,
            reason_code: str,
            error: str,
            strategy: Optional[str] = None,
            symbol: Optional[str] = None,
            side: Optional[str] = None,
            timeframe: Optional[str] = None,
            missing_fields: Optional[List[str]] = None,
        ) -> None:
            now_ms = self._now_ms()
            payload: Dict[str, Any] = {
                "intent": "soft_deferral",
            }
            if strategy:
                payload["strategy_name"] = str(strategy)
            if symbol:
                payload["symbol"] = str(symbol)
            if side:
                payload["side"] = str(side)
            if timeframe:
                payload["timeframe"] = str(timeframe)
            item = {
                "pending_id": uuid.uuid4().hex,
                "payload": payload,
                "first_seen_ts_ms": now_ms,
                "attempts": 0,
                "reason_code": str(reason_code),
                "dedupe_key": None,
                "ttl_seconds": None,
                "ctx_hash": None,
            }
            extra: Dict[str, Any] = {
                "drop_kind": "soft_deferral_reject",
                "drop_reason": str(reason_code),
                "error": str(error),
            }
            if missing_fields:
                extra["missing_fields"] = list(missing_fields)
            self._emit_waiting_room_event("waiting_room_drop", item, **extra)

        if not isinstance(event, dict):
            _emit_reject(reason_code="soft_deferral.invalid_event", error="invalid_event_type_non_dict")
            return {"status": "rejected", "reason": "invalid_event", "stage": "soft_deferral"}

        event_type = event.get("event_type")
        if event_type and str(event_type) != "soft_deferral_event":
            strategy_hint = event.get("strategy") or event.get("strategy_name")
            symbol_hint = event.get("symbol")
            side_hint = event.get("side")
            timeframe_hint = event.get("timeframe") or event.get("tf")
            _emit_reject(
                reason_code="soft_deferral.invalid_event_type",
                error=f"invalid_event_type:{event_type}",
                strategy=strategy_hint,
                symbol=symbol_hint,
                side=side_hint,
                timeframe=timeframe_hint,
            )
            return {"status": "rejected", "reason": "invalid_event_type", "stage": "soft_deferral"}

        strategy = event.get("strategy") or event.get("strategy_name")
        symbol = event.get("symbol")
        side = event.get("side")
        timeframe = event.get("timeframe") or event.get("tf")
        setup_anchor_ts_ms = event.get("setup_anchor_ts_ms")
        reason_code = event.get("reason_code") or "strategy.soft_deferral"
        condition_data = event.get("condition_data") if isinstance(event.get("condition_data"), dict) else {}
        refresh_policy = event.get("refresh_policy")

        missing = []
        if not strategy:
            missing.append("strategy")
        if not symbol:
            missing.append("symbol")
        if not side:
            missing.append("side")
        if not timeframe:
            missing.append("timeframe")
        if setup_anchor_ts_ms is None:
            missing.append("setup_anchor_ts_ms")
        if not reason_code:
            missing.append("reason_code")
        if missing:
            subcode = "soft_deferral.schema_invalid"
            if len(missing) == 1:
                subcode = f"soft_deferral.missing_{missing[0]}"
            _emit_reject(
                reason_code=subcode,
                error="missing_required_fields",
                strategy=strategy,
                symbol=symbol,
                side=side,
                timeframe=timeframe,
                missing_fields=missing,
            )
            return {"status": "rejected", "reason": "invalid_schema", "stage": "soft_deferral", "missing": missing}

        if not refresh_policy:
            if self._fast_watch_enabled and isinstance(condition_data, dict) and condition_data.get("trigger_price") is not None:
                refresh_policy = "FAST_PRICE_WATCH"
            else:
                refresh_policy = "STRATEGY_RECHECK"

        normalized_side = self._normalize_side(side) or str(side).strip().lower()
        if normalized_side not in ("long", "short"):
            _emit_reject(
                reason_code="soft_deferral.invalid_side",
                error=f"invalid_side:{side}",
                strategy=strategy,
                symbol=symbol,
                side=side,
                timeframe=timeframe,
            )
            return {"status": "rejected", "reason": "invalid_side", "stage": "soft_deferral"}

        try:
            anchor_ms = int(float(setup_anchor_ts_ms))
        except Exception:
            _emit_reject(
                reason_code="soft_deferral.invalid_setup_anchor_ts_ms",
                error=f"invalid_setup_anchor_ts_ms:{setup_anchor_ts_ms}",
                strategy=strategy,
                symbol=symbol,
                side=normalized_side,
                timeframe=timeframe,
            )
            return {"status": "rejected", "reason": "invalid_setup_anchor_ts_ms", "stage": "soft_deferral"}

        if anchor_ms <= 0:
            _emit_reject(
                reason_code="soft_deferral.invalid_setup_anchor_ts_ms",
                error=f"invalid_setup_anchor_ts_ms:{setup_anchor_ts_ms}",
                strategy=strategy,
                symbol=symbol,
                side=normalized_side,
                timeframe=timeframe,
            )
            return {"status": "rejected", "reason": "invalid_setup_anchor_ts_ms", "stage": "soft_deferral"}

        synthetic_signal = {
            "symbol": str(symbol),
            "side": normalized_side,
            "timeframe": str(timeframe),
            "setup_anchor_ts_ms": anchor_ms,
            "intent": "soft_deferral",
            "reason": str(event.get("reason") or reason_code),
            "condition_data": dict(condition_data),
            "timestamp": anchor_ms,
        }

        return await self.incubate_signal(
            strategy_name=str(strategy),
            signal=synthetic_signal,
            reason_code=str(reason_code),
            refresh_policy=str(refresh_policy),
            stage="soft_deferral",
        )

    async def incubator_tick(self, max_items: Optional[int] = None, time_budget_ms: Optional[int] = None) -> int:
        if not self._incubator_enabled:
            return 0

        max_items = self._incubator_tick_max_items if max_items is None else int(max_items)
        time_budget_ms = self._incubator_tick_time_budget_ms if time_budget_ms is None else int(time_budget_ms)
        if max_items <= 0 or time_budget_ms <= 0:
            return 0

        start_ms = self._now_ms()
        now_ms = start_ms
        processed = 0
        checked = 0
        skipped = 0
        salvaged = 0
        dropped = 0
        due_total = 0
        price_cache: Dict[tuple[str, str], Optional[float]] = {}
        fast_watch_active = bool(self._fast_watch_task and not self._fast_watch_task.done())
        micro_watch_active = bool(self._micro_gate_watch_task and not self._micro_gate_watch_task.done())

        expired_items: List[Dict[str, Any]] = []
        due_keys: List[str] = []

        async with self._incubator_lock:
            for key, item in list(self._incubator_items.items()):
                try:
                    expires_at = int(item.get("expires_at_ms") or 0)
                except Exception:
                    expires_at = 0
                if expires_at and now_ms >= expires_at:
                    expired_items.append(item)
                    self._incubator_items.pop(key, None)
                    continue

                if fast_watch_active and str(item.get("refresh_policy") or "").upper() == "FAST_PRICE_WATCH":
                    continue

                if micro_watch_active and str(item.get("watch_kind") or "").lower() == "micro_gate_watch":
                    continue

                try:
                    next_check = int(item.get("next_check_at_ms") or 0)
                except Exception:
                    next_check = 0
                if now_ms >= next_check:
                    due_keys.append(key)

            due_total = len(due_keys)

        for item in expired_items:
            self._emit_waiting_room_event("waiting_room_drop", item, drop_reason="ttl_expired")
        dropped += len(expired_items)

        due_keys.sort(key=lambda k: int(self._incubator_items.get(k, {}).get("next_check_at_ms", now_ms) or now_ms))
        for dedupe_key in due_keys[:max_items]:
            if self._now_ms() - start_ms >= time_budget_ms:
                break

            prev_ctx_hash: Optional[str] = None
            async with self._incubator_lock:
                item = self._incubator_items.get(dedupe_key)
                if not item:
                    continue
                payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
                prev_ctx_hash = item.get("ctx_hash")
                attempts = int(item.get("attempts", 0) or 0)
                max_attempts = int(item.get("max_attempts", 0) or 0)
                if max_attempts and attempts >= max_attempts:
                    self._incubator_items.pop(dedupe_key, None)
                    drop_item = dict(item)
                    item = None
                else:
                    drop_item = None

            if drop_item is not None:
                self._emit_waiting_room_event("waiting_room_drop", drop_item, drop_reason="max_attempts_exceeded")
                dropped += 1
                processed += 1
                continue

            if not isinstance(payload, dict):
                processed += 1
                continue

            reason_code = str(item.get("reason_code") or "")
            policy = self._incubator_policies.get(reason_code, {})
            base_delay_ms = int(policy.get("base_delay_ms", 1000) or 1000)
            max_delay_ms = int(policy.get("max_delay_ms", 10_000) or 10_000)
            refresh_policy = str(item.get("refresh_policy") or "NONE").upper()
            if refresh_policy == "FAST_PRICE_WATCH" and not fast_watch_active:
                self._emit_fast_watch_downgrade(item, reason="watcher_inactive")
                refresh_policy = "STRATEGY_RECHECK"

            condition_met = False
            check_detail: Dict[str, Any] = {}

            ctx_hash, ctx_detail = await self._compute_ctx_hash(payload, now_ms=now_ms, price_cache=price_cache, reason_code=reason_code)
            if ctx_hash:
                async with self._incubator_lock:
                    existing = self._incubator_items.get(dedupe_key)
                    if existing:
                        existing["ctx_hash"] = ctx_hash
                check_detail["ctx"] = ctx_detail

            heat_reason_codes = {
                "risk.planner.heat_exhausted",
                "risk.concurrent.portfolio_heat",
                "risk.concurrent.portfolio_heat_exceeded",
            }

            if reason_code in heat_reason_codes and ctx_hash and ctx_hash == prev_ctx_hash:
                # Optimization: for heat gates, only re-evaluate when portfolio or rounded-price context changes.
                condition_met = False
                check_detail["skip_reason"] = "ctx_hash_unchanged"

            if reason_code == "volume.low_vol_tight_stop" and ctx_hash and ctx_hash == prev_ctx_hash:
                # Optimization: if symbol+price+candle open didn't change, skip heavy checks.
                condition_met = False
                check_detail["skip_reason"] = "ctx_hash_unchanged"

            if reason_code in ("queue.capacity", "queue.symbol_pending_limit"):
                try:
                    can_accept, _, _ = await self.signal_queue.can_accept(
                        {"signal": {"symbol": payload.get("symbol"), "intent": payload.get("intent"), "priority": payload.get("priority", 1)}}
                    )
                    condition_met = bool(can_accept)
                    check_detail["queue_qsize"] = self.signal_queue.qsize()
                except Exception as exc:
                    condition_met = False
                    check_detail["error"] = str(exc)

            elif reason_code in ("risk.concurrent.max_open_positions", "risk.concurrent.max_positions_per_symbol"):
                condition_met, concurrent_detail = self._check_concurrent_release_condition(payload, reason_code)
                check_detail.update(concurrent_detail)

            elif reason_code == "volume.low_vol_tight_stop":
                if check_detail.get("skip_reason") != "ctx_hash_unchanged":
                    condition_met, volume_detail = await self._check_volume_release_condition(payload)
                    check_detail.update(volume_detail)

            elif reason_code in heat_reason_codes:
                if check_detail.get("skip_reason") != "ctx_hash_unchanged":
                    condition_met = True

            elif refresh_policy == "STRATEGY_RECHECK":
                condition_met = True

            if not check_detail.get("skip_reason"):
                async with self._incubator_lock:
                    existing = self._incubator_items.get(dedupe_key)
                    if existing:
                        existing["attempts"] = int(existing.get("attempts", 0) or 0) + 1
                        existing["last_attempt_ts_ms"] = now_ms
                        item = existing
                checked += 1
            else:
                skipped += 1

            self._emit_waiting_room_event("waiting_room_retry", item, check_detail=check_detail)

            if condition_met:
                if refresh_policy == "STRATEGY_RECHECK":
                    async with self._incubator_lock:
                        removed = self._incubator_items.pop(dedupe_key, None)
                    if removed:
                        self._emit_strategy_recheck_request(removed, check_detail=check_detail)
                        self._emit_waiting_room_event(
                            "waiting_room_drop",
                            removed,
                            drop_reason="strategy_recheck_requested",
                            drop_kind="strategy_recheck",
                        )
                        salvaged += 1
                    processed += 1
                    continue

                refreshed_signal = await self._apply_refresh_policy(dict(payload), str(item.get("refresh_policy") or "NONE"))
                refreshed_signal["incubator_replay"] = True
                try:
                    result = await self.process_strategy_signal(strategy_name=str(payload.get("strategy_name") or ""), signal=refreshed_signal)
                except Exception as exc:
                    result = {"status": "error", "reason": str(exc), "stage": "incubator_replay"}

                status_raw = result.get("status")
                status = str(status_raw or "").strip().lower()
                queued_statuses = {"queued", "enqueued"}
                accepted_statuses = {"accepted", "active", "executing", "executed"}
                success_statuses = queued_statuses | accepted_statuses
                if status in success_statuses:
                    success_kind = "queued" if status in queued_statuses else "accepted"
                    async with self._incubator_lock:
                        removed = self._incubator_items.pop(dedupe_key, None)
                    item_for_telemetry = removed or item
                    item_payload = item_for_telemetry.get("payload") if isinstance(item_for_telemetry, dict) else None
                    if removed:
                        self._emit_waiting_room_event("waiting_room_accept", removed)
                        salvaged += 1
                    parent_pending_id = None
                    try:
                        parent_pending_id = (
                            item_for_telemetry.get("parent_pending_id") if isinstance(item_for_telemetry, dict) else None
                        )
                        pending_reason_code = (
                            item_for_telemetry.get("pending_reason_code") if isinstance(item_for_telemetry, dict) else None
                        )
                        meta = item_payload.get("meta") if isinstance(item_payload, dict) else None
                        if not parent_pending_id and isinstance(meta, dict):
                            parent_pending_id = meta.get("parent_pending_id")
                        if not pending_reason_code and isinstance(meta, dict):
                            pending_reason_code = meta.get("pending_reason_code")
                        if parent_pending_id and pending_reason_code:
                            self._remember_soft_deferral_pending_reason(str(parent_pending_id), str(pending_reason_code))
                    except Exception:
                        parent_pending_id = None
                    if parent_pending_id:
                        try:
                            salvage_signal_id = result.get("signal_id")
                            if not salvage_signal_id and isinstance(item_payload, dict):
                                salvage_signal_id = item_payload.get("signal_id")
                            self._emit_soft_deferral_salvaged(
                                parent_pending_id=parent_pending_id,
                                signal_id=salvage_signal_id,
                                final_status=status,
                                signal_payload=item_payload if isinstance(item_payload, dict) else None,
                            )
                        except Exception:
                            pass
                    self._emit_waiting_room_event(
                        "waiting_room_outcome",
                        item_for_telemetry,
                        outcome="success",
                        success_kind=success_kind,
                        final_status=status,
                        final_reason=result.get("reason") or "queued",
                        final_reason_code=result.get("reason_code"),
                    )
                elif status == "incubated":
                    # process_strategy_signal re-incubated/update the item; keep it
                    pass
                else:
                    async with self._incubator_lock:
                        removed = self._incubator_items.pop(dedupe_key, None)
                    item_for_telemetry = removed or item
                    if status == "rejected":
                        self._emit_waiting_room_event(
                            "waiting_room_outcome",
                            item_for_telemetry,
                            outcome="failed_replay",
                            success_kind="none",
                            final_status=status,
                            final_reason=result.get("reason"),
                            final_reason_code=result.get("reason_code"),
                        )
                    if removed:
                        self._emit_waiting_room_event(
                            "waiting_room_drop",
                            removed,
                            drop_reason="replay_failed",
                            replay_status=status,
                            replay_stage=result.get("stage"),
                            replay_reason=result.get("reason"),
                        )
                        dropped += 1

            else:
                delay = min(max_delay_ms, max(250, int(base_delay_ms * (2 ** min(int(item.get("attempts", 1) or 1) - 1, 6)))))
                async with self._incubator_lock:
                    existing = self._incubator_items.get(dedupe_key)
                    if existing:
                        existing["next_check_at_ms"] = self._now_ms() + delay

            processed += 1

        if processed or dropped:
            elapsed_ms = self._now_ms() - start_ms
            async with self._incubator_lock:
                pending = len(self._incubator_items)
            logger.info(
                "Incubator Tick: pending=%s, due=%s, checked=%s, skipped=%s, salvaged=%s, dropped=%s, elapsed_ms=%s",
                pending,
                due_total,
                checked,
                skipped,
                salvaged,
                dropped,
                elapsed_ms,
            )

        return processed

    def set_recheck_ready_callback(self, callback: Optional[Callable[[], None]]) -> None:
        self._recheck_ready_callback = callback if callable(callback) else None

    def _notify_recheck_ready(self) -> None:
        callback = getattr(self, "_recheck_ready_callback", None)
        if callback:
            try:
                callback()
            except Exception as exc:
                logger.debug("[FAST-WATCH] Recheck wakeup callback failed: %s", exc)

    def _resolve_volume_heartbeat_symbols(self) -> List[str]:
        hb_cfg: Dict[str, Any] = {}
        try:
            va_cfg = self.config.get("volume_analyzer", {}) if isinstance(self.config, dict) else {}
            hb_cfg = (va_cfg or {}).get("heartbeat", {}) if isinstance(va_cfg, dict) else {}
        except Exception:
            hb_cfg = {}

        explicit = hb_cfg.get("symbols")
        if isinstance(explicit, (list, tuple)) and explicit:
            return [str(sym).strip() for sym in explicit if str(sym).strip()]

        try:
            fixed = (self.config.get("universe", {}) or {}).get("fixed_symbols") if isinstance(self.config, dict) else None
            if isinstance(fixed, (list, tuple)) and fixed:
                return [str(sym).strip() for sym in fixed if str(sym).strip()]
        except Exception:
            pass

        env_symbols = os.environ.get("TRADING_SYMBOLS", "").strip()
        if env_symbols:
            symbols = [s.strip() for s in env_symbols.split(",") if s.strip()]
            if symbols:
                return symbols

        return []

    async def _emit_volume_heartbeat(self, symbol: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        run_id = get_current_run_id()
        trade_tf = self._volume_heartbeat_trade_tf
        key = f"{symbol}:{trade_tf}"

        if not (self.volume_analyzer and self._volume_analyzer_enabled):
            payload = {
                "event": "volume_context_heartbeat_error",
                "timestamp": now_iso,
                "run_id": run_id,
                "symbol": symbol,
                "timeframe": trade_tf,
                "reason": "volume_analyzer_disabled",
            }
            logger.info("⚠️📊 volume_context_heartbeat_error %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return

        try:
            ctx = await self.volume_analyzer.compute_context(symbol=symbol, trade_timeframe=trade_tf)
        except Exception as exc:
            payload = {
                "event": "volume_context_heartbeat_error",
                "timestamp": now_iso,
                "run_id": run_id,
                "symbol": symbol,
                "timeframe": trade_tf,
                "reason": "compute_context_exception",
                "error": str(exc),
            }
            logger.info("⚠️📊 volume_context_heartbeat_error %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return

        if not ctx:
            payload = {
                "event": "volume_context_heartbeat_error",
                "timestamp": now_iso,
                "run_id": run_id,
                "symbol": symbol,
                "timeframe": trade_tf,
                "reason": "no_context",
            }
            logger.info("⚠️📊 volume_context_heartbeat_error %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return

        prev = self._volume_heartbeat_last_ctx.get(key) or {}
        prev_bucket = prev.get("volume_bucket")
        prev_strength = prev.get("volume_strength")

        try:
            strength = float(ctx.volume_strength)
        except Exception:
            strength = None
        bucket = getattr(ctx, "bucket", None)

        changed_bucket = (prev_bucket is not None) and (bucket != prev_bucket)
        changed_strength = False
        if prev_strength is not None and strength is not None:
            try:
                changed_strength = abs(float(strength) - float(prev_strength)) >= float(self._volume_heartbeat_min_strength_delta)
            except Exception:
                changed_strength = False
        changed = changed_bucket or changed_strength or (prev_bucket is None and prev_strength is None)

        payload: Dict[str, Any] = {
            "event": "volume_context_heartbeat",
            "timestamp": now_iso,
            "run_id": run_id,
            "symbol": symbol,
            "timeframe": trade_tf,
            "volume_bucket": bucket,
            "volume_strength": strength,
            "prev_volume_bucket": prev_bucket,
            "prev_volume_strength": prev_strength,
            "changed": bool(changed),
            "changed_bucket": bool(changed_bucket),
            "changed_strength": bool(changed_strength),
            "min_strength_delta": float(self._volume_heartbeat_min_strength_delta),
            "source": "analyzer",
        }

        if self._volume_heartbeat_include_debug_fields:
            payload.update(
                {
                    "ratio_short": getattr(ctx, "ratio_short", None),
                    "ratio_medium": getattr(ctx, "ratio_medium", None),
                    "ratio_combined": getattr(ctx, "ratio_combined", None),
                    "current_window_volume": getattr(ctx, "current_window_volume", None),
                    "short_baseline_volume": getattr(ctx, "short_baseline_volume", None),
                    "medium_baseline_volume": getattr(ctx, "medium_baseline_volume", None),
                    "baseline_short_last_bar_ts": getattr(ctx, "baseline_short_last_bar_ts", None),
                    "baseline_medium_last_bar_ts": getattr(ctx, "baseline_medium_last_bar_ts", None),
                    "baseline_calc_mode": getattr(ctx, "baseline_calc_mode", None),
                }
            )

        self._volume_heartbeat_last_ctx[key] = {"volume_bucket": bucket, "volume_strength": strength}

        prefix = "🔔📊" if changed else "💓📊"
        logger.info("%s volume_context_heartbeat %s", prefix, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def start_volume_heartbeat(self) -> bool:
        if not self._volume_heartbeat_enabled:
            return False
        task = getattr(self, "_volume_heartbeat_task", None)
        if task and not task.done():
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[VOLUME-HEARTBEAT] No running event loop; heartbeat not started")
            return False
        self._volume_heartbeat_task = loop.create_task(self._volume_heartbeat_loop())
        return True

    def stop_volume_heartbeat(self) -> None:
        task = getattr(self, "_volume_heartbeat_task", None)
        if task and not task.done():
            task.cancel()
        self._volume_heartbeat_task = None

    async def _volume_heartbeat_loop(self) -> None:
        logger.info(
            "💓📊 [VOLUME-HEARTBEAT] Started | interval_sec=%s trade_tf=%s",
            self._volume_heartbeat_interval_sec,
            self._volume_heartbeat_trade_tf,
        )
        warned_empty = False
        try:
            while True:
                symbols = self._resolve_volume_heartbeat_symbols()
                if not symbols:
                    if not warned_empty:
                        warned_empty = True
                        logger.info("💓📊 [VOLUME-HEARTBEAT] No symbols resolved; waiting...")
                    await asyncio.sleep(min(60, int(self._volume_heartbeat_interval_sec)))
                    continue

                warned_empty = False
                now = time.time()
                for symbol in symbols:
                    last = self._volume_heartbeat_last_emit_ts.get(symbol, 0.0)
                    if (now - last) < float(self._volume_heartbeat_interval_sec):
                        continue
                    await self._emit_volume_heartbeat(symbol)
                    self._volume_heartbeat_last_emit_ts[symbol] = now

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("💓📊 [VOLUME-HEARTBEAT] Stopped")
            raise

    def start_fast_watcher(self) -> bool:
        if not self._fast_watch_enabled:
            return False
        task = getattr(self, "_fast_watch_task", None)
        if task and not task.done():
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[FAST-WATCH] No running event loop; watcher not started")
            return False
        self._fast_watch_task = loop.create_task(self._fast_watch_loop())
        return True

    def stop_fast_watcher(self) -> None:
        task = getattr(self, "_fast_watch_task", None)
        if task and not task.done():
            task.cancel()
        self._fast_watch_task = None

    def start_micro_gate_watcher(self) -> bool:
        if not self._micro_gate_watch_enabled:
            return False
        task = getattr(self, "_micro_gate_watch_task", None)
        if task and not task.done():
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[MICRO-GATE] No running event loop; watcher not started")
            return False
        self._micro_gate_watch_task = loop.create_task(self._micro_gate_watch_loop())
        return True

    def stop_micro_gate_watcher(self) -> None:
        task = getattr(self, "_micro_gate_watch_task", None)
        if task and not task.done():
            task.cancel()
        self._micro_gate_watch_task = None

    async def _fast_watch_loop(self) -> None:
        logger.info(
            "[FAST-WATCH] Started | interval_ms=%s max_items=%s time_budget_ms=%s",
            self._fast_watch_interval_ms,
            self._fast_watch_max_items_per_tick,
            self._fast_watch_time_budget_ms,
        )
        try:
            while True:
                start_ms = self._now_ms()
                try:
                    await self._fast_watch_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("[FAST-WATCH] Tick failed: %s", exc)
                elapsed_ms = self._now_ms() - start_ms
                sleep_ms = max(0, int(self._fast_watch_interval_ms) - int(elapsed_ms))
                if sleep_ms:
                    await asyncio.sleep(sleep_ms / 1000)
        except asyncio.CancelledError:
            logger.info("[FAST-WATCH] Stopped")
            raise

    async def _micro_gate_watch_loop(self) -> None:
        logger.info(
            "[MICRO-GATE] Started | loop_interval_ms=%s max_items=%s time_budget_ms=%s",
            self._micro_gate_watch_loop_interval_ms,
            self._micro_gate_watch_max_items_per_tick,
            self._micro_gate_watch_time_budget_ms,
        )
        try:
            while True:
                start_ms = self._now_ms()
                try:
                    await self._micro_gate_watch_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("[MICRO-GATE] Tick failed: %s", exc)
                elapsed_ms = self._now_ms() - start_ms
                sleep_ms = max(0, int(self._micro_gate_watch_loop_interval_ms) - int(elapsed_ms))
                if sleep_ms:
                    await asyncio.sleep(sleep_ms / 1000)
        except asyncio.CancelledError:
            logger.info("[MICRO-GATE] Stopped")
            raise

    async def _fast_watch_tick(self) -> int:
        if not self._fast_watch_enabled:
            return 0

        start_ms = self._now_ms()
        now_ms = start_ms
        max_items = self._fast_watch_max_items_per_tick
        time_budget_ms = self._fast_watch_time_budget_ms
        due_items: List[Dict[str, Any]] = []
        expired_items: List[Tuple[Dict[str, Any], Dict[str, Any], Optional[int], Optional[str], int]] = []

        async with self._incubator_lock:
            for dedupe_key, item in list(self._incubator_items.items()):
                if self._now_ms() - start_ms >= time_budget_ms:
                    break
                payload = item.get("payload") if isinstance(item.get("payload"), dict) else None
                if not isinstance(payload, dict):
                    continue
                if str(item.get("refresh_policy") or "").upper() != "FAST_PRICE_WATCH":
                    continue
                condition_data = payload.get("condition_data") if isinstance(payload.get("condition_data"), dict) else {}
                if not isinstance(condition_data, dict):
                    continue
                symbol = payload.get("symbol")
                side = payload.get("side")
                timeframe = payload.get("timeframe") or payload.get("tf")
                trigger_price = condition_data.get("trigger_price")
                if not symbol or not side or not timeframe or trigger_price is None:
                    continue

                expires_at_ms, expire_reason, created_ts_ms = self._resolve_fast_watch_expiry(item, payload, condition_data, now_ms)
                if created_ts_ms:
                    item.setdefault("fast_created_ts_ms", created_ts_ms)
                if expire_reason:
                    item["fast_expire_reason"] = expire_reason
                if expires_at_ms and expires_at_ms != item.get("expires_at_ms"):
                    item["expires_at_ms"] = expires_at_ms
                    item["fast_expires_at_ms"] = expires_at_ms
                if expires_at_ms and now_ms >= expires_at_ms:
                    removed = self._incubator_items.pop(dedupe_key, None)
                    if removed:
                        expired_items.append((removed, condition_data, expires_at_ms, expire_reason, created_ts_ms))
                    continue

                state = str(item.get("state") or "watching").strip().lower()
                if state != "watching":
                    continue

                try:
                    next_check_at_ms = int(item.get("next_check_at_ms") or 0)
                except Exception:
                    next_check_at_ms = 0
                if now_ms >= next_check_at_ms:
                    due_items.append(
                        {
                            "dedupe_key": dedupe_key,
                            "strategy": payload.get("strategy_name") or payload.get("strategy"),
                            "symbol": str(symbol),
                            "side": str(side),
                            "timeframe": str(timeframe),
                            "trigger_price": trigger_price,
                            "trigger_kind": str(condition_data.get("trigger_kind") or ""),
                            "eps_bps": condition_data.get("eps_bps"),
                            "exchange": payload.get("exchange") or payload.get("exchange_name"),
                            "condition_data": dict(condition_data),
                            "created_ts_ms": created_ts_ms,
                            "expires_at_ms": expires_at_ms,
                            "expire_reason": expire_reason,
                        }
                    )
                    if len(due_items) >= max_items:
                        break

        expired_price_by_key: Dict[Tuple[str, str, Optional[str]], Tuple[Optional[float], Optional[str]]] = {}
        for removed, _condition_data, _expires_at_ms, _expire_reason, _created_ts_ms in expired_items:
            payload = removed.get("payload") if isinstance(removed.get("payload"), dict) else {}
            symbol = payload.get("symbol")
            timeframe = payload.get("timeframe") or payload.get("tf") or "5m"
            exchange = payload.get("exchange") or payload.get("exchange_name")
            if not symbol:
                continue
            key = (str(symbol), str(timeframe), str(exchange) if exchange else None)
            if key in expired_price_by_key:
                continue
            expired_price_by_key[key] = await self._get_cache_only_price(
                str(symbol),
                timeframe=str(timeframe),
                exchange=str(exchange) if exchange else None,
            )

        for removed, condition_data, expires_at_ms, expire_reason, created_ts_ms in expired_items:
            payload = removed.get("payload") if isinstance(removed.get("payload"), dict) else {}
            symbol = payload.get("symbol")
            timeframe = payload.get("timeframe") or payload.get("tf") or "5m"
            exchange = payload.get("exchange") or payload.get("exchange_name")
            price = None
            cache_source = None
            if symbol:
                price_key = (str(symbol), str(timeframe), str(exchange) if exchange else None)
                price, cache_source = expired_price_by_key.get(price_key, (None, None))
            cache_hit = price is not None

            self._emit_fast_watch_outcome(
                removed,
                condition_data=condition_data,
                outcome="expired",
                checks_done=int(removed.get("fast_checks_done") or removed.get("attempts", 0) or 0),
                price=price,
                created_ts_ms=created_ts_ms,
                now_ts_ms=now_ms,
                expires_at_ms=expires_at_ms,
                expire_reason=expire_reason or "expired",
                cache_hit=cache_hit,
                cache_source=cache_source,
                miss_count=int(removed.get("fast_miss_count") or 0),
            )
            self._emit_fast_watch_waiting_room_compat(
                removed,
                outcome="expired",
                expire_reason=expire_reason,
                price=price,
                now_ts_ms=now_ms,
            )

        if not due_items:
            return 0

        price_by_key: Dict[Tuple[str, str, Optional[str]], Tuple[Optional[float], Optional[str]]] = {}
        bidask_by_key: Dict[Tuple[str, str, Optional[str]], Optional[Dict[str, Any]]] = {}
        for item in due_items:
            key = (item["symbol"], item["timeframe"], item.get("exchange"))
            if key in price_by_key:
                continue
            price_by_key[key] = await self._get_cache_only_price(
                item["symbol"],
                timeframe=item["timeframe"],
                exchange=item.get("exchange"),
            )

        outcomes: List[Dict[str, Any]] = []
        triggers: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        cache_hits = 0
        cache_misses = 0

        async with self._incubator_lock:
            for item in due_items:
                if self._now_ms() - start_ms >= time_budget_ms:
                    break
                dedupe_key = item["dedupe_key"]
                live_item = self._incubator_items.get(dedupe_key)
                if not live_item:
                    continue
                state = str(live_item.get("state") or "watching").strip().lower()
                if state != "watching":
                    continue
                payload = live_item.get("payload") if isinstance(live_item.get("payload"), dict) else None
                if not isinstance(payload, dict):
                    continue
                condition_data = payload.get("condition_data") if isinstance(payload.get("condition_data"), dict) else {}
                if not isinstance(condition_data, dict):
                    continue

                expires_at_ms, expire_reason, created_ts_ms = self._resolve_fast_watch_expiry(live_item, payload, condition_data, now_ms)
                if created_ts_ms:
                    live_item.setdefault("fast_created_ts_ms", created_ts_ms)
                if expire_reason:
                    live_item["fast_expire_reason"] = expire_reason
                if expires_at_ms and now_ms >= expires_at_ms:
                    removed = self._incubator_items.pop(dedupe_key, None)
                    if removed:
                        outcomes.append(
                            {
                                "item": removed,
                                "condition_data": condition_data,
                                "outcome": "expired",
                                "checks_done": int(removed.get("fast_checks_done") or removed.get("attempts", 0) or 0),
                                "price": None,
                                "created_ts_ms": created_ts_ms,
                                "expires_at_ms": expires_at_ms,
                                "expire_reason": expire_reason or "expired",
                                "cache_hit": False,
                                "cache_source": None,
                                "miss_count": int(removed.get("fast_miss_count") or 0),
                            }
                        )
                    continue

                price_key = (item["symbol"], item["timeframe"], item.get("exchange"))
                price, cache_source = price_by_key.get(price_key, (None, None))
                cache_hit = price is not None
                if not cache_hit:
                    cache_misses += 1
                else:
                    cache_hits += 1
                    live_item["last_known_price"] = price
                    live_item["last_known_ts_ms"] = now_ms

                strategy_name = item.get("strategy")
                v2 = self._resolve_fast_watch_v2_settings(
                    strategy_name=str(strategy_name) if strategy_name else None,
                    condition_data=condition_data,
                )

                interval_ms = self._resolve_fast_watch_interval_ms(condition_data)
                max_checks = self._resolve_fast_watch_max_checks(condition_data, live_item)

                checks_done = int(live_item.get("fast_checks_done") or live_item.get("attempts", 0) or 0)
                miss_count = int(live_item.get("fast_miss_count") or 0)
                if not cache_hit and not v2:
                    miss_count += 1
                    live_item["fast_miss_count"] = miss_count
                    live_item["next_check_at_ms"] = self._now_ms() + interval_ms
                    outcomes.append(
                        {
                            "item": live_item,
                            "condition_data": condition_data,
                            "outcome": "cache_miss",
                            "checks_done": checks_done,
                            "price": None,
                            "created_ts_ms": created_ts_ms,
                            "expires_at_ms": expires_at_ms,
                            "expire_reason": "cache_miss",
                            "cache_hit": False,
                            "cache_source": cache_source,
                            "miss_count": miss_count,
                        }
                    )
                    continue
                if not cache_hit and v2:
                    miss_count += 1
                    live_item["fast_miss_count"] = miss_count

                trigger_price = self._coerce_float(condition_data.get("trigger_price"))
                eps_bps = self._coerce_float(condition_data.get("eps_bps")) or 0.0
                trigger_kind = str(condition_data.get("trigger_kind") or "")
                side = str(payload.get("side") or "")
                if trigger_price is None or trigger_price <= 0:
                    removed = self._incubator_items.pop(dedupe_key, None)
                    if removed:
                        outcomes.append(
                            {
                                "item": removed,
                                "condition_data": condition_data,
                                "outcome": "expired",
                                "checks_done": checks_done,
                                "price": price,
                                "created_ts_ms": created_ts_ms,
                                "expires_at_ms": expires_at_ms,
                                "expire_reason": "expired",
                                "cache_hit": True,
                                "cache_source": cache_source,
                                "miss_count": miss_count,
                            }
                        )
                    continue

                near_hit = None
                touch_candidate = None
                touch_confirmed = None
                trigger_action = "none"
                stale_reason = None
                refresh_attempted = False
                ticker_age_ms = None
                bid = ask = None
                px_used = None
                dist_to_band_bps = None
                touch_eps_bps_used = None

                hit = False
                if v2:
                    side_norm = str(side or "").strip().lower()
                    exchange = self._normalize_exchange_key(payload.get("exchange") or payload.get("exchange_name"))
                    bidask_key = (item["symbol"], item["timeframe"], exchange)
                    metrics = bidask_by_key.get(bidask_key)
                    if bidask_key not in bidask_by_key:
                        metrics = await self._get_bidask_metrics(str(payload.get("symbol")), exchange=exchange)
                        bidask_by_key[bidask_key] = metrics

                    if isinstance(metrics, dict):
                        bid = metrics.get("bid")
                        ask = metrics.get("ask")
                        try:
                            bid = float(bid) if bid is not None else None
                        except Exception:
                            bid = None
                        try:
                            ask = float(ask) if ask is not None else None
                        except Exception:
                            ask = None
                        age_val = metrics.get("age_ms")
                        try:
                            ticker_age_ms = int(age_val) if age_val is not None else None
                        except Exception:
                            ticker_age_ms = None

                    if side_norm in ("long", "buy"):
                        px_used = bid
                    elif side_norm in ("short", "sell"):
                        px_used = ask
                    else:
                        px_used = None
                        stale_reason = "side_invalid"

                    try:
                        touch_eps_bps_used = float(v2["touch_eps_bps"])
                    except Exception:
                        touch_eps_bps_used = None

                    if px_used is None:
                        near_hit = False
                        touch_candidate = False
                        touch_confirmed = False
                        if stale_reason is None:
                            stale_reason = "bidask_missing"
                    else:
                        sigma_used = self._coerce_float(condition_data.get("trigger_sigma"))
                        if sigma_used is not None and sigma_used <= 0:
                            sigma_used = None

                        touch_sigma_delta_used = v2.get("touch_sigma_delta")
                        near_sigma_delta_used = v2.get("near_sigma_delta")

                        bps_near_eps_px = abs(float(trigger_price)) * (float(v2["near_bps"]) / 10000.0)
                        bps_touch_eps_px = abs(float(trigger_price)) * (float(v2["touch_eps_bps"]) / 10000.0)

                        def _clamp_eps(val: Optional[float], *, min_abs_px: Any, max_abs_px: Any) -> Optional[float]:
                            if val is None:
                                return None
                            out = float(val)
                            if not math.isfinite(out):
                                return None
                            min_v = self._coerce_float(min_abs_px)
                            if min_v is not None and min_v > 0:
                                out = max(out, float(min_v))
                            max_v = self._coerce_float(max_abs_px)
                            if max_v is not None and max_v > 0:
                                out = min(out, float(max_v))
                            return out

                        sigma_touch_eps_px = None
                        if sigma_used is not None and touch_sigma_delta_used is not None:
                            try:
                                sigma_touch_eps_px = float(sigma_used) * float(touch_sigma_delta_used)
                            except Exception:
                                sigma_touch_eps_px = None
                            sigma_touch_eps_px = _clamp_eps(
                                sigma_touch_eps_px,
                                min_abs_px=v2.get("min_touch_abs_px"),
                                max_abs_px=v2.get("max_touch_abs_px"),
                            )

                        sigma_near_eps_px = None
                        if sigma_used is not None and near_sigma_delta_used is not None:
                            try:
                                sigma_near_eps_px = float(sigma_used) * float(near_sigma_delta_used)
                            except Exception:
                                sigma_near_eps_px = None
                            sigma_near_eps_px = _clamp_eps(
                                sigma_near_eps_px,
                                min_abs_px=v2.get("min_near_abs_px"),
                                max_abs_px=v2.get("max_near_abs_px"),
                            )

                        effective_touch_eps_px = (
                            max(float(bps_touch_eps_px), float(sigma_touch_eps_px))
                            if sigma_touch_eps_px is not None
                            else float(bps_touch_eps_px)
                        )
                        effective_near_eps_px = (
                            max(float(bps_near_eps_px), float(sigma_near_eps_px))
                            if sigma_near_eps_px is not None
                            else float(bps_near_eps_px)
                        )

                        dist_to_band_px = None
                        try:
                            if side_norm in ("long", "buy"):
                                dist_to_band_px = float(px_used) - float(trigger_price)
                            else:
                                dist_to_band_px = float(trigger_price) - float(px_used)
                        except Exception:
                            dist_to_band_px = None

                        try:
                            dist_to_band_bps = (
                                (float(dist_to_band_px) / float(trigger_price) * 10000.0) if dist_to_band_px is not None else None
                            )
                        except Exception:
                            dist_to_band_bps = None

                        near_hit = bool(dist_to_band_px is not None and float(dist_to_band_px) <= float(effective_near_eps_px))
                        touch_candidate = bool(dist_to_band_px is not None and float(dist_to_band_px) <= float(effective_touch_eps_px))

                    if touch_candidate:
                        max_age = int(v2.get("max_ticker_age_ms") or 0)
                        if ticker_age_ms is None:
                            touch_confirmed = False
                            stale_reason = "ticker_age_missing"
                            trigger_action = "skip_stale"
                        elif max_age and int(ticker_age_ms) > max_age:
                            touch_confirmed = False
                            stale_reason = "ticker_stale"
                            trigger_action = "skip_stale"
                        elif side_norm in ("long", "buy") and bid is None:
                            touch_confirmed = False
                            stale_reason = "bid_missing"
                            trigger_action = "no_touch"
                        elif side_norm in ("short", "sell") and ask is None:
                            touch_confirmed = False
                            stale_reason = "ask_missing"
                            trigger_action = "no_touch"
                        else:
                            if side_norm in ("long", "buy"):
                                touch_confirmed = (
                                    bool(float(bid) <= float(trigger_price) + float(effective_touch_eps_px)) if bid is not None else False
                                )
                            else:
                                touch_confirmed = (
                                    bool(float(ask) >= float(trigger_price) - float(effective_touch_eps_px)) if ask is not None else False
                                )
                            trigger_action = "recheck" if touch_confirmed else "no_touch"

                        if touch_confirmed:
                            last_ts = self._coerce_int(live_item.get("fast_watch_last_trigger_ts_ms"))
                            cooldown_ms = int(v2.get("recheck_cooldown_ms") or 0)
                            if cooldown_ms and last_ts and now_ms - int(last_ts) < cooldown_ms:
                                outcomes.append(
                                    {
                                        "item": live_item,
                                        "condition_data": condition_data,
                                        "outcome": "cooldown_skip",
                                        "checks_done": checks_done,
                                        "price": px_used,
                                        "created_ts_ms": created_ts_ms,
                                        "expires_at_ms": expires_at_ms,
                                        "expire_reason": "cooldown",
                                        "cache_hit": True,
                                        "cache_source": cache_source,
                                        "miss_count": miss_count,
                                        "extra": {
                                            "near_hit": near_hit,
                                            "touch_candidate": touch_candidate,
                                            "touch_confirmed": touch_confirmed,
                                            "trigger_action": "cooldown",
                                            "stale_reason": None,
                                            "ticker_age_ms": ticker_age_ms,
                                            "px_used": px_used,
                                            "touch_eps_bps": touch_eps_bps_used,
                                            "dist_to_band_bps": dist_to_band_bps,
                                            "bid": bid,
                                            "ask": ask,
                                        },
                                    }
                                )
                                live_item["next_check_at_ms"] = self._now_ms() + interval_ms
                                continue

                            hit = True

                        if trigger_action == "skip_stale":
                            refresh_attempted = True
                            metrics2 = await self._get_bidask_metrics(str(payload.get("symbol")), exchange=exchange)
                            if isinstance(metrics2, dict):
                                age_val = metrics2.get("age_ms")
                                try:
                                    ticker_age_ms = int(age_val) if age_val is not None else None
                                except Exception:
                                    ticker_age_ms = None
                            outcomes.append(
                                {
                                    "item": live_item,
                                    "condition_data": condition_data,
                                    "outcome": "stale_skip",
                                    "checks_done": checks_done,
                                    "price": px_used,
                                    "created_ts_ms": created_ts_ms,
                                    "expires_at_ms": expires_at_ms,
                                    "expire_reason": "stale",
                                    "cache_hit": True,
                                    "cache_source": cache_source,
                                    "miss_count": miss_count,
                                    "extra": {
                                        "near_hit": near_hit,
                                        "touch_candidate": touch_candidate,
                                        "touch_confirmed": False,
                                        "trigger_action": "skip_stale",
                                        "stale_reason": stale_reason,
                                        "refresh_attempted": refresh_attempted,
                                        "ticker_age_ms": ticker_age_ms,
                                        "px_used": px_used,
                                        "touch_eps_bps": touch_eps_bps_used,
                                        "dist_to_band_bps": dist_to_band_bps,
                                        "bid": bid,
                                        "ask": ask,
                                    },
                                }
                            )
                            live_item["next_check_at_ms"] = self._now_ms() + interval_ms
                            continue

                if not v2:
                    eps = abs(trigger_price) * (eps_bps / 10000.0) if eps_bps else 0.0
                    hit = price is not None and self._fast_watch_hit(
                        price=price,
                        trigger_price=trigger_price,
                        eps=eps,
                        trigger_kind=trigger_kind,
                        side=side,
                    )

                checks_done += 1
                live_item["fast_checks_done"] = checks_done
                live_item["attempts"] = checks_done
                live_item["last_attempt_ts_ms"] = now_ms

                if hit:
                    live_item["state"] = "awaiting_recheck"
                    live_item["fast_watch_last_trigger_ts_ms"] = now_ms
                    check_detail = {
                        "fast_watch": {
                            "price": px_used if v2 else price,
                            "trigger_price": trigger_price,
                            "eps_bps": eps_bps,
                            "trigger_kind": trigger_kind,
                            "near_hit": near_hit,
                            "touch_candidate": touch_candidate,
                            "touch_confirmed": touch_confirmed,
                            "trigger_action": trigger_action,
                            "stale_reason": stale_reason,
                            "refresh_attempted": refresh_attempted,
                            "ticker_age_ms": ticker_age_ms,
                            "bid": bid,
                            "ask": ask,
                            "px_used": px_used,
                            "touch_eps_bps": touch_eps_bps_used,
                            "dist_to_band_bps": dist_to_band_bps,
                            "allow_touch_entry": (v2.get("allow_touch_entry") if v2 else None),
                        }
                    }
                    triggers.append((live_item, check_detail))
                    outcomes.append(
                        {
                            "item": live_item,
                            "condition_data": condition_data,
                            "outcome": "triggered",
                            "checks_done": checks_done,
                            "price": px_used if v2 else price,
                            "created_ts_ms": created_ts_ms,
                            "expires_at_ms": expires_at_ms,
                            "expire_reason": "hit",
                            "cache_hit": True,
                            "cache_source": cache_source,
                            "miss_count": miss_count,
                            "extra": {
                                "near_hit": near_hit,
                                "touch_candidate": touch_candidate,
                                "touch_confirmed": touch_confirmed,
                                "trigger_action": trigger_action,
                                "stale_reason": stale_reason,
                                "refresh_attempted": refresh_attempted,
                                "ticker_age_ms": ticker_age_ms,
                                "px_used": px_used,
                                "touch_eps_bps": touch_eps_bps_used,
                                "dist_to_band_bps": dist_to_band_bps,
                                "bid": bid,
                                "ask": ask,
                            }
                            if v2
                            else None,
                        }
                    )
                    continue

                if max_checks and checks_done >= max_checks:
                    removed = self._incubator_items.pop(dedupe_key, None)
                    if removed:
                        outcomes.append(
                            {
                                "item": removed,
                                "condition_data": condition_data,
                                "outcome": "max_checks",
                                "checks_done": checks_done,
                                "price": price,
                                "created_ts_ms": created_ts_ms,
                                "expires_at_ms": expires_at_ms,
                                "expire_reason": "max_checks",
                                "cache_hit": True,
                                "cache_source": cache_source,
                                "miss_count": miss_count,
                            }
                        )
                    continue

                live_item["next_check_at_ms"] = self._now_ms() + interval_ms

        for outcome in outcomes:
            self._emit_fast_watch_outcome(
                outcome["item"],
                condition_data=outcome.get("condition_data") or {},
                outcome=outcome.get("outcome") or "expired",
                checks_done=int(outcome.get("checks_done") or 0),
                price=outcome.get("price"),
                created_ts_ms=outcome.get("created_ts_ms"),
                now_ts_ms=now_ms,
                expires_at_ms=outcome.get("expires_at_ms"),
                expire_reason=outcome.get("expire_reason"),
                cache_hit=outcome.get("cache_hit"),
                cache_source=outcome.get("cache_source"),
                miss_count=outcome.get("miss_count"),
                extra=outcome.get("extra"),
            )
            if outcome.get("outcome") in ("expired", "max_checks"):
                self._emit_fast_watch_waiting_room_compat(
                    outcome["item"],
                    outcome=str(outcome.get("outcome")),
                    expire_reason=outcome.get("expire_reason"),
                    price=outcome.get("price"),
                    now_ts_ms=now_ms,
                )

        for removed, check_detail in triggers:
            self._emit_strategy_recheck_request(removed, check_detail=check_detail)
            self._notify_recheck_ready()

        self._fast_watch_check_counter += 1
        if self._fast_watch_check_sample_rate and self._fast_watch_check_counter % self._fast_watch_check_sample_rate == 0:
            try:
                run_id = get_current_run_id()
            except Exception:
                run_id = None
            sample = {
                "event": "fast_watch_check_sampled",
                "ts_ms": self._now_ms(),
                "run_id": run_id,
                "due_items": len(due_items),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            }
            safe_sample = self._json_sanitize(sample)
            try:
                logger.info("fast_watch_check_sampled %s", json.dumps(safe_sample, ensure_ascii=False, sort_keys=True))
            except Exception:
                logger.info("fast_watch_check_sampled %s", safe_sample)

        return len(due_items)

    async def _micro_gate_watch_tick(self) -> int:
        if not self._micro_gate_watch_enabled:
            return 0

        start_ms = self._now_ms()
        now_ms = start_ms
        max_items = self._micro_gate_watch_max_items_per_tick
        time_budget_ms = self._micro_gate_watch_time_budget_ms
        due_items: List[Dict[str, Any]] = []
        expired_items: List[Tuple[Dict[str, Any], int, int]] = []

        async with self._incubator_lock:
            for dedupe_key, item in list(self._incubator_items.items()):
                if self._now_ms() - start_ms >= time_budget_ms:
                    break
                if str(item.get("watch_kind") or "").lower() != "micro_gate_watch":
                    continue
                payload = item.get("payload") if isinstance(item.get("payload"), dict) else None
                if not isinstance(payload, dict):
                    continue

                expires_at_ms, created_ts_ms = self._resolve_micro_gate_watch_expiry(item, payload, now_ms)
                if expires_at_ms and now_ms >= expires_at_ms:
                    removed = self._incubator_items.pop(dedupe_key, None)
                    if removed:
                        expired_items.append((removed, expires_at_ms, created_ts_ms))
                    continue

                try:
                    next_check_at_ms = int(item.get("next_check_at_ms") or 0)
                except Exception:
                    next_check_at_ms = 0
                if now_ms >= next_check_at_ms:
                    due_items.append(
                        {
                            "dedupe_key": dedupe_key,
                            "pending_id": item.get("pending_id"),
                            "payload": dict(payload),
                            "watch_interval_ms": item.get("watch_interval_ms"),
                            "max_checks": item.get("max_checks"),
                            "checks_done": int(item.get("checks_done", 0) or 0),
                            "expires_at_ms": expires_at_ms,
                            "created_ts_ms": created_ts_ms,
                        }
                    )
                    if len(due_items) >= max_items:
                        break

        for removed, expires_at_ms, created_ts_ms in expired_items:
            payload = removed.get("payload") if isinstance(removed.get("payload"), dict) else {}
            price_ctx = self._build_micro_gate_watch_price_context(
                removed,
                payload,
                price=None,
                price_source=None,
                now_ts_ms=now_ms,
            )
            remaining_ttl_ms = None
            if expires_at_ms:
                remaining_ttl_ms = int(expires_at_ms - now_ms)
            self._emit_micro_gate_watch_outcome(
                removed,
                outcome="expired",
                drop_reason="ttl",
                checks_done=int(removed.get("checks_done", 0) or removed.get("attempts", 0) or 0),
                max_checks=removed.get("max_checks"),
                interval_ms=removed.get("watch_interval_ms"),
                remaining_ttl_ms=remaining_ttl_ms,
                price_ctx=price_ctx,
                gate_detail=None,
            )
            extra = {
                "drop_kind": "micro_gate_watch",
                "drop_reason": "ttl",
                "price": price_ctx.get("price"),
                "price_source": price_ctx.get("price_source"),
            }
            if price_ctx.get("price_imputed"):
                extra["price_imputed"] = True
                extra["imputed_from"] = price_ctx.get("imputed_from")
                if price_ctx.get("last_price_age_ms") is not None:
                    extra["last_price_age_ms"] = price_ctx.get("last_price_age_ms")
            self._emit_waiting_room_event("waiting_room_drop", removed, **extra)

        if not due_items:
            return 0

        price_by_key: Dict[Tuple[str, str, Optional[str]], Tuple[Optional[float], Optional[str]]] = {}
        for item in due_items:
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            symbol = payload.get("symbol")
            timeframe = payload.get("timeframe") or "5m"
            exchange = payload.get("exchange") or payload.get("exchange_name")
            if not symbol:
                continue
            key = (str(symbol), str(timeframe), str(exchange) if exchange else None)
            if key in price_by_key:
                continue
            price_by_key[key] = await self._get_micro_gate_watch_price(
                str(symbol),
                timeframe=str(timeframe),
                exchange=str(exchange) if exchange else None,
            )

        processed = 0

        for entry in due_items:
            if self._now_ms() - start_ms >= time_budget_ms:
                break
            dedupe_key = entry.get("dedupe_key")
            pending_id = entry.get("pending_id")
            payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
            if not dedupe_key or not isinstance(payload, dict):
                continue
            symbol = payload.get("symbol")
            timeframe = payload.get("timeframe") or "5m"
            exchange = payload.get("exchange") or payload.get("exchange_name")
            key = (str(symbol), str(timeframe), str(exchange) if exchange else None)
            raw_price, raw_source = price_by_key.get(key, (None, None))

            async with self._incubator_lock:
                live_item = self._incubator_items.get(dedupe_key)
                if not live_item:
                    continue
                if pending_id is not None and str(live_item.get("pending_id")) != str(pending_id):
                    continue
                if str(live_item.get("watch_kind") or "").lower() != "micro_gate_watch":
                    continue
                interval_ms = int(live_item.get("watch_interval_ms") or self._micro_gate_watch_interval_ms_default)
                max_checks = int(live_item.get("max_checks") or self._micro_gate_watch_max_checks_default)
                checks_done = int(live_item.get("checks_done", 0) or 0)
                expires_at_ms = live_item.get("expires_at_ms") or entry.get("expires_at_ms")
                if not expires_at_ms:
                    expires_at_ms, _ = self._resolve_micro_gate_watch_expiry(live_item, payload, now_ms)
                remaining_ttl_ms = None
                if expires_at_ms:
                    remaining_ttl_ms = int(expires_at_ms - now_ms)

            price_ctx = self._build_micro_gate_watch_price_context(
                live_item,
                payload,
                price=raw_price,
                price_source=raw_source,
                now_ts_ms=now_ms,
            )
            if raw_price is not None and raw_source:
                async with self._incubator_lock:
                    live_item = self._incubator_items.get(dedupe_key)
                    if live_item and str(live_item.get("watch_kind") or "").lower() == "micro_gate_watch":
                        live_item["last_known_price"] = price_ctx.get("price")
                        live_item["last_known_ts_ms"] = now_ms
            if not live_item:
                continue

            checks_done += 1
            gate_clear = False
            gate_detail: Dict[str, Any] = {}
            if symbol:
                gate_clear, gate_detail = await self._check_micro_gate_watch_release_condition(
                    payload,
                    price=price_ctx.get("price"),
                    price_source=price_ctx.get("price_source"),
                )

            interval_policy = "timer_only"
            next_check_in_ms: Optional[int] = None
            if not gate_clear and checks_done < max_checks:
                next_check_in_ms = int(interval_ms)
                if remaining_ttl_ms is not None:
                    next_check_in_ms = max(0, min(int(interval_ms), int(remaining_ttl_ms)))

            tick_detail = {
                "micro_gate_watch": {
                    "checks_done": checks_done,
                    "max_checks": max_checks,
                    "interval_ms": interval_ms,
                    "price": price_ctx.get("price"),
                    "price_source": price_ctx.get("price_source"),
                    "price_imputed": price_ctx.get("price_imputed"),
                }
            }
            self._emit_micro_gate_watch_tick(
                live_item,
                checks_done=checks_done,
                max_checks=max_checks,
                interval_ms=interval_ms,
                remaining_ttl_ms=remaining_ttl_ms,
                price_ctx=price_ctx,
                gate_detail=gate_detail,
                check_detail=tick_detail,
                interval_policy=interval_policy,
                next_check_in_ms=next_check_in_ms,
            )

            if gate_clear:
                refreshed_signal = await self._apply_refresh_policy(
                    dict(payload),
                    str(live_item.get("refresh_policy") or "NONE"),
                    price_override=price_ctx.get("price"),
                    price_source=price_ctx.get("price_source"),
                )
                refreshed_signal["incubator_replay"] = True
                strategy_name = payload.get("strategy_name") or payload.get("strategy") or ""
                try:
                    result = await self.process_strategy_signal(strategy_name=str(strategy_name), signal=refreshed_signal)
                except Exception as exc:
                    result = {"status": "error", "reason": str(exc), "stage": "micro_gate_watch_replay"}

                status_raw = result.get("status")
                status = str(status_raw or "").strip().lower()
                replay_reason_code = str(result.get("reason_code") or "")
                replay_blocked = status == "incubated" and replay_reason_code == "volume.low_vol_tight_stop"
                if replay_blocked:
                    gate_detail = dict(gate_detail or {})
                    gate_detail["replay_status"] = status
                    gate_detail["replay_reason_code"] = replay_reason_code
                else:
                    queued_statuses = {"queued", "enqueued"}
                    accepted_statuses = {"accepted", "active", "executing", "executed"}
                    success_statuses = queued_statuses | accepted_statuses
                    drop_reason = None
                    outcome = "accepted" if status in success_statuses else "dropped"
                    if outcome == "dropped":
                        drop_reason = result.get("reason_code") or result.get("reason") or "replay_failed"

                    async with self._incubator_lock:
                        live_item = self._incubator_items.get(dedupe_key)
                        if live_item and str(live_item.get("watch_kind") or "").lower() == "micro_gate_watch":
                            if pending_id is None or str(live_item.get("pending_id")) == str(pending_id):
                                self._incubator_items.pop(dedupe_key, None)

                    self._emit_micro_gate_watch_outcome(
                        live_item or entry,
                        outcome=outcome,
                        drop_reason=drop_reason,
                        checks_done=checks_done,
                        max_checks=max_checks,
                        interval_ms=interval_ms,
                        remaining_ttl_ms=remaining_ttl_ms,
                        price_ctx=price_ctx,
                        gate_detail=gate_detail,
                    )
                    if outcome == "accepted":
                        self._emit_waiting_room_event("waiting_room_accept", live_item or entry)
                    else:
                        self._emit_waiting_room_event(
                            "waiting_room_drop",
                            live_item or entry,
                            drop_kind="micro_gate_watch",
                            drop_reason=drop_reason or "replay_failed",
                        )
                    processed += 1
                    continue

            if checks_done >= max_checks:
                async with self._incubator_lock:
                    live_item = self._incubator_items.get(dedupe_key)
                    if live_item and str(live_item.get("watch_kind") or "").lower() == "micro_gate_watch":
                        if pending_id is None or str(live_item.get("pending_id")) == str(pending_id):
                            self._incubator_items.pop(dedupe_key, None)
                self._emit_micro_gate_watch_outcome(
                    live_item or entry,
                    outcome="dropped",
                    drop_reason="gate_still_blocked",
                    checks_done=checks_done,
                    max_checks=max_checks,
                    interval_ms=interval_ms,
                    remaining_ttl_ms=remaining_ttl_ms,
                    price_ctx=price_ctx,
                    gate_detail=gate_detail,
                )
                self._emit_waiting_room_event(
                    "waiting_room_drop",
                    live_item or entry,
                    drop_kind="micro_gate_watch",
                    drop_reason="gate_still_blocked",
                    checks_done=checks_done,
                    max_checks=max_checks,
                    interval_ms=interval_ms,
                )
                processed += 1
            else:
                async with self._incubator_lock:
                    live_item = self._incubator_items.get(dedupe_key)
                    if live_item and str(live_item.get("watch_kind") or "").lower() == "micro_gate_watch":
                        if pending_id is None or str(live_item.get("pending_id")) == str(pending_id):
                            live_item["checks_done"] = checks_done
                            live_item["attempts"] = checks_done
                            live_item["last_attempt_ts_ms"] = now_ms
                            next_delay_ms = int(next_check_in_ms) if next_check_in_ms is not None else int(interval_ms)
                            live_item["next_check_at_ms"] = now_ms + next_delay_ms
                processed += 1

        return processed

    def _resolve_fast_watch_expiry(
        self,
        item: Dict[str, Any],
        payload: Dict[str, Any],
        condition_data: Dict[str, Any],
        now_ms: int,
    ) -> Tuple[Optional[int], Optional[str], int]:
        created_ts_ms = self._coerce_int(
            item.get("fast_created_ts_ms")
            or item.get("first_seen_ts_ms")
            or payload.get("timestamp")
            or payload.get("setup_anchor_ts_ms")
        )
        if created_ts_ms is None or created_ts_ms <= 0:
            created_ts_ms = now_ms

        ttl_raw = condition_data.get("ttl_ms") or condition_data.get("watch_ttl_ms")
        ttl_val = self._coerce_int(ttl_raw) if ttl_raw is not None else None
        if ttl_val is None or ttl_val <= 0:
            ttl_val = self._fast_watch_ttl_ms_default

        expires_at_ms: Optional[int] = None
        expire_reason: Optional[str] = None

        if ttl_val and ttl_val > 0:
            expires_at_ms = created_ts_ms + int(ttl_val)
            expire_reason = "ttl"

        raw_exp = condition_data.get("expires_at_ms") or condition_data.get("watch_expires_at_ms")
        exp_override = self._coerce_int(raw_exp)
        if exp_override and exp_override > now_ms:
            if expires_at_ms is None or exp_override < expires_at_ms:
                expires_at_ms = exp_override
                expire_reason = "explicit"

        timeframe = payload.get("timeframe") or payload.get("tf")
        tf_ms = self._parse_timeframe_ms(timeframe) or 0
        if tf_ms:
            next_boundary = int(now_ms - (now_ms % int(tf_ms)) + int(tf_ms))
            if next_boundary > now_ms:
                if expires_at_ms is None or next_boundary < expires_at_ms:
                    expires_at_ms = next_boundary
                    expire_reason = "candle_boundary"

        if expires_at_ms is not None and expires_at_ms <= created_ts_ms:
            if ttl_val and ttl_val > 0:
                expires_at_ms = created_ts_ms + int(ttl_val)
                expire_reason = "ttl"
            elif self._fast_watch_ttl_ms_default:
                expires_at_ms = created_ts_ms + int(self._fast_watch_ttl_ms_default)
                expire_reason = "ttl"

        return expires_at_ms, expire_reason, int(created_ts_ms)

    def _resolve_fast_watch_interval_ms(self, condition_data: Dict[str, Any]) -> int:
        interval_ms = condition_data.get("watch_interval_ms") or condition_data.get("interval_ms")
        interval_val = self._coerce_int(interval_ms)
        if interval_val is None or interval_val <= 0:
            interval_val = self._fast_watch_interval_ms
        return max(250, int(interval_val))

    def _resolve_fast_watch_max_checks(self, condition_data: Dict[str, Any], item: Dict[str, Any]) -> int:
        raw = condition_data.get("max_checks")
        max_checks = self._coerce_int(raw)
        if max_checks is None or max_checks <= 0:
            max_checks = self._coerce_int(item.get("max_attempts"))
        if max_checks is None or max_checks <= 0:
            max_checks = self._fast_watch_max_checks_default
        return max(1, int(max_checks))

    def _resolve_micro_gate_watch_settings(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        condition_data = payload.get("condition_data") if isinstance(payload.get("condition_data"), dict) else {}
        watch_kind = None
        if isinstance(condition_data, dict):
            watch_kind = condition_data.get("watch_kind")
        if watch_kind is None:
            watch_kind = payload.get("watch_kind")
        if str(watch_kind or "").lower() != "micro_gate_watch":
            return None
        interval_ms = self._resolve_micro_gate_watch_interval_ms(condition_data if isinstance(condition_data, dict) else {})
        max_checks = self._resolve_micro_gate_watch_max_checks(condition_data if isinstance(condition_data, dict) else {})
        ttl_ms = self._resolve_micro_gate_watch_ttl_ms(condition_data if isinstance(condition_data, dict) else {})
        return {
            "watch_kind": "micro_gate_watch",
            "interval_ms": interval_ms,
            "max_checks": max_checks,
            "ttl_ms": ttl_ms,
        }

    def _resolve_micro_gate_watch_interval_ms(self, condition_data: Dict[str, Any]) -> int:
        interval_ms = condition_data.get("watch_interval_ms") or condition_data.get("interval_ms")
        interval_val = self._coerce_int(interval_ms)
        if interval_val is None or interval_val <= 0:
            interval_val = self._micro_gate_watch_interval_ms_default
        return max(250, int(interval_val))

    def _resolve_micro_gate_watch_max_checks(self, condition_data: Dict[str, Any]) -> int:
        raw = condition_data.get("max_checks")
        max_checks = self._coerce_int(raw)
        if max_checks is None or max_checks <= 0:
            max_checks = self._micro_gate_watch_max_checks_default
        return max(1, int(max_checks))

    def _resolve_micro_gate_watch_ttl_ms(self, condition_data: Dict[str, Any]) -> int:
        ttl_ms = condition_data.get("ttl_ms") or condition_data.get("watch_ttl_ms")
        ttl_val = self._coerce_int(ttl_ms)
        if ttl_val is None or ttl_val <= 0:
            ttl_val = self._micro_gate_watch_ttl_ms_default
        return max(1_000, int(ttl_val))

    def _resolve_micro_gate_watch_expiry(
        self,
        item: Dict[str, Any],
        payload: Dict[str, Any],
        now_ms: int,
    ) -> Tuple[Optional[int], int]:
        created_ts_ms = self._coerce_int(
            item.get("watch_created_ts_ms")
            or item.get("first_seen_ts_ms")
            or payload.get("timestamp")
            or payload.get("setup_anchor_ts_ms")
        )
        if created_ts_ms is None or created_ts_ms <= 0:
            created_ts_ms = now_ms

        condition_data = payload.get("condition_data") if isinstance(payload.get("condition_data"), dict) else {}
        ttl_val = self._coerce_int(item.get("watch_ttl_ms")) if item.get("watch_ttl_ms") is not None else None
        if ttl_val is None or ttl_val <= 0:
            ttl_val = self._resolve_micro_gate_watch_ttl_ms(condition_data if isinstance(condition_data, dict) else {})

        expires_at_ms = self._coerce_int(item.get("expires_at_ms"))
        if expires_at_ms is None or expires_at_ms <= 0:
            expires_at_ms = created_ts_ms + int(ttl_val)

        if expires_at_ms <= created_ts_ms:
            expires_at_ms = created_ts_ms + int(ttl_val)

        item.setdefault("watch_created_ts_ms", created_ts_ms)
        item.setdefault("watch_ttl_ms", ttl_val)
        item.setdefault("expires_at_ms", expires_at_ms)

        return expires_at_ms, int(created_ts_ms)

    async def _get_cache_only_price(
        self,
        symbol: str,
        *,
        timeframe: str,
        exchange: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        pipeline = getattr(self, "market_data_pipeline", None)
        if pipeline and hasattr(pipeline, "get_latest_price_cache_only"):
            try:
                price = await pipeline.get_latest_price_cache_only(symbol, timeframe=timeframe, exchange=exchange)
                return price, "cache_only"
            except Exception:
                return None, "cache_only"
        if pipeline and hasattr(pipeline, "get_realtime_price"):
            try:
                price = pipeline.get_realtime_price(symbol, timeframe=timeframe, exchange=exchange)
                return price, "forming_candle"
            except Exception:
                return None, "forming_candle"
        return None, None

    async def _get_micro_gate_watch_price(
        self,
        symbol: str,
        *,
        timeframe: str,
        exchange: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        price, source = await self._get_cache_only_price(symbol, timeframe=timeframe, exchange=exchange)
        if price is not None:
            return price, source
        pipeline = getattr(self, "market_data_pipeline", None)
        if pipeline and hasattr(pipeline, "get_latest_price"):
            try:
                price = await pipeline.get_latest_price(symbol, timeframe=timeframe)
                return price, "market_price"
            except Exception:
                return None, "market_price"
        return None, None

    @staticmethod
    def _normalize_exchange_key(exchange: Optional[str]) -> Optional[str]:
        if exchange is None:
            return None
        try:
            ex = str(exchange).strip().lower()
        except Exception:
            return None
        return ex or None

    def _get_strategy_fast_watch_v2_cfg(self, strategy_name: Optional[str]) -> Dict[str, Any]:
        if not strategy_name:
            return {}
        cfg = self.config if isinstance(self.config, dict) else {}
        strat_cfg = cfg.get("strategies", {}) if isinstance(cfg, dict) else {}
        if not isinstance(strat_cfg, dict):
            return {}
        entry = strat_cfg.get(str(strategy_name))
        if not isinstance(entry, dict):
            return {}
        fw = entry.get("fast_watch")
        if fw is None:
            return {}
        if not isinstance(fw, dict):
            return {}
        return dict(fw)

    def _resolve_fast_watch_v2_settings(
        self,
        *,
        strategy_name: Optional[str],
        condition_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        base = self._get_strategy_fast_watch_v2_cfg(strategy_name)
        override = condition_data.get("fast_watch")
        if override is not None and not isinstance(override, dict):
            override = None
        merged = {**(base or {}), **(override or {})}

        v2_keys = {
            "near_bps",
            "touch_eps_bps",
            "near_sigma_delta",
            "touch_sigma_delta",
            "min_touch_abs_px",
            "max_touch_abs_px",
            "min_near_abs_px",
            "max_near_abs_px",
            "touch_price_source",
            "recheck_freshness_ms",
            "max_ticker_age_ms",
            "recheck_cooldown_ms",
            "allow_touch_entry",
        }
        if not any(k in merged for k in v2_keys):
            return None

        def _coerce_float_local(val: Any) -> Optional[float]:
            try:
                fval = float(val)
                if not math.isfinite(fval):
                    return None
                return fval
            except Exception:
                return None

        def _coerce_int_local(val: Any) -> Optional[int]:
            try:
                ival = int(val)
                return ival
            except Exception:
                return None

        near_bps = _coerce_float_local(merged.get("near_bps"))
        if near_bps is None:
            near_bps = _coerce_float_local(condition_data.get("eps_bps"))
        if near_bps is None:
            near_bps = 10.0
        near_bps = max(0.0, float(near_bps))

        touch_eps_bps = _coerce_float_local(merged.get("touch_eps_bps"))
        if touch_eps_bps is None:
            touch_eps_bps = 3.0
        touch_eps_bps = max(0.0, float(touch_eps_bps))

        near_sigma_delta = _coerce_float_local(merged.get("near_sigma_delta"))
        if near_sigma_delta is not None:
            near_sigma_delta = max(0.0, float(near_sigma_delta))

        touch_sigma_delta = _coerce_float_local(merged.get("touch_sigma_delta"))
        if touch_sigma_delta is not None:
            touch_sigma_delta = max(0.0, float(touch_sigma_delta))

        min_touch_abs_px = _coerce_float_local(merged.get("min_touch_abs_px"))
        if min_touch_abs_px is not None:
            min_touch_abs_px = max(0.0, float(min_touch_abs_px))

        max_touch_abs_px = _coerce_float_local(merged.get("max_touch_abs_px"))
        if max_touch_abs_px is not None:
            max_touch_abs_px = max(0.0, float(max_touch_abs_px))

        min_near_abs_px = _coerce_float_local(merged.get("min_near_abs_px"))
        if min_near_abs_px is not None:
            min_near_abs_px = max(0.0, float(min_near_abs_px))

        max_near_abs_px = _coerce_float_local(merged.get("max_near_abs_px"))
        if max_near_abs_px is not None:
            max_near_abs_px = max(0.0, float(max_near_abs_px))

        freshness_ms = _coerce_int_local(merged.get("max_ticker_age_ms"))
        if freshness_ms is None:
            freshness_ms = _coerce_int_local(merged.get("recheck_freshness_ms"))
        if freshness_ms is None:
            freshness_ms = 500
        freshness_ms = max(0, int(freshness_ms))

        cooldown_ms = _coerce_int_local(merged.get("recheck_cooldown_ms"))
        if cooldown_ms is None:
            cooldown_ms = 1000
        cooldown_ms = max(0, int(cooldown_ms))

        try:
            src = str(merged.get("touch_price_source", "bidask") or "").strip().lower()
        except Exception:
            src = "bidask"
        if not src:
            src = "bidask"

        try:
            allow_touch_entry = bool(merged.get("allow_touch_entry", True))
        except Exception:
            allow_touch_entry = True

        return {
            "near_bps": near_bps,
            "touch_eps_bps": touch_eps_bps,
            "near_sigma_delta": near_sigma_delta,
            "touch_sigma_delta": touch_sigma_delta,
            "min_touch_abs_px": min_touch_abs_px,
            "max_touch_abs_px": max_touch_abs_px,
            "min_near_abs_px": min_near_abs_px,
            "max_near_abs_px": max_near_abs_px,
            "touch_price_source": src,
            "max_ticker_age_ms": freshness_ms,
            "recheck_cooldown_ms": cooldown_ms,
            "allow_touch_entry": allow_touch_entry,
        }

    async def _get_bidask_metrics(
        self,
        symbol: str,
        *,
        exchange: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        pipeline = getattr(self, "market_data_pipeline", None)
        if not pipeline or not hasattr(pipeline, "get_spread_metrics"):
            return None
        try:
            return await pipeline.get_spread_metrics(
                symbol,
                exchange=exchange,
                allow_rest_fallback=False,
            )
        except Exception:
            return None

    def _build_fast_watch_price_context(
        self,
        item: Dict[str, Any],
        price: Optional[float],
        *,
        now_ts_ms: Optional[int],
        outcome: str,
    ) -> Dict[str, Any]:
        ts_ms = int(now_ts_ms if now_ts_ms is not None else self._now_ms())
        last_price = self._coerce_float(item.get("last_known_price") if isinstance(item, dict) else None)
        last_price_ts_ms = self._coerce_int(item.get("last_known_ts_ms") if isinstance(item, dict) else None)
        price_used = price
        price_imputed = False
        imputed_from = None
        last_price_age_ms = None
        if price is None and outcome == "expired" and last_price is not None:
            price_used = last_price
            price_imputed = True
            imputed_from = "last_known_price"
            if last_price_ts_ms is not None:
                last_price_age_ms = max(0, int(ts_ms - last_price_ts_ms))
        return {
            "price": price_used,
            "last_price": last_price,
            "last_price_ts_ms": last_price_ts_ms,
            "price_imputed": price_imputed,
            "imputed_from": imputed_from,
            "last_price_age_ms": last_price_age_ms,
        }

    def _build_micro_gate_watch_price_context(
        self,
        item: Dict[str, Any],
        payload: Dict[str, Any],
        *,
        price: Optional[float],
        price_source: Optional[str],
        now_ts_ms: Optional[int],
    ) -> Dict[str, Any]:
        ts_ms = int(now_ts_ms if now_ts_ms is not None else self._now_ms())
        last_price = self._coerce_float(item.get("last_known_price") if isinstance(item, dict) else None)
        last_price_ts_ms = self._coerce_int(item.get("last_known_ts_ms") if isinstance(item, dict) else None)
        price_used = price
        price_src = price_source
        price_imputed = False
        imputed_from = None
        if price_used is None:
            if last_price is not None:
                price_used = last_price
                price_src = "last_known_price"
                price_imputed = True
                imputed_from = "last_known_price"
            else:
                entry_val = None
                if isinstance(payload, dict):
                    entry_val = self._coerce_float(
                        payload.get("entry") or payload.get("entry_price") or payload.get("price")
                    )
                if entry_val is not None:
                    price_used = entry_val
                    price_src = "payload_entry"
                    price_imputed = True
                    imputed_from = "payload_entry"

        last_price_age_ms = None
        if price_imputed and last_price_ts_ms is not None:
            try:
                last_price_age_ms = max(0, int(ts_ms - int(last_price_ts_ms)))
            except Exception:
                last_price_age_ms = None

        return {
            "price": price_used,
            "price_source": price_src,
            "price_imputed": price_imputed,
            "imputed_from": imputed_from,
            "last_price": last_price,
            "last_price_ts_ms": last_price_ts_ms,
            "last_price_age_ms": last_price_age_ms,
        }

    def _emit_fast_watch_outcome(
        self,
        item: Dict[str, Any],
        *,
        condition_data: Dict[str, Any],
        outcome: str,
        checks_done: int,
        price: Optional[float],
        created_ts_ms: Optional[int] = None,
        now_ts_ms: Optional[int] = None,
        expires_at_ms: Optional[int] = None,
        expire_reason: Optional[str] = None,
        cache_hit: Optional[bool] = None,
        cache_source: Optional[str] = None,
        miss_count: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        ts_ms = now_ts_ms if now_ts_ms is not None else self._now_ms()
        try:
            run_id = get_current_run_id()
        except Exception:
            run_id = None

        parent_pending_id = item.get("parent_pending_id") if isinstance(item, dict) else None
        if parent_pending_id is not None:
            try:
                parent_pending_id = str(parent_pending_id)
            except Exception:
                parent_pending_id = None

        trigger_price = self._coerce_float(condition_data.get("trigger_price"))
        eps_bps = self._coerce_float(condition_data.get("eps_bps"))
        near_val = condition_data.get("near") if isinstance(condition_data, dict) else None
        try:
            near_str = str(near_val) if near_val is not None else "unknown"
        except Exception:
            near_str = "unknown"
        if not near_str.strip():
            near_str = "unknown"
        if created_ts_ms is None:
            created_ts_ms = self._coerce_int(item.get("fast_created_ts_ms") or item.get("first_seen_ts_ms"))
        if expires_at_ms is None:
            expires_at_ms = self._coerce_int(item.get("fast_expires_at_ms") or item.get("expires_at_ms"))
        if expire_reason is None:
            expire_reason = item.get("fast_expire_reason")
        if cache_hit is None:
            cache_hit = price is not None
        if miss_count is None:
            miss_count = self._coerce_int(item.get("fast_miss_count") or 0) or 0

        price_ctx = self._build_fast_watch_price_context(
            item,
            price,
            now_ts_ms=ts_ms,
            outcome=str(outcome),
        )
        out = {
            "event": "fast_watch_outcome",
            "ts_ms": ts_ms,
            "now_ts_ms": ts_ms,
            "created_ts_ms": created_ts_ms,
            "expires_at_ms": expires_at_ms,
            "expire_reason": expire_reason,
            "run_id": run_id,
            "pending_id": item.get("pending_id") if isinstance(item, dict) else None,
            "parent_pending_id": parent_pending_id,
            "signal_id": payload.get("signal_id") if isinstance(payload, dict) else None,
            "strategy": payload.get("strategy_name") or payload.get("strategy"),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "timeframe": payload.get("timeframe") or payload.get("tf"),
            "outcome": str(outcome),
            "checks_done": int(checks_done),
            "elapsed_ms": ts_ms - int(item.get("first_seen_ts_ms", ts_ms) or ts_ms),
            "price": price_ctx.get("price"),
            "last_price": price_ctx.get("last_price"),
            "last_price_ts_ms": price_ctx.get("last_price_ts_ms"),
            "trigger_price": trigger_price,
            "eps_bps": eps_bps,
            "near": near_str,
            "cache_hit": cache_hit,
            "cache_source": cache_source,
            "miss_count": miss_count,
        }
        if isinstance(extra, dict) and extra:
            out.update(extra)
        if price_ctx.get("price_imputed"):
            out["price_imputed"] = True
            out["imputed_from"] = price_ctx.get("imputed_from")
            if price_ctx.get("last_price_age_ms") is not None:
                out["last_price_age_ms"] = price_ctx.get("last_price_age_ms")
        safe_out = self._json_sanitize(out)
        try:
            logger.info("fast_watch_outcome %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("fast_watch_outcome %s", safe_out)

    def _emit_fast_watch_rearm(
        self,
        item: Dict[str, Any],
        *,
        interval_ms: int,
        rearm_count: int,
        remaining_ttl_ms: Optional[int],
        reason: str,
    ) -> None:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        out = {
            "event": "fast_watch_rearm",
            "ts_ms": self._now_ms(),
            "run_id": get_current_run_id(),
            "pending_id": item.get("pending_id") if isinstance(item, dict) else None,
            "parent_pending_id": item.get("parent_pending_id") if isinstance(item, dict) else None,
            "signal_id": payload.get("signal_id") if isinstance(payload, dict) else None,
            "strategy": payload.get("strategy_name") or payload.get("strategy"),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "timeframe": payload.get("timeframe") or payload.get("tf"),
            "dedupe_key": item.get("dedupe_key"),
            "interval_ms": int(interval_ms),
            "rearm_count": int(rearm_count),
            "remaining_ttl_ms": remaining_ttl_ms,
            "reason": str(reason),
        }
        safe_out = self._json_sanitize(out)
        try:
            logger.info("fast_watch_rearm %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("fast_watch_rearm %s", safe_out)

    def _emit_fast_watch_waiting_room_compat(
        self,
        item: Dict[str, Any],
        *,
        outcome: str,
        expire_reason: Optional[str] = None,
        price: Optional[float] = None,
        now_ts_ms: Optional[int] = None,
    ) -> None:
        if not item or outcome not in ("triggered", "expired", "max_checks"):
            return
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        near_str = "unknown"
        if isinstance(payload, dict):
            condition_data = payload.get("condition_data")
            if isinstance(condition_data, dict):
                near_val = condition_data.get("near")
                try:
                    near_str = str(near_val) if near_val is not None else "unknown"
                except Exception:
                    near_str = "unknown"
                if not near_str.strip():
                    near_str = "unknown"
        drop_reason = outcome
        if outcome == "triggered":
            drop_reason = "hit"
        elif outcome == "max_checks":
            drop_reason = "max_checks"
        elif outcome == "expired":
            drop_reason = "expired"
        price_ctx = self._build_fast_watch_price_context(
            item,
            price,
            now_ts_ms=now_ts_ms,
            outcome=str(outcome),
        )
        extra = {
            "drop_kind": "fast_watch",
            "drop_reason": drop_reason,
            "fast_watch_outcome": outcome,
            "near": near_str,
            "price": price_ctx.get("price"),
            "last_price": price_ctx.get("last_price"),
            "last_price_ts_ms": price_ctx.get("last_price_ts_ms"),
        }
        if price_ctx.get("price_imputed"):
            extra["price_imputed"] = True
            extra["imputed_from"] = price_ctx.get("imputed_from")
            if price_ctx.get("last_price_age_ms") is not None:
                extra["last_price_age_ms"] = price_ctx.get("last_price_age_ms")
        if expire_reason:
            extra["expire_reason"] = expire_reason
        self._emit_waiting_room_event("waiting_room_drop", item, **extra)

    def _emit_micro_gate_watch_add(self, item: Dict[str, Any]) -> None:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        ts_ms = self._now_ms()
        try:
            run_id = get_current_run_id()
        except Exception:
            run_id = None
        expires_at_ms = self._coerce_int(item.get("expires_at_ms"))
        remaining_ttl_ms = int(expires_at_ms - ts_ms) if expires_at_ms else None
        out = {
            "event": "micro_gate_watch_add",
            "ts_ms": ts_ms,
            "run_id": run_id,
            "pending_id": item.get("pending_id") if isinstance(item, dict) else None,
            "dedupe_key": item.get("dedupe_key") if isinstance(item, dict) else None,
            "reason_code": item.get("reason_code") if isinstance(item, dict) else None,
            "checks_done": int(item.get("checks_done", 0) or 0),
            "max_checks": item.get("max_checks"),
            "interval_ms": item.get("watch_interval_ms"),
            "remaining_ttl_ms": remaining_ttl_ms,
            "symbol": payload.get("symbol") if isinstance(payload, dict) else None,
            "side": payload.get("side") if isinstance(payload, dict) else None,
            "timeframe": payload.get("timeframe") if isinstance(payload, dict) else None,
        }
        condition_data = payload.get("condition_data") if isinstance(payload, dict) else None
        if isinstance(condition_data, dict):
            for key in ("gate_margin_bps", "gate_threshold", "stop_distance", "stop_pct"):
                if condition_data.get(key) is not None:
                    out[key] = condition_data.get(key)
        safe_out = self._json_sanitize(out)
        try:
            logger.info("micro_gate_watch_add %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("micro_gate_watch_add %s", safe_out)

    def _emit_micro_gate_watch_dedupe_drop_incoming(
        self,
        *,
        dedupe_key: Optional[str],
        existing_pending_id: Optional[str],
        incoming_signal_id: Optional[str],
        reason: str,
    ) -> None:
        ts_ms = self._now_ms()
        try:
            run_id = get_current_run_id()
        except Exception:
            run_id = None
        out = {
            "event": "micro_gate_watch_dedupe_drop_incoming",
            "ts_ms": ts_ms,
            "run_id": run_id,
            "dedupe_key": dedupe_key,
            "existing_pending_id": existing_pending_id,
            "incoming_signal_id": incoming_signal_id,
            "reason": reason,
        }
        safe_out = self._json_sanitize(out)
        try:
            logger.info("micro_gate_watch_dedupe_drop_incoming %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("micro_gate_watch_dedupe_drop_incoming %s", safe_out)

    def _emit_micro_gate_watch_tick(
        self,
        item: Dict[str, Any],
        *,
        checks_done: int,
        max_checks: int,
        interval_ms: int,
        remaining_ttl_ms: Optional[int],
        price_ctx: Dict[str, Any],
        gate_detail: Optional[Dict[str, Any]],
        check_detail: Optional[Dict[str, Any]] = None,
        interval_policy: Optional[str] = None,
        next_check_in_ms: Optional[int] = None,
    ) -> None:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        ts_ms = self._now_ms()
        try:
            run_id = get_current_run_id()
        except Exception:
            run_id = None
        out = {
            "event": "micro_gate_watch_tick",
            "ts_ms": ts_ms,
            "run_id": run_id,
            "pending_id": item.get("pending_id") if isinstance(item, dict) else None,
            "dedupe_key": item.get("dedupe_key") if isinstance(item, dict) else None,
            "reason_code": item.get("reason_code") if isinstance(item, dict) else None,
            "checks_done": checks_done,
            "max_checks": max_checks,
            "interval_ms": interval_ms,
            "remaining_ttl_ms": remaining_ttl_ms,
            "price": price_ctx.get("price"),
            "price_source": price_ctx.get("price_source"),
            "symbol": payload.get("symbol") if isinstance(payload, dict) else None,
            "side": payload.get("side") if isinstance(payload, dict) else None,
            "timeframe": payload.get("timeframe") if isinstance(payload, dict) else None,
        }
        if interval_policy:
            out["interval_policy"] = interval_policy
        if next_check_in_ms is not None:
            out["next_check_in_ms"] = int(next_check_in_ms)
        if price_ctx.get("price_imputed"):
            out["price_imputed"] = True
            out["imputed_from"] = price_ctx.get("imputed_from")
            if price_ctx.get("last_price_age_ms") is not None:
                out["last_price_age_ms"] = price_ctx.get("last_price_age_ms")
        if gate_detail:
            out.update(gate_detail)
        if check_detail:
            out["check_detail"] = check_detail
        safe_out = self._json_sanitize(out)
        try:
            logger.info("micro_gate_watch_tick %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("micro_gate_watch_tick %s", safe_out)

    def _emit_micro_gate_watch_outcome(
        self,
        item: Dict[str, Any],
        *,
        outcome: str,
        drop_reason: Optional[str],
        checks_done: int,
        max_checks: Optional[int],
        interval_ms: Optional[int],
        remaining_ttl_ms: Optional[int],
        price_ctx: Dict[str, Any],
        gate_detail: Optional[Dict[str, Any]],
    ) -> None:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        ts_ms = self._now_ms()
        try:
            run_id = get_current_run_id()
        except Exception:
            run_id = None
        out = {
            "event": "micro_gate_watch_outcome",
            "ts_ms": ts_ms,
            "run_id": run_id,
            "pending_id": item.get("pending_id") if isinstance(item, dict) else None,
            "dedupe_key": item.get("dedupe_key") if isinstance(item, dict) else None,
            "reason_code": item.get("reason_code") if isinstance(item, dict) else None,
            "outcome": outcome,
            "drop_reason": drop_reason,
            "checks_done": checks_done,
            "max_checks": max_checks,
            "interval_ms": interval_ms,
            "remaining_ttl_ms": remaining_ttl_ms,
            "price": price_ctx.get("price"),
            "price_source": price_ctx.get("price_source"),
            "symbol": payload.get("symbol") if isinstance(payload, dict) else None,
            "side": payload.get("side") if isinstance(payload, dict) else None,
            "timeframe": payload.get("timeframe") if isinstance(payload, dict) else None,
        }
        if price_ctx.get("price_imputed"):
            out["price_imputed"] = True
            out["imputed_from"] = price_ctx.get("imputed_from")
            if price_ctx.get("last_price_age_ms") is not None:
                out["last_price_age_ms"] = price_ctx.get("last_price_age_ms")
        if gate_detail:
            out.update(gate_detail)
        safe_out = self._json_sanitize(out)
        try:
            logger.info("micro_gate_watch_outcome %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.info("micro_gate_watch_outcome %s", safe_out)

    async def _check_micro_gate_watch_release_condition(
        self,
        payload: Dict[str, Any],
        *,
        price: Optional[float],
        price_source: Optional[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        symbol = payload.get("symbol") if isinstance(payload, dict) else None
        timeframe = (payload.get("timeframe") or "5m") if isinstance(payload, dict) else "5m"
        detail: Dict[str, Any] = {}

        bucket_label = None
        bucket_rank = None
        normal_rank = None
        if symbol and self.volume_analyzer:
            try:
                ctx = await self.volume_analyzer.compute_context(symbol, timeframe)
                bucket = getattr(ctx, "bucket", None) if ctx else None
                bucket_label = str(bucket or "").upper() if bucket is not None else None
                if bucket_label:
                    bucket_rank = get_bucket_rank(bucket_label)
                    normal_rank = get_bucket_rank("NORMAL")
            except Exception as exc:
                detail["volume_error"] = str(exc)
        else:
            detail["volume_error"] = "missing_symbol_or_volume_analyzer"

        if bucket_label is not None:
            detail["bucket"] = bucket_label
        if bucket_rank is not None:
            detail["bucket_rank"] = bucket_rank
        if normal_rank is not None:
            detail["normal_rank"] = normal_rank

        stop_pct, stop_distance, entry_used = self._calc_stop_metrics(payload, price_override=price)
        gate_threshold_bps = LOW_VOL_TIGHT_STOP_THRESHOLD * 10000.0
        gate_margin_bps = None
        if stop_pct is not None and math.isfinite(stop_pct):
            gate_margin_bps = max(0.0, (LOW_VOL_TIGHT_STOP_THRESHOLD - stop_pct) * 10000.0)
        px_used = price if price is not None else entry_used
        px_source = price_source if price is not None else ("entry_price" if entry_used is not None else None)
        market_price = None
        if price is not None and price_source in ("market_price", "cache_only", "forming_candle"):
            market_price = price

        detail.update(
            {
                "gate_threshold": LOW_VOL_TIGHT_STOP_THRESHOLD,
                "gate_threshold_bps": gate_threshold_bps,
                "gate_margin_bps": gate_margin_bps,
                "stop_distance": stop_distance,
                "stop_pct": stop_pct,
                "market_price": market_price,
                "px_used": px_used,
                "px_source": px_source,
            }
        )

        volume_ok = bool(bucket_rank is not None and normal_rank is not None and bucket_rank >= normal_rank)
        stop_ok = bool(stop_pct is not None and stop_pct >= LOW_VOL_TIGHT_STOP_THRESHOLD)
        detail["volume_ok"] = volume_ok
        detail["stop_ok"] = stop_ok

        if volume_ok:
            detail["gate_reason"] = "volume_bucket_normal"
        elif stop_ok:
            detail["gate_reason"] = "stop_distance_ok"
        else:
            detail["gate_reason"] = "gate_still_blocked"

        return volume_ok or stop_ok, detail

    async def on_recheck_result(
        self,
        pending_id: Optional[str],
        *,
        rearm: bool,
        interval_hint_ms: Optional[int] = None,
        decision_meta: Optional[Dict[str, Any]] = None,
        final_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not pending_id:
            return {"status": "error", "reason": "missing_pending_id"}
        pending_id_str = str(pending_id)
        item = None
        dedupe_key = None
        async with self._incubator_lock:
            for key, cand in self._incubator_items.items():
                cand_pending = cand.get("pending_id")
                if cand_pending is not None and str(cand_pending) == pending_id_str:
                    item = cand
                    dedupe_key = key
                    break
        if not item or not dedupe_key:
            return {"status": "missing", "reason": "pending_not_found"}
        if str(item.get("refresh_policy") or "").upper() != "FAST_PRICE_WATCH":
            return {"status": "ignored", "reason": "not_fast_watch"}

        now_ms = self._now_ms()
        expires_at_ms = self._coerce_int(item.get("expires_at_ms") or item.get("fast_expires_at_ms"))
        remaining_ttl_ms = None
        if expires_at_ms:
            remaining_ttl_ms = int(expires_at_ms - now_ms)
        rearm_count = int(item.get("rearm_count") or 0)
        max_rearms = int(item.get("max_rearms") or self._fast_watch_max_rearms)

        if rearm:
            if expires_at_ms and now_ms >= expires_at_ms:
                rearm = False
                final_reason = final_reason or "expired"
            elif rearm_count >= max_rearms:
                rearm = False
                final_reason = final_reason or "max_rearms"

        if rearm:
            base_interval = int(item.get("fast_watch_interval_ms") or self._fast_watch_interval_ms)
            interval_ms = base_interval
            if interval_hint_ms is not None:
                try:
                    interval_ms = int(interval_hint_ms)
                except Exception:
                    interval_ms = base_interval
            else:
                interval_ms = int(base_interval * self._fast_watch_rearm_backoff_mult)
            interval_ms = max(1, interval_ms)
            if self._fast_watch_rearm_max_interval_ms:
                interval_ms = min(interval_ms, int(self._fast_watch_rearm_max_interval_ms))

            async with self._incubator_lock:
                live_item = self._incubator_items.get(dedupe_key)
                if live_item:
                    live_item["state"] = "watching"
                    live_item["rearm_count"] = rearm_count + 1
                    live_item["fast_watch_interval_ms"] = interval_ms
                    live_item["next_check_at_ms"] = now_ms + interval_ms
            self._emit_fast_watch_rearm(
                item,
                interval_ms=interval_ms,
                rearm_count=rearm_count + 1,
                remaining_ttl_ms=remaining_ttl_ms,
                reason=str(decision_meta.get("rearm_reason") if isinstance(decision_meta, dict) else "rearm"),
            )
            return {
                "status": "rearmed",
                "dedupe_key": dedupe_key,
                "rearm_count": rearm_count + 1,
                "remaining_ttl_ms": remaining_ttl_ms,
                "next_interval_ms": interval_ms,
                "expires_at_ms": expires_at_ms,
            }

        drop_reason = final_reason or "recheck_hold"
        if drop_reason == "rearm_limit":
            drop_reason = "max_rearms"
        if rearm_count >= max_rearms:
            drop_reason = "max_rearms"
        async with self._incubator_lock:
            removed = self._incubator_items.pop(dedupe_key, None)
        if removed:
            removed["state"] = "finalized"
            self._emit_waiting_room_event(
                "waiting_room_drop",
                removed,
                drop_kind="fast_watch_final",
                drop_reason=drop_reason,
                rearm_count=rearm_count,
                remaining_ttl_ms=remaining_ttl_ms,
            )
        return {
            "status": "finalized",
            "dedupe_key": dedupe_key,
            "drop_reason": drop_reason,
            "rearm_count": rearm_count,
            "remaining_ttl_ms": remaining_ttl_ms,
        }

    def _emit_fast_watch_downgrade(self, item: Dict[str, Any], *, reason: str) -> None:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        condition_data = payload.get("condition_data") if isinstance(payload, dict) else None
        trigger_price = None
        if isinstance(condition_data, dict):
            trigger_price = condition_data.get("trigger_price")
        out = {
            "event": "fast_watch_downgrade",
            "ts_ms": self._now_ms(),
            "run_id": get_current_run_id(),
            "pending_id": item.get("pending_id") if isinstance(item, dict) else None,
            "parent_pending_id": item.get("parent_pending_id") if isinstance(item, dict) else None,
            "signal_id": payload.get("signal_id") if isinstance(payload, dict) else None,
            "strategy": payload.get("strategy_name") or payload.get("strategy"),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "timeframe": payload.get("timeframe") or payload.get("tf"),
            "reason_code": item.get("reason_code") if isinstance(item, dict) else None,
            "pending_reason_code": item.get("pending_reason_code") if isinstance(item, dict) else None,
            "dedupe_key": item.get("dedupe_key") if isinstance(item, dict) else None,
            "refresh_policy": item.get("refresh_policy") if isinstance(item, dict) else None,
            "downgrade_reason": str(reason),
            "trigger_price": trigger_price,
        }
        safe_out = self._json_sanitize(out)
        try:
            logger.warning("fast_watch_downgrade %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))
        except Exception:
            logger.warning("fast_watch_downgrade %s", safe_out)

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _calc_stop_metrics(
        self, signal: Dict[str, Any], *, price_override: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        entry_val = self._coerce_float(price_override) if price_override is not None else None
        if entry_val is None:
            entry_val = self._coerce_float(
                signal.get("entry") or signal.get("entry_price") or signal.get("price")
            )
        stop_val = self._coerce_float(signal.get("stop") or signal.get("stop_loss"))
        stop_distance = None
        stop_pct = None
        if entry_val is not None and entry_val > 0 and stop_val is not None and stop_val > 0:
            stop_distance = abs(entry_val - stop_val)
            stop_pct = stop_distance / entry_val if entry_val else None
        else:
            stop_pct = self._coerce_float(signal.get("stop_loss_pct"))
        return stop_pct, stop_distance, entry_val

    @staticmethod
    def _fast_watch_hit(
        *,
        price: float,
        trigger_price: float,
        eps: float,
        trigger_kind: str,
        side: str,
    ) -> bool:
        kind = (trigger_kind or "").strip().lower()
        side_norm = (side or "").strip().lower()
        if "lower" in kind:
            return price <= trigger_price + eps
        if "upper" in kind:
            return price >= trigger_price - eps
        if kind in ("leq", "lte", "below"):
            return price <= trigger_price + eps
        if kind in ("geq", "gte", "above"):
            return price >= trigger_price - eps
        if kind in ("band_touch", "touch", "cross"):
            if side_norm in ("long", "buy"):
                return price <= trigger_price + eps
            if side_norm in ("short", "sell"):
                return price >= trigger_price - eps
        return abs(price - trigger_price) <= eps

    def _check_concurrent_release_condition(self, payload: Dict[str, Any], reason_code: str) -> Tuple[bool, Dict[str, Any]]:
        limits = getattr(self.risk_manager, "concurrent_limits", None)
        if limits is None or self.portfolio_manager is None:
            return False, {"error": "missing_limits_or_portfolio_manager"}

        symbol = payload.get("symbol")
        try:
            max_open_positions = getattr(limits, "max_open_positions", None)
            max_positions_per_symbol = getattr(limits, "max_positions_per_symbol", None)
        except Exception:
            max_open_positions = None
            max_positions_per_symbol = None

        try:
            if hasattr(self.portfolio_manager, "count_open_positions"):
                total_open = int(self.portfolio_manager.count_open_positions())
                symbol_open = int(self.portfolio_manager.count_open_positions(symbol)) if symbol else 0
            elif hasattr(self.portfolio_manager, "get_open_positions"):
                open_positions = self.portfolio_manager.get_open_positions() or {}
                total_open = len(open_positions) if isinstance(open_positions, dict) else 0
                symbol_open = sum(1 for pos in (open_positions or {}).values() if pos.get("symbol") == symbol) if symbol else 0
            else:
                total_open = len(getattr(self.risk_manager, "active_positions", {}) or {})
                symbol_open = 0
        except Exception as exc:
            return False, {"error": str(exc)}

        if reason_code == "risk.concurrent.max_open_positions":
            if max_open_positions is None:
                return False, {"error": "missing_max_open_positions", "total_open": total_open}
            return total_open < int(max_open_positions), {"total_open": total_open, "max_open_positions": int(max_open_positions)}

        if reason_code == "risk.concurrent.max_positions_per_symbol":
            if max_positions_per_symbol is None or not symbol:
                return False, {"error": "missing_symbol_or_max_positions_per_symbol", "symbol": symbol}
            return (
                symbol_open < int(max_positions_per_symbol),
                {"symbol": symbol, "symbol_open": symbol_open, "max_positions_per_symbol": int(max_positions_per_symbol)},
            )

        return False, {"error": "unknown_reason_code", "reason_code": reason_code}

    async def _compute_ctx_hash(
        self,
        payload: Dict[str, Any],
        *,
        now_ms: Optional[int] = None,
        price_cache: Optional[Dict[tuple[str, str], Optional[float]]] = None,
        reason_code: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        heat_reason_codes = {
            "risk.planner.heat_exhausted",
            "risk.concurrent.portfolio_heat",
            "risk.concurrent.portfolio_heat_exceeded",
        }
        if reason_code in heat_reason_codes:
            symbol = payload.get("symbol")
            timeframe = payload.get("timeframe") or "5m"

            summary: Any = None
            try:
                risk_mgr = getattr(self, "risk_manager", None)
                if risk_mgr is not None and hasattr(risk_mgr, "get_portfolio_summary"):
                    summary = risk_mgr.get_portfolio_summary(portfolio_manager=getattr(self, "portfolio_manager", None))
                elif hasattr(self.portfolio_manager, "get_portfolio_summary"):
                    summary = self.portfolio_manager.get_portfolio_summary()
            except Exception as exc:
                summary = {"error": str(exc)}

            if not isinstance(summary, dict):
                summary = {"value": str(summary)}

            active_positions_val = summary.get("active_positions")
            total_risk_val = summary.get("total_risk")
            portfolio_heat_val = summary.get("portfolio_heat")

            try:
                active_positions = int(active_positions_val) if active_positions_val is not None else None
            except Exception:
                active_positions = None

            try:
                total_risk = float(total_risk_val) if total_risk_val is not None else None
            except Exception:
                total_risk = None

            try:
                portfolio_heat = float(portfolio_heat_val) if portfolio_heat_val is not None else None
            except Exception:
                portfolio_heat = None

            max_heat = None
            try:
                limits = getattr(self.risk_manager, "concurrent_limits", None)
                max_heat = getattr(limits, "max_total_risk_pct", None) if limits is not None else None
            except Exception:
                max_heat = None

            cache_key = None
            if symbol:
                cache_key = (str(symbol), str(timeframe))
            last_price_val = (
                price_cache.get(cache_key)
                if (price_cache is not None and cache_key is not None)
                else None
            )
            if last_price_val is None:
                last_price = None
                if self.market_data_pipeline and symbol:
                    try:
                        last_price = await self.market_data_pipeline.get_latest_price(str(symbol), timeframe=str(timeframe))
                    except Exception:
                        last_price = None
                if last_price is None:
                    last_price = payload.get("entry") or payload.get("entry_price") or payload.get("price")

                try:
                    last_price_val = float(last_price) if last_price is not None else None
                except Exception:
                    last_price_val = None

                if price_cache is not None and cache_key is not None:
                    price_cache[cache_key] = last_price_val

            last_price_rounded = None
            if last_price_val is not None:
                try:
                    last_price_rounded = float(round(float(last_price_val), 2))
                except Exception:
                    last_price_rounded = None

            heat_tag = "none" if portfolio_heat is None else f"{portfolio_heat:.6f}"
            risk_tag = "none" if total_risk is None else f"{total_risk:.2f}"
            positions_tag = "none" if active_positions is None else str(active_positions)
            price_tag = "none" if last_price_rounded is None else f"{last_price_rounded:.2f}"
            ctx_hash = f"portfolio:{positions_tag}:{heat_tag}:{risk_tag}:price:{price_tag}"
            return ctx_hash, {
                "portfolio": {
                    "active_positions": active_positions,
                    "portfolio_heat": portfolio_heat,
                    "total_risk": total_risk,
                    "max_total_risk_pct": max_heat,
                },
                "price": {
                    "symbol": str(symbol) if symbol else None,
                    "timeframe": str(timeframe) if timeframe else None,
                    "last_price": last_price_val,
                    "last_price_rounded": last_price_rounded,
                },
            }

        symbol = payload.get("symbol")
        timeframe = payload.get("timeframe") or "5m"
        timeframe_ms = self._parse_timeframe_ms(timeframe) or 300_000
        now_ms = self._now_ms() if now_ms is None else int(now_ms)
        candle_open_ms = int(now_ms - (now_ms % int(timeframe_ms)))

        cache_key = None
        if symbol:
            cache_key = (str(symbol), str(timeframe))
        last_price_val = price_cache.get(cache_key) if (price_cache is not None and cache_key is not None) else None
        if last_price_val is None:
            last_price = None
            if self.market_data_pipeline and symbol:
                try:
                    last_price = await self.market_data_pipeline.get_latest_price(str(symbol), timeframe=str(timeframe))
                except Exception:
                    last_price = None
            if last_price is None:
                last_price = payload.get("entry") or payload.get("entry_price") or payload.get("price")

            try:
                last_price_val = float(last_price) if last_price is not None else None
            except Exception:
                last_price_val = None

            if price_cache is not None and cache_key is not None:
                price_cache[cache_key] = last_price_val

        price_tag = "none" if last_price_val is None else f"{last_price_val:.8f}"
        ctx_hash = None if not symbol else f"{symbol}:{timeframe}:{candle_open_ms}:{price_tag}"
        return ctx_hash, {
            "symbol": symbol,
            "timeframe": timeframe,
            "candle_open_ts_ms": candle_open_ms,
            "last_price": last_price_val,
        }

    async def _check_volume_release_condition(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        symbol = payload.get("symbol")
        timeframe = payload.get("timeframe") or "5m"
        if not symbol or not self.volume_analyzer:
            return False, {"error": "missing_symbol_or_volume_analyzer"}
        try:
            ctx = await self.volume_analyzer.compute_context(symbol, timeframe)
        except Exception as exc:
            return False, {"error": str(exc)}
        bucket = getattr(ctx, "bucket", None) if ctx else None
        bucket_label = str(bucket or "").upper()
        try:
            bucket_rank = get_bucket_rank(bucket_label) if bucket_label else None
            normal_rank = get_bucket_rank("NORMAL")
        except Exception:
            bucket_rank = None
            normal_rank = None
        if bucket_rank is None or normal_rank is None:
            return False, {"bucket": bucket_label, "error": "bucket_rank_unavailable"}
        return bucket_rank >= normal_rank, {"bucket": bucket_label, "bucket_rank": bucket_rank, "normal_rank": normal_rank}

    async def _apply_refresh_policy(
        self,
        payload: Dict[str, Any],
        refresh_policy: str,
        *,
        price_override: Optional[float] = None,
        price_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        if refresh_policy != "REPRICE_AND_RESIZE":
            return payload

        symbol = payload.get("symbol")
        timeframe = payload.get("timeframe") or "5m"
        latest_price = price_override
        if latest_price is None and self.market_data_pipeline and symbol:
            try:
                latest_price = await self.market_data_pipeline.get_latest_price(symbol, timeframe=timeframe)
            except Exception:
                latest_price = None

        try:
            old_entry = float(payload.get("entry") or payload.get("entry_price") or 0.0)
        except Exception:
            old_entry = 0.0

        if latest_price is not None:
            try:
                latest_price = float(latest_price)
            except Exception:
                latest_price = None

        if latest_price and latest_price > 0:
            payload["entry"] = float(latest_price)

            stop_pct = payload.get("stop_loss_pct")
            if stop_pct is None:
                try:
                    old_stop = float(payload.get("stop") or payload.get("stop_loss") or 0.0)
                except Exception:
                    old_stop = 0.0
                if old_entry > 0 and old_stop > 0:
                    stop_pct = abs(old_entry - old_stop) / old_entry
                    payload["stop_loss_pct"] = float(stop_pct)

            try:
                stop_pct = float(stop_pct) if stop_pct is not None else None
            except Exception:
                stop_pct = None

            if stop_pct is not None and stop_pct > 0:
                side = str(payload.get("side") or "").lower()
                if side in ("short", "sell"):
                    payload["stop"] = float(latest_price) * (1 + stop_pct)
                else:
                    payload["stop"] = float(latest_price) * (1 - stop_pct)

            target_pct = payload.get("target_pct")
            if target_pct is None:
                try:
                    old_target = float(payload.get("target") or payload.get("take_profit") or 0.0)
                except Exception:
                    old_target = 0.0
                if old_entry > 0 and old_target > 0:
                    target_pct = abs(old_target - old_entry) / old_entry
                    payload["target_pct"] = float(target_pct)

            try:
                target_pct = float(target_pct) if target_pct is not None else None
            except Exception:
                target_pct = None

            if target_pct is not None and target_pct > 0:
                side = str(payload.get("side") or "").lower()
                if side in ("short", "sell"):
                    payload["target"] = float(latest_price) * (1 - target_pct)
                else:
                    payload["target"] = float(latest_price) * (1 + target_pct)

        # Force re-sizing on replay
        for k in (
            "amount",
            "position_size",
            "notional",
            "sizing_meta",
            "planner_active",
            "planner_raw_notional",
            "planner_planned_notional",
            "planner_cap_flags",
        ):
            payload.pop(k, None)

        return payload

    def _initialize_gemma(self):
        """Initialize GEMMA adapter with manifest configuration."""
        try:
            # Import inside function to avoid circular dependency
            from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
            from src.ml.manifest_manager import ManifestManager
            
            gemma_config = self.config['ml']['gemma'].copy()

            # Single operational flag: ml.gemma.enabled
            if isinstance(gemma_config, dict) and gemma_config.get("enabled", True) is False:
                logger.info("🧠 GEMMA disabled via config (ml.gemma.enabled=false); skipping adapter init.")
                self.gemma_adapter = None
                return
            
            # Load manifest for GEMMA configuration
            try:
                manifest_mgr = ManifestManager()
                bundle_path = self.config.get('models', {}).get('active_bundle', 'artifacts/legacy')
                manifest = manifest_mgr.load_manifest(bundle_path)
                
                # Update GEMMA config with manifest
                gemma_config['feature_count'] = manifest['feature_count']
                gemma_config['feature_names'] = manifest['feature_names_ordered']
                
                # Override paths if specified in manifest
                if 'gemma_price_model_path' in manifest:
                    gemma_config['model_path'] = Path(bundle_path) / manifest['gemma_price_model_path']
                if 'gemma_price_scaler_path' in manifest:
                    gemma_config['scaler_path'] = Path(bundle_path) / manifest['gemma_price_scaler_path']
                
                logger.info(f"✅ GEMMA config updated with manifest: {manifest.get('version')}")
                logger.info(f"   Feature count: {manifest['feature_count']}")
                
            except Exception as e:
                logger.warning(f"Failed to load manifest for GEMMA, using defaults: {e}")
            
            self.gemma_adapter = GemmaTorchScriptAdapter(gemma_config)
            logger.info("✅ GEMMA adapter successfully initialized in StrategyCoordinator.")
        except ImportError:
            logger.error("❌ GemmaTorchScriptAdapter could not be imported. Is the file created?")
            self.gemma_adapter = None
        except Exception as e:
            logger.error(f"❌ GEMMA adapter initialization failed: {e}", exc_info=True)
            self.gemma_adapter = None

    def _determine_intent(self, signal: Dict, strategy_name: str) -> str:
        """
        Classify signal intent (entry vs scale-in) based on current open positions.

        Shadow mode when pyramiding is disabled: always returns entry but logs if a scale-in
        candidate is detected. Behavior-changing branches only apply when config enables pyramiding.
        """
        symbol = signal.get("symbol")
        side = str(signal.get("side", "")).lower()
        scale_profile = (signal.get("scale_profile") or signal.get("dca_metadata", {}).get("profile"))
        is_dca_signal = scale_profile == "dca"

        cfg_source = {}
        try:
            if hasattr(self.portfolio_manager, "cfg"):
                cfg_source = self.portfolio_manager.cfg or {}
        except Exception:
            cfg_source = {}
        if not cfg_source and isinstance(self.config, dict):
            cfg_source = self.config

        pyramiding_cfg = cfg_source.get("pyramiding", {}) if isinstance(cfg_source, dict) else {}
        pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False))

        # Gather open positions for this symbol
        open_positions: List[Dict[str, Any]] = []
        pm = getattr(self, "portfolio_manager", None)
        try:
            if pm and hasattr(pm, "get_open_positions_for_symbol") and symbol:
                open_positions = pm.get_open_positions_for_symbol(symbol) or []
        except Exception as exc:
            logger.debug("Intent classification: unable to fetch open positions for %s: %s", symbol, exc)
            open_positions = []

        def _matches(pos: Dict[str, Any]) -> bool:
            pos_side = str(pos.get("side", "")).lower()
            pos_strategy = (pos.get("strategy_name") or pos.get("strategy") or "").lower()
            return (not side or pos_side == side) and pos_strategy == strategy_name.lower()

        candidate_exists = any(_matches(p) for p in open_positions if isinstance(p, dict))

        def _matches_symbol_side(pos: Dict[str, Any]) -> bool:
            pos_side = str(pos.get("side", "")).lower()
            return (not side or pos_side == side)

        candidate_exists_symbol_side = any(
            _matches_symbol_side(p) for p in open_positions if isinstance(p, dict)
        )

        # DCA signals should stay as scale_in regardless of pyramiding toggle (v1).
        if is_dca_signal:
            if not candidate_exists:
                logger.info(
                    "DCA intent requested but no open base position detected | sym=%s | strat=%s",
                    symbol,
                    strategy_name,
                )
            return INTENT_SCALE_IN

        if not pyramiding_enabled:
            if candidate_exists:
                logger.info(
                    "Pyramiding shadow: signal would be classified as scale_in if enabled | sym=%s | strat=%s | side=%s",
                    symbol,
                    strategy_name,
                    side,
                )
            return INTENT_ENTRY

        # When pyramiding is enabled, treat any existing same-side position as a scale-in candidate.
        # Risk is shared at symbol+side level, even if the opening strategy differs.
        if candidate_exists_symbol_side:
            return INTENT_SCALE_IN
        return INTENT_ENTRY
    
    def validate_duplicate(self, signal: Dict, strategy_name: str) -> Tuple[bool, str]:
        """
        Validate signal for duplicates using cooldown and price movement checks.
        
        FIXED: Correct config path detection and threshold units
        - Bug #1: Empty dict check fixed
        - Bug #2: No double division
        - Bug #3: Consistent threshold units
        
        Args:
            signal: Trading signal dictionary
            strategy_name: Name of the strategy generating the signal
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        import time

        intent = signal.get("intent", INTENT_ENTRY)
        scale_profile = signal.get("scale_profile") or (signal.get("dca_metadata") or {}).get("profile")
        if scale_profile == "dca":
            return self._validate_duplicate_dca(signal, strategy_name)

        # Maintenance intents should not be blocked by duplicate logic
        if intent in MAINTENANCE_INTENTS:
            return True, "ok_maintenance_intent"

        # Step 1: Get config
        config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
        pyramiding_cfg = config.get("pyramiding", {}) if isinstance(config, dict) else {}
        pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False))

        # ✅ FIX #1: Proper config path detection
        # Check if signals.duplicate_prevention exists (not just empty dict)
        has_signals_config = (
            'signals' in config and
            'duplicate_prevention' in config.get('signals', {}) and
            config['signals']['duplicate_prevention']  # Not empty
        )

        if has_signals_config:
            # ✅ Use signals config (NEW location, CORRECT values)
            dup_config = config['signals']['duplicate_prevention']
            enabled = dup_config.get('enabled', True)
            base_cooldown = float(dup_config.get('cooldown_seconds', 20))
            base_min_price_change = float(dup_config.get('min_price_change_pct', 0.0005))

            # Dedicated bypass threshold
            price_delta_bypass_threshold = float(dup_config.get('price_delta_bypass_threshold', 0.0015))
            price_delta_bypass_enabled = dup_config.get('price_delta_bypass_enabled', True)

            # Optional scale-in specific settings (defaults mirror base)
            scale_in_min_price_change = float(dup_config.get('scale_in_min_price_change_pct', base_min_price_change))
            scale_in_cooldown = float(dup_config.get('scale_in_cooldown_seconds', base_cooldown))

            logger.debug(f"✓ Using signals.duplicate_prevention config")
        else:
            # ✅ Fallback to monitoring config (OLD location, backward compatibility)
            dup_config = config.get('monitoring', {}).get('duplicate_prevention', {})
            enabled = dup_config.get('enabled', True)
            base_cooldown = float(dup_config.get('same_symbol_cooldown', 60))
            base_min_price_change = float(dup_config.get('min_price_change_pct', 0.0005))
            price_delta_bypass_enabled = dup_config.get('price_delta_bypass_enabled', True)

            # monitoring config uses different unit (0.0015 = 0.15%)
            price_delta_bypass_threshold = float(dup_config.get('price_delta_bypass_threshold', 0.0015))

            # Legacy path has no scale-in overrides; mirror base
            scale_in_min_price_change = base_min_price_change
            scale_in_cooldown = base_cooldown

            logger.debug(f"⚠️ Using monitoring.duplicate_prevention config (legacy)")

        if not enabled:
            return True, "OK"

        # Select effective thresholds by intent (conservative: scale_in mirrors entry)
        if intent in (INTENT_ENTRY, INTENT_REENTRY):
            effective_min_price_change = base_min_price_change
            cooldown = base_cooldown
        elif intent == INTENT_SCALE_IN:
            effective_min_price_change = scale_in_min_price_change
            cooldown = scale_in_cooldown
            if pyramiding_enabled:
                cooldown = min(scale_in_cooldown, base_cooldown)
        else:
            effective_min_price_change = base_min_price_change
            cooldown = base_cooldown

        symbol = signal.get('symbol')
        entry_price = signal.get('entry', 0)
        current_time = time.time()
        
        # Create combined key: "symbol:strategy"
        signal_key = f"{symbol}:{strategy_name}"
        prev_signal_time = self.last_signal_time.get(signal_key)

        # Dynamic cooldown sensitivity based on RSI delta
        effective_cooldown = cooldown
        current_rsi = None
        try:
            rsi_candidate = (signal.get('features') or {}).get('rsi')
            if rsi_candidate is None:
                rsi_candidate = signal.get('rsi')
            if rsi_candidate is not None:
                current_rsi = float(rsi_candidate)
        except (TypeError, ValueError):
            current_rsi = None

        # RSI session-based low-point tracking (symbol-scoped).
        # When RSI is oversold, track the lowest price seen and optionally use it as the
        # reference for price-delta comparisons during cooldown (to avoid expensive lookbacks).
        ref_price_override = None
        try:
            rsi_limit = float(dup_config.get("rsi_session_oversold_threshold", 30.0))
        except (TypeError, ValueError, AttributeError):
            rsi_limit = 30.0

        if symbol and entry_price > 0 and current_rsi is not None:
            signal_side = self._normalize_side(signal.get("side")) or "long"
            if signal_side == "long":
                session = self.rsi_session_state.get(symbol)
                if current_rsi > rsi_limit and session:
                    self.rsi_session_state.pop(symbol, None)
                    session = None

                if current_rsi <= rsi_limit:
                    if not session:
                        session = {"active": True, "anchor_price": float(entry_price), "side": "long"}
                        self.rsi_session_state[symbol] = session
                    else:
                        try:
                            anchor = float(session.get("anchor_price", entry_price))
                        except (TypeError, ValueError):
                            anchor = float(entry_price)
                        if entry_price < anchor:
                            session["anchor_price"] = float(entry_price)
                        session["active"] = True
                        session["side"] = "long"

                    try:
                        ref_price_override = float(self.rsi_session_state[symbol].get("anchor_price"))
                    except (TypeError, ValueError, KeyError):
                        ref_price_override = None

        dynamic_cfg = dup_config.get('dynamic_cooldown', {}) if isinstance(dup_config, dict) else {}
        if current_rsi is not None and dynamic_cfg.get('enabled', True):
            last_rsi = self.last_signal_rsi.get(signal_key, current_rsi)
            rsi_delta = abs(current_rsi - last_rsi)
            high_delta = float(dynamic_cfg.get('high_delta_threshold', 15.0))
            medium_delta = float(dynamic_cfg.get('medium_delta_threshold', 8.0))
            fast_seconds = float(dynamic_cfg.get('fast_cooldown_seconds', 15.0))
            medium_seconds = float(dynamic_cfg.get('medium_cooldown_seconds', 45.0))
            slow_seconds = float(dynamic_cfg.get('slow_cooldown_seconds', cooldown))

            if rsi_delta > high_delta:
                effective_cooldown = fast_seconds
            elif rsi_delta > medium_delta:
                effective_cooldown = medium_seconds
            else:
                effective_cooldown = slow_seconds

            self.last_signal_rsi[signal_key] = current_rsi
            logger.info(
                "⚡ [DUPLICATE] Dynamic cooldown applied | symbol=%s | strategy=%s | rsi=%.2f | Δrsi=%.2f | cooldown=%.1fs",
                symbol,
                strategy_name,
                current_rsi,
                rsi_delta,
                effective_cooldown,
            )

        cooldown = effective_cooldown
        
        # Step 2: Calculate cooldown status
        within_cooldown = False
        remaining = 0
        elapsed_time = None
        
        if signal_key in self.last_signal_time:
            elapsed = current_time - self.last_signal_time[signal_key]
            if elapsed < cooldown:
                within_cooldown = True
                remaining = cooldown - elapsed
                elapsed_time = elapsed
        
        # Step 3: IF within cooldown, check for price delta bypass
        if within_cooldown:
            # Scale-in soft guard when pyramiding enabled: reject only spammy repeats
            if pyramiding_enabled and intent == INTENT_SCALE_IN:
                spam_window = float(pyramiding_cfg.get("spam_window_seconds", 3.0)) if isinstance(pyramiding_cfg, dict) else 3.0
                spam_delta_threshold = min(
                    effective_min_price_change,
                    float(pyramiding_cfg.get("spam_delta_pct", 0.0002) if isinstance(pyramiding_cfg, dict) else 0.0002),
                )

                price_delta = None
                if symbol in self.signal_price_history and entry_price > 0 and self.signal_price_history[symbol]:
                    _, last_price = self.signal_price_history[symbol][-1]
                    price_delta = abs(entry_price - last_price) / last_price

                if (elapsed_time is not None and elapsed_time < spam_window) and (price_delta is None or price_delta < spam_delta_threshold):
                    logger.warning(
                        "❌ [DUPLICATE-REJECT] Scale-in rejected (spam window) | sym=%s | strat=%s | intent=%s | elapsed=%.2fs | delta=%s | threshold=%.5f",
                        symbol,
                        strategy_name,
                        intent,
                        elapsed_time,
                        f"{price_delta:.5f}" if price_delta is not None else "n/a",
                        spam_delta_threshold,
                    )
                    return False, "duplicate_scale_in_spam_window"

                # Not spam: allow; update tracking and continue
                self.last_signal_time[signal_key] = current_time
                if current_rsi is not None:
                    self.last_signal_rsi[signal_key] = current_rsi
                if entry_price > 0:
                    self.signal_price_history[symbol].append((current_time, entry_price))
                logger.info(
                    "✅ [DUPLICATE] Scale-in cooldown skipped under pyramiding | sym=%s | strat=%s | intent=%s | elapsed=%.2fs | cooldown=%.2fs",
                    symbol,
                    strategy_name,
                    intent,
                    elapsed_time or 0.0,
                    cooldown,
                )
                return True, "OK (scale_in_soft_guard)"
            # Step 3a: Get last price from history
            if symbol in self.signal_price_history and entry_price > 0:
                # Find last price for this symbol
                if self.signal_price_history[symbol]:
                    last_timestamp, last_price = self.signal_price_history[symbol][-1]

                    # Use RSI-session anchor as reference during cooldown (when available).
                    ref_price = ref_price_override if ref_price_override is not None else last_price

                    # Directional better-price bypass: accept immediately if price improved for the signal side.
                    # Noise filter: require a meaningful improvement (0.01%) to avoid microscopic tick spam.
                    signal_side = self._normalize_side(signal.get("side")) or "long"
                    IMPROVEMENT_THRESHOLD = 0.0001  # 0.01%
                    is_better_price = False
                    if last_price > 0:
                        if signal_side == "long" and entry_price < (last_price * (1 - IMPROVEMENT_THRESHOLD)):
                            is_better_price = True
                        elif signal_side == "short" and entry_price > (last_price * (1 + IMPROVEMENT_THRESHOLD)):
                            is_better_price = True

                    if is_better_price:
                        logger.info(
                            f"✅ [DUPLICATE-BYPASS] Better price found\n"
                            f"   Symbol: {symbol}\n"
                            f"   Strategy: {strategy_name}\n"
                            f"   Intent: {intent}\n"
                            f"   Side: {signal_side}\n"
                            f"   Last Price: ${last_price:.2f}\n"
                            f"   New Price: ${entry_price:.2f}\n"
                            f"   Better Price Threshold: {IMPROVEMENT_THRESHOLD*100:.2f}%\n"
                            f"   Cooldown Remaining: {remaining:.1f}s\n"
                            f"   ✅ SIGNAL ACCEPTED"
                        )

                        # Update tracking (treat as accepted)
                        self.last_signal_time[signal_key] = current_time
                        self.signal_price_history[symbol].append((current_time, entry_price))
                        if current_rsi is not None:
                            self.last_signal_rsi[signal_key] = current_rsi

                        return (
                            True,
                            f"Better price found (diff > {IMPROVEMENT_THRESHOLD*100:.2f}%) "
                            f"({signal_side}: {last_price:.2f} -> {entry_price:.2f})",
                        )

                    if price_delta_bypass_enabled:
                        # Step 3b: Calculate price_delta (in decimal, e.g., 0.0005 = 0.05%)
                        # TODO (pyramiding/DCA): consider signed delta for certain strategies
                        if ref_price > 0:
                            price_delta = abs(entry_price - ref_price) / ref_price
                        else:
                            price_delta = 0.0

                        # Step 3c: IF price_delta >= threshold, BYPASS
                        if price_delta >= price_delta_bypass_threshold:
                            # Log bypass event with details
                            logger.info(
                                f"✅ [DUPLICATE-BYPASS] Cooldown bypassed\n"
                                f"   Symbol: {symbol}\n"
                                f"   Strategy: {strategy_name}\n"
                                f"   Intent: {intent}\n"
                                f"   Ref Price: ${ref_price:.2f}\n"
                                f"   New Price: ${entry_price:.2f}\n"
                                f"   Delta: {price_delta*100:.2f}% (>= {price_delta_bypass_threshold*100:.2f}%)\n"
                                f"   Cooldown Remaining: {remaining:.1f}s\n"
                                f"   ✅ SIGNAL ACCEPTED"
                            )

                            # Update statistics
                            self.processing_stats['cooldown_bypasses'] += 1
                            self.processing_stats['last_bypass_time'] = current_time

                            # Update running average
                            bypass_count = self.processing_stats['cooldown_bypasses']
                            current_avg = self.processing_stats['avg_bypass_price_delta']
                            new_avg = ((current_avg * (bypass_count - 1)) + (price_delta * 100)) / bypass_count
                            self.processing_stats['avg_bypass_price_delta'] = new_avg

                            # Update bypass success rate
                            total_signals = self.processing_stats['total_signals']
                            if total_signals > 0:
                                self.processing_stats['bypass_success_rate'] = (bypass_count / total_signals) * 100

                            # Update tracking
                            self.last_signal_time[signal_key] = current_time
                            self.signal_price_history[symbol].append((current_time, entry_price))
                            if current_rsi is not None:
                                self.last_signal_rsi[signal_key] = current_rsi

                            return True, f"OK (price delta bypass: {price_delta*100:.2f}%)"

                        # Step 3d: ELSE, reject with price delta info
                        logger.warning(
                            f"❌ [DUPLICATE-REJECT] Signal rejected - insufficient price movement\n"
                            f"   Symbol: {symbol}\n"
                            f"   Strategy: {strategy_name}\n"
                            f"   Intent: {intent}\n"
                            f"   Price Change: {price_delta*100:.2f}% (< {price_delta_bypass_threshold*100:.2f}%)\n"
                            f"   Cooldown Remaining: {remaining:.1f}s\n"
                            f"   ❌ SIGNAL REJECTED"
                        )

                        self.processing_stats['rejected_price_delta'] += 1
                        if current_rsi is not None:
                            self.last_signal_rsi[signal_key] = current_rsi

                        return (
                            False,
                            f"Duplicate prevention: Signal cooldown: {remaining:.0f}s remaining (price change {price_delta*100:.2f}% < threshold)",
                        )
            
            # No price history or bypass disabled
            logger.warning(
                f"❌ [DUPLICATE-REJECT] Signal rejected - cooldown active\n"
                f"   Symbol: {symbol}\n"
                f"   Strategy: {strategy_name}\n"
                f"   Intent: {intent}\n"
                f"   Cooldown Remaining: {remaining:.1f}s\n"
                f"   ❌ SIGNAL REJECTED"
            )
            self.processing_stats['rejected_cooldown'] += 1
            if current_rsi is not None:
                self.last_signal_rsi[signal_key] = current_rsi
            return False, f"Duplicate prevention: Signal cooldown: {remaining:.0f}s remaining (same symbol+strategy)"
        
        # Step 4: IF outside cooldown, accept and update tracking
        self.last_signal_time[signal_key] = current_time
        if current_rsi is not None:
            self.last_signal_rsi[signal_key] = current_rsi
        if entry_price > 0:
            self.signal_price_history[symbol].append((current_time, entry_price))

        elapsed = None
        if prev_signal_time is not None:
            elapsed = current_time - prev_signal_time
        logger.info(
            "Duplicate check accepted outside cooldown | sym=%s | strat=%s | intent=%s | cooldown=%.2fs | min_price_change_pct=%.5f",
            symbol,
            strategy_name,
            intent,
            cooldown,
            effective_min_price_change,
        )
        
        return True, "OK"

    def _validate_duplicate_dca(self, signal: Dict, strategy_name: str) -> Tuple[bool, str]:
        """
        DCA-specific duplicate prevention:
        - Blocks replays of the same layer.
        - Ensures adverse movement vs anchor.
        - Applies DCA cooldown per symbol.
        """
        import time

        symbol = signal.get("symbol")
        dca_meta = signal.get("dca_metadata") or {}
        layer_index_raw = dca_meta.get("layer_index")
        try:
            layer_index = int(layer_index_raw) if layer_index_raw is not None else 0
        except (TypeError, ValueError):
            layer_index = 0

        dca_cfg = (self.config.get("dca") or {}) if isinstance(self.config, dict) else {}
        strategy_cfg = dca_cfg.get("strategy", {}) if isinstance(dca_cfg, dict) else {}
        cooldown_seconds = float(strategy_cfg.get("cooldown_seconds", 0) or 0)

        # 1) Same layer already live?
        try:
            pm_positions = self.portfolio_manager.get_open_positions_for_symbol(symbol) if hasattr(self.portfolio_manager, "get_open_positions_for_symbol") else []
        except Exception:
            pm_positions = []
        for pos in pm_positions or []:
            meta = pos.get("dca_metadata") or {}
            profile = pos.get("scale_profile") or meta.get("profile")
            if profile == "dca":
                try:
                    pos_layer = int(meta.get("layer_index") or 0)
                except (TypeError, ValueError):
                    pos_layer = 0
                if layer_index and pos_layer == layer_index:
                    logger.warning(
                        "[DUPLICATE-DCA] Rejecting duplicate layer | sym=%s | layer=%s",
                        symbol,
                        layer_index,
                    )
                    return False, "dca_layer_duplicate"

        # 2) Rapid repeat guard for same layer
        now = time.time()
        recent_layers = self._dca_recent_layers.get(symbol) or {}
        last_layer_ts = recent_layers.get(layer_index)
        if last_layer_ts and cooldown_seconds > 0 and (now - last_layer_ts) < cooldown_seconds:
            return False, "dca_layer_recent"

        # 3) Adverse movement check (lightweight; RiskManager does full gating)
        anchor_price = dca_meta.get("anchor_price")
        entry_price = signal.get("entry") or signal.get("price") or signal.get("entry_price")
        direction = (signal.get("side") or signal.get("direction") or "").lower()
        price_drop_pct = None
        try:
            price_drop_pct = float(dca_meta.get("price_drop_pct")) if dca_meta.get("price_drop_pct") is not None else None
        except (TypeError, ValueError):
            price_drop_pct = None

        if price_drop_pct is not None and price_drop_pct <= 0:
            return False, "dca_not_adverse_enough"
        if anchor_price and entry_price:
            try:
                anchor_val = float(anchor_price)
                entry_val = float(entry_price)
                if direction in ("long", "buy") and entry_val >= anchor_val:
                    return False, "dca_not_adverse_enough"
                if direction in ("short", "sell") and entry_val <= anchor_val:
                    return False, "dca_not_adverse_enough"
            except (TypeError, ValueError):
                pass

        # 4) Cooldown per symbol
        last_ts = self._dca_last_signal_time.get(symbol, 0.0)
        if cooldown_seconds > 0 and (now - last_ts) < cooldown_seconds:
            return False, "dca_cooldown_not_passed"

        self._dca_last_signal_time[symbol] = now
        if layer_index:
            self._dca_recent_layers[symbol][layer_index] = now

        return True, "OK"
    
    def get_duplicate_prevention_stats(self) -> Dict[str, Any]:
        """
        Get duplicate prevention statistics including bypass metrics.
        Phase 3.4 - Issue #118: Enhanced statistics tracking
        
        Returns:
            Dictionary with comprehensive duplicate prevention metrics:
            - total_signals_processed: Total number of signals processed
            - total_duplicate_rejections: Total signals rejected due to duplication
            - cooldown_bypasses: Number of times cooldown was bypassed
            - bypass_rate: Percentage of bypasses relative to total signals (e.g., 5.5 means 5.5%)
            - avg_bypass_price_delta: Average price change for bypasses as percentage (e.g., 0.35 means 0.35%)
            - rejected_by_cooldown: Signals rejected by cooldown (no price history)
            - rejected_by_price_delta: Signals rejected due to insufficient price movement
            - rejection_breakdown: Detailed breakdown of rejections
            - last_bypass_time: Timestamp of last bypass event
        """
        total_signals = self.processing_stats.get('total_signals', 0)
        cooldown_bypasses = self.processing_stats.get('cooldown_bypasses', 0)
        rejected_cooldown = self.processing_stats.get('rejected_cooldown', 0)
        rejected_price_delta = self.processing_stats.get('rejected_price_delta', 0)
        duplicate_rejections = self.processing_stats.get('duplicate_rejections', 0)
        
        # Calculate rates
        bypass_rate = (cooldown_bypasses / total_signals * 100) if total_signals > 0 else 0.0
        
        return {
            'total_signals_processed': total_signals,
            'total_duplicate_rejections': duplicate_rejections,
            'cooldown_bypasses': cooldown_bypasses,
            'bypass_rate': round(bypass_rate, 2),
            'avg_bypass_price_delta': round(self.processing_stats.get('avg_bypass_price_delta', 0.0), 2),
            'rejected_by_cooldown': rejected_cooldown,
            'rejected_by_price_delta': rejected_price_delta,
            'rejection_breakdown': {
                'cooldown_only': rejected_cooldown,
                'insufficient_price_delta': rejected_price_delta,
                'total': rejected_cooldown + rejected_price_delta
            },
            'last_bypass_time': self.processing_stats.get('last_bypass_time')
        }
    
    def _apply_ai_gate(self, signal: Dict[str, Any]) -> bool:
        """
        Apply AI-Gate filtering with GEMMA if available, otherwise use legacy ML.
        Signal flow: GEMMA → AI-Gate → RL-Veto → Execution
        
        GEMMA operates in two modes:
        - Active Mode: GEMMA predictions override legacy ML
        - Shadow Mode: GEMMA predictions logged but not used (for validation)
        """
        gemma_prediction = None
        gemma_active = False
        
        # 1. GEMMA tahminini al (eğer adaptör aktifse)
        if self.gemma_adapter:
            try:
                features = signal.get('features', {})
                if not features:
                    logger.warning("No features in signal for GEMMA. AI-Gate might be ineffective.")
                else:
                    gemma_prediction = self.gemma_adapter.predict(features)
                    gemma_probabilities = gemma_prediction.get('probabilities', [])
                    signal['gemma_probabilities'] = gemma_probabilities
                    logger.debug(
                        "DEBUG_ML_RL | symbol=%s | gemma_conf=%.3f | gemma_pred=%s | gemma_probs=%s",
                        signal.get('symbol'),
                        gemma_prediction.get('price_confidence', 0.0),
                        gemma_prediction.get('prediction_label', 'neutral'),
                        [round(p, 4) for p in gemma_probabilities[:3]]
                    )
                    
                    # Check if GEMMA is in shadow mode
                    if self.gemma_adapter.shadow_mode:
                        # Shadow mode: Log but don't use
                        logger.info(
                            f"👻 [GEMMA-SHADOW] {signal.get('symbol', 'N/A')} | "
                            f"Prediction: {gemma_prediction.get('prediction_label')} | "
                            f"Confidence: {gemma_prediction.get('price_confidence', 0):.3f} | "
                            f"Legacy: {signal.get('ml_confidence', 0):.3f}"
                        )
                        
                        # Store for analysis but don't use
                        signal['gemma_shadow'] = {
                            'confidence': gemma_prediction.get('price_confidence'),
                            'prediction': gemma_prediction.get('prediction_label')
                        }
                    else:
                        # Active mode: Use GEMMA predictions
                        signal['gemma_confidence'] = gemma_prediction.get('price_confidence')
                        signal['gemma_prediction'] = gemma_prediction.get('prediction_label')
                        gemma_active = True
                        
                        logger.info(
                            f"🧬 [GEMMA-ACTIVE] {signal.get('symbol', 'N/A')} | "
                            f"Prediction: {gemma_prediction.get('prediction_label')} | "
                            f"Confidence: {gemma_prediction.get('price_confidence', 0):.3f}"
                        )
            except Exception as e:
                logger.error(f"GEMMA prediction failed in AI-Gate: {e}", exc_info=True)

        # 2. Güven skorlarını belirle (GEMMA öncelikli sadece active mode'da)
        if gemma_active:
            price_confidence = signal.get('gemma_confidence', 0.5)
            confidence_source = "GEMMA"
        else:
            price_confidence = signal.get('ml_confidence', 0.5)
            confidence_source = "Legacy ML"

        # 3. Eşik değerlerini config'den al
        price_threshold = self.config.get('ml', {}).get('price', {}).get('min_confidence', 0.66)

        # 4. Karar ver
        if price_confidence >= price_threshold:
            logger.info(
                f"✅ [AI-GATE] PASSED | {signal.get('symbol', 'N/A')} | "
                f"Source: {confidence_source} | "
                f"Confidence: {price_confidence:.3f} >= Threshold: {price_threshold:.2f}"
            )
            return True
        else:
            self.processing_stats['ai_gate_rejections'] = self.processing_stats.get('ai_gate_rejections', 0) + 1
            logger.warning(
                f"🛡️ [AI-GATE] REJECTED | {signal.get('symbol', 'N/A')} | "
                f"Source: {confidence_source} | "
                f"Confidence: {price_confidence:.3f} < Threshold: {price_threshold:.2f}"
            )
            return False

    async def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enhanced signal processing with GEMMA integration.
        Signal flow: GEMMA → AI-Gate → RL-Veto → Execution
        
        This is a simplified processing pipeline that can be used when direct signal processing is needed.
        For full strategy signal processing, use process_strategy_signal instead.
        """
        try:
            # Ordering (simplified path):
            # 1) AI-Gate → 2) RL (if present) → 3) Risk → 4) Duplicate → 5) Queue/Execution
            # Ensure intent is set for downstream components
            signal.setdefault("intent", INTENT_ENTRY)
            strategy_name = signal.get('strategy_name') or signal.get('strategy') or 'unknown'
            signal["intent"] = self._determine_intent(signal, strategy_name)

            # ADIM 1: AI-Gate (GEMMA veya eski ML modeli ile filtreleme)
            if not self._apply_ai_gate(signal):
                return None  # Sinyal AI-Gate tarafından reddedildi.
            
            # ADIM 2: RL-Veto (ML enhancement içinde yapılıyor)
            if hasattr(self, 'ml_integration') and self.ml_integration:
                enhanced_signal = await self._enhance_signal_with_ml(signal)
                if enhanced_signal is None:
                    logger.warning(f"Signal for {signal.get('symbol')} rejected by ML/RL enhancement.")
                    return None
                signal = enhanced_signal
            
            # ADIM 2.5: Quality hesapla (RiskManager'in girdi kalitesini iyileştir)
            self._compute_signal_quality(signal)

            # ADIM 3: Risk kontrolleri (mevcut risk_manager üzerinden)
            risk_assessment = await self._assess_signal_risk(signal, strategy_name)
            if not risk_assessment['acceptable']:
                logger.warning(f"Signal for {signal.get('symbol')} rejected by risk assessment: {risk_assessment['reason']}")
                return None
            
            # ADIM 4: Cooldown kontrolleri (validate_duplicate üzerinden)
            strategy_name = signal.get('strategy_name', 'unknown')
            is_valid, reason = self.validate_duplicate(signal, strategy_name)
            if not is_valid:
                logger.warning(f"Signal for {signal.get('symbol')} rejected by duplicate check: {reason}")
                return None
            
            # Tüm kontrollerden geçen sinyal onaylandı
            self.processing_stats['approved_signals'] = self.processing_stats.get('approved_signals', 0) + 1
            
            logger.info(f"Signal for {signal.get('symbol')} approved for execution.")
            return signal
            
        except Exception as e:
            logger.error(f"Critical error in signal processing pipeline: {e}", exc_info=True)
            return None


    def emit_signal_breakdown(self, signal: Dict[str, Any], quality_result: Dict[str, Any]) -> None:
        """
        Log structured JSON breakdown of the signal for observability.
        """
        quality_score = quality_result.get("value", 0.0)
        
        # Alert on zero quality
        if quality_score <= 0.0:
            logger.warning(f"⚠️ [QUALITY-ALERT] Signal quality is 0.0 for {signal.get('symbol')}! Reasons: {quality_result.get('reason')}")

        def _sf(v: Any) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        entry_price = _sf(signal.get("entry_price") or signal.get("entry") or signal.get("price"))
        stop_price = _sf(
            signal.get("stop_price")
            or signal.get("stop")
            or signal.get("stop_loss")
            or signal.get("stop_loss_price")
        )
        target_price = _sf(
            signal.get("target_price")
            or signal.get("target")
            or signal.get("take_profit")
            or signal.get("take_profit_price")
        )

        breakdown = {
            "event": "signal_breakdown",
            "signal_id": signal.get("signal_id"),
            "symbol": signal.get("symbol"),
            "strategy": signal.get("strategy_name"),
            "side": signal.get("side"),
            "quality_score": quality_score,
            "quality_components": quality_result.get("components"),
            "quality_reasons": quality_result.get("reason"),
            "ml_score": signal.get("ml_confidence"),
            "ml_regime": signal.get("predicted_regime"),
            "ml_metadata": signal.get("ml_metadata"),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Advanced volatility telemetry (best-effort): strategies may populate
        # `signal["meta"]["vol_telemetry"]` but logs previously omitted it.
        meta = signal.get("meta") if isinstance(signal, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        vol_tel = meta.get("vol_telemetry")
        if isinstance(vol_tel, dict):
            breakdown["volatility"] = {
                "timeframe": signal.get("timeframe"),
                "window": (self.config.get("indicators", {}).get("advanced_volatility", {}) or {}).get("window")
                if isinstance(getattr(self, "config", None), dict)
                else None,
                "ddof": (self.config.get("indicators", {}).get("advanced_volatility", {}) or {}).get("ddof")
                if isinstance(getattr(self, "config", None), dict)
                else None,
                "selected_estimator": meta.get("vol_selected_estimator") or signal.get("vol_selected_estimator") or "std",
                "vol_rs_bps": _sf(vol_tel.get("rs_bps")),
                "vol_gk_bps": _sf(vol_tel.get("gk_bps")),
                "vol_yz_bps": _sf(vol_tel.get("yz_bps")),
                "vol_atr_bps": _sf(vol_tel.get("atr_bps")),
                "vol_std_bps": _sf(vol_tel.get("std_bps")),
            }
        vsa_shadow = meta.get("vsa_shadow")
        if isinstance(vsa_shadow, dict):
            breakdown["vsa_shadow"] = {
                "selected_class": vsa_shadow.get("selected_class"),
                "probabilities": vsa_shadow.get("probabilities"),
                "scores": vsa_shadow.get("scores"),
                "edge": vsa_shadow.get("edge"),
                "status": vsa_shadow.get("status"),
            }
        logger.info(f"SIGNAL_BREAKDOWN {json.dumps(breakdown)}")

    def _compute_signal_quality(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Compute and attach quality metrics to a signal (single source of truth)."""
        cfg = self.config if isinstance(self.config, dict) else {}
        directional_bias_cfg = (cfg.get("signals") or {}).get("directional_bias") or {}

        def _apply_directional_bias_adjustment(value: float, quality_result: Dict[str, Any]) -> float:
            try:
                base_value = float(value)
            except Exception:
                base_value = 0.0
            base_value = max(0.0, min(1.0, base_value))

            adj = compute_directional_bias_adjustment(signal, directional_bias_cfg if isinstance(directional_bias_cfg, dict) else {})
            try:
                delta = float(adj.get("delta", 0.0) or 0.0)
            except Exception:
                delta = 0.0

            applied = bool(adj.get("applied", False) and abs(delta) > 1e-12)
            adjusted_value = max(0.0, min(1.0, base_value + delta)) if applied else base_value

            quality_result["directional_bias"] = {
                "enabled": bool(adj.get("enabled", False)),
                "applied": applied,
                "delta": round(delta, 4),
                "would_delta": round(float(adj.get("would_delta", 0.0) or 0.0), 4),
                "zone": adj.get("zone"),
                "bias_score": round(float(adj.get("bias_score", 0.0) or 0.0), 4),
                "confidence": round(float(adj.get("confidence", 0.0) or 0.0), 4),
                "reason": adj.get("reason"),
            }
            if applied:
                quality_result.setdefault("components", {})
                quality_result["components"]["directional_bias_component"] = round(
                    float(adj.get("bias_score", 0.0) or 0.0), 4
                )
                if isinstance(quality_result.get("reason"), list):
                    quality_result["reason"].append(str(adj.get("reason") or "directional_bias.adjusted"))
            return adjusted_value

        # Extreme bypass profile (skip ML, rely on non-ML components)
        if signal.get("extreme_bypass"):
            cfg = self.config.get("signals", {}).get("signal_scoring", {}) if isinstance(self.config, dict) else {}
            weights = cfg.get("extreme_weights", {}) if isinstance(cfg, dict) else {}
            w_regime = float(weights.get("regime", 0.4))
            w_vol = float(weights.get("volume", 0.3))
            w_mom = float(weights.get("momentum", 0.3))
            w_rr = float(weights.get("risk_reward", 0.0))
            total_w = max(w_regime + w_vol + w_mom + w_rr, 1e-6)

            def _clamp(v: Any, lo: float = 0.0, hi: float = 1.0) -> float:
                try:
                    return max(lo, min(hi, float(v)))
                except Exception:
                    return lo

            q_regime = _clamp(signal.get("regime_confidence", signal.get("regime_weight", 0.5)), 0.0, 1.0)
            q_vol = _clamp(signal.get("volume_strength", signal.get("volume_score", 0.5)), 0.0, 1.0)
            q_mom = _clamp(signal.get("momentum_strength", 0.5), 0.0, 1.0)
            rr_ratio = signal.get("rr_ratio")
            try:
                q_rr = _clamp((float(rr_ratio) / 3.0) if rr_ratio is not None else 0.5, 0.0, 1.0)
            except Exception:
                q_rr = 0.5

            q_base = (w_regime * q_regime + w_vol * q_vol + w_mom * q_mom + w_rr * q_rr) / total_w
            extreme_min = 0.0
            try:
                extreme_min = float(cfg.get("extreme_min_quality", 0.0) or 0.0)
            except Exception:
                extreme_min = 0.0
            quality_value = max(q_base, extreme_min)
            quality_result = {
                "value": round(quality_value, 4),
                "components": {
                    "regime_component": round(q_regime, 4),
                    "volume_component": round(q_vol, 4),
                    "momentum_component": round(q_mom, 4),
                    "risk_reward_component": round(q_rr, 4),
                },
                "reason": [],
            }
            quality_value = _apply_directional_bias_adjustment(quality_value, quality_result)
            quality_result["value"] = round(quality_value, 4)
            signal["quality_score"] = quality_result["value"]
            signal["quality_breakdown"] = quality_result
            return quality_result

        # Normal profile (config-backed, neutral fallbacks, enriched field mapping)
        cfg = self.config if isinstance(self.config, dict) else {}
        scoring_cfg = (cfg.get("signals") or {}).get("signal_scoring") or {}

        default_normal_weights = {
            "ml": 0.25,
            "volume": 0.20,
            "momentum": 0.20,
            "regime": 0.15,
            "ppo_rl": 0.15,
            "spread": 0.05,
        }
        normal_weights_cfg = scoring_cfg.get("normal_weights") or {}
        weights = {k: float(normal_weights_cfg.get(k, v)) for k, v in default_normal_weights.items()}
        total_w = max(sum(weights.values()), 1e-6)

        def _clamp(v: Any, lo: float = 0.0, hi: float = 1.0) -> float:
            try:
                return max(lo, min(hi, float(v)))
            except Exception:
                return lo

        def _normalize_vol(v: Any) -> float:
            # Volume strength is typically 0..~2; clamp to [0,1] with soft cap
            try:
                val = float(v)
                if val < 0:
                    val = 0.0
                if val > 1.0:
                    return min(val, 2.0) / 2.0
                return val
            except Exception:
                return 0.5

        features = signal.get("features") or {}

        # Components (neutral defaults = 0.5)
        ml_component = _clamp(signal.get("ml_confidence", 0.5), 0.0, 1.0)
        volume_component = _clamp(
            _normalize_vol(signal.get("volume_strength", signal.get("volume_score", 0.5))),
            0.0,
            1.0,
        )
        momentum_component = _clamp(signal.get("momentum_strength", 0.5), 0.0, 1.0)
        regime_component = _clamp(signal.get("regime_confidence", signal.get("regime_weight", 0.5)), 0.0, 1.0)

        # PPO/RL: prefer PPO score; blend with explicit RL agreement if present
        ppo_val = signal.get("ppo_long_score")
        ppo_component = _clamp(ppo_val if ppo_val is not None else 0.5, 0.0, 1.0)
        rl_agree = signal.get("rl_is_agree")
        rl_component = _clamp(1.0 if rl_agree else 0.0, 0.0, 1.0) if rl_agree is not None else None
        if rl_component is None:
            ppo_rl_component = ppo_component
        else:
            ppo_rl_component = _clamp((ppo_component + rl_component) / 2.0, 0.0, 1.0)

        spread_component = _clamp(signal.get("spread_component", features.get("spread", 0.5)), 0.0, 1.0)

        raw_score = (
            weights["ml"] * ml_component
            + weights["volume"] * volume_component
            + weights["momentum"] * momentum_component
            + weights["regime"] * regime_component
            + weights["ppo_rl"] * ppo_rl_component
            + weights["spread"] * spread_component
        )

        base_quality = raw_score / total_w

        # Optional R/R adjustment (small, controlled)
        rr_ratio = signal.get("rr_ratio") or signal.get("risk_reward_ratio")
        rr_adj = 1.0
        try:
            if rr_ratio is not None:
                rr_val = float(rr_ratio)
                if rr_val >= 2.5:
                    rr_adj = min(1.10, 1.05 + (rr_val - 2.5) * 0.02)
                elif rr_val <= 1.2:
                    rr_adj = max(0.90, 0.95 - (1.2 - rr_val) * 0.05)
        except Exception:
            rr_adj = 1.0

        quality_value = _clamp(base_quality * rr_adj, 0.0, 1.0)

        quality_result = {
            "value": round(quality_value, 4),
            "components": {
                "ml_component": round(ml_component, 4),
                "volume_component": round(volume_component, 4),
                "momentum_component": round(momentum_component, 4),
                "regime_component": round(regime_component, 4),
                "ppo_rl_component": round(ppo_rl_component, 4),
                "spread_component": round(spread_component, 4),
            },
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "rr_adjustment": round(rr_adj, 4),
            "reason": [],
        }
        quality_value = _apply_directional_bias_adjustment(quality_value, quality_result)
        quality_result["value"] = round(quality_value, 4)
        signal["quality_score"] = quality_result["value"]
        signal["quality_breakdown"] = quality_result
        return quality_result

    # ===============================================================
    # ====================   DÜZELTİLMİŞ METOT   ====================
    # ===============================================================
    async def process_strategy_signal(self, strategy_name: str, signal: Dict) -> Dict[str, Any]:
        """
        Process incoming signals from registered strategies.
        (GÜNCELLENDİ: 'self.logger' -> 'logger' hatası düzeltildi)
        
        Ordering (full path, default entry/reentry):
          1) Validate format
          2) Enrich signal
          3) Volume gating
          4) Duplicate check
          5) ML enhancement
          6) Conflict resolution
          7) Risk sizing/validation
          8) Quality calc + queue
        
        For scale_in when pyramiding is enabled, duplicate is deferred to a late stage (after risk)
        to let RiskManager's dynamic scaling be the primary gate.
        """
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            self._normalize_signal_side(signal)
            log_prefix = f"[{strategy_name.upper()}/{symbol}]"
            self._ensure_signal_id(strategy_name, signal)
            logger.info(
                "??  %s Signal ingress | side=%s | intent_hint=%s | reason=%s",
                log_prefix,
                signal.get('side', 'N/A'),
                signal.get('intent', INTENT_ENTRY),
                signal.get('reason', 'N/A'),
            )

            cooldown_active, cooldown_until = self._is_strategy_in_cooldown(
                strategy_name, symbol, side=signal.get("side"), return_expiry=True
            )
            if symbol and symbol != 'UNKNOWN' and cooldown_active:
                self.processing_stats['rejected_signals'] += 1
                self.processing_stats['rejected_cooldown'] += 1
                until_str = cooldown_until.isoformat() if isinstance(cooldown_until, datetime) else 'unknown'
                logger.info(
                    "??  %s DROPPED (Cooldown Active) | until=%s",
                    log_prefix,
                    until_str,
                )
                return {
                    'status': 'dropped',
                    'reason': 'stop_loss_cooldown_active',
                    'stage': 'cooldown',
                    'cooldown_until': until_str
                }

            # Default all signals to entry intent unless explicitly provided
            signal.setdefault("intent", INTENT_ENTRY)
            signal["intent"] = self._determine_intent(signal, strategy_name)
            intent = signal.get("intent", INTENT_ENTRY)

            dedupe_key = self._derive_dedupe_key(strategy_name, signal)
            if dedupe_key and not bool(signal.get("incubator_replay")):
                signal["dedupe_key"] = dedupe_key
                active_signal_id = self._active_dedupe_by_key.get(dedupe_key)
                if active_signal_id:
                    drop_item = {
                        "payload": self._json_sanitize(signal),
                        "first_seen_ts_ms": self._now_ms(),
                        "attempts": 0,
                        "reason_code": "incubator.dedupe.active_exists",
                        "dedupe_key": dedupe_key,
                    }
                    self._emit_waiting_room_event(
                        "waiting_room_drop",
                        drop_item,
                        drop_kind="suppression",
                        drop_reason="incubator.dedupe.active_exists",
                        active_signal_id=active_signal_id,
                    )
                    return {
                        "status": "dropped",
                        "reason": "incubator.dedupe.active_exists",
                        "stage": "incubator_dedupe",
                        "dedupe_key": dedupe_key,
                        "active_signal_id": active_signal_id,
                    }

                # If an identical setup is already waiting, refresh its payload and prompt a near-term recheck.
                refreshed = None
                dedupe_drop = None
                dedupe_drop_reason = None
                dedupe_drop_extra: Dict[str, Any] = {}
                dedupe_micro_drop = None
                async with self._incubator_lock:
                    existing = self._incubator_items.get(dedupe_key)
                    if existing:
                        if not existing.get("pending_id"):
                            existing["pending_id"] = uuid.uuid4().hex

                        pending_id_existing = existing.get("pending_id")

                        safe_payload = self._json_sanitize(signal)
                        if isinstance(safe_payload, dict):
                            safe_payload.setdefault("strategy_name", strategy_name)
                            if not safe_payload.get("signal_id"):
                                self._ensure_signal_id(strategy_name, safe_payload)
                            safe_payload["dedupe_key"] = dedupe_key
                            safe_payload["setup_anchor_ts_ms"] = int(self._derive_setup_anchor_ts_ms(safe_payload))

                            incoming_signal_id = safe_payload.get("signal_id")
                            if not incoming_signal_id or (pending_id_existing and str(incoming_signal_id) == str(pending_id_existing)):
                                safe_payload["signal_id"] = uuid.uuid4().hex
                        if str(existing.get("watch_kind") or "").lower() == "micro_gate_watch":
                            dedupe_drop = {
                                "payload": safe_payload,
                                "first_seen_ts_ms": self._now_ms(),
                                "attempts": 0,
                                "reason_code": "micro_watch_active",
                                "dedupe_key": dedupe_key,
                            }
                            dedupe_drop_reason = "micro_watch_active"
                            dedupe_drop_extra = {
                                "existing_pending_id": existing.get("pending_id"),
                                "incoming_signal_id": safe_payload.get("signal_id") if isinstance(safe_payload, dict) else None,
                            }
                            dedupe_micro_drop = {
                                "dedupe_key": dedupe_key,
                                "existing_pending_id": existing.get("pending_id"),
                                "incoming_signal_id": safe_payload.get("signal_id") if isinstance(safe_payload, dict) else None,
                                "reason": "micro_watch_active",
                            }
                        else:
                            existing["payload"] = safe_payload
                            existing["ctx_hash"] = None
                            existing["next_check_at_ms"] = self._now_ms()
                            refreshed = dict(existing)

                if dedupe_drop is not None:
                    if dedupe_micro_drop:
                        self._emit_micro_gate_watch_dedupe_drop_incoming(**dedupe_micro_drop)
                    self._emit_waiting_room_event(
                        "waiting_room_drop",
                        dedupe_drop,
                        drop_reason=dedupe_drop_reason,
                        **dedupe_drop_extra,
                    )
                    return {
                        "status": "dropped",
                        "reason": dedupe_drop_reason,
                        "stage": "incubator_dedupe",
                        "dedupe_key": dedupe_key,
                    }

                if refreshed:
                    return {
                        "status": "incubated",
                        "reason_code": refreshed.get("reason_code"),
                        "stage": "incubator_dedupe",
                        "dedupe_key": dedupe_key,
                    }

            cfg_source = self.config if isinstance(self.config, dict) else {}
            if hasattr(self.portfolio_manager, "cfg") and isinstance(getattr(self.portfolio_manager, "cfg"), dict):
                cfg_source = getattr(self.portfolio_manager, "cfg")
            pyramiding_cfg = cfg_source.get("pyramiding", {}) if isinstance(cfg_source, dict) else {}
            pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False))

            self.processing_stats['total_signals'] += 1
            self.processing_stats['last_signal_time'] = datetime.now(timezone.utc)
            
            # --- TELEMETRİ: Sinyal ilk alındığında logla (DÜZELTİLDİ) ---
            logger.info(f"➡️  {log_prefix} Signal Received. Side: {signal.get('side', 'N/A')}, Reason: '{signal.get('reason', 'N/A')}'")
            
            # Adım 1: Sinyal Formatını Doğrula
            validation_result = self._validate_signal_format(signal)
            if not validation_result['valid']:
                self.processing_stats['rejected_signals'] += 1
                # --- TELEMETRİ: Ret Sebebi (DÜZELTİLDİ) ---
                logger.warning(f"🛡️  {log_prefix} REJECTED (Invalid Format): {validation_result['reason']}")
                return {'status': 'rejected', 'reason': validation_result['reason'], 'stage': 'validation'}
            
            # Adım 2: Sinyali Zenginleştir
            enriched_signal = await self._enrich_signal(strategy_name, signal)

            # --- Integrity Guard (Price deviation + staleness) ---
            current_position = None
            try:
                pm = getattr(self, "portfolio_manager", None)
                if pm is not None and hasattr(pm, "get_open_positions_for_symbol") and symbol:
                    positions = pm.get_open_positions_for_symbol(symbol) or []
                    if isinstance(positions, list):
                        current_position = positions[0] if positions else None
                    elif isinstance(positions, dict):
                        current_position = positions
                elif pm is not None and hasattr(pm, "get_open_positions") and symbol:
                    positions_dict = pm.get_open_positions() or {}
                    if isinstance(positions_dict, dict):
                        for pos in positions_dict.values():
                            if isinstance(pos, dict) and pos.get("symbol") == symbol:
                                current_position = pos
                                break
            except Exception:
                current_position = None

            if self.integrity_guard:
                integrity = await self.integrity_guard.validate(enriched_signal, current_position=current_position)
                if integrity.get("action") == "convert_reverse_to_close":
                    enriched_signal["intent"] = INTENT_CLOSE
                    enriched_signal.setdefault("meta", {}).setdefault("integrity", {})
                    enriched_signal["meta"]["integrity"].update(
                        {
                            "status": "converted",
                            "reason": integrity.get("reason"),
                            "deviation_pct": (integrity.get("metadata") or {}).get("deviation_pct"),
                            "original_intent": "reverse",
                        }
                    )
                    intent = INTENT_CLOSE
                elif not integrity.get("valid", True) or integrity.get("action") == "reject":
                    self.processing_stats['rejected_signals'] += 1
                    logger.warning(
                        "🛡️  [%s/%s] REJECTED (IntegrityGuard): %s",
                        strategy_name.upper(),
                        symbol,
                        integrity.get("reason"),
                    )
                    return {
                        'status': 'rejected',
                        'reason': integrity.get('reason'),
                        'stage': 'integrity_guard'
                    }

            # --- Regime Filter ---
            if self.regime_filter:
                regime_result = await self.regime_filter.validate(enriched_signal)
                if not regime_result.get("valid", True):
                    self.processing_stats['rejected_signals'] += 1
                    logger.warning(
                        "🛡️  [%s/%s] REJECTED (RegimeFilter): %s",
                        strategy_name.upper(),
                        symbol,
                        regime_result.get("reason"),
                    )
                    return {
                        'status': 'rejected',
                        'reason': regime_result.get('reason'),
                        'stage': 'regime_filter'
                    }

            # --- Stop-and-Reverse (Auto Reversal) ---
            # If enabled and we already have an open position for this symbol on the
            # opposite side, tag this signal as INTENT_REVERSE so execution can
            # close then reopen atomically. This also ensures risk gating treats
            # the operation as de-risking (bypasses concurrent limits).
            try:
                signals_cfg = cfg_source.get("signals", {}) if isinstance(cfg_source, dict) else {}
                allow_auto_reversal = bool(
                    signals_cfg.get("allow_auto_reversal", False) if isinstance(signals_cfg, dict) else False
                )
                if allow_auto_reversal and intent == INTENT_ENTRY:
                    incoming_side = str(enriched_signal.get("side", "")).lower()
                    incoming_symbol = enriched_signal.get("symbol") or symbol
                    open_positions_for_symbol: List[Dict[str, Any]] = []

                    pm = getattr(self, "portfolio_manager", None)
                    if pm is not None and hasattr(pm, "get_open_positions_for_symbol") and incoming_symbol:
                        try:
                            open_positions_for_symbol = pm.get_open_positions_for_symbol(incoming_symbol) or []
                        except Exception:
                            open_positions_for_symbol = []
                    elif pm is not None and hasattr(pm, "get_open_positions") and incoming_symbol:
                        try:
                            positions_dict = pm.get_open_positions() or {}
                            open_positions_for_symbol = [
                                dict(pos, position_id=pid)
                                for pid, pos in (positions_dict or {}).items()
                                if isinstance(pos, dict) and pos.get("symbol") == incoming_symbol
                            ]
                        except Exception:
                            open_positions_for_symbol = []

                    reverse_target = None
                    for pos in open_positions_for_symbol:
                        if not isinstance(pos, dict):
                            continue
                        pos_side = str(pos.get("side", "")).lower()
                        if incoming_side and pos_side and self._are_sides_opposite(incoming_side, pos_side):
                            reverse_target = pos
                            break

                    reverse_from_position_id = None
                    if isinstance(reverse_target, dict):
                        reverse_from_position_id = (
                            reverse_target.get("position_id")
                            or reverse_target.get("id")
                            or reverse_target.get("positionId")
                        )

                    if reverse_from_position_id:
                        policy_result = None
                        if self.transition_policy:
                            policy_result = self.transition_policy.evaluate(
                                reverse_target,
                                enriched_signal,
                                inferred_intent="reverse",
                            )

                        if policy_result and not policy_result.get("allowed", True):
                            if policy_result.get("action") == "convert_to_close":
                                enriched_signal["intent"] = INTENT_CLOSE
                                enriched_signal.setdefault("meta", {}).setdefault("transition_policy", {})
                                enriched_signal["meta"]["transition_policy"] = {
                                    "original_intent": "reverse",
                                    "action": "converted",
                                    "reason": policy_result.get("reason"),
                                }
                                intent = INTENT_CLOSE
                                logger.info(
                                    "[AUTO-REVERSE] Blocked reverse, converted to close | sym=%s | reason=%s",
                                    incoming_symbol,
                                    policy_result.get("reason"),
                                )
                            elif policy_result.get("action") == "reject":
                                self.processing_stats['rejected_signals'] += 1
                                logger.info(
                                    "[AUTO-REVERSE] Rejected by transition policy | sym=%s | reason=%s",
                                    incoming_symbol,
                                    policy_result.get("reason"),
                                )
                                return {
                                    'status': 'rejected',
                                    'reason': policy_result.get('reason'),
                                    'stage': 'transition_policy'
                                }
                        else:
                            enriched_signal["intent"] = INTENT_REVERSE
                            enriched_signal["reverse_from_position_id"] = reverse_from_position_id
                            enriched_signal.setdefault("meta", {})["auto_reversal"] = True
                            logger.info(
                                "[AUTO-REVERSE] Tagged reverse intent | sym=%s | %s -> %s | close_position_id=%s",
                                incoming_symbol,
                                str(reverse_target.get("side", "")).lower() if isinstance(reverse_target, dict) else "n/a",
                                incoming_side,
                                reverse_from_position_id,
                            )
                            intent = INTENT_REVERSE
            except Exception as exc:
                logger.warning("[AUTO-REVERSE] Failed to evaluate auto reversal: %s", exc)

            # --- TrendGuard Veto (Fast Trend Validation) ---
            if self.trend_guard and intent in (INTENT_ENTRY, INTENT_REENTRY, INTENT_REVERSE):
                try:
                    if self.trend_guard.should_check(strategy_name, enriched_signal):
                        guard_tf = self.trend_guard.resolve_timeframe(enriched_signal)
                        guard_df = None
                        if self.market_data_pipeline:
                            guard_df = await self.market_data_pipeline.get_latest_ohlcv(symbol, guard_tf)
                        if guard_df is not None and not guard_df.empty:
                            guard_result = self.trend_guard.check_veto(
                                symbol=symbol,
                                side=enriched_signal.get("side"),
                                current_candle=None,
                                dataframe=guard_df,
                                timeframe=guard_tf,
                            )
                            enriched_signal.setdefault("meta", {})["trend_guard"] = guard_result.meta_data
                            if guard_result.is_vetoed:
                                self.processing_stats['rejected_signals'] += 1
                                logger.warning(
                                    "??  [%s/%s] REJECTED (TrendGuard): %s",
                                    strategy_name.upper(),
                                    symbol,
                                    guard_result.reason,
                                )
                                if self._trend_guard_veto_diag_enabled:
                                    try:
                                        now_mono = time.monotonic()
                                        meta = guard_result.meta_data or {}
                                        attrs = guard_df.attrs if hasattr(guard_df, "attrs") else {}

                                        key_map = {
                                            "symbol": symbol,
                                            "timeframe": guard_tf,
                                            "side": meta.get("side") or enriched_signal.get("side"),
                                            "reason": guard_result.reason,
                                            "strategy": strategy_name,
                                        }
                                        key = tuple(key_map.get(field) for field in self._trend_guard_veto_diag_key_fields)
                                        last_ts = self._trend_guard_veto_diag_last_log.get(key, 0.0)
                                        throttle_s = max(float(self._trend_guard_veto_diag_throttle_s), 0.0)
                                        if throttle_s <= 0 or (now_mono - float(last_ts or 0.0)) >= throttle_s:
                                            self._trend_guard_veto_diag_last_log[key] = now_mono
                                            level = getattr(logging, self._trend_guard_veto_diag_log_level, logging.WARNING)

                                            def _fmt(val: Any, precision: int = 6) -> str:
                                                if val is None:
                                                    return "n/a"
                                                if isinstance(val, bool):
                                                    return "true" if val else "false"
                                                if isinstance(val, (int, np.integer)):
                                                    return str(val)
                                                if isinstance(val, str):
                                                    return val
                                                try:
                                                    fval = float(val)
                                                    if math.isfinite(fval):
                                                        return f"{fval:.{precision}f}"
                                                except Exception:
                                                    return str(val)
                                                return str(val)

                                            logger.log(
                                                level,
                                                "[TREND-GUARD][VETO] sym=%s strat=%s side=%s tf=%s reason=%s "
                                                "squeeze_recent=%s breakout_dir=%s close=%s upper=%s lower=%s "
                                                "bbw=%s bbw_ratio=%s bbw_squeeze_thr=%s bbw_expand_thr=%s "
                                                "slope=%s slope_up_thr=%s slope_dn_thr=%s body_ratio=%s "
                                                "src=%s last_closed_ts=%s forming_ts=%s gap_count=%s retrieved_at=%s",
                                                symbol,
                                                strategy_name,
                                                _fmt(meta.get("side") or enriched_signal.get("side")),
                                                guard_tf,
                                                guard_result.reason,
                                                _fmt(meta.get("squeeze_recent")),
                                                _fmt(meta.get("breakout_dir"), precision=0),
                                                _fmt(meta.get("close")),
                                                _fmt(meta.get("upper")),
                                                _fmt(meta.get("lower")),
                                                _fmt(meta.get("bbw")),
                                                _fmt(meta.get("bbw_ratio")),
                                                _fmt(meta.get("bbw_squeeze_thr")),
                                                _fmt(meta.get("bbw_expand_thr")),
                                                _fmt(meta.get("slope"), precision=8),
                                                _fmt(meta.get("slope_up_thr"), precision=8),
                                                _fmt(meta.get("slope_dn_thr"), precision=8),
                                                _fmt(meta.get("body_ratio")),
                                                _fmt(attrs.get("ohlcv_source"), precision=0),
                                                _fmt(attrs.get("last_closed_ts"), precision=0),
                                                _fmt(attrs.get("forming_ts"), precision=0),
                                                _fmt(attrs.get("gap_count"), precision=0),
                                                _fmt(attrs.get("retrieved_at"), precision=0),
                                            )
                                    except Exception as exc:
                                        logger.debug("[TREND-GUARD] Veto diagnostics skipped: %s", exc)
                                return {
                                    'status': 'rejected',
                                    'reason': guard_result.reason,
                                    'stage': 'trend_guard',
                                }
                        else:
                            logger.debug(
                                "[TREND-GUARD] No data for %s %s; skipping guard",
                                symbol,
                                guard_tf,
                            )
                except Exception as exc:
                    logger.warning("[TREND-GUARD] Evaluation failed: %s", exc)

            # --- Safety Override (Adaptive/Aggressive Mode) ---
            if self.safety_override and intent in (INTENT_ENTRY, INTENT_REENTRY, INTENT_REVERSE):
                try:
                    if self.safety_override.should_check(strategy_name, enriched_signal):
                        guard_result = self.safety_override.check_veto(strategy_name, enriched_signal)
                        enriched_signal.setdefault("meta", {})["safety_override"] = guard_result.meta_data
                        if guard_result.is_vetoed:
                            self.processing_stats['rejected_signals'] += 1
                            self.processing_stats.setdefault('safety_override_blocks', 0)
                            self.processing_stats['safety_override_blocks'] += 1
                            # INFO-level so operators can see blocks in normal live logs.
                            logger.info(
                                "??  [%s/%s] REJECTED (SafetyOverride): %s | reason=%s score=%s passes=%s fails=%s",
                                strategy_name.upper(),
                                symbol,
                                guard_result.reason,
                                (guard_result.meta_data or {}).get("reason"),
                                (guard_result.meta_data or {}).get("score"),
                                (guard_result.meta_data or {}).get("passes"),
                                (guard_result.meta_data or {}).get("fails"),
                            )
                            return {
                                'status': 'rejected',
                                'reason': guard_result.reason,
                                'stage': 'safety_override',
                            }
                        else:
                            # Emit a trace for PASS (only when aggressive mode is active) so live ops can confirm
                            # the guard is running even when it doesn't block.
                            try:
                                meta = guard_result.meta_data or {}
                                if bool(meta.get("aggressive", False)):
                                    logger.info(
                                        "✓  [%s/%s] SafetyOverride PASSED | reason=%s score=%s passes=%s fails=%s na=%s",
                                        strategy_name.upper(),
                                        symbol,
                                        meta.get("reason", guard_result.reason),
                                        meta.get("score"),
                                        meta.get("passes"),
                                        meta.get("fails"),
                                        meta.get("na"),
                                    )
                            except Exception:
                                pass
                except Exception as exc:
                    logger.warning("[SAFETY-OVERRIDE] Evaluation failed: %s", exc)
             
            # --- Volume Gating (Issue #450) ---
            strat_cfg = self.config.get('strategies', {}).get(strategy_name, {})
            vol_filters = strat_cfg.get('volume_filters', {})
            volume_bucket = enriched_signal.get('volume_bucket')
            
            if volume_bucket:
                decision = "accepted"
                rejection_reason = None
                
                # 1. Check min_bucket if filters enabled
                if vol_filters.get('enabled', False):
                    min_bucket = vol_filters.get('min_bucket', 'NORMAL')
                    current_rank = get_bucket_rank(volume_bucket)
                    min_rank = get_bucket_rank(min_bucket)
                    
                    if current_rank < min_rank:
                        # Check for override
                        allow_low = strat_cfg.get('allow_low_volume', False)
                        if allow_low:
                            require_confirmation = bool(
                                strat_cfg.get('low_volume_requires_reversal_confirmation', False)
                                or strat_cfg.get('low_volume_requires_confirmation', False)
                            )
                            confirmation_ok = True
                            if require_confirmation and volume_bucket.lower() in {'low', 'very_low'}:
                                meta = enriched_signal.get('meta') or {}
                                confirmation_ok = bool(
                                    meta.get('low_volume_reversal_confirmed')
                                    or meta.get('reversal_confirmed')
                                    or meta.get('reversal_confirmation', {}).get('confirmed')
                                )

                            # Optional: Downtrend guard only applies on LOW/very_low volume.
                            # This is deliberately downstream so we can use `volume_bucket`.
                            if confirmation_ok and volume_bucket.lower() in {'low', 'very_low'}:
                                extreme_bypass = bool(
                                    enriched_signal.get('extreme_bypass')
                                    or (enriched_signal.get('features') or {}).get('extreme_bypass')
                                    or (enriched_signal.get('meta') or {}).get('extreme_bypass')
                                )

                                require_strong_in_downtrend = bool(
                                    strat_cfg.get('adaptive_ob_require_reversal_confirmation_in_downtrend', False)
                                    or strat_cfg.get('downtrend_low_volume_requires_strong_reversal', False)
                                )
                                adx_threshold = float(strat_cfg.get('downtrend_guard_adx_threshold', 30.0))

                                if require_strong_in_downtrend and not extreme_bypass:
                                    downtrend_ctx = (meta.get('downtrend_context') or {})
                                    downtrend_active = bool(downtrend_ctx.get('active'))
                                    adx_val = downtrend_ctx.get('adx')
                                    try:
                                        adx_ok = (adx_val is not None) and (float(adx_val) >= adx_threshold)
                                    except Exception:
                                        adx_ok = False

                                    if downtrend_active and adx_ok:
                                        rev = (meta.get('reversal_confirmation') or {})
                                        strong_ok = bool(rev.get('strong_confirmed'))
                                        if not strong_ok:
                                            decision = "rejected_low_bucket"
                                            rejection_reason = (
                                                f"LOW volume downtrend guard: strong reversal confirmation missing "
                                                f"(ADX>={adx_threshold})"
                                            )

                            if confirmation_ok:
                                logger.info(
                                    f"⚠️ [VOLUME-OVERRIDE] {log_prefix} | Bucket '{volume_bucket}' < min '{min_bucket}' "
                                    f"but allow_low_volume=True. Accepting."
                                )
                            else:
                                decision = "rejected_low_bucket"
                                rejection_reason = (
                                    f"Volume bucket '{volume_bucket}' < min '{min_bucket}' and "
                                    "low-volume reversal confirmation missing"
                                )
                        else:
                            decision = "rejected_low_bucket"
                            rejection_reason = f"Volume bucket '{volume_bucket}' < min '{min_bucket}' and allow_low_volume=False"

                # 2. Check very_low/LOW hard floor if not already rejected
                # If bucket is LOW (or very_low) and allow_low_volume is False, we should reject 
                # even if min_bucket is set to LOW? 
                # That seems to be what the user implied with "If volume_bucket is very_low and allow_low_volume is false...".
                # But if min_bucket is LOW, then we explicitly allowed LOW.
                # Let's assume 'very_low' is a special case or alias for LOW in the user's mind.
                # If we stick to the override logic above, it handles the most common case (min=NORMAL, bucket=LOW).
                # If min=LOW, then bucket=LOW passes. 
                # If the user wants to block LOW, they should set min=NORMAL.
                # So I will stick to the override logic above as the primary mechanism.
                
                # However, to be safe and strictly follow "If volume_bucket is very_low ... rejection",
                # I will add a check for 'very_low' specifically, in case it appears (e.g. from a custom analyzer).
                if decision == "accepted" and volume_bucket.lower() == 'very_low':
                     allow_low = strat_cfg.get('allow_low_volume', False)
                     if not allow_low:
                         decision = "rejected_very_low"
                         rejection_reason = "Volume bucket is 'very_low' and allow_low_volume=False"

                # Log the decision check (Structured for analysis)
                audit_payload = {
                    'event': 'volume_decision_check',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'run_id': get_current_run_id(),
                    'strategy_name': strategy_name,
                    'symbol': symbol,
                    'timeframe': enriched_signal.get('timeframe', '5m'),
                    'volume_bucket': volume_bucket,
                    'volume_strength': enriched_signal.get('volume_strength', 0.0),
                    'volume_ctx_source': 'analyzer',
                    'volume_ratio_short': enriched_signal.get('volume_ratio_short'),
                    'volume_ratio_medium': enriched_signal.get('volume_ratio_medium'),
                    'volume_ratio_combined': enriched_signal.get('volume_ratio_combined'),
                    'current_window_volume': enriched_signal.get('volume_current_window'),
                    'short_baseline_volume': enriched_signal.get('volume_short_baseline'),
                    'medium_baseline_volume': enriched_signal.get('volume_medium_baseline'),
                    'central_bucket_decision': decision
                }
                logger.info(f"volume_decision_check {json.dumps(audit_payload)}")

                if decision != "accepted":
                    # Optional: during fast shock states, avoid hard rejects on LOW buckets.
                    # Instead, short-incubate so the signal can be repriced/resized as volume confirms.
                    try:
                        vol_policy_cfg = (
                            (self.config.get("signals", {}) or {}).get("volume_policy", {})
                            if isinstance(self.config, dict)
                            else {}
                        )
                    except Exception:
                        vol_policy_cfg = {}
                    defer_on_shock_low = bool(
                        (vol_policy_cfg.get("defer_on_shock_low_bucket", False) if isinstance(vol_policy_cfg, dict) else False)
                    )
                    shock_state = None
                    try:
                        meta = enriched_signal.get("meta") if isinstance(enriched_signal, dict) else None
                        if isinstance(meta, dict):
                            shock_state = meta.get("shock_state") or meta.get("shock_status")
                    except Exception:
                        shock_state = None

                    if defer_on_shock_low and str(shock_state or "").upper() == "ARMED":
                        logger.warning(
                            "⚠️  %s DEFERRED (Shock Low Bucket) | decision=%s reason=%s",
                            log_prefix,
                            decision,
                            rejection_reason,
                        )
                        incubated = await self.incubate_signal(
                            strategy_name=strategy_name,
                            signal=enriched_signal,
                            reason_code="volume.shock_low_bucket",
                            refresh_policy="REPRICE_AND_RESIZE",
                            stage="volume_gating",
                        )
                        if incubated.get("status") == "incubated":
                            return {
                                "status": "incubated",
                                "reason_code": "volume.shock_low_bucket",
                                "stage": "volume_gating",
                                "dedupe_key": incubated.get("dedupe_key"),
                            }
                        return incubated

                    self.processing_stats['rejected_signals'] += 1
                    logger.warning(f"🛡️  {log_prefix} REJECTED (Volume Gating): {rejection_reason}")
                    return {'status': 'rejected', 'reason': rejection_reason, 'stage': 'volume_gating'}

            # --- Crash Guard (Panic / Falling-Knife Protection) ---
            crash_cfg = self._get_crash_guard_cfg(strategy_name)
            crash_enabled = bool(crash_cfg.get("enabled", False))
            if (
                crash_enabled
                and strategy_name == "adaptive_ob"
                and intent in (INTENT_ENTRY, INTENT_REENTRY, INTENT_SCALE_IN)
            ):
                side_norm = str(enriched_signal.get("side") or "").lower().strip()
                if side_norm in {"long", "buy"}:
                    cooldown_mode = str(crash_cfg.get("cooldown_mode", "off") or "off").strip().lower()
                    if cooldown_mode == "reversal_only":
                        sl_key = f"{strategy_name}:{symbol}:{side_norm}"
                        with self._strategy_cooldowns_lock:
                            armed_since = self._stop_loss_reversal_required.get(sl_key)
                        if armed_since:
                            meta = enriched_signal.get("meta") or {}
                            if not isinstance(meta, dict):
                                meta = {}
                            rsi_hook = bool(meta.get("rsi_hook"))
                            bull_candle = bool(meta.get("bull_candle"))
                            reclaim = bool(meta.get("reclaim"))
                            reversal_ok = bool(rsi_hook and (bull_candle or reclaim))
                            if not reversal_ok:
                                self.processing_stats['rejected_signals'] += 1
                                logger.warning(
                                    "🛡️  %s REJECTED (StopLoss Guard): stop_loss_reversal_required | missing_reversal=%s",
                                    log_prefix,
                                    "rsi_hook" if not rsi_hook else "bull_candle_or_reclaim",
                                )
                                return {
                                    "status": "rejected",
                                    "reason": "stop_loss_reversal_required",
                                    "reason_code": "stop_loss_reversal_required",
                                    "stage": "cooldown",
                                    "armed_since_ts": armed_since,
                                }

                    is_panic_state, panic_meta = await self._compute_panic_state(
                        symbol=symbol,
                        volume_bucket=volume_bucket,
                        crash_cfg=crash_cfg,
                    )
                    enriched_signal.setdefault("meta", {})["panic_guard"] = panic_meta

                    if is_panic_state:
                        meta = enriched_signal.get("meta") or {}
                        if not isinstance(meta, dict):
                            meta = {}

                        # --- TP/RR Phantom Fix (gated: only during panic states) ---
                        rr_fix_mode = str(crash_cfg.get("tp_rr_fix_mode", "off") or "off").strip().lower()
                        if rr_fix_mode != "off":
                            try:
                                gap_th = float(crash_cfg.get("tp_rr_fix_ema_gap_atr_threshold", 0.0) or 0.0)
                            except Exception:
                                gap_th = 0.0

                            gap_atr = panic_meta.get("ema_fast_gap_atr")
                            gap_atr_f = None
                            try:
                                if gap_atr is not None:
                                    gap_atr_f = float(gap_atr)
                            except Exception:
                                gap_atr_f = None

                            if gap_th > 0 and gap_atr_f is not None and gap_atr_f >= gap_th:
                                try:
                                    entry_val = float(enriched_signal.get("entry") or 0.0)
                                except Exception:
                                    entry_val = 0.0
                                try:
                                    stop_val = float(enriched_signal.get("stop") or 0.0)
                                except Exception:
                                    stop_val = 0.0
                                try:
                                    target_val = float(
                                        enriched_signal.get("target")
                                        or enriched_signal.get("take_profit")
                                        or 0.0
                                    )
                                except Exception:
                                    target_val = 0.0

                                risk_val = abs(entry_val - stop_val) if entry_val > 0 and stop_val > 0 else 0.0

                                rr_raw = None
                                try:
                                    rr_raw = float(enriched_signal.get("rr_ratio") or 0.0)
                                except Exception:
                                    rr_raw = None
                                if rr_raw is None or rr_raw <= 0:
                                    rr_raw = (
                                        (abs(target_val - entry_val) / risk_val)
                                        if (risk_val > 0 and target_val > 0)
                                        else None
                                    )

                                fix_meta = {
                                    "mode": rr_fix_mode,
                                    "ema_fast_gap_atr": gap_atr_f,
                                    "ema_fast_gap_atr_threshold": gap_th,
                                    "rr_raw": rr_raw,
                                    "applied": False,
                                }

                                if rr_fix_mode == "clamp_tp":
                                    try:
                                        max_tp_atr = float(crash_cfg.get("tp_rr_fix_max_tp_atr_mult", 0.0) or 0.0)
                                    except Exception:
                                        max_tp_atr = 0.0

                                    atr_val = panic_meta.get("atr")
                                    atr_f = None
                                    try:
                                        if atr_val is not None:
                                            atr_f = float(atr_val)
                                    except Exception:
                                        atr_f = None

                                    if (
                                        max_tp_atr > 0
                                        and atr_f is not None
                                        and atr_f > 0
                                        and entry_val > 0
                                        and target_val > 0
                                        and risk_val > 0
                                    ):
                                        max_target = entry_val + (max_tp_atr * atr_f)
                                        if target_val > max_target:
                                            enriched_signal["target"] = max_target
                                            enriched_signal["rr_ratio"] = (max_target - entry_val) / risk_val
                                            fix_meta.update(
                                                {
                                                    "applied": True,
                                                    "max_tp_atr_mult": max_tp_atr,
                                                    "atr": atr_f,
                                                    "target_raw": target_val,
                                                    "target_clamped": max_target,
                                                    "rr_new": enriched_signal.get("rr_ratio"),
                                                }
                                            )

                                elif rr_fix_mode == "penalize_rr":
                                    try:
                                        penalty = float(crash_cfg.get("tp_rr_fix_rr_penalty", 1.0) or 1.0)
                                    except Exception:
                                        penalty = 1.0
                                    if rr_raw is not None and rr_raw > 0 and penalty > 0 and penalty != 1.0:
                                        enriched_signal["rr_ratio"] = rr_raw * penalty
                                        fix_meta.update(
                                            {
                                                "applied": True,
                                                "rr_penalty": penalty,
                                                "rr_new": enriched_signal.get("rr_ratio"),
                                            }
                                        )

                                if fix_meta.get("applied"):
                                    meta.setdefault("tp_rr_fix", fix_meta)

                        rsi_hook = bool(meta.get("rsi_hook"))
                        bull_candle = bool(meta.get("bull_candle"))
                        reclaim = bool(meta.get("reclaim"))

                        # Tiered acceptance:
                        # - HIGH: rsi_hook AND (bull_candle OR reclaim)
                        # - EXTREME (or very large EMA gap): rsi_hook AND reclaim
                        volume_bucket_label = str(volume_bucket or "").upper().strip()
                        ema_gap_atr = None
                        try:
                            ema_gap_atr = float(panic_meta.get("ema_fast_gap_atr")) if panic_meta.get("ema_fast_gap_atr") is not None else None
                        except Exception:
                            ema_gap_atr = None
                        try:
                            extreme_gap_th = float(crash_cfg.get("extreme_gap_atr_threshold", 0.0) or 0.0)
                        except Exception:
                            extreme_gap_th = 0.0
                        strict_extreme = bool(volume_bucket_label == "EXTREME")
                        if not strict_extreme and extreme_gap_th > 0 and ema_gap_atr is not None and ema_gap_atr >= extreme_gap_th:
                            strict_extreme = True

                        if strict_extreme:
                            reversal_ok = bool(rsi_hook and reclaim)
                        else:
                            reversal_ok = bool(rsi_hook and (bull_candle or reclaim))

                        if not reversal_ok:
                            self.processing_stats['rejected_signals'] += 1
                            missing = []
                            if not rsi_hook:
                                missing.append("rsi_hook")
                            if strict_extreme:
                                if not reclaim:
                                    missing.append("reclaim")
                            else:
                                if not (bull_candle or reclaim):
                                    missing.append("bull_candle_or_reclaim")

                            logger.warning(
                                "🛡️  %s REJECTED (PanicGuard): panic_veto_no_reversal | bucket=%s tf=%s drop=%s atr=%s bear_body=%s ema_gap_atr=%s strict_extreme=%s missing=%s",
                                log_prefix,
                                (volume_bucket or "n/a"),
                                panic_meta.get("tf"),
                                panic_meta.get("fast_drop_pct"),
                                panic_meta.get("atr_pct"),
                                panic_meta.get("bearish_body_ratio"),
                                panic_meta.get("ema_fast_gap_atr"),
                                bool(strict_extreme),
                                ",".join(missing) if missing else "n/a",
                            )
                            return {
                                "status": "rejected",
                                "reason": "panic_veto_no_reversal",
                                "reason_code": "panic_veto_no_reversal",
                                "stage": "panic_guard",
                                "panic_guard": panic_meta,
                                "missing_reversal": missing,
                            }

            # --- Volume Policy Matrix (Phase 3) ---
            def _calc_stop_pct(sig: Dict[str, Any]) -> Optional[float]:
                try:
                    entry_val = float(sig.get('entry') or 0.0)
                    stop_val = float(sig.get('stop') or 0.0)
                    if entry_val > 0 and stop_val > 0:
                        return abs(entry_val - stop_val) / entry_val
                except Exception:
                    return None
                return None

            def _calc_rr(sig: Dict[str, Any]) -> Optional[float]:
                try:
                    entry_val = float(sig.get('entry') or 0.0)
                    stop_val = float(sig.get('stop') or 0.0)
                    target_val = float(sig.get('target') or sig.get('take_profit') or 0.0)
                    if entry_val > 0 and stop_val > 0 and target_val > 0:
                        risk_val = abs(entry_val - stop_val)
                        if risk_val <= 0:
                            return None
                        side_val = (sig.get('side') or '').lower()
                        if side_val in ['short', 'sell']:
                            reward_val = entry_val - target_val
                        else:
                            reward_val = target_val - entry_val
                        return abs(reward_val) / risk_val if reward_val is not None else None
                except Exception:
                    pass

                try:
                    rr_existing = float(sig.get('rr_ratio'))
                    if rr_existing > 0:
                        return rr_existing
                except Exception:
                    return None
                return None

            volume_bucket_label = (volume_bucket or "").upper()
            volume_rank = get_bucket_rank(volume_bucket_label) if volume_bucket else None
            normal_rank = get_bucket_rank('NORMAL')
            is_low_bucket = volume_bucket_label == 'LOW'
            is_deferred = bool((enriched_signal.get('queue_meta') or {}).get('is_deferred'))

            tight_stop_threshold = LOW_VOL_TIGHT_STOP_THRESHOLD
            wide_stop_threshold = LOW_VOL_WIDE_STOP_THRESHOLD
            micro_gate_margin_bps = LOW_VOL_MICRO_GATE_MARGIN_BPS
            RESCUE_MULTIPLIER = 0.25
            LOW_LIMIT_MULTIPLIER = 0.35
            COOLDOWN_SECONDS = 300  # 1 bar

            if volume_bucket:
                # Check 1: Deferred signals returning from waiting room
                if is_deferred:
                    if volume_rank is not None and volume_rank >= normal_rank:
                        logger.info(f"?? {log_prefix} Deferred signal resumed at {volume_bucket_label} volume (standard profile).")
                    elif is_low_bucket:
                        stop_pct, _, _ = self._calc_stop_metrics(enriched_signal)
                        stop_pct = stop_pct or 0.0
                        adjusted_pct = max(stop_pct, tight_stop_threshold)

                        entry_val = float(enriched_signal.get('entry') or 0.0)
                        if entry_val > 0 and stop_pct < tight_stop_threshold:
                            side_val = (enriched_signal.get('side') or '').lower()
                            if side_val in ['short', 'sell']:
                                enriched_signal['stop'] = entry_val * (1 + adjusted_pct)
                            else:
                                enriched_signal['stop'] = entry_val * (1 - adjusted_pct)

                        enriched_signal['stop_loss_pct'] = adjusted_pct
                        enriched_signal['position_size_multiplier'] = RESCUE_MULTIPLIER
                        enriched_signal['execution_params'] = {'type': 'LIMIT', 'post_only': True}

                        new_rr = _calc_rr(enriched_signal)
                        if new_rr is not None:
                            enriched_signal['rr_ratio'] = new_rr
                        if new_rr is None or new_rr < 3.0:
                            self.processing_stats['rejected_signals'] += 1
                            reason = f"Deferred low-volume check failed RR ({new_rr if new_rr is not None else 'n/a'})"
                            logger.warning(f"???  {log_prefix} REJECTED (Rescue Recheck): {reason}")
                            return {'status': 'rejected', 'reason': reason, 'stage': 'volume_policy'}
                        logger.info(f"?? {log_prefix} Deferred signal passed rescue recheck (RR={new_rr:.2f}).")

                # Check 2: New signals
                elif volume_rank is not None:
                    if volume_rank >= normal_rank:
                        pass  # Standard profile, proceed
                    elif is_low_bucket:
                        stop_pct, stop_distance, stop_px_used = self._calc_stop_metrics(enriched_signal)
                        tight_stop = stop_pct is not None and stop_pct < tight_stop_threshold
                        wide_stop = stop_pct is not None and stop_pct > wide_stop_threshold

                        if tight_stop:
                            gate_margin_bps = None
                            if stop_pct is not None and math.isfinite(stop_pct):
                                gate_margin_bps = max(0.0, (tight_stop_threshold - stop_pct) * 10000.0)
                            if gate_margin_bps is not None and gate_margin_bps > micro_gate_margin_bps:
                                self.processing_stats['rejected_signals'] += 1
                                drop_item = {
                                    "payload": self._json_sanitize(enriched_signal),
                                    "first_seen_ts_ms": self._now_ms(),
                                    "attempts": 0,
                                    "reason_code": "volume.low_vol_tight_stop_far",
                                    "dedupe_key": enriched_signal.get("dedupe_key"),
                                    "stage": "volume_policy",
                                }
                                self._emit_waiting_room_event(
                                    "waiting_room_drop",
                                    drop_item,
                                    drop_reason="gate_far_from_pass",
                                    gate_margin_bps=gate_margin_bps,
                                    gate_threshold=tight_stop_threshold,
                                    gate_threshold_bps=tight_stop_threshold * 10000.0,
                                    stop_distance=stop_distance,
                                    stop_pct=stop_pct,
                                    px_used=stop_px_used,
                                    px_source="entry_price" if stop_px_used is not None else None,
                                )
                                logger.warning(
                                    "??  %s DROPPED (Low Vol Tight Stop Far) | margin_bps=%.2f threshold_bps=%.2f",
                                    log_prefix,
                                    gate_margin_bps,
                                    tight_stop_threshold * 10000.0,
                                )
                                return {
                                    "status": "dropped",
                                    "reason": "gate_far_from_pass",
                                    "reason_code": "volume.low_vol_tight_stop_far",
                                    "stage": "volume_policy",
                                }
                            logger.warning(
                                f"{log_prefix} Signal DEFERRED due to Low Volatility (tight stop). "
                                f"Reason=volume.low_vol_tight_stop strategy={strategy_name} symbol={symbol}"
                            )
                            condition_data = enriched_signal.get("condition_data")
                            if not isinstance(condition_data, dict):
                                condition_data = {}
                            condition_data.update(
                                {
                                    "gate_margin_bps": gate_margin_bps,
                                    "gate_threshold": tight_stop_threshold,
                                    "stop_distance": stop_distance,
                                    "stop_pct": stop_pct,
                                    "watch_kind": "micro_gate_watch",
                                    "watch_interval_ms": self._micro_gate_watch_interval_ms_default,
                                    "max_checks": self._micro_gate_watch_max_checks_default,
                                    "ttl_ms": self._micro_gate_watch_ttl_ms_default,
                                }
                            )
                            enriched_signal["condition_data"] = condition_data
                            incubated = await self.incubate_signal(
                                strategy_name=strategy_name,
                                signal=enriched_signal,
                                reason_code="volume.low_vol_tight_stop",
                                refresh_policy="REPRICE_AND_RESIZE",
                                stage="volume_policy",
                            )
                            if incubated.get("status") == "incubated":
                                return {
                                    "status": "incubated",
                                    "reason_code": "volume.low_vol_tight_stop",
                                    "stage": "volume_policy",
                                    "dedupe_key": incubated.get("dedupe_key"),
                                }
                            return incubated

                        elif wide_stop:
                            enriched_signal['execution_params'] = {'type': 'LIMIT', 'post_only': True}
                            enriched_signal['position_size_multiplier'] = LOW_LIMIT_MULTIPLIER
                            logger.info(f"?? {log_prefix} Applying LOW_WIDE_STOP_STRICT (mult={LOW_LIMIT_MULTIPLIER})")
                        else:
                            enriched_signal['execution_params'] = {'type': 'LIMIT', 'post_only': True}
                            enriched_signal['position_size_multiplier'] = 0.50
                            logger.info(f"?? {log_prefix} Applying LOW_NORMAL_STOP (mult=0.50)")
            
            # Adım 3: Duplikasyon ve Cooldown Kontrolü
            run_duplicate_early = not (pyramiding_enabled and intent == INTENT_SCALE_IN)
            duplicate_checked = False
            if run_duplicate_early:
                is_valid_duplicate, duplicate_reason = self.validate_duplicate(enriched_signal, strategy_name)
                duplicate_checked = True
                if not is_valid_duplicate:
                    self.processing_stats['rejected_signals'] += 1
                    self.processing_stats['duplicate_rejections'] += 1
                    return {'status': 'rejected', 'reason': duplicate_reason, 'stage': 'duplicate_validation'}
            
            # Adım 4: ML ile Sinyali Geliştirme
            if hasattr(self, 'ml_integration') and self.ml_integration:
                enriched_signal = await self._enhance_signal_with_ml(enriched_signal)
                if enriched_signal is None:
                    self.processing_stats['rejected_signals'] += 1
                    self.processing_stats['ml_blocked_signals'] = self.processing_stats.get('ml_blocked_signals', 0) + 1
                    rejection_reason = self._consume_ml_rejection_reason() or 'ML/RL enhancement blocked signal'
                    return {'status': 'rejected', 'reason': rejection_reason, 'stage': 'ml_enhancement'}
            
            # Adım 5: Çatışma Kontrolü
            conflict_check = await self._check_signal_conflicts(enriched_signal)
            if conflict_check['has_conflict']:
                self.processing_stats['conflicted_signals'] += 1
                # (DÜZELTİLDİ)
                logger.info(f"🚦 {log_prefix} Conflict Detected: {conflict_check['conflicts']}. Resolving...")
                
                resolution = await self.resolve_signal_conflicts(enriched_signal, conflict_check['conflicting_signals'])
                
                if resolution['action'] == 'reject':
                    self.processing_stats['rejected_signals'] += 1
                    # --- TELEMETRİ: Ret Sebebi (DÜZELTİLDİ) ---
                    logger.warning(f"🛡️  {log_prefix} REJECTED (Conflict): {resolution['reason']}")
                    return {'status': 'rejected', 'reason': resolution['reason'], 'stage': 'conflict_resolution'}
            
            # Adım 6: Risk Değerlendirmesi
            quality_result = self._compute_signal_quality(enriched_signal)

            risk_assessment = await self._assess_signal_risk(enriched_signal, strategy_name)
            if not risk_assessment['acceptable']:
                risk_reason_code = risk_assessment.get("reason_code")
                risk_reason = str(risk_assessment.get("reason") or "")
                risk_metrics = risk_assessment.get("metrics", {}) if isinstance(risk_assessment, dict) else {}
                if not isinstance(risk_metrics, dict):
                    risk_metrics = {}

                if (
                    risk_reason_code == "risk.concurrent.max_positions_per_symbol"
                    and (
                        risk_reason == "scale_in_quality_below_threshold"
                        or str(risk_metrics.get("dynamic_scaling_denial") or "") == "scale_in_quality_below_threshold"
                        or str(risk_metrics.get("blocked_by") or "") == "RiskManager._can_dynamic_scale"
                    )
                ):
                    corrected_code = "risk.scale_in.quality_below_threshold"
                    logger.error(
                        "[INCUBATOR] Sanity guard: scale-in quality rejection misdiagnosed as %s | sym=%s strat=%s reason=%s",
                        risk_reason_code,
                        enriched_signal.get("symbol"),
                        strategy_name,
                        risk_reason,
                    )
                    pseudo_item = {
                        "payload": self._json_sanitize(enriched_signal),
                        "first_seen_ts_ms": self._now_ms(),
                        "attempts": 0,
                        "reason_code": corrected_code,
                        "dedupe_key": enriched_signal.get("dedupe_key"),
                    }
                    self._emit_waiting_room_event(
                        "waiting_room_drop",
                        pseudo_item,
                        drop_reason="incubator.sanity_guard.misdiagnosed_scale_in_quality",
                        observed_reason_code=risk_reason_code,
                        observed_reason=risk_reason,
                    )
                    self.processing_stats['rejected_signals'] += 1
                    return {
                        'status': 'rejected',
                        'reason': risk_reason,
                        'reason_code': corrected_code,
                        'stage': 'risk_assessment',
                    }

                if risk_reason_code == "risk.scale_in.quality_below_threshold":
                    self.processing_stats['rejected_signals'] += 1
                    blocked_item = {
                        "payload": self._json_sanitize(enriched_signal),
                        "first_seen_ts_ms": self._now_ms(),
                        "attempts": 0,
                        "reason_code": "incubator.blocked.risk.scale_in.quality_below_threshold",
                        "dedupe_key": enriched_signal.get("dedupe_key"),
                        "stage": "risk_assessment",
                    }
                    self._emit_waiting_room_event(
                        "waiting_room_drop",
                        blocked_item,
                        drop_kind="sanity_guard",
                        blocked_reason_code=risk_reason_code,
                        blocked_reason=risk_reason,
                    )
                    logger.warning(
                        "🛡️  %s REJECTED (Scale-In Quality) | reason_code=%s | reason=%s",
                        log_prefix,
                        risk_reason_code,
                        risk_reason,
                    )
                    return {
                        'status': 'rejected',
                        'reason': risk_reason,
                        'reason_code': risk_reason_code,
                        'stage': 'risk_assessment',
                    }

                if risk_reason_code in (
                    "risk.concurrent.max_open_positions",
                    "risk.concurrent.max_positions_per_symbol",
                ):
                    logger.warning(
                        "🕒 %s DEFERRED (Risk Concurrent Limit) | reason_code=%s | reason=%s",
                        log_prefix,
                        risk_reason_code,
                        risk_assessment.get("reason"),
                    )
                    incubated = await self.incubate_signal(
                        strategy_name=strategy_name,
                        signal=enriched_signal,
                        reason_code=str(risk_reason_code),
                        refresh_policy="NONE",
                        stage="risk_assessment",
                    )
                    if incubated.get("status") == "incubated":
                        return {
                            "status": "incubated",
                            "reason_code": str(risk_reason_code),
                            "reason": risk_reason,
                            "stage": "risk_assessment",
                            "dedupe_key": incubated.get("dedupe_key"),
                        }
                    return incubated

                if risk_reason_code in (
                    "risk.planner.heat_exhausted",
                    "risk.concurrent.portfolio_heat",
                    "risk.concurrent.portfolio_heat_exceeded",
                ):
                    logger.warning(
                        "🕒 %s DEFERRED (Risk Heat) | reason_code=%s | reason=%s",
                        log_prefix,
                        risk_reason_code,
                        risk_reason,
                    )
                    incubated = await self.incubate_signal(
                        strategy_name=strategy_name,
                        signal=enriched_signal,
                        reason_code=str(risk_reason_code),
                        refresh_policy="REPRICE_AND_RESIZE",
                        stage="risk_assessment",
                    )
                    if incubated.get("status") == "incubated":
                        return {
                            "status": "incubated",
                            "reason_code": str(risk_reason_code),
                            "reason": risk_reason,
                            "stage": "risk_assessment",
                            "dedupe_key": incubated.get("dedupe_key"),
                        }
                    return incubated

                self.processing_stats['rejected_signals'] += 1
                # --- TELEMETRİ: Ret Sebebi (DÜZELTİLDİ) ---
                logger.warning(f"🛡️  {log_prefix} REJECTED (Risk Check): {risk_assessment['reason']}")
                return {
                    'status': 'rejected',
                    'reason': risk_assessment['reason'],
                    'reason_code': risk_reason_code,
                    'stage': 'risk_assessment',
                }

            # Late duplicate check for scale-in when pyramiding is enabled (soft guard)
            if not duplicate_checked:
                is_valid_duplicate, duplicate_reason = self.validate_duplicate(enriched_signal, strategy_name)
                if not is_valid_duplicate:
                    self.processing_stats['rejected_signals'] += 1
                    self.processing_stats['duplicate_rejections'] += 1
                    return {'status': 'rejected', 'reason': duplicate_reason, 'stage': 'duplicate_validation'}
            
            # Adim 7: Sinyali ve Rota Bilgisini Hazirla
            routing_result = self._route_signal(enriched_signal, risk_assessment)
            signal_id = self._ensure_signal_id(strategy_name, enriched_signal)
            dedupe_key = self._derive_dedupe_key(strategy_name, enriched_signal)
            if dedupe_key:
                enriched_signal["dedupe_key"] = dedupe_key
                self._active_dedupe_by_key[dedupe_key] = signal_id
                self._active_dedupe_by_signal_id[signal_id] = dedupe_key

            # --- Emit Signal Breakdown ---
            self.emit_signal_breakdown(enriched_signal, quality_result)
            
            self.active_signals[signal_id] = {
                'signal': enriched_signal, 'risk_assessment': risk_assessment,
                'routing': routing_result, 'timestamp': datetime.now(timezone.utc), 'status': 'active'
            }

            # Adım 8: Sinyali Yürütme Kuyruğuna Ekle
            put_result = await self.signal_queue.put({
                'signal_id': signal_id,
                'signal': enriched_signal,
                'risk_assessment': risk_assessment,
                'routing': routing_result
            })
            queued = False
            queue_reason = None
            queue_reason_code = None
            if isinstance(put_result, tuple):
                if len(put_result) >= 1:
                    queued = bool(put_result[0])
                if len(put_result) >= 2:
                    queue_reason = put_result[1]
                if len(put_result) >= 3:
                    queue_reason_code = put_result[2]
            else:
                queued = bool(put_result)

            if not queued:
                if queue_reason_code in ("queue.capacity", "queue.symbol_pending_limit"):
                    # Keep active registry aligned with true queue outcome
                    self.discard_active_signal(signal_id)
                    logger.warning(
                        "🕒 %s DEFERRED (Queue Limit) | reason_code=%s | reason=%s | signal_id=%s",
                        log_prefix,
                        queue_reason_code,
                        queue_reason,
                        signal_id,
                    )
                    incubated = await self.incubate_signal(
                        strategy_name=strategy_name,
                        signal=enriched_signal,
                        reason_code=str(queue_reason_code),
                        refresh_policy="NONE",
                        stage="queue",
                    )
                    if incubated.get("status") == "incubated":
                        return {
                            "status": "incubated",
                            "reason_code": str(queue_reason_code),
                            "reason": queue_reason,
                            "stage": "queue",
                            "dedupe_key": incubated.get("dedupe_key"),
                        }
                    return incubated

                self.processing_stats['rejected_signals'] += 1
                self.processing_stats['queue_rejections'] += 1
                # Keep active registry aligned with true queue outcome
                self.discard_active_signal(signal_id)
                logger.info(
                    "🚫 %s REJECTED | reason=%s | symbol=%s strategy=%s side=%s tf=%s signal_id=%s prio=%s",
                    log_prefix,
                    queue_reason or "queue_rejected",
                    enriched_signal.get('symbol'),
                    strategy_name,
                    enriched_signal.get('side'),
                    enriched_signal.get('timeframe', '5m'),
                    signal_id,
                    enriched_signal.get('priority'),
                )
                return {
                    'status': 'rejected',
                    'reason': queue_reason,
                    'reason_code': queue_reason_code,
                    'stage': 'queue',
                }

            logger.info(
                "✅ %s ENQUEUED | symbol=%s strategy=%s side=%s tf=%s signal_id=%s prio=%s",
                log_prefix,
                enriched_signal.get('symbol'),
                strategy_name,
                enriched_signal.get('side'),
                enriched_signal.get('timeframe', '5m'),
                signal_id,
                enriched_signal.get('priority'),
            )
            
            self.processing_stats['accepted_signals'] += 1
            self._add_signal_history_entry({
                'signal_id': signal_id, 'strategy_name': strategy_name,
                'symbol': enriched_signal.get('symbol'), 'timestamp': datetime.now(timezone.utc), 'status': 'accepted'
            })

            try:
                meta = enriched_signal.get("meta") if isinstance(enriched_signal, dict) else None
                parent_pending_id = meta.get("parent_pending_id") if isinstance(meta, dict) else None
            except Exception:
                parent_pending_id = None
            if parent_pending_id:
                self._emit_soft_deferral_salvaged(
                    parent_pending_id=parent_pending_id,
                    signal_id=signal_id,
                    final_status="accepted",
                    signal_payload=enriched_signal,
                )

            # Bu log aslında gereksiz çünkü yukarıda daha detaylı ENQUEUED log'u var.
            # Ama yine de hatayı düzeltelim. (DÜZELTİLDİ)
            # Orijinal kodda bu satır vardı, ancak process_strategy_signal içinde logger.info(f"Signal accepted and queued: {signal_id}") satırı yoktu.
            # Muhtemelen başka bir yerden karıştırıldı, yine de benim eklediğim ve sonra kaldırdığım bir satır olabilir.
            # En güncel dosyada bu satır bulunmuyor. Bu yüzden yorum satırı olarak bırakmak en temizi.
            # logger.info(f"Signal accepted and queued: {signal_id}")
            
            return {
                'status': 'accepted', 'signal_id': signal_id,
                'enriched_signal': enriched_signal, 'risk_assessment': risk_assessment, 'routing': routing_result
            }
            
        except Exception as e:
            # (DÜZELTİLDİ)
            logger.error(f"💥 Error processing signal from {strategy_name}: {e}", exc_info=True)
            self.processing_stats['rejected_signals'] += 1
            return {'status': 'error', 'reason': str(e), 'stage': 'processing'}

    def _remember_ml_rejection(self, reason: str) -> None:
        """Store ML/RL rejection context so the caller can surface a precise reason."""
        self._last_ml_rejection_reason = reason

    def _consume_ml_rejection_reason(self) -> Optional[str]:
        """Pop the last ML/RL rejection reason, if any."""
        reason = self._last_ml_rejection_reason
        self._last_ml_rejection_reason = None
        return reason

    def _record_rl_telemetry(
        self,
        rl_meta: Dict[str, Any],
        rl_action: str,
        q_std: Optional[float] = None,
        q_range: Optional[float] = None
    ) -> None:
        """Track RL decision telemetry for observability dashboards."""

        telemetry = self.rl_telemetry
        telemetry['total_decisions'] += 1

        if rl_meta.get('bias_applied'):
            telemetry['bias_applied_count'] += 1
        else:
            telemetry['bias_skipped_count'] += 1

        if q_std is not None:
            telemetry['q_std_values'].append(float(q_std))
        if q_range is not None:
            telemetry['q_range_values'].append(float(q_range))

        if rl_meta.get('bypassed'):
            telemetry['bypass_count'] += 1

        if rl_action == 'hold':
            telemetry['veto_count'] += 1
            regime_key = rl_meta.get('regime_label') or rl_meta.get('market_regime') or 'neutral'
            telemetry['veto_by_regime'].setdefault(regime_key, 0)
            telemetry['veto_by_regime'][regime_key] += 1

    def _record_ppo_telemetry(self, score: float) -> None:
        """Track PPO adapter activity for debugging dashboards."""
        metrics = self.rl_telemetry.setdefault('ppo', {
            'samples': 0,
            'long_votes': 0,
            'flat_votes': 0,
            'score_sum': 0.0
        })
        metrics['samples'] += 1
        metrics['score_sum'] += float(score)
        if score >= 0.5:
            metrics['long_votes'] += 1
        else:
            metrics['flat_votes'] += 1

    def on_ml_components_connected(self) -> None:
        """Hook invoked when ML pipelines are wired in (lazily start PPO)."""
        self._initialize_ppo_adapter_if_ready()

    def _initialize_ppo_adapter_if_ready(self) -> None:
        """Instantiate the PPO adapter once all prerequisites are available."""
        if self.ppo_adapter or self._ppo_adapter_failed:
            return
        rl_cfg = getattr(self, '_rl_config', {}) or {}
        if not rl_cfg.get('ppo_enabled'):
            # Only log once to avoid spam
            if not getattr(self, '_ppo_disabled_logged', False):
                logger.info("ℹ️ [PPO] Adapter disabled in config.")
                self._ppo_disabled_logged = True
            return
        if PPOTradingAdapter is None:
            if not self._ppo_missing_dependency_logged:
                logger.warning("⚠️ [PPO] Adapter requested but dependency unavailable (import failed).")
                self._ppo_missing_dependency_logged = True
            return
        if not self.market_data_pipeline or not self.feature_pipeline:
            return
        try:
            self.ppo_adapter = PPOTradingAdapter(
                rl_cfg,
                market_data_pipeline=self.market_data_pipeline,
                feature_pipeline=self.feature_pipeline,
            )
            logger.info(f"✅ [PPO] Adapter initialized successfully. Symbols: {rl_cfg.get('ppo_symbols')}")
        except Exception as exc:  # pragma: no cover - adapter init safety
            self._ppo_adapter_failed = True
            logger.error(
                "❌ [PPO] StrategyCoordinator failed to initialize PPO adapter: %s",
                exc,
                exc_info=True,
            )

    @staticmethod
    def _map_signal_side_to_rl_action(side: str) -> str:
        """Map original strategy side to RL action semantics."""

        normalized = (side or '').lower()
        if normalized in ('buy', 'long'):
            return 'buy'
        if normalized in ('sell', 'short'):
            return 'sell'
        return 'hold'

    def get_rl_telemetry_stats(self) -> Dict[str, Any]:
        """Summarize RL telemetry insights for diagnostics."""
        telemetry = self.rl_telemetry
        total = telemetry['total_decisions']
        q_std_values = telemetry.get('q_std_values', [])
        q_range_values = telemetry.get('q_range_values', [])

        std_mean = float(np.mean(q_std_values)) if q_std_values else 0.0
        std_median = float(np.median(q_std_values)) if q_std_values else 0.0
        range_mean = float(np.mean(q_range_values)) if q_range_values else 0.0
        range_median = float(np.median(q_range_values)) if q_range_values else 0.0

        rl_bypass_rate = (telemetry['bypass_count'] / total) if total else 0.0

        ppo_metrics = telemetry.get('ppo', {})
        ppo_samples = ppo_metrics.get('samples', 0)
        ppo_avg_score = (
            ppo_metrics.get('score_sum', 0.0) / ppo_samples if ppo_samples else 0.0
        )

        return {
            'rl_veto_rate': (telemetry['veto_count'] / total) if total else 0.0,
            'bias_applied_rate': (telemetry['bias_applied_count'] / total) if total else 0.0,
            'bias_skipped_rate': (telemetry['bias_skipped_count'] / total) if total else 0.0,
            'q_std_mean': std_mean,
            'q_std_median': std_median,
            'q_range_mean': range_mean,
            'q_range_median': range_median,
            'bypass_rate': rl_bypass_rate,
            'rl_bypass_rate': rl_bypass_rate,
            'veto_by_regime': telemetry['veto_by_regime'],
            'samples': total,
            'q_std_values': list(q_std_values),
            'q_range_values': list(q_range_values),
            'ppo_samples': ppo_samples,
            'ppo_long_votes': ppo_metrics.get('long_votes', 0),
            'ppo_flat_votes': ppo_metrics.get('flat_votes', 0),
            'ppo_avg_score': ppo_avg_score
        }
    
    async def _enhance_signal_with_ml(self, signal: Dict) -> Optional[Dict]:
        """
        Enhance signal with all ML predictions and apply RL agent's decision as a gatekeeper.
        Uses price prediction, regime prediction, and RL agent recommendations.
        
        UPDATED: Added extreme condition bypass to prevent RL veto during obvious market conditions.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Enhanced signal dict or None if ML/RL blocks the signal.
        """
        if not hasattr(self, 'ml_integration') or not self.ml_integration:
            return signal
        
        try:
            import numpy as np
            import pandas as pd
            current_price = signal.get('entry', 0)
            symbol = signal.get('symbol')
            original_side = (signal.get('side') or '').lower()

            # --- EXTREME CONDITION BYPASS CHECK (BEFORE ANY ML/RL PROCESSING) ---
            rsi_value = await self._extract_rsi_from_market_data(symbol)
            if rsi_value is not None:
                bypass_triggered = await self._check_extreme_condition_bypass(
                    signal, rsi_value, symbol, original_side
                )
                if bypass_triggered:
                    logger.warning(
                        "🔥 [EXTREME-BYPASS] Signal APPROVED without ML/RL checks | symbol=%s | side=%s | rsi=%.2f",
                        symbol,
                        original_side.upper(),
                        rsi_value
                    )
                    # Bypass confirmed - skip RL veto and return signal with minimal ML enhancement
                    signal['bypass_triggered'] = True
                    signal['bypass_rsi'] = rsi_value
                    signal.setdefault('extreme_bypass', True)
                    signal.setdefault('extreme_type', signal.get('extreme_type'))
                    signal.setdefault('extreme_rsi', rsi_value)
                    signal.setdefault('ml_confidence', 0.8)
                    signal['ml_strength'] = signal.get('ml_strength', signal.get('strength'))
                    self.processing_stats['bypass_approvals'] += 1
                    total_signals = self.processing_stats.get('total_signals', 0)
                    if total_signals > 0:
                        self.processing_stats['bypass_success_rate'] = (
                            self.processing_stats['bypass_approvals'] / total_signals
                        ) * 100
                    return signal

            # --- 1. ML ZENGİNLEŞTİRMESİ (MEVCUT YAPI KORUNUYOR) ---
            try:
                enhancement = await self.ml_integration.enhance_strategy_signal(
                    symbol, signal, current_price
                )
                signal.update(enhancement)
            except Exception as e:
                logger.debug(f"ML strategy integration failed: {e}")

            # --- 2. RL AGENT'IN AYRI OLARAK ÇAĞRILMASI VE KONTROLÜ ---
            rl_advice = None
            rl_meta: Optional[Dict[str, Any]] = None
            if self.legacy_rl_enabled and hasattr(self, 'rl_agent') and self.rl_agent:
                try:
                    if hasattr(self.rl_agent, 'set_inference_mode') and not getattr(self.rl_agent, '_inference_locked', False):
                        try:
                            self.rl_agent.set_inference_mode()
                            logger.debug("🤖 [RL] Inference mode re-asserted before decision flow.")
                        except Exception as lock_err:
                            logger.warning(f"⚠️ [RL] Unable to force inference mode before signal: {lock_err}")
                    # ✅ DÜZELTME: 'await' eklendi.
                    state_features = await self._extract_rl_state(symbol, current_price)
                    
                    # 💡 YENİ LOGLAMA: Ajanın "gördüğü" durumu logla
                    if state_features is None:
                        logger.warning(
                            f"⚠️ [RL-SKIP] No state features available for {symbol}. Skipping RL agent (signal continues with ML only)."
                        )
                        signal['rl_recommendation'] = 'n/a'
                        signal['rl_skipped'] = True
                        self.processing_stats['rl_skipped_signals'] += 1
                    else:
                        logger.info(f"🤖 [RL-DEBUG] State vector for {symbol} (first 5 features): {np.round(state_features[:5], 4)}")
                        logger.info(f"🤖 [RL-DEBUG] Original Signal: {original_side.upper()}, Strategy: {signal.get('strategy_name')}")
                        logger.info(f"🤖 [RL-DEBUG] Market Regime: {signal.get('predicted_regime', 'neutral')}")

                        rl_meta = {}
                        rl_training_flag = getattr(self.rl_agent, 'training_mode', False)
                        if getattr(self.rl_agent, '_inference_locked', False):
                            rl_training_flag = False
                        regime_context = signal.get('predicted_regime', 'neutral')
                        regime_confidence = signal.get('regime_confidence')
                        if isinstance(regime_confidence, (int, float)):
                            market_regime_payload: Any = {
                                'predicted_regime': regime_context,
                                'confidence': float(regime_confidence)
                            }
                        else:
                            market_regime_payload = regime_context

                        rl_kwargs = {
                            'market_regime': market_regime_payload,
                            'risk_constraints': signal.get('risk_constraints'),
                            'training': rl_training_flag
                        }
                        if hasattr(self.rl_agent, 'get_action_with_meta'):
                            rl_action_index, rl_meta = self.rl_agent.get_action_with_meta(state_features, **rl_kwargs)
                        else:
                            rl_action_index = self.rl_agent.act(state_features, **rl_kwargs)
                            rl_meta = {}

                        rl_advice_str = ['buy', 'hold', 'sell'][rl_action_index]
                        rl_advice = rl_advice_str.lower()

                        raw_q_values = rl_meta.get('raw_q_values') or []
                        q_std = None
                        q_range = None
                        if raw_q_values:
                            q_array = np.array(raw_q_values, dtype=float)
                            q_std = float(np.std(q_array))
                            q_range = float(np.max(q_array) - np.min(q_array))

                        q_std_threshold = (
                            self.config
                            .get('ml', {})
                            .get('reinforcement_learning', {})
                            .get('q_std_bypass_threshold', 1e-4)
                        )
                        q_range_threshold = (
                            self.config
                            .get('ml', {})
                            .get('reinforcement_learning', {})
                            .get('q_range_bypass_threshold', 1e-3)
                        )

                        bypass_reason = None
                        if q_std is not None and q_std < q_std_threshold:
                            bypass_reason = f"low_q_std:{q_std:.6f}"
                        elif q_range is not None and q_range < q_range_threshold:
                            bypass_reason = f"low_q_range:{q_range:.6f}"

                        if bypass_reason:
                            fallback_advice = self._map_signal_side_to_rl_action(original_side)
                            logger.warning(
                                "⚠️ [RL-BYPASS] Model frozen detected for %s (q_std=%.6f, q_range=%.6f) -> fallback=%s",
                                symbol,
                                q_std if q_std is not None else float('nan'),
                                q_range if q_range is not None else float('nan'),
                                fallback_advice.upper()
                            )
                            rl_meta['bypassed'] = True
                            rl_meta['bypass_reason'] = 'frozen_model'
                            rl_advice = fallback_advice
                            rl_advice_str = rl_advice.upper()
                            signal['rl_bypassed'] = True
                            signal['rl_bypass_reason'] = 'frozen_model'
                        else:
                            rl_meta['bypassed'] = False
                            signal['rl_bypassed'] = False
                            signal['rl_bypass_reason'] = None

                        rl_meta['q_std'] = q_std
                        rl_meta['q_range'] = q_range

                        signal['rl_recommendation'] = rl_advice
                        signal['rl_decision_meta'] = rl_meta
                        self._record_rl_telemetry(rl_meta or {}, rl_advice, q_std=q_std, q_range=q_range)
                        
                        # CRITICAL: Store RL decision for enrichment
                        self._last_rl_decision = {
                            'action': rl_advice.upper(),
                            'confidence': rl_meta.get('best_probability', DEFAULT_RL_CONFIDENCE) if rl_meta else DEFAULT_RL_CONFIDENCE,
                            'timestamp': datetime.utcnow().isoformat()
                        }

                        meta_preview = {}
                        if rl_meta:
                            meta_preview = {
                                'probabilities': [round(p, 4) for p in rl_meta.get('probabilities', [])[:3]],
                                'best_probability': rl_meta.get('best_probability'),
                                'exploration': rl_meta.get('exploration'),
                                'epsilon': rl_meta.get('epsilon'),
                                'raw_q_values': [round(v, 4) for v in rl_meta.get('raw_q_values', [])[:3]] if rl_meta.get('raw_q_values') else None,
                                'adjusted_q_values': [round(v, 4) for v in rl_meta.get('adjusted_q_values', [])[:3]] if rl_meta.get('adjusted_q_values') else None,
                                'bias_applied': rl_meta.get('bias_applied'),
                                'effective_bias': rl_meta.get('effective_bias'),
                                'regime_confidence': rl_meta.get('regime_confidence')
                            }
                        logger.debug(
                            "DEBUG_ML_RL | symbol=%s | rl_decision=%s | meta=%s",
                            symbol,
                            rl_advice_str.upper(),
                            meta_preview
                        )

                        # 💡 YENİ LOGLAMA: Ajanın kararını logla
                        logger.info(f"🤖 [RL-DECISION] For {symbol}, Agent decided: {rl_advice.upper()}")
                        probabilities_preview = meta_preview.get('probabilities', []) or None
                        epsilon_preview = meta_preview.get('epsilon') if meta_preview else None
                        confidence_preview = meta_preview.get('best_probability') if meta_preview else None
                        q_values_preview = (
                            meta_preview.get('adjusted_q_values')
                            or meta_preview.get('raw_q_values')
                            or 'N/A'
                        )
                        logger.info(
                            "🤖 [RL-META] %s | decision=%s | epsilon=%s | confidence=%s | bias=%s | q_values=%s | probs=%s",
                            symbol,
                            rl_advice_str.lower(),
                            f"{epsilon_preview:.3f}" if isinstance(epsilon_preview, (int, float)) else 'N/A',
                            f"{confidence_preview:.3f}" if isinstance(confidence_preview, (int, float)) else 'N/A',
                            {
                                'applied': rl_meta.get('bias_applied'),
                                'effective_bias': rl_meta.get('effective_bias'),
                                'threshold': rl_meta.get('regime_confidence_threshold')
                            },
                            q_values_preview,
                            probabilities_preview or 'N/A'
                        )

                except Exception as e:
                    logger.warning(f"RL recommendation failed: {e}", exc_info=True)

            # --- 3. Legacy RL disagreements now apply soft penalties only ---
            if self.legacy_rl_enabled and rl_advice == 'hold':
                logger.warning(f"🤖 [RL-HOLD] Agent advised HOLD for {symbol}; applying soft strength penalty.")
                current_strength = signal.get('ml_strength', signal.get('strength', 0.5) or 0.5)
                signal['ml_strength'] = max(0.05, current_strength * 0.7)
                signal['rl_hold_recommendation'] = True

            if self.legacy_rl_enabled:
                is_opposite = (
                    (original_side in ['buy', 'long'] and rl_advice in ['sell', 'short']) or
                    (original_side in ['sell', 'short'] and rl_advice in ['buy', 'long'])
                )
                if rl_advice and is_opposite:
                    logger.warning(f"⚠️ [RL-DISAGREE] RL Agent disagrees with signal for {symbol}.")
                    logger.warning(f"    Signal Side: {original_side.upper()}, RL Advice: {rl_advice.upper()}")
                    current_strength = signal.get('ml_strength', signal.get('strength', 0.5))
                    new_strength = current_strength * 0.5
                    signal['ml_strength'] = new_strength
                    logger.warning(f"    Signal strength reduced from {current_strength:.2f} to {new_strength:.2f} due to disagreement.")

            # --- 4. PPO tabanlı soft filtre uygulanır (yalnızca long sinyaller) ---
            await self._apply_ppo_long_filter(signal)

            # --- 5. SON LOGLAMA VE SİNYALİ DÖNDÜRME ---
            if 'position_size' in signal and signal.get('ml_confidence'):
                signal['position_size'] *= signal['ml_confidence']

            logger.info(f"🧠 [ML] Signal enhanced: {symbol}")
            if 'ml_strength' in signal:
                logger.info(f"   Strength: {signal.get('strength', 0.5):.2f} → {signal['ml_strength']:.2f}")
            if 'predicted_regime' in signal:
                 logger.info(f"   Regime: {signal['predicted_regime']} ({signal.get('regime_confidence', 0):.2%})")
            if 'rl_recommendation' in signal:
                logger.info(f"   RL Says: {signal['rl_recommendation']}")
            
            return signal
            
        except Exception as e:
            logger.error(f"ML enhancement failed critically: {e}", exc_info=True)
            return signal

    async def _apply_ppo_long_filter(self, signal: Dict[str, Any]) -> None:
        """Apply PPO-based soft gating for BTC/USDT long signals."""
        side = (signal.get('side') or '').lower()
        requested_symbol = signal.get('symbol')
        
        # Log PPO check attempt
        logger.debug(f"🔍 [PPO-CHECK] Checking PPO for {requested_symbol} ({side})")

        if side not in ('buy', 'long'):
            logger.debug(f"ℹ️ [PPO-SKIP] Signal is {side}, PPO only filters LONGs.")
            return
            
        if not requested_symbol:
            return

        normalized_symbol = self._normalize_symbol_for_ppo(requested_symbol)
        tail_overrides, overrides_meta = self._build_ppo_state_overrides(normalized_symbol)
        self._initialize_ppo_adapter_if_ready()
        adapter = getattr(self, 'ppo_adapter', None)

        score: float
        metadata: Dict[str, Any]
        if adapter:
            try:
                score, metadata = await adapter.get_long_score(
                    normalized_symbol,
                    position_fraction=tail_overrides.get('position_fraction'),
                    normalized_pv=tail_overrides.get('normalized_pv'),
                    override_meta=overrides_meta,
                )
                
                # Log if symbol was unsupported by adapter
                if metadata.get('reason') == 'unsupported_symbol':
                    logger.debug(f"ℹ️ [PPO-SKIP] Symbol {requested_symbol} not in PPO universe.")
                    
            except Exception as exc:  # pragma: no cover - extra safety
                logger.warning(f"⚠️ [PPO-LONG] Adapter error for {requested_symbol}: {exc}")
                score = self.ppo_fallback_score
                metadata = {'reason': 'exception', 'error': str(exc)}
        else:
            score = self.ppo_fallback_score
            metadata = {'reason': 'adapter_unavailable'}
            logger.debug("ℹ️ [PPO-SKIP] Adapter unavailable.")

        score = float(score)
        metadata = dict(metadata or {})
        if tail_overrides:
            metadata['state_tail'] = tail_overrides
        metadata.setdefault('symbol', normalized_symbol)
        metadata.setdefault('normalized_symbol', normalized_symbol)
        metadata.setdefault('requested_symbol', requested_symbol)

        signal['ppo_long_score'] = score
        signal['ppo_meta'] = metadata
        lookback_meta = metadata.get('lookback') if isinstance(metadata, dict) else None
        if lookback_meta:
            signal['ppo_lookback_meta'] = lookback_meta

        action_label = 'BUY' if score >= 0.5 else 'HOLD'
        
        # Enhanced logging for active PPO decisions
        if metadata.get('reason') not in ('unsupported_symbol', 'disabled'):
            logger.info(
                "🤖 [PPO-DECISION] %s | Action: %s | Score: %.2f | Conf: %.2f",
                requested_symbol,
                action_label,
                score,
                metadata.get('confidence', 0.0)
            )

        self._record_ppo_telemetry(score)
        signal['rl_recommendation'] = action_label.lower()
        base_meta = signal.get('rl_decision_meta')
        if not isinstance(base_meta, dict):
            base_meta = {}
        signal['rl_decision_meta'] = {**base_meta, 'ppo': metadata}
        self._last_rl_decision = {
            'action': action_label,
            'confidence': score,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'ppo'
        }

    async def monitor_ppo_state(self, symbol: str) -> None:
        """
        Force PPO inference for telemetry purposes (Shadow Mode).
        This ensures PPO logs appear even when no trade signal is present.
        """
        if not symbol:
            return

        # 1. Check if PPO is enabled in config
        rl_cfg = getattr(self, '_rl_config', {}) or {}
        if not rl_cfg.get('ppo_enabled'):
            return

        # 2. Initialize adapter if needed
        self._initialize_ppo_adapter_if_ready()
        adapter = getattr(self, 'ppo_adapter', None)
        
        if not adapter:
            return

        # 3. Run inference (Shadow Mode)
        try:
            normalized_symbol = self._normalize_symbol_for_ppo(symbol)
            tail_overrides, overrides_meta = self._build_ppo_state_overrides(normalized_symbol)

            score, metadata = await adapter.get_long_score(
                normalized_symbol,
                position_fraction=tail_overrides.get('position_fraction'),
                normalized_pv=tail_overrides.get('normalized_pv'),
                override_meta=overrides_meta,
            )

            if metadata.get("cache_hit"):
                return
            
            # 4. Log result with specific tag for monitoring
            if metadata.get('reason') != 'unsupported_symbol':
                action = "BUY" if score >= 0.5 else "HOLD"
                logger.info(
                    f"👀 [PPO-MONITOR] {symbol} | Score(Raw): {score:.4f} | Action: {action} | "
                    f"Conf: {metadata.get('confidence', 0.0):.2f}"
                )
                
                # Record telemetry
                self._record_ppo_telemetry(score)
                debug_meta = metadata.get("debug", {}) if isinstance(metadata, dict) else {}
                override_info = debug_meta.get("override_meta") or overrides_meta or {}
                logger.info(
                    "[PPO-DEBUG] sym=%s tf=%s src=%s last_candle_ts=%s prev_last_candle_ts=%s age_s=%s rows=%s "
                    "state_hash=%s state_mean=%.4f state_std=%.4f state_min=%.4f state_max=%.4f "
                    "feat_std=%.4f feat_min=%.4f feat_max=%.4f extra_std=%.4f tail_pf=%.3f tail_pv=%.3f "
                    "tail_default=%s nan=%s inf=%s head3=%s tail3=%s action_int=%s "
                    "p_flat=%.6f p_long=%.6f p_margin=%.6f conf_raw=%.6f entropy_raw=%.6f "
                    "health_ok=%s health_reasons=%s p_long_std=%.6f "
                    "obs_norm_present=%s vecnorm_loaded=%s obs_norm_applied=%s obs_clip_frac=%.6f clip_mean=%.6f "
                    "z_abs_mean=%.6f z_abs_p99=%.6f cache_hit=%s override_ok=%s override_reason=%s",
                    symbol,
                    debug_meta.get("timeframe", "1h"),
                    debug_meta.get("source", "unknown"),
                    debug_meta.get("last_ts"),
                    debug_meta.get("prev_last_ts"),
                    debug_meta.get("age_sec"),
                    debug_meta.get("rows"),
                    debug_meta.get("state_hash"),
                    self._safe_float(debug_meta.get("state_mean")),
                    self._safe_float(debug_meta.get("state_std")),
                    self._safe_float(debug_meta.get("state_min")),
                    self._safe_float(debug_meta.get("state_max")),
                    self._safe_float(debug_meta.get("feat_std")),
                    self._safe_float(debug_meta.get("feat_min")),
                    self._safe_float(debug_meta.get("feat_max")),
                    self._safe_float(debug_meta.get("extra_std")),
                    self._safe_float(debug_meta.get("tail_pf")),
                    self._safe_float(debug_meta.get("tail_pv")),
                    bool(debug_meta.get("tail_default")),
                    debug_meta.get("nan_count"),
                    debug_meta.get("inf_count"),
                    debug_meta.get("state_head3"),
                    debug_meta.get("state_tail3"),
                    debug_meta.get("action_int"),
                    self._safe_float(debug_meta.get("p_flat")),
                    self._safe_float(debug_meta.get("p_long")),
                    self._safe_float(debug_meta.get("p_margin")),
                    self._safe_float(debug_meta.get("conf_raw")),
                    self._safe_float(debug_meta.get("entropy_raw")),
                    bool(debug_meta.get("health_ok")),
                    debug_meta.get("health_reasons"),
                    self._safe_float(debug_meta.get("p_long_std")),
                    bool(debug_meta.get("obs_norm_present")),
                    bool(debug_meta.get("vecnorm_loaded")),
                    bool(debug_meta.get("obs_norm_applied")),
                    self._safe_float(debug_meta.get("obs_clip_frac")),
                    self._safe_float(debug_meta.get("clip_mean")),
                    self._safe_float(debug_meta.get("z_abs_mean")),
                    self._safe_float(debug_meta.get("z_abs_p99")),
                    bool(debug_meta.get("cache_hit")),
                    bool(override_info.get("override_ok")),
                    override_info.get("reason"),
                )
            else:
                 # Log unsupported symbol at debug level
                 logger.debug(f"ℹ️ [PPO-MONITOR] Unsupported symbol: {symbol} (norm: {normalized_symbol})")

        except Exception as e:
            # Log at WARNING level to avoid spamming errors if PPO fails frequently in shadow mode
            logger.warning(f"⚠️ [PPO-MONITOR] Failed for {symbol}: {e}", exc_info=True)

    @staticmethod
    def _normalize_symbol_for_ppo(symbol: Optional[str]) -> str:
        if not symbol:
            return ""
        normalized = symbol.strip().upper().replace('-', '/')
        if ':' in normalized:
            normalized = normalized.split(':', 1)[0]
        return normalized

    def _build_ppo_state_overrides(self, symbol: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
        overrides: Dict[str, float] = {}
        meta: Dict[str, Any] = {"override_ok": False, "reason": None}
        normalized_symbol = self._normalize_symbol_for_ppo(symbol)
        if not self.portfolio_manager or not hasattr(self.portfolio_manager, 'get_open_positions'):
            meta["reason"] = "no_portfolio_manager"
        else:
            pos_fraction = self._compute_symbol_position_fraction(normalized_symbol)
            if pos_fraction is not None:
                overrides['position_fraction'] = pos_fraction
            else:
                meta["reason"] = meta.get("reason") or "position_fraction_missing"

        if not self.portfolio_manager or not hasattr(self.portfolio_manager, 'get_current_equity'):
            meta["reason"] = meta.get("reason") or "no_equity_provider"
        else:
            normalized_pv = self._compute_normalized_equity()
            if normalized_pv is not None:
                overrides['normalized_pv'] = normalized_pv
            else:
                meta["reason"] = meta.get("reason") or "normalized_pv_missing"
        if overrides:
            meta["override_ok"] = True
        return overrides, meta

    def _compute_symbol_position_fraction(self, symbol: str) -> Optional[float]:
        if not self.portfolio_manager or not hasattr(self.portfolio_manager, 'get_open_positions'):
            return None
        try:
            positions = self.portfolio_manager.get_open_positions()
        except Exception as exc:
            logger.debug("Failed to fetch open positions for PPO tail: %s", exc)
            return None

        total_equity = self._safe_float(getattr(self.portfolio_manager, 'get_current_equity', lambda: 0.0)())
        if total_equity <= 0:
            return None

        symbol_norm = self._normalize_symbol_for_ppo(symbol)
        symbol_notional = 0.0
        for pos in positions.values():
            if self._normalize_symbol_for_ppo(pos.get('symbol')) != symbol_norm:
                continue
            side = (pos.get('side') or pos.get('direction') or '').lower()
            if side and side not in ('buy', 'long'):
                continue
            notional = self._extract_notional(pos)
            symbol_notional += notional

        if symbol_notional <= 0:
            return 0.0
        return min(1.0, symbol_notional / total_equity)

    def _compute_normalized_equity(self) -> Optional[float]:
        if not hasattr(self.portfolio_manager, 'get_current_equity'):
            return None
        try:
            current_equity = self._safe_float(self.portfolio_manager.get_current_equity())
        except Exception as exc:
            logger.debug("Failed to fetch equity for PPO tail: %s", exc)
            return None
        base = self._initial_equity or 1.0
        if base <= 0:
            base = 1.0
        ratio = current_equity / base if current_equity > 0 else 0.0
        return max(0.1, min(5.0, ratio))

    def _derive_initial_equity(self) -> float:
        try:
            if self.portfolio_manager and hasattr(self.portfolio_manager, 'get_current_equity'):
                equity = self._safe_float(self.portfolio_manager.get_current_equity())
                if equity > 0:
                    return equity
        except Exception:
            pass
        risk_cfg = (self.config.get('risk') or {}) if isinstance(self.config, dict) else {}
        fallback_equity = self._safe_float(risk_cfg.get('equity_usd'))
        if fallback_equity and fallback_equity > 0:
            return fallback_equity
        return 1.0

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_notional(position: Dict[str, Any]) -> float:
        if not position:
            return 0.0
        notional = position.get('notional') or position.get('position_notional')
        if notional:
            try:
                return float(notional)
            except (TypeError, ValueError):
                notional = None
        size = position.get('size') or position.get('amount') or position.get('position_size')
        price = position.get('entry_price') or position.get('entry') or position.get('avg_entry_price')
        if size and price:
            try:
                return float(size) * float(price)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _compute_ppo_position_multiplier(self, signal: Dict[str, Any]) -> float:
        side = (signal.get('side') or '').lower()
        if side not in ('buy', 'long'):
            return 1.0
        score = float(signal.get('ppo_long_score', self.ppo_fallback_score))
        base = self.ppo_multipliers['position_base']
        bonus = self.ppo_multipliers['position_bonus']
        multiplier = base + (bonus * score)
        return max(0.1, min(multiplier, 1.0))
    
    async def _extract_rl_state(self, symbol: str, current_price: float) -> Optional['np.ndarray']:
        """
        Extracts a real feature vector from market data for the RL agent state.
        
        This method replaces the random placeholder data with a standardized
        feature set derived from the market data pipeline and feature engine.

        Args:
            symbol: The trading symbol.
            current_price: The current price of the asset (not used directly but good practice).

        Returns:
            A numpy array representing the agent's state, or None if data is unavailable.
        """
        import numpy as np

        # 1. Gerekli bileşenlerin varlığını kontrol et
        if not hasattr(self, 'market_data_pipeline') or not self.market_data_pipeline:
            logger.warning("[RL-STATE] MarketDataPipeline not available. Cannot create RL state.")
            return None
        if not hasattr(self, 'feature_pipeline') or not self.feature_pipeline:
            logger.warning("[RL-STATE] FeatureEngineeringPipeline not available. Cannot create RL state.")
            return None

        try:
            # 2. MarketDataPipeline'dan en güncel, indikatörlü veriyi al
            # Stratejilerle tutarlı olması için '30m' zaman dilimini kullanıyoruz.
            df = await self.market_data_pipeline.get_latest_ohlcv(symbol, "30m")

            if df is None or df.empty:
                logger.warning(f"[RL-STATE] No 30m data available for {symbol} to create RL state.")
                return None

            # 3. FeatureEngineeringPipeline kullanarak standart özellikleri çıkar
            # NOT: Bu metot zaten içinde 'add_indicators' çağırıyor, bu yüzden ek indikatör eklemeye gerek yok.
            # RL state için price feature set kullanılır
            features_df = self.feature_pipeline.extract_features(df, mode='price')
            
            if features_df.empty:
                logger.warning(f"[RL-STATE] Feature extraction failed for {symbol}.")
                return None
            
            # 4. En son (en güncel) özellik satırını al ve eksik verileri temizle
            latest_features = features_df.iloc[-1].values
            
            # NaN (Not a Number) değerler varsa, RL modeli hata verir.
            # Şimdilik bu durumu loglayıp state oluşturmuyoruz.
            if np.isnan(latest_features).any():
                logger.warning(f"[RL-STATE] Latest features for {symbol} contain NaN values. Skipping state creation.")
                return None
            
            logger.debug(f"[RL-STATE] Successfully created RL state for {symbol} with {len(latest_features)} features.")
            return latest_features

        except Exception as e:
            logger.error(f"❌ [RL-STATE] Critical error extracting RL state for {symbol}: {e}", exc_info=True)
            return None
    
    async def _extract_rsi_from_market_data(self, symbol: str) -> Optional[float]:
        """
        Extract current RSI value from market data for extreme condition bypass check.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current RSI value or None if unavailable
        """
        if not hasattr(self, 'market_data_pipeline') or not self.market_data_pipeline:
            return None
        
        try:
            # Get latest 30m data with indicators (same timeframe used by strategies)
            df = await self.market_data_pipeline.get_latest_ohlcv(symbol, "30m")
            
            if df is None or df.empty:
                return None
            
            # Check if RSI column exists
            if 'rsi' not in df.columns:
                return None
            
            # Get the most recent RSI value
            latest_rsi = float(df['rsi'].iloc[-1])
            
            # Sanity check (RSI should be between 0 and 100)
            if 0 <= latest_rsi <= 100:
                return latest_rsi
            
            logger.warning(f"[BYPASS] Invalid RSI value {latest_rsi} for {symbol} (out of 0-100 range)")
            return None
            
        except Exception as e:
            logger.debug(f"[BYPASS] Failed to extract RSI for {symbol}: {e}")
            return None
    
    async def _check_extreme_condition_bypass(
        self, signal: Dict, rsi_value: float, symbol: str, original_side: str
    ) -> bool:
        """
        Check if extreme condition bypass should be triggered.
        
        This method implements the bypass logic that allows obvious trading signals
        to skip RL Agent veto when RSI reaches extreme oversold/overbought levels.
        
        Args:
            signal: Trading signal dictionary
            rsi_value: Current RSI value
            symbol: Trading symbol
            original_side: Signal side ('buy', 'long', 'sell', 'short')
            
        Returns:
            True if bypass is triggered, False otherwise
        """
        # Get config
        config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
        bypass_config = config.get('signals', {}).get('bypass', {})
        
        # Check if bypass is enabled (default: True as per issue requirements)
        if not bypass_config.get('enabled', True):
            return False
        
        # Get thresholds with validation
        oversold_threshold = float(bypass_config.get('rsi_oversold_threshold', 20))
        overbought_threshold = float(bypass_config.get('rsi_overbought_threshold', 80))
        
        # Validate thresholds
        if not (0 <= oversold_threshold <= 100 and 0 <= overbought_threshold <= 100):
            logger.error(
                f"[BYPASS] Invalid RSI thresholds: oversold={oversold_threshold}, "
                f"overbought={overbought_threshold}. Must be in range [0, 100]."
            )
            return False
        
        if oversold_threshold >= overbought_threshold:
            logger.error(
                f"[BYPASS] Invalid RSI thresholds: oversold ({oversold_threshold}) "
                f"must be < overbought ({overbought_threshold})"
            )
            return False
        
        # Normalize side for comparison
        normalized_side = original_side.lower()
        is_buy_signal = normalized_side in ['buy', 'long']
        is_sell_signal = normalized_side in ['sell', 'short']
        
        force_swap_enabled = bypass_config.get('force_swap_enabled', True)

        # Check EXTREME OVERSOLD condition (RSI < oversold_threshold) with BUY signal
        if rsi_value < oversold_threshold and is_buy_signal:
            logger.warning(
                f"🚨 [EXTREME-OVERSOLD-BYPASS] RSI={rsi_value:.2f} < {oversold_threshold}\n"
                f"   Symbol: {symbol}\n"
                f"   Signal: {original_side.upper()}\n"
                f"   Strategy: {signal.get('strategy_name', 'unknown')}\n"
                f"   Entry: ${signal.get('entry', 0):.2f}\n"
                f"   Bypassing all ML/RL checks - SIGNAL CONFIRMED\n"
                f"   Reason: Extreme oversold condition detected"
            )
            signal["extreme_bypass"] = True
            signal["extreme_type"] = "oversold"
            signal["extreme_rsi"] = rsi_value
            if force_swap_enabled:
                self._prepare_force_swap_slot(signal, symbol)
            return True
        
        # Check EXTREME OVERBOUGHT condition (RSI > overbought_threshold) with SELL signal
        if rsi_value > overbought_threshold and is_sell_signal:
            logger.warning(
                f"🚨 [EXTREME-OVERBOUGHT-BYPASS] RSI={rsi_value:.2f} > {overbought_threshold}\n"
                f"   Symbol: {symbol}\n"
                f"   Signal: {original_side.upper()}\n"
                f"   Strategy: {signal.get('strategy_name', 'unknown')}\n"
                f"   Entry: ${signal.get('entry', 0):.2f}\n"
                f"   Bypassing all ML/RL checks - SIGNAL CONFIRMED\n"
                f"   Reason: Extreme overbought condition detected"
            )
            signal["extreme_bypass"] = True
            signal["extreme_type"] = "overbought"
            signal["extreme_rsi"] = rsi_value
            if force_swap_enabled:
                self._prepare_force_swap_slot(signal, symbol)
            return True
        
        # No bypass triggered
        return False

    def _prepare_force_swap_slot(self, signal: Dict, symbol: Optional[str]) -> None:
        """Tag signal for force swap when symbol slots are fully occupied."""
        if not symbol or not isinstance(signal, dict):
            return
        portfolio_manager = getattr(self, 'portfolio_manager', None)
        if portfolio_manager is None:
            return

        max_limit = None
        try:
            if hasattr(self, 'risk_manager') and getattr(self.risk_manager, 'concurrent_limits', None):
                max_limit = getattr(self.risk_manager.concurrent_limits, 'max_positions_per_symbol', None)
        except Exception:
            max_limit = None

        if max_limit is None:
            risk_section = (self.config or {}).get('risk') if isinstance(self.config, dict) else None
            if isinstance(risk_section, dict):
                max_limit = risk_section.get('concurrent_limits', {}).get('max_positions_per_symbol')

        if not max_limit:
            return

        try:
            getter = getattr(portfolio_manager, 'get_open_positions_for_symbol', None)
            if callable(getter):
                positions = getter(symbol)
            else:
                positions = []
        except Exception as exc:
            logger.debug(f"[EXTREME-SWAP] Unable to inspect positions for {symbol}: {exc}")
            return

        if not positions or len(positions) < max_limit:
            return

        incoming_side = str(signal.get('side', '')).lower()

        def _pnl_value(position: Dict[str, Any]) -> float:
            pnl_val = position.get('unrealized_pnl_pct')
            if pnl_val is None:
                metrics = position.get('metrics') or {}
                pnl_val = metrics.get('unrealized_pnl_pct')
            try:
                return float(pnl_val or 0.0)
            except (TypeError, ValueError):
                return 0.0

        weakest = min(positions, key=_pnl_value)
        target_id = weakest.get('position_id')
        if not target_id:
            return

        target_side = str(weakest.get('side', '')).lower()
        same_direction = incoming_side and target_side and incoming_side == target_side

        if same_direction:
            cfg_source = self.config if isinstance(self.config, dict) else {}
            if hasattr(self.portfolio_manager, "cfg") and isinstance(getattr(self.portfolio_manager, "cfg"), dict):
                cfg_source = getattr(self.portfolio_manager, "cfg")
            pyramiding_cfg = cfg_source.get("pyramiding", {}) if isinstance(cfg_source, dict) else {}
            pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False))

            if pyramiding_enabled:
                signal['intent'] = INTENT_SCALE_IN
                signal.pop('swap_target_id', None)
                logger.info(
                    "⚖️ [EXTREME-SAME-SIDE] Keeping position open; converting bypass signal to SCALE_IN | sym=%s | side=%s",
                    symbol,
                    incoming_side,
                )
            else:
                signal['intent'] = INTENT_HOLD
                signal.pop('swap_target_id', None)
                logger.info(
                    "⚖️ [EXTREME-SAME-SIDE] Keeping position open; marking bypass signal as HOLD | sym=%s | side=%s",
                    symbol,
                    incoming_side,
                )
            return

        signal['intent'] = INTENT_FORCE_SWAP
        signal['swap_target_id'] = target_id
        logger.warning(
            "⚡ [EXTREME-SWAP] Marking %s on %s for closure before opening new extreme signal (PnL=%.2f%%)",
            target_id,
            symbol,
            _pnl_value(weakest) * 100,
        )
    
    async def resolve_signal_conflicts(self, new_signal: Dict, 
                                      conflicting_signals: List[Dict],
                                      resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_PRIORITY) -> Dict[str, Any]:
        """
        Resolve conflicts between competing strategy signals.
        
        Args:
            new_signal: New incoming signal
            conflicting_signals: List of conflicting existing signals
            resolution_strategy: Strategy for conflict resolution
            
        Returns:
            Resolution decision and reasoning
        """
        try:
            logger.info(f"Resolving signal conflict using {resolution_strategy.value} strategy")
            
            all_signals = [new_signal] + conflicting_signals
            
            # Apply resolution strategy
            if resolution_strategy == ConflictResolutionStrategy.HIGHEST_PRIORITY:
                winner = self._resolve_by_priority(all_signals)
            elif resolution_strategy == ConflictResolutionStrategy.BEST_RISK_REWARD:
                winner = self._resolve_by_risk_reward(all_signals)
            elif resolution_strategy == ConflictResolutionStrategy.PERFORMANCE_WEIGHTED:
                winner = self._resolve_by_performance(all_signals)
            elif resolution_strategy == ConflictResolutionStrategy.FIRST_IN_FIRST_OUT:
                winner = self._resolve_by_fifo(all_signals)
            else:
                winner = self._resolve_by_priority(all_signals)  # Default
            
            # Determine action for new signal
            if winner['signal_id'] == new_signal.get('signal_id', 'new'):
                action = 'accept'
                reason = f"Won conflict resolution ({resolution_strategy.value})"

                # Reverse intent handling: if the winning signal is
                # opposite side of an existing open position on the
                # same symbol, annotate it as a reverse so that the
                # engine can close then reopen atomically.
                try:
                    symbol = winner.get('symbol') or new_signal.get('symbol')
                    side = str(winner.get('side', '')).lower()
                    position_manager = getattr(self, 'position_manager', None)
                    portfolio_manager = getattr(self, 'portfolio_manager', None)

                    open_position = None
                    if portfolio_manager is not None and hasattr(portfolio_manager, 'get_open_positions_for_symbol'):
                        open_position = portfolio_manager.get_open_positions_for_symbol(symbol)
                    elif position_manager is not None and hasattr(position_manager, 'get_open_position_for_symbol'):
                        open_position = position_manager.get_open_position_for_symbol(symbol)

                    if isinstance(open_position, dict) and open_position.get('position_id'):
                        existing_side = str(open_position.get('side', '')).lower()
                        if existing_side and side and existing_side != side:
                            policy_result = None
                            if getattr(self, "transition_policy", None):
                                policy_result = self.transition_policy.evaluate(
                                    open_position,
                                    winner,
                                    inferred_intent="reverse",
                                )

                            if policy_result and not policy_result.get("allowed", True):
                                if policy_result.get("action") == "convert_to_close":
                                    winner['intent'] = INTENT_CLOSE
                                    winner.setdefault('meta', {}).setdefault('transition_policy', {})
                                    winner['meta']['transition_policy'] = {
                                        'original_intent': 'reverse',
                                        'action': 'converted',
                                        'reason': policy_result.get('reason')
                                    }
                                    logger.info(
                                        "[CONFLICT-RESOLUTION] Reverse blocked, converted to close | signal=%s | reason=%s",
                                        winner.get('signal_id'),
                                        policy_result.get('reason'),
                                    )
                                elif policy_result.get("action") == "reject":
                                    action = 'reject'
                                    reason = f"TransitionPolicy reject: {policy_result.get('reason')}"
                                    logger.info(
                                        "[CONFLICT-RESOLUTION] Reverse rejected by policy | signal=%s | reason=%s",
                                        winner.get('signal_id'),
                                        policy_result.get('reason'),
                                    )
                            else:
                                winner['intent'] = INTENT_REVERSE
                                winner['reverse_from_position_id'] = open_position['position_id']
                                logger.info(
                                    "[CONFLICT-RESOLUTION] Marking winning signal %s as reverse of position %s on %s",
                                    winner.get('signal_id'),
                                    open_position['position_id'],
                                    symbol,
                                )
                except Exception as reverse_error:
                    logger.warning("[CONFLICT-RESOLUTION] Failed to annotate reverse intent: %s", reverse_error)
            else:
                action = 'reject'
                reason = f"Lost conflict resolution to {winner['strategy_name']} ({resolution_strategy.value})"
            
            # Record conflict
            conflict_record = {
                'timestamp': datetime.now(timezone.utc),
                'new_signal': new_signal.get('signal_id', 'new'),
                'conflicting_signals': [s.get('signal_id', 'unknown') for s in conflicting_signals],
                'winner': winner['signal_id'],
                'strategy': resolution_strategy.value,
                'action': action
            }
            self.conflict_history.append(conflict_record)
            
            # Keep last 200 conflicts
            if len(self.conflict_history) > 200:
                self.conflict_history = self.conflict_history[-200:]
            
            logger.info(f"Conflict resolved: {action} new signal (winner: {winner['strategy_name']})")
            
            return {
                'action': action,
                'reason': reason,
                'winner': winner,
                'resolution_strategy': resolution_strategy.value,
                'conflict_record': conflict_record
            }
            
        except Exception as e:
            logger.error(f"Error resolving signal conflict: {e}")
            return {
                'action': 'reject',
                'reason': f"Conflict resolution error: {str(e)}",
                'winner': None
            }
    
    def _validate_signal_format(self, signal: Dict) -> Dict[str, Any]:
        """Validate signal has required fields."""
        required_fields = ['symbol', 'side']
        
        for field in required_fields:
            if field not in signal:
                return {
                    'valid': False,
                    'reason': f"Missing required field: {field}"
                }
        
        # Validate side
        if signal['side'] not in ['long', 'short', 'buy', 'sell']:
            return {
                'valid': False,
                'reason': f"Invalid side: {signal['side']}"
            }
        
        # Entry yoksa da geçerli say (enrich'te eklenecek)
        if 'entry' in signal and signal.get('entry', 0) <= 0:
            return {
                'valid': False,
                'reason': "Entry price must be positive"
            }
        
        return {'valid': True}
    
    async def _enrich_signal(self, strategy_name: str, signal: Dict) -> Dict:
        """Enrich signal with additional metadata."""
        enriched = signal.copy()

        # Entry yoksa, mevcut fiyatı al ve ekle
        if 'entry' not in enriched and 'symbol' in enriched:
            try:
                # Exchange client'ı bul
                for ex_name, client in self.portfolio_manager.exchange_clients.items():
                    try:
                        ticker = client.fetch_ticker(enriched['symbol'])
                        last_price = ticker.get('last', ticker.get('close', 0))
                        if last_price > 0:
                            enriched['entry'] = float(last_price)
                            logger.info(f"Added entry price {last_price} to signal for {enriched['symbol']}")
                            break
                    except:
                        continue
                
                # Hala entry yoksa varsayılan değer
                if 'entry' not in enriched:
                    logger.warning(f"Could not fetch entry price for {enriched.get('symbol')}, signal may be rejected")
                    enriched['entry'] = 0  # Risk manager reddedecek
                    
            except Exception as e:
                logger.error(f"Error fetching entry price: {e}")
        
        # Add strategy information
        enriched['strategy_name'] = strategy_name
        enriched['signal_timestamp'] = datetime.now(timezone.utc)
        
        # Add strategy allocation
        allocation = self.portfolio_manager.get_strategy_allocation(strategy_name)
        enriched['strategy_allocation'] = allocation if allocation is not None else 0.0
        
        # Add priority based on strategy performance
        enriched['priority'] = self._calculate_signal_priority(strategy_name, signal)
        
        # Add market regime if available
        if self.portfolio_manager.performance_monitor:
            summary = self.portfolio_manager.performance_monitor.get_strategy_summary(strategy_name)
            enriched['strategy_metrics'] = summary.get('metrics', {})
        
        # --- Volume Analysis Injection (Issue #450) ---
        if self.volume_analyzer and 'symbol' in enriched:
            symbol = enriched['symbol']
            try:
                # Get volume context
                trade_tf = enriched.get('timeframe', '5m')
                vol_context = await self.volume_analyzer.compute_context(symbol, trade_tf)
                
                if vol_context:
                    enriched['volume_bucket'] = vol_context.bucket
                    enriched['volume_strength'] = vol_context.volume_strength
                    enriched['volume_ratio_short'] = vol_context.ratio_short
                    enriched['volume_ratio_medium'] = vol_context.ratio_medium
                    enriched['volume_ratio_combined'] = vol_context.ratio_combined
                    enriched['volume_short_baseline'] = vol_context.short_baseline_volume
                    enriched['volume_medium_baseline'] = vol_context.medium_baseline_volume
                    enriched['volume_current_window'] = vol_context.current_window_volume
                    
                    # Log volume context injection
                    logger.info(
                        f"📊 [VOLUME-CONTEXT] {symbol} | Bucket: {vol_context.bucket} | "
                        f"Strength: {vol_context.volume_strength:.2f} | Source: analyzer"
                    )
                    
                    # Calculate volume score based on strategy config
                    strat_cfg = self.config.get('strategies', {}).get(strategy_name, {})
                    vol_filters = strat_cfg.get('volume_filters', {})
                    
                    if vol_filters.get('enabled', False) and vol_filters.get('use_volume_strength_in_score', False):
                        weight = float(vol_filters.get('volume_score_weight', 0.0))
                        # Normalize strength: 1.0 -> 0.5, 2.0 -> 1.0
                        strength = enriched['volume_strength']
                        raw_vol_score = min(1.0, strength / 2.0)
                        
                        # Apply weight
                        volume_score = raw_vol_score * weight
                        enriched['volume_score'] = volume_score
                        
                        # Adjust base score if present
                        base_score = enriched.get('score') or enriched.get('signal_score')
                        if base_score is not None:
                            enriched['score'] = float(base_score) + volume_score
                            logger.debug(f"   Volume Score Adjustment: {base_score:.3f} + {volume_score:.3f} -> {enriched['score']:.3f}")

                        # Also adjust quality_score if present (for RiskManager visibility)
                        base_quality = enriched.get('quality_score')
                        if base_quality is not None:
                            enriched['quality_score'] = float(base_quality) + volume_score
                    else:
                        enriched['volume_score'] = 0.0

            except Exception as e:
                logger.warning(f"Failed to inject volume context for {symbol}: {e}")
        
        return enriched
    
    def _calculate_signal_priority(self, strategy_name: str, signal: Dict) -> SignalPriority:
        """Calculate signal priority based on strategy performance and signal quality."""
        # Default priority
        priority = SignalPriority.MEDIUM
        
        # Check if performance monitor available
        if not self.portfolio_manager.performance_monitor:
            return priority
        
        summary = self.portfolio_manager.performance_monitor.get_strategy_summary(strategy_name)
        metrics = summary.get('metrics', {})
        
        if not metrics:
            return priority
        
        # Calculate priority based on metrics
        win_rate = metrics.get('win_rate', 0.5)
        sharpe = metrics.get('sharpe_ratio', 0)
        profit_factor = metrics.get('profit_factor', 1.0)
        
        # High priority: excellent performance
        if win_rate > 0.65 and sharpe > 1.5 and profit_factor > 2.0:
            priority = SignalPriority.HIGH
        # Low priority: poor performance
        elif win_rate < 0.40 or profit_factor < 1.0:
            priority = SignalPriority.LOW
        
        # Adjust for signal quality and regime weight
        if signal.get('confidence'):
            confidence = signal['confidence']
            if confidence > 0.8 and priority == SignalPriority.HIGH:
                priority = SignalPriority.CRITICAL
            elif confidence < 0.3:
                priority = SignalPriority.LOW
        
        # Further adjust based on regime_weight if available
        regime_weight = signal.get('regime_weight')
        if regime_weight is not None:
            regime_weight = float(regime_weight)
            if regime_weight < 0.5 and priority > SignalPriority.LOW:
                # Low regime confidence: downgrade priority
                priority = SignalPriority.LOW
            elif regime_weight > 0.8 and priority == SignalPriority.HIGH:
                # High regime confidence with high priority: upgrade to critical
                priority = SignalPriority.CRITICAL
        
        return priority
    
    async def _check_signal_conflicts(self, signal: Dict) -> Dict[str, Any]:
        """Check for conflicts with existing positions and signals."""
        conflicts = []
        conflicting_signals = []
        
        symbol = signal.get('symbol')
        side = signal.get('side')
        
        # Check active signals
        for signal_id, signal_data in self.active_signals.items():
            existing_signal = signal_data['signal']
            
            # Same symbol conflict
            if existing_signal.get('symbol') == symbol:
                # Opposite side conflict
                if self._are_sides_opposite(side, existing_signal.get('side')):
                    conflicts.append('opposite_direction')
                    conflicting_signals.append({
                        'signal_id': signal_id,
                        'signal': existing_signal,
                        'conflict_type': 'opposite_direction'
                    })
                # Same side - check if too close
                elif side == existing_signal.get('side'):
                    conflicts.append('same_direction')
                    conflicting_signals.append({
                        'signal_id': signal_id,
                        'signal': existing_signal,
                        'conflict_type': 'same_direction'
                    })
        
        # Check active positions (prefer PortfolioManager as source of truth)
        if self.portfolio_manager and hasattr(self.portfolio_manager, 'get_open_positions'):
            active_positions = self.portfolio_manager.get_open_positions() or {}
        else:
            active_positions = getattr(self.risk_manager, 'active_positions', {}) or {}

        for position_id, position in active_positions.items():
            if position.get('symbol') == symbol:
                position_side = position.get('side')
                if self._are_sides_opposite(side, position_side):
                    conflicts.append('opposite_to_position')
                    conflicting_signals.append({
                        'position_id': position_id,
                        'position': position,
                        'conflict_type': 'opposite_to_position'
                    })
        
        return {
            'has_conflict': len(conflicts) > 0,
            'conflicts': conflicts,
            'conflicting_signals': conflicting_signals
        }
    
    def _are_sides_opposite(self, side1: str, side2: str) -> bool:
        """Check if two sides are opposite."""
        long_sides = ['long', 'buy']
        short_sides = ['short', 'sell']
        
        return (side1 in long_sides and side2 in short_sides) or \
               (side1 in short_sides and side2 in long_sides)

    def _apply_regime_route_hint(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Attach regime-based routing metadata and urgency hints to a signal."""
        if not self.regime_routing_rules:
            return signal

        regime_name = (signal.get('regime_name') or '').lower()
        if not regime_name:
            return signal

        raw_rule = self.regime_routing_rules.get(regime_name) or self.regime_routing_rules.get('default')
        self.regime_route_stats['evaluated'] += 1

        if not raw_rule:
            self.regime_route_stats['unmatched'] += 1
            return signal

        route_hint = self._normalize_route_hint(raw_rule)
        if not route_hint:
            self.regime_route_stats['unmatched'] += 1
            return signal

        self.regime_route_stats['matched'] += 1
        self.regime_route_stats['by_regime'][regime_name] += 1
        signal['regime_route_hint'] = route_hint
        self._apply_strategy_urgency(signal, route_hint)
        return signal

    def _normalize_route_hint(self, raw_rule: Any) -> Optional[Dict[str, Any]]:
        """Normalize routing rule definitions to a dictionary structure."""
        if isinstance(raw_rule, dict):
            return deepcopy(raw_rule)
        if isinstance(raw_rule, str):
            return {'preferred_strategies': [raw_rule]}
        if isinstance(raw_rule, (list, tuple, set)):
            return {'preferred_strategies': list(raw_rule)}
        return None

    def _apply_strategy_urgency(self, signal: Dict[str, Any], route_hint: Dict[str, Any]) -> None:
        """Derive a strategy urgency score from routing metadata for queue prioritization."""
        strategy_name = signal.get('strategy_name')
        preferred = route_hint.get('preferred_strategies') or []
        if strategy_name and preferred:
            pref_priority = float(route_hint.get('preferred_priority', 0.8))
            fallback_priority = float(route_hint.get('fallback_priority', 0.4))
            if strategy_name in preferred:
                signal['strategy_urgency'] = max(signal.get('strategy_urgency', 0.5), pref_priority)
            else:
                signal['strategy_urgency'] = min(signal.get('strategy_urgency', fallback_priority), 1.0)

        boost = route_hint.get('queue_priority_boost')
        if boost is not None:
            try:
                boost_value = float(boost)
                signal['strategy_urgency'] = max(signal.get('strategy_urgency', 0.5), boost_value)
            except (TypeError, ValueError):
                pass

    def _record_rr_telemetry(self, signal: Dict[str, Any]) -> None:
        """Store rolling averages of actual vs. target R/R for analytics."""
        actual_rr = signal.get('rr_ratio')
        target_rr = signal.get('dynamic_rr_target')
        if actual_rr is None or target_rr is None:
            return

        self._increment_rr_stats(self.rr_telemetry, actual_rr, target_rr)

        strategy_bucket = self.rr_telemetry['by_strategy'].setdefault(
            signal.get('strategy_name', 'unknown'),
            {'samples': 0, 'avg_actual_rr': 0.0, 'avg_target_rr': 0.0}
        )
        self._increment_rr_stats(strategy_bucket, actual_rr, target_rr)

    @staticmethod
    def _increment_rr_stats(bucket: Dict[str, Any], actual_rr: float, target_rr: float) -> None:
        """Update running averages for a telemetry bucket."""
        samples = bucket.get('samples', 0) + 1
        bucket['samples'] = samples

        current_actual = bucket.get('avg_actual_rr', 0.0)
        bucket['avg_actual_rr'] = current_actual + (actual_rr - current_actual) / samples

        current_target = bucket.get('avg_target_rr', 0.0)
        bucket['avg_target_rr'] = current_target + (target_rr - current_target) / samples
    
    async def _enrich_signal_for_dynamic_rr(self, signal: Dict) -> Dict:
        """
        Enrich signal with intelligence metrics for dynamic R/R calculation.
        
        Adds the following metrics:
        - ML confidence and regime prediction
        - RL agreement and action probability
        - Market volume and momentum strength
        
        Args:
            signal: Trading signal to enrich
            
        Returns:
            Enriched signal with intelligence metrics
        """
        symbol = signal.get('symbol', 'UNKNOWN')
        
        # 1. ML Metrics
        try:
            if hasattr(self, 'ml_integration') and self.ml_integration:
                ml_context = await self.ml_integration.get_ml_context(symbol)
                if ml_context:
                    # Use actual ML values
                    signal['ml_confidence'] = float(ml_context.get('consensus_score', 0.5))
                    signal['regime_name'] = str(ml_context.get('regime', 'neutral'))
                    signal['regime_confidence'] = float(ml_context.get('regime_confidence', 0.3))
                    quality_score = float(ml_context.get('quality_score', 0.0) or 0.0)
                    if quality_score > 1:
                        quality_score /= 100.0
                    ml_quality = max(0.0, min(1.0, quality_score))
                    signal['ml_quality_score'] = ml_quality
                    if 'quality_score' not in signal:
                        signal['quality_score'] = ml_quality
                    logger.debug(
                        "✅ ML metrics added: conf=%.2f | quality=%.2f",
                        signal['ml_confidence'],
                        signal.get('quality_score', 0.0),
                    )
                else:
                    # No ML context, use explicit fallbacks
                    signal['ml_confidence'] = 0.5
                    signal['regime_name'] = 'neutral'
                    signal['regime_confidence'] = 0.3
                    signal['ml_quality_score'] = 0.0
                    if 'quality_score' not in signal:
                        signal['quality_score'] = 0.0
                    logger.debug("⚠️ Using ML fallback values")
            else:
                signal['ml_confidence'] = 0.5
                signal['regime_name'] = 'neutral'
                signal['regime_confidence'] = 0.3
                signal['ml_quality_score'] = signal.get('ml_quality_score', 0.0)
                if 'quality_score' not in signal:
                    signal['quality_score'] = signal.get('ml_quality_score', 0.0)
        except Exception as e:
            logger.debug(f"ML enrichment error: {e}")
            signal.update({'ml_confidence': 0.5, 'regime_name': 'neutral', 'regime_confidence': 0.3})
            signal['ml_quality_score'] = signal.get('ml_quality_score', 0.0)
            if 'quality_score' not in signal:
                signal['quality_score'] = 0.0
        
        # 2. RL/PPO Metrics
        try:
            side = (signal.get('side') or '').lower()
            if side in ('buy', 'long') and 'ppo_long_score' in signal:
                score = float(signal.get('ppo_long_score', self.ppo_fallback_score))
                signal['rl_is_agree'] = score >= 0.5
                signal['rl_action_prob'] = score
                logger.debug(f"✅ PPO metrics: long_score={score:.2f}")
            elif self.legacy_rl_enabled and hasattr(self, '_last_rl_decision') and self._last_rl_decision:
                rl_action = self._last_rl_decision.get('action', 'hold')
                rl_confidence = float(self._last_rl_decision.get('confidence', 0.5))
                signal_side = signal.get('side', '').lower()
                rl_action_normalized = rl_action.lower().replace('long', 'buy').replace('short', 'sell')
                signal_side_normalized = signal_side.replace('long', 'buy').replace('short', 'sell')
                signal['rl_is_agree'] = (rl_action_normalized == signal_side_normalized)
                signal['rl_action_prob'] = rl_confidence
                logger.debug(f"✅ RL metrics: agree={signal['rl_is_agree']}, prob={rl_confidence:.2f}")
            else:
                signal['rl_is_agree'] = False
                signal['rl_action_prob'] = 0.5
                logger.debug("⚠️ Using RL fallback values")
        except Exception as e:
            logger.debug(f"RL enrichment error: {e}")
            signal.update({'rl_is_agree': False, 'rl_action_prob': 0.5})
        
        # 3. Market Metrics (Volume & Momentum)
        volume_strength = 0.5
        volume_bucket = "NORMAL"
        volume_ctx_source = "unknown"
        momentum_strength = 0.5
        trade_tf = signal.get('timeframe') or signal.get('tf') or '5m'
        as_of_ts = signal.get('timestamp') or signal.get('ts')

        try:
            now_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            run_id = get_current_run_id()
            computed_by_analyzer = False
            volume_ctx_log = None
            if self.volume_analyzer and self._volume_analyzer_enabled:
                # Reuse existing volume context if present to avoid double computation
                if all(
                    key in signal
                    for key in (
                        "volume_strength",
                        "volume_bucket",
                        "volume_ratio_short",
                        "volume_ratio_medium",
                        "volume_ratio_combined",
                    )
                ):
                    volume_strength = float(signal.get("volume_strength", volume_strength))
                    volume_bucket = signal.get("volume_bucket", volume_bucket)
                    volume_ctx_source = "cached"
                    computed_by_analyzer = True
                    volume_ctx_log = None  # already logged in first enrichment
                else:
                    shock_state = None
                    try:
                        meta = signal.get("meta") if isinstance(signal, dict) else None
                        if isinstance(meta, dict):
                            shock_state = meta.get("shock_state") or meta.get("shock_status")
                    except Exception:
                        shock_state = None

                    include_forming_trade = False
                    try:
                        vol_cfg = self.config.get("volume_analyzer", {}) if isinstance(self.config, dict) else {}
                        include_forming_trade = bool(vol_cfg.get("include_forming_numerator_when_armed", False))
                    except Exception:
                        include_forming_trade = False

                    ctx = await self.volume_analyzer.compute_context(
                        symbol=symbol,
                        trade_timeframe=trade_tf,
                        as_of_ts=as_of_ts,
                        shock_state=shock_state,
                        include_forming_trade=include_forming_trade,
                    )
                    if ctx:
                        volume_strength = float(ctx.volume_strength)
                        volume_bucket = ctx.bucket
                        computed_by_analyzer = True
                        volume_ctx_source = "analyzer"
                        volume_ctx_log = {
                            "event": "volume_context",
                            "timestamp": now_ts,
                            "run_id": run_id,
                            "symbol": symbol,
                            "timeframe": trade_tf,
                            "volume_bucket": ctx.bucket,
                            "volume_strength": ctx.volume_strength,
                            "ratio_short": ctx.ratio_short,
                            "ratio_medium": ctx.ratio_medium,
                            "ratio_combined": ctx.ratio_combined,
                            "current_window_volume": ctx.current_window_volume,
                            "short_baseline_volume": ctx.short_baseline_volume,
                            "medium_baseline_volume": ctx.medium_baseline_volume,
                            "baseline_short_last_bar_ts": getattr(ctx, "baseline_short_last_bar_ts", None),
                            "baseline_medium_last_bar_ts": getattr(ctx, "baseline_medium_last_bar_ts", None),
                            "baseline_calc_mode": getattr(ctx, "baseline_calc_mode", None),
                            "source": "analyzer",
                        }

            data = None
            if self.market_data_pipeline:
                data = await self.market_data_pipeline.get_latest_ohlcv(symbol, trade_tf)

            if data is not None and len(data) >= 20:
                # Momentum: price change normalized
                price_change_pct = data['close'].pct_change(10).iloc[-1]
                momentum_strength = max(0, min(1, (price_change_pct + MOMENTUM_PRICE_CHANGE_OFFSET) / MOMENTUM_PRICE_CHANGE_RANGE))

                # Fallback volume strength if analyzer unavailable or returned no context
                if not computed_by_analyzer:
                    volume_ctx_source = "fallback"
                    recent_vol = data['volume'].tail(5).mean()
                    avg_vol = data['volume'].tail(20).mean()
                    volume_strength = min(recent_vol / avg_vol, VOLUME_NORMALIZATION_MAX) / VOLUME_NORMALIZATION_DIVISOR if avg_vol > 0 else volume_strength
            if volume_ctx_log:
                logger.info("volume_context %s", json.dumps(volume_ctx_log))

            logger.debug(
                "✅ Market metrics: vol=%.2f bucket=%s, mom=%.2f (analyzer=%s)",
                volume_strength,
                volume_bucket,
                momentum_strength,
                computed_by_analyzer,
            )
        except Exception as e:
            logger.debug(f"Market metrics error: {e}")

        signal['volume_strength'] = volume_strength
        signal['volume_bucket'] = volume_bucket
        signal['volume_ctx_source'] = volume_ctx_source
        signal['momentum_strength'] = momentum_strength

        # 4. PPO Multipliers for downstream risk modules
        side = (signal.get('side') or '').lower()
        ppo_rr_multiplier = 1.0
        if side in ('buy', 'long') and 'ppo_long_score' in signal:
            score = float(signal['ppo_long_score'])
            ppo_rr_multiplier = (
                self.ppo_multipliers['rr_up_mult'] if score < 0.5 else self.ppo_multipliers['rr_down_mult']
            )
        signal['ppo_rr_multiplier'] = ppo_rr_multiplier

        signal = self._apply_regime_route_hint(signal)

        logger.info(
            f"📊 [Signal Enriched] {symbol}: ML={signal.get('ml_confidence', 0):.2f}, "
            f"RL_agree={signal.get('rl_is_agree', False)}, Regime={signal.get('regime_name', 'N/A')} "
            f"({signal.get('regime_confidence', 0):.2f}), Vol={signal.get('volume_strength', 0):.2f} "
            f"[{signal.get('volume_bucket', 'N/A')}], "
            f"Mom={signal.get('momentum_strength', 0):.2f}, PPO_RR={ppo_rr_multiplier:.2f}"
        )
        
        return signal
    
    async def _assess_signal_risk(self, signal: Dict, strategy_name: str) -> Dict[str, Any]:
        """
        UPDATED: Use AdvancedPositionSizing instead of risk_manager for sizing.
        Also enriches signals with ML/RL intelligence for dynamic R/R calculation.
        """
        try:
            # Initialize position sizing if not already done
            if not hasattr(self, 'position_sizing'):
                from core.position_sizing import AdvancedPositionSizing
                self.position_sizing: PositionSizingProtocol = AdvancedPositionSizing(self.risk_manager)
                logger.info("✅ AdvancedPositionSizing initialized in StrategyCoordinator")
            
            # CRITICAL: Enrich signal BEFORE risk validation
            signal = await self._enrich_signal_for_dynamic_rr(signal)
            position_multiplier = self._compute_ppo_position_multiplier(signal)
            signal['ppo_position_multiplier'] = position_multiplier

            # Leverage SSOT: default from centralized config unless strategy/signal explicitly sets it.
            if not signal.get('leverage'):
                try:
                    trading_cfg = self.config.get('trading', {}) if isinstance(self.config, dict) else {}
                    if isinstance(trading_cfg, dict) and trading_cfg.get('leverage') is not None:
                        signal['leverage'] = float(trading_cfg.get('leverage') or 1.0)
                except Exception:
                    signal['leverage'] = signal.get('leverage') or 1.0

            # Apply volume bucket filters/boosts per-strategy
            # NOTE: Volume gating and scoring are now handled in process_strategy_signal and _enrich_signal (Issue #450).
            # The legacy logic here has been removed to prevent duplication and errors.
            
            # 3. Calculate R/R ratio if not already present
            if 'rr_ratio' not in signal and signal.get('entry') and signal.get('stop') and signal.get('target'):
                entry = float(signal['entry'])
                stop = float(signal['stop'])
                target = float(signal['target'])
                risk = abs(entry - stop)
                reward = abs(target - entry)
                signal['rr_ratio'] = reward / risk if risk > 0 else 0
                logger.debug(f"📊 Signal R/R: {signal['rr_ratio']:.2f}")
            
            # Calculate position size using the new module
            sized_signal = await self.position_sizing.calculate_optimal_size(
                signal,
                return_signal=True,
                portfolio_manager=getattr(self, "portfolio_manager", None),
            )

            if position_multiplier != 1.0:
                sized_signal.setdefault('sizing_meta', {})
                sized_signal['sizing_meta']['ppo_position_multiplier'] = position_multiplier
                if 'amount' in sized_signal:
                    sized_signal['amount'] *= position_multiplier
                if 'notional' in sized_signal:
                    sized_signal['notional'] *= position_multiplier
            
            # Check if sizing was successful
            if sized_signal.get('amount', 0) <= 0:
                return {
                    'acceptable': False,
                    'reason': 'Unable to calculate valid position size',
                    'metrics': sized_signal.get('sizing_meta', {})
                }

            logger.info(
                "[QUALITY-BEFORE-RISK] strat=%s | sym=%s | intent=%s | extreme_bypass=%s | quality=%.3f",
                strategy_name,
                sized_signal.get('symbol'),
                sized_signal.get('intent'),
                sized_signal.get('extreme_bypass', False),
                float(sized_signal.get('quality_score', 0.0) or 0.0),
            )

            # Validate AND enforce planner limits via RiskManager
            allowed, final_size, meta = await self.risk_manager.size_and_validate_position(
                sized_signal,
                portfolio_manager=self.portfolio_manager,
            )

            planner_value = meta.get('planner')
            if isinstance(planner_value, dict):
                planner_dict = planner_value
            elif planner_value:
                try:
                    planner_dict = asdict(planner_value)
                except TypeError:
                    planner_dict = None
            else:
                planner_dict = None
            risk_metrics = meta.get('risk_metrics', {}) or {}
            if planner_dict:
                risk_metrics['planner'] = planner_dict
            risk_metrics['planner_raw_notional'] = meta.get('planner_raw_notional')
            risk_metrics['planner_delta_abs'] = meta.get('planner_delta_abs')
            risk_metrics['planner_delta_ratio'] = meta.get('planner_delta_ratio')
            risk_metrics['planner_reason'] = meta.get('planner_reason')
            risk_metrics['sizing_meta'] = sized_signal.get('sizing_meta', {})
            risk_metrics['sizing_meta']['ppo_position_multiplier'] = position_multiplier

            # Final execution size fields (canonical when planner is active)
            risk_metrics['final_position_size'] = final_size
            risk_metrics['final_notional'] = sized_signal.get('notional')

            planner_mode = 'active' if self.risk_manager._is_size_planner_enabled() else 'shadow'
            logger.info(
                "[RISK-PLANNER] strategy_path",
                extra={
                    'symbol': sized_signal.get('symbol'),
                    'mode': planner_mode,
                    'raw_notional': meta.get('planner_raw_notional'),
                    'final_notional': sized_signal.get('notional'),
                    'planner_reason': meta.get('planner_reason'),
                },
            )

            self._record_rr_telemetry(sized_signal)

            if not allowed:
                reason = (
                    meta.get('validation_reason')
                    or meta.get('planner_reason')
                    or meta.get('blocked_by')
                    or 'Risk validation failed'
                )
                reason_code = meta.get('reason_code') or (risk_metrics.get('reason_code') if isinstance(risk_metrics, dict) else None)
                return {
                    'acceptable': False,
                    'reason': reason,
                    'reason_code': reason_code,
                    'metrics': risk_metrics
                }
            
            return {
                'acceptable': True,
                'position_size': final_size,
                'notional': sized_signal.get('notional'),
                'metrics': risk_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}", exc_info=True)
            return {
                'acceptable': False,
                'reason': f"Risk assessment error: {str(e)}",
                'metrics': {}
            }
    
    def _route_signal(self, signal: Dict, risk_assessment: Dict) -> Dict[str, Any]:
        """Route signal based on priority and risk assessment."""
        priority = signal.get('priority', SignalPriority.MEDIUM)
        
        # Determine execution priority
        if priority == SignalPriority.CRITICAL:
            execution_priority = 'immediate'
        elif priority == SignalPriority.HIGH:
            execution_priority = 'high'
        else:
            execution_priority = 'normal'
        
        # Determine execution method
        position_size = risk_assessment.get('position_size', 0)
        if position_size > 0:
            execution_method = 'limit_order'  # Default to limit orders
        else:
            execution_method = 'skip'
        
        return {
            'execution_priority': execution_priority,
            'execution_method': execution_method,
            'position_size': position_size
        }
    
    def _generate_signal_id(self, strategy_name: str, signal: Dict) -> str:
        """Generate unique signal identifier."""
        return uuid.uuid4().hex
    
    def _resolve_by_priority(self, signals: List[Dict]) -> Dict:
        """Resolve conflict by selecting highest priority signal."""
        # Add signal_id if not present
        for i, signal in enumerate(signals):
            if 'signal_id' not in signal:
                signal['signal_id'] = f"signal_{i}"
        
        # Sort by priority (highest first)
        priority_map = {
            SignalPriority.CRITICAL: 4,
            SignalPriority.HIGH: 3,
            SignalPriority.MEDIUM: 2,
            SignalPriority.LOW: 1
        }
        
        sorted_signals = sorted(
            signals,
            key=lambda s: priority_map.get(s.get('priority', SignalPriority.MEDIUM), 2),
            reverse=True
        )
        
        winner = sorted_signals[0]
        return {
            'signal_id': winner.get('signal_id'),
            'strategy_name': winner.get('strategy_name', 'unknown'),
            'priority': winner.get('priority', SignalPriority.MEDIUM)
        }
    
    def _resolve_by_risk_reward(self, signals: List[Dict]) -> Dict:
        """Resolve conflict by selecting best risk/reward ratio."""
        best_signal = None
        best_rr = 0
        
        for signal in signals:
            entry = signal.get('entry', 0)
            stop = signal.get('stop', 0)
            target = signal.get('target', entry * 1.02)
            
            if entry > 0 and stop > 0:
                risk = abs(entry - stop)
                reward = abs(target - entry)
                rr = reward / risk if risk > 0 else 0
                
                if rr > best_rr:
                    best_rr = rr
                    best_signal = signal
        
        if best_signal:
            if 'signal_id' not in best_signal:
                best_signal['signal_id'] = 'best_rr'
            return {
                'signal_id': best_signal.get('signal_id'),
                'strategy_name': best_signal.get('strategy_name', 'unknown'),
                'risk_reward': best_rr
            }
        
        # Fallback to first signal
        if signals:
            if 'signal_id' not in signals[0]:
                signals[0]['signal_id'] = 'fallback'
            return {
                'signal_id': signals[0].get('signal_id'),
                'strategy_name': signals[0].get('strategy_name', 'unknown'),
                'risk_reward': 0
            }
        
        return {'signal_id': 'none', 'strategy_name': 'none', 'risk_reward': 0}
    
    def _resolve_by_performance(self, signals: List[Dict]) -> Dict:
        """Resolve conflict by strategy performance."""
        if not self.portfolio_manager.performance_monitor:
            return self._resolve_by_priority(signals)
        
        best_signal = None
        best_score = -1
        
        for signal in signals:
            strategy_name = signal.get('strategy_name')
            if not strategy_name:
                continue
            
            summary = self.portfolio_manager.performance_monitor.get_strategy_summary(strategy_name)
            metrics = summary.get('metrics', {})
            
            # Calculate performance score
            win_rate = metrics.get('win_rate', 0.5)
            sharpe = max(metrics.get('sharpe_ratio', 0), 0)
            profit_factor = metrics.get('profit_factor', 1.0)
            
            score = (win_rate * 0.4) + (min(sharpe / 2.0, 0.3) * 0.3) + (min(profit_factor / 3.0, 0.3) * 0.3)
            
            if score > best_score:
                best_score = score
                best_signal = signal
        
        if best_signal:
            if 'signal_id' not in best_signal:
                best_signal['signal_id'] = 'best_performance'
            return {
                'signal_id': best_signal.get('signal_id'),
                'strategy_name': best_signal.get('strategy_name', 'unknown'),
                'performance_score': best_score
            }
        
        return self._resolve_by_priority(signals)
    
    def _resolve_by_fifo(self, signals: List[Dict]) -> Dict:
        """Resolve conflict by first-in-first-out."""
        # Existing signals (from conflicting_signals) come first
        for signal in signals[1:]:  # Skip new signal
            if 'signal_id' in signal:
                return {
                    'signal_id': signal.get('signal_id'),
                    'strategy_name': signal.get('strategy_name', 'unknown'),
                    'reason': 'existing_signal'
                }
        
        # New signal wins if no existing signals
        if signals:
            if 'signal_id' not in signals[0]:
                signals[0]['signal_id'] = 'new'
            return {
                'signal_id': signals[0].get('signal_id'),
                'strategy_name': signals[0].get('strategy_name', 'unknown'),
                'reason': 'new_signal'
            }
        
        return {'signal_id': 'none', 'strategy_name': 'none', 'reason': 'no_signals'}
    
    async def get_next_signal(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Get next signal from queue."""
        try:
            signal = await self.signal_queue.get(timeout=timeout)
            return signal
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error getting next signal: {e}")
            return None

    def _has_execution_capacity(self, signal: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> bool:
        if not self.risk_manager:
            return True

        risk_assessment = None
        cached_metrics = None
        if isinstance(payload, dict):
            risk_assessment = payload.get('risk_assessment') or {}
            cached_metrics = risk_assessment.get('metrics') if isinstance(risk_assessment, dict) else None

        gating_fn = getattr(self.risk_manager, 'can_open_new_position', None)

        try:
            if callable(gating_fn):
                ok, reason, updated_metrics = gating_fn(signal, self.portfolio_manager, cached_metrics)
                if isinstance(risk_assessment, dict) and isinstance(updated_metrics, dict):
                    risk_assessment['metrics'] = updated_metrics
            elif hasattr(self.risk_manager, 'has_execution_capacity'):
                ok, reason = self.risk_manager.has_execution_capacity(signal, self.portfolio_manager)
            else:
                return True
        except Exception as exc:
            logger.error(f"Error during dispatch gating: {exc}")
            return False

        if not ok:
            logger.debug(f"⏸️ Dispatch paused for {signal.get('symbol')}: {reason}")
        return ok

    async def try_dispatch_next(self, timeout: Optional[float] = 1.0) -> Optional[Dict[str, Any]]:
        """Get next signal only when concurrent limits have spare capacity."""
        loop = asyncio.get_running_loop()
        deadline = None
        if timeout is not None:
            deadline = loop.time() + timeout

        async with self._dispatch_lock:
            while True:
                remaining = None
                if deadline is not None:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        return None

                try:
                    payload = await self.signal_queue.get(timeout=remaining)
                except asyncio.TimeoutError:
                    return None

                signal = payload.get('signal') or {}
                if self._has_execution_capacity(signal, payload):
                    return payload

                await self.signal_queue.requeue(payload)

                if deadline is not None and loop.time() >= deadline:
                    return None

                sleep_for = self._dispatch_retry_delay
                if deadline is not None:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        return None
                    sleep_for = min(sleep_for, remaining)

                await asyncio.sleep(sleep_for)
    
    def mark_signal_executed(self, signal_id: str, execution_result: Dict):
        """Mark signal as executed and remove from active signals."""
        if not signal_id:
            logger.warning("Attempted to mark execution with empty signal_id")
            return

        if signal_id not in self.active_signals:
            logger.debug(f"Signal {signal_id} not found in active signals during execution mark")
            return

        signal_entry = self.active_signals[signal_id]
        signal_entry['status'] = 'executed'
        signal_entry['execution_result'] = execution_result
        signal_entry['execution_time'] = datetime.now(timezone.utc)

        # Clear stop-loss reversal-only guard once an entry is actually executed.
        try:
            sig = signal_entry.get("signal") or {}
            strat = sig.get("strategy_name") or sig.get("strategy")
            sym = sig.get("symbol")
            s_norm = self._normalize_side(sig.get("side"))
            if strat and sym and s_norm:
                sl_key = f"{strat}:{sym}:{s_norm}"
                with self._strategy_cooldowns_lock:
                    self._stop_loss_reversal_required.pop(sl_key, None)
        except Exception:
            pass

        # Update history entry if present to reflect execution
        history_entry = self._signal_history_lookup.get(signal_id)

        if history_entry:
            history_entry.update({
                'status': 'executed',
                'execution_time': signal_entry['execution_time'],
                'execution_result': execution_result
            })
        else:
            self._add_signal_history_entry({
                'signal_id': signal_id,
                'strategy_name': signal_entry['signal'].get('strategy') or signal_entry['signal'].get('strategy_name'),
                'symbol': signal_entry['signal'].get('symbol'),
                'timestamp': signal_entry.get('timestamp', datetime.now(timezone.utc)),
                'status': 'executed',
                'execution_time': signal_entry['execution_time'],
                'execution_result': execution_result
            })

        # Remove from active signals to prevent conflicts and unbounded growth
        self.active_signals.pop(signal_id, None)
        dedupe_key = self._active_dedupe_by_signal_id.pop(signal_id, None)
        if dedupe_key:
            self._active_dedupe_by_key.pop(str(dedupe_key), None)

        logger.info(f"Signal {signal_id} marked as executed and removed from active registry")

    def _on_signal_queue_expire(self, payload: Any) -> None:
        """Queue lifecycle hook to keep active_signals aligned with TTL purges."""
        if not payload or not isinstance(payload, dict):
            return
        signal_id = payload.get("signal_id")
        if not signal_id:
            return
        self.discard_active_signal(str(signal_id))

    def discard_active_signal(self, signal_id: str) -> None:
        """Remove a signal from the active registry without raising errors."""
        if not signal_id:
            return

        removed = self.active_signals.pop(signal_id, None)

        dedupe_key = self._active_dedupe_by_signal_id.pop(signal_id, None)
        if not dedupe_key and isinstance(removed, dict):
            signal = removed.get("signal") if isinstance(removed.get("signal"), dict) else {}
            dedupe_key = signal.get("dedupe_key")
        if dedupe_key:
            self._active_dedupe_by_key.pop(str(dedupe_key), None)

        if removed:
            logger.warning(
                "Signal %s discarded from active registry",
                signal_id
            )
        else:
            logger.debug(
                "Attempted to discard signal %s from active registry but it was not present",
                signal_id
            )

    def _add_signal_history_entry(self, entry: Dict[str, Any]) -> None:
        """Add or replace a signal history entry while maintaining lookup cache."""
        signal_id = entry.get('signal_id')
        if not signal_id:
            logger.debug("Attempted to add history entry without signal_id; skipping")
            return

        existing_entry = self._signal_history_lookup.get(signal_id)
        if existing_entry:
            existing_entry.update(entry)
        else:
            self.signal_history.append(entry)
            self._signal_history_lookup[signal_id] = entry

        if len(self.signal_history) > 500:
            # Retain only the most recent 500 entries and rebuild lookup map accordingly
            self.signal_history = self.signal_history[-500:]
            self._signal_history_lookup = {
                item.get('signal_id'): item
                for item in self.signal_history
                if item.get('signal_id')
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get signal processing statistics."""
        queue_stats = self.signal_queue.get_stats() if hasattr(self.signal_queue, 'get_stats') else {}
        return {
            'stats': self.processing_stats.copy(),
            'active_signals': len(self.active_signals),
            'queued_signals': self.signal_queue.qsize(),
            'signal_history_count': len(self.signal_history),
            'conflict_history_count': len(self.conflict_history),
            'queue_stats': queue_stats
        }
    
    def get_active_signals_summary(self) -> List[Dict]:
        """Get summary of active signals."""
        return [
            {
                'signal_id': signal_id,
                'strategy': data['signal'].get('strategy_name'),
                'symbol': data['signal'].get('symbol'),
                'side': data['signal'].get('side'),
                'priority': data['signal'].get('priority', SignalPriority.MEDIUM).name,
                'timestamp': data['timestamp'],
                'status': data['status']
            }
            for signal_id, data in self.active_signals.items()
        ]
