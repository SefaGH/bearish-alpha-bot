"""
Strategy Coordination Engine.
Coordinates signals and positions across multiple strategies.
"""

import asyncio
import heapq
import itertools
import logging
import time
import json
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
from copy import deepcopy

from src.quality.quality_calculator import compute_quality
from core.volume_analyzer import VolumeAnalyzer
from src.core.interfaces import PositionSizingProtocol
from src.utils.volume_utils import get_bucket_rank
from core.logger import get_current_run_id
from src.core.signal_intents import (
    INTENT_ENTRY,
    INTENT_REENTRY,
    INTENT_SCALE_IN,
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
        self._heap: List[Tuple[float, float, int, Dict[str, Any]]] = []
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

    async def put(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        symbol = self._extract_symbol(payload)
        intent = self._extract_intent(payload)
        async with self._condition:
            now = time.time()
            self._purge_expired_locked()
            if symbol:
                pending_totals = self._pending_by_symbol[symbol]
                pending_total = pending_totals["total"]
                pending_scale = pending_totals["scale_in"]

                if not self._pyramiding_enabled:
                    if self._max_pending_per_symbol and pending_total >= self._max_pending_per_symbol:
                        self._stats['rejected_symbol_limit'] += 1
                        reason = f"Queue limit reached for {symbol}"
                        self._logger.warning(f"🚫 [QUEUE] {reason}")
                        return False, reason
                else:
                    if intent == INTENT_SCALE_IN:
                        max_allowed = self._max_pending_per_symbol + self._max_pending_scale_in_per_symbol
                        if self._max_pending_scale_in_per_symbol <= 0:
                            max_allowed = self._max_pending_per_symbol
                        if pending_total >= max_allowed or pending_scale >= self._max_pending_scale_in_per_symbol > 0:
                            self._stats['rejected_symbol_limit'] += 1
                            reason = "Scale-in queue limit reached"
                            self._logger.info(
                                "[PYRAMID-QUEUE] scale-in rejected at enqueue | sym=%s | pending_total=%d | pending_scale_in=%d | max_entry=%d | max_scale_in=%d",
                                symbol,
                                pending_total,
                                pending_scale,
                                self._max_pending_per_symbol,
                                self._max_pending_scale_in_per_symbol,
                            )
                            return False, reason
                    else:
                        if self._max_pending_per_symbol and pending_total >= self._max_pending_per_symbol:
                            self._stats['rejected_symbol_limit'] += 1
                            reason = f"Queue limit reached for {symbol}"
                            self._logger.warning(f"🚫 [QUEUE] {reason}")
                            return False, reason

            now = time.time()
            meta = payload.setdefault('queue_meta', {})
            meta['enqueued_at'] = now
            meta['expiration'] = now + self._ttl

            priority_score = self._compute_priority(payload, now)
            entry = (-priority_score, meta['enqueued_at'], next(self._sequence), payload)

            if len(self._heap) >= self._max_depth:
                replaced = self._maybe_replace_lowest(entry, priority_score)
                if not replaced:
                    self._stats['rejected_capacity'] += 1
                    reason = "Signal queue at capacity"
                    self._logger.warning(f"🚫 [QUEUE] {reason} (score={priority_score:.3f})")
                    return False, reason
            else:
                heapq.heappush(self._heap, entry)

            if symbol:
                self._pending_by_symbol[symbol]["total"] += 1
                if intent == INTENT_SCALE_IN:
                    self._pending_by_symbol[symbol]["scale_in"] += 1
            self._stats['accepted'] += 1
            self._condition.notify()
            self._logger.info(
                f"📥 [QUEUE] Signal enqueued: symbol={symbol}, score={priority_score:.3f}, depth={len(self._heap)}"
            )
            return True, None

    async def get(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        async with self._condition:
            deadline = None
            loop = None
            if timeout is not None:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + timeout

            while True:
                self._purge_expired_locked()
                self._refresh_priorities_locked()

                if self._heap:
                    entry = heapq.heappop(self._heap)
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
                    return payload

                if timeout is None:
                    await self._condition.wait()
                else:
                    loop = loop or asyncio.get_running_loop()
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)

    def qsize(self) -> int:
        return len(self._heap)

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    async def requeue(self, payload: Dict[str, Any]) -> None:
        symbol = self._extract_symbol(payload)
        async with self._condition:
            now = time.time()
            self._purge_expired_locked()
            meta = payload.setdefault('queue_meta', {})
            meta.setdefault('enqueued_at', now)
            meta.setdefault('expiration', now + self._ttl)

            priority_score = self._compute_priority(payload, now)
            entry = (-priority_score, meta['enqueued_at'], next(self._sequence), payload)
            heapq.heappush(self._heap, entry)

            if symbol:
                self._pending_by_symbol[symbol]["total"] += 1
                if self._extract_intent(payload) == INTENT_SCALE_IN:
                    self._pending_by_symbol[symbol]["scale_in"] += 1

            self._stats['requeued'] = self._stats.get('requeued', 0) + 1
            self._condition.notify()

    def _maybe_replace_lowest(self, entry, new_score: float) -> bool:
        if not self._heap:
            heapq.heappush(self._heap, entry)
            return True

        lowest_index = max(range(len(self._heap)), key=lambda idx: self._heap[idx][0])
        lowest_entry = self._heap[lowest_index]
        lowest_score = -lowest_entry[0]
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
        self._heap[lowest_index] = entry
        heapq.heapify(self._heap)
        self._logger.info(
            f"♻️ [QUEUE] Replaced low-priority signal (score={lowest_score:.3f}) with higher score {new_score:.3f}"
        )
        return True

    def _purge_expired_locked(self) -> None:
        if not self._heap:
            return

        now = time.time()
        kept: List[Tuple[float, float, int, Dict[str, Any]]] = []
        expired = 0
        while self._heap:
            entry = heapq.heappop(self._heap)
            payload = entry[3]
            expiration = payload.get('queue_meta', {}).get('expiration', now + 1)
            if expiration < now:
                expired += 1
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
            heapq.heappush(self._heap, entry)

        if expired:
            self._stats['expired'] += expired
            self._logger.warning(f"⏳ [QUEUE] Dropped {expired} expired signals")

    def _refresh_priorities_locked(self) -> None:
        if not self._heap:
            return

        now = time.time()
        refreshed = [
            (-self._compute_priority(entry[3], now), entry[1], entry[2], entry[3])
            for entry in self._heap
        ]
        heapq.heapify(refreshed)
        self._heap = refreshed

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
        self._initial_equity = self._derive_initial_equity()

        va_cfg = self.config.get('volume_analyzer') if isinstance(self.config, dict) else {}
        self._volume_analyzer_enabled = bool(va_cfg.get('enabled', True) if isinstance(va_cfg, dict) else True)
        # Volume analyzer (async) for dynamic volume context
        self.volume_analyzer = kwargs.get('volume_analyzer') if self._volume_analyzer_enabled else None
        if not self.volume_analyzer and self.market_data_pipeline and self._volume_analyzer_enabled:
            self.volume_analyzer = VolumeAnalyzer(self.market_data_pipeline, va_cfg or {})

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
    
        logger.info("StrategyCoordinator initialized (market_data_pipeline=%s)", bool(self.market_data_pipeline))
    
    def _initialize_gemma(self):
        """Initialize GEMMA adapter with manifest configuration."""
        try:
            # Import inside function to avoid circular dependency
            from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
            from src.ml.manifest_manager import ManifestManager
            
            gemma_config = self.config['ml']['gemma'].copy()
            
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

        if not pyramiding_enabled:
            if candidate_exists:
                logger.info(
                    "Pyramiding shadow: signal would be classified as scale_in if enabled | sym=%s | strat=%s | side=%s",
                    symbol,
                    strategy_name,
                    side,
                )
            return INTENT_ENTRY

        if candidate_exists:
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
            if symbol in self.signal_price_history and entry_price > 0 and price_delta_bypass_enabled:
                # Find last price for this symbol
                if self.signal_price_history[symbol]:
                    last_timestamp, last_price = self.signal_price_history[symbol][-1]
                    
                    # Step 3b: Calculate price_delta (in decimal, e.g., 0.0005 = 0.05%)
                    # TODO (pyramiding/DCA): consider signed delta for certain strategies
                    price_delta = abs(entry_price - last_price) / last_price
                    
                    # Step 3c: IF price_delta >= threshold, BYPASS
                    if price_delta >= price_delta_bypass_threshold:
                        # Log bypass event with details
                        logger.info(
                            f"✅ [DUPLICATE-BYPASS] Cooldown bypassed\n"
                            f"   Symbol: {symbol}\n"
                            f"   Strategy: {strategy_name}\n"
                            f"   Intent: {intent}\n"
                            f"   Last Price: ${last_price:.2f}\n"
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
                    else:
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
                        
                        return False, f"Duplicate prevention: Signal cooldown: {remaining:.0f}s remaining (price change {price_delta*100:.2f}% < threshold)"
            
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"SIGNAL_BREAKDOWN {json.dumps(breakdown)}")

    def _compute_signal_quality(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Compute and attach quality metrics to a signal (single source of truth)."""
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
            signal["quality_score"] = quality_result["value"]
            signal["quality_breakdown"] = quality_result
            return quality_result

        # Normal profile (includes ML component if available)
        features = signal.get('features', {}) or {}

        def _get_val(primary, secondary=None):
            return primary if primary is not None else secondary

        quality_features = {
            "ml_component": signal.get("ml_confidence"),
            "volume_component": _get_val(features.get("volume_score"), signal.get("volume_24h")),
            "momentum_component": features.get("momentum"),
            "spread_component": features.get("spread"),
        }
        quality_result = compute_quality(quality_features, logger)
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
            log_prefix = f"[{strategy_name.upper()}/{symbol}]"

            # Default all signals to entry intent unless explicitly provided
            signal.setdefault("intent", INTENT_ENTRY)
            signal["intent"] = self._determine_intent(signal, strategy_name)
            intent = signal.get("intent", INTENT_ENTRY)

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
                            logger.info(
                                f"⚠️ [VOLUME-OVERRIDE] {log_prefix} | Bucket '{volume_bucket}' < min '{min_bucket}' "
                                f"but allow_low_volume=True. Accepting."
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
                    'central_bucket_decision': decision
                }
                logger.info(f"volume_decision_check {json.dumps(audit_payload)}")

                if decision != "accepted":
                    self.processing_stats['rejected_signals'] += 1
                    logger.warning(f"🛡️  {log_prefix} REJECTED (Volume Gating): {rejection_reason}")
                    return {'status': 'rejected', 'reason': rejection_reason, 'stage': 'volume_gating'}
            
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
                self.processing_stats['rejected_signals'] += 1
                # --- TELEMETRİ: Ret Sebebi (DÜZELTİLDİ) ---
                logger.warning(f"🛡️  {log_prefix} REJECTED (Risk Check): {risk_assessment['reason']}")
                return {'status': 'rejected', 'reason': risk_assessment['reason'], 'stage': 'risk_assessment'}

            # Late duplicate check for scale-in when pyramiding is enabled (soft guard)
            if not duplicate_checked:
                is_valid_duplicate, duplicate_reason = self.validate_duplicate(enriched_signal, strategy_name)
                if not is_valid_duplicate:
                    self.processing_stats['rejected_signals'] += 1
                    self.processing_stats['duplicate_rejections'] += 1
                    return {'status': 'rejected', 'reason': duplicate_reason, 'stage': 'duplicate_validation'}
            
            # Adim 7: Sinyali ve Rota Bilgisini Hazirla
            routing_result = self._route_signal(enriched_signal, risk_assessment)
            signal_id = self._generate_signal_id(strategy_name, enriched_signal)
            enriched_signal["signal_id"] = signal_id

            # --- Emit Signal Breakdown ---
            self.emit_signal_breakdown(enriched_signal, quality_result)
            
            self.active_signals[signal_id] = {
                'signal': enriched_signal, 'risk_assessment': risk_assessment,
                'routing': routing_result, 'timestamp': datetime.now(timezone.utc), 'status': 'active'
            }
            
            # --- TELEMETRİ: Sinyal kuyruğa eklenirken (DÜZELTİLDİ) ---
            display_notional = risk_assessment.get('notional')
            if display_notional is None:
                try:
                    display_notional = float(risk_assessment.get('position_size', 0) or 0) * float(enriched_signal.get('entry') or 0)
                except Exception:
                    display_notional = 0.0

            logger.info(
                f"✅ {log_prefix} ENQUEUED. Side: {enriched_signal.get('side')}, "
                f"Entry: ${enriched_signal.get('entry'):.2f}, SL: ${enriched_signal.get('stop'):.2f}, TP: ${enriched_signal.get('target'):.2f}, "
                f"Size: ${display_notional:.2f}"
            )

            # Adım 8: Sinyali Yürütme Kuyruğuna Ekle
            queued, queue_reason = await self.signal_queue.put({
                'signal_id': signal_id,
                'signal': enriched_signal,
                'risk_assessment': risk_assessment,
                'routing': routing_result
            })

            if not queued:
                self.processing_stats['rejected_signals'] += 1
                self.processing_stats['queue_rejections'] += 1
                return {'status': 'rejected', 'reason': queue_reason, 'stage': 'queue'}
            
            self.processing_stats['accepted_signals'] += 1
            self._add_signal_history_entry({
                'signal_id': signal_id, 'strategy_name': strategy_name,
                'symbol': enriched_signal.get('symbol'), 'timestamp': datetime.now(timezone.utc), 'status': 'accepted'
            })

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
        tail_overrides = self._build_ppo_state_overrides(normalized_symbol)
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
            tail_overrides = self._build_ppo_state_overrides(normalized_symbol)
            
            score, metadata = await adapter.get_long_score(
                normalized_symbol,
                position_fraction=tail_overrides.get('position_fraction'),
                normalized_pv=tail_overrides.get('normalized_pv'),
            )
            
            # 4. Log result with specific tag for monitoring
            if metadata.get('reason') != 'unsupported_symbol':
                action = "BUY" if score >= 0.5 else "HOLD"
                logger.info(
                    f"👀 [PPO-MONITOR] {symbol} | Score: {score:.4f} | Action: {action} | "
                    f"Conf: {metadata.get('confidence', 0.0):.2f}"
                )
                
                # Record telemetry
                self._record_ppo_telemetry(score)
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

    def _build_ppo_state_overrides(self, symbol: str) -> Dict[str, float]:
        overrides: Dict[str, float] = {}
        normalized_symbol = self._normalize_symbol_for_ppo(symbol)
        pos_fraction = self._compute_symbol_position_fraction(normalized_symbol)
        if pos_fraction is not None:
            overrides['position_fraction'] = pos_fraction
        normalized_pv = self._compute_normalized_equity()
        if normalized_pv is not None:
            overrides['normalized_pv'] = normalized_pv
        return overrides

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
                ctx = await self.volume_analyzer.compute_context(
                    symbol=symbol,
                    trade_timeframe=trade_tf,
                    as_of_ts=as_of_ts,
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
                logger.info(f"volume_context {volume_ctx_log}")

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
            sized_signal = await self.position_sizing.calculate_optimal_size(signal, return_signal=True)

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

            planner_result = meta.get('planner')
            planner_dict = asdict(planner_result) if planner_result else None
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
                return {
                    'acceptable': False,
                    'reason': reason,
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
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        symbol = signal.get('symbol', 'UNKNOWN').replace('/', '_').replace(':', '_')
        return f"{strategy_name}_{symbol}_{timestamp}"
    
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

        logger.info(f"Signal {signal_id} marked as executed and removed from active registry")

    def discard_active_signal(self, signal_id: str) -> None:
        """Remove a signal from the active registry without raising errors."""
        if not signal_id:
            return

        removed = self.active_signals.pop(signal_id, None)

        if removed:
            logger.warning(
                "Signal %s discarded from active registry after lifecycle callback failure",
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
