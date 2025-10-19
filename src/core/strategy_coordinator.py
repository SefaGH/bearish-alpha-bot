"""
Strategy Coordination Engine.
Coordinates signals and positions across multiple strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


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


class StrategyCoordinator:
    """Coordinate signals and positions across multiple strategies."""
    
    def __init__(self, portfolio_manager, risk_manager):
        """
        Initialize strategy coordinator.
        
        Args:
            portfolio_manager: PortfolioManager instance
            risk_manager: RiskManager instance
        """
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        
        # Signal management
        self.active_signals = {}  # signal_id -> signal_data
        self.signal_queue = asyncio.Queue()
        self.signal_history = []
        
        # Conflict tracking
        self.conflict_history = []
        
        # Duplicate prevention tracking (Phase 3.4 - Issue #104)
        self.last_signal_time = {}  # "symbol:strategy" -> timestamp
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
            'rejected_price_delta': 0
        }
        
        logger.info("StrategyCoordinator initialized")
    
    def validate_duplicate(self, signal: Dict, strategy_name: str) -> Tuple[bool, str]:
        """
        Validate signal for duplicates using cooldown and price movement checks.
        Phase 3.4 - Issue #104: Fixed Cooldown Logic
        Phase 3.4 - Issue #118: Enhanced with Price Delta Bypass
        
        Uses combined key "symbol:strategy" to allow:
        - BTC+strategy1 → ETH+strategy1 ✅
        - BTC+strategy1 → BTC+strategy2 ✅
        - BTC+strategy1 → BTC+strategy1 ❌ (60s cooldown, unless price moved >0.15%)
        
        Args:
            signal: Trading signal dictionary
            strategy_name: Name of the strategy generating the signal
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        import time
        
        # Step 1: Check if duplicate prevention enabled
        config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
        monitoring_config = config.get('monitoring', {}).get('duplicate_prevention', {})
        
        enabled = monitoring_config.get('enabled', True)
        if not enabled:
            return True, "OK"
        
        symbol = signal.get('symbol')
        entry_price = signal.get('entry', 0)
        current_time = time.time()
        
        # Configuration values
        cooldown = float(monitoring_config.get('same_symbol_cooldown', 60))
        price_delta_bypass_enabled = monitoring_config.get('price_delta_bypass_enabled', True)
        price_delta_bypass_threshold = float(monitoring_config.get('price_delta_bypass_threshold', 0.0015))
        
        # Create combined key: "symbol:strategy"
        signal_key = f"{symbol}:{strategy_name}"
        
        # Step 2: Calculate cooldown status
        within_cooldown = False
        remaining = 0
        
        if signal_key in self.last_signal_time:
            elapsed = current_time - self.last_signal_time[signal_key]
            if elapsed < cooldown:
                within_cooldown = True
                remaining = cooldown - elapsed
        
        # Step 3: IF within cooldown, check for price delta bypass
        if within_cooldown:
            # Step 3a: Get last price from history
            if symbol in self.signal_price_history and entry_price > 0 and price_delta_bypass_enabled:
                # Find last price for this symbol (from any recent signal)
                if self.signal_price_history[symbol]:
                    last_timestamp, last_price = self.signal_price_history[symbol][-1]
                    
                    # Step 3b: Calculate price_delta
                    price_delta = abs(entry_price - last_price) / last_price
                    
                    # Step 3c: IF price_delta >= threshold, BYPASS
                    if price_delta >= price_delta_bypass_threshold:
                        # Log bypass event with details
                        logger.info(
                            f"[DUPLICATE-CHECK] Price Delta Check\n"
                            f"   Symbol: {symbol}\n"
                            f"   Last Price: ${last_price:.2f}\n"
                            f"   New Price: ${entry_price:.2f}\n"
                            f"   Delta: {price_delta:.4f} ({price_delta*100:.2f}%)\n"
                            f"   Threshold: {price_delta_bypass_threshold:.4f} ({price_delta_bypass_threshold*100:.2f}%)"
                        )
                        logger.info(
                            f"✅ [DUPLICATE-BYPASS] Cooldown bypassed due to significant price movement\n"
                            f"   Symbol: {symbol}\n"
                            f"   Strategy: {strategy_name}\n"
                            f"   Price Change: {price_delta*100:.2f}% (>= {price_delta_bypass_threshold*100:.2f}%)\n"
                            f"   Cooldown Remaining: {remaining:.1f}s\n"
                            f"   ✅ SIGNAL ACCEPTED"
                        )
                        
                        # Update statistics (cooldown_bypasses counter)
                        self.processing_stats['cooldown_bypasses'] += 1
                        self.processing_stats['last_bypass_time'] = current_time
                        
                        # Update running average for bypass price delta
                        bypass_count = self.processing_stats['cooldown_bypasses']
                        current_avg = self.processing_stats['avg_bypass_price_delta']
                        new_avg = ((current_avg * (bypass_count - 1)) + (price_delta * 100)) / bypass_count
                        self.processing_stats['avg_bypass_price_delta'] = new_avg
                        
                        # Update bypass success rate
                        total_signals = self.processing_stats['total_signals']
                        if total_signals > 0:
                            self.processing_stats['bypass_success_rate'] = (bypass_count / total_signals) * 100
                        
                        # Update tracking (last_signal_time, price_history)
                        self.last_signal_time[signal_key] = current_time
                        self.signal_price_history[symbol].append((current_time, entry_price))
                        
                        return True, f"OK (price delta bypass: {price_delta*100:.2f}%)"
                    
                    # Step 3d: ELSE, reject with price delta info
                    else:
                        # Log rejection with price delta and threshold
                        logger.warning(
                            f"❌ [DUPLICATE-REJECT] Signal rejected - insufficient price movement\n"
                            f"   Symbol: {symbol}\n"
                            f"   Strategy: {strategy_name}\n"
                            f"   Price Change: {price_delta*100:.2f}% (< {price_delta_bypass_threshold*100:.2f}%)\n"
                            f"   Cooldown Remaining: {remaining:.1f}s\n"
                            f"   ❌ SIGNAL REJECTED"
                        )
                        
                        # Update statistics (rejected_price_delta counter)
                        self.processing_stats['rejected_price_delta'] += 1
                        
                        return False, f"Signal cooldown: {remaining:.0f}s remaining (price change {price_delta*100:.2f}% < threshold)"
            
            # No price history available or bypass disabled - reject with cooldown
            logger.warning(
                f"❌ [DUPLICATE-REJECT] Signal rejected - cooldown active\n"
                f"   Symbol: {symbol}\n"
                f"   Strategy: {strategy_name}\n"
                f"   Cooldown Remaining: {remaining:.1f}s\n"
                f"   ❌ SIGNAL REJECTED"
            )
            self.processing_stats['rejected_cooldown'] += 1
            return False, f"Signal cooldown: {remaining:.0f}s remaining (same symbol+strategy)"
        
        # Step 4: IF outside cooldown, accept and update tracking
        self.last_signal_time[signal_key] = current_time
        if entry_price > 0:
            self.signal_price_history[symbol].append((current_time, entry_price))
        
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
    
    async def process_strategy_signal(self, strategy_name: str, signal: Dict) -> Dict[str, Any]:
        """
        Process incoming signals from registered strategies.
        
        Args:
            strategy_name: Name of the strategy generating the signal
            signal: Trading signal dictionary
            
        Returns:
            Processing result with validation status and actions
        """
        try:
            self.processing_stats['total_signals'] += 1
            self.processing_stats['last_signal_time'] = datetime.now(timezone.utc)
            
            logger.info(f"Processing signal from {strategy_name}: {signal.get('symbol', 'UNKNOWN')}")
            
            # Step 1: Validate signal format
            validation_result = self._validate_signal_format(signal)
            if not validation_result['valid']:
                self.processing_stats['rejected_signals'] += 1
                logger.warning(f"Signal validation failed: {validation_result['reason']}")
                return {
                    'status': 'rejected',
                    'reason': validation_result['reason'],
                    'stage': 'validation'
                }
            
            # Step 2: Enrich signal with metadata
            enriched_signal = self._enrich_signal(strategy_name, signal)
            
            # Step 2.5: Validate for duplicates (Phase 3.4 - Issue #103)
            is_valid_duplicate, duplicate_reason = self.validate_duplicate(enriched_signal, strategy_name)
            if not is_valid_duplicate:
                self.processing_stats['rejected_signals'] += 1
                self.processing_stats['duplicate_rejections'] += 1
                logger.info(f"Signal rejected due to duplicate prevention: {duplicate_reason}")
                return {
                    'status': 'rejected',
                    'reason': f"Duplicate prevention: {duplicate_reason}",
                    'stage': 'duplicate_validation'
                }
            
            # Step 3: Check for conflicts with existing positions/signals
            conflict_check = await self._check_signal_conflicts(enriched_signal)
            
            if conflict_check['has_conflict']:
                self.processing_stats['conflicted_signals'] += 1
                logger.info(f"Signal conflict detected: {conflict_check['conflicts']}")
                
                # Resolve conflicts
                resolution = await self.resolve_signal_conflicts(
                    enriched_signal, 
                    conflict_check['conflicting_signals']
                )
                
                if resolution['action'] == 'reject':
                    self.processing_stats['rejected_signals'] += 1
                    return {
                        'status': 'rejected',
                        'reason': resolution['reason'],
                        'stage': 'conflict_resolution',
                        'conflict_details': conflict_check
                    }
            
            # Step 4: Risk assessment
            risk_assessment = await self._assess_signal_risk(enriched_signal)
            
            if not risk_assessment['acceptable']:
                self.processing_stats['rejected_signals'] += 1
                logger.warning(f"Signal rejected by risk assessment: {risk_assessment['reason']}")
                return {
                    'status': 'rejected',
                    'reason': risk_assessment['reason'],
                    'stage': 'risk_assessment',
                    'risk_metrics': risk_assessment.get('metrics', {})
                }
            
            # Step 5: Priority-based routing
            routing_result = self._route_signal(enriched_signal, risk_assessment)
            
            # Step 6: Register signal
            signal_id = self._generate_signal_id(strategy_name, enriched_signal)
            self.active_signals[signal_id] = {
                'signal': enriched_signal,
                'risk_assessment': risk_assessment,
                'routing': routing_result,
                'timestamp': datetime.now(timezone.utc),
                'status': 'active'
            }
            
            # Add to signal queue for execution
            await self.signal_queue.put({
                'signal_id': signal_id,
                'signal': enriched_signal,
                'risk_assessment': risk_assessment,
                'routing': routing_result
            })
            
            # Update stats
            self.processing_stats['accepted_signals'] += 1
            
            # Record in history
            self.signal_history.append({
                'signal_id': signal_id,
                'strategy_name': strategy_name,
                'symbol': enriched_signal.get('symbol'),
                'timestamp': datetime.now(timezone.utc),
                'status': 'accepted'
            })
            
            # Keep last 500 signals
            if len(self.signal_history) > 500:
                self.signal_history = self.signal_history[-500:]
            
            logger.info(f"Signal accepted and queued: {signal_id}")
            
            return {
                'status': 'accepted',
                'signal_id': signal_id,
                'enriched_signal': enriched_signal,
                'risk_assessment': risk_assessment,
                'routing': routing_result
            }
            
        except Exception as e:
            logger.error(f"Error processing signal from {strategy_name}: {e}")
            self.processing_stats['rejected_signals'] += 1
            return {
                'status': 'error',
                'reason': str(e),
                'stage': 'processing'
            }
    
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
    
    def _enrich_signal(self, strategy_name: str, signal: Dict) -> Dict:
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
        
        # Adjust for signal quality
        if signal.get('confidence'):
            confidence = signal['confidence']
            if confidence > 0.8 and priority == SignalPriority.HIGH:
                priority = SignalPriority.CRITICAL
            elif confidence < 0.3:
                priority = SignalPriority.LOW
        
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
        
        # Check active positions from risk manager
        for position_id, position in self.risk_manager.active_positions.items():
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
    
    async def _assess_signal_risk(self, signal: Dict) -> Dict[str, Any]:
        """Assess risk for the signal using risk manager."""
        try:
            # Calculate position size
            position_size = await self.risk_manager.calculate_position_size(signal)
            
            if position_size <= 0:
                return {
                    'acceptable': False,
                    'reason': 'Unable to calculate valid position size',
                    'metrics': {}
                }
            
            # Add position size to signal
            signal['position_size'] = position_size
            
            # Validate position with risk manager
            is_valid, reason, risk_metrics = await self.risk_manager.validate_new_position(
                signal, 
                {}
            )
            
            if not is_valid:
                return {
                    'acceptable': False,
                    'reason': reason,
                    'metrics': risk_metrics
                }
            
            return {
                'acceptable': True,
                'position_size': position_size,
                'metrics': risk_metrics
            }
            
        except Exception as e:
            logger.error(f"Error assessing signal risk: {e}")
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
            if timeout:
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=timeout)
            else:
                signal = await self.signal_queue.get()
            return signal
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error getting next signal: {e}")
            return None
    
    def mark_signal_executed(self, signal_id: str, execution_result: Dict):
        """Mark signal as executed and remove from active signals."""
        if signal_id in self.active_signals:
            self.active_signals[signal_id]['status'] = 'executed'
            self.active_signals[signal_id]['execution_result'] = execution_result
            self.active_signals[signal_id]['execution_time'] = datetime.now(timezone.utc)
            
            # Move to history after short delay
            # In production, clean up periodically
            logger.info(f"Signal {signal_id} marked as executed")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get signal processing statistics."""
        return {
            'stats': self.processing_stats.copy(),
            'active_signals': len(self.active_signals),
            'queued_signals': self.signal_queue.qsize(),
            'signal_history_count': len(self.signal_history),
            'conflict_history_count': len(self.conflict_history)
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
