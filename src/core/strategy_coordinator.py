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
    
        # Signal management
        self.active_signals = {}  # signal_id -> signal_data
        self.signal_queue = asyncio.Queue()
        self.signal_history = []
        self._signal_history_lookup: Dict[str, Dict[str, Any]] = {}
    
        # Conflict tracking
        self.conflict_history = []
    
        # Duplicate prevention tracking
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
        
        # ML integration placeholders
        self.ml_integration = None
        self.feature_pipeline = None
        self.rl_agent = None
    
        logger.info("StrategyCoordinator initialized (market_data_pipeline=%s)", bool(self.market_data_pipeline))
    
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
        
        # Step 1: Get config
        config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
        
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
            cooldown = float(dup_config.get('cooldown_seconds', 20))
            
            # ✅ FIX #2: No double division - value is already in decimal
            # Config has min_price_change_pct: 0.0005 which means 0.05% in decimal
            price_delta_bypass_threshold = float(dup_config.get('min_price_change_pct', 0.0005))
            price_delta_bypass_enabled = dup_config.get('price_delta_bypass_enabled', True)
            
            logger.debug(f"✓ Using signals.duplicate_prevention config")
        else:
            # ✅ Fallback to monitoring config (OLD location, backward compatibility)
            dup_config = config.get('monitoring', {}).get('duplicate_prevention', {})
            enabled = dup_config.get('enabled', True)
            cooldown = float(dup_config.get('same_symbol_cooldown', 60))
            price_delta_bypass_enabled = dup_config.get('price_delta_bypass_enabled', True)
            
            # ✅ FIX #3: monitoring config uses different unit (0.0015 = 0.15%)
            # Keep as-is for backward compatibility
            price_delta_bypass_threshold = float(dup_config.get('price_delta_bypass_threshold', 0.0015))
            
            logger.debug(f"⚠️ Using monitoring.duplicate_prevention config (legacy)")
        
        if not enabled:
            return True, "OK"
        
        symbol = signal.get('symbol')
        entry_price = signal.get('entry', 0)
        current_time = time.time()
        
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
                # Find last price for this symbol
                if self.signal_price_history[symbol]:
                    last_timestamp, last_price = self.signal_price_history[symbol][-1]
                    
                    # Step 3b: Calculate price_delta (in decimal, e.g., 0.0005 = 0.05%)
                    price_delta = abs(entry_price - last_price) / last_price
                    
                    # Step 3c: IF price_delta >= threshold, BYPASS
                    if price_delta >= price_delta_bypass_threshold:
                        # Log bypass event with details
                        logger.info(
                            f"✅ [DUPLICATE-BYPASS] Cooldown bypassed\n"
                            f"   Symbol: {symbol}\n"
                            f"   Strategy: {strategy_name}\n"
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
                        
                        return True, f"OK (price delta bypass: {price_delta*100:.2f}%)"
                    
                    # Step 3d: ELSE, reject with price delta info
                    else:
                        logger.warning(
                            f"❌ [DUPLICATE-REJECT] Signal rejected - insufficient price movement\n"
                            f"   Symbol: {symbol}\n"
                            f"   Strategy: {strategy_name}\n"
                            f"   Price Change: {price_delta*100:.2f}% (< {price_delta_bypass_threshold*100:.2f}%)\n"
                            f"   Cooldown Remaining: {remaining:.1f}s\n"
                            f"   ❌ SIGNAL REJECTED"
                        )
                        
                        self.processing_stats['rejected_price_delta'] += 1
                        
                        return False, f"Duplicate prevention: Signal cooldown: {remaining:.0f}s remaining (price change {price_delta*100:.2f}% < threshold)"
            
            # No price history or bypass disabled
            logger.warning(
                f"❌ [DUPLICATE-REJECT] Signal rejected - cooldown active\n"
                f"   Symbol: {symbol}\n"
                f"   Strategy: {strategy_name}\n"
                f"   Cooldown Remaining: {remaining:.1f}s\n"
                f"   ❌ SIGNAL REJECTED"
            )
            self.processing_stats['rejected_cooldown'] += 1
            return False, f"Duplicate prevention: Signal cooldown: {remaining:.0f}s remaining (same symbol+strategy)"
        
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

    # ===============================================================
    # ====================   DÜZELTİLMİŞ METOT   ====================
    # ===============================================================
    async def process_strategy_signal(self, strategy_name: str, signal: Dict) -> Dict[str, Any]:
        """
        Process incoming signals from registered strategies.
        (GÜNCELLENDİ: 'self.logger' -> 'logger' hatası düzeltildi)
        """
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            log_prefix = f"[{strategy_name.upper()}/{symbol}]"

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
            enriched_signal = self._enrich_signal(strategy_name, signal)
            
            # Adım 3: Duplikasyon ve Cooldown Kontrolü
            is_valid_duplicate, duplicate_reason = self.validate_duplicate(enriched_signal, strategy_name)
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
                    return {'status': 'rejected', 'reason': 'ML/RL enhancement blocked signal', 'stage': 'ml_enhancement'}
            
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
            risk_assessment = await self._assess_signal_risk(enriched_signal)
            if not risk_assessment['acceptable']:
                self.processing_stats['rejected_signals'] += 1
                # --- TELEMETRİ: Ret Sebebi (DÜZELTİLDİ) ---
                logger.warning(f"🛡️  {log_prefix} REJECTED (Risk Check): {risk_assessment['reason']}")
                return {'status': 'rejected', 'reason': risk_assessment['reason'], 'stage': 'risk_assessment'}
            
            # Adım 7: Sinyali ve Rota Bilgisini Hazırla
            routing_result = self._route_signal(enriched_signal, risk_assessment)
            signal_id = self._generate_signal_id(strategy_name, enriched_signal)
            
            self.active_signals[signal_id] = {
                'signal': enriched_signal, 'risk_assessment': risk_assessment,
                'routing': routing_result, 'timestamp': datetime.now(timezone.utc), 'status': 'active'
            }
            
            # --- TELEMETRİ: Sinyal kuyruğa eklenirken (DÜZELTİLDİ) ---
            logger.info(
                f"✅ {log_prefix} ENQUEUED. Side: {enriched_signal.get('side')}, "
                f"Entry: ${enriched_signal.get('entry'):.2f}, SL: ${enriched_signal.get('stop'):.2f}, TP: ${enriched_signal.get('target'):.2f}, "
                f"Size: ${risk_assessment.get('position_size'):.2f}"
            )

            # Adım 8: Sinyali Yürütme Kuyruğuna Ekle
            await self.signal_queue.put({
                'signal_id': signal_id, 'signal': enriched_signal,
                'risk_assessment': risk_assessment, 'routing': routing_result
            })
            
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
    
    async def _enhance_signal_with_ml(self, signal: Dict) -> Optional[Dict]:
        """
        Enhance signal with all ML predictions and apply RL agent's decision as a gatekeeper.
        Uses price prediction, regime prediction, and RL agent recommendations.
        
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
            original_side = signal.get('side').lower()

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
            if hasattr(self, 'rl_agent') and self.rl_agent:
                try:
                    # ✅ DÜZELTME: 'await' eklendi.
                    state_features = await self._extract_rl_state(symbol, current_price)
                    
                    # 💡 YENİ LOGLAMA: Ajanın "gördüğü" durumu logla
                    if state_features is not None:
                        logger.info(f"🤖 [RL-DEBUG] State vector for {symbol} (first 5 features): {np.round(state_features[:5], 4)}")
                        logger.info(f"🤖 [RL-DEBUG] Original Signal: {original_side.upper()}, Strategy: {signal.get('strategy_name')}")
                        logger.info(f"🤖 [RL-DEBUG] Market Regime: {signal.get('predicted_regime', 'neutral')}")
                    else:
                        logger.warning(f"🤖 [RL-DEBUG] State features are None for {symbol}. RL Agent cannot make a decision.")

                    rl_action_index = self.rl_agent.act(
                        state_features,
                        market_regime=signal.get('predicted_regime', 'neutral')
                    )
                    rl_advice_str = ['buy', 'hold', 'sell'][rl_action_index]
                    signal['rl_recommendation'] = rl_advice_str
                    rl_advice = rl_advice_str.lower()
                    
                    # CRITICAL: Store RL decision for enrichment
                    from datetime import datetime
                    self._last_rl_decision = {
                        'action': rl_advice_str,
                        'confidence': 0.7,  # Default confidence, can be enhanced if RL agent provides it
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # 💡 YENİ LOGLAMA: Ajanın kararını logla
                    logger.info(f"🤖 [RL-DECISION] For {symbol}, Agent decided: {rl_advice.upper()}")

                except Exception as e:
                    logger.warning(f"RL recommendation failed: {e}", exc_info=True)

            # --- 3. RL VETO VE ANLAŞMAZLIK KONTROLLERİ ---
            if rl_advice == 'hold':
                logger.warning(f"🤖 [RL-VETO] Signal for {symbol} rejected. Reason: RL Agent advised 'hold'.")
                signal['ml_blocked'] = True
                signal['ml_rejection_reason'] = 'RL VETO (HOLD)'
                return None

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

            # --- 4. SON LOGLAMA VE SİNYALİ DÖNDÜRME ---
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
            features_df = self.feature_pipeline.extract_features(df)
            
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
                ml_context = self.ml_integration.get_ml_context(symbol)
                if ml_context:
                    # Use actual ML values
                    signal['ml_confidence'] = float(ml_context.get('consensus_score', 0.5))
                    signal['regime_name'] = str(ml_context.get('regime', 'neutral'))
                    signal['regime_confidence'] = float(ml_context.get('regime_confidence', 0.3))
                    logger.debug(f"✅ ML metrics added: conf={signal['ml_confidence']:.2f}")
                else:
                    # No ML context, use explicit fallbacks
                    signal['ml_confidence'] = 0.5
                    signal['regime_name'] = 'neutral'
                    signal['regime_confidence'] = 0.3
                    logger.debug("⚠️ Using ML fallback values")
            else:
                signal['ml_confidence'] = 0.5
                signal['regime_name'] = 'neutral'
                signal['regime_confidence'] = 0.3
        except Exception as e:
            logger.debug(f"ML enrichment error: {e}")
            signal.update({'ml_confidence': 0.5, 'regime_name': 'neutral', 'regime_confidence': 0.3})
        
        # 2. RL Metrics
        try:
            # Check if we stored the RL decision
            if hasattr(self, '_last_rl_decision') and self._last_rl_decision:
                rl_action = self._last_rl_decision.get('action', 'hold')
                rl_confidence = float(self._last_rl_decision.get('confidence', 0.5))
                signal_side = signal.get('side', '').lower()
                
                # Normalize action strings for comparison
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
        try:
            if hasattr(self, 'market_data_pipeline') and self.market_data_pipeline:
                # Get latest market data
                data = await self.market_data_pipeline.get_latest_ohlcv(symbol, '5m')
                if data is not None and len(data) >= 20:
                    # Volume strength: recent vs average
                    recent_vol = data['volume'].tail(5).mean()
                    avg_vol = data['volume'].tail(20).mean()
                    signal['volume_strength'] = min(recent_vol / avg_vol, 2.0) / 2.0 if avg_vol > 0 else 0.5
                    
                    # Momentum: price change normalized
                    price_change_pct = data['close'].pct_change(10).iloc[-1]
                    signal['momentum_strength'] = max(0, min(1, (price_change_pct + 0.1) / 0.2))
                    logger.debug(f"✅ Market metrics: vol={signal['volume_strength']:.2f}, mom={signal['momentum_strength']:.2f}")
                else:
                    signal.update({'volume_strength': 0.5, 'momentum_strength': 0.5})
                    logger.debug("⚠️ Insufficient market data, using defaults")
            else:
                signal.update({'volume_strength': 0.5, 'momentum_strength': 0.5})
        except Exception as e:
            logger.debug(f"Market metrics error: {e}")
            signal.update({'volume_strength': 0.5, 'momentum_strength': 0.5})
        
        # Log enriched signal
        logger.info(f"📊 [Signal Enriched] {symbol}: "
                    f"ML={signal.get('ml_confidence', 0):.2f}, "
                    f"RL_agree={signal.get('rl_is_agree', False)}, "
                    f"Regime={signal.get('regime_name', 'N/A')} "
                    f"({signal.get('regime_confidence', 0):.2f}), "
                    f"Vol={signal.get('volume_strength', 0):.2f}, "
                    f"Mom={signal.get('momentum_strength', 0):.2f}")
        
        return signal
    
    async def _assess_signal_risk(self, signal: Dict) -> Dict[str, Any]:
        """
        UPDATED: Use AdvancedPositionSizing instead of risk_manager for sizing.
        Also enriches signals with ML/RL intelligence for dynamic R/R calculation.
        """
        try:
            # Initialize position sizing if not already done
            if not hasattr(self, 'position_sizing'):
                from core.position_sizing import AdvancedPositionSizing
                self.position_sizing = AdvancedPositionSizing(self.risk_manager)
                logger.info("✅ AdvancedPositionSizing initialized in StrategyCoordinator")
            
            # CRITICAL: Enrich signal BEFORE risk validation
            signal = await self._enrich_signal_for_dynamic_rr(signal)
            
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
            sized_signal = await self.position_sizing.calculate_optimal_size(signal)
            
            # Check if sizing was successful
            if sized_signal.get('amount', 0) <= 0:
                return {
                    'acceptable': False,
                    'reason': 'Unable to calculate valid position size',
                    'metrics': sized_signal.get('sizing_meta', {})
                }
            
            # NOW validate the sized position with risk rules
            is_valid, reason, risk_metrics = await self.risk_manager.validate_new_position(
                sized_signal,
                portfolio_manager=self.portfolio_manager
            )
            
            if not is_valid:
                return {
                    'acceptable': False,
                    'reason': reason,
                    'metrics': risk_metrics
                }
            
            # Add sizing metadata to risk metrics
            risk_metrics['sizing_meta'] = sized_signal.get('sizing_meta', {})
            
            return {
                'acceptable': True,
                'position_size': sized_signal['amount'],
                'notional': sized_signal['notional'],
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
