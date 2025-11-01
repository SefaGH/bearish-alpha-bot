"""
Adaptive ShortTheRip strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
from typing import Optional, Dict
from .short_the_rip import ShortTheRip

# Default market regime for fallback
DEFAULT_MARKET_REGIME = {
    'trend': 'neutral',
    'momentum': 'sideways', 
    'volatility': 'normal',
    'micro_trend_strength': 0.5,
    'entry_score': 0.5,
    'risk_multiplier': 1.0
}

logger = logging.getLogger(__name__)


class AdaptiveShortTheRip(ShortTheRip):
    """
    Market regime-aware ShortTheRip strategy.
    
    Adapts RSI thresholds, position sizing, and EMA requirements
    based on real-time market regime analysis.
    """
    
    # Maximum adjustment to base threshold (in RSI points)
    MAX_THRESHOLD_ADJUSTMENT = 5
    
    def __init__(self, cfg: Dict, regime_analyzer=None):
        """
        Initialize adaptive ShortTheRip strategy.
        
        Args:
            cfg: Strategy configuration dictionary
            regime_analyzer: MarketRegimeAnalyzer instance for regime detection
        """
        super().__init__(cfg)
        self.regime_analyzer = regime_analyzer
        self.base_cfg = cfg.copy()
        self.debug_logging = self.base_cfg.get('debug', {}).get('strategy_logging', False)

    def _validate_input_data(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame, regime_data: Dict, symbol: str) -> tuple[bool, str]:
        """Gerekli tüm verilerin varlığını ve geçerliliğini kontrol eder."""
        if df_30m is None or df_30m.empty:
            return False, "Input data 'df_30m' is missing or empty."
        
        required_cols = ['close', 'rsi', 'atr', 'ema_fast', 'ema21', 'ema50', 'ema200']
        missing_cols = [col for col in required_cols if col not in df_30m.columns]
        if missing_cols:
            return False, f"df_30m is missing required indicator columns: {missing_cols}."
            
        if self.debug_logging:
            if df_1h is None or df_1h.empty:
                logger.info(f"[{self.strategy_name.upper()}-INFO] {symbol} - 'df_1h' is missing. Market regime analysis may be less accurate.")
            if regime_data is None:
                logger.info(f"[{self.strategy_name.upper()}-INFO] {symbol} - 'regime_data' is missing. Strategy will fallback to non-adaptive mode.")

        return True, "All required data is present."
    
    def get_symbol_specific_threshold(self, symbol: str) -> Optional[float]:
        """
        Get symbol-specific RSI threshold override if configured.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            
        Returns:
            Symbol-specific threshold or None
        """
        if not symbol:
            return None
        
        # Check if symbol-specific config exists
        symbols_cfg = self.base_cfg.get('symbols', {})
        if symbol in symbols_cfg:
            symbol_cfg = symbols_cfg[symbol]
            if 'rsi_threshold' in symbol_cfg:
                return float(symbol_cfg['rsi_threshold'])
        
        return None
    
    def get_adaptive_rsi_threshold(self, market_regime: Dict) -> float:
        """
        Dynamic RSI thresholds based on market conditions.
        Now respects config values and uses gentler adjustments.
        
        Args:
            market_regime: Dictionary with 'trend', 'momentum', 'volatility'
            
        Returns:
            Adaptive RSI threshold for overbought detection
        """
        # Get config values with proper fallbacks
        base_rsi = float(self.base_cfg.get('adaptive_rsi_base',
                         self.base_cfg.get('rsi_min', 50)))
        
        # Get adjustment range from config (default ±10)
        adapt_range = float(self.base_cfg.get('adaptive_rsi_range', 10))
        
        trend = market_regime.get('trend', 'neutral')
        momentum = market_regime.get('momentum', 'sideways')
        
        # Start with base value
        threshold = base_rsi
        
        # Gentler adjustments based on regime
        # For short strategy: bearish = more aggressive (lower threshold), bullish = more selective (higher threshold)
        if trend == 'bearish':
            # In downtrends, be slightly more aggressive with shorts
            if momentum == 'strong':
                threshold = base_rsi - min(self.MAX_THRESHOLD_ADJUSTMENT, adapt_range/2)
            else:
                threshold = base_rsi - min(self.MAX_THRESHOLD_ADJUSTMENT * 0.6, adapt_range/3)
        
        elif trend == 'bullish':
            # In uptrends, be more selective (need higher RSI)
            if momentum == 'strong':
                threshold = base_rsi + min(self.MAX_THRESHOLD_ADJUSTMENT, adapt_range/2)
            else:
                threshold = base_rsi + min(self.MAX_THRESHOLD_ADJUSTMENT * 0.6, adapt_range/3)
        
        # Clamp to reasonable range for shorts (55-85 range)
        min_threshold = max(55, base_rsi - adapt_range)  # STR için minimum 55
        max_threshold = min(85, base_rsi + adapt_range)  # STR için maximum 85
        
        return max(min_threshold, min(max_threshold, threshold))
    
    def calculate_dynamic_position_size(self, volatility_regime: str, 
                                       base_multiplier: float = 1.0) -> float:
        """
        Volatility-adjusted position sizing multiplier.
        
        Args:
            volatility_regime: 'high', 'normal', or 'low'
            base_multiplier: Base position size multiplier
            
        Returns:
            Adjusted position size multiplier
        """
        # High volatility: Reduce position size for risk management
        if volatility_regime == 'high':
            return base_multiplier * 0.75
        
        # Low volatility: Can increase position size slightly
        elif volatility_regime == 'low':
            return base_multiplier * 1.25
        
        # Normal volatility: Use base multiplier
        else:
            return base_multiplier
    
    def adapt_ema_requirements(self, trend_strength: float) -> Dict[str, any]:
        """
        EMA alignment requirements based on trend strength.
        
        Args:
            trend_strength: Trend strength metric (0.0 to 1.0)
            
        Returns:
            Dictionary with EMA requirement parameters
        """
        # Strong trends: Require strict EMA alignment
        if trend_strength > 0.7:
            return {
                'require_strict_ema_align': True,
                'ema_tolerance': 0.001  # 0.1% tolerance
            }
        
        # Weak trends: Relax EMA requirements
        elif trend_strength < 0.3:
            return {
                'require_strict_ema_align': False,
                'ema_tolerance': 0.01  # 1% tolerance
            }
        
        # Moderate trends: Standard requirements
        else:
            return {
                'require_strict_ema_align': True,
                'ema_tolerance': 0.005  # 0.5% tolerance
            }
    
    def signal(self, df_30m: pd.DataFrame, 
               df_1h: pd.DataFrame = None,
               regime_data: Optional[Dict] = None,
               symbol: str = None,
               market_data: Optional[Dict] = None,
               ml_context=None) -> Optional[Dict]:
        """
        Generate adaptive trading signal based on market regime and ML insights.
        
        Args:
            df_30m: 30-minute OHLCV dataframe with indicators
            df_1h: Optional 1-hour OHLCV dataframe with indicators
            regime_data: Optional market regime data for adaptation
                        If None, falls back to base strategy
            symbol: Symbol name for debug logging
            market_data: Optional dictionary containing all available timeframes
                        Format: {'1m': df, '5m': df, '30m': df, '1h': df, '4h': df}
                        Allows strategies to access additional timeframes if needed
            ml_context: Optional MLContext with ML predictions and insights
        
        Returns:
            Signal dictionary or None
        """
        # Log symbol for debugging multi-symbol trading
        symbol_display = symbol or "UNKNOWN"
        logger.info(f"[STR-DEBUG] {symbol_display}")
        
        # --- Kapsamlı Veri Doğrulama ---
        if self.debug_logging:
            validation_passed, reason = self._validate_input_data(df_30m, df_1h, regime_data, symbol_display)
            if not validation_passed:
                logger.warning(f"[{self.strategy_name.upper()}-REJECT] {symbol_display} - {reason}")
                return None
        
        # Safely get last row
        try:
            last30 = df_30m.dropna().iloc[-1]
        except IndexError:
            logger.info(f"  ❌ Insufficient 30m data")
            return None
        
        # 1h data is optional for adaptive strategy
        last1h = None
        if df_1h is not None and not df_1h.empty:
            try:
                last1h = df_1h.dropna().iloc[-1]
            except IndexError:
                # Continue without 1h data
                pass
        
        # Analyze market regime with available data
        if regime_data is None:
            if last1h is not None:
                # Try to analyze regime if we have regime analyzer
                if self.regime_analyzer:
                    try:
                        regime_data = self.regime_analyzer.analyze_regime(last30, last1h)
                    except Exception as e:
                        logger.debug(f"Failed to analyze regime: {e}")
                        regime_data = None
            
            if regime_data is None:
                # Use default neutral regime
                regime_data = DEFAULT_MARKET_REGIME.copy()
        
        try:
            # Ensure we have valid data with critical columns
            if 'rsi' not in last30.index or 'close' not in last30.index:
                logger.info(f"  ❌ Missing required columns (RSI or close)")
                return None
            
            # Get price and RSI data
            close_price = float(last30['close'])
            rsi_val = float(last30['rsi'])
            ema_fast = float(last30.get('ema_fast', 0))
            
            # Get adaptive RSI threshold
            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }
            
            adaptive_rsi_threshold = self.get_adaptive_rsi_threshold(market_regime)

            # --- Detaylı Teşhis Logları (Tam Hali) ---
            if self.debug_logging:
                logger.info(f"🔍 [{self.strategy_name.upper()}-CHECK] {symbol_display}")
                logger.info(f"  - Market Regime: {market_regime.get('trend', 'N/A')}")
                logger.info(f"  - Current Price: ${close_price:,.2f}")
                logger.info(f"  - Current 30m RSI: {rsi_val:.2f}")
                logger.info(f"  - Adaptive RSI Threshold: {adaptive_rsi_threshold:.2f}")

            # 1. Ana Koşul: RSI
            rsi_condition = rsi_val >= adaptive_rsi_threshold
            if self.debug_logging:
                logger.info(f"  - Condition 'RSI >= Threshold': {rsi_condition}")
            
            if not rsi_condition:
                if self.debug_logging: logger.info(f"  └─ ❌ REJECT: RSI ({rsi_val:.2f}) is below the adaptive threshold ({adaptive_rsi_threshold:.2f}).")
                return None

            # 2. Teyit Koşulu: EMA Hizalaması
            trend_strength = regime_data.get('micro_trend_strength', 0.5)
            ema_params = self.adapt_ema_requirements(trend_strength)
            ema_ok = True
            if ema_params['require_strict_ema_align']:
                ema21 = float(last30['ema21'])
                ema50 = float(last30['ema50'])
                ema200 = float(last30['ema200'])
                ema_ok = ema21 < ema50 <= ema200
                if self.debug_logging:
                    logger.info(f"  - Condition 'Strict EMA Align (21<50<=200)': {ema_ok} (Values: {ema21:.1f} < {ema50:.1f} <= {ema200:.1f})")
            
            if not ema_ok:
                if self.debug_logging: logger.info(f"  └─ ❌ REJECT: EMA alignment check failed.")
                return None
            
            if self.debug_logging: logger.info("  └─ ✅ ACCEPT: All conditions met. Proceeding to signal generation.")
            
            # Get symbol-specific threshold override if available
            symbol_specific_threshold = self.get_symbol_specific_threshold(symbol_display)
            if symbol_specific_threshold is not None:
                adaptive_rsi_threshold = symbol_specific_threshold
                logger.info(f"  📌 Using symbol-specific RSI threshold: {adaptive_rsi_threshold:.2f}")
            
            # Log current state
            logger.info(f"  RSI: {rsi_val:.2f} (threshold: {adaptive_rsi_threshold:.2f})")
            
            # Check RSI condition
            if rsi_val < adaptive_rsi_threshold:
                logger.info(f"  ❌ Signal: NONE - RSI {rsi_val:.2f} < threshold {adaptive_rsi_threshold:.2f}")
                return None
            else:
                logger.info(f"  ✅ RSI check passed: {rsi_val:.2f} >= {adaptive_rsi_threshold:.2f}")
            
            # Get trend strength for EMA adaptation
            trend_strength = regime_data.get('micro_trend_strength', 0.5)
            ema_params = self.adapt_ema_requirements(trend_strength)
            
            # Check EMA alignment if required
            ema_ok = True
            if ema_params['require_strict_ema_align']:
                if all(col in last30.index for col in ('ema21','ema50','ema200')):
                    ema21 = float(last30['ema21'])
                    ema50 = float(last30['ema50'])
                    ema200 = float(last30['ema200'])
                    # Strict alignment: 21 < 50 <= 200 (bearish alignment)
                    ema_ok = ema21 < ema50 <= ema200
                    ema_status = "✅" if ema_ok else "❌"
                    logger.info(f"  EMA Align: {ema_status} (21={ema21:.2f}, 50={ema50:.2f}, 200={ema200:.2f})")
                else:
                    logger.info(f"  ⚠️ EMA Align: Missing EMA columns")
                    ema_ok = False
            else:
                logger.info(f"  EMA Align: ✅ (not required for this regime)")
            
            if not ema_ok:
                logger.info(f"  ❌ Signal: NONE - EMA alignment check failed")
                return None
            
            # Check volume if available
            volume_ok = True
            if 'volume' in last30.index:
                volume_val = float(last30['volume'])
                logger.info(f"  Volume: {volume_val:.2f}")
                # Volume check can be added here if needed
                volume_ok = volume_val > 0
            else:
                logger.info(f"  Volume: N/A")
            
            if not volume_ok:
                logger.info(f"  ❌ Signal: NONE - Volume check failed")
                return None
            
            # ===== ML-AWARE DECISION MAKING (NEW) =====
            # Base signal would be SELL (short) since we passed all checks above
            base_signal_direction = 'short'
            position_size_modifier = 1.0  # Default: no adjustment
            ml_enhanced = False
            
            # Check if we have healthy ML context
            MIN_ML_CONFIDENCE_THRESHOLD = 0.60 # Güvenilir ML tahmini için minimum eşik

            if ml_context and \
               ml_context.get('is_healthy', False) and \
               ml_context.get('regime_confidence', 0) >= MIN_ML_CONFIDENCE_THRESHOLD:
                
                ml_enhanced = True
                
                # VETO: ML strongly disagrees with our short signal
                if ml_context.get('regime_prediction') == 'bullish' and ml_context.get('regime_confidence', 0) > 0.7:
                    if self.debug_logging:
                        logger.info(
                            f"  🧠 [ML-VETO] {symbol_display}: ML regime is BULLISH "
                            f"(conf={ml_context.get('regime_confidence', 0):.2%}), vetoing SHORT signal"
                        )
                    return None
                
                if ml_context.get('price_direction') == 'up' and ml_context.get('price_confidence', 0) > 0.7:
                    if self.debug_logging:
                        logger.info(
                            f"  🧠 [ML-VETO] {symbol_display}: ML predicts price UP "
                            f"(conf={ml_context.get('price_confidence', 0):.2%}), vetoing SHORT signal"
                        )
                    return None
                
                # CONFIRMATION: ML agrees with our short signal - increase position size
                if (ml_context.get('regime_prediction') == 'bearish' and ml_context.get('regime_confidence', 0) > 0.6) or \
                   (ml_context.get('price_direction') == 'down' and ml_context.get('price_confidence', 0) > 0.6):
                    # Increase position size by up to 25% based on consensus
                    position_size_modifier = 1.0 + (0.25 * ml_context.get('consensus_score', 0))
                    if self.debug_logging:
                        logger.info(
                            f"  🧠 [ML-CONFIRM] {symbol_display}: ML confirms SHORT signal "
                            f"(regime={ml_context.get('regime_prediction')}, price={ml_context.get('price_direction')}), "
                            f"increasing position size by {(position_size_modifier - 1.0) * 100:.1f}%"
                        )
                
                # WEAK CONSENSUS: Reduce position size if ML is uncertain
                if ml_context.get('consensus_score', 1.0) < 0.5:
                    position_size_modifier *= 0.75  # Reduce by 25%
                    if self.debug_logging:
                        logger.info(
                            f"  🧠 [ML-CAUTION] {symbol_display}: Low ML consensus "
                            f"({ml_context.get('consensus_score', 0):.2%}), reducing position size by 25%"
                        )
            elif ml_context and not ml_context.get('is_healthy', False):
                if self.debug_logging:
                    logger.info(
                        f"  🧠 [ML-UNAVAILABLE] {symbol_display}: ML context unhealthy, "
                        f"proceeding with base strategy only"
                    )
            
            # Calculate position size adjustment
            volatility = regime_data.get('volatility', 'normal')
            position_mult = self.calculate_dynamic_position_size(volatility)
            
            # Apply ML-based position size modifier
            position_mult *= position_size_modifier
            
            # ===== ATR-BASED TP/SL CALCULATION FOR SHORT =====
            entry_price = float(last30['close'])
            atr_value = float(last30['atr']) if 'atr' in last30.index else entry_price * 0.02
            logger.info(f"  ATR: {atr_value:.4f}")
            
            # Get ATR multipliers from config
            tp_atr_mult = float(self.cfg.get("tp_atr_mult", 3.0))
            sl_atr_mult = float(self.cfg.get("sl_atr_mult", 1.5))
            
            # Calculate TP and SL from ATR (SHORT: TP below entry, SL above entry)
            target_price = entry_price - (atr_value * tp_atr_mult)
            stop_price = entry_price + (atr_value * sl_atr_mult)
            
            # Safety boundaries
            min_tp_pct = float(self.cfg.get("min_tp_pct", 0.010))
            max_sl_pct = float(self.cfg.get("max_sl_pct", 0.020))
            
            # Enforce minimum TP (for short, target is below entry)
            if (entry_price - target_price) / entry_price < min_tp_pct:
                target_price = entry_price * (1 - min_tp_pct)
            
            # Enforce maximum SL (for short, stop is above entry)
            if (stop_price - entry_price) / entry_price > max_sl_pct:
                stop_price = entry_price * (1 + max_sl_pct)
            
            # Calculate and validate R/R ratio
            rr_ratio = (entry_price - target_price) / (stop_price - entry_price)

            # --- 🔥 IYILESTIRME: Olası 'division by zero' hatasını önle ---
            risk_amount = stop_price - entry_price
            reward_amount = entry_price - target_price
            if risk_amount <= 0:
                rr_ratio = float('inf') # Risk yoksa R/R sonsuzdur
            else:
                rr_ratio = reward_amount / risk_amount
            
            # --- 🔥 YENİ EKLENECEK BÖLÜM BAŞLANGICI 🔥 ---
            min_rr_ratio = self.cfg.get('min_rr_ratio', 1.2) # config'den min oranı oku, yoksa 1.2 kullan
            if rr_ratio < min_rr_ratio:
                if self.debug_logging: 
                    logger.info(f"  └─ ❌ REJECT: R/R ratio ({rr_ratio:.2f}) is below minimum required ({min_rr_ratio}).")
                return None
            # --- 🔥 YENİ EKLENECEK BÖLÜM SONU 🔥 ---
            
            # Calculate percentages for signal
            tp_pct = (entry_price - target_price) / entry_price
            sl_pct = (stop_price - entry_price) / entry_price
            
            # Build adaptive signal with ATR-based TP/SL
            signal = {
                "strategy_name": self.strategy_name, # 🔥 BU SATIRI EKLEYİN
                "side": "sell",
                "entry": entry_price,
                "stop": stop_price,
                "target": target_price,
                "reason": f"Adaptive RSI overbought {rsi_val:.1f} (threshold: {adaptive_rsi_threshold:.1f}, regime: {market_regime['trend']}, R/R: {rr_ratio:.2f})",
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_atr_mult": tp_atr_mult,
                "sl_atr_mult": sl_atr_mult,
                "atr": atr_value,
                "rr_ratio": rr_ratio,
                "is_adaptive": True,
                "adaptive_threshold": adaptive_rsi_threshold,
                "position_multiplier": position_mult,
                "market_regime": market_regime,
                "ema_params": ema_params,
                "ml_enhanced": ml_enhanced
            }
            
            # Add ML metadata if available
            if ml_context and ml_context.get('is_healthy', False):
                signal['ml_regime'] = ml_context.get('regime_prediction')
                signal['ml_regime_confidence'] = ml_context.get('regime_confidence')
                signal['ml_price_direction'] = ml_context.get('price_direction')
                signal['ml_consensus'] = ml_context.get('consensus_score')
                signal['ml_position_modifier'] = position_size_modifier
            
            logger.info(f"  ✅ Signal: SELL (RSI {rsi_val:.1f} >= {adaptive_rsi_threshold:.1f}, regime={market_regime['trend']})")
            if ml_enhanced:
                logger.info(f"  🧠 ML-Enhanced: regime={ml_context.get('regime_prediction')}, price={ml_context.get('price_direction')}, modifier={position_size_modifier:.2f}x")
            logger.info(f"  Entry: ${entry_price:.2f}, Target: ${target_price:.2f}, Stop: ${stop_price:.2f}, R/R: {rr_ratio:.2f}")
            
            # Strategy type ekle ve signal'i döndür
            signal['strategy_type'] = 'adaptive'
            signal['symbol'] = symbol  # Add symbol field required by StrategyCoordinator
            return signal
            
        except Exception as e:
            logger.error(f"Adaptive strategy failed for {symbol_display}: {e}", exc_info=True)
            
            # FALLBACK TO BASE STRATEGY
            try:
                # Base ShortTheRip için
                if hasattr(super(), 'signal'):
                    base_signal = super().signal(df_30m, df_1h)
                    if base_signal:
                        base_signal['strategy_type'] = 'base_fallback'
                        base_signal['fallback_reason'] = str(e)
                        base_signal['symbol'] = symbol  # Add symbol field for fallback signals
                        logger.info("✅ Fallback to base strategy successful")
                        return base_signal
            except Exception as fallback_error:
                logger.error(f"Base strategy also failed: {fallback_error}")
                
        return None
    
    def get_strategy_state(self) -> Dict:
        """
        Get current strategy state and parameters.
        
        Returns:
            Dictionary with current adaptive parameters
        """
        return {
            'strategy': 'adaptive_short_the_rip',
            'base_config': self.base_cfg,
            'has_regime_analyzer': self.regime_analyzer is not None
        }
