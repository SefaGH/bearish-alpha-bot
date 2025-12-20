"""
Adaptive ShortTheRip strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
from collections import OrderedDict
from typing import Optional, Dict, Tuple, ClassVar, Any
from .short_the_rip import ShortTheRip
from core.indicators import rsi as calc_rsi, ema as calc_ema

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

    DEFAULT_VOLATILITY_STOP_CFG: ClassVar[Dict[str, Any]] = {
        'enabled': True,
        'min_sl_pct': 0.0025,
        'max_sl_pct': 0.02,
        'overrides': {
            'low': {
                'atr_scale': 0.75,
                'min_sl_pct': 0.0015,
                'max_sl_pct': 0.015,
            },
            'normal': {
                'atr_scale': 1.0,
            },
            'high': {
                'atr_scale': 1.15,
                'min_sl_pct': 0.003,
                'max_sl_pct': 0.025,
            },
        },
    }

    DEFAULT_MTF_MIN_BARS: ClassVar[Dict[str, int]] = {
        'rsi': 20,
        'ema21': 30,
        'ema50': 100,
        'ema200': 250,
    }
    MTF_CACHE_LIMIT: ClassVar[int] = 100
    
    def __init__(self, cfg: Dict, regime_analyzer=None):
        """
        Initialize adaptive ShortTheRip strategy.
        
        Args:
            cfg: Strategy configuration dictionary
            regime_analyzer: MarketRegimeAnalyzer instance for regime detection
            
        🔥 GÜNCELLEME: Kalıtım zinciri `super().__init__(cfg)` çağrısıyla onarıldı.
        Minimum R/R oranı artık başlangıçta config'den okunup `self.min_rr_ratio` olarak saklanıyor.
        """
        # `ShortTheRip` sınıfının __init__ metodunu çağırarak doğru kurulumu sağla.
        super().__init__(cfg)
        
        # Üst sınıf "short_the_rip" olarak ayarlayabileceği için, bu alt sınıfın adını yeniden tanımla.
        self.strategy_name = "adaptive_str"
        
        self.regime_analyzer = regime_analyzer
        self.base_cfg = cfg.copy()
        self.debug_logging = self.base_cfg.get('debug', {}).get('strategy_logging', False)
        
        # Minimum R/R oranını başlangıçta, bir kez olmak üzere config'den oku.
        # `super()` çağrısı `self.strategy_config`'i oluşturduğu için artık bunu güvenle kullanabiliriz.
        self.min_rr_ratio = self.strategy_config.get('min_rr_ratio', 1.2)
        logger.info(f"[{self.strategy_name.upper()}] Minimum R/R Ratio initialized to: {self.min_rr_ratio}")

        self.volatility_stop_cfg = self._build_volatility_stop_cfg(self.strategy_config)
        self._mtf_indicator_cache: OrderedDict[Tuple[Any, ...], Dict[str, pd.Series]] = OrderedDict()
        self._mtf_cache_limit = self.MTF_CACHE_LIMIT
        self._mtf_telemetry = {
            'mtf_15m_fallback_attempted': 0,
            'mtf_15m_fallback_skipped_insufficient_bars': 0,
            'mtf_15m_fallback_computed': 0,
            'mtf_15m_cache_hit': 0,
            'mtf_1h_fallback_attempted': 0,
            'mtf_1h_fallback_skipped_insufficient_bars': 0,
            'mtf_1h_fallback_computed': 0,
            'mtf_1h_cache_hit': 0,
        }

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

    def _build_volatility_stop_cfg(self, strategy_config: Dict) -> Dict:
        overrides = dict(self.DEFAULT_VOLATILITY_STOP_CFG)
        user_cfg = strategy_config.get('volatility_stop') or {}
        merged = {
            **overrides,
            **{k: v for k, v in user_cfg.items() if k not in ('overrides',)},
        }
        merged_overrides = dict(self.DEFAULT_VOLATILITY_STOP_CFG.get('overrides', {}))
        merged_overrides.update(user_cfg.get('overrides', {}) or {})
        merged['overrides'] = merged_overrides

        # Respect global max_sl_pct if user didn't override explicitly.
        if 'max_sl_pct' not in merged or merged['max_sl_pct'] is None:
            merged['max_sl_pct'] = strategy_config.get('max_sl_pct', self.DEFAULT_VOLATILITY_STOP_CFG['max_sl_pct'])

        if 'min_sl_pct' not in merged or merged['min_sl_pct'] is None:
            merged['min_sl_pct'] = min(
                merged['max_sl_pct'],
                self.DEFAULT_VOLATILITY_STOP_CFG['min_sl_pct'],
            )

        return merged
    
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
    
    def get_adaptive_rsi_threshold(self, market_regime: Dict, *, return_reason: bool = False):
        """Sniper-mode SELL signals only trigger on stretched pumps."""
        base_rsi = float(self.base_cfg.get('adaptive_rsi_base', 68.0))
        adapt_range = float(self.base_cfg.get('adaptive_rsi_range', 8.0))

        trend = market_regime.get('trend', 'neutral')
        momentum = market_regime.get('momentum', 'sideways')
        threshold = base_rsi
        adjustment_reason = None

        if trend == 'bearish':
            if momentum == 'strong':
                threshold = base_rsi - 3.0
                adjustment_reason = "Bearish trend with strong momentum: lowering RSI threshold"
            else:
                threshold = base_rsi
        elif trend == 'bullish':
            if momentum == 'strong':
                threshold = base_rsi + 7.0  # Target ~75
                adjustment_reason = "Bullish trend with strong momentum: demanding higher RSI"
            else:
                threshold = base_rsi + 3.0  # Target ~71
                adjustment_reason = "Bullish trend: raising RSI threshold"

        min_threshold = max(60, base_rsi - adapt_range)
        max_threshold = min(90, base_rsi + adapt_range)
        clamped_threshold = max(min_threshold, min(max_threshold, threshold))

        if adjustment_reason is None and clamped_threshold != base_rsi:
            adjustment_reason = "Adaptive RSI clamp applied"

        if not return_reason:
            return clamped_threshold

        return clamped_threshold, {
            'base_threshold': base_rsi,
            'reason': adjustment_reason,
            'trend': trend,
            'momentum': momentum
        }
    
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
    
    def adapt_ema_requirements(self, trend_strength: float) -> Dict[str, Any]:
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

    def _apply_volatility_stop_tuning(self, volatility: str, base_sl_pct: float, atr_pct: float) -> Tuple[float, Dict[str, float]]:
        cfg = self.volatility_stop_cfg or {}
        if not cfg.get('enabled', False):
            return base_sl_pct, {}

        overrides = cfg.get('overrides', {})
        vol_cfg = overrides.get(volatility, {})

        atr_scale = float(vol_cfg.get('atr_scale', 1.0))
        tuned_sl_pct = base_sl_pct * atr_scale

        min_sl_pct = float(vol_cfg.get('min_sl_pct', cfg.get('min_sl_pct', base_sl_pct)))
        max_sl_pct = float(vol_cfg.get('max_sl_pct', cfg.get('max_sl_pct', base_sl_pct)))

        strategy_cap = float(self.strategy_config.get('max_sl_pct', max_sl_pct))
        max_sl_pct = min(max_sl_pct, strategy_cap)

        tuned_sl_pct = max(min_sl_pct, min(max_sl_pct, tuned_sl_pct))

        metadata = {
            'volatility': volatility,
            'atr_pct': atr_pct,
            'base_sl_pct': base_sl_pct,
            'min_sl_pct': min_sl_pct,
            'max_sl_pct': max_sl_pct,
            'applied_scale': atr_scale,
            'final_sl_pct': tuned_sl_pct,
        }
        return tuned_sl_pct, metadata

    def _resolve_mtf_missing_action(self, cfg: Dict, key: str, required: bool) -> str:
        on_missing = str(cfg.get(key, 'skip')).strip().lower()
        if on_missing not in ('skip', 'reject'):
            on_missing = 'skip'
        return 'reject' if required else on_missing

    def _mtf_missing_result(self, action: str, reason: str, code: str, meta: Dict, required: bool) -> Tuple[bool, str, Dict]:
        meta['status'] = 'missing'
        meta['action'] = action
        if required and action == 'reject':
            reason = f"{reason} (required)"
        else:
            reason = f"{reason} (policy={action})"
        if action == 'skip':
            meta['skipped'] = True
            return True, reason, meta
        meta['code'] = code
        return False, reason, meta

    def _get_mtf_min_bars(self, cfg: Dict, key: str) -> int:
        cfg_key = f"min_bars_{key}"
        default = self.DEFAULT_MTF_MIN_BARS.get(key, 0)
        try:
            value = int(cfg.get(cfg_key, default))
        except (TypeError, ValueError):
            value = default
        return max(0, value)

    def _mtf_min_bars_for_indicators(self, cfg: Dict, indicators: set) -> int:
        min_bars = 0
        for indicator in indicators:
            min_bars = max(min_bars, self._get_mtf_min_bars(cfg, indicator))
        return min_bars

    def _get_df_last_timestamp(self, df: pd.DataFrame):
        if df is None or df.empty:
            return None
        if 'timestamp' in df.columns:
            return df['timestamp'].iloc[-1]
        if df.index is not None and len(df.index) > 0:
            return df.index[-1]
        return None

    def _inc_mtf_telemetry(self, key: str, amount: int = 1) -> None:
        self._mtf_telemetry[key] = self._mtf_telemetry.get(key, 0) + amount

    def _apply_mtf_indicator_fallback(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        need_rsi: bool = False,
        ema_periods: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        if 'close' not in df.columns:
            return df

        ema_periods = ema_periods or []
        required_cols = []
        if need_rsi:
            required_cols.append('rsi')
        for period in ema_periods:
            required_cols.append(f"ema{period}")
        if not required_cols:
            return df

        last_ts = self._get_df_last_timestamp(df)
        cache_key = (
            symbol,
            timeframe,
            last_ts,
            len(df),
            14,
            tuple(sorted(ema_periods)),
        )

        cached = self._mtf_indicator_cache.get(cache_key)
        if cached:
            self._inc_mtf_telemetry(f"mtf_{timeframe}_cache_hit")
            self._mtf_indicator_cache.move_to_end(cache_key)
            df_local = df.copy()
            for col, series in cached.items():
                if col not in df_local.columns:
                    df_local[col] = series
            logger.debug(
                "[%s-MTF] %s %s fallback cache hit (len=%s, last_ts=%s)",
                self.strategy_name.upper(),
                symbol,
                timeframe,
                len(df),
                last_ts,
            )
            return df_local

        df_local = df.copy()
        computed = {}
        if need_rsi and 'rsi' not in df_local.columns:
            df_local['rsi'] = calc_rsi(df_local['close'], period=14)
            computed['rsi'] = df_local['rsi']
        for period in ema_periods:
            col = f"ema{period}"
            if col not in df_local.columns:
                df_local[col] = calc_ema(df_local['close'], period=period)
                computed[col] = df_local[col]

        if computed:
            self._mtf_indicator_cache[cache_key] = computed
            self._mtf_indicator_cache.move_to_end(cache_key)
            while len(self._mtf_indicator_cache) > self._mtf_cache_limit:
                self._mtf_indicator_cache.popitem(last=False)
            self._inc_mtf_telemetry(f"mtf_{timeframe}_fallback_computed")
            logger.debug(
                "[%s-MTF] %s %s fallback computed (len=%s, last_ts=%s, cols=%s)",
                self.strategy_name.upper(),
                symbol,
                timeframe,
                len(df),
                last_ts,
                sorted(computed.keys()),
            )

        return df_local

    def _mtf_confirm_15m(self, df_15m: pd.DataFrame, symbol: str, cfg: Dict) -> Tuple[bool, str, Dict]:
        require_15m = bool(cfg.get('require_15m', False))
        action = self._resolve_mtf_missing_action(cfg, 'on_missing_15m', require_15m)
        meta = {
            'timeframe': '15m',
            'required': require_15m,
            'on_missing': action,
        }

        if df_15m is None or not isinstance(df_15m, pd.DataFrame) or df_15m.empty:
            return self._mtf_missing_result(action, 'missing_15m_data', 'mtf_15m_missing', meta, require_15m)

        try:
            rsi_min = float(cfg.get('rsi_15m_min', 62.0))
        except (TypeError, ValueError):
            rsi_min = 62.0

        try:
            min_ext_pct = float(cfg.get('min_15m_close_over_ema50_pct', 0.0) or 0.0)
        except (TypeError, ValueError):
            min_ext_pct = 0.0

        use_extension = min_ext_pct > 0.0
        required_cols = ['close', 'rsi']
        if use_extension:
            required_cols.append('ema50')

        missing_cols = [col for col in required_cols if col not in df_15m.columns]
        if missing_cols:
            fallback_indicators = set()
            if 'rsi' in missing_cols:
                fallback_indicators.add('rsi')
            if 'ema50' in missing_cols:
                fallback_indicators.add('ema50')

            if 'close' in missing_cols or not fallback_indicators:
                return self._mtf_missing_result(
                    action,
                    f"missing_15m_columns ({', '.join(missing_cols)})",
                    'mtf_15m_missing',
                    meta,
                    require_15m,
                )

            self._inc_mtf_telemetry('mtf_15m_fallback_attempted')
            min_required = self._mtf_min_bars_for_indicators(cfg, fallback_indicators)
            if len(df_15m) < min_required:
                self._inc_mtf_telemetry('mtf_15m_fallback_skipped_insufficient_bars')
                meta['code'] = 'mtf_15m_insufficient_bars'
                logger.info(
                    "[%s-MTF] %s 15m fallback skipped (len=%s < min=%s)",
                    self.strategy_name.upper(),
                    symbol,
                    len(df_15m),
                    min_required,
                )
                return self._mtf_missing_result(
                    action,
                    f"insufficient_bars (len={len(df_15m)} < min={min_required})",
                    'mtf_15m_insufficient_bars',
                    meta,
                    require_15m,
                )

            df_15m = self._apply_mtf_indicator_fallback(
                df_15m,
                symbol=symbol,
                timeframe='15m',
                need_rsi=('rsi' in missing_cols),
                ema_periods=[50] if 'ema50' in missing_cols else [],
            )

            missing_cols = [col for col in required_cols if col not in df_15m.columns]
            if missing_cols:
                return self._mtf_missing_result(
                    action,
                    f"missing_15m_columns ({', '.join(missing_cols)})",
                    'mtf_15m_missing',
                    meta,
                    require_15m,
                )

        df_valid = df_15m.dropna(subset=required_cols)
        if df_valid.empty:
            return self._mtf_missing_result(action, 'missing_15m_values', 'mtf_15m_missing', meta, require_15m)

        last = df_valid.iloc[-1]
        close = float(last['close'])
        rsi_val = float(last['rsi'])
        meta.update({
            'rsi_15m': rsi_val,
            'rsi_15m_min': rsi_min,
            'close_15m': close,
        })

        if rsi_val < rsi_min:
            meta['code'] = 'mtf_15m_rsi'
            return False, f"rsi_15m_below_min (rsi={rsi_val:.2f}, min={rsi_min:.2f})", meta

        if use_extension:
            ema50 = float(last['ema50'])
            if ema50 <= 0:
                return self._mtf_missing_result(action, 'invalid_15m_ema50', 'mtf_15m_missing', meta, require_15m)
            close_over = (close / ema50) - 1.0
            meta.update({
                'ema50_15m': ema50,
                'close_over_ema50_pct': close_over,
                'min_15m_close_over_ema50_pct': min_ext_pct,
            })
            if close < ema50 * (1.0 + min_ext_pct):
                meta['code'] = 'mtf_15m_extension'
                return False, (
                    f"close_not_extended_over_ema50 (close={close:.2f}, ema50={ema50:.2f}, min_pct={min_ext_pct:.4f})"
                ), meta

        meta['status'] = 'passed'
        return True, 'passed', meta

    def _mtf_confirm_1h(self, df_1h: pd.DataFrame, symbol: str, cfg: Dict) -> Tuple[bool, str, Dict]:
        require_1h = bool(cfg.get('require_1h', False))
        action = self._resolve_mtf_missing_action(cfg, 'on_missing_1h', require_1h)
        meta = {
            'timeframe': '1h',
            'required': require_1h,
            'on_missing': action,
        }

        if df_1h is None or not isinstance(df_1h, pd.DataFrame) or df_1h.empty:
            return self._mtf_missing_result(action, 'missing_1h_data', 'mtf_1h_missing', meta, require_1h)

        require_ema_stack = bool(cfg.get('require_1h_bearish_ema_stack', True))
        raw_rsi_max = cfg.get('rsi_1h_max', 60.0)
        rsi_max = None
        if raw_rsi_max is not None:
            try:
                rsi_max = float(raw_rsi_max)
            except (TypeError, ValueError):
                rsi_max = 60.0

        required_cols = []
        if require_ema_stack:
            required_cols.extend(['ema21', 'ema50', 'ema200'])
        if rsi_max is not None:
            required_cols.append('rsi')

        if required_cols:
            missing_cols = [col for col in required_cols if col not in df_1h.columns]
            if missing_cols:
                fallback_indicators = set()
                if 'rsi' in missing_cols:
                    fallback_indicators.add('rsi')
                if 'ema21' in missing_cols:
                    fallback_indicators.add('ema21')
                if 'ema50' in missing_cols:
                    fallback_indicators.add('ema50')
                if 'ema200' in missing_cols:
                    fallback_indicators.add('ema200')

                if 'close' in missing_cols or not fallback_indicators:
                    return self._mtf_missing_result(
                        action,
                        f"missing_1h_columns ({', '.join(missing_cols)})",
                        'mtf_1h_missing',
                        meta,
                        require_1h,
                    )

                self._inc_mtf_telemetry('mtf_1h_fallback_attempted')
                min_required = self._mtf_min_bars_for_indicators(cfg, fallback_indicators)
                if len(df_1h) < min_required:
                    self._inc_mtf_telemetry('mtf_1h_fallback_skipped_insufficient_bars')
                    meta['code'] = 'mtf_1h_insufficient_bars'
                    logger.info(
                        "[%s-MTF] %s 1h fallback skipped (len=%s < min=%s)",
                        self.strategy_name.upper(),
                        symbol,
                        len(df_1h),
                        min_required,
                    )
                    return self._mtf_missing_result(
                        action,
                        f"insufficient_bars (len={len(df_1h)} < min={min_required})",
                        'mtf_1h_insufficient_bars',
                        meta,
                        require_1h,
                    )

                df_1h = self._apply_mtf_indicator_fallback(
                    df_1h,
                    symbol=symbol,
                    timeframe='1h',
                    need_rsi=('rsi' in missing_cols),
                    ema_periods=[p for p in (21, 50, 200) if f"ema{p}" in missing_cols],
                )

                missing_cols = [col for col in required_cols if col not in df_1h.columns]
                if missing_cols:
                    return self._mtf_missing_result(
                        action,
                        f"missing_1h_columns ({', '.join(missing_cols)})",
                        'mtf_1h_missing',
                        meta,
                        require_1h,
                    )
            df_valid = df_1h.dropna(subset=required_cols)
        else:
            df_valid = df_1h.dropna()

        if df_valid.empty:
            return self._mtf_missing_result(action, 'missing_1h_values', 'mtf_1h_missing', meta, require_1h)

        last = df_valid.iloc[-1]

        if require_ema_stack:
            ema21 = float(last['ema21'])
            ema50 = float(last['ema50'])
            ema200 = float(last['ema200'])
            ema_stack_ok = ema21 < ema50 <= ema200
            meta.update({
                'ema21_1h': ema21,
                'ema50_1h': ema50,
                'ema200_1h': ema200,
                'ema_stack_ok': ema_stack_ok,
            })
            if not ema_stack_ok:
                meta['code'] = 'mtf_1h_ema'
                return False, (
                    f"ema_stack_not_bearish (ema21={ema21:.2f}, ema50={ema50:.2f}, ema200={ema200:.2f})"
                ), meta

        if rsi_max is not None:
            rsi_val = float(last['rsi'])
            meta.update({
                'rsi_1h': rsi_val,
                'rsi_1h_max': rsi_max,
            })
            if rsi_val > rsi_max:
                meta['code'] = 'mtf_1h_rsi'
                return False, f"rsi_1h_above_max (rsi={rsi_val:.2f}, max={rsi_max:.2f})", meta

        meta['status'] = 'passed'
        return True, 'passed', meta
    
    def signal(self, df_30m: pd.DataFrame, 
               df_1h: pd.DataFrame = None,
               regime_data: Optional[Dict] = None,
               symbol: str = None,
               market_data: Optional[Dict] = None,
               ml_context=None) -> Optional[Dict]:
        """
        Generate adaptive trading signal for a short position based on market regime and ML insights.
        Logs the specific reason if no signal is generated for transparency.
        """
        symbol_display = symbol or "UNKNOWN"
        log_prefix = f"[{self.strategy_name.upper()}/{symbol_display}]"
        # Volume logic centralized in StrategyCoordinator (Issue #450)

        # --- Data Validation Step ---
        validation_passed, reason = self._validate_input_data(df_30m, df_1h, regime_data, symbol_display)
        if not validation_passed:
            logger.info(f"🚫 {log_prefix} No Signal: {reason}")
            return None

        try:
            last30 = df_30m.dropna().iloc[-1]
        except IndexError:
            logger.info(f"🚫 {log_prefix} No Signal: Insufficient 30m data to generate a signal.")
            return None

        if regime_data is None:
            regime_data = DEFAULT_MARKET_REGIME.copy()

        try:
            signal = {
                "strategy_name": self.strategy_name,
                "side": "sell",
                "symbol": symbol,
                "features": {}
            }
            # --- Initial Data Extraction ---
            if 'rsi' not in last30.index or 'close' not in last30.index:
                logger.warning(f"🚫 {log_prefix} No Signal: 'rsi' or 'close' column missing in the latest data.")
                return None
            
            close_price = float(last30['close'])
            rsi_val = float(last30['rsi'])
            
            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }
            
            # --- Adaptive Threshold Calculation ---
            adaptive_rsi_threshold = self.get_symbol_specific_threshold(symbol_display)
            if adaptive_rsi_threshold is not None:
                base_threshold = float(self.base_cfg.get('adaptive_rsi_base', 68.0))
                logger.info(
                    f"⚠️ {log_prefix} Symbol-specific RSI threshold active: base {base_threshold:.2f} → {adaptive_rsi_threshold:.2f}."
                )
            else:
                adaptive_rsi_threshold, threshold_meta = self.get_adaptive_rsi_threshold(
                    market_regime,
                    return_reason=True
                )
                meta_base = threshold_meta.get('base_threshold', adaptive_rsi_threshold)
                reason = threshold_meta.get('reason')
                if reason and abs(adaptive_rsi_threshold - meta_base) >= 0.1:
                    logger.info(
                        f"⚠️ {log_prefix} {reason} (base {meta_base:.2f} → {adaptive_rsi_threshold:.2f})."
                    )
                elif abs(adaptive_rsi_threshold - meta_base) >= 0.1:
                    logger.info(
                        f"⚠️ {log_prefix} Adaptive RSI adjustment applied (base {meta_base:.2f} → {adaptive_rsi_threshold:.2f})."
                    )
                else:
                    logger.info(
                        f"ℹ️ {log_prefix} RSI threshold steady at {adaptive_rsi_threshold:.2f} (trend={market_regime['trend']}, momentum={market_regime['momentum']})."
                    )

            # --- Core Signal Condition Checks with Tracing ---
            if self.debug_logging:
                # Calculate EMA alignment for debug log
                ema21 = float(last30.get('ema21', 0))
                ema50 = float(last30.get('ema50', 0))
                ema200 = float(last30.get('ema200', 0))
                ema_aligned = (ema21 < ema50 <= ema200)
                
                atr_val = float(last30.get('atr', 0))
                vol = float(last30.get('volume', 0))
                
                logger.info(f"🔍 [STR-DEBUG] {symbol_display}")
                logger.info(f"   RSI: {rsi_val:.2f} (threshold: {adaptive_rsi_threshold:.2f})")
                logger.info(f"   EMA Align: {ema_aligned} (21={ema21:.1f}, 50={ema50:.1f}, 200={ema200:.1f})")
                logger.info(f"   Volume: {vol:.2f}")
                logger.info(f"   ATR: {atr_val:.4f}")
                logger.info(f"   Regime: {market_regime['trend']}, Volatility: {market_regime['volatility']}")

            # 1. RSI Condition Check
            if rsi_val < adaptive_rsi_threshold:
                logger.info(f"🚫 {log_prefix} No Signal: RSI ({rsi_val:.2f}) is below the threshold ({adaptive_rsi_threshold:.2f}).")
                return None

            # 2. Dynamic Rip Check (Replaces strict EMA alignment)
            # "Rip" is defined as price extending significantly above a Long-Term EMA
            
            # Get Long-Term EMA (using EMA 50 as proxy for trend baseline)
            long_ema_value = float(last30.get('ema50', 0))
            atr_value = float(last30.get('atr', 0))
            
            if long_ema_value > 0 and atr_value > 0:
                # Determine ATR Multiplier based on volatility
                volatility_regime = market_regime.get('volatility', 'normal')
                
                if volatility_regime == 'high':
                    atr_multiplier = 1.5  # High vol: require more aggressive extension
                elif volatility_regime == 'low':
                    atr_multiplier = 0.75 # Low vol: smaller extension is enough
                else:
                    atr_multiplier = 1.0  # Normal
                
                # Calculate Rip Threshold
                rip_value = atr_value * atr_multiplier
                required_price_threshold = long_ema_value + rip_value
                
                if close_price < required_price_threshold:
                    logger.info(
                        f"🚫 {log_prefix} No Signal: Rip Check Failed. Price ${close_price:,.2f} "
                        f"is not above EMA50+Rip (${required_price_threshold:,.2f}). "
                        f"(EMA50=${long_ema_value:,.2f}, Rip=${rip_value:,.2f}, Vol={volatility_regime})"
                    )
                    return None
                else:
                    if self.debug_logging:
                        logger.info(
                            f"✅ {log_prefix} Rip Check Passed: Price ${close_price:,.2f} > "
                            f"Threshold ${required_price_threshold:,.2f} (Rip=${rip_value:,.2f})"
                        )
            else:
                # Fallback if indicators missing (should be caught by validation, but safe guard)
                logger.warning(f"⚠️ {log_prefix} Missing EMA50 or ATR for Rip Check. Skipping.")

            # --- Optional Multi-Timeframe (MTF) Confirmation ---
            mtf_cfg = self.strategy_config.get("mtf_confirmation", {}) or {}
            if isinstance(mtf_cfg, dict) and mtf_cfg.get("enabled", False):
                df_15m = None
                if market_data:
                    df_15m = market_data.get('15m')

                df_1h_local = df_1h
                if (df_1h_local is None or (hasattr(df_1h_local, 'empty') and df_1h_local.empty)) and market_data:
                    df_1h_local = market_data.get("df_1h") or market_data.get("1h")

                passed_15m, reason_15m, meta_15m = self._mtf_confirm_15m(df_15m, symbol_display, mtf_cfg)
                signal.setdefault("features", {})["mtf_15m"] = meta_15m
                if not passed_15m:
                    code = meta_15m.get("code", "mtf_15m_block")
                    logger.info(f"?? {log_prefix} No Signal: MTF-15m block: {reason_15m}. [code={code}]")
                    return None

                passed_1h, reason_1h, meta_1h = self._mtf_confirm_1h(df_1h_local, symbol_display, mtf_cfg)
                signal.setdefault("features", {})["mtf_1h"] = meta_1h
                if not passed_1h:
                    code = meta_1h.get("code", "mtf_1h_block")
                    logger.info(f"?? {log_prefix} No Signal: MTF-1h block: {reason_1h}. [code={code}]")
                    return None
            
            logger.info(f"✅ {log_prefix} Base conditions met. Proceeding to ML & Risk checks.")

            # --- ML-Aware Decision Making ---
            position_size_modifier = 1.0
            signal["rsi"] = float(rsi_val)
            signal.setdefault("features", {})["rsi"] = float(rsi_val)
            ml_enhanced = False
            MIN_ML_CONFIDENCE_THRESHOLD = 0.60

            if ml_context and ml_context.get('is_healthy', False) and ml_context.get('regime_confidence', 0) >= MIN_ML_CONFIDENCE_THRESHOLD:
                ml_enhanced = True
                
                if ml_context.get('regime_prediction') == 'bullish' and ml_context.get('regime_confidence', 0) > 0.7:
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong bullish regime detected (confidence: {ml_context.get('regime_confidence', 0):.2%}).")
                    return None
                
                if ml_context.get('price_direction') == 'up' and ml_context.get('price_confidence', 0) > 0.7:
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong price up prediction (confidence: {ml_context.get('price_confidence', 0):.2%}).")
                    return None
                
                if (ml_context.get('regime_prediction') == 'bearish') or (ml_context.get('price_direction') == 'down'):
                    position_size_modifier = 1.0 + (0.25 * ml_context.get('consensus_score', 0))
                    if self.debug_logging: logger.info(f"🧠 {log_prefix} ML Confirmation: Increasing position size modifier to {position_size_modifier:.2f}x.")
                elif ml_context.get('consensus_score', 1.0) < 0.5:
                    position_size_modifier *= 0.75
                    if self.debug_logging: logger.info(f"🧠 {log_prefix} ML Caution: Reducing position size modifier to {position_size_modifier:.2f}x.")

            # --- Final Risk/Reward and Position Sizing ---
            volatility = regime_data.get('volatility', 'normal')
            position_mult = self.calculate_dynamic_position_size(volatility) * position_size_modifier
            
            entry_price = float(last30['close'])
            atr_value = float(last30.get('atr', entry_price * 0.02))
            atr_pct = (atr_value / entry_price) if entry_price else 0.0
            
            # 🔥 GÜNCELLEME: Config okumaları artık tutarlı bir şekilde `self.strategy_config` üzerinden yapılıyor.
            tp_atr_mult = float(self.strategy_config.get("tp_atr_mult", 3.0))
            sl_atr_mult = float(self.strategy_config.get("sl_atr_mult", 1.5))
            
            # Calculate theoretical ATR-based levels
            theoretical_sl_distance = atr_value * sl_atr_mult
            theoretical_tp_distance = atr_value * tp_atr_mult
            
            theoretical_sl_pct = theoretical_sl_distance / entry_price
            theoretical_tp_pct = theoretical_tp_distance / entry_price
            
            min_tp_pct = float(self.strategy_config.get("min_tp_pct", 0.010))
            max_sl_pct = float(self.strategy_config.get("max_sl_pct", 0.020))
            
            intended_rr = tp_atr_mult / sl_atr_mult if sl_atr_mult else self.min_rr_ratio

            # Apply stop-loss cap CORRECTLY and realign TP if needed
            if theoretical_sl_pct > max_sl_pct:
                logger.info(f"📊 {log_prefix} [SL Cap Applied] {theoretical_sl_pct:.1%} → {max_sl_pct:.1%}")
                actual_sl_pct = max_sl_pct
            else:
                actual_sl_pct = theoretical_sl_pct

            volatility_stop_meta = None
            if self.volatility_stop_cfg.get('enabled', False):
                tuned_sl_pct, volatility_stop_meta = self._apply_volatility_stop_tuning(
                    volatility,
                    actual_sl_pct,
                    atr_pct,
                )
                if abs(tuned_sl_pct - actual_sl_pct) >= 1e-6:
                    logger.info(
                        f"⚙️ {log_prefix} [VolStop] {volatility.capitalize()} volatility adjusted SL {actual_sl_pct:.1%} → {tuned_sl_pct:.1%}."
                    )
                actual_sl_pct = tuned_sl_pct

            scaled_tp_pct = actual_sl_pct * intended_rr
            actual_tp_pct = max(min_tp_pct, min(theoretical_tp_pct, scaled_tp_pct))
            
            # SHORT: Stop is ABOVE entry
            stop_price = entry_price * (1 + actual_sl_pct)
            
            # ✅ CRITICAL FIX: Use min() for SHORT to prevent stop going too high
            # This safety net ensures we never exceed max_sl_pct even with rounding errors
            # or edge cases in the logic above (defensive programming for financial systems)
            stop_price = min(stop_price, entry_price * (1 + max_sl_pct))
            
            # SHORT: Target is BELOW entry
            target_price = entry_price * (1 - actual_tp_pct)
            target_price = min(target_price, entry_price * (1 - min_tp_pct))
            
            # Calculate final R/R
            risk_amount = stop_price - entry_price
            reward_amount = entry_price - target_price
            rr_ratio = (reward_amount / risk_amount) if risk_amount > 0 else float('inf')

            # Enhanced logging with R/R details
            logger.info(
                f"🔎 {log_prefix} [STR R/R] SHORT Entry=${entry_price:.2f}, "
                f"Stop=${stop_price:.2f} (+{actual_sl_pct:.1%}), "
                f"Target=${target_price:.2f} (-{actual_tp_pct:.1%}), "
                f"R/R={rr_ratio:.2f}"
            )    

            # 3. R/R Ratio Check
            # 🔥 GÜNCELLEME: R/R kontrolü artık __init__ içinde ayarlanan `self.min_rr_ratio` özelliğini kullanıyor.
            if rr_ratio < self.min_rr_ratio:
                logger.info(f"🚫 {log_prefix} No Signal: Calculated R/R Ratio ({rr_ratio:.2f}) is below the minimum required ({self.min_rr_ratio}).")
                return None

            # --- Signal Generation ---
            logger.info(f"✅ {log_prefix} All checks passed. Generating SELL signal.")
            
            signal.update({
                "entry": entry_price,
                "stop": stop_price,
                "target": target_price,
                "reason": f"Adaptive RSI {rsi_val:.1f} >= {adaptive_rsi_threshold:.1f}",
                "rr_ratio": rr_ratio,
                "is_adaptive": True,
                "position_multiplier": position_mult,
                "ml_enhanced": ml_enhanced,
                "strategy_type": 'adaptive',
                "strategy_min_rr": self.min_rr_ratio,
            })
            if volatility_stop_meta:
                signal['volatility_stop_meta'] = volatility_stop_meta
            
            if ml_enhanced:
                signal['ml_consensus'] = ml_context.get('consensus_score')
                signal['ml_position_modifier'] = position_size_modifier
            
            return signal
            
        except Exception as e:
            logger.error(f"💥 {log_prefix} Critical error during signal generation: {e}", exc_info=True)
            # Fallback logic remains unchanged
            try:
                if hasattr(super(), 'signal'):
                    base_signal = super().signal(df_30m, df_1h)
                    if base_signal:
                        base_signal.update({'strategy_type': 'base_fallback', 'fallback_reason': str(e), 'symbol': symbol})
                        logger.warning(f"⚠️ {log_prefix} Fallback to base strategy successful.")
                        return base_signal
            except Exception as fallback_error:
                logger.error(f"💥 {log_prefix} Fallback to base strategy also failed: {fallback_error}")
                
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
