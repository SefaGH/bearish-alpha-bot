"""
Adaptive ShortTheRip strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
import math
import json
from collections import OrderedDict
from typing import Optional, Dict, Tuple, ClassVar, Any
from .short_the_rip import ShortTheRip
from config.mtf_policy import MtfConfirmationConfig, MtfTimeframePolicy, MtfMinBars
from core.indicators import rsi as calc_rsi, ema as calc_ema
from core.strategy_shadow_eval import (
    shadow_enabled,
    extract_last_closed_ts_ms,
    extract_df_meta,
    emit_shadow_log,
)

# Default market regime for fallback
DEFAULT_MARKET_REGIME = {
    'trend': 'neutral',
    'momentum': 'sideways', 
    'volatility': 'normal',
    'trend_strength': 0.0,
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
        self._last_hybrid_meta_state = None
        
        # Minimum R/R oranını başlangıçta, bir kez olmak üzere config'den oku.
        # `super()` çağrısı `self.strategy_config`'i oluşturduğu için artık bunu güvenle kullanabiliriz.
        self.min_rr_ratio = self.strategy_config.get('min_rr_ratio', 1.2)
        logger.info(f"[{self.strategy_name.upper()}] Minimum R/R Ratio initialized to: {self.min_rr_ratio}")
        if not hasattr(self, "signal"):
            self.signal = self._default_signal_wrapper  # type: ignore
        assert callable(getattr(self, "signal", None)), f"{self.strategy_name}: signal method not callable"

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
        self._guard_telemetry = {
            "guard_rollover_defer_count": 0,
            "guard_rollover_skip_count": 0,
        }
        self._mtf_policy: Optional[MtfConfirmationConfig] = None
        mtf_effective = self.strategy_config.get("mtf_confirmation_effective")
        if isinstance(mtf_effective, MtfConfirmationConfig):
            self._mtf_policy = mtf_effective
        elif isinstance(self.strategy_config.get("mtf_confirmation"), MtfConfirmationConfig):
            self._mtf_policy = self.strategy_config.get("mtf_confirmation")

    def _inc_guard_telemetry(self, key: str, amount: int = 1) -> None:
        try:
            amount = int(amount)
        except Exception:
            amount = 1
        try:
            current = int(self._guard_telemetry.get(key, 0))
        except Exception:
            current = 0
        self._guard_telemetry[key] = current + amount

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
        trend_strength_raw = market_regime.get('trend_strength', market_regime.get('adx', 0.0))
        try:
            trend_strength = float(trend_strength_raw or 0.0)
        except Exception:
            trend_strength = 0.0
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

        strong_trend = trend_strength > 30.0
        strong_trend_floor = 85.0
        if strong_trend:
            threshold = max(threshold, strong_trend_floor)
            adjustment_reason = "Strong trend override: forcing higher RSI threshold"

        min_threshold = max(60, base_rsi - adapt_range)
        if strong_trend:
            max_threshold = 90.0
        else:
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
            'momentum': momentum,
            'trend_strength': trend_strength,
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

    def _mtf_missing_result(self, action: str, reason: str, code: str, meta: Dict) -> Tuple[bool, str, Dict]:
        meta['status'] = 'missing'
        meta['action'] = action
        reason = f"{reason} (policy={action})"
        if action == 'skip':
            meta['skipped'] = True
            return True, reason, meta
        meta['code'] = code
        return False, reason, meta

    def _mtf_min_bars_for_indicators(self, min_bars: MtfMinBars, indicators: set) -> int:
        min_bars_map = min_bars.as_dict()
        min_required = 0
        for indicator in indicators:
            min_required = max(min_required, min_bars_map.get(indicator, 0))
        return min_required

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

    def _mtf_confirm_15m(
        self,
        df_15m: pd.DataFrame,
        symbol: str,
        policy: MtfTimeframePolicy,
        min_bars: MtfMinBars,
    ) -> Tuple[bool, str, Dict]:
        action = policy.missing_policy
        meta = {
            'timeframe': '15m',
            'mode': policy.mode,
            'on_missing': policy.on_missing,
            'missing_policy': action,
            'missing_is_fatal': policy.missing_is_fatal,
        }

        if df_15m is None or not isinstance(df_15m, pd.DataFrame) or df_15m.empty:
            return self._mtf_missing_result(action, 'missing_15m_data', 'mtf_15m_missing', meta)

        rsi_min = policy.rsi_min
        min_ext_pct = policy.min_close_over_ema50_pct

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
                )

            self._inc_mtf_telemetry('mtf_15m_fallback_attempted')
            min_required = self._mtf_min_bars_for_indicators(min_bars, fallback_indicators)
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
                )

        df_valid = df_15m.dropna(subset=required_cols)
        if df_valid.empty:
            return self._mtf_missing_result(action, 'missing_15m_values', 'mtf_15m_missing', meta)

        last = df_valid.iloc[-1]
        close = float(last['close'])
        rsi_val = float(last['rsi'])
        meta.update({
            'rsi_15m': rsi_val,
            'rsi_15m_min': rsi_min,
            'close_15m': close,
        })

        if rsi_val < rsi_min:
            meta['status'] = 'failed'
            meta['code'] = 'mtf_15m_rsi'
            return False, f"rsi_15m_below_min (rsi={rsi_val:.2f}, min={rsi_min:.2f})", meta

        if use_extension:
            ema50 = float(last['ema50'])
            if ema50 <= 0:
                return self._mtf_missing_result(action, 'invalid_15m_ema50', 'mtf_15m_missing', meta)
            close_over = (close / ema50) - 1.0
            meta.update({
                'ema50_15m': ema50,
                'close_over_ema50_pct': close_over,
                'min_15m_close_over_ema50_pct': min_ext_pct,
            })
            if close < ema50 * (1.0 + min_ext_pct):
                meta['status'] = 'failed'
                meta['code'] = 'mtf_15m_extension'
                return False, (
                    f"close_not_extended_over_ema50 (close={close:.2f}, ema50={ema50:.2f}, min_pct={min_ext_pct:.4f})"
                ), meta

        meta['status'] = 'passed'
        return True, 'passed', meta

    def _mtf_confirm_1h(
        self,
        df_1h: pd.DataFrame,
        symbol: str,
        policy: MtfTimeframePolicy,
        min_bars: MtfMinBars,
    ) -> Tuple[bool, str, Dict]:
        action = policy.missing_policy
        meta = {
            'timeframe': '1h',
            'mode': policy.mode,
            'on_missing': policy.on_missing,
            'missing_policy': action,
            'missing_is_fatal': policy.missing_is_fatal,
        }

        if df_1h is None or not isinstance(df_1h, pd.DataFrame) or df_1h.empty:
            return self._mtf_missing_result(action, 'missing_1h_data', 'mtf_1h_missing', meta)

        require_ema_stack = policy.require_bearish_ema_stack
        rsi_max = policy.rsi_max

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
                    )

                self._inc_mtf_telemetry('mtf_1h_fallback_attempted')
                min_required = self._mtf_min_bars_for_indicators(min_bars, fallback_indicators)
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
                    )
            df_valid = df_1h.dropna(subset=required_cols)
        else:
            df_valid = df_1h.dropna()

        if df_valid.empty:
            return self._mtf_missing_result(action, 'missing_1h_values', 'mtf_1h_missing', meta)

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
                meta['status'] = 'failed'
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
                meta['status'] = 'failed'
                meta['code'] = 'mtf_1h_rsi'
                return False, f"rsi_1h_above_max (rsi={rsi_val:.2f}, max={rsi_max:.2f})", meta

        meta['status'] = 'passed'
        return True, 'passed', meta

    def _check_extreme_bypass_mtf_hard_veto(
        self,
        *,
        signal: Dict[str, Any],
        df_eval: pd.DataFrame,
        close_price: float,
        atr_value: float,
        rsi_value: float,
        timeframe: str,
        failure_kind: str,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Optional (opt-in) bypass for MTF hard veto.

        Note: The existing coordinator-level "extreme bypass" (signals.bypass.*)
        runs *after* a strategy returns a signal. If the strategy returns `None`
        due to an MTF hard veto, downstream bypass logic is unreachable.
        """
        if failure_kind == "missing-data":
            return False, None, {}

        mtf_cfg = self.strategy_config.get("mtf_confirmation")
        if not isinstance(mtf_cfg, dict):
            mtf_cfg = self.base_cfg.get("mtf_confirmation")
        if not isinstance(mtf_cfg, dict):
            return False, None, {}

        bypass_cfg = mtf_cfg.get("extreme_bypass", {})
        if not isinstance(bypass_cfg, dict):
            return False, None, {}

        if not bool(bypass_cfg.get("enabled", False)):
            return False, None, {}

        def _as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        min_directional_move_pct = _as_float(bypass_cfg.get("min_directional_move_pct", 0.0), 0.0)
        min_abs_move_pct = _as_float(bypass_cfg.get("min_abs_move_pct", 0.0), 0.0)
        min_atr_pct = _as_float(bypass_cfg.get("min_atr_pct", 0.0), 0.0)
        rsi_oversold_threshold = _as_float(bypass_cfg.get("rsi_oversold_threshold", 0.0), 0.0)
        rsi_overbought_threshold = _as_float(bypass_cfg.get("rsi_overbought_threshold", 0.0), 0.0)

        side = (signal.get("side") or "").lower()

        prev_close = None
        try:
            if df_eval is not None and hasattr(df_eval, "__len__") and len(df_eval) >= 2 and "close" in df_eval.columns:
                prev_close = float(df_eval["close"].iloc[-2])
        except Exception:
            prev_close = None

        move_pct = None
        abs_move_pct = None
        directional_move_pct = None
        if prev_close is not None and prev_close > 0:
            try:
                move_pct = (close_price - prev_close) / prev_close
                abs_move_pct = abs(move_pct)
                if side in ("sell", "short"):
                    directional_move_pct = move_pct
                elif side in ("buy", "long"):
                    directional_move_pct = -move_pct
            except Exception:
                move_pct = None
                abs_move_pct = None
                directional_move_pct = None

        atr_pct = None
        try:
            atr_pct = (atr_value / close_price) if close_price else None
        except Exception:
            atr_pct = None

        # Safety gate: if ATR isn't present/finite in the evaluation frame, treat it as fallback
        # (the strategy may substitute close*0.02 upstream). We do not allow bypass decisions
        # to hinge on fallback ATR.
        atr_is_fallback = True
        try:
            if (
                df_eval is not None
                and hasattr(df_eval, "__len__")
                and len(df_eval) >= 1
                and "atr" in df_eval.columns
            ):
                last_atr = df_eval["atr"].iloc[-1]
                if last_atr is not None:
                    last_atr_f = float(last_atr)
                    if last_atr_f > 0 and math.isfinite(last_atr_f):
                        atr_is_fallback = False
        except Exception:
            atr_is_fallback = True

        def _meets_threshold(metric: Optional[float], threshold: float) -> bool:
            if threshold <= 0:
                return False
            if metric is None:
                return False
            try:
                return float(metric) >= float(threshold)
            except Exception:
                return False

        # Two-factor bypass: require ATR validity and at least 2 independent signals.
        # This avoids low-threshold directional blips (like ~0.4-0.5%) bypassing 1h hard veto.
        rsi_oversold_ok = (
            side in ("buy", "long")
            and rsi_oversold_threshold > 0
            and rsi_value <= rsi_oversold_threshold
        )
        rsi_overbought_ok = (
            side in ("sell", "short")
            and rsi_overbought_threshold > 0
            and rsi_value >= rsi_overbought_threshold
        )
        directional_ok = _meets_threshold(directional_move_pct, min_directional_move_pct)
        abs_ok = _meets_threshold(abs_move_pct, min_abs_move_pct)
        atr_ok = (not atr_is_fallback) and _meets_threshold(atr_pct, min_atr_pct)

        factor_hits = [
            ("rsi_oversold", rsi_oversold_ok),
            ("rsi_overbought", rsi_overbought_ok),
            ("directional_move_pct", directional_ok),
            ("abs_move_pct", abs_ok),
            ("atr_pct", atr_ok),
        ]
        hit_reasons = [name for name, ok in factor_hits if ok]

        triggered = False
        reason = None
        if atr_is_fallback:
            triggered = False
            reason = None
        elif len(hit_reasons) >= 2:
            triggered = True
            reason = "+".join(hit_reasons)

        meta = {
            "timeframe": timeframe,
            "side": side,
            "rsi": float(rsi_value),
            "prev_close": prev_close,
            "close": float(close_price),
            "move_pct": move_pct,
            "abs_move_pct": abs_move_pct,
            "directional_move_pct": directional_move_pct,
            "atr": float(atr_value),
            "atr_pct": atr_pct,
            "atr_is_fallback": atr_is_fallback,
            "thresholds": {
                "min_directional_move_pct": min_directional_move_pct,
                "min_abs_move_pct": min_abs_move_pct,
                "min_atr_pct": min_atr_pct,
                "rsi_oversold_threshold": rsi_oversold_threshold,
                "rsi_overbought_threshold": rsi_overbought_threshold,
            },
        }

        return triggered, reason, meta
    
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

        df_30m_closed = market_data.get("30m_closed") if market_data else None
        df_30m_hybrid = market_data.get("30m_hybrid") if market_data else None
        hybrid_df_provided = df_30m_hybrid is not None
        if df_30m_closed is None:
            df_30m_closed = df_30m
        if df_30m_hybrid is None:
            df_30m_hybrid = df_30m

        df_hybrid_attrs = getattr(df_30m_hybrid, "attrs", {}) or {}
        includes_forming = bool(df_hybrid_attrs.get("includes_forming", False))
        fallback_reason = df_hybrid_attrs.get("fallback_reason", None)
        if isinstance(fallback_reason, str) and fallback_reason.strip().lower() in ("none", ""):
            fallback_reason = None
        merge_action = df_hybrid_attrs.get("merge_action", "none")
        forming_ts = df_hybrid_attrs.get("forming_ts")
        forming_last_update_ts = df_hybrid_attrs.get("forming_last_update_ts")
        forming_update_age_ms = df_hybrid_attrs.get("forming_update_age_ms")

        used_forming = bool(includes_forming and fallback_reason is None)
        df_eval = df_30m_hybrid if used_forming else df_30m_closed
        rsi_source = "live" if used_forming else "closed"
        trigger_price_source = "forming_close" if used_forming else "closed_close"
        eval_source = "hybrid" if used_forming else "closed"

        # --- Data Validation Step ---
        validation_passed, reason = self._validate_input_data(df_eval, df_1h, regime_data, symbol_display)
        if not validation_passed:
            logger.info(f"🚫 {log_prefix} No Signal: {reason}")
            return None

        try:
            if self.debug_logging:
                logger.debug(f"{log_prefix} Data sufficiency check: total_rows={len(df_eval)}")
            last30 = df_eval.iloc[-1]
            required_cols = ['close', 'rsi', 'atr', 'ema_fast', 'ema21', 'ema50', 'ema200']
            missing = [c for c in required_cols if c not in last30.index]
            if missing:
                logger.warning(f"{log_prefix} Missing required columns in latest row: {missing}")
                return None
            if any(pd.isna(last30[c]) for c in required_cols):
                logger.info(f"{log_prefix} Latest row has NaN in required columns; skipping this tick.")
                return None
        except IndexError:
            logger.info(f"🚫 {log_prefix} No Signal: Insufficient 30m data to generate a signal (IndexError).")
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
            atr_value = float(last30.get('atr', close_price * 0.02))
            ema21 = float(last30.get('ema21', 0))
            ema50 = float(last30.get('ema50', 0))
            ema200 = float(last30.get('ema200', 0))
            long_ema_value = ema50

            adx_val = None
            try:
                adx_val = float(last30.get("adx")) if last30 is not None and "adx" in last30 else None
            except Exception:
                adx_val = None

            trend_strength = regime_data.get('trend_strength')
            if trend_strength is None:
                trend_strength = regime_data.get('adx')
            try:
                trend_strength = float(trend_strength or 0.0)
            except Exception:
                trend_strength = 0.0

            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal'),
                'trend_strength': trend_strength,
            }

            used_row_ts = None
            try:
                if "timestamp" in last30:
                    used_row_ts = last30.get("timestamp")
                if used_row_ts is None and hasattr(df_eval, "index") and len(df_eval.index) > 0:
                    idx = df_eval.index[-1]
                    used_row_ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
            except Exception:
                used_row_ts = None

            meta_state = (includes_forming, fallback_reason, merge_action)
            should_log_hybrid_meta = used_forming or (meta_state != self._last_hybrid_meta_state)
            if should_log_hybrid_meta:
                entry_log = f"{close_price:.2f}" if close_price is not None else "None"
                rsi_log = f"{rsi_val:.2f}" if rsi_val is not None else "None"
                logger.info(
                    f"{log_prefix} Hybrid meta | includes_forming={includes_forming} "
                    f"used_forming={used_forming} fallback_reason={(fallback_reason if fallback_reason is not None else 'none')} "
                    f"merge_action={merge_action} rsi_source={rsi_source} trigger_price_source={trigger_price_source} "
                    f"used_row_ts={used_row_ts} forming_ts={forming_ts} "
                    f"forming_last_update_ts={forming_last_update_ts} forming_update_age_ms={forming_update_age_ms} "
                    f"entry={entry_log} rsi={rsi_log}"
                )
                self._last_hybrid_meta_state = meta_state
            
            # --- Adaptive Threshold Calculation ---
            base_threshold_for_meta = None
            adaptive_rsi_threshold = self.get_symbol_specific_threshold(symbol_display)
            if adaptive_rsi_threshold is not None:
                base_threshold = float(self.base_cfg.get('adaptive_rsi_base', 68.0))
                base_threshold_for_meta = float(base_threshold)
                logger.info(
                    f"⚠️ {log_prefix} Symbol-specific RSI threshold active: base {base_threshold:.2f} → {adaptive_rsi_threshold:.2f}."
                )
            else:
                adaptive_rsi_threshold, threshold_meta = self.get_adaptive_rsi_threshold(
                    market_regime,
                    return_reason=True
                )
                meta_base = threshold_meta.get('base_threshold', adaptive_rsi_threshold)
                try:
                    base_threshold_for_meta = float(meta_base)
                except Exception:
                    base_threshold_for_meta = None
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

            # -------------------------------------------------------------------------
            # CUSTOM TREND VETO (Step 2 Implementation)
            # Purpose: In strong uptrends (Price > EMA50 and ADX > 25),
            # force the RSI threshold to a minimum of 75.0 to prevent premature shorts.
            # -------------------------------------------------------------------------
            # 'ema50' variable is assumed to be defined earlier in the method
            ema_long = ema50 

            if (
                adx_val is not None             # Data exists
                and not pd.isna(adx_val)        # Not NaN
                and ema_long and ema_long > 0   # EMA is valid
                and close_price > ema_long      # Bullish Structure
                and adx_val > 25.0              # Strong Trend Strength
                and adaptive_rsi_threshold < 75.0 # Current threshold is too low
            ):
                # Log the intervention for transparency
                # Using module-level logger directly for visibility
                logger.info(f"{symbol_display} Trend Veto Triggered: Raising RSI Threshold from {adaptive_rsi_threshold:.1f} to 75.0 (Price > EMA50 & ADX {adx_val:.1f} > 25)")

                adaptive_rsi_threshold = 75.0
            # -------------------------------------------------------------------------

            volatility_regime = market_regime.get('volatility', 'normal')
            rip_atr_multiplier = 1.0
            if volatility_regime == 'high':
                rip_atr_multiplier = 1.5
            elif volatility_regime == 'low':
                rip_atr_multiplier = 0.75

            rip_threshold_value = None
            rip_delta_value = None
            rip_pass_shadow = None
            if ema50 > 0 and atr_value > 0:
                rip_threshold_value = ema50 + (atr_value * rip_atr_multiplier)
                rip_delta_value = close_price - rip_threshold_value
                rip_pass_shadow = close_price >= rip_threshold_value

            timeframe = "30m"
            last_closed_ts_ms = extract_last_closed_ts_ms(df_eval)
            df_meta = extract_df_meta(df_eval)

            def _shadow_str(decision: str, fail_reason: str = "", extra: Optional[Dict] = None) -> None:
                if not shadow_enabled():
                    return
                try:
                    payload = {
                        "event": "strategy_shadow_eval",
                        "strategy": "adaptive_str",
                        "symbol": symbol_display,
                        "timeframe": timeframe,
                        "last_closed_ts": last_closed_ts_ms,
                        **df_meta,
                        "close": close_price,
                        "rsi": rsi_val,
                        "rsi_threshold": adaptive_rsi_threshold,
                        "rsi_delta": rsi_val - adaptive_rsi_threshold if adaptive_rsi_threshold is not None else None,
                        "ema21": ema21,
                        "ema50": ema50,
                        "ema200": ema200,
                        "atr": atr_value,
                        "rip_vol_mult": rip_atr_multiplier,
                        "rip_threshold": rip_threshold_value,
                        "rip_delta": rip_delta_value,
                        "rip_pass_shadow": rip_pass_shadow,
                        "volatility": volatility_regime,
                        "decision": decision,
                        "fail_reason": fail_reason,
                    }
                    if extra:
                        payload.update(extra)
                    emit_shadow_log(logger, payload, "adaptive_str", symbol_display, timeframe, last_closed_ts_ms)
                except Exception:
                    return

            # --- Panic/Pump detector for shorts ---
            side = (signal.get("side") or "").lower()
            if side in ("sell", "short"):
                current_volume = None
                avg_volume = None
                if df_eval is not None and hasattr(df_eval, "columns") and "volume" in df_eval.columns:
                    try:
                        current_volume = float(last30.get("volume"))
                    except Exception:
                        current_volume = None
                    try:
                        vol_series = df_eval["volume"].dropna()
                        if not vol_series.empty:
                            avg_volume = float(vol_series.tail(20).mean())
                    except Exception:
                        avg_volume = None

                if (
                    current_volume is not None
                    and avg_volume is not None
                    and avg_volume > 0
                    and current_volume > (avg_volume * 3.0)
                ):
                    logger.info(
                        f"{log_prefix} Panic veto: volume spike (current={current_volume:.2f}, avg20={avg_volume:.2f})."
                    )
                    _shadow_str("no_signal_volume", "volume_spike")
                    return None

                panic_cfg = self.strategy_config.get("panic_veto") or {}
                volume_5m_enabled = bool(panic_cfg.get("volume_5m_enabled", True))
                volume_5m_mult = float(panic_cfg.get("volume_5m_mult", 3.0))
                volume_5m_window = int(panic_cfg.get("volume_5m_window", 20))

                df_5m = None
                if market_data and isinstance(market_data, dict):
                    df_5m = market_data.get("5m")
                if (
                    volume_5m_enabled
                    and df_5m is not None
                    and hasattr(df_5m, "empty")
                    and not df_5m.empty
                ):
                    try:
                        vol_series_5m = df_5m["volume"].dropna()
                        if not vol_series_5m.empty:
                            current_volume_5m = float(vol_series_5m.iloc[-1])
                            avg_volume_5m = float(vol_series_5m.tail(volume_5m_window).mean())
                            if avg_volume_5m > 0 and current_volume_5m > (avg_volume_5m * volume_5m_mult):
                                logger.info(
                                    f"{log_prefix} Panic veto: 5m volume spike (current={current_volume_5m:.2f}, avg{volume_5m_window}={avg_volume_5m:.2f})."
                                )
                                _shadow_str("no_signal_volume_5m", "volume_spike_5m")
                                return None
                    except Exception:
                        pass

                if ema50 > 0 and close_price > (ema50 * 1.02):
                    logger.info(
                        f"{log_prefix} Panic veto: price {close_price:.2f} > EMA50*1.02 ({ema50:.2f})."
                    )
                    _shadow_str("no_signal_parabolic", "parabolic_pump")
                    return None

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
                logger.info(
                    f"   Regime: {market_regime['trend']}, "
                    f"Trend Strength (ADX): {market_regime.get('trend_strength', 0.0):.1f}, "
                    f"Volatility: {market_regime['volatility']}"
                )

            # 1. RSI Condition Check
            if rsi_val < adaptive_rsi_threshold:
                logger.info(f"🚫 {log_prefix} No Signal: RSI ({rsi_val:.2f}) is below the threshold ({adaptive_rsi_threshold:.2f}).")
                _shadow_str("no_signal_rsi", "rsi_below_threshold")
                return None

            # 2. Dynamic Rip Check (Replaces strict EMA alignment)
            # "Rip" is defined as price extending significantly above a Long-Term EMA
            
            # Get Long-Term EMA (using EMA 50 as proxy for trend baseline)
            if long_ema_value > 0 and atr_value > 0 and rip_threshold_value is not None:
                rip_value = atr_value * rip_atr_multiplier
                required_price_threshold = rip_threshold_value
                
                if close_price < required_price_threshold:
                    logger.info(
                        f"🚫 {log_prefix} No Signal: Rip Check Failed. Price ${close_price:,.2f} "
                        f"is not above EMA50+Rip (${required_price_threshold:,.2f}). "
                        f"(EMA50=${long_ema_value:,.2f}, Rip=${rip_value:,.2f}, Vol={volatility_regime})"
                    )
                    _shadow_str(
                        "no_signal_rip",
                        "rip_check_failed",
                        {"rip_pass": rip_pass_shadow},
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

            mtf_skipped = False
            mtf_meta_15m = None
            mtf_meta_1h = None

            # --- Optional Multi-Timeframe (MTF) Confirmation ---
            mtf_policy = self._mtf_policy
            if isinstance(mtf_policy, MtfConfirmationConfig):
                tf_15m = mtf_policy.tf_15m
                tf_1h = mtf_policy.tf_1h

                df_15m = None
                if market_data and tf_15m.mode != "off":
                    df_15m = market_data.get('15m')

                df_1h_local = df_1h
                if (
                    tf_1h.mode != "off"
                    and (df_1h_local is None or (hasattr(df_1h_local, 'empty') and df_1h_local.empty))
                    and market_data
                ):
                    df_1h_local = market_data.get("df_1h")
                    if df_1h_local is None:
                        df_1h_local = market_data.get("1h")

                    if df_1h_local is None:
                        logger.warning("[Adaptive_STR] Missing 1h dataframe (df_1h/1h).")
                    elif hasattr(df_1h_local, "empty") and df_1h_local.empty:
                        logger.warning("[Adaptive_STR] Empty 1h dataframe.")

                if tf_15m.mode == "off":
                    mtf_meta_15m = {"timeframe": "15m", "mode": "off", "status": "skipped"}
                    signal.setdefault("features", {})["mtf_15m"] = mtf_meta_15m
                    logger.info(f"ℹ️ {log_prefix} MTF-15m skipped (mode=off).")
                    mtf_skipped = True
                else:
                    passed_15m, reason_15m, meta_15m = self._mtf_confirm_15m(
                        df_15m,
                        symbol_display,
                        tf_15m,
                        mtf_policy.min_bars,
                    )
                    mtf_meta_15m = meta_15m
                    signal.setdefault("features", {})["mtf_15m"] = meta_15m
                    if meta_15m and meta_15m.get("action") == "skip":
                        mtf_skipped = True
                    if (not passed_15m) and tf_15m.mode == "hard":
                        bypassed, bypass_reason, bypass_meta = self._check_extreme_bypass_mtf_hard_veto(
                            signal=signal,
                            df_eval=df_eval,
                            close_price=close_price,
                            atr_value=atr_value,
                            rsi_value=rsi_val,
                            timeframe="15m",
                            failure_kind=(
                                "missing-data" if meta_15m.get("status") == "missing" else "threshold"
                            ),
                        )
                        if bypassed:
                            passed_15m = True
                            meta_15m["bypass"] = True
                            meta_15m["bypass_reason"] = bypass_reason
                            meta_15m["bypass_meta"] = bypass_meta
                            try:
                                bypass_ctx = {
                                    "event": "mtf_hard_veto_bypass_ctx",
                                    "strategy": self.strategy_name,
                                    "symbol": symbol_display,
                                    "timeframe": "15m",
                                    "failure_kind": (
                                        "missing-data" if meta_15m.get("status") == "missing" else "threshold"
                                    ),
                                    "side": signal.get("side"),
                                    "bypass_reason": bypass_reason,
                                    "bypass_meta": bypass_meta,
                                    "eval": {
                                        "eval_tf": "30m",
                                        "eval_source": eval_source,
                                        "includes_forming": includes_forming,
                                        "used_forming": used_forming,
                                        "merge_action": merge_action,
                                        "fallback_reason": fallback_reason,
                                        "forming_ts": forming_ts,
                                        "forming_last_update_ts": forming_last_update_ts,
                                        "forming_update_age_ms": forming_update_age_ms,
                                        "rsi_source": rsi_source,
                                        "trigger_price_source": trigger_price_source,
                                    },
                                    "snapshot": {
                                        "close": float(close_price),
                                        "rsi": float(rsi_val),
                                        "atr_value": float(atr_value),
                                        "atr_pct": (float(atr_value) / float(close_price)) if close_price else None,
                                    },
                                }
                                logger.info(
                                    "mtf_hard_veto_bypass_ctx %s",
                                    json.dumps(bypass_ctx, separators=(",", ":"), default=str),
                                )
                            except Exception:
                                pass
                            logger.warning(
                                "🚨 %s MTF-15m hard veto BYPASSED | code=%s | bypass=%s",
                                log_prefix,
                                meta_15m.get("code", "mtf_15m_block"),
                                bypass_reason,
                            )
                    if not passed_15m:
                        code = meta_15m.get("code", "mtf_15m_block")
                        failure_kind = "missing-data" if meta_15m.get("status") == "missing" else "threshold"
                        if tf_15m.mode == "hard":
                            logger.info(
                                f"🚫 {log_prefix} No Signal: MTF-15m veto ({failure_kind}): {reason_15m}. [code={code}]"
                            )
                            _shadow_str(
                                "no_signal_mtf",
                                "mtf_15m_block",
                                {"mtf_15m": meta_15m},
                            )
                            return None
                        meta_15m["soft_fail"] = True
                        logger.info(
                            f"⚠️ {log_prefix} MTF-15m soft-fail ({failure_kind}): {reason_15m}. [code={code}]"
                        )

                if tf_1h.mode == "off":
                    mtf_meta_1h = {"timeframe": "1h", "mode": "off", "status": "skipped"}
                    signal.setdefault("features", {})["mtf_1h"] = mtf_meta_1h
                    logger.info(f"ℹ️ {log_prefix} MTF-1h skipped (mode=off).")
                    mtf_skipped = True
                else:
                    passed_1h, reason_1h, meta_1h = self._mtf_confirm_1h(
                        df_1h_local,
                        symbol_display,
                        tf_1h,
                        mtf_policy.min_bars,
                    )
                    mtf_meta_1h = meta_1h
                    signal.setdefault("features", {})["mtf_1h"] = meta_1h
                    if meta_1h and meta_1h.get("action") == "skip":
                        mtf_skipped = True
                    if (not passed_1h) and tf_1h.mode == "hard":
                        bypassed, bypass_reason, bypass_meta = self._check_extreme_bypass_mtf_hard_veto(
                            signal=signal,
                            df_eval=df_eval,
                            close_price=close_price,
                            atr_value=atr_value,
                            rsi_value=rsi_val,
                            timeframe="1h",
                            failure_kind=(
                                "missing-data" if meta_1h.get("status") == "missing" else "threshold"
                            ),
                        )
                        if bypassed:
                            passed_1h = True
                            meta_1h["bypass"] = True
                            meta_1h["bypass_reason"] = bypass_reason
                            meta_1h["bypass_meta"] = bypass_meta
                            try:
                                bypass_ctx = {
                                    "event": "mtf_hard_veto_bypass_ctx",
                                    "strategy": self.strategy_name,
                                    "symbol": symbol_display,
                                    "timeframe": "1h",
                                    "failure_kind": (
                                        "missing-data" if meta_1h.get("status") == "missing" else "threshold"
                                    ),
                                    "side": signal.get("side"),
                                    "bypass_reason": bypass_reason,
                                    "bypass_meta": bypass_meta,
                                    "eval": {
                                        "eval_tf": "30m",
                                        "eval_source": eval_source,
                                        "includes_forming": includes_forming,
                                        "used_forming": used_forming,
                                        "merge_action": merge_action,
                                        "fallback_reason": fallback_reason,
                                        "forming_ts": forming_ts,
                                        "forming_last_update_ts": forming_last_update_ts,
                                        "forming_update_age_ms": forming_update_age_ms,
                                        "rsi_source": rsi_source,
                                        "trigger_price_source": trigger_price_source,
                                    },
                                    "snapshot": {
                                        "close": float(close_price),
                                        "rsi": float(rsi_val),
                                        "atr_value": float(atr_value),
                                        "atr_pct": (float(atr_value) / float(close_price)) if close_price else None,
                                    },
                                }
                                logger.info(
                                    "mtf_hard_veto_bypass_ctx %s",
                                    json.dumps(bypass_ctx, separators=(",", ":"), default=str),
                                )
                            except Exception:
                                pass
                            logger.warning(
                                "🚨 %s MTF-1h hard veto BYPASSED | code=%s | bypass=%s",
                                log_prefix,
                                meta_1h.get("code", "mtf_1h_block"),
                                bypass_reason,
                            )
                    if not passed_1h:
                        code = meta_1h.get("code", "mtf_1h_block")
                        failure_kind = "missing-data" if meta_1h.get("status") == "missing" else "threshold"
                        if tf_1h.mode == "hard":
                            logger.info(
                                f"🚫 {log_prefix} No Signal: MTF-1h veto ({failure_kind}): {reason_1h}. [code={code}]"
                            )
                            _shadow_str(
                                "no_signal_mtf",
                                "mtf_1h_block",
                                {
                                    "mtf_15m": mtf_meta_15m,
                                    "mtf_1h": meta_1h,
                                },
                            )
                            return None
                        meta_1h["soft_fail"] = True
                        logger.info(
                            f"⚠️ {log_prefix} MTF-1h soft-fail ({failure_kind}): {reason_1h}. [code={code}]"
                        )
            
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
                    _shadow_str(
                        "no_signal_ml",
                        "ml_veto",
                        {
                            "ml_regime": ml_context.get('regime_prediction'),
                            "ml_confidence": ml_context.get('regime_confidence'),
                        },
                    )
                    return None
                
                if ml_context.get('price_direction') == 'up' and ml_context.get('price_confidence', 0) > 0.7:
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong price up prediction (confidence: {ml_context.get('price_confidence', 0):.2%}).")
                    _shadow_str(
                        "no_signal_ml",
                        "ml_veto",
                        {
                            "ml_price_direction": ml_context.get('price_direction'),
                            "ml_price_confidence": ml_context.get('price_confidence'),
                        },
                    )
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

            # --- Smart Guard: RSI rollover check for risky shorts ---
            # Risky context = forming-trigger evaluation and/or MTF soft-fail.
            # This prevents "top-tick" shorts when RSI is still rising/flat during formation.
            rollover_cfg = self.strategy_config.get("rsi_rollover_guard") if isinstance(self.strategy_config, dict) else None
            if rollover_cfg is True:
                rollover_cfg = {"enabled": True}
            if not isinstance(rollover_cfg, dict):
                rollover_cfg = {}

            rollover_enabled = bool(rollover_cfg.get("enabled", True))
            if rollover_enabled:
                eps = rollover_cfg.get("eps", 0.2)
                try:
                    eps = float(eps)
                except Exception:
                    eps = 0.2
                eps = max(0.0, eps)

                require_risky_context = bool(rollover_cfg.get("require_risky_context", True))

                side_norm = str(signal.get("side") or "").strip().lower()
                is_forming_trigger = str(trigger_price_source or "") == "forming_close"
                mtf_soft_fail = bool(isinstance(mtf_meta_15m, dict) and mtf_meta_15m.get("soft_fail"))
                risky_context = bool(is_forming_trigger or mtf_soft_fail)

                if side_norm in ("sell", "short") and (risky_context or not require_risky_context):
                    rsi_prev = None
                    rsi_now = None
                    rsi_pair_source = "unknown"
                    guard_anchor_tf = None
                    guard_anchor_source = None

                    def _try_fast_anchor(tf_key: str) -> bool:
                        """
                        Use a faster TF RSI last2 as the rollover reference.
                        Prefer closed-only RSI values: if the TF dataframe includes a forming row,
                        exclude it before selecting last2.
                        """
                        nonlocal rsi_prev, rsi_now, rsi_pair_source, guard_anchor_tf, guard_anchor_source

                        if not market_data or not isinstance(market_data, dict):
                            return False

                        df_fast = market_data.get(tf_key)
                        if df_fast is None or not isinstance(df_fast, pd.DataFrame) or df_fast.empty:
                            return False
                        if "rsi" not in df_fast.columns:
                            return False

                        series = df_fast["rsi"]
                        try:
                            includes_fast_forming = bool(getattr(df_fast, "attrs", {}).get("includes_forming", False))
                        except Exception:
                            includes_fast_forming = False

                        guard_anchor_tf = tf_key
                        if includes_fast_forming and len(series) >= 2:
                            # Exclude last row (forming) to avoid mixing intrabar updates into reference.
                            series = series.iloc[:-1]
                            guard_anchor_source = "hybrid_excl_forming"
                        else:
                            guard_anchor_source = "closed_only"

                        series = series.dropna()
                        if len(series) < 2:
                            guard_anchor_tf = None
                            guard_anchor_source = None
                            return False

                        try:
                            rsi_prev = float(series.iloc[-2])
                            rsi_now = float(series.iloc[-1])
                        except Exception:
                            rsi_prev = None
                            rsi_now = None
                            guard_anchor_tf = None
                            guard_anchor_source = None
                            return False

                        rsi_pair_source = f"fast_{tf_key}_last2"
                        return True

                    # If signal is based on forming_close, prefer a fast RSI anchor to avoid stale 30m prev.
                    if is_forming_trigger:
                        if not _try_fast_anchor("5m"):
                            _try_fast_anchor("15m")

                    # Preferred: prev from CLOSED, now from HYBRID (forming).
                    if (rsi_prev is None or rsi_now is None) and used_forming:
                        try:
                            if (
                                df_30m_closed is not None
                                and hasattr(df_30m_closed, "columns")
                                and "rsi" in df_30m_closed.columns
                                and len(df_30m_closed) >= 1
                            ):
                                rsi_prev = float(df_30m_closed["rsi"].iloc[-1])
                                rsi_pair_source = "closed_prev"
                        except Exception:
                            rsi_prev = None
                        try:
                            if (
                                df_eval is not None
                                and hasattr(df_eval, "columns")
                                and "rsi" in df_eval.columns
                                and len(df_eval) >= 1
                            ):
                                rsi_now = float(df_eval["rsi"].iloc[-1])
                                rsi_pair_source = f"{rsi_pair_source}+eval_now"
                        except Exception:
                            rsi_now = None

                    # Fallback: use last two RSI values from the eval series (closed or hybrid).
                    if rsi_prev is None or rsi_now is None:
                        try:
                            if (
                                df_eval is not None
                                and hasattr(df_eval, "columns")
                                and "rsi" in df_eval.columns
                                and len(df_eval) >= 2
                            ):
                                rsi_prev = float(df_eval["rsi"].iloc[-2])
                                rsi_now = float(df_eval["rsi"].iloc[-1])
                                rsi_pair_source = "eval_last2"
                        except Exception:
                            rsi_prev = None
                            rsi_now = None

                    if rsi_prev is None or rsi_now is None:
                        # Fail-open: do not kill the signal on missing/NaN RSI history.
                        self._inc_guard_telemetry("guard_rollover_skip_count")
                        logger.info(
                            f"{log_prefix} [GUARD] Skip RSI rollover (fail-open): insufficient/NaN RSI history "
                            f"trigger={trigger_price_source} mtf_soft_fail={mtf_soft_fail}"
                        )
                    else:
                        # Guard: require RSI to be actually falling (rollover)
                        if float(rsi_now) >= (float(rsi_prev) - float(eps)):
                            self._inc_guard_telemetry("guard_rollover_defer_count")
                            logger.info(
                                f"{log_prefix} [GUARD] Defer Risky SHORT: RSI not rolling over. "
                                f"rsi_prev={float(rsi_prev):.2f} rsi_now={float(rsi_now):.2f} eps={float(eps):.2f} "
                                f"trigger_price_source={trigger_price_source} mtf_soft_fail={mtf_soft_fail} "
                                f"reason=guard.rsi_rollover_defer rsi_pair={rsi_pair_source} "
                                f"guard_anchor_tf={guard_anchor_tf} guard_anchor_source={guard_anchor_source} "
                                f"guard_rollover_defer_count={self._guard_telemetry.get('guard_rollover_defer_count', 0)}"
                            )
                            _shadow_str(
                                "guard_rollover_defer",
                                "guard.rsi_rollover_defer",
                                {
                                    "rsi_prev": float(rsi_prev),
                                    "rsi_now": float(rsi_now),
                                    "eps": float(eps),
                                    "trigger_price_source": trigger_price_source,
                                    "mtf_soft_fail": mtf_soft_fail,
                                    "rsi_pair_source": rsi_pair_source,
                                    "guard_anchor_tf": guard_anchor_tf,
                                    "guard_anchor_source": guard_anchor_source,
                                    "guard_rollover_defer_count": int(self._guard_telemetry.get("guard_rollover_defer_count", 0) or 0),
                                },
                            )
                            return None
             
            entry_price = close_price
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
                _shadow_str(
                    "no_signal_rr",
                    "rr_below_min",
                    {
                        "rr_ratio": rr_ratio,
                        "min_rr_required": self.min_rr_ratio,
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                        "volatility_stop_meta": volatility_stop_meta,
                    },
                )
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

            # Coordinator-level safety layers (e.g., SafetyOverride) rely on a small snapshot in signal meta.
            try:
                ema_fast = None
                try:
                    ema_fast = float(last30.get("ema_fast")) if last30 is not None and "ema_fast" in last30 else None
                except Exception:
                    ema_fast = None

                candle_open = None
                candle_close = None
                try:
                    candle_open = float(last30.get("open")) if last30 is not None and "open" in last30 else None
                except Exception:
                    candle_open = None
                try:
                    candle_close = float(last30.get("close")) if last30 is not None and "close" in last30 else None
                except Exception:
                    candle_close = None

                current_volume = None
                volume_ma20 = None
                if df_eval is not None and hasattr(df_eval, "columns") and "volume" in df_eval.columns:
                    try:
                        current_volume = float(last30.get("volume"))
                    except Exception:
                        current_volume = None
                    try:
                        vol_series = df_eval["volume"].dropna()
                        if not vol_series.empty:
                            volume_ma20 = float(vol_series.tail(20).mean())
                    except Exception:
                        volume_ma20 = None

                resistance_level = None
                resistance_distance_bps = None
                try:
                    if df_30m_closed is not None and hasattr(df_30m_closed, "columns") and "high" in df_30m_closed.columns:
                        highs = df_30m_closed["high"].dropna()
                        if not highs.empty:
                            resistance_level = float(highs.tail(20).max())
                            if close_price and close_price > 0:
                                resistance_distance_bps = ((resistance_level - float(close_price)) / float(close_price)) * 10000.0
                except Exception:
                    resistance_level = None
                    resistance_distance_bps = None

                mtf_soft_fail = bool(isinstance(mtf_meta_15m, dict) and mtf_meta_15m.get("soft_fail"))

                base_thr = float(base_threshold_for_meta) if base_threshold_for_meta is not None else None
                cur_thr = float(adaptive_rsi_threshold) if adaptive_rsi_threshold is not None else None
                delta_thr = (base_thr - cur_thr) if (base_thr is not None and cur_thr is not None) else None

                signal_meta = signal.setdefault("meta", {})
                signal_meta["adaptive_threshold"] = {
                    "base_threshold": base_thr,
                    "current_threshold": cur_thr,
                    "delta": delta_thr,
                    "lowered": bool(delta_thr is not None and delta_thr > 0),
                }
                signal_meta["safety_snapshot"] = {
                    "close": float(close_price) if close_price is not None else None,
                    "rsi": float(rsi_val) if rsi_val is not None else None,
                    "ema_fast": ema_fast,
                    "ema21": float(ema21) if ema21 is not None else None,
                    "ema50": float(ema50) if ema50 is not None else None,
                    "candle_open": candle_open,
                    "candle_close": candle_close,
                    "volume": current_volume,
                    "volume_ma20": volume_ma20,
                    "resistance_level": resistance_level,
                    "resistance_distance_bps": resistance_distance_bps,
                    "trigger_price_source": trigger_price_source,
                    "mtf_soft_fail": mtf_soft_fail,
                }
            except Exception:
                pass
            
            if ml_enhanced:
                signal['ml_consensus'] = ml_context.get('consensus_score')
                signal['ml_position_modifier'] = position_size_modifier

            decision_to_log = "mtf_skipped" if mtf_skipped else "pass"
            _shadow_str(
                decision_to_log,
                "mtf_policy_skip" if mtf_skipped else "",
                {
                    "rr_ratio": rr_ratio,
                    "min_rr_required": self.min_rr_ratio,
                    "volatility_stop_meta": volatility_stop_meta,
                    "mtf_15m": mtf_meta_15m,
                    "mtf_1h": mtf_meta_1h,
                },
            )
            
            return signal
            
        except Exception as e:
            logger.error(
                f"?? {log_prefix} Critical error during signal generation: {e} | "
                f"eval_source={eval_source} includes_forming={includes_forming} "
                f"used_forming={used_forming} fallback_reason={(fallback_reason if fallback_reason is not None else 'none')} "
                f"merge_action={merge_action}",
                exc_info=True
            )

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
            'has_regime_analyzer': self.regime_analyzer is not None,
            'guard_telemetry': dict(self._guard_telemetry) if isinstance(getattr(self, "_guard_telemetry", None), dict) else {},
        }
