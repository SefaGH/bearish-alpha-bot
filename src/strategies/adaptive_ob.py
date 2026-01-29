"""
Adaptive OversoldBounce strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import json
import pandas as pd
import logging
import math
import time
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List
from .oversold_bounce import OversoldBounce
from core.strategy_shadow_eval import (
    shadow_enabled,
    extract_last_closed_ts_ms,
    extract_df_meta,
    emit_shadow_log,
)
from core.data_validator import TIMEFRAME_SECONDS
from core.indicators import rsi
from core.resistance_band import compute_band, confirmed_pivot_highs, confirmed_pivot_lows

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


class AdaptiveOversoldBounce(OversoldBounce):
    """
    Market regime-aware OversoldBounce strategy.
    
    Adapts RSI thresholds, position sizing, and EMA requirements
    based on real-time market regime analysis.
    """
    
    # Maximum adjustment to base threshold (in RSI points)
    MAX_THRESHOLD_ADJUSTMENT = 5
    
    def __init__(self, cfg: Dict, regime_analyzer=None, market_data_pipeline=None):
        """
        Initialize adaptive OversoldBounce strategy.
        
        Args:
            cfg: Strategy configuration dictionary
            regime_analyzer: MarketRegimeAnalyzer instance for regime detection
        
        🔥 GÜNCELLEME: Kalıtım zinciri `super().__init__(cfg)` çağrısıyla onarıldı.
        Minimum R/R oranı artık başlangıçta config'den okunup `self.min_rr_ratio` olarak saklanıyor.
        """
        # `OversoldBounce` sınıfının __init__ metodunu çağırarak doğru kurulumu sağla.
        super().__init__(cfg)
        
        # Üst sınıf "oversold_bounce" olarak ayarlayabileceği için, bu alt sınıfın adını yeniden tanımla.
        self.strategy_name = "adaptive_ob"
        
        self.regime_analyzer = regime_analyzer
        self.market_data_pipeline = market_data_pipeline
        self.base_cfg = cfg.copy()
        self.debug_logging = self.base_cfg.get('debug', {}).get('strategy_logging', False)
        self._persistency_state: Dict[str, Dict[str, float]] = {}
        self._persistency_skip_log_ts: Dict[str, float] = {}
        self._persistency_cfg = {
            "mode": "time",
            "seconds": 5.0,
            "min_samples": 2,
            "wick_closeness_k": 0.25,
        }
        # Dynamic RSI Phase 0 shadow state (per-symbol)
        self._dyn_state_by_symbol: Dict[str, str] = {}
        self._dyn_armed_until_ms_by_symbol: Dict[str, int] = {}
        self._dyn_cooldown_until_ms_by_symbol: Dict[str, int] = {}
        self._dyn_last_fast_status_log_ts: Dict[Tuple[str, str], float] = {}
        self._dyn_last_fast_status_snapshot: Dict[Tuple[str, str], Tuple[bool, bool, bool]] = {}
        self._dyn_last_seen_ts: Dict[str, int] = {}
        self._dyn_last_shock_score_by_symbol: Dict[str, float] = {}
        self._dyn_last_shock_tf_by_symbol: Dict[str, Optional[str]] = {}
        self._dyn_last_shock_close_source_by_symbol: Dict[str, str] = {}
        self._trend_penalty_state_by_symbol: Dict[str, bool] = {}

        # Minimum R/R oranını başlangıçta, bir kez olmak üzere config'den oku.
        # `super()` çağrısı `self.strategy_config`'i oluşturduğu için artık bunu güvenle kullanabiliriz.
        self.min_rr_ratio = self.strategy_config.get('min_rr_ratio', 1.2)
        logger.info(f"[{self.strategy_name.upper()}] Minimum R/R Ratio initialized to: {self.min_rr_ratio}")
        # Interface guard
        if not hasattr(self, "signal"):
            self.signal = self._default_signal_wrapper  # type: ignore
        assert callable(getattr(self, "signal", None)), f"{self.strategy_name}: signal method not callable"

    def _update_dyn_shadow(
        self,
        symbol: str,
        df_30m: pd.DataFrame,
        now_ms: int,
        slow_fallback_reason: Optional[str],
        bypass_perf_gate: bool = False,
    ) -> None:
        """Phase 0 shadow telemetry for dynamic RSI gate (no decision impact)."""
        try:
            cfg = self.base_cfg.get("dynamic_rsi_gate") or {}
            if not isinstance(cfg, dict) or not cfg.get("enabled", False):
                return

            log_cfg = cfg.get("logging", {}) or {}
            if not isinstance(log_cfg, dict):
                log_cfg = {}
            mode = str(log_cfg.get("mode", "state_changes")).lower()
            if mode == "off":
                return

            pipeline = getattr(self, "market_data_pipeline", None)
            ws_manager = getattr(pipeline, "websocket_manager", None) if pipeline else None
            collector = getattr(ws_manager, "collector", None) if ws_manager else None
            if collector is None:
                return

            gate = float(cfg.get("compute_fast_only_if_rsi_slow_within", 0.0) or 0.0)
            if gate > 0 and not bypass_perf_gate:
                rsi_val = None
                try:
                    if df_30m is not None and not df_30m.empty:
                        includes_forming = bool(getattr(df_30m, "attrs", {}).get("includes_forming", False))
                        row = df_30m.iloc[-2] if includes_forming and len(df_30m) >= 2 else df_30m.iloc[-1]
                        if "rsi" in row:
                            rsi_val = float(row["rsi"])
                except Exception:
                    rsi_val = None
                if rsi_val is not None:
                    base = float(self.base_cfg.get("adaptive_rsi_base", 32.0))
                    if abs(rsi_val - base) > gate:
                        return

            exchange = str(cfg.get("exchange", "bingx") or "bingx").lower()
            fast_tfs = cfg.get("fast_timeframes", ["1m", "5m"])
            if isinstance(fast_tfs, str):
                fast_tfs = [x.strip() for x in fast_tfs.split(",") if x.strip()]
            elif isinstance(fast_tfs, (list, tuple)):
                fast_tfs = [str(x).strip() for x in fast_tfs if str(x).strip()]
            else:
                fast_tfs = ["1m", "5m"]
            if not fast_tfs:
                return

            min_bars_fast = int(cfg.get("min_bars_fast", 50) or 0)
            shock_cfg = cfg.get("shock", {}) or {}
            lookback_bars = int(shock_cfg.get("lookback_bars", 5) or 0)
            price_move_pct = float(shock_cfg.get("price_move_pct", 0.0) or 0.0)
            arm_threshold = float(shock_cfg.get("arm_score_threshold", 1.0) or 1.0)
            max_forming_update_age_ms = int(cfg.get("max_forming_update_age_ms", 0) or 0)

            armed_cfg = cfg.get("armed", {}) or {}
            ttl_s = int(armed_cfg.get("ttl_s", 0) or 0)
            max_ttl_s = int(armed_cfg.get("max_ttl_s", ttl_s) or ttl_s)
            cooldown_s = int(armed_cfg.get("cooldown_s", 0) or 0)
            rearm_policy = str(armed_cfg.get("rearm_policy", "extend")).lower()

            throttle_s = float(log_cfg.get("throttle_s", 60) or 60.0)
            limit_small = max(min_bars_fast, lookback_bars + 1, 60)

            best_shock_score = None
            best_tf = None
            best_last_close_source = "closed"

            for tf in fast_tfs:
                ohlcv_list = collector.get_latest_ohlcv(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=tf,
                    limit=limit_small,
                )
                closed_len = len(ohlcv_list) if ohlcv_list else 0
                has_data = closed_len > 0
                sufficient = closed_len >= min_bars_fast if min_bars_fast > 0 else has_data

                state = {}
                if hasattr(collector, "get_state"):
                    try:
                        state = collector.get_state(exchange, symbol, tf) or {}
                    except Exception:
                        state = {}

                last_closed_ts = state.get("last_closed_ts")
                forming_last_update_ts = state.get("forming_last_update_ts")
                gap_count = state.get("gap_count")
                out_of_order = state.get("out_of_order_drops")

                tf_sec = TIMEFRAME_SECONDS.get(tf)
                tf_ms = int(tf_sec * 1000) if tf_sec else 0
                since_last_close_ms = None
                if last_closed_ts is not None and tf_ms > 0:
                    try:
                        since_last_close_ms = max(0, int(now_ms - (int(last_closed_ts) + tf_ms)))
                    except Exception:
                        since_last_close_ms = None
                since_last_kline_update_ms = None
                if forming_last_update_ts is not None:
                    try:
                        since_last_kline_update_ms = max(0, int(now_ms - int(forming_last_update_ts)))
                    except Exception:
                        since_last_kline_update_ms = None
                update_known = forming_last_update_ts is not None

                snap_key = (symbol, tf)
                snapshot = (has_data, bool(sufficient), update_known)
                prev_snapshot = self._dyn_last_fast_status_snapshot.get(snap_key)
                status_changed = snapshot != prev_snapshot

                now_ts = time.time()
                last_log = self._dyn_last_fast_status_log_ts.get(snap_key, 0.0)
                should_log = status_changed or ((now_ts - last_log) >= throttle_s)
                if should_log:
                    payload = {
                        "event": "ob_dyn_fast_data_status",
                        "exchange": exchange,
                        "symbol": symbol,
                        "tf": tf,
                        "closed_len": closed_len,
                        "last_closed_ts": last_closed_ts,
                        "since_last_close_ms": since_last_close_ms,
                        "since_last_kline_update_ms": since_last_kline_update_ms,
                        "last_kline_update_ts": forming_last_update_ts,
                        "gap_count": gap_count,
                        "out_of_order_drops": out_of_order,
                    }
                    logger.info(json.dumps(payload, separators=(",", ":")))
                    self._dyn_last_fast_status_log_ts[snap_key] = now_ts
                    self._dyn_last_fast_status_snapshot[snap_key] = snapshot

                if best_shock_score is None and has_data and sufficient and update_known:
                    if (
                        ohlcv_list
                        and lookback_bars > 0
                        and len(ohlcv_list) >= (lookback_bars + 1)
                        and price_move_pct > 0
                    ):
                        shock_score = 0.0
                        last_close_source = "closed"
                        try:
                            base = float(ohlcv_list[-(lookback_bars + 1)][4])
                            last = None
                            # Prefer forming close (when fresh) to arm shock earlier during intrabar moves.
                            if (
                                max_forming_update_age_ms > 0
                                and since_last_kline_update_ms is not None
                                and int(since_last_kline_update_ms) <= int(max_forming_update_age_ms)
                                and hasattr(collector, "get_forming_ohlcv")
                            ):
                                try:
                                    forming = collector.get_forming_ohlcv(exchange, symbol, tf)
                                except Exception:
                                    forming = None
                                if isinstance(forming, (list, tuple)) and len(forming) >= 6:
                                    try:
                                        forming_open_ms = int(forming[0])
                                        forming_close = float(forming[4])
                                        # Basic bucket sanity: forming should be current (or near-current) bucket.
                                        if last_closed_ts is not None and tf_ms > 0:
                                            expected_forming_open = int(last_closed_ts) + int(tf_ms)
                                            if abs(forming_open_ms - expected_forming_open) <= int(tf_ms):
                                                last = forming_close
                                                last_close_source = "forming_close"
                                        else:
                                            last = forming_close
                                            last_close_source = "forming_close"
                                    except Exception:
                                        last = None
                            if last is None:
                                last = float(ohlcv_list[-1][4])
                            if base != 0:
                                move_pct = abs((last - base) / base)
                                shock_score = max(0.0, min(move_pct / price_move_pct, 1.0))
                        except Exception:
                            shock_score = 0.0
                        best_shock_score = shock_score
                        best_tf = tf
                        best_last_close_source = last_close_source

            state = self._dyn_state_by_symbol.get(symbol, "DISARMED")
            armed_until = int(self._dyn_armed_until_ms_by_symbol.get(symbol, 0) or 0)
            cooldown_until = int(self._dyn_cooldown_until_ms_by_symbol.get(symbol, 0) or 0)
            new_state = state
            reason = "none"

            ttl_ms = max(ttl_s, 0) * 1000
            max_ttl_ms = max(max_ttl_s, ttl_s, 0) * 1000
            cooldown_ms = max(cooldown_s, 0) * 1000

            shock_score = best_shock_score if best_shock_score is not None else 0.0
            shock_ready = best_shock_score is not None and shock_score >= arm_threshold
            self._dyn_last_shock_score_by_symbol[symbol] = float(shock_score)
            self._dyn_last_shock_tf_by_symbol[symbol] = best_tf
            self._dyn_last_shock_close_source_by_symbol[symbol] = str(best_last_close_source or "closed")

            if state == "DISARMED":
                if shock_ready:
                    new_state = "ARMED"
                    armed_until = now_ms + ttl_ms
                    reason = f"shock_{best_tf}" if best_tf else "shock"
            elif state == "ARMED":
                if armed_until and now_ms >= armed_until:
                    new_state = "COOLDOWN"
                    cooldown_until = now_ms + cooldown_ms
                    reason = "ttl_expired"
                elif shock_ready:
                    if rearm_policy == "extend":
                        target = now_ms + ttl_ms
                        max_target = now_ms + max_ttl_ms
                        armed_until = min(max(armed_until, target), max_target)
                    elif rearm_policy == "reset":
                        armed_until = min(now_ms + ttl_ms, now_ms + max_ttl_ms)
                    elif rearm_policy == "ignore":
                        pass
            elif state == "COOLDOWN":
                if cooldown_until and now_ms >= cooldown_until:
                    new_state = "DISARMED"
                    reason = "cooldown_complete"

            self._dyn_state_by_symbol[symbol] = new_state
            self._dyn_armed_until_ms_by_symbol[symbol] = armed_until
            self._dyn_cooldown_until_ms_by_symbol[symbol] = cooldown_until
            self._dyn_last_seen_ts[symbol] = now_ms

            if new_state != state:
                # Unified eval telemetry (low-noise): only on state transitions.
                try:
                    slow_attrs = getattr(df_30m, "attrs", {}) or {}
                    slow_includes_forming = bool(slow_attrs.get("includes_forming", False))
                    slow_forming_open_time = slow_attrs.get("forming_open_time") or slow_attrs.get("forming_ts")
                    slow_forming_update_age_ms = slow_attrs.get("forming_update_age_ms")
                    slow_rsi = None
                    try:
                        if df_30m is not None and not df_30m.empty:
                            row = df_30m.iloc[-2] if slow_includes_forming and len(df_30m) >= 2 else df_30m.iloc[-1]
                            if "rsi" in row:
                                slow_rsi = float(row["rsi"])
                    except Exception:
                        slow_rsi = None
                    eval_payload = {
                        "event": "ob_dyn_fast_gate_eval",
                        "symbol": symbol,
                        "armed_state": new_state,
                        "state_from": state,
                        "state_to": new_state,
                        "reason": reason,
                        "shock_score": shock_score if best_shock_score is not None else None,
                        "shock_tf": best_tf,
                        "shock_last_close_source": best_last_close_source,
                        "policy": "closed_only" if max_forming_update_age_ms <= 0 else "closed_plus_forming_shock",
                        "max_forming_update_age_ms": max_forming_update_age_ms if max_forming_update_age_ms > 0 else None,
                        "slow_rsi": slow_rsi,
                        "slow_includes_forming": slow_includes_forming,
                        "slow_fallback_reason": slow_fallback_reason,
                        "slow_forming_open_time": slow_forming_open_time,
                        "slow_forming_update_age_ms": slow_forming_update_age_ms,
                        "ts_ms": now_ms,
                    }
                    logger.info(json.dumps(eval_payload, separators=(",", ":")))
                except Exception:
                    pass

                ttl_left_s = max(0, int((armed_until - now_ms) / 1000)) if armed_until else 0
                cooldown_left_s = max(0, int((cooldown_until - now_ms) / 1000)) if cooldown_until else 0
                payload = {
                    "event": "ob_dyn_armed_state_change",
                    "symbol": symbol,
                    "state_from": state,
                    "state_to": new_state,
                    "reason": reason,
                    "ttl_left": ttl_left_s,
                    "cooldown_left": cooldown_left_s,
                    "shock_score": shock_score if best_shock_score is not None else None,
                    "policy": "closed_only",
                    "slow_fallback_reason": slow_fallback_reason,
                }
                logger.info(json.dumps(payload, separators=(",", ":")))
        except Exception:
            return

    def update_dyn_gate(self, *, symbol: str, df_30m: pd.DataFrame, now_ms: int, slow_fallback_reason: Optional[str]) -> None:
        """Public wrapper to update fast-gate state before hybrid fetch decisions."""
        self._update_dyn_shadow(symbol, df_30m, now_ms, slow_fallback_reason, bypass_perf_gate=True)

    def get_dyn_gate_state(self, symbol: str) -> str:
        return str(self._dyn_state_by_symbol.get(symbol, "DISARMED") or "DISARMED")

    def get_dyn_gate_snapshot(self, symbol: str) -> Dict[str, Any]:
        return {
            "state": self.get_dyn_gate_state(symbol),
            "shock_score": self._dyn_last_shock_score_by_symbol.get(symbol),
            "shock_tf": self._dyn_last_shock_tf_by_symbol.get(symbol),
            "shock_last_close_source": self._dyn_last_shock_close_source_by_symbol.get(symbol),
            "last_seen_ts_ms": self._dyn_last_seen_ts.get(symbol),
        }

    def _validate_input_data(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame, regime_data: Dict, symbol: str) -> tuple[bool, str]:
        """Gerekli tüm verilerin varlığını ve geçerliliğini kontrol eder."""
        # 1. Ana DataFrame Kontrolü
        if df_30m is None or df_30m.empty:
            return False, "Input data 'df_30m' is missing or empty."
        if self.debug_logging:
            logger.debug(f"[{self.strategy_name.upper()}] Validation data lengths: total_rows={len(df_30m)}")
            
        # 2. Zorunlu Sütunların Kontrolü
        required_cols = ['close', 'rsi', 'atr', 'ema_fast']
        missing_cols = [col for col in required_cols if col not in df_30m.columns]
        if missing_cols:
            return False, f"df_30m is missing required indicator columns: {missing_cols}."
            
        # 3. Yüksek Önemli "Adaptive" Verilerin Kontrolü (Uyarı Niteliğinde)
        if self.debug_logging:
            if df_1h is None or df_1h.empty:
                logger.info(f"[{self.strategy_name.upper()}-INFO] {symbol} - 'df_1h' is missing. Market regime analysis may be less accurate.")
            if regime_data is None:
                logger.info(f"[{self.strategy_name.upper()}-INFO] {symbol} - 'regime_data' is missing. Strategy will fallback to non-adaptive mode.")

        return True, "All required data is present."
        
    def get_adaptive_rsi_threshold(self, market_regime: Dict) -> float:
        """Sniper-mode RSI floors force patience for deep oversold prints."""
        base_rsi = float(self.base_cfg.get('adaptive_rsi_base', 32.0))
        adapt_range = float(self.base_cfg.get('adaptive_rsi_range', 8.0))

        trend = market_regime.get('trend', 'neutral')
        momentum = market_regime.get('momentum', 'sideways')
        threshold = base_rsi
        if trend == 'bullish':
            if momentum == 'strong':
                threshold = base_rsi + 2.0
            else:
                threshold = base_rsi
        elif trend == 'bearish':
            if momentum == 'strong':
                threshold = base_rsi - 5.0  # Target ~27
            else:
                threshold = base_rsi - 2.0  # Target ~30

        min_threshold = max(20, base_rsi - adapt_range)
        max_threshold = min(40, base_rsi + adapt_range)
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
            return base_multiplier * 0.50
        
        # Low volatility: Can increase position size slightly
        elif volatility_regime == 'low':
            return base_multiplier * 1.25
        
        # Normal volatility: Use base multiplier
        else:
            return base_multiplier
    
    def adapt_ema_distances(self, trend_strength: float) -> Dict[str, float]:
        """
        EMA distance requirements based on trend strength.
        
        Args:
            trend_strength: Trend strength metric (0.0 to 1.0)
            
        Returns:
            Dictionary with EMA distance multipliers
        """
        # Strong trends: Require larger EMA distances (more confirmation)
        if trend_strength > 0.7:
            return {
                'ema_distance_mult': 1.5,
                'require_ema_separation': True
            }
        
        # Weak trends: Smaller EMA distances acceptable
        elif trend_strength < 0.3:
            return {
                'ema_distance_mult': 0.7,
                'require_ema_separation': False
            }
        
        # Moderate trends: Standard requirements
        else:
            return {
                'ema_distance_mult': 1.0,
                'require_ema_separation': False
            }
    
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

    def _check_extreme_bypass(
        self,
        *,
        symbol: str,
        close_price: float,
        atr_value: float,
        market_data: Optional[Dict[str, Any]],
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Extreme bypass (panic mode) for Adaptive OB.

        Used to:
        - skip trend penalty that tightens RSI threshold (ema_fast < ema_mid)
        - bypass ML veto returns during flash-crash / high-volatility moves

        Config (signals.oversold_bounce.extreme_bypass):
          enabled: bool
          triggers:
            price_drop_15m_pct: float   # percent, e.g. 0.8 == 0.8%
            rsi_15m_below: float        # RSI threshold
            min_atr_pct: float          # ATR/price, e.g. 0.006 == 0.6%
        """
        cfg = self.strategy_config if isinstance(self.strategy_config, dict) else {}
        bypass_cfg = cfg.get("extreme_bypass", {})
        if not isinstance(bypass_cfg, dict):
            return False, {}

        if not bool(bypass_cfg.get("enabled", False)):
            return False, {}

        triggers = bypass_cfg.get("triggers", {})
        if not isinstance(triggers, dict):
            triggers = {}

        def _float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        price_drop_15m_pct = _float(triggers.get("price_drop_15m_pct", 0.0), 0.0)
        rsi_15m_below = _float(triggers.get("rsi_15m_below", 0.0), 0.0)
        min_atr_pct = _float(triggers.get("min_atr_pct", 0.0), 0.0)

        if rsi_15m_below and not (0.0 <= rsi_15m_below <= 100.0):
            logger.warning("[OB-BYPASS] Invalid rsi_15m_below=%s for %s", rsi_15m_below, symbol)
            return False, {}

        atr_pct = None
        try:
            atr_pct = (float(atr_value) / float(close_price)) if close_price else None
        except Exception:
            atr_pct = None

        df_15m = None
        if isinstance(market_data, dict):
            df_15m = market_data.get("15m")
            if df_15m is None:
                df_15m = market_data.get("df_15m")

        prev_close_15m = None
        last_close_15m = None
        price_drop_calc = None
        rsi_15m = None

        if isinstance(df_15m, pd.DataFrame) and not df_15m.empty and "close" in df_15m.columns:
            try:
                closes = df_15m["close"].astype(float)
                if len(closes) >= 2:
                    prev_close_15m = float(closes.iloc[-2])
                    last_close_15m = float(closes.iloc[-1])
            except Exception:
                prev_close_15m = None
                last_close_15m = None

            try:
                if "rsi" in df_15m.columns:
                    rsi_15m = float(df_15m["rsi"].iloc[-1])
                else:
                    rsi_15m = float(rsi(df_15m["close"]).iloc[-1])
            except Exception:
                rsi_15m = None

        if prev_close_15m is not None and prev_close_15m > 0 and last_close_15m is not None:
            price_drop_calc = ((prev_close_15m - last_close_15m) / prev_close_15m) * 100.0

        price_drop_ok = True
        if price_drop_15m_pct > 0:
            price_drop_ok = price_drop_calc is not None and price_drop_calc >= price_drop_15m_pct

        rsi_ok = True
        if rsi_15m_below > 0:
            rsi_ok = rsi_15m is not None and rsi_15m <= rsi_15m_below

        atr_ok = True
        if min_atr_pct > 0:
            atr_ok = atr_pct is not None and atr_pct >= min_atr_pct

        triggered = bool(price_drop_ok and rsi_ok and atr_ok)
        meta = {
            "enabled": True,
            "symbol": symbol,
            "price_drop_15m_pct": price_drop_calc,
            "rsi_15m": rsi_15m,
            "atr_pct": atr_pct,
            "thresholds": {
                "price_drop_15m_pct": price_drop_15m_pct,
                "rsi_15m_below": rsi_15m_below,
                "min_atr_pct": min_atr_pct,
            },
            "inputs": {
                "prev_close_15m": prev_close_15m,
                "last_close_15m": last_close_15m,
            },
        }

        return triggered, meta

    @staticmethod
    def _as_closed_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Drop a trailing forming candle when present (lookahead-safe)."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        try:
            includes_forming = bool(getattr(df, "attrs", {}).get("includes_forming", False))
        except Exception:
            includes_forming = False
        if includes_forming and len(df) >= 2:
            return df.iloc[:-1]
        return df

    def _detect_crash_leg(
        self,
        *,
        df: pd.DataFrame,
        timeframe: str,
        lookback_bars: int,
        pivot_left: int,
        pivot_right: int,
        min_drop_pct: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect the most recent "crash leg" (swing high -> swing low) using confirmed pivots.

        This is used by Smart Recovery TP to anchor fibo retracements and structure targets.
        """
        df_closed = self._as_closed_df(df)
        if not isinstance(df_closed, pd.DataFrame) or df_closed.empty:
            return None

        lookback = int(lookback_bars or 0)
        if lookback > 0 and len(df_closed) > lookback:
            df_window = df_closed.tail(lookback).copy()
            offset = int(len(df_closed) - len(df_window))
        else:
            df_window = df_closed.copy()
            offset = 0

        high_col = "high" if "high" in df_window.columns else "close"
        low_col = "low" if "low" in df_window.columns else "close"
        try:
            highs = df_window[high_col].astype(float).values
            lows = df_window[low_col].astype(float).values
        except Exception:
            return None

        left = int(pivot_left or 0)
        right = int(pivot_right or 0)
        n = int(len(df_window))
        if n < max(10, left + right + 3):
            return None

        try:
            piv_hi = confirmed_pivot_highs(highs, left=left, right=right)
            piv_lo = confirmed_pivot_lows(lows, left=left, right=right)
        except Exception:
            return None

        piv_hi_idx = np.flatnonzero(piv_hi)
        piv_lo_idx = np.flatnonzero(piv_lo)

        # Crash low: choose the lowest confirmed pivot low (prefer most recent if ties).
        if piv_lo_idx.size > 0:
            low_prices = lows[piv_lo_idx]
            min_low = float(np.min(low_prices))
            # Tolerance for float comparisons
            tol = max(1e-9, abs(min_low) * 1e-9)
            tied = [int(i) for i, lp in zip(piv_lo_idx.tolist(), low_prices.tolist()) if abs(float(lp) - min_low) <= tol]
            crash_low_idx = int(tied[-1]) if tied else int(piv_lo_idx[int(np.argmin(low_prices))])
        else:
            crash_low_idx = int(np.argmin(lows))
        crash_low = float(lows[crash_low_idx])

        # Crash high: choose the most recent confirmed pivot high before the crash low.
        crash_high_idx = None
        if piv_hi_idx.size > 0:
            prior = piv_hi_idx[piv_hi_idx < crash_low_idx]
            if prior.size > 0:
                crash_high_idx = int(prior[-1])
        if crash_high_idx is None:
            if crash_low_idx <= 0:
                return None
            crash_high_idx = int(np.argmax(highs[: crash_low_idx + 1]))
        crash_high = float(highs[crash_high_idx])

        if crash_high <= 0 or crash_high <= crash_low:
            return None

        drop_pct = float((crash_high - crash_low) / crash_high)
        if float(min_drop_pct or 0.0) > 0 and drop_pct < float(min_drop_pct):
            return None

        ts_high = None
        ts_low = None
        try:
            ts_high = df_window.index[crash_high_idx]
        except Exception:
            ts_high = None
        try:
            ts_low = df_window.index[crash_low_idx]
        except Exception:
            ts_low = None

        return {
            "timeframe": str(timeframe),
            "high": float(crash_high),
            "low": float(crash_low),
            "drop_pct": float(drop_pct),
            "idx_high": int(crash_high_idx + offset),
            "idx_low": int(crash_low_idx + offset),
            "ts_high": str(ts_high) if ts_high is not None else None,
            "ts_low": str(ts_low) if ts_low is not None else None,
            "source_cols": {"high": high_col, "low": low_col},
            "pivots": {
                "high_idx": [int(i + offset) for i in piv_hi_idx.tolist()],
                "low_idx": [int(i + offset) for i in piv_lo_idx.tolist()],
            },
        }

    def _normalize_smart_recovery_config(self, raw_cfg: Any) -> Dict[str, Any]:
        """
        Normalize Smart Recovery config into the internal schema used by the strategy.

        Supports two schemas:
        - Legacy/internal: smart_recovery.trigger/crash_leg/candidates/reachability/barrier/...
        - User-friendly:  smart_recovery.activation/targets/filters (mapped to internal)
        """
        if not isinstance(raw_cfg, dict):
            return {}

        if not any(key in raw_cfg for key in ("activation", "targets", "filters")):
            return raw_cfg

        def _safe_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        def _pct_or_fraction(value: Any, default_fraction: float) -> float:
            v = _safe_float(value, default_fraction)
            # If user passes "1.2" meaning 1.2%, normalize to 0.012.
            return (v / 100.0) if v > 1.0 else v

        activation = raw_cfg.get("activation") or {}
        if not isinstance(activation, dict):
            activation = {}
        targets = raw_cfg.get("targets") or {}
        if not isinstance(targets, dict):
            targets = {}
        filters = raw_cfg.get("filters") or {}
        if not isinstance(filters, dict):
            filters = {}

        drop_magnitude_pct = activation.get("drop_magnitude_pct")
        band_compression_ratio = activation.get("band_compression_ratio")

        lookback_bars = int(targets.get("lookback_bars", 240) or 240)
        min_leg_atr_mult = _safe_float(targets.get("min_leg_atr_mult", 2.0), 2.0)
        fibo_levels = targets.get("fibo_levels", [0.236, 0.382])
        atr_ext = targets.get("atr_extensions", [1.5, 2.0])

        max_reachability_atr = _safe_float(filters.get("max_reachability_atr", 3.0), 3.0)
        barrier_penalty = _safe_float(filters.get("barrier_penalty", 1.0), 1.0)
        penalty_points = int(round(max(0.0, float(barrier_penalty))))

        norm: Dict[str, Any] = {
            "enabled": bool(raw_cfg.get("enabled", False)),
            "trigger": {
                "shock": {
                    "enabled": True,
                    "min_drop_pct": _pct_or_fraction(drop_magnitude_pct, 0.05),
                    "min_shock_score": _safe_float(activation.get("min_shock_score", 0.60), 0.60),
                    "require_dyn_gate_armed": bool(activation.get("require_dyn_gate_armed", False)),
                },
                "compression": {
                    "enabled": True,
                    "band_width_atr_ratio_max": _safe_float(band_compression_ratio, 1.20),
                },
            },
            "crash_leg": {
                "timeframe": str(targets.get("timeframe", "5m") or "5m").strip().lower(),
                "lookback_bars": lookback_bars,
                "pivot_left": int(targets.get("pivot_left", 3) or 3),
                "pivot_right": int(targets.get("pivot_right", 3) or 3),
                "min_leg_atr_mult": min_leg_atr_mult,
            },
            "candidates": {
                "fibo_levels": fibo_levels,
                "atr_mults": atr_ext,
                "include_pivot": True,
                "include_band": True,
            },
            "reachability": {
                "max_atr_mult": max_reachability_atr,
            },
            "barrier": {
                "penalty_points": penalty_points,
            },
            "logging": raw_cfg.get("logging", {"mode": "decisions"}),
            "on_no_valid_candidates": raw_cfg.get("on_no_valid_candidates", "skip_trade"),
        }
        return norm

    def _calculate_smart_recovery_tp(
        self,
        *,
        symbol: str,
        entry_price: float,
        stop_price: float,
        atr_value: float,
        min_tp_pct: float,
        baseline_target_price: float,
        current_target_price: float,
        tp_band_meta: Optional[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        crash_leg: Optional[Dict[str, Any]],
        triggers: Dict[str, Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Smart Recovery TP: structural target selection for post-crash consolidation.

        Candidate pool:
          - Fibo retracements of the crash leg (0.236, 0.382)
          - ATR projections (Entry + k*ATR)
          - Nearest pivot/liquidity levels (pivot highs)
          - Optional: current/band targets as low-priority fallbacks

        Filters:
          - Reachability: distance_to_tp <= max_atr_mult * ATR
          - Strategy min R/R: rr >= self.min_rr_ratio
        """
        risk = float(entry_price - stop_price)
        if risk <= 0:
            return None, {
                "enabled": True,
                "error": "invalid_risk_distance",
                "entry": entry_price,
                "stop": stop_price,
            }

        def _safe_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        reach_cfg = cfg.get("reachability", {}) if isinstance(cfg.get("reachability", {}), dict) else {}
        max_atr_mult = _safe_float(reach_cfg.get("max_atr_mult", 3.0), 3.0)
        max_distance = float(max_atr_mult) * float(atr_value) if atr_value and atr_value > 0 else None

        cand_cfg = cfg.get("candidates", {}) if isinstance(cfg.get("candidates", {}), dict) else {}
        fibo_levels = cand_cfg.get("fibo_levels", [0.236, 0.382])
        if isinstance(fibo_levels, str):
            fibo_levels = [x.strip() for x in fibo_levels.split(",") if x.strip()]
        fibo_levels = [float(x) for x in fibo_levels if x is not None]

        atr_mults = cand_cfg.get("atr_mults", [1.5, 2.0])
        if isinstance(atr_mults, str):
            atr_mults = [x.strip() for x in atr_mults.split(",") if x.strip()]
        atr_mults = [float(x) for x in atr_mults if x is not None]

        include_band = bool(cand_cfg.get("include_band", True))
        include_pivot = bool(cand_cfg.get("include_pivot", True))

        pri_cfg = cfg.get("priority", {}) if isinstance(cfg.get("priority", {}), dict) else {}
        score_map = {
            "fibo": int(pri_cfg.get("fibo", 3) or 3),
            "pivot": int(pri_cfg.get("pivot", 3) or 3),
            "atr": int(pri_cfg.get("atr", 2) or 2),
            "band": int(pri_cfg.get("band", 1) or 1),
        }

        barrier_cfg = cfg.get("barrier", {}) if isinstance(cfg.get("barrier", {}), dict) else {}
        barrier_penalty_points = int(barrier_cfg.get("penalty_points", 1) or 1)

        nearest_pivot_high: Optional[Dict[str, Any]] = None
        if include_pivot and isinstance(market_data, dict):
            df_5m = market_data.get("5m") or market_data.get("df_5m")
            df_5m = self._as_closed_df(df_5m) if isinstance(df_5m, pd.DataFrame) else None
            if isinstance(df_5m, pd.DataFrame) and not df_5m.empty:
                try:
                    piv_cfg = cfg.get("crash_leg", {}) if isinstance(cfg.get("crash_leg", {}), dict) else {}
                    left = int(piv_cfg.get("pivot_left", 3) or 3)
                    right = int(piv_cfg.get("pivot_right", 3) or 3)
                    lookback = int(piv_cfg.get("lookback_bars", 240) or 240)
                    df_w = df_5m.tail(lookback) if lookback > 0 and len(df_5m) > lookback else df_5m
                    high_col = "high" if "high" in df_w.columns else "close"
                    highs = df_w[high_col].astype(float).values
                    piv_hi = confirmed_pivot_highs(highs, left=left, right=right)
                    idx = np.flatnonzero(piv_hi)
                    if idx.size > 0:
                        pivot_prices = highs[idx]
                        above = [(float(px), int(i)) for px, i in zip(pivot_prices.tolist(), idx.tolist()) if float(px) > float(entry_price)]
                        if above:
                            px_sel, i_sel = min(above, key=lambda x: x[0] - float(entry_price))
                            dist = float(px_sel - float(entry_price))
                            dist_atr = (dist / float(atr_value)) if atr_value and atr_value > 0 else None
                            nearest_pivot_high = {
                                "price": float(px_sel),
                                "idx": int(i_sel),
                                "distance": dist,
                                "distance_atr": dist_atr,
                                "timeframe": "5m",
                                "high_col": high_col,
                            }
                except Exception:
                    nearest_pivot_high = None

        min_tp_target = float(entry_price) * (1.0 + float(min_tp_pct or 0.0))

        candidates: List[Dict[str, Any]] = []

        def _add_candidate(*, kind: str, price: float, meta: Optional[Dict[str, Any]] = None) -> None:
            if price is None or not math.isfinite(float(price)):
                return
            px = float(price)
            dist = float(px - float(entry_price))
            rr = dist / float(risk) if float(risk) > 0 else 0.0
            rr_margin = rr - float(self.min_rr_ratio)
            dist_atr = (dist / float(atr_value)) if atr_value and atr_value > 0 else None
            reachable = True
            if max_distance is not None:
                reachable = dist <= float(max_distance)
            pass_min_tp = px >= min_tp_target
            pass_min_rr = rr >= float(self.min_rr_ratio)
            pass_side = px > float(entry_price)

            reject_reasons: List[str] = []
            if not pass_side:
                reject_reasons.append("tp_not_above_entry")
            if not pass_min_tp:
                reject_reasons.append("tp_below_min_tp_pct")
            if not reachable:
                if dist_atr is not None:
                    reject_reasons.append(f"unreachable_{float(dist_atr):.1f}ATR")
                else:
                    reject_reasons.append("unreachable_by_atr")
            if not pass_min_rr:
                reject_reasons.append("rr_below_strategy_min")

            kind_base = kind.split("_", 1)[0]
            score = int(score_map.get(kind_base, 0))
            barrier_meta: Dict[str, Any] = {}
            if (
                nearest_pivot_high is not None
                and kind != "pivot_high"
                and (nearest_pivot_high.get("price") is not None)
                and float(entry_price) < float(nearest_pivot_high["price"]) < float(px)
                and barrier_penalty_points > 0
            ):
                penalty = min(int(barrier_penalty_points), int(score))
                score = int(score) - int(penalty)
                barrier_meta = {
                    "nearest_pivot_high": float(nearest_pivot_high["price"]),
                    "penalty_points": int(penalty),
                }

            candidates.append(
                {
                    "type": str(kind),
                    "price": px,
                    "distance": dist,
                    "distance_atr": dist_atr,
                    "rr": rr,
                    "rr_margin": rr_margin,
                    "reachable": bool(reachable),
                    "pass_min_rr": bool(pass_min_rr),
                    "pass_min_tp": bool(pass_min_tp),
                    "score": int(score),
                    "rejected_reasons": reject_reasons,
                    "meta": {**(meta or {}), **({"barrier": barrier_meta} if barrier_meta else {})},
                }
            )

        crash_cfg = cfg.get("crash_leg", {}) if isinstance(cfg.get("crash_leg", {}), dict) else {}
        min_leg_atr_mult = _safe_float(crash_cfg.get("min_leg_atr_mult", 2.0), 2.0)
        leg_span = None
        leg_height_atr = None
        leg_valid = False

        # 1) Fibo retracements from crash leg
        if crash_leg and crash_leg.get("high") and crash_leg.get("low"):
            hi = float(crash_leg["high"])
            lo = float(crash_leg["low"])
            span = hi - lo
            leg_span = float(span)
            if atr_value and atr_value > 0:
                leg_height_atr = float(leg_span / float(atr_value)) if leg_span is not None else None
                leg_valid = bool(leg_span >= (float(min_leg_atr_mult) * float(atr_value)))
            else:
                leg_height_atr = None
                leg_valid = False

            if leg_span > 0 and leg_valid:
                for lvl in fibo_levels:
                    try:
                        lvl_f = float(lvl)
                    except Exception:
                        continue
                    if not (0.0 < lvl_f < 1.0):
                        continue
                    px = lo + leg_span * lvl_f
                    _add_candidate(kind=f"fibo_{lvl_f:.3f}", price=px, meta={"level": lvl_f})

        # 2) ATR projections
        if atr_value and atr_value > 0:
            for k in atr_mults:
                try:
                    kf = float(k)
                except Exception:
                    continue
                if kf <= 0:
                    continue
                _add_candidate(kind=f"atr_{kf:.2f}", price=float(entry_price) + (kf * float(atr_value)), meta={"k": kf})

        # Always include baseline TP as a reference candidate (kept compatible with existing TP realignment logic)
        _add_candidate(kind="atr_baseline", price=float(baseline_target_price), meta={"source": "baseline"})

        # 3) Liquidity/pivot: nearest pivot high above entry (5m)
        if nearest_pivot_high is not None:
            _add_candidate(kind="pivot_high", price=float(nearest_pivot_high["price"]), meta={"idx": int(nearest_pivot_high["idx"])})

        # Optional: standard-band targets as lowest-priority fallbacks
        if include_band and tp_band_meta and isinstance(tp_band_meta.get("band"), dict):
            try:
                band = tp_band_meta["band"]
                _add_candidate(kind="band_high", price=float(band.get("band_high")))
            except Exception:
                pass

        valid = [c for c in candidates if not c.get("rejected_reasons")]
        selected = None
        if valid:
            # Prefer stronger structure first, then the closest reachable target (realism over distance)
            selected = sorted(
                valid,
                key=lambda c: (
                    -int(c.get("score", 0)),
                    float(c.get("distance", 0.0)),
                    -float(c.get("rr_margin", 0.0)),
                ),
            )[0]

        rejection_reasons: Dict[str, Any] = {}
        for cand in candidates:
            reasons = cand.get("rejected_reasons") or []
            if reasons:
                rejection_reasons[str(cand.get("type"))] = "|".join([str(r) for r in reasons])

        # If fibo candidates were suppressed due to a low-quality crash leg, surface that explicitly.
        if crash_leg and fibo_levels and leg_span is not None and not leg_valid:
            for lvl in fibo_levels:
                try:
                    lvl_f = float(lvl)
                except Exception:
                    continue
                if not (0.0 < lvl_f < 1.0):
                    continue
                rejection_reasons.setdefault(f"fibo_{lvl_f:.3f}", "leg_too_small")

        meta_out: Dict[str, Any] = {
            "enabled": True,
            "tp_mode": "Smart_Recovery",
            "triggers": triggers,
            "crash_leg_levels": crash_leg,
            "leg_quality": {
                "height_atr": float(leg_height_atr) if leg_height_atr is not None else None,
                "min_leg_atr_mult": float(min_leg_atr_mult),
                "valid": bool(leg_valid),
            },
            "reachability": {
                "max_atr_mult": float(max_atr_mult),
                "max_distance": float(max_distance) if max_distance is not None else None,
            },
            "barrier": {
                "nearest_pivot_high": nearest_pivot_high,
                "penalty_points": int(barrier_penalty_points),
            },
            "min_rr_required": float(self.min_rr_ratio),
            "min_tp_pct": float(min_tp_pct),
            "candidates": candidates,
            "rejection_reasons": rejection_reasons,
            "risk_data": {
                "cap": None,
                "floor": float(self.min_rr_ratio),
                "ppo_mult": None,
            },
            "selected_tp": selected,
        }

        if not selected:
            return None, meta_out
        return float(selected["price"]), meta_out
    
    def signal(self, df_30m: pd.DataFrame, 
               df_1h: pd.DataFrame = None,
               regime_data: Optional[Dict] = None,
               symbol: str = None,
               market_data: Optional[Dict] = None,
               ml_context=None) -> Optional[Dict]:
        """
        Generate adaptive trading signal based on market regime and ML insights.
        Logs the specific reason if no signal is generated.
        """
        symbol_display = symbol or "UNKNOWN"
        log_prefix = f"[{self.strategy_name.upper()}/{symbol_display}]"

        # --- Data Validation Step ---
        validation_passed, reason = self._validate_input_data(df_30m, df_1h, regime_data, symbol_display)
        if not validation_passed:
            logger.info(f"🚫 {log_prefix} No Signal: {reason}")
            return None
        
        try:
            includes_forming = bool(df_30m.attrs.get("includes_forming", False))
            forming_open_ms_attr = df_30m.attrs.get("forming_ts")

            fallback_reason = df_30m.attrs.get("fallback_reason", None)
            # Backwards-compat: older pipeline versions stored the string "none".
            if isinstance(fallback_reason, str) and fallback_reason.strip().lower() in ("none", ""):
                fallback_reason = None

            forming_last_update_ts = df_30m.attrs.get("forming_last_update_ts")
            forming_update_age_ms = df_30m.attrs.get("forming_update_age_ms")
            now_ms = int(time.time() * 1000)
            self._update_dyn_shadow(symbol_display, df_30m, now_ms, fallback_reason)

            df_closed = df_30m
            df_used = df_30m
            forming_row = None
            forming_open_ms = None
            used_forming = False
            rsi_source = "closed"
            trigger_price_source = "closed_close"

            if includes_forming and len(df_30m) >= 2:
                df_closed = df_30m.iloc[:-1]
                forming_row = df_30m.iloc[-1]
                try:
                    forming_open_ms = int(forming_open_ms_attr or int(forming_row.name.timestamp() * 1000))
                except Exception:
                    forming_open_ms = None
            else:
                includes_forming = False
                forming_open_ms = None
                df_closed = df_30m

            # Strategy should only *use* forming data when it exists AND pipeline indicates no fallback.
            used_forming = bool(includes_forming and fallback_reason is None)
            if used_forming:
                df_used = df_30m
                rsi_source = "live"
                trigger_price_source = "forming_close"
            else:
                df_used = df_closed
                rsi_source = "closed"
                trigger_price_source = "closed_close"

            if self.debug_logging:
                logger.debug(f"{log_prefix} Data sufficiency check: total_rows={len(df_closed)}")

            try:
                # Always use the last CLOSED candle for indicator context (EMA/ATR/etc).
                trend_row = df_closed.iloc[-1]
            except IndexError:
                logger.info(f"🚫 {log_prefix} No Signal: Insufficient 30m data to generate a signal (IndexError).")
                return None

            required_cols = ['close', 'rsi', 'atr', 'ema_fast']
            missing = [c for c in required_cols if c not in trend_row.index]
            if missing:
                logger.warning(f"{log_prefix} Missing required columns in latest closed row: {missing}")
                return None
            if any(pd.isna(trend_row[c]) for c in required_cols):
                logger.info(f"{log_prefix} Latest closed row has NaN in required columns; skipping this tick.")
                return None

            if regime_data is None:
                regime_data = DEFAULT_MARKET_REGIME.copy()

            # --- Initial Data Extraction ---
            if 'close' not in trend_row.index:
                logger.warning(f"🚫 {log_prefix} No Signal: 'close' column missing in the latest data.")
                return None

            if used_forming:
                rsi_series = rsi(df_used['close'])
            elif 'rsi' in df_closed.columns:
                rsi_series = df_closed['rsi']
            else:
                rsi_series = rsi(df_closed['close'])

            rsi_val = float(rsi_series.iloc[-1])
            rsi_prev = None
            try:
                if len(rsi_series) >= 2:
                    prev_raw = rsi_series.iloc[-2]
                    if prev_raw is not None and not pd.isna(prev_raw):
                        rsi_prev = float(prev_raw)
            except Exception:
                rsi_prev = None
            close_price = float(trend_row['close'])
            forming_price = float(forming_row['close']) if forming_row is not None else close_price
            forming_low = float(forming_row['low']) if forming_row is not None and 'low' in forming_row else None
            forming_high = float(forming_row['high']) if forming_row is not None and 'high' in forming_row else None
            ema_fast = float(trend_row.get('ema_fast', 0))
            ema_mid = float(trend_row.get('ema_mid', 0))
            atr_value = float(trend_row.get('atr', close_price * 0.02))
            volume_val = None
            if forming_row is not None and 'volume' in forming_row:
                volume_val = float(forming_row['volume'])
            elif 'volume' in trend_row:
                volume_val = float(trend_row['volume'])

            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }

            # Hardening (Issue #454):
            # - Never log "Hybrid fallback: none"
            # - Only emit "reverting to closed-only" when we explicitly chose NOT to use forming
            #   AND the pipeline provided a real fallback reason.
            if (not used_forming) and (fallback_reason is not None):
                if fallback_reason == "pivot_grace_prev_bucket":
                    logger.info(
                        f"{log_prefix} Hybrid downgrade: pivot_grace_prev_bucket. "
                        "Previous bucket still updating within grace window; using closed-only for safety."
                    )
                else:
                    logger.warning(
                        f"{log_prefix} Hybrid fallback: {fallback_reason}. Reverting to closed-only data."
                    )

            trigger_cfg_source = str(self.base_cfg.get("adaptive_ob_trigger_price_source", "mid")).lower()
            resolved_trigger_source = trigger_cfg_source
            trigger_fallback_chain = "none"
            trigger_price = forming_price
            if used_forming:
                if self.market_data_pipeline:
                    trigger_price, resolved_trigger_source, trigger_fallback_chain = self.market_data_pipeline.get_live_trigger_price(
                        symbol=symbol_display,
                        timeframe="30m",
                        source=trigger_cfg_source,
                        forming_close=forming_price,
                    )
                if trigger_price is None:
                    trigger_price = forming_price
                    resolved_trigger_source = "forming_close"
                    trigger_fallback_chain = f"{trigger_fallback_chain}->forming_close_none" if trigger_fallback_chain != "none" else "forming_close_none"
            else:
                resolved_trigger_source = "closed_close"
                trigger_price = close_price
                trigger_fallback_chain = "closed_only"

            trigger_price_source = resolved_trigger_source

            trigger_price_log = f"{trigger_price:.2f}" if trigger_price is not None else "None"

            if includes_forming or self.debug_logging:
                logger.info(
                    f"{log_prefix} Hybrid meta | includes_forming={includes_forming} "
                    f"used_forming={used_forming} "
                    f"forming_open_time={forming_open_ms} rsi_source={rsi_source} "
                    f"trigger_price_source={resolved_trigger_source} fallback_reason={(fallback_reason if fallback_reason is not None else 'none')} "
                    f"forming_last_update_ts={forming_last_update_ts} forming_update_age_ms={forming_update_age_ms} "
                    f"fallback_chain={trigger_fallback_chain} trigger_price={trigger_price_log}"
                )

            persist_mode = str(self.base_cfg.get("adaptive_ob_persistency_mode", "time")).lower()
            persist_seconds = float(self.base_cfg.get("adaptive_ob_persistency_seconds", 5))
            persist_min_samples = int(self.base_cfg.get("adaptive_ob_persistency_min_samples", 2))
            persist_wick_k = float(self.base_cfg.get("adaptive_ob_wick_closeness_k", 0.25))
            self._persistency_cfg = {
                "mode": persist_mode,
                "seconds": max(persist_seconds, 0.0),
                "min_samples": max(persist_min_samples, 1),
                "wick_closeness_k": max(min(persist_wick_k, 1.0), 0.0),
            }
            
            # --- Adaptive Threshold Calculation ---
            adaptive_rsi_threshold = self.get_symbol_specific_threshold(symbol_display)
            if adaptive_rsi_threshold is not None:
                if self.debug_logging: logger.info(f"ℹ️ {log_prefix} Using symbol-specific RSI threshold: {adaptive_rsi_threshold:.2f}")
            else:
                adaptive_rsi_threshold = self.get_adaptive_rsi_threshold(market_regime)

            extreme_bypass_active, extreme_bypass_meta = self._check_extreme_bypass(
                symbol=symbol_display,
                close_price=close_price,
                atr_value=atr_value,
                market_data=market_data,
            )

            # Trend confirmation: if market is in a strong downswing (ema_fast < ema_mid), demand deeper RSI
            ema_trend_penalty = float(self.strategy_config.get('trend_confirmation_rsi_penalty', 5.0))
            min_adaptive_rsi = float(self.strategy_config.get('trend_confirmation_min_rsi', 8.0))
            ema_gap_pct_min = float(self.strategy_config.get('trend_confirmation_ema_gap_pct_min', 0.0) or 0.0)
            gap_on_raw = self.strategy_config.get('trend_confirmation_ema_gap_pct_on', None)
            gap_off_raw = self.strategy_config.get('trend_confirmation_ema_gap_pct_off', None)
            gap_on = None
            gap_off = None
            if gap_on_raw is not None:
                try:
                    gap_on = float(gap_on_raw)
                except Exception:
                    gap_on = None
            if gap_off_raw is not None:
                try:
                    gap_off = float(gap_off_raw)
                except Exception:
                    gap_off = None
            use_hysteresis = (gap_on is not None) or (gap_off is not None)
            if use_hysteresis and gap_on is None:
                gap_on = gap_off
            if use_hysteresis and gap_off is None:
                gap_off = gap_on
            if use_hysteresis and gap_on is not None and gap_off is not None and gap_off > gap_on:
                gap_off = gap_on
            ema_gap_pct = None
            if ema_mid > 0:
                ema_gap_pct = (ema_mid - ema_fast) / ema_mid
            trend_bias_active = False
            prev_active = self._trend_penalty_state_by_symbol.get(symbol_display, False)
            active = prev_active
            reason = None
            if ema_fast > 0 and ema_mid > 0 and ema_fast < ema_mid and extreme_bypass_active:
                try:
                    extreme_bypass_meta["trend_penalty_skipped"] = True
                except Exception:
                    pass
                logger.warning(
                    "[OB-EXTREME-BYPASS] %s skipping trend RSI penalty (ema_fast < ema_mid)",
                    log_prefix,
                )
            if extreme_bypass_active:
                active = False
                if prev_active:
                    reason = "turned_off_extreme_bypass"
            elif not (ema_fast > 0 and ema_mid > 0):
                active = False
                if prev_active:
                    reason = "ema_mid_zero"
            elif not (ema_fast < ema_mid):
                active = False
                if prev_active:
                    reason = "turned_off_ema_not_below_mid"
            elif ema_gap_pct is None:
                active = False
                if prev_active:
                    reason = "ema_mid_zero"
            else:
                if use_hysteresis:
                    if (not prev_active) and (ema_gap_pct >= gap_on):
                        active = True
                        reason = "turned_on_gap_ge_on"
                    elif prev_active and (ema_gap_pct <= gap_off):
                        active = False
                        reason = "turned_off_gap_le_off"
                    else:
                        active = prev_active
                else:
                    active = ema_gap_pct >= ema_gap_pct_min
                    if (not prev_active) and active:
                        reason = "turned_on_gap_ge_min"
                    elif prev_active and (not active):
                        reason = "turned_off_gap_below_min"

            base_threshold = adaptive_rsi_threshold
            if active:
                new_threshold = max(min_adaptive_rsi, adaptive_rsi_threshold - ema_trend_penalty)
                if new_threshold != adaptive_rsi_threshold:
                    logger.debug(
                        f"⚠️ {log_prefix} Trend confirmation active: EMA fast ${ema_fast:,.2f} below EMA mid ${ema_mid:,.2f}."
                        f" Adjusting RSI threshold {adaptive_rsi_threshold:.2f} → {new_threshold:.2f}."
                    )
                    adaptive_rsi_threshold = new_threshold
                    trend_bias_active = True

            if active != prev_active:
                self._trend_penalty_state_by_symbol[symbol_display] = active
                if reason is None:
                    reason = "state_change"
                ema_gap_pct_log = f"{ema_gap_pct:.6f}" if ema_gap_pct is not None else "na"
                gap_on_log = f"{gap_on:.6f}" if gap_on is not None else "na"
                gap_off_log = f"{gap_off:.6f}" if gap_off is not None else "na"
                logger.info(
                    "Trend penalty state | symbol=%s | active=%s | reason=%s | ema_fast=%.5f | ema_mid=%.5f | "
                    "ema_gap_pct=%s | gap_min=%.6f | gap_on=%s | gap_off=%s | base_thr=%.2f | "
                    "effective_thr=%.2f | extreme_bypass=%s",
                    symbol_display,
                    active,
                    reason,
                    ema_fast,
                    ema_mid,
                    ema_gap_pct_log,
                    ema_gap_pct_min,
                    gap_on_log,
                    gap_off_log,
                    base_threshold,
                    adaptive_rsi_threshold,
                    extreme_bypass_active,
                )
            else:
                self._trend_penalty_state_by_symbol[symbol_display] = active

            # Volume logic centralized in StrategyCoordinator (Issue #450)
            # Legacy volume confirmation removed.

            timeframe = "30m"
            last_closed_ts_ms = extract_last_closed_ts_ms(df_closed)
            df_meta = extract_df_meta(df_closed)

            def _shadow_ob(decision: str, fail_reason: str = "", extra: Optional[Dict] = None) -> None:
                if not shadow_enabled():
                    return
                try:
                    payload = {
                        "event": "strategy_shadow_eval",
                        "strategy": "adaptive_ob",
                        "symbol": symbol_display,
                        "timeframe": timeframe,
                        "last_closed_ts": last_closed_ts_ms,
                        **df_meta,
                        "close": trigger_price,
                        "rsi": rsi_val,
                        "rsi_threshold": adaptive_rsi_threshold,
                        "rsi_delta": rsi_val - adaptive_rsi_threshold if adaptive_rsi_threshold is not None else None,
                        "ema_fast": ema_fast,
                        "ema_mid": ema_mid,
                        "atr": atr_value,
                        "includes_forming": includes_forming,
                        "used_forming": used_forming,
                        "forming_open_time": forming_open_ms,
                        "forming_last_update_ts": forming_last_update_ts,
                        "forming_update_age_ms": forming_update_age_ms,
                        "rsi_source": rsi_source,
                        "trigger_price_source": trigger_price_source,
                        "fallback_reason": fallback_reason,
                        "trend_penalty_applied": trend_bias_active,
                        "trend_penalty": ema_trend_penalty,
                        "decision": decision,
                        "fail_reason": fail_reason,
                    }
                    if extra:
                        payload.update(extra)
                    emit_shadow_log(logger, payload, "adaptive_ob", symbol_display, timeframe, last_closed_ts_ms)
                except Exception:
                    return

            # --- Core Signal Condition Checks with Tracing ---
            if self.debug_logging:
                logger.info(f"🔍 {log_prefix} Checking conditions...")
                logger.info(f"  - Regime: {market_regime['trend']}, Volatility: {market_regime['volatility']}")
                logger.info(f"  - Price: ${trigger_price:,.2f}, RSI: {rsi_val:.2f}, EMA Fast: ${ema_fast:,.2f}, EMA Mid: ${ema_mid:,.2f}")
                logger.info(f"  - RSI Threshold: {adaptive_rsi_threshold:.2f}")

            # -------------------------------------------------------------------------
            # CUSTOM DOWNTREND VETO (Safety Patch)
            # Purpose: In strong downtrends (Price < EMA50 and ADX > 30),
            # force the RSI threshold down to a maximum of 25.0.
            # This prevents buying "mild dips" during a crash.
            # -------------------------------------------------------------------------
            try:
                # Safely fetch ADX from trend_row (returns None if missing)
                # trend_row is available in local scope
                adx_val = float(trend_row.get("adx")) if trend_row is not None and "adx" in trend_row else None
            except Exception:
                adx_val = None

            # 'ema_mid' is the alias for EMA50 in this file
            ema_long = ema_mid 

            # Use close_price as reference
            price_ref = close_price

            if (
                adx_val is not None             # Data exists
                and not pd.isna(adx_val)        # Not NaN
                and ema_long and ema_long > 0   # EMA is valid
                and price_ref < ema_long        # Bearish Structure (Price below EMA)
                and adx_val > 30.0              # Strong Trend Strength
                and adaptive_rsi_threshold > 25.0 # Current threshold is too generous (high)
            ):
                # Log the intervention using module-level logger
                logger.info(f"{symbol_display} Downtrend Veto: lowering RSI threshold {adaptive_rsi_threshold:.1f} -> 25.0 (Price < EMA50 & ADX {adx_val:.1f} > 30)")

                adaptive_rsi_threshold = 25.0
            # -------------------------------------------------------------------------

            # --- Low-volume / downtrend reversal confirmation (optional guardrail) ---
            reversal_confirmed = False
            reversal_meta: Dict[str, Any] = {
                "used_forming": used_forming,
                "bullish_candle": None,
                "close_above_prev_close": None,
                "close_in_upper_range": None,
                "two_consecutive_bullish_closes": None,
                "close_above_ema_fast": None,
                "strong_confirmed": False,
                "confirmed": False,
            }
            try:
                if used_forming and forming_row is not None:
                    f_open = float(forming_row.get('open')) if 'open' in forming_row else None
                    f_close = float(forming_row.get('close')) if 'close' in forming_row else None
                    f_low = float(forming_row.get('low')) if 'low' in forming_row else None
                    f_high = float(forming_row.get('high')) if 'high' in forming_row else None

                    bullish = (f_open is not None and f_close is not None and f_close > f_open)
                    prev_close = float(close_price)
                    close_above_prev = (f_close is not None and f_close > prev_close)
                    close_in_upper = None
                    if f_close is not None and f_low is not None and f_high is not None and f_high > f_low:
                        close_in_upper = ((f_close - f_low) / (f_high - f_low)) >= 0.60

                    close_above_ema_fast = None
                    if f_close is not None and ema_fast is not None and ema_fast > 0:
                        close_above_ema_fast = f_close > float(ema_fast)

                    two_bullish = None
                    try:
                        if len(df_closed) >= 2:
                            last_closed = df_closed.iloc[-1]
                            prev_closed = df_closed.iloc[-2]
                            l_open = float(last_closed.get('open')) if 'open' in last_closed else None
                            l_close = float(last_closed.get('close')) if 'close' in last_closed else None
                            p_open = float(prev_closed.get('open')) if 'open' in prev_closed else None
                            p_close = float(prev_closed.get('close')) if 'close' in prev_closed else None
                            two_bullish = bool(
                                l_open is not None and l_close is not None and l_close > l_open
                                and p_open is not None and p_close is not None and p_close > p_open
                            )
                    except Exception:
                        two_bullish = None

                    reversal_confirmed = bool(bullish and close_above_prev and (close_in_upper is not False))
                    strong_confirmed = bool(
                        reversal_confirmed
                        and (
                            close_in_upper is True
                            or close_above_ema_fast is True
                            or two_bullish is True
                        )
                    )
                    reversal_meta.update(
                        {
                            "bullish_candle": bullish,
                            "close_above_prev_close": close_above_prev,
                            "close_in_upper_range": close_in_upper,
                            "two_consecutive_bullish_closes": two_bullish,
                            "close_above_ema_fast": close_above_ema_fast,
                            "strong_confirmed": strong_confirmed,
                            "confirmed": reversal_confirmed,
                        }
                    )
                else:
                    # Closed-only fallback: require last closed candle to be bullish and above previous close.
                    last_row = df_closed.iloc[-1]
                    prev_row = df_closed.iloc[-2] if len(df_closed) >= 2 else None
                    last_open = float(last_row.get('open')) if 'open' in last_row else None
                    last_close = float(last_row.get('close')) if 'close' in last_row else None
                    prev_close = float(prev_row.get('close')) if prev_row is not None and 'close' in prev_row else None

                    bullish = (
                        last_open is not None and last_close is not None and last_close > last_open
                    )
                    close_above_prev = (
                        prev_close is not None and last_close is not None and last_close > prev_close
                    )
                    reversal_confirmed = bool(bullish and close_above_prev)
                    close_above_ema_fast = None
                    if last_close is not None and ema_fast is not None and ema_fast > 0:
                        close_above_ema_fast = last_close > float(ema_fast)

                    two_bullish = None
                    try:
                        if len(df_closed) >= 2:
                            last_closed = df_closed.iloc[-1]
                            prev_closed = df_closed.iloc[-2]
                            l_open = float(last_closed.get('open')) if 'open' in last_closed else None
                            l_close = float(last_closed.get('close')) if 'close' in last_closed else None
                            p_open = float(prev_closed.get('open')) if 'open' in prev_closed else None
                            p_close = float(prev_closed.get('close')) if 'close' in prev_closed else None
                            two_bullish = bool(
                                l_open is not None and l_close is not None and l_close > l_open
                                and p_open is not None and p_close is not None and p_close > p_open
                            )
                    except Exception:
                        two_bullish = None

                    strong_confirmed = bool(
                        reversal_confirmed
                        and (
                            close_above_ema_fast is True
                            or two_bullish is True
                        )
                    )
                    reversal_meta.update(
                        {
                            "bullish_candle": bullish,
                            "close_above_prev_close": close_above_prev,
                            "close_in_upper_range": None,
                            "two_consecutive_bullish_closes": two_bullish,
                            "close_above_ema_fast": close_above_ema_fast,
                            "strong_confirmed": strong_confirmed,
                            "confirmed": reversal_confirmed,
                        }
                    )
            except Exception:
                reversal_confirmed = False
                reversal_meta["confirmed"] = False

            downtrend_context = bool(
                adx_val is not None
                and not pd.isna(adx_val)
                and ema_long
                and ema_long > 0
                and price_ref < ema_long
                and adx_val > 30.0
            )

            # NOTE: Downtrend+LOW-volume guardrails are enforced downstream (StrategyCoordinator)
            # because volume_bucket is computed there. Here we only emit metadata.

            # 1. RSI Condition Check
            if rsi_val > adaptive_rsi_threshold:
                logger.info(f"🚫 {log_prefix} No Signal: RSI ({rsi_val:.2f}) is above the threshold ({adaptive_rsi_threshold:.2f}).")
                self._reset_persistency(symbol_display)
                _shadow_ob("no_signal_rsi", "rsi_above_threshold")
                self._log_persistency_skipped(
                    symbol_display,
                    log_prefix,
                    "rsi_above_threshold",
                    includes_forming,
                    forming_open_ms,
                    {
                        "rsi": f"{rsi_val:.2f}",
                        "threshold": f"{adaptive_rsi_threshold:.2f}",
                    },
                )
                return None

            # 2. Price vs. EMA Condition Check
            if ema_fast > 0 and trigger_price >= ema_fast:
                logger.info(f"🚫 {log_prefix} No Signal: Price (${trigger_price:,.2f}) is not below the fast EMA (${ema_fast:,.2f}).")
                self._reset_persistency(symbol_display)
                _shadow_ob(
                    "no_signal_price_vs_ema",
                    "price_not_below_ema_fast",
                    {"price_vs_ema_fast": trigger_price - ema_fast},
                )
                self._log_persistency_skipped(
                    symbol_display,
                    log_prefix,
                    "price_not_below_ema_fast",
                    includes_forming,
                    forming_open_ms,
                    {
                        "price": f"{trigger_price:.2f}",
                        "ema_fast": f"{ema_fast:.2f}",
                    },
                )
                return None

            # 3. Volume Check (basic data sanity)
            if volume_val is not None and volume_val <= 0:
                logger.info(f"🚫 {log_prefix} No Signal: Volume is zero or negative.")
                self._reset_persistency(symbol_display)
                _shadow_ob(
                    "no_signal_volume",
                    "non_positive_volume",
                    {"volume": volume_val},
                )
                self._log_persistency_skipped(
                    symbol_display,
                    log_prefix,
                    "non_positive_volume",
                    includes_forming,
                    forming_open_ms,
                    {"volume": volume_val},
                )
                return None
            
            # --- Persistency Guard ---
            guard_passed, persist_meta = self._apply_persistency_guard(
                symbol_display,
                includes_forming,
                forming_open_ms,
                rsi_val,
                adaptive_rsi_threshold,
                trigger_price,
                ema_fast,
                forming_price,
                forming_low,
                forming_high,
            )

            logger.info(
                f"{log_prefix} Persistency | mode={persist_meta['mode']} condition_true={persist_meta['condition_true']} "
                f"elapsed_s={persist_meta['elapsed_s']:.2f} samples={persist_meta['samples']} passed={guard_passed} "
                f"wick_close_to_low={persist_meta.get('wick_closeness')} k={persist_meta.get('wick_closeness_k')} "
                f"wick_passed={persist_meta.get('wick_closeness_passed')} "
                f"wick_skipped={persist_meta.get('wick_closeness_skipped')}"
            )
            if not guard_passed:
                _shadow_ob(
                    "no_signal_persistency",
                    "persistency_not_met",
                    persist_meta,
                )
                return None

            logger.info(f"✅ {log_prefix} Base conditions met. Proceeding to ML & Risk checks.")

            # --- ML-Aware Decision Making ---
            position_size_modifier = 1.0
            ml_enhanced = False
            MIN_ML_CONFIDENCE_THRESHOLD = 0.60

            if ml_context and ml_context.get('is_healthy', False) and ml_context.get('regime_confidence', 0) >= MIN_ML_CONFIDENCE_THRESHOLD:
                ml_enhanced = True

                if (
                    extreme_bypass_active
                    and ml_context.get('regime_prediction') == 'bearish'
                    and ml_context.get('regime_confidence', 0) > 0.7
                ):
                    try:
                        extreme_bypass_meta["ml_veto_bypassed"] = True
                        extreme_bypass_meta["ml_veto_reason"] = "bearish_regime"
                    except Exception:
                        pass
                    logger.warning(
                        "[OB-EXTREME-BYPASS] %s bypassing ML veto (bearish regime)",
                        log_prefix,
                    )

                if (
                    extreme_bypass_active
                    and ml_context.get('price_direction') == 'down'
                    and ml_context.get('price_confidence', 0) > 0.7
                ):
                    try:
                        extreme_bypass_meta["ml_veto_bypassed"] = True
                        extreme_bypass_meta["ml_veto_reason"] = "price_direction_down"
                    except Exception:
                        pass
                    logger.warning(
                        "[OB-EXTREME-BYPASS] %s bypassing ML veto (price_direction=down)",
                        log_prefix,
                    )
                 
                if (
                    ml_context.get('regime_prediction') == 'bearish'
                    and ml_context.get('regime_confidence', 0) > 0.7
                    and (not extreme_bypass_active)
                ):
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong bearish regime detected (confidence: {ml_context.get('regime_confidence', 0):.2%}).")
                    _shadow_ob(
                        "no_signal_ml",
                        "ml_veto",
                        {
                            "ml_regime": ml_context.get('regime_prediction'),
                            "ml_confidence": ml_context.get('regime_confidence'),
                        },
                    )
                    return None
                
                if (
                    ml_context.get('price_direction') == 'down'
                    and ml_context.get('price_confidence', 0) > 0.7
                    and (not extreme_bypass_active)
                ):
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong price down prediction (confidence: {ml_context.get('price_confidence', 0):.2%}).")
                    _shadow_ob(
                        "no_signal_ml",
                        "ml_veto",
                        {
                            "ml_price_direction": ml_context.get('price_direction'),
                            "ml_price_confidence": ml_context.get('price_confidence'),
                        },
                    )
                    return None
                
                if (ml_context.get('regime_prediction') == 'bullish') or (ml_context.get('price_direction') == 'up'):
                    position_size_modifier = 1.0 + (0.25 * ml_context.get('consensus_score', 0))
                    if self.debug_logging: logger.info(f"🧠 {log_prefix} ML Confirmation: Increasing position size modifier to {position_size_modifier:.2f}x.")
                elif ml_context.get('consensus_score', 1.0) < 0.5:
                    position_size_modifier *= 0.75
                    if self.debug_logging: logger.info(f"🧠 {log_prefix} ML Caution: Reducing position size modifier to {position_size_modifier:.2f}x.")

            # --- Final Risk/Reward and Position Sizing ---
            volatility = regime_data.get('volatility', 'normal')
            position_mult = self.calculate_dynamic_position_size(volatility) * position_size_modifier
            
            entry_price = float(trigger_price)
            
            # 🔥 GÜNCELLEME: Config okumaları artık tutarlı bir şekilde `self.strategy_config` üzerinden yapılıyor.
            tp_atr_mult = float(self.strategy_config.get("tp_atr_mult", 2.5))
            sl_atr_mult = float(self.strategy_config.get("sl_atr_mult", 1.2))
            
            # Calculate theoretical ATR-based levels
            theoretical_sl_distance = atr_value * sl_atr_mult
            theoretical_tp_distance = atr_value * tp_atr_mult
            
            theoretical_sl_pct = theoretical_sl_distance / entry_price
            theoretical_tp_pct = theoretical_tp_distance / entry_price
            
            min_tp_pct = float(self.strategy_config.get("min_tp_pct", 0.008))
            max_sl_pct = float(self.strategy_config.get("max_sl_pct", 0.015))
            
            # Apply stop-loss cap CORRECTLY and realign TP if needed
            if theoretical_sl_pct > max_sl_pct:
                # Risk needs capping
                logger.info(f"📊 {log_prefix} [SL Cap Applied] {theoretical_sl_pct:.1%} → {max_sl_pct:.1%}")
                actual_sl_pct = max_sl_pct
                actual_sl_distance = entry_price * actual_sl_pct
                
                # CRITICAL - Realign TP to maintain intended R/R ratio
                intended_rr = tp_atr_mult / sl_atr_mult  # e.g., 2.5/1.2 = 2.08
                adjusted_tp_distance = actual_sl_distance * intended_rr
                actual_tp_pct = adjusted_tp_distance / entry_price
                
                logger.info(f"📊 {log_prefix} [TP Realigned] Maintaining R/R={intended_rr:.2f}")
            else:
                # No capping needed
                actual_sl_pct = theoretical_sl_pct
                actual_tp_pct = theoretical_tp_pct
            
            # Calculate final prices
            stop_price = entry_price * (1 - actual_sl_pct)
            
            # ✅ CRITICAL FIX: Use max() for LONG positions to prevent stop going too low
            # This safety net ensures we never exceed max_sl_pct even with rounding errors
            # or edge cases in the logic above (defensive programming for financial systems)
            stop_price = max(stop_price, entry_price * (1 - max_sl_pct))

            # Baseline TP (ATR-based) - kept as fallback/reference
            baseline_tp_pct = float(actual_tp_pct)
            baseline_target_price = entry_price * (1 + baseline_tp_pct)
            baseline_target_price = max(baseline_target_price, entry_price * (1 + min_tp_pct))

            target_price = float(baseline_target_price)
            tp_mode = "atr"
            tp_band_meta: Optional[Dict[str, Any]] = None

            # Optional: resistance-band-based TP candidate (feature-flagged; default off)
            tp_band_cfg = self.strategy_config.get("tp_band") or {}
            if isinstance(tp_band_cfg, dict) and tp_band_cfg.get("enabled", False):
                mode = str(tp_band_cfg.get("mode", "shadow")).lower()
                timeframes = tp_band_cfg.get("timeframes", ["1m", "5m", "30m"])
                if isinstance(timeframes, str):
                    timeframes = [x.strip() for x in timeframes.split(",") if x.strip()]
                elif isinstance(timeframes, (list, tuple)):
                    timeframes = [str(x).strip() for x in timeframes if str(x).strip()]
                else:
                    timeframes = ["1m", "5m", "30m"]

                select_policy = str(tp_band_cfg.get("select_policy", "closest_level")).lower()
                method = str(tp_band_cfg.get("method", "kmeans")).lower()

                pivot_left = int(tp_band_cfg.get("pivot_left", 3) or 3)
                pivot_right = int(tp_band_cfg.get("pivot_right", 3) or 3)
                lookback_bars = int(tp_band_cfg.get("lookback_bars", 300) or 300)
                band_pct = float(tp_band_cfg.get("band_pct", 0.003) or 0.003)
                smc_cluster_pct = float(tp_band_cfg.get("smc_cluster_pct", 0.0015) or 0.0015)
                min_cluster_n = int(tp_band_cfg.get("min_cluster_n", 2) or 2)
                kmin = int(tp_band_cfg.get("kmin", 3) or 3)
                kmax = int(tp_band_cfg.get("kmax", 8) or 8)
                random_state = int(tp_band_cfg.get("random_state", 42) or 42)

                require_band_low_above_entry = bool(tp_band_cfg.get("require_band_low_above_entry", True))
                min_distance_atr_mult = float(tp_band_cfg.get("min_distance_atr_mult", 0.0) or 0.0)
                min_width_atr_mult = float(tp_band_cfg.get("min_width_atr_mult", 0.0) or 0.0)

                def _clamp(v: float, lo: float, hi: float) -> float:
                    return max(lo, min(hi, float(v)))

                # Spread snapshot (WS ticker) without async.
                spread_pct = None
                bid = ask = mid = None
                ticker_age_ms = None
                try:
                    pipeline = getattr(self, "market_data_pipeline", None)
                    ws_manager = getattr(pipeline, "websocket_manager", None) if pipeline else None
                    collector = getattr(ws_manager, "collector", None) if ws_manager else None
                    exchange = str(tp_band_cfg.get("exchange", getattr(pipeline, "DEFAULT_EXCHANGE", "bingx")) or "bingx").lower()
                    ticker_sample = collector.get_latest_ticker_sample(exchange, symbol_display) if collector else None
                    ticker = ticker_sample.get("data") if isinstance(ticker_sample, dict) else None
                    sample_ts = ticker_sample.get("timestamp") if isinstance(ticker_sample, dict) else None
                    if isinstance(sample_ts, datetime):
                        try:
                            ticker_age_ms = max(0, int((datetime.now(timezone.utc) - sample_ts).total_seconds() * 1000))
                        except Exception:
                            ticker_age_ms = None
                    if isinstance(ticker, dict):
                        for k in ("bestBid", "bid", "bidPrice", "best_bid", "B"):
                            if k in ticker and bid is None:
                                try:
                                    bid = float(ticker[k])
                                except Exception:
                                    bid = None
                        for k in ("bestAsk", "ask", "askPrice", "best_ask", "A"):
                            if k in ticker and ask is None:
                                try:
                                    ask = float(ticker[k])
                                except Exception:
                                    ask = None
                    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
                        mid = (bid + ask) / 2.0
                        if mid > 0:
                            spread_pct = (ask - bid) / mid
                except Exception:
                    spread_pct = None

                # Momentum/volume strength (fallback approximation) for quantile auto-selection.
                momentum_strength = None
                volume_strength = None
                strength_tf = None
                try:
                    df_fast = None
                    if isinstance(market_data, dict):
                        for tf in ("5m", "1m", "30m"):
                            candidate = market_data.get(tf)
                            if isinstance(candidate, pd.DataFrame) and not candidate.empty and "close" in candidate.columns:
                                df_fast = candidate
                                strength_tf = tf
                                break
                    if isinstance(df_fast, pd.DataFrame) and not df_fast.empty:
                        try:
                            includes_forming_fast = bool(getattr(df_fast, "attrs", {}).get("includes_forming", False))
                        except Exception:
                            includes_forming_fast = False
                        df_fast_closed = df_fast.iloc[:-1] if includes_forming_fast and len(df_fast) >= 2 else df_fast
                        closes = df_fast_closed["close"].astype(float)
                        if len(closes) >= 11:
                            price_change_pct = float(closes.pct_change(10).iloc[-1])
                            # Mirror StrategyCoordinator defaults
                            momentum_strength = _clamp((price_change_pct + 0.1) / 0.2, 0.0, 1.0)
                        if "volume" in df_fast_closed.columns and len(df_fast_closed) >= 20:
                            recent_vol = float(df_fast_closed["volume"].astype(float).tail(5).mean())
                            avg_vol = float(df_fast_closed["volume"].astype(float).tail(20).mean())
                            if avg_vol > 0:
                                volume_strength = _clamp(min(recent_vol / avg_vol, 2.0) / 2.0, 0.0, 1.0)
                except Exception:
                    momentum_strength = None
                    volume_strength = None

                volatility_bps = None
                try:
                    if entry_price > 0 and atr_value > 0:
                        volatility_bps = float((atr_value / entry_price) * 10_000.0)
                except Exception:
                    volatility_bps = None

                quantile_raw = tp_band_cfg.get("quantile", 0.5)
                quantile_mode = "fixed"
                if isinstance(quantile_raw, str) and quantile_raw.strip().lower() == "auto":
                    quantile_mode = "auto"
                    q_low = 0.25
                    q_mid = 0.50
                    q_high = 0.75
                    low_thr = float(tp_band_cfg.get("low_strength_threshold", 0.33) or 0.33)
                    high_thr = float(tp_band_cfg.get("high_strength_threshold", 0.66) or 0.66)
                    spread_low_thr = tp_band_cfg.get("spread_low_threshold_pct", None)
                    try:
                        spread_low_thr = float(spread_low_thr) if spread_low_thr is not None else None
                    except Exception:
                        spread_low_thr = None

                    mom = float(momentum_strength) if momentum_strength is not None else 0.5
                    vol_s = float(volume_strength) if volume_strength is not None else 0.5
                    q = q_mid
                    if (mom < low_thr) and (vol_s < low_thr):
                        q = q_low
                    elif (mom > high_thr) and (vol_s > high_thr) and (
                        spread_low_thr is None or (spread_pct is not None and float(spread_pct) < spread_low_thr)
                    ):
                        q = q_high

                    vol_bps_thr = tp_band_cfg.get("volatility_bps_high_threshold", None)
                    try:
                        vol_bps_thr = float(vol_bps_thr) if vol_bps_thr is not None else None
                    except Exception:
                        vol_bps_thr = None
                    if vol_bps_thr is not None and volatility_bps is not None and float(volatility_bps) >= float(vol_bps_thr):
                        if q >= q_high:
                            q = q_mid
                        elif q >= q_mid:
                            q = q_low
                    quantile = float(q)
                else:
                    try:
                        quantile = float(quantile_raw)
                    except Exception:
                        quantile = 0.5
                    quantile = _clamp(quantile, 0.0, 1.0)

                band_candidates: List[Tuple[str, Any]] = []
                for tf in timeframes:
                    df_tf = None
                    if isinstance(market_data, dict):
                        df_tf = market_data.get(tf)
                        if df_tf is None:
                            df_tf = market_data.get(f"df_{tf}")
                    if df_tf is None and tf == "30m":
                        df_tf = df_30m

                    if df_tf is None:
                        logger.warning(f"[Adaptive_OB] Missing dataframe for tf={tf}. Skipping.")
                        continue

                    if hasattr(df_tf, "empty") and df_tf.empty:
                        logger.warning(f"[Adaptive_OB] Empty dataframe for tf={tf}. Skipping.")
                        continue
                    if not isinstance(df_tf, pd.DataFrame) or df_tf.empty:
                        continue

                    band = compute_band(
                        df=df_tf,
                        timeframe=tf,
                        price=entry_price,
                        side="buy",
                        method=method,
                        pivot_left=pivot_left,
                        pivot_right=pivot_right,
                        lookback_bars=lookback_bars,
                        band_pct=band_pct,
                        smc_cluster_pct=smc_cluster_pct,
                        min_cluster_n=min_cluster_n,
                        kmin=kmin,
                        kmax=kmax,
                        random_state=random_state,
                    )
                    if band is None:
                        continue

                    if require_band_low_above_entry and float(band.band_low) <= float(entry_price):
                        continue
                    if min_width_atr_mult > 0 and atr_value > 0 and (float(band.band_high) - float(band.band_low)) < (min_width_atr_mult * float(atr_value)):
                        continue
                    if min_distance_atr_mult > 0 and atr_value > 0 and (float(band.band_low) - float(entry_price)) < (min_distance_atr_mult * float(atr_value)):
                        continue

                    band_candidates.append((tf, band))

                selected_tf = None
                selected_band = None
                if band_candidates:
                    if select_policy == "prefer_tf_order":
                        order = {tf: i for i, tf in enumerate(timeframes)}
                        selected_tf, selected_band = min(band_candidates, key=lambda x: order.get(x[0], 10_000))
                    else:
                        selected_tf, selected_band = min(
                            band_candidates,
                            key=lambda x: float(x[1].level) - float(entry_price),
                        )

                tp_candidate = None
                if selected_band is not None:
                    tp_candidate = float(selected_band.band_low) + float(quantile) * (
                        float(selected_band.band_high) - float(selected_band.band_low)
                    )
                    if tp_candidate <= entry_price:
                        tp_candidate = None

                tp_band_meta = {
                    "enabled": True,
                    "mode": mode,
                    "selected_tf": selected_tf,
                    "select_policy": select_policy,
                    "method": method,
                    "quantile_mode": quantile_mode,
                    "quantile": float(quantile),
                    "band": (
                        {
                            "level": float(selected_band.level),
                            "band_low": float(selected_band.band_low),
                            "band_high": float(selected_band.band_high),
                            "meta": selected_band.meta,
                        }
                        if selected_band is not None
                        else None
                    ),
                    "tp_candidate_raw": tp_candidate,
                    "tp_candidate_applied": None,
                    "target_applied": None,
                    "inputs": {
                        "momentum_strength": momentum_strength,
                        "volume_strength": volume_strength,
                        "strength_tf": strength_tf,
                        "spread_pct": spread_pct,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "ticker_age_ms": ticker_age_ms,
                        "volatility_bps": volatility_bps,
                    },
                    "filters": {
                        "require_band_low_above_entry": require_band_low_above_entry,
                        "min_distance_atr_mult": min_distance_atr_mult,
                        "min_width_atr_mult": min_width_atr_mult,
                        "pivot_left": pivot_left,
                        "pivot_right": pivot_right,
                        "lookback_bars": lookback_bars,
                        "band_pct": band_pct,
                        "smc_cluster_pct": smc_cluster_pct,
                        "min_cluster_n": min_cluster_n,
                        "kmin": kmin,
                        "kmax": kmax,
                        "random_state": random_state,
                    },
                }

                tp_candidate_applied = None
                if tp_candidate is not None and selected_band is not None:
                    tp_candidate_applied = float(tp_candidate)
                    # Do NOT push TP beyond the selected band's high just to satisfy min_tp_pct.
                    # If min_tp_pct is higher than band_high, keep TP inside the band and let RR veto it.
                    min_tp_target = float(entry_price * (1 + min_tp_pct))
                    if tp_candidate_applied < min_tp_target and min_tp_target <= float(selected_band.band_high):
                        tp_candidate_applied = float(min_tp_target)
                    tp_band_meta["tp_candidate_applied"] = tp_candidate_applied
                    if mode == "apply":
                        target_price = float(tp_candidate_applied)
                        tp_mode = "band"
                        tp_band_meta["target_applied"] = float(target_price)

                if selected_band is not None:
                    logger.info(
                        f"?? {log_prefix} [OB TP-BAND] mode={mode} tf={selected_tf} method={method} "
                        f"band_low={float(selected_band.band_low):.2f} band_high={float(selected_band.band_high):.2f} "
                        f"q={float(quantile):.2f} tp_raw={(tp_candidate if tp_candidate is not None else 'N/A')} "
                        f"tp_apply={(tp_candidate_applied if tp_candidate_applied is not None else 'N/A')} "
                        f"spread_pct={(spread_pct if spread_pct is not None else 'N/A')} "
                        f"mom={(momentum_strength if momentum_strength is not None else 'N/A')} "
                        f"vol={(volume_strength if volume_strength is not None else 'N/A')} "
                        f"vol_bps={(volatility_bps if volatility_bps is not None else 'N/A')}"
                    )
                else:
                    logger.info(f"?? {log_prefix} [OB TP-BAND] mode={mode} no_band_found (tfs={timeframes})")

            # Smart Recovery TP (structural targets) - optional, feature-flagged
            smart_recovery_meta: Optional[Dict[str, Any]] = None
            smart_cfg_raw = self.strategy_config.get("smart_recovery") or {}
            smart_cfg = self._normalize_smart_recovery_config(smart_cfg_raw)
            if isinstance(smart_cfg, dict) and bool(smart_cfg.get("enabled", False)):
                trigger_cfg = smart_cfg.get("trigger") or {}
                if not isinstance(trigger_cfg, dict):
                    trigger_cfg = {}

                # Crash leg detection (stable 5m pivots; avoid 1m noise)
                crash_cfg = smart_cfg.get("crash_leg") or {}
                if not isinstance(crash_cfg, dict):
                    crash_cfg = {}
                crash_tf = str(crash_cfg.get("timeframe", "5m") or "5m").strip().lower()
                crash_df = None
                if isinstance(market_data, dict):
                    crash_df = market_data.get(crash_tf) or market_data.get(f"df_{crash_tf}")
                if crash_df is None and crash_tf in ("30m", "df_30m"):
                    crash_df = df_30m
                crash_leg = None
                if isinstance(crash_df, pd.DataFrame) and not crash_df.empty:
                    crash_leg = self._detect_crash_leg(
                        df=crash_df,
                        timeframe=crash_tf,
                        lookback_bars=int(crash_cfg.get("lookback_bars", 240) or 240),
                        pivot_left=int(crash_cfg.get("pivot_left", 3) or 3),
                        pivot_right=int(crash_cfg.get("pivot_right", 3) or 3),
                        min_drop_pct=0.0,  # evaluate drop threshold in trigger logic
                    )

                # Trigger A: Shock state / drop magnitude
                shock_cfg = trigger_cfg.get("shock") or {}
                if not isinstance(shock_cfg, dict):
                    shock_cfg = {}
                shock_enabled = bool(shock_cfg.get("enabled", True))
                min_drop_pct = float(shock_cfg.get("min_drop_pct", 0.05) or 0.05)
                min_shock_score = float(shock_cfg.get("min_shock_score", 0.60) or 0.60)
                require_armed = bool(shock_cfg.get("require_dyn_gate_armed", False))

                shock_state = self.get_dyn_gate_state(symbol_display)
                shock_score = self._dyn_last_shock_score_by_symbol.get(symbol_display)
                crash_drop_pct = crash_leg.get("drop_pct") if isinstance(crash_leg, dict) else None

                shock_active_reasons: List[str] = []
                shock_active = False
                try:
                    if shock_score is not None and float(shock_score) >= float(min_shock_score):
                        shock_active = True
                        shock_active_reasons.append("shock_score")
                except Exception:
                    pass
                try:
                    if crash_drop_pct is not None and float(crash_drop_pct) >= float(min_drop_pct):
                        shock_active = True
                        shock_active_reasons.append("crash_leg_drop_pct")
                except Exception:
                    pass
                if require_armed and str(shock_state).upper() != "ARMED":
                    shock_active = False
                    shock_active_reasons = ["require_dyn_gate_armed"]

                # Trigger B: Band compression (band_width / ATR < threshold)
                comp_cfg = trigger_cfg.get("compression") or {}
                if not isinstance(comp_cfg, dict):
                    comp_cfg = {}
                comp_enabled = bool(comp_cfg.get("enabled", True))
                width_atr_max = float(comp_cfg.get("band_width_atr_ratio_max", 1.20) or 1.20)
                band_width = None
                width_atr = None
                try:
                    if tp_band_meta and isinstance(tp_band_meta.get("band"), dict):
                        band = tp_band_meta["band"]
                        lo = float(band.get("band_low"))
                        hi = float(band.get("band_high"))
                        band_width = float(hi - lo)
                        if atr_value and atr_value > 0:
                            width_atr = float(band_width / float(atr_value))
                except Exception:
                    band_width = None
                    width_atr = None

                comp_active = bool(width_atr is not None and float(width_atr) <= float(width_atr_max))

                smart_active = (shock_enabled and shock_active) or (comp_enabled and comp_active)
                triggers = {
                    "active": bool(smart_active),
                    "reasons": (shock_active_reasons if shock_active else []) + (["band_compression"] if comp_active else []),
                    "shock": {
                        "enabled": bool(shock_enabled),
                        "active": bool(shock_active),
                        "dyn_state": str(shock_state),
                        "shock_score": float(shock_score) if shock_score is not None else None,
                        "min_shock_score": float(min_shock_score),
                        "crash_drop_pct": float(crash_drop_pct) if crash_drop_pct is not None else None,
                        "min_drop_pct": float(min_drop_pct),
                        "require_dyn_gate_armed": bool(require_armed),
                    },
                    "compression": {
                        "enabled": bool(comp_enabled),
                        "active": bool(comp_active),
                        "band_width": float(band_width) if band_width is not None else None,
                        "atr": float(atr_value) if atr_value is not None else None,
                        "band_width_atr_ratio": float(width_atr) if width_atr is not None else None,
                        "band_width_atr_ratio_max": float(width_atr_max),
                    },
                }

                if smart_active:
                    selected_tp, smart_recovery_meta = self._calculate_smart_recovery_tp(
                        symbol=symbol_display,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        atr_value=atr_value,
                        min_tp_pct=min_tp_pct,
                        baseline_target_price=baseline_target_price,
                        current_target_price=target_price,
                        tp_band_meta=tp_band_meta,
                        market_data=market_data,
                        cfg=smart_cfg,
                        crash_leg=crash_leg,
                        triggers=triggers,
                    )

                    log_cfg = smart_cfg.get("logging") or {}
                    if not isinstance(log_cfg, dict):
                        log_cfg = {}
                    log_mode = str(log_cfg.get("mode", "decisions") or "decisions").strip().lower()
                    if log_mode in ("decisions", "all"):
                        try:
                            logger.info(
                                json.dumps(
                                    {
                                        "event": "ob_smart_recovery_tp",
                                        "symbol": symbol_display,
                                        "tp_mode": "Smart_Recovery",
                                        "entry": float(entry_price),
                                        "stop": float(stop_price),
                                        "min_rr_required": float(self.min_rr_ratio),
                                        "atr": float(atr_value) if atr_value is not None else None,
                                        "triggers": triggers,
                                        "selected_tp": smart_recovery_meta.get("selected_tp") if isinstance(smart_recovery_meta, dict) else None,
                                        "candidates": smart_recovery_meta.get("candidates") if isinstance(smart_recovery_meta, dict) else None,
                                        "rejection_reasons": smart_recovery_meta.get("rejection_reasons") if isinstance(smart_recovery_meta, dict) else None,
                                        "risk_data": smart_recovery_meta.get("risk_data") if isinstance(smart_recovery_meta, dict) else None,
                                        "leg_quality": smart_recovery_meta.get("leg_quality") if isinstance(smart_recovery_meta, dict) else None,
                                        "crash_leg_levels": smart_recovery_meta.get("crash_leg_levels") if isinstance(smart_recovery_meta, dict) else None,
                                    },
                                    separators=(",", ":"),
                                )
                            )
                        except Exception:
                            pass

                    if selected_tp is None:
                        on_fail = str(smart_cfg.get("on_no_valid_candidates", "skip_trade") or "skip_trade").strip().lower()
                        if on_fail in ("skip", "skip_trade", "reject"):
                            logger.info(f"🚫 {log_prefix} No Signal: Smart Recovery - no valid reachable TP candidates.")
                            _shadow_ob(
                                "no_signal_tp",
                                "smart_recovery_no_candidate",
                                {
                                    "tp_mode": "smart_recovery",
                                    "entry": entry_price,
                                    "stop": stop_price,
                                    "baseline_target": baseline_target_price,
                                    "target_pre_smart": target_price,
                                },
                            )
                            return None
                        # fallback_to_standard: keep pre-smart target_price
                    else:
                        target_price = float(selected_tp)
                        tp_mode = "smart_recovery"

            # Final TP pct for logging/RR calculation (may be overridden by TP-band apply mode)
            actual_tp_pct = (target_price / entry_price) - 1.0 if entry_price > 0 else 0.0

            # Calculate final R/R
            rr_numerator = target_price - entry_price
            rr_denominator = entry_price - stop_price
            rr_ratio = (rr_numerator / rr_denominator) if rr_denominator > 0 else 0

            # Enhanced logging with R/R details
            logger.info(
                f"🔎 {log_prefix} [OB R/R] Entry=${entry_price:.2f}, "
                f"Stop=${stop_price:.2f} (-{actual_sl_pct:.1%}), "
                f"Target=${target_price:.2f} (+{actual_tp_pct:.1%}), "
                f"R/R={rr_ratio:.2f} tp_mode={tp_mode}"
            )

            # 4. R/R Ratio Check
            # 🔥 GÜNCELLEME: R/R kontrolü artık __init__ içinde ayarlanan `self.min_rr_ratio` özelliğini kullanıyor.
            if rr_ratio < self.min_rr_ratio:
                logger.info(f"🚫 {log_prefix} No Signal: Calculated R/R Ratio ({rr_ratio:.2f}) is below the minimum required ({self.min_rr_ratio}).")
                _shadow_ob(
                    "no_signal_rr",
                    "rr_below_min",
                    {
                        "rr_ratio": rr_ratio,
                        "min_rr_required": self.min_rr_ratio,
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                    },
                )
                return None

            # --- Signal Generation ---
            logger.info(f"✅ {log_prefix} All checks passed. Generating BUY signal.")
            
            signal = {
                "strategy_name": self.strategy_name, "side": "buy", "symbol": symbol,
                "entry": entry_price, "stop": stop_price, "target": target_price,
                "reason": f"Adaptive RSI {rsi_val:.1f} <= {adaptive_rsi_threshold:.1f}",
                "rr_ratio": rr_ratio, "is_adaptive": True, "position_multiplier": position_mult,
                "ml_enhanced": ml_enhanced, "strategy_type": 'adaptive',
                "strategy_min_rr": self.min_rr_ratio,  # NEW: Strategy's own minimum R/R
            }

            if extreme_bypass_active:
                signal["extreme_bypass"] = True
                signal["extreme_bypass_meta"] = extreme_bypass_meta
                signal.setdefault("features", {})["extreme_bypass"] = extreme_bypass_meta

            # Expose RSI telemetry so downstream duplicate logic can react dynamically
            signal["rsi"] = float(rsi_val)
            signal.setdefault("features", {})["rsi"] = float(rsi_val)

            hook_delta = None
            try:
                hook_delta = float(self.strategy_config.get("hook_delta", 1.0))
            except Exception:
                hook_delta = 1.0
            rsi_hook = bool(
                rsi_prev is not None
                and hook_delta is not None
                and (rsi_val - rsi_prev) >= float(hook_delta)
            )
            bull_candle = bool(
                (reversal_meta.get("bullish_candle") is True)
                and (reversal_meta.get("close_above_prev_close") is True)
            )
            reclaim = bool(reversal_meta.get("close_above_ema_fast") is True)

            # Prefer fast-TF (1m/5m) reversal telemetry when available to avoid 30m lag.
            fast_df = None
            fast_tf = None
            try:
                if isinstance(market_data, dict):
                    for key in ("1m", "df_1m", "5m", "df_5m"):
                        cand = market_data.get(key)
                        if isinstance(cand, pd.DataFrame) and not cand.empty:
                            fast_df = cand
                            fast_tf = "1m" if "1m" in key else "5m"
                            break
            except Exception:
                fast_df = None
                fast_tf = None

            if isinstance(fast_df, pd.DataFrame) and not fast_df.empty:
                try:
                    last_fast = fast_df.iloc[-1]
                    prev_fast = fast_df.iloc[-2] if len(fast_df) >= 2 else None

                    last_open = float(last_fast.get("open")) if "open" in fast_df.columns else None
                    last_close = float(last_fast.get("close")) if "close" in fast_df.columns else None
                    prev_close = (
                        float(prev_fast.get("close"))
                        if prev_fast is not None and "close" in fast_df.columns
                        else None
                    )

                    if last_open is not None and last_close is not None and prev_close is not None:
                        bull_candle = bool(last_close > last_open and last_close > prev_close)

                    if "close" in fast_df.columns and hook_delta is not None:
                        if "rsi" in fast_df.columns:
                            fast_rsi = fast_df["rsi"].astype(float)
                        else:
                            fast_rsi = rsi(fast_df["close"].astype(float))
                        if len(fast_rsi) >= 2:
                            fr_prev = float(fast_rsi.iloc[-2])
                            fr_now = float(fast_rsi.iloc[-1])
                            if not pd.isna(fr_prev) and not pd.isna(fr_now):
                                rsi_hook = bool((fr_now - fr_prev) >= float(hook_delta))

                    ema_fast_val = None
                    if "ema_fast" in fast_df.columns:
                        ema_fast_val = float(last_fast.get("ema_fast"))
                    elif "ema21" in fast_df.columns:
                        ema_fast_val = float(last_fast.get("ema21"))
                    elif "close" in fast_df.columns and len(fast_df) >= 21:
                        ema_fast_val = float(
                            fast_df["close"]
                            .astype(float)
                            .ewm(span=21, adjust=False, min_periods=21)
                            .mean()
                            .iloc[-1]
                        )
                    if ema_fast_val is not None and last_close is not None and not pd.isna(ema_fast_val):
                        reclaim = bool(last_close > float(ema_fast_val))
                except Exception:
                    pass

            meta = signal.setdefault('meta', {})
            meta.update(
                {
                    'includes_forming': includes_forming,
                    'forming_open_time': forming_open_ms,
                    'rsi_source': rsi_source,
                    'trigger_price_source': trigger_price_source,
                    'fallback_reason': fallback_reason,
                    # Volume gating can optionally require this when allow_low_volume overrides min_bucket.
                    'low_volume_reversal_confirmed': bool(reversal_confirmed),
                    'reversal_confirmation': reversal_meta,
                    'rsi_hook': rsi_hook,
                    'bull_candle': bull_candle,
                    'reclaim': reclaim,
                    'reversal_tf': fast_tf,
                    'downtrend_context': {
                        'active': downtrend_context,
                        'adx': adx_val,
                        'ema_mid': ema_long,
                        'price_ref': price_ref,
                    },
                }
            )
            if tp_band_meta is not None:
                meta["tp_band"] = tp_band_meta
            if smart_recovery_meta is not None:
                meta["smart_recovery"] = smart_recovery_meta
            if trend_bias_active:
                meta['trend_confirmation'] = {
                    'ema_fast': ema_fast,
                    'ema_mid': ema_mid,
                    'rsi_threshold': adaptive_rsi_threshold,
                }
            
            if ml_enhanced:
                signal['ml_consensus'] = ml_context.get('consensus_score')
                signal['ml_position_modifier'] = position_size_modifier

            _shadow_ob(
                "pass",
                "",
                {
                    "rr_ratio": rr_ratio,
                    "min_rr_required": self.min_rr_ratio,
                    "position_multiplier": position_mult,
                },
            )
            
            # Persistency must not stay "passed" after emitting a signal.
            self._reset_persistency(symbol_display)
            return signal
            
        except Exception as e:
            logger.error(f"💥 {log_prefix} Critical error during signal generation: {e}", exc_info=True)
            # Fallback logic remains unchanged
            try:
                if hasattr(super(), 'signal'):
                    base_signal = super().signal(df_30m)
                    if base_signal:
                        base_signal.update({'strategy_type': 'base_fallback', 'fallback_reason': str(e), 'symbol': symbol})
                        logger.warning(f"⚠️ {log_prefix} Fallback to base strategy successful.")
                        return base_signal
            except Exception as fallback_error:
                logger.error(f"💥 {log_prefix} Fallback to base strategy also failed: {fallback_error}")
                
        return None

    def _persistency_key(self, symbol: str) -> str:
        return f"30m::{symbol}"

    def _reset_persistency(self, symbol: str) -> None:
        try:
            self._persistency_state.pop(self._persistency_key(symbol), None)
        except Exception:
            return

    def _log_persistency_skipped(
        self,
        symbol: str,
        log_prefix: str,
        reason: str,
        includes_forming: bool,
        forming_open_ms: Optional[int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        now_ts = time.time()
        throttle_key = f"{symbol}:{reason}"
        last_ts = self._persistency_skip_log_ts.get(throttle_key, 0.0)
        if (now_ts - last_ts) < 60.0:
            return
        self._persistency_skip_log_ts[throttle_key] = now_ts

        cfg = getattr(self, "_persistency_cfg", {}) or {}
        mode = str(cfg.get("mode", "time"))
        seconds = float(cfg.get("seconds", 5.0))
        min_samples = int(cfg.get("min_samples", 2))
        wick_k = float(cfg.get("wick_closeness_k", 0.25))

        state = self._persistency_state.get(self._persistency_key(symbol)) or {}
        first_ts = state.get("first_true_ts")
        samples = state.get("samples")
        bucket = state.get("bucket")

        state_bits = []
        if first_ts is not None:
            state_bits.append(f"state_first_ts={first_ts:.0f}")
        if samples is not None:
            state_bits.append(f"state_samples={samples}")
        if bucket is not None:
            state_bits.append(f"state_bucket={bucket}")
        state_suffix = " " + " ".join(state_bits) if state_bits else ""

        extra_bits = ""
        if extra:
            extra_bits = " " + " ".join(f"{k}={v}" for k, v in extra.items())

        logger.debug(
            f"{log_prefix} PersistencySkipped | reason={reason} mode={mode} seconds={seconds:.2f} "
            f"min_samples={min_samples} wick_k={wick_k:.2f} includes_forming={includes_forming} "
            f"bucket={forming_open_ms}{state_suffix}{extra_bits}"
        )

    def _apply_persistency_guard(
        self,
        symbol: str,
        includes_forming: bool,
        forming_open_ms: Optional[int],
        rsi_val: float,
        adaptive_rsi_threshold: float,
        trigger_price: float,
        entry_threshold: float,
        forming_close: Optional[float],
        forming_low: Optional[float],
        forming_high: Optional[float],
    ) -> tuple[bool, Dict[str, Any]]:
        cfg = getattr(self, "_persistency_cfg", {"mode": "time", "seconds": 5.0, "min_samples": 1})
        mode = str(cfg.get("mode", "time")).lower()
        seconds = max(float(cfg.get("seconds", 5.0)), 0.0)
        min_samples = max(int(cfg.get("min_samples", 1)), 1)
        wick_k = max(min(float(cfg.get("wick_closeness_k", 0.25)), 1.0), 0.0)
        eps = 1e-9

        meta = {
            "mode": mode,
            "includes_forming": includes_forming,
            "forming_open_time": forming_open_ms,
            "condition_true": False,
            "elapsed_s": 0.0,
            "samples": 0,
            "threshold_seconds": seconds,
            "threshold_samples": min_samples,
            "trigger_price": trigger_price,
            "entry_threshold": entry_threshold,
            "forming_close": forming_close,
            "forming_low": forming_low,
            "forming_high": forming_high,
            "wick_closeness": None,
            "wick_closeness_k": wick_k,
            "wick_closeness_passed": None,
            "wick_closeness_skipped": False,
            "wick_closeness_reason": None,
            "bucket_changed": False,
        }

        if mode == "off":
            self._reset_persistency(symbol)
            meta["condition_true"] = True
            meta["elapsed_s"] = 0.0
            meta["samples"] = 0
            return True, meta

        if not includes_forming:
            self._reset_persistency(symbol)
            meta["condition_true"] = True
            return True, meta

        key = self._persistency_key(symbol)

        had_state = key in self._persistency_state
        state = self._persistency_state.get(
            key, {"first_true_ts": None, "samples": 0, "bucket": forming_open_ms}
        )

        prev_bucket = state.get("bucket")
        bucket_changed = had_state and (prev_bucket != forming_open_ms)

        if bucket_changed:
            state = {"first_true_ts": None, "samples": 0, "bucket": forming_open_ms}

        condition_true = False
        if mode == "bar_low_and_close":
            if includes_forming and forming_low is not None and forming_close is not None and forming_close > 0:
                # Canonical wick filter
                dip_occurred = forming_low <= entry_threshold
                no_snap_back = trigger_price <= entry_threshold
                if forming_high is None or pd.isna(forming_high) or pd.isna(forming_low):
                    meta["wick_closeness_skipped"] = True
                    meta["wick_closeness_reason"] = "missing_high_or_low"
                    closeness = None
                    closeness_pass = True  # fall back to dip + no_snap_back only
                    meta["wick_closeness_passed"] = closeness_pass
                meta = {
                    "mode": mode,
                    "includes_forming": includes_forming,
                    "forming_open_time": forming_open_ms,
                    "condition_true": False,
                    "elapsed_s": 0.0,
                    "samples": 0,
                    "threshold_seconds": seconds,
                    "threshold_samples": min_samples,
                    "trigger_price": trigger_price,
                    "entry_threshold": entry_threshold,
                    "forming_close": forming_close,
                    "forming_low": forming_low,
                    "forming_high": forming_high,
                    "wick_closeness": None,
                    "wick_closeness_k": wick_k,
                    "wick_closeness_passed": None,
                    "wick_closeness_skipped": False,
                    "wick_closeness_reason": None,
                    "bucket_changed": False,
                    "prev_bucket": prev_bucket,
                }

                meta["prev_bucket"] = prev_bucket
                meta["bucket_changed"] = bucket_changed
            self._reset_persistency(symbol)
            meta["bucket_changed"] = bucket_changed
            return False, meta

        now_ts = time.time()
        if state.get("first_true_ts") is None:
            state["first_true_ts"] = now_ts
            state["samples"] = 1
        else:
            state["samples"] = int(state.get("samples", 0)) + 1
        elapsed = max(0.0, now_ts - float(state.get("first_true_ts", now_ts)))
        passed = (elapsed >= seconds) and (state["samples"] >= min_samples)

        state["bucket"] = forming_open_ms
        self._persistency_state[key] = state

        meta.update(
            {
                "condition_true": True,
                "elapsed_s": elapsed,
                "samples": state["samples"],
                "bucket_changed": bucket_changed,
            }
        )
        return passed, meta
    
    def get_strategy_state(self) -> Dict:
        """
        Get current strategy state and parameters.
        
        Returns:
            Dictionary with current adaptive parameters
        """
        return {
            'strategy': 'adaptive_oversold_bounce',
            'base_config': self.base_cfg,
            'has_regime_analyzer': self.regime_analyzer is not None
        }
