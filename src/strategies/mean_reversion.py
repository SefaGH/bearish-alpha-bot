import json
import asyncio
import math
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import pandas as pd

from .base_strategy import BaseStrategy
from .mr_controller import DynamicMRController, MRControllerDecision
from core.indicators import rsi as calc_rsi
from core.logger import get_current_run_id
from core.rsi_zone_router import is_strategy_allowed, snapshot_log_context as rsi_snapshot_log_context

logger = logging.getLogger(__name__)


class TradeAggregatorLite:
    """
    Incremental 15s trade-to-bar aggregator (pure Python, O(1) per trade).

    - Dynamic state is limited to `current_bar` and `last_closed_ts`.
    - On bucket rollover, emits the closed bar(s) to the MR controller.
    - Gap-filling is safety-capped to avoid CPU spikes after long gaps.
    """

    BUCKET_MS = 15_000

    def __init__(
        self,
        *,
        symbol: str,
        controller: DynamicMRController,
        gap_fill_cap: int = 50,
    ) -> None:
        self.symbol = str(symbol)
        self._controller = controller
        try:
            cap = int(gap_fill_cap)
        except Exception:
            cap = 50
        # Safety cap: never generate more than 50 synthetic bars per rollover.
        self._gap_fill_cap = max(0, min(cap, 50))

        self.current_bar: Optional[Dict[str, float]] = None
        self.last_closed_ts: Optional[int] = None

    def get_current_volume(self) -> Optional[float]:
        bar = self.current_bar
        if not isinstance(bar, dict):
            return None
        try:
            return float(bar.get("volume", 0.0))
        except Exception:
            return None

    def process_trade(self, *, ts_ms: int, price: float, qty: float) -> None:
        try:
            ts_ms_int = int(ts_ms)
        except Exception:
            return
        if ts_ms_int <= 0:
            return

        try:
            price_f = float(price)
            qty_f = float(qty)
        except Exception:
            return
        if not math.isfinite(price_f) or price_f <= 0:
            return
        if not math.isfinite(qty_f) or qty_f < 0:
            return

        bucket_start = ts_ms_int - (ts_ms_int % self.BUCKET_MS)
        bar = self.current_bar

        # First trade initializes the current bar.
        if not isinstance(bar, dict):
            self.current_bar = {
                "start_ts": float(bucket_start),
                "open": price_f,
                "high": price_f,
                "low": price_f,
                "close": price_f,
                "volume": qty_f,
            }
            return

        try:
            cur_start = int(bar.get("start_ts", 0.0))
        except Exception:
            cur_start = 0

        # Late/out-of-order trade: ignore (no retroactive fixes).
        if bucket_start < cur_start:
            return

        # Same bucket: incremental O/H/L/C/V update.
        if bucket_start == cur_start:
            try:
                bar["high"] = max(float(bar.get("high", price_f)), price_f)
                bar["low"] = min(float(bar.get("low", price_f)), price_f)
                bar["close"] = price_f
                bar["volume"] = float(bar.get("volume", 0.0)) + qty_f
            except Exception:
                # If bar got corrupted, reset minimally to current trade.
                self.current_bar = {
                    "start_ts": float(bucket_start),
                    "open": price_f,
                    "high": price_f,
                    "low": price_f,
                    "close": price_f,
                    "volume": qty_f,
                }
            return

        # New bucket: close current bar and emit (plus safe-capped gap fill).
        try:
            prev_close = float(bar.get("close", price_f))
        except Exception:
            prev_close = price_f
        try:
            prev_vol = float(bar.get("volume", 0.0) or 0.0)
        except Exception:
            prev_vol = 0.0

        self._emit_closed_bar(start_ts=cur_start, close=prev_close, volume=prev_vol)

        missing_slots = (bucket_start - cur_start) // self.BUCKET_MS - 1
        if missing_slots > 0 and self._gap_fill_cap > 0 and math.isfinite(prev_close):
            fill_count = min(int(missing_slots), int(self._gap_fill_cap))
            for i in range(fill_count):
                fill_start = cur_start + ((i + 1) * self.BUCKET_MS)
                self._emit_closed_bar(start_ts=fill_start, close=prev_close, volume=0.0)

        # Start new current bar for this bucket.
        self.current_bar = {
            "start_ts": float(bucket_start),
            "open": price_f,
            "high": price_f,
            "low": price_f,
            "close": price_f,
            "volume": qty_f,
        }

    def _emit_closed_bar(self, *, start_ts: int, close: float, volume: float) -> None:
        self.last_closed_ts = int(start_ts)
        ingest = getattr(self._controller, "ingest_15s_bar", None)
        if not callable(ingest):
            return
        try:
            ingest(symbol=self.symbol, start_ts_ms=int(start_ts), close=float(close), volume=float(volume))
        except Exception:
            return


class VWAPMeanReversion(BaseStrategy):
    """
    VWAP-band mean reversion strategy (1m VWAP, 5m signal default).
    Uses a weak-trend filter via ADX to avoid fighting strong trends.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(strategy_name="mean_reversion", config=cfg)
        self.vwap_tf = cfg.get("timeframe", "1m")
        self.signal_tf = cfg.get("signal_timeframe", "5m")
        raw_price_source = cfg.get("price_source", "signal_close")
        try:
            raw_price_source = str(raw_price_source or "").strip().lower()
        except Exception:
            raw_price_source = "signal_close"
        if raw_price_source in ("close", "closed", "closed_close", "candle_close", "signal_close", ""):
            raw_price_source = "signal_close"
        allowed_price_sources = {"signal_close", "forming_close", "mid", "mark", "last"}
        if raw_price_source not in allowed_price_sources:
            logger.warning(
                "[MeanReversion] Unsupported price_source=%r; falling back to 'signal_close' (allowed=%s)",
                raw_price_source,
                sorted(allowed_price_sources),
            )
            raw_price_source = "signal_close"
        self.price_source = raw_price_source
        self.band_mult = float(cfg.get("band_multiplier", 2.0))
        self.stop_loss_std_delta = float(cfg.get("stop_loss_std_delta", 0.5))
        if not math.isfinite(self.stop_loss_std_delta) or self.stop_loss_std_delta < 0:
            self.stop_loss_std_delta = 0.5
        self.vwap_lookback = int(cfg.get("vwap_lookback", 1440))
        self.adx_threshold = float(cfg.get("adx_threshold", 30))
        try:
            self._high_adx_z_threshold = float(cfg.get("high_adx_z_threshold", 2.0) or 2.0)
        except Exception:
            self._high_adx_z_threshold = 2.0
        if not math.isfinite(self._high_adx_z_threshold) or self._high_adx_z_threshold <= 0:
            self._high_adx_z_threshold = 2.0
        # Safety: preserve intent that higher ADX requires stricter Z.
        self._high_adx_z_threshold = max(1.60, float(self._high_adx_z_threshold))
        self.min_rr_ratio = float(cfg.get("min_rr_ratio", 1.0))

        net_profit_cfg = cfg.get("net_profit_filter", {})
        if net_profit_cfg is not None and not isinstance(net_profit_cfg, dict):
            net_profit_cfg = {}
        net_profit_cfg = dict(net_profit_cfg) if isinstance(net_profit_cfg, dict) else {}
        self.net_profit_filter_enabled = bool(net_profit_cfg.get("enabled", False))
        try:
            self.net_profit_cost_bps = float(net_profit_cfg.get("cost_bps_assumed", cfg.get("cost_bps_assumed", 6.0)))
        except Exception:
            self.net_profit_cost_bps = 6.0
        if not math.isfinite(self.net_profit_cost_bps) or self.net_profit_cost_bps < 0:
            self.net_profit_cost_bps = 6.0
        try:
            self.min_net_reward_bps = float(net_profit_cfg.get("min_net_reward_bps", 0.0))
        except Exception:
            self.min_net_reward_bps = 0.0
        if not math.isfinite(self.min_net_reward_bps):
            self.min_net_reward_bps = 0.0
        self.soft_deferral_threshold = float(cfg.get("soft_deferral_threshold", 0.005))
        if not math.isfinite(self.soft_deferral_threshold) or self.soft_deferral_threshold < 0:
            self.soft_deferral_threshold = 0.005
        try:
            self.fast_watch_eps_bps = int(cfg.get("fast_watch_eps_bps", 10) or 10)
        except Exception:
            self.fast_watch_eps_bps = 10
        self.fast_watch_eps_bps = max(0, self.fast_watch_eps_bps)

        mr_fast_watch_cfg = cfg.get("fast_watch", {})
        if mr_fast_watch_cfg is not None and not isinstance(mr_fast_watch_cfg, dict):
            mr_fast_watch_cfg = {}
        mr_fast_watch_cfg = dict(mr_fast_watch_cfg) if isinstance(mr_fast_watch_cfg, dict) else {}
        promote_override_cfg = mr_fast_watch_cfg.get("promote_override", {})
        if promote_override_cfg is not None and not isinstance(promote_override_cfg, dict):
            promote_override_cfg = {}
        promote_override_cfg = dict(promote_override_cfg) if isinstance(promote_override_cfg, dict) else {}
        self._fast_watch_v2_cfg = mr_fast_watch_cfg
        v2_keys = {"near_bps", "touch_eps_bps", "touch_price_source", "recheck_freshness_ms", "allow_touch_entry"}
        self._fast_watch_v2_enabled = any(k in mr_fast_watch_cfg for k in v2_keys)
        self._promote_override_enabled = bool(promote_override_cfg.get("enabled", True))
        try:
            promote_mode = str(promote_override_cfg.get("mode", "observe") or "observe").strip().lower()
        except Exception:
            promote_mode = "observe"
        if promote_mode not in {"observe", "enforce", "off", "disabled"}:
            promote_mode = "observe"
        self._promote_override_mode = promote_mode

        raw_canary_symbols = promote_override_cfg.get("canary_symbols", [])
        canary_symbols: set[str] = set()
        if isinstance(raw_canary_symbols, str):
            tokens = [p.strip() for p in raw_canary_symbols.split(",")]
        elif isinstance(raw_canary_symbols, (list, tuple, set)):
            tokens = list(raw_canary_symbols)
        else:
            tokens = []
        for token in tokens:
            try:
                key = str(token).strip().upper()
            except Exception:
                key = ""
            if key:
                canary_symbols.add(key)
        self._promote_override_canary_symbols = canary_symbols
        try:
            self._promote_override_min_z_score = float(promote_override_cfg.get("min_z_score", 2.0) or 2.0)
        except Exception:
            self._promote_override_min_z_score = 2.0
        if not math.isfinite(self._promote_override_min_z_score) or self._promote_override_min_z_score <= 0:
            self._promote_override_min_z_score = 2.0
        try:
            self._promote_override_max_dist_bps = float(promote_override_cfg.get("max_dist_bps", 2.0) or 2.0)
        except Exception:
            self._promote_override_max_dist_bps = 2.0
        if not math.isfinite(self._promote_override_max_dist_bps) or self._promote_override_max_dist_bps < 0:
            self._promote_override_max_dist_bps = 2.0
        try:
            self._promote_override_max_adx = float(promote_override_cfg.get("max_adx", 20.0) or 20.0)
        except Exception:
            self._promote_override_max_adx = 20.0
        if not math.isfinite(self._promote_override_max_adx) or self._promote_override_max_adx <= 0:
            self._promote_override_max_adx = 20.0
        raw_min_volume = promote_override_cfg.get("min_volume_strength")
        self._promote_override_min_volume_strength: Optional[float] = None
        if raw_min_volume is not None:
            try:
                parsed_min_volume = float(raw_min_volume)
                if math.isfinite(parsed_min_volume):
                    self._promote_override_min_volume_strength = max(0.0, parsed_min_volume)
            except Exception:
                self._promote_override_min_volume_strength = None
        self._promote_override_respect_trend_veto = bool(promote_override_cfg.get("respect_trend_veto", False))
        blocked_states_raw = promote_override_cfg.get("blocked_shock_states", ["ARMED", "TRIGGERED"])
        blocked_states: set[str] = set()
        if isinstance(blocked_states_raw, str):
            blocked_states = {s.strip().upper() for s in blocked_states_raw.split(",") if s and str(s).strip()}
        elif isinstance(blocked_states_raw, (list, tuple, set)):
            for state in blocked_states_raw:
                try:
                    state_norm = str(state).strip().upper()
                except Exception:
                    state_norm = ""
                if state_norm:
                    blocked_states.add(state_norm)
        if not blocked_states:
            blocked_states = {"ARMED", "TRIGGERED"}
        self._promote_override_blocked_shock_states = blocked_states

        self._fast_watch_near_bps_default: Optional[float] = None
        self._fast_watch_touch_eps_bps_default: Optional[float] = None
        self._fast_watch_recheck_freshness_ms_default: Optional[int] = None
        self._fast_watch_touch_price_source_default: Optional[str] = None
        self._fast_watch_allow_touch_entry_default: Optional[bool] = None

        if self._fast_watch_v2_enabled:
            try:
                near_bps = mr_fast_watch_cfg.get("near_bps")
                self._fast_watch_near_bps_default = float(near_bps) if near_bps is not None else float(self.fast_watch_eps_bps)
            except Exception:
                self._fast_watch_near_bps_default = float(self.fast_watch_eps_bps)

            try:
                touch_eps_bps = mr_fast_watch_cfg.get("touch_eps_bps")
                self._fast_watch_touch_eps_bps_default = float(touch_eps_bps) if touch_eps_bps is not None else 2.0
            except Exception:
                self._fast_watch_touch_eps_bps_default = 2.0
            if self._fast_watch_touch_eps_bps_default is not None:
                self._fast_watch_touch_eps_bps_default = max(0.0, float(self._fast_watch_touch_eps_bps_default))

            try:
                fresh_ms = mr_fast_watch_cfg.get("recheck_freshness_ms")
                self._fast_watch_recheck_freshness_ms_default = int(fresh_ms) if fresh_ms is not None else 1000
            except Exception:
                self._fast_watch_recheck_freshness_ms_default = 1000
            if self._fast_watch_recheck_freshness_ms_default is not None:
                self._fast_watch_recheck_freshness_ms_default = max(0, int(self._fast_watch_recheck_freshness_ms_default))

            try:
                raw_src = mr_fast_watch_cfg.get("touch_price_source", "bidask")
                raw_src = str(raw_src or "").strip().lower()
            except Exception:
                raw_src = "bidask"
            if not raw_src:
                raw_src = "bidask"
            self._fast_watch_touch_price_source_default = raw_src

            try:
                self._fast_watch_allow_touch_entry_default = bool(mr_fast_watch_cfg.get("allow_touch_entry", True))
            except Exception:
                self._fast_watch_allow_touch_entry_default = True
        try:
            self.fast_watch_interval_ms = int(cfg.get("fast_watch_interval_ms", 3000) or 3000)
        except Exception:
            self.fast_watch_interval_ms = 3000
        self.fast_watch_interval_ms = max(250, self.fast_watch_interval_ms)
        try:
            self.fast_watch_max_checks = int(cfg.get("fast_watch_max_checks", 9) or 9)
        except Exception:
            self.fast_watch_max_checks = 9
        self.fast_watch_max_checks = max(1, self.fast_watch_max_checks)
        try:
            self.fast_watch_ttl_ms = int(cfg.get("fast_watch_ttl_ms", 30000) or 30000)
        except Exception:
            self.fast_watch_ttl_ms = 30000
        self.fast_watch_ttl_ms = max(1000, self.fast_watch_ttl_ms)
        # Increase baseline so VWAP window (1440) + signal buffer can coexist
        self.min_rows = int(cfg.get("min_rows", 2100))
        self.min_signal_rows = int(cfg.get("min_signal_rows", 50))

        controller_cfg = cfg.get("dynamic_controller", {})
        if controller_cfg is not None and not isinstance(controller_cfg, dict):
            logger.warning(
                "[MeanReversion] dynamic_controller config must be a dict; disabling controller."
            )
            controller_cfg = {}
        controller_cfg = dict(controller_cfg) if isinstance(controller_cfg, dict) else {}
        controller_cfg.setdefault("adx_freeze_threshold", self.adx_threshold)

        adaptive_cfg = cfg.get("adaptive_settings", {})
        if adaptive_cfg is not None and not isinstance(adaptive_cfg, dict):
            logger.warning("[MeanReversion] adaptive_settings config must be a dict; disabling adaptive settings.")
            adaptive_cfg = {}
        adaptive_cfg = dict(adaptive_cfg) if isinstance(adaptive_cfg, dict) else {}
        self._adaptive_lite_enabled = bool(adaptive_cfg.get("enabled", False))
        try:
            gap_cap = int(adaptive_cfg.get("gap_fill_cap", 50) or 50)
        except Exception:
            gap_cap = 50
        # Safety cap: at most 50 synthetic bars per rollover.
        self._trade_gap_fill_cap = max(0, min(gap_cap, 50))
        self._trade_aggregators: Dict[str, TradeAggregatorLite] = {}
        self._trade_callback_registered = False
        self._trade_subscribed_symbols: set[str] = set()
        # Pass adaptive settings into controller config (controller ignores when disabled).
        controller_cfg.setdefault("adaptive_settings", adaptive_cfg)

        # Regime-adaptive ADX veto support (squeeze exception + slope check).
        # - Uses controller vol_state when available.
        # - Uses dynamic_controller.adx_freeze_threshold as the relaxed ceiling in squeeze only.
        # - Falls back to static adx_threshold when inputs are missing/invalid.
        try:
            self._adx_slope_lookback = int(cfg.get("adx_slope_lookback", 5) or 5)
        except Exception:
            self._adx_slope_lookback = 5
        self._adx_slope_lookback = max(1, int(self._adx_slope_lookback))
        try:
            self._adx_slope_eps = float(cfg.get("adx_slope_eps", 0.0) or 0.0)
        except Exception:
            self._adx_slope_eps = 0.0
        if not math.isfinite(self._adx_slope_eps):
            self._adx_slope_eps = 0.0

        self._adx_squeeze_threshold = None
        try:
            raw_squeeze_th = controller_cfg.get("adx_freeze_threshold")
            if raw_squeeze_th is not None:
                squeeze_th = float(raw_squeeze_th)
                if math.isfinite(squeeze_th) and squeeze_th > 0:
                    self._adx_squeeze_threshold = float(squeeze_th)
        except Exception:
            self._adx_squeeze_threshold = None
        self._mr_controller = DynamicMRController(
            controller_cfg,
            static_band_multiplier=self.band_mult,
            static_lookback=self.vwap_lookback,
        )
        self._pipeline_cfg_warned = False
        self._controller_fallback_warned = False
        self._last_soft_deferral_anchor_by_key: Dict[str, int] = {}
        guard_cfg = cfg.get("rsi_rebound_guard", {})
        if guard_cfg is not None and not isinstance(guard_cfg, dict):
            logger.warning("[MeanReversion] rsi_rebound_guard config must be a dict; disabling guard.")
            guard_cfg = {}
        guard_cfg = dict(guard_cfg) if isinstance(guard_cfg, dict) else {}

        def _guard_float(value: Any, default: float) -> float:
            try:
                value = float(value)
            except Exception:
                return float(default)
            if not math.isfinite(value):
                return float(default)
            return float(value)

        def _guard_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        activation_rsi = _guard_float(guard_cfg.get("activation_rsi", 25.0), 25.0)
        rebound_rsi = _guard_float(guard_cfg.get("rebound_rsi", 27.0), 27.0)
        if activation_rsi < 0 or activation_rsi > 100:
            activation_rsi = 25.0
        if rebound_rsi < 0 or rebound_rsi > 100:
            rebound_rsi = 27.0

        self._rsi_guard_cfg = {
            "enabled": bool(guard_cfg.get("enabled", True)),
            "tf": str(guard_cfg.get("tf", "1m") or "1m"),
            "use_closed_only": bool(guard_cfg.get("use_closed_only", True)),
            "activation_rsi": activation_rsi,
            "activation_z_score": _guard_float(guard_cfg.get("activation_z_score", 2.2), 2.2),
            "rebound_rsi": rebound_rsi,
            "max_wait_s": max(0, _guard_int(guard_cfg.get("max_wait_s", 120), 120)),
        }
        self._rsi_guard_state_by_symbol: Dict[str, str] = {}
        self._rsi_guard_armed_ts_ms_by_symbol: Dict[str, int] = {}

        reentry_cfg = cfg.get("reentry_guard", {})
        if reentry_cfg is not None and not isinstance(reentry_cfg, dict):
            logger.warning("[MeanReversion] reentry_guard config must be a dict; disabling guard.")
            reentry_cfg = {}
        reentry_cfg = dict(reentry_cfg) if isinstance(reentry_cfg, dict) else {}
        self._reentry_guard_enabled = bool(reentry_cfg.get("enabled", True))
        self._reentry_guard_require_vwap_reclaim = bool(
            reentry_cfg.get("require_vwap_reclaim_after_stop", True)
        )
        short_side_cfg = reentry_cfg.get("short_side", {})
        if short_side_cfg is not None and not isinstance(short_side_cfg, dict):
            short_side_cfg = {}
        short_side_cfg = dict(short_side_cfg) if isinstance(short_side_cfg, dict) else {}
        self._reentry_guard_short_enabled = bool(short_side_cfg.get("enabled", False))
        self._reentry_guard_short_clear_on_band_breach = bool(
            short_side_cfg.get("clear_on_band_breach", True)
        )
        try:
            self._reentry_guard_short_clear_on_z_threshold = float(
                short_side_cfg.get("clear_on_z_threshold", 0.0) or 0.0
            )
        except Exception:
            self._reentry_guard_short_clear_on_z_threshold = 0.0
        if (
            not math.isfinite(self._reentry_guard_short_clear_on_z_threshold)
            or self._reentry_guard_short_clear_on_z_threshold < 0
        ):
            self._reentry_guard_short_clear_on_z_threshold = 0.0
        self._reentry_guard_long_by_symbol: Dict[str, bool] = {}
        self._reentry_guard_short_by_symbol: Dict[str, bool] = {}

        # Phase-1 shadow classifier telemetry (no execution impact).
        vsa_cfg = cfg.get("vsa_shadow", {})
        if vsa_cfg is not None and not isinstance(vsa_cfg, dict):
            logger.warning("[MeanReversion] vsa_shadow config must be a dict; using defaults.")
            vsa_cfg = {}
        vsa_cfg = dict(vsa_cfg) if isinstance(vsa_cfg, dict) else {}
        self._vsa_shadow_enabled = bool(vsa_cfg.get("enabled", True))
        try:
            self._vsa_shadow_rejection_window = int(vsa_cfg.get("rejection_window", 12) or 12)
        except Exception:
            self._vsa_shadow_rejection_window = 12
        self._vsa_shadow_rejection_window = max(3, int(self._vsa_shadow_rejection_window))
        try:
            self._vsa_shadow_z_entry = float(vsa_cfg.get("z_entry", 1.6) or 1.6)
        except Exception:
            self._vsa_shadow_z_entry = 1.6
        try:
            self._vsa_shadow_z_cap = float(vsa_cfg.get("z_cap", 3.0) or 3.0)
        except Exception:
            self._vsa_shadow_z_cap = 3.0
        if not math.isfinite(self._vsa_shadow_z_entry):
            self._vsa_shadow_z_entry = 1.6
        if not math.isfinite(self._vsa_shadow_z_cap):
            self._vsa_shadow_z_cap = 3.0
        if self._vsa_shadow_z_cap <= self._vsa_shadow_z_entry:
            self._vsa_shadow_z_cap = float(self._vsa_shadow_z_entry + 1.4)
        try:
            self._vsa_shadow_rr_span = float(vsa_cfg.get("rr_span", 0.5) or 0.5)
        except Exception:
            self._vsa_shadow_rr_span = 0.5
        if not math.isfinite(self._vsa_shadow_rr_span) or self._vsa_shadow_rr_span <= 0:
            self._vsa_shadow_rr_span = 0.5
        self._vsa_rejection_pass_hist_by_symbol: Dict[str, list[int]] = {}

        if self.vwap_lookback > 1000:
            logger.warning(
                f"[MeanReversion] High Lookback detected (L={self.vwap_lookback}). "
                "Note that L is in BARS (in VWAP timeframe), not minutes."
            )

        # Interface guard
        if not hasattr(self, "signal"):
            self.signal = self._default_signal_wrapper  # type: ignore
        assert callable(getattr(self, "signal", None)), "MeanReversion: signal method not callable"
        print("MeanReversion: signal method bound successfully")

    def arm_reentry_guard(self, symbol: str, side: str = "long") -> None:
        if not self._reentry_guard_enabled:
            return
        try:
            sym = str(symbol or "").strip()
        except Exception:
            return
        if not sym:
            return
        try:
            side_norm = str(side or "long").strip().lower()
        except Exception:
            side_norm = "long"
        if side_norm in {"short", "sell"}:
            if self._reentry_guard_short_enabled:
                self._reentry_guard_short_by_symbol[sym] = True
            return
        self._reentry_guard_long_by_symbol[sym] = True

    async def _ensure_trade_wiring(self, symbol: str) -> None:
        """Ensure BingX @trade subscription + callback registration (best effort, non-fatal)."""
        if not getattr(self, "_adaptive_lite_enabled", False):
            return

        pipeline = getattr(self, "market_data_pipeline", None)
        ws_manager = getattr(pipeline, "websocket_manager", None) if pipeline is not None else None
        if ws_manager is None:
            return

        ws = None
        try:
            clients = getattr(ws_manager, "clients", {}) or {}
            client = clients.get("bingx")
            ws = getattr(client, "bingx_ws", None) if client is not None else None
        except Exception:
            ws = None

        if ws is None:
            return

        # Register callback once.
        if not getattr(self, "_trade_callback_registered", False):
            try:
                if hasattr(ws, "on_trade"):
                    ws.on_trade(self.on_trade)
                    self._trade_callback_registered = True
            except Exception:
                self._trade_callback_registered = False

        # Subscribe once per symbol.
        try:
            sym = str(symbol or "").strip()
        except Exception:
            return
        if not sym:
            return
        if sym in self._trade_subscribed_symbols:
            return

        subscribe = getattr(ws, "subscribe_trade", None) or getattr(ws, "subscribe_trades", None)
        if not callable(subscribe):
            return

        try:
            if asyncio.iscoroutinefunction(subscribe):
                await subscribe(sym)
            else:
                subscribe(sym)
            self._trade_subscribed_symbols.add(sym)
        except Exception:
            return

    def on_trade(self, symbol: str, trade: Dict[str, Any]) -> None:
        """Trade callback invoked by BingXWebSocket; feeds TradeAggregatorLite (pure python)."""
        if not getattr(self, "_adaptive_lite_enabled", False):
            return
        if not symbol or not isinstance(trade, dict):
            return

        symbol_key = str(symbol)
        if ":" not in symbol_key and symbol_key.endswith("/USDT"):
            symbol_key = f"{symbol_key}:USDT"

        ts_ms_int = 0
        try:
            ts_ms = trade.get("timestamp")
            if ts_ms is None:
                ts_ms = trade.get("ts")
            if ts_ms is None:
                ts_ms = trade.get("T")
            ts_ms_int = int(ts_ms) if ts_ms is not None else 0
        except Exception:
            ts_ms_int = 0

        price_f = float("nan")
        try:
            price = trade.get("price")
            if price is None:
                price = trade.get("p")
            price_f = float(price) if price is not None else float("nan")
        except Exception:
            price_f = float("nan")

        qty_f = float("nan")
        try:
            qty = trade.get("quantity")
            if qty is None:
                qty = trade.get("qty")
            if qty is None:
                qty = trade.get("q")
            qty_f = float(qty) if qty is not None else float("nan")
        except Exception:
            qty_f = float("nan")

        if ts_ms_int <= 0 or not math.isfinite(price_f) or not math.isfinite(qty_f):
            return

        agg = self._trade_aggregators.get(symbol_key)
        if agg is None:
            agg = TradeAggregatorLite(
                symbol=symbol_key,
                controller=self._mr_controller,
                gap_fill_cap=int(getattr(self, "_trade_gap_fill_cap", 50) or 50),
            )
            self._trade_aggregators[symbol_key] = agg

        agg.process_trade(ts_ms=ts_ms_int, price=price_f, qty=qty_f)

    @staticmethod
    def _parse_timeframe_ms(timeframe: str) -> int:
        raw = str(timeframe or "").strip().lower()
        if not raw:
            return 300_000
        try:
            num = ""
            unit = ""
            for ch in raw:
                if ch.isdigit():
                    num += ch
                else:
                    unit += ch
            value = int(num) if num else 1
        except Exception:
            return 300_000

        unit = unit.strip() or "m"
        if unit in ("m", "min", "mins", "minute", "minutes"):
            return value * 60_000
        if unit in ("h", "hr", "hrs", "hour", "hours"):
            return value * 3_600_000
        if unit in ("d", "day", "days"):
            return value * 86_400_000
        return 300_000

    @staticmethod
    def _subset_dropna(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
        cols = [c for c in subset if c in df.columns]
        if cols:
            return df.dropna(subset=cols)
        return df.dropna()

    def _clean_vwap_df(self, df_vwap: pd.DataFrame) -> pd.DataFrame:
        # Only require columns used for VWAP-band context.
        subset = ["close", "volume", "vwap", "vwap_upper", "vwap_lower"]
        return self._subset_dropna(df_vwap, subset)

    def _clean_sig_df(self, df_sig: pd.DataFrame) -> pd.DataFrame:
        # Only require columns used for signal evaluation (avoid collateral NaNs from unused indicators).
        subset = ["close", "adx"]
        return self._subset_dropna(df_sig, subset)

    @staticmethod
    def _normalize_ema_stack(ema_stack: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not isinstance(ema_stack, dict):
            return None
        out: Dict[str, float] = {}
        for key in ("ema21", "ema50", "ema200"):
            try:
                val = float(ema_stack.get(key))
            except Exception:
                return None
            if not math.isfinite(val):
                return None
            out[key] = float(val)
        return out

    @classmethod
    def _extract_ema_stack_from_df(cls, df: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None

        normalized_cols: Dict[str, str] = {}
        for col in df.columns:
            try:
                col_norm = str(col).strip().lower().replace("_", "")
            except Exception:
                continue
            normalized_cols[col_norm] = str(col)

        key_map = {
            "ema21": normalized_cols.get("ema21"),
            "ema50": normalized_cols.get("ema50"),
            "ema200": normalized_cols.get("ema200"),
        }
        if not all(key_map.values()):
            return None

        out: Dict[str, float] = {}
        for ema_key, col_name in key_map.items():
            try:
                series = pd.to_numeric(df[col_name], errors="coerce").dropna()
            except Exception:
                return None
            if series.empty:
                return None
            try:
                val = float(series.iloc[-1])
            except Exception:
                return None
            if not math.isfinite(val):
                return None
            out[ema_key] = float(val)
        return out

    def _get_ema_stack(
        self,
        *,
        symbol: Optional[str] = None,
        df_sig: Optional[pd.DataFrame] = None,
        market_data: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        # 1) Explicit caller-provided stack wins.
        explicit = None
        if isinstance(kwargs, dict):
            explicit = kwargs.get("ema_stack")
        normalized_explicit = self._normalize_ema_stack(explicit if isinstance(explicit, dict) else None)
        if normalized_explicit is not None:
            return normalized_explicit

        # 2) Current signal dataframe.
        cand = self._extract_ema_stack_from_df(df_sig)
        if cand is not None:
            return cand

        # 3) Additional dataframes in market_data (best-effort, no extra fetch).
        candidates: list[pd.DataFrame] = []
        if isinstance(market_data, dict):
            for key in ("df_sig", "5m", "15m", "30m", "1h", "df_30m", "df_1h"):
                maybe_df = market_data.get(key)
                if isinstance(maybe_df, pd.DataFrame):
                    candidates.append(maybe_df)

            if symbol is not None:
                symbol_bucket = market_data.get(symbol)
                if isinstance(symbol_bucket, dict):
                    for key in ("5m", "15m", "30m", "1h", str(self.signal_tf), str(self.vwap_tf)):
                        maybe_df = symbol_bucket.get(key)
                        if isinstance(maybe_df, pd.DataFrame):
                            candidates.append(maybe_df)
                try:
                    symbol_short = str(symbol).split(":", 1)[0]
                except Exception:
                    symbol_short = None
                if symbol_short and symbol_short != symbol:
                    symbol_bucket = market_data.get(symbol_short)
                    if isinstance(symbol_bucket, dict):
                        for key in ("5m", "15m", "30m", "1h", str(self.signal_tf), str(self.vwap_tf)):
                            maybe_df = symbol_bucket.get(key)
                            if isinstance(maybe_df, pd.DataFrame):
                                candidates.append(maybe_df)

        for candidate_df in candidates:
            cand = self._extract_ema_stack_from_df(candidate_df)
            if cand is not None:
                return cand
        return None

    def _is_trend_against_mr(
        self,
        direction: str,
        *,
        ema_stack: Optional[Dict[str, Any]] = None,
        regime_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        near = str(direction or "").strip().lower()
        if near not in {"lower", "upper"}:
            return False

        # Prefer explicit EMA stack if caller provides it.
        if isinstance(ema_stack, dict):
            try:
                ema21 = float(ema_stack.get("ema21"))
                ema50 = float(ema_stack.get("ema50"))
                ema200 = float(ema_stack.get("ema200"))
            except Exception:
                ema21 = ema50 = ema200 = float("nan")
            if math.isfinite(ema21) and math.isfinite(ema50) and math.isfinite(ema200):
                bullish_stack = ema21 > ema50 > ema200
                bearish_stack = ema21 < ema50 < ema200
                if near == "upper" and bullish_stack:
                    return True
                if near == "lower" and bearish_stack:
                    return True

        # Fallback: use already-computed regime label.
        trend_label = None
        if isinstance(regime_data, dict):
            try:
                trend_label = str(regime_data.get("trend") or "").strip().lower()
            except Exception:
                trend_label = None
        if near == "upper" and trend_label == "bullish":
            return True
        if near == "lower" and trend_label == "bearish":
            return True
        return False

    @staticmethod
    def _coerce_finite_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except Exception:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    @staticmethod
    def _normalize_volume_bucket(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            label = str(value).strip().upper()
        except Exception:
            return None
        if not label:
            return None
        return label

    @staticmethod
    def _clip01(value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
        except Exception:
            val = float(default)
        if not math.isfinite(val):
            val = float(default)
        return max(0.0, min(1.0, float(val)))

    @staticmethod
    def _safe_sigmoid(value: float) -> float:
        try:
            if value >= 0:
                z = math.exp(-float(value))
                return 1.0 / (1.0 + z)
            z = math.exp(float(value))
            return z / (1.0 + z)
        except Exception:
            return 0.5

    def _update_vsa_rejection_history(
        self,
        *,
        symbol: Any,
        rejection_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        key = ""
        try:
            key = str(symbol or "").strip()
        except Exception:
            key = ""
        if not key:
            return {
                "rej_pass": None,
                "pass_rate": 0.0,
                "has_two_consecutive_passes": False,
                "window": int(self._vsa_shadow_rejection_window),
                "history_size": 0,
            }

        rej_pass: Optional[bool] = None
        upper_wick_component = 0.0
        if isinstance(rejection_meta, dict):
            try:
                has_red = bool(rejection_meta.get("has_red"))
                close_back_inside = bool(rejection_meta.get("close_back_inside_band"))
                upper_wick_ratio = self._coerce_finite_float(rejection_meta.get("upper_wick_ratio"))
                wick_thr = self._coerce_finite_float(rejection_meta.get("threshold_wick_ratio"))
                if wick_thr is None or not math.isfinite(float(wick_thr)) or wick_thr <= 0:
                    wick_thr = 0.8
                if upper_wick_ratio is None or not math.isfinite(float(upper_wick_ratio)):
                    upper_wick_ratio = 0.0
                rej_pass = bool(has_red and (close_back_inside or float(upper_wick_ratio) >= float(wick_thr)))
                upper_wick_component = self._clip01(float(upper_wick_ratio) / 0.8, default=0.0)
            except Exception:
                rej_pass = None
                upper_wick_component = 0.0

        hist = self._vsa_rejection_pass_hist_by_symbol.setdefault(key, [])
        if rej_pass is not None:
            hist.append(1 if rej_pass else 0)
            max_len = max(3, int(self._vsa_shadow_rejection_window))
            if len(hist) > max_len:
                del hist[:-max_len]

        pass_rate = (float(sum(hist)) / float(len(hist))) if hist else 0.0
        has_two = bool(len(hist) >= 2 and hist[-1] == 1 and hist[-2] == 1)
        return {
            "rej_pass": rej_pass,
            "pass_rate": self._clip01(pass_rate, default=0.0),
            "has_two_consecutive_passes": has_two,
            "window": int(self._vsa_shadow_rejection_window),
            "history_size": int(len(hist)),
            "upper_wick_component": self._clip01(upper_wick_component, default=0.0),
        }

    def _compute_vsa_shadow_meta(
        self,
        *,
        symbol: Any,
        side: str,
        clean_vwap: Optional[pd.DataFrame],
        regime_data: Optional[Dict[str, Any]],
        adx_val: Optional[float],
        atr_val: Optional[float],
        z_val: Optional[float],
        volume_analysis: Optional[Dict[str, Any]],
        reward_bps: Optional[float],
        risk_bps: Optional[float],
        rejection_shadow: Optional[Dict[str, Any]],
        timeframe_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "_vsa_shadow_enabled", True)):
            return None

        if clean_vwap is None or not isinstance(clean_vwap, pd.DataFrame) or clean_vwap.empty:
            return {
                "enabled": True,
                "status": "insufficient_data",
                "reason": "missing_clean_vwap",
            }

        side_norm = str(side or "").strip().lower()
        if side_norm in {"sell", "short"}:
            side_norm = "short"
        elif side_norm in {"buy", "long"}:
            side_norm = "long"
        else:
            side_norm = "unknown"

        # --- Impulse I ---
        last_close = prev_close = None
        high_last = low_last = None
        try:
            close_series = pd.to_numeric(clean_vwap["close"], errors="coerce").dropna()
            if len(close_series) >= 1:
                last_close = float(close_series.iloc[-1])
            if len(close_series) >= 2:
                prev_close = float(close_series.iloc[-2])
        except Exception:
            pass
        try:
            high_series = pd.to_numeric(clean_vwap["high"], errors="coerce").dropna()
            low_series = pd.to_numeric(clean_vwap["low"], errors="coerce").dropna()
            if len(high_series) >= 1 and len(low_series) >= 1:
                high_last = float(high_series.iloc[-1])
                low_last = float(low_series.iloc[-1])
        except Exception:
            pass

        r_abs = None
        if last_close is not None and prev_close is not None and prev_close > 0:
            try:
                r_abs = abs((float(last_close) - float(prev_close)) / float(prev_close))
            except Exception:
                r_abs = None

        tr_last = None
        if high_last is not None and low_last is not None:
            if prev_close is not None:
                try:
                    tr_last = max(
                        float(high_last) - float(low_last),
                        abs(float(high_last) - float(prev_close)),
                        abs(float(low_last) - float(prev_close)),
                    )
                except Exception:
                    tr_last = None
            else:
                try:
                    tr_last = max(0.0, float(high_last) - float(low_last))
                except Exception:
                    tr_last = None

        atr_ref = self._coerce_finite_float(atr_val)
        if atr_ref is None:
            try:
                if "atr" in clean_vwap.columns:
                    atr_series = pd.to_numeric(clean_vwap["atr"], errors="coerce").dropna()
                    if not atr_series.empty:
                        atr_ref = float(atr_series.iloc[-1])
            except Exception:
                atr_ref = None
        if atr_ref is not None and (not math.isfinite(float(atr_ref)) or float(atr_ref) <= 0):
            atr_ref = None

        vr_ratio = None
        if isinstance(volume_analysis, dict):
            try:
                local_ctx = volume_analysis.get("local")
                if isinstance(local_ctx, dict):
                    vr_ratio = self._coerce_finite_float(local_ctx.get("ratio_recent_to_baseline"))
            except Exception:
                vr_ratio = None
            if vr_ratio is None:
                try:
                    vol_strength = self._coerce_finite_float(volume_analysis.get("volume_strength"))
                    if vol_strength is not None and math.isfinite(float(vol_strength)):
                        vr_ratio = max(0.0, float(vol_strength) * 2.0)
                except Exception:
                    vr_ratio = None

        impulse_components: Dict[str, float] = {}
        if r_abs is not None:
            impulse_components["ret"] = self._clip01(float(r_abs) / 0.0025, default=0.0)
        if tr_last is not None and atr_ref is not None and atr_ref > 0:
            impulse_components["tr_atr"] = self._clip01(float(tr_last) / (2.5 * float(atr_ref)), default=0.0)
        if vr_ratio is not None:
            impulse_components["vr"] = self._clip01(float(vr_ratio) / 2.0, default=0.0)
        impulse_score = max(impulse_components.values()) if impulse_components else 0.0
        impulse_score = self._clip01(impulse_score, default=0.0)

        # --- Trend T (directional alignment) ---
        adx_coord = None
        if isinstance(regime_data, dict):
            adx_coord = self._coerce_finite_float(regime_data.get("trend_strength"))
            if adx_coord is None:
                adx_coord = self._coerce_finite_float(regime_data.get("adx"))
        if adx_coord is None:
            adx_coord = self._coerce_finite_float(adx_val)
        trend_core = 0.0
        if adx_coord is not None:
            trend_core = self._clip01((float(adx_coord) - 25.0) / 20.0, default=0.0)

        slope_raw = None
        slope_source = "none"
        try:
            if "vwap" in clean_vwap.columns and len(clean_vwap) >= 2:
                vwap_series = pd.to_numeric(clean_vwap["vwap"], errors="coerce").dropna()
                if len(vwap_series) >= 2:
                    slope_raw = float(vwap_series.iloc[-1]) - float(vwap_series.iloc[-2])
                    slope_source = "vwap"
        except Exception:
            slope_raw = None
            slope_source = "none"
        if slope_raw is None:
            try:
                if "ema50" in clean_vwap.columns and len(clean_vwap) >= 2:
                    ema_series = pd.to_numeric(clean_vwap["ema50"], errors="coerce").dropna()
                    if len(ema_series) >= 2:
                        slope_raw = float(ema_series.iloc[-1]) - float(ema_series.iloc[-2])
                        slope_source = "ema50"
            except Exception:
                slope_raw = None
                slope_source = "none"
        slope_align = 0.0
        if slope_raw is not None and math.isfinite(float(slope_raw)):
            if side_norm == "short":
                slope_align = 1.0 if float(slope_raw) > 0 else 0.0
            elif side_norm == "long":
                slope_align = 1.0 if float(slope_raw) < 0 else 0.0
        trend_score = self._clip01(float(trend_core) * float(slope_align), default=0.0)

        # --- Rejection R (with persistency penalty) ---
        pass_rate = 0.0
        has_two_consecutive = False
        upper_wick_component = 0.0
        if isinstance(rejection_shadow, dict):
            pass_rate = self._clip01(rejection_shadow.get("pass_rate"), default=0.0)
            has_two_consecutive = bool(rejection_shadow.get("has_two_consecutive_passes", False))
            upper_wick_component = self._clip01(rejection_shadow.get("upper_wick_component"), default=0.0)
        rejection_raw = self._clip01((0.7 * pass_rate) + (0.3 * upper_wick_component), default=0.0)
        rejection_score = rejection_raw if has_two_consecutive else self._clip01(0.6 * rejection_raw, default=0.0)

        # --- Acceptance A (time above/below VWAP for recent hold) ---
        tf_ms = self._parse_timeframe_ms(str(timeframe_hint or self.vwap_tf))
        tf_seconds = max(1.0, float(tf_ms) / 1000.0)
        hold_seconds = 0.0
        acceptance_base = 0.0
        bars_in_120s = max(1, int(math.ceil(120.0 / tf_seconds)))
        try:
            if {"close", "vwap"}.issubset(set(clean_vwap.columns)):
                sub = clean_vwap.tail(max(2, bars_in_120s))
                close_vals = pd.to_numeric(sub["close"], errors="coerce").tolist()
                vwap_vals = pd.to_numeric(sub["vwap"], errors="coerce").tolist()
                streak = 0
                for idx in range(len(close_vals) - 1, -1, -1):
                    c = close_vals[idx]
                    v = vwap_vals[idx]
                    if c is None or v is None:
                        break
                    try:
                        c = float(c)
                        v = float(v)
                    except Exception:
                        break
                    if not math.isfinite(c) or not math.isfinite(v):
                        break
                    if side_norm == "short":
                        cond = c > v
                    elif side_norm == "long":
                        cond = c < v
                    else:
                        cond = False
                    if not cond:
                        break
                    streak += 1
                hold_seconds = min(120.0, float(streak) * float(tf_seconds))
                acceptance_base = self._clip01(float(hold_seconds) / 120.0, default=0.0)
        except Exception:
            hold_seconds = 0.0
            acceptance_base = 0.0
        acceptance_score = self._clip01(float(acceptance_base) * (1.0 - float(rejection_score)), default=0.0)

        # --- z normalization ---
        abs_z = 0.0
        z_raw = self._coerce_finite_float(z_val)
        if z_raw is not None:
            abs_z = abs(float(z_raw))
        z_entry = float(self._vsa_shadow_z_entry)
        z_cap = float(self._vsa_shadow_z_cap)
        z_norm = self._clip01((float(abs_z) - z_entry) / max(1e-9, (z_cap - z_entry)), default=0.0)

        s_ba = (1.2 * impulse_score) + (1.0 * trend_score) + (0.8 * acceptance_score) - (1.0 * rejection_score)
        s_go = (1.0 * rejection_score) + (0.8 * z_norm) - (1.0 * impulse_score) - (0.8 * trend_score)
        s_fr = (1.0 * impulse_score) + (1.2 * rejection_score) - (1.0 * acceptance_score)

        # Softmax with numerical stabilization.
        max_s = max(s_ba, s_go, s_fr)
        try:
            exp_ba = math.exp(s_ba - max_s)
            exp_go = math.exp(s_go - max_s)
            exp_fr = math.exp(s_fr - max_s)
            denom = exp_ba + exp_go + exp_fr
            if denom <= 0 or not math.isfinite(denom):
                p_ba = p_go = p_fr = (1.0 / 3.0)
            else:
                p_ba = exp_ba / denom
                p_go = exp_go / denom
                p_fr = exp_fr / denom
        except Exception:
            p_ba = p_go = p_fr = (1.0 / 3.0)

        probs = {"BA": float(p_ba), "GO": float(p_go), "FR": float(p_fr)}
        selected_class = max(probs, key=probs.get)
        p_selected = float(probs[selected_class])

        # E = p_selected * Q * M (shadow-only approximation).
        quality_raw = 0.60  # neutral fallback before coordinator quality scoring.
        quality_source = "neutral_fallback"
        q_in = self._coerce_finite_float(quality_raw)
        q_comp = self._clip01((float(q_in) - 0.50) / 0.20, default=0.5) if q_in is not None else 0.5

        rr_ratio = None
        rb = self._coerce_finite_float(reward_bps)
        rk = self._coerce_finite_float(risk_bps)
        if rb is not None and rk is not None and float(rk) > 0:
            rr_ratio = float(rb) / float(rk)
        rr_min = self._coerce_finite_float(self.min_rr_ratio)
        if rr_min is None:
            rr_min = 1.0
        rr_span = float(self._vsa_shadow_rr_span)
        rr_deficit = 0.0
        if rr_ratio is not None:
            rr_deficit = max(0.0, float(rr_min) - float(rr_ratio))
        pen_rr = self._clip01(float(rr_deficit) / float(rr_span), default=0.0)
        m_rr = 1.0 - pen_rr
        m_fill = 1.0  # pre-fill signal stage
        m_comp = self._clip01(float(m_fill) * float(m_rr), default=1.0)

        edge_e = self._clip01(float(p_selected) * float(q_comp) * float(m_comp), default=0.0)
        risk_mult = self._clip01(self._safe_sigmoid(8.0 * (float(edge_e) - 0.55)), default=0.1)
        risk_mult = max(0.1, min(1.0, float(risk_mult)))

        return {
            "enabled": True,
            "status": "ok",
            "inputs": {
                "side": side_norm,
                "timeframe": str(timeframe_hint or self.signal_tf),
                "symbol": str(symbol) if symbol is not None else None,
                "rejection_window": int(self._vsa_shadow_rejection_window),
                "z_entry": float(z_entry),
                "z_cap": float(z_cap),
                "rr_span": float(rr_span),
            },
            "scores": {
                "I": float(impulse_score),
                "T": float(trend_score),
                "A": float(acceptance_score),
                "R": float(rejection_score),
                "z_norm": float(z_norm),
            },
            "probabilities": probs,
            "selected_class": str(selected_class),
            "edge": {
                "E": float(edge_e),
                "Q": float(q_comp),
                "M": float(m_comp),
                "M_fill": float(m_fill),
                "M_rr": float(m_rr),
                "RR": float(rr_ratio) if rr_ratio is not None else None,
                "RR_min": float(rr_min),
                "risk_mult_shadow": float(risk_mult),
                "quality_source": quality_source,
            },
            "diagnostics": {
                "impulse_components": impulse_components,
                "rejection_pass_rate": float(pass_rate),
                "rejection_persistency_ok": bool(has_two_consecutive),
                "hold_seconds_vwap": float(hold_seconds),
                "trend_adx_coord": float(adx_coord) if adx_coord is not None else None,
                "trend_core": float(trend_core),
                "trend_slope_source": slope_source,
                "trend_slope_raw": float(slope_raw) if slope_raw is not None else None,
                "upper_wick_component": float(upper_wick_component),
            },
        }

    @staticmethod
    def _symbol_rollout_keys(symbol: Any) -> set[str]:
        keys: set[str] = set()
        if symbol is None:
            return keys
        try:
            raw = str(symbol).strip().upper()
        except Exception:
            raw = ""
        if not raw:
            return keys
        keys.add(raw)
        if ":" in raw:
            left = raw.split(":", 1)[0].strip()
            if left:
                keys.add(left)
        return keys

    def _is_promote_override_enforced_for_symbol(self, symbol: Any) -> tuple[bool, str]:
        if not bool(getattr(self, "_promote_override_enabled", True)):
            return False, "disabled"
        mode = str(getattr(self, "_promote_override_mode", "observe") or "observe").strip().lower()
        if mode in {"off", "disabled"}:
            return False, "mode_disabled"
        if mode == "enforce":
            return True, "mode_enforce"
        canary_set = getattr(self, "_promote_override_canary_symbols", set()) or set()
        if "*" in canary_set:
            return True, "canary_all"
        symbol_keys = self._symbol_rollout_keys(symbol)
        if symbol_keys and any(k in canary_set for k in symbol_keys):
            return True, "canary_symbol"
        return False, "observe_only"

    def _estimate_local_volume_context(self, df_sig: Optional[pd.DataFrame]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "available": False,
            "volume_strength": None,
            "volume_bucket": None,
            "recent_volume": None,
            "baseline_volume": None,
            "ratio_recent_to_baseline": None,
            "reason": "unavailable",
        }
        if df_sig is None or not isinstance(df_sig, pd.DataFrame) or df_sig.empty:
            return out
        if "volume" not in df_sig.columns:
            out["reason"] = "missing_volume_column"
            return out

        try:
            vol = pd.to_numeric(df_sig["volume"], errors="coerce").dropna()
        except Exception:
            vol = pd.Series(dtype="float64")
        if len(vol) < 3:
            out["reason"] = "insufficient_rows"
            return out

        recent_window = min(5, max(1, len(vol) - 1))
        baseline_series = vol.iloc[:-recent_window] if recent_window < len(vol) else vol.iloc[:-1]
        if baseline_series.empty:
            out["reason"] = "insufficient_baseline"
            return out
        baseline_window = min(20, len(baseline_series))
        baseline_tail = baseline_series.tail(baseline_window)
        if baseline_tail.empty:
            out["reason"] = "insufficient_baseline"
            return out

        try:
            recent_avg = float(vol.tail(recent_window).mean())
            baseline_avg = float(baseline_tail.mean())
        except Exception:
            out["reason"] = "aggregation_failed"
            return out
        if not math.isfinite(recent_avg) or not math.isfinite(baseline_avg) or baseline_avg <= 0:
            out["reason"] = "non_finite_or_zero_baseline"
            return out

        ratio = recent_avg / baseline_avg
        strength = min(max(ratio, 0.0), 2.0) / 2.0
        if ratio < 0.80:
            bucket = "LOW"
        elif ratio < 1.20:
            bucket = "NORMAL"
        elif ratio < 1.80:
            bucket = "HIGH"
        else:
            bucket = "EXTREME"
        out.update(
            {
                "available": True,
                "volume_strength": float(strength),
                "volume_bucket": bucket,
                "recent_volume": float(recent_avg),
                "baseline_volume": float(baseline_avg),
                "ratio_recent_to_baseline": float(ratio),
                "reason": "ok",
            }
        )
        return out

    def _resolve_volume_analysis(
        self,
        *,
        kwargs: Dict[str, Any],
        check_detail: Optional[Dict[str, Any]],
        df_sig: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        upstream_strength = self._coerce_finite_float(kwargs.get("volume_strength"))
        upstream_bucket = self._normalize_volume_bucket(kwargs.get("volume_bucket"))
        upstream_source = kwargs.get("volume_source")
        upstream_analysis = kwargs.get("volume_analysis")
        if isinstance(upstream_analysis, dict):
            if upstream_strength is None:
                upstream_strength = self._coerce_finite_float(
                    upstream_analysis.get("volume_strength", upstream_analysis.get("strength"))
                )
            if upstream_bucket is None:
                upstream_bucket = self._normalize_volume_bucket(
                    upstream_analysis.get("volume_bucket", upstream_analysis.get("bucket"))
                )
            if upstream_source is None:
                upstream_source = upstream_analysis.get("source")

        detail_candidates = []
        if isinstance(check_detail, dict):
            detail_candidates.extend(
                [
                    check_detail,
                    check_detail.get("volume"),
                    check_detail.get("volume_analysis"),
                    check_detail.get("volume_context"),
                    check_detail.get("volume_detail"),
                ]
            )
        for cand in detail_candidates:
            if not isinstance(cand, dict):
                continue
            if upstream_strength is None:
                upstream_strength = self._coerce_finite_float(
                    cand.get("volume_strength", cand.get("strength"))
                )
            if upstream_bucket is None:
                upstream_bucket = self._normalize_volume_bucket(
                    cand.get("volume_bucket", cand.get("bucket"))
                )
            if upstream_source is None:
                upstream_source = cand.get("source")
            if upstream_strength is not None and upstream_bucket is not None and upstream_source is not None:
                break

        local_ctx = self._estimate_local_volume_context(df_sig)
        local_strength = self._coerce_finite_float(local_ctx.get("volume_strength"))
        local_bucket = self._normalize_volume_bucket(local_ctx.get("volume_bucket"))

        resolved_strength = upstream_strength if upstream_strength is not None else local_strength
        resolved_bucket = upstream_bucket if upstream_bucket is not None else local_bucket

        if upstream_strength is not None or upstream_bucket is not None:
            resolved_source = str(upstream_source or "upstream_recheck")
        elif local_strength is not None or local_bucket is not None:
            resolved_source = "local_df_sig_fallback"
        else:
            resolved_source = "unavailable"

        return {
            "source": resolved_source,
            "volume_strength": resolved_strength,
            "volume_bucket": resolved_bucket,
            "upstream": {
                "volume_strength": upstream_strength,
                "volume_bucket": upstream_bucket,
                "source": str(upstream_source) if upstream_source is not None else None,
            },
            "local": local_ctx,
        }

    def check_promotion_override(
        self,
        *,
        touch_confirmed: bool,
        near: str,
        dist_bps: Optional[float],
        z: Optional[float],
        adx: Optional[float],
        shock_state: Optional[str] = None,
        regime_data: Optional[Dict[str, Any]] = None,
        volume_strength: Optional[float] = None,
        ema_stack: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self._promote_override_enabled:
            return False
        if not touch_confirmed:
            return False
        near_norm = str(near or "").strip().lower()
        if near_norm not in {"lower", "upper"}:
            return False
        if dist_bps is None or not math.isfinite(float(dist_bps)):
            logger.info("[MeanReversion] PROMOTE reject: dist_bps missing/invalid")
            return False
        # FastWatch dist_to_band_bps can be signed; compare using absolute distance.
        if abs(float(dist_bps)) > float(self._promote_override_max_dist_bps):
            logger.info(
                "[MeanReversion] PROMOTE reject: abs(dist_bps)=%.2f > %.2f",
                abs(float(dist_bps)),
                float(self._promote_override_max_dist_bps),
            )
            return False
        if z is None or not math.isfinite(float(z)):
            logger.info("[MeanReversion] PROMOTE reject: z missing/invalid")
            return False
        if abs(float(z)) < float(self._promote_override_min_z_score):
            logger.info(
                "[MeanReversion] PROMOTE reject: z=%.2f < %.2f",
                float(z),
                float(self._promote_override_min_z_score),
            )
            return False
        if adx is None or not math.isfinite(float(adx)):
            logger.info("[MeanReversion] PROMOTE reject: adx missing/invalid")
            return False
        if float(adx) > float(self._promote_override_max_adx):
            logger.info(
                "[MeanReversion] PROMOTE reject: ADX=%.2f > %.2f",
                float(adx),
                float(self._promote_override_max_adx),
            )
            return False
        if self._promote_override_respect_trend_veto and self._is_trend_against_mr(
            near_norm,
            ema_stack=ema_stack,
            regime_data=regime_data,
        ):
            logger.info("[MeanReversion] PROMOTE reject: trend veto near=%s", near_norm)
            return False
        if (
            volume_strength is not None
            and self._promote_override_min_volume_strength is not None
            and math.isfinite(float(volume_strength))
            and float(volume_strength) < float(self._promote_override_min_volume_strength)
        ):
            logger.info(
                "[MeanReversion] PROMOTE reject: volume_strength=%.2f < %.2f",
                float(volume_strength),
                float(self._promote_override_min_volume_strength),
            )
            return False
        if shock_state is not None:
            try:
                shock_state_norm = str(shock_state).strip().upper()
            except Exception:
                shock_state_norm = ""
            if shock_state_norm and shock_state_norm in self._promote_override_blocked_shock_states:
                logger.info("[MeanReversion] PROMOTE reject: shock_state=%s", shock_state_norm)
                return False

        logger.info(
            "[MeanReversion] PROMOTE approved: near=%s z=%.2f adx=%.2f dist_bps=%.2f",
            near_norm,
            float(z),
            float(adx),
            float(dist_bps),
        )
        return True

    async def signal(self, symbol: str, market_data: Optional[Dict[str, Any]] = None, ml_context=None, **kwargs) -> Optional[Dict[str, Any]]:
        """Interface method expected by ProductionCoordinator."""
        rows_hint = 0
        if isinstance(market_data, dict):
            rows_hint = len(next(iter(market_data.values()), [])) if market_data else 0
        logger.info(f"[MeanReversion] Processing signal for {symbol}... Input Data Rows: {rows_hint}")
        result = await self.generate_signal(symbol=symbol, market_data=market_data, ml_context=ml_context, **kwargs)
        logger.info(f"[MeanReversion] Cycle complete for {symbol}. Action: {'HOLD' if result is None else 'SIGNAL'}")
        return result

    async def generate_signal(self, symbol: str, market_data: Optional[Dict[str, Any]] = None, ml_context=None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate a mean-reversion signal using VWAP bands and ADX trend filter.
        """
        if getattr(self, "_adaptive_lite_enabled", False):
            try:
                await self._ensure_trade_wiring(symbol)
            except Exception:
                pass

        parent_pending_id = kwargs.get("parent_pending_id")
        if parent_pending_id is not None:
            try:
                parent_pending_id = str(parent_pending_id)
            except Exception:
                parent_pending_id = None
        is_recheck = bool(parent_pending_id)
        condition_data = kwargs.get("condition_data") if isinstance(kwargs.get("condition_data"), dict) else {}
        check_detail = kwargs.get("check_detail") if isinstance(kwargs.get("check_detail"), dict) else {}
        pending_id = kwargs.get("pending_id") or parent_pending_id
        side_hint = kwargs.get("side")
        timeframe_hint = kwargs.get("timeframe") or kwargs.get("tf") or self.signal_tf
        regime_data = kwargs.get("regime_data") or kwargs.get("regime")
        if regime_data is not None and not isinstance(regime_data, dict):
            regime_data = None
        shock_state = kwargs.get("shock_state")
        shock_score = kwargs.get("shock_score")
        rsi_zone_snapshot = kwargs.get("rsi_zone_snapshot")
        if rsi_zone_snapshot is not None and not isinstance(rsi_zone_snapshot, dict):
            rsi_zone_snapshot = None
        rsi_zone_router_cfg = kwargs.get("rsi_zone_router_cfg")
        if rsi_zone_router_cfg is not None and not isinstance(rsi_zone_router_cfg, dict):
            rsi_zone_router_cfg = None
        rsi_zone_router_cfg = dict(rsi_zone_router_cfg) if isinstance(rsi_zone_router_cfg, dict) else {}
        rsi_router_enabled = bool(rsi_zone_router_cfg.get("enabled", False))
        try:
            shock_score = float(shock_score) if shock_score is not None else None
        except Exception:
            shock_score = None

        def _coerce_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except Exception:
                return None

        def _bps_delta(px: Optional[float], ref: Optional[float]) -> Optional[float]:
            if px is None or ref is None or ref == 0:
                return None
            try:
                return (float(px) - float(ref)) / float(ref) * 10000.0
            except Exception:
                return None

        near_val = condition_data.get("near")
        try:
            near_str = str(near_val) if near_val is not None else "unknown"
        except Exception:
            near_str = "unknown"
        if not near_str.strip():
            near_str = "unknown"

        trigger_price = _coerce_float(condition_data.get("trigger_price"))
        eps_bps = _coerce_float(condition_data.get("eps_bps"))
        condition_price = _coerce_float(condition_data.get("price"))
        fast_watch_price = None
        micro_gate_watch_price = None
        fast_watch_touch_confirmed = False
        fast_watch_touch_eps_bps = None
        fast_watch_px_used = None
        fast_watch_dist_to_band_bps = None
        if isinstance(check_detail, dict):
            fast_watch = check_detail.get("fast_watch")
            if isinstance(fast_watch, dict):
                fast_watch_price = _coerce_float(fast_watch.get("price"))
                fast_watch_touch_confirmed = bool(fast_watch.get("touch_confirmed", False))
                fast_watch_touch_eps_bps = _coerce_float(fast_watch.get("touch_eps_bps"))
                fast_watch_px_used = _coerce_float(fast_watch.get("px_used"))
                fast_watch_dist_to_band_bps = _coerce_float(fast_watch.get("dist_to_band_bps"))
            micro_gate_watch = check_detail.get("micro_gate_watch")
            if isinstance(micro_gate_watch, dict):
                micro_gate_watch_price = _coerce_float(micro_gate_watch.get("price"))

        market_price = None
        recheck_eval_emitted = False
        volume_analysis: Dict[str, Any] = {
            "source": "unresolved",
            "volume_strength": None,
            "volume_bucket": None,
            "upstream": {},
            "local": {"available": False, "reason": "not_evaluated"},
        }
        promote_override_meta: Dict[str, Any] = {
            "enabled": bool(getattr(self, "_promote_override_enabled", True)),
            "configured_mode": str(getattr(self, "_promote_override_mode", "observe")),
            "scope_reason": "not_evaluated",
            "mode_enforced": False,
            "candidate": False,
            "applied": False,
            "canary_symbols_count": len(getattr(self, "_promote_override_canary_symbols", set()) or set()),
            # Snapshot inputs for downstream TRADE_CLOSED observability.
            "near": None,
            "touch_confirmed": None,
            "dist_bps": None,
            "z": None,
            "adx": None,
            "volume_strength": None,
            "volume_bucket": None,
            "shock_state": None,
        }

        def _emit_recheck_eval(
            *,
            action: str,
            gate_reasons: Optional[list[str]] = None,
            px: Optional[float] = None,
            px_source: Optional[str] = None,
            lower: Optional[float] = None,
            upper: Optional[float] = None,
            vwap: Optional[float] = None,
            vwap_std: Optional[float] = None,
            z: Optional[float] = None,
            side_value: Optional[str] = None,
            rearm_recommended: Optional[bool] = None,
            rearm_reason: Optional[str] = None,
        ) -> None:
            nonlocal recheck_eval_emitted
            if not is_recheck or recheck_eval_emitted:
                return None
            recheck_eval_emitted = True
            reasons = gate_reasons or []
            if not reasons:
                reasons = ["unknown"]
            if px_source is None:
                if px is not None:
                    if (
                        is_recheck
                        and fast_watch_price is not None
                        and math.isfinite(fast_watch_price)
                        and px == fast_watch_price
                    ):
                        px_source_val = "fast_watch"
                    elif (
                        is_recheck
                        and micro_gate_watch_price is not None
                        and math.isfinite(micro_gate_watch_price)
                        and px == micro_gate_watch_price
                    ):
                        px_source_val = "micro_gate_watch"
                    else:
                        px_source_val = "market_price"
                elif fast_watch_price is not None:
                    px_source_val = "fast_watch"
                elif micro_gate_watch_price is not None:
                    px_source_val = "micro_gate_watch"
                elif condition_price is not None:
                    px_source_val = "condition_data_price"
                else:
                    px_source_val = "unknown"
            else:
                px_source_val = px_source
            dist_to_lower_bps = _bps_delta(px, lower)
            dist_to_upper_bps = _bps_delta(px, upper)
            dist_to_trigger_bps = _bps_delta(px, trigger_price)
            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            out = {
                "event": "mr_recheck_eval",
                "ts_ms": ts_ms,
                "run_id": get_current_run_id(),
                "symbol": symbol,
                "timeframe": str(timeframe_hint) if timeframe_hint is not None else None,
                "side": str(side_value or side_hint) if (side_value or side_hint) is not None else None,
                "pending_id": str(pending_id) if pending_id is not None else None,
                "parent_pending_id": str(parent_pending_id) if parent_pending_id is not None else None,
                "near": near_str,
                "trigger_price": trigger_price,
                "market_price": market_price,
                "fast_watch_price": fast_watch_price,
                "micro_gate_watch_price": micro_gate_watch_price,
                "eps_bps": eps_bps,
                "px": px,
                "px_used": px,
                "px_source": px_source_val,
                "lower": lower,
                "upper": upper,
                "vwap": vwap,
                "vwap_std": vwap_std,
                "z": z,
                "dist_to_lower_bps": dist_to_lower_bps,
                "dist_to_upper_bps": dist_to_upper_bps,
                "dist_to_trigger_bps": dist_to_trigger_bps,
                "action": str(action),
                "gate_reasons": list(reasons),
                "primary_gate_reason": reasons[0] if reasons else "unknown",
                "rearm_recommended": rearm_recommended,
                "rearm_reason": rearm_reason,
                "volume_strength": volume_analysis.get("volume_strength"),
                "volume_bucket": volume_analysis.get("volume_bucket"),
                "volume_source": volume_analysis.get("source"),
                "promote_override_mode": promote_override_meta.get("configured_mode"),
                "promote_override_scope": promote_override_meta.get("scope_reason"),
                "promote_override_candidate": promote_override_meta.get("candidate"),
                "promote_override_applied": promote_override_meta.get("applied"),
            }
            try:
                logger.info("mr_recheck_eval %s", json.dumps(out, ensure_ascii=True, sort_keys=True))
            except Exception:
                logger.info("mr_recheck_eval %s", out)
            return out

        if rsi_router_enabled:
            allowed_by_router, router_reason = is_strategy_allowed(
                self.strategy_name,
                side_hint,
                rsi_zone_snapshot,
                rsi_zone_router_cfg,
            )
            if not allowed_by_router:
                router_reason_code = str(router_reason or "rsi_router.zone_mismatch")
                router_log_ctx = rsi_snapshot_log_context(rsi_zone_snapshot)
                logger.info(
                    "[MeanReversion] RSI router veto %s: reason=%s zone=%s rsi_level=%s rsi_slow=%s rsi_fast=%s ob_threshold=%s str_threshold=%s consensus_status=%s",
                    symbol,
                    router_reason_code,
                    (rsi_zone_snapshot or {}).get("zone") if isinstance(rsi_zone_snapshot, dict) else None,
                    router_log_ctx.get("rsi_level"),
                    router_log_ctx.get("rsi_slow"),
                    router_log_ctx.get("rsi_fast"),
                    router_log_ctx.get("ob_threshold"),
                    router_log_ctx.get("str_threshold"),
                    router_log_ctx.get("consensus_status"),
                )
                eval_out = _emit_recheck_eval(
                    action="HOLD",
                    gate_reasons=[router_reason_code],
                    rearm_recommended=False,
                    rearm_reason="rsi_router.deferral_cancelled",
                )
                if is_recheck:
                    decision_meta = {
                        "action": "HOLD",
                        "rearm_fast_watch": False,
                        "rearm_reason": "rsi_router.deferral_cancelled",
                        "reason_code": router_reason_code,
                        "rsi_zone_snapshot": dict(rsi_zone_snapshot) if isinstance(rsi_zone_snapshot, dict) else None,
                    }
                    if isinstance(eval_out, dict):
                        decision_meta.setdefault("mr_recheck_eval", eval_out)
                    return {
                        "event_type": "strategy_recheck_decision",
                        "decision_meta": decision_meta,
                    }
                return None

        df_vwap = None
        df_sig = None

        # Prefer provided market_data or direct kwargs if present; be flexible on symbol key format
        if isinstance(market_data, dict) and market_data:
            variants = [symbol]
            if ':' in symbol:
                variants.append(symbol.split(':')[0])
            variants.extend(list(market_data.keys()))
            def _lookup(tf: str):
                for key in variants:
                    if key in market_data and isinstance(market_data[key], dict):
                        cand = market_data[key].get(tf)
                        if cand is not None:
                            return cand
                    if key == tf and tf in market_data:
                        return market_data[tf]
                # direct tf-level lookup
                if tf in market_data:
                    return market_data[tf]
                return None
            df_vwap = _lookup(self.vwap_tf)
            if df_vwap is None:
                df_vwap = market_data.get("df_vwap")
            df_sig = _lookup(self.signal_tf)
            if df_sig is None:
                df_sig = market_data.get("df_sig")
        if df_vwap is None and "df_vwap" in kwargs:
            df_vwap = kwargs.get("df_vwap")
        if df_sig is None and "df_sig" in kwargs:
            df_sig = kwargs.get("df_sig")
        # Fallback to pipeline if still missing
        if (df_vwap is None or df_sig is None) and self.market_data_pipeline:
            df_vwap = df_vwap or await self.market_data_pipeline.get_latest_ohlcv(symbol, self.vwap_tf, limit=self.min_rows)
            df_sig = df_sig or await self.market_data_pipeline.get_latest_ohlcv(symbol, self.signal_tf, limit=max(self.min_signal_rows, 1000))
        elif df_vwap is None or df_sig is None:
            logger.warning(f"[MeanReversion] MarketDataPipeline missing and no data supplied for {symbol}. Aborting.")
            _emit_recheck_eval(action="HOLD", gate_reasons=["data_missing"])
            return None

        if df_vwap is None or df_vwap.empty or df_sig is None or df_sig.empty:
            logger.warning(f"[MeanReversion] Data missing: vwap_empty={df_vwap is None or df_vwap.empty} "
                           f"sig_empty={df_sig is None or df_sig.empty} for {symbol}")
            _emit_recheck_eval(action="HOLD", gate_reasons=["data_missing"])
            return None

        # Ensure time order and required columns for VWAP math
        if not df_vwap.index.is_monotonic_increasing:
            df_vwap = df_vwap.sort_index()
        if not df_sig.index.is_monotonic_increasing:
            df_sig = df_sig.sort_index()
        if "volume" not in df_vwap.columns:
            logger.warning(f"[MeanReversion] Missing volume column in VWAP dataframe for {symbol}")
            _emit_recheck_eval(action="HOLD", gate_reasons=["missing_volume"])
            return None

        logger.info(
            f"[MeanReversion] Data rows: vwap={len(df_vwap)}, signal_tf={len(df_sig)}, "
            f"min_vwap={self.min_rows}, min_signal={self.min_signal_rows}"
        )

        repair_attempted = False
        used_fallback_fetch = False
        repair_reason = None

        def _safe_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        safe_sig_min = max(1000, 255, _safe_int(getattr(self, "min_signal_rows", 0) or 0, 0))
        safe_vwap_min = max(1000, _safe_int(getattr(self, "min_rows", 0) or 0, 0))
        try:
            # Optional warmup-aware bump (min_periods for VWAP/std is lookback//2).
            safe_vwap_min = max(safe_vwap_min, int(max(int(self.vwap_lookback or 0) // 2, 0) + 50))
        except Exception:
            pass

        target_sig_min = safe_sig_min if is_recheck else int(self.min_signal_rows)
        target_vwap_min = safe_vwap_min if is_recheck else int(self.min_rows)

        clean_vwap = self._clean_vwap_df(df_vwap)
        clean_sig = self._clean_sig_df(df_sig)

        async def _maybe_repair_fetch(reason: str) -> None:
            nonlocal df_vwap, df_sig, clean_vwap, clean_sig, repair_attempted, used_fallback_fetch, repair_reason
            if repair_attempted:
                return
            if not self.market_data_pipeline:
                return
            repair_attempted = True
            repair_reason = reason
            try:
                refreshed_vwap = None
                refreshed_sig = None
                try:
                    refreshed_vwap = await self.market_data_pipeline.get_latest_ohlcv(
                        symbol,
                        self.vwap_tf,
                        limit=int(target_vwap_min),
                    )
                except Exception:
                    refreshed_vwap = None
                try:
                    refreshed_sig = await self.market_data_pipeline.get_latest_ohlcv(
                        symbol,
                        self.signal_tf,
                        limit=int(target_sig_min),
                    )
                except Exception:
                    refreshed_sig = None
                if refreshed_vwap is not None and not refreshed_vwap.empty:
                    df_vwap = refreshed_vwap
                if refreshed_sig is not None and not refreshed_sig.empty:
                    df_sig = refreshed_sig
                used_fallback_fetch = True
                if not df_vwap.index.is_monotonic_increasing:
                    df_vwap = df_vwap.sort_index()
                if not df_sig.index.is_monotonic_increasing:
                    df_sig = df_sig.sort_index()
                clean_vwap = self._clean_vwap_df(df_vwap)
                clean_sig = self._clean_sig_df(df_sig)
                logger.info(f"[MeanReversion] Repair fetch: vwap={len(df_vwap)}, sig={len(df_sig)} reason={reason}")
            except Exception as e:
                logger.warning(f"[MeanReversion] Repair fetch failed: {e}")

        if is_recheck:
            if len(df_vwap) < target_vwap_min or len(df_sig) < target_sig_min:
                await _maybe_repair_fetch("injected_too_small")
            elif clean_vwap.empty or clean_sig.empty:
                await _maybe_repair_fetch("clean_empty")
        else:
            if self.market_data_pipeline and (len(df_vwap) < self.min_rows or len(df_sig) < self.min_signal_rows):
                await _maybe_repair_fetch("min_rows_insufficient")

        if len(df_vwap) < self.min_rows:
            logger.warning(f"[MeanReversion] VWAP data insufficient. Have vwap={len(df_vwap)}, "
                           f"Need>={self.min_rows}. Aborting.")
            _emit_recheck_eval(action="HOLD", gate_reasons=["min_rows_insufficient"])
            return None
        if len(df_sig) < self.min_signal_rows:
            logger.warning(f"[MeanReversion] Signal data insufficient. Have sig={len(df_sig)}, "
                           f"Need>={self.min_signal_rows}. Aborting.")
            _emit_recheck_eval(action="HOLD", gate_reasons=["min_signal_rows_insufficient"])
            return None

        if clean_vwap.empty:
            logger.warning(
                f"[MeanReversion] Indicator calculation resulted in empty VWAP dataframe after dropna "
                f"(input rows={len(df_vwap)})."
            )
            logger.debug(f"[MeanReversion] VWAP columns: {list(df_vwap.columns)}")
            if is_recheck:
                logger.info(
                    "[MeanReversion] Recheck debug: vwap_clean_rows=0 sig_clean_rows=%s used_fallback_fetch=%s repair_reason=%s",
                    len(clean_sig),
                    used_fallback_fetch,
                    repair_reason,
                )
            _emit_recheck_eval(action="HOLD", gate_reasons=["clean_vwap_empty"])
            return None
        if clean_sig.empty:
            logger.warning(
                f"[MeanReversion] Indicator calculation resulted in empty signal dataframe after dropna "
                f"(input rows={len(df_sig)})."
            )
            logger.debug(f"[MeanReversion] Signal columns: {list(df_sig.columns)}")
            if is_recheck:
                logger.info(
                    "[MeanReversion] Recheck debug: vwap_clean_rows=%s sig_clean_rows=0 used_fallback_fetch=%s repair_reason=%s",
                    len(clean_vwap),
                    used_fallback_fetch,
                    repair_reason,
                )
            _emit_recheck_eval(action="HOLD", gate_reasons=["clean_sig_empty"])
            return None

        last_vwap = clean_vwap.iloc[-1]
        last_sig = clean_sig.iloc[-1]

        self._maybe_warn_pipeline_indicator_mismatch()

        required_cols_vwap = {"vwap", "vwap_lower", "vwap_upper"}
        if not required_cols_vwap.issubset(set(last_vwap.index)):
            logger.warning(f"[MeanReversion] Missing required VWAP columns for {symbol}: "
                           f"{required_cols_vwap - set(last_vwap.index)}")
            _emit_recheck_eval(action="HOLD", gate_reasons=["missing_vwap_columns"])
            return None
        if "adx" not in last_sig.index:
            logger.warning(f"[MeanReversion] Missing ADX column for {symbol}")
            _emit_recheck_eval(action="HOLD", gate_reasons=["missing_adx"])
            return None

        sig_close_price = float(last_sig["close"])
        price = sig_close_price
        px_source = "signal_close"
        market_price_source = "signal_close"
        market_price_fallback_chain = None
        market_price = sig_close_price

        if self.price_source != "signal_close" and self.market_data_pipeline:
            forming_close = None
            try:
                forming_close = self.market_data_pipeline.get_realtime_price(symbol, timeframe=self.signal_tf)
            except Exception:
                forming_close = None

            live_price = None
            resolved_source = None
            fallback_chain = None
            try:
                if self.price_source == "forming_close":
                    live_price = forming_close
                    resolved_source = "forming_close"
                    fallback_chain = "forming_close"
                else:
                    live_price, resolved_source, fallback_chain = self.market_data_pipeline.get_live_trigger_price(
                        symbol=symbol,
                        timeframe=self.signal_tf,
                        source=self.price_source,
                        forming_close=forming_close,
                    )
            except Exception:
                live_price = None

            try:
                if live_price is not None and math.isfinite(float(live_price)) and float(live_price) > 0:
                    market_price = float(live_price)
                    market_price_source = str(resolved_source or self.price_source)
                    market_price_fallback_chain = fallback_chain
                    price = market_price
                    px_source = market_price_source
            except Exception:
                pass
        vwap_main = float(last_vwap["vwap"])
        vwap_lower = float(last_vwap["vwap_lower"])
        vwap_upper = float(last_vwap["vwap_upper"])
        adx_val = float(last_sig["adx"])

        if math.isnan(vwap_main) or math.isnan(vwap_lower) or math.isnan(vwap_upper) or math.isnan(adx_val):
            logger.warning(f"[MeanReversion] NaN detected in indicators for {symbol}: "
                           f"vwap={vwap_main}, lower={vwap_lower}, upper={vwap_upper}, adx={adx_val}")
            _emit_recheck_eval(action="HOLD", gate_reasons=["nan_indicator"])
            return None

        atr_val = float(last_sig["atr"]) if "atr" in last_sig.index else None

        controller_decision = self._maybe_apply_dynamic_controller(
            symbol=symbol,
            df_vwap=df_vwap,
            df_sig=df_sig,
            price=price,
            vwap=vwap_main,
            vwap_std=float(last_vwap["vwap_std"]) if "vwap_std" in last_vwap.index else None,
            adx=adx_val,
            atr=atr_val,
        )
        lower = float(controller_decision.lower) if controller_decision else vwap_lower
        upper = float(controller_decision.upper) if controller_decision else vwap_upper
        vwap_target = float(controller_decision.vwap) if controller_decision else vwap_main
        vwap_std_val = None
        try:
            if controller_decision is not None and controller_decision.vwap_std is not None:
                vwap_std_val = float(controller_decision.vwap_std)
        except Exception:
            vwap_std_val = None
        if vwap_std_val is None:
            try:
                if "vwap_std" in last_vwap.index:
                    vwap_std_val = float(last_vwap["vwap_std"])
            except Exception:
                vwap_std_val = None
        if is_recheck and micro_gate_watch_price is not None and math.isfinite(micro_gate_watch_price):
            price = float(micro_gate_watch_price)
            px_source = "micro_gate_watch"
        elif is_recheck and fast_watch_price is not None and math.isfinite(fast_watch_price):
            price = float(fast_watch_price)
            px_source = "fast_watch"

        fw_override = condition_data.get("fast_watch") if isinstance(condition_data, dict) else None
        if fw_override is not None and not isinstance(fw_override, dict):
            fw_override = None

        allow_touch_entry = bool(self._fast_watch_allow_touch_entry_default) if self._fast_watch_v2_enabled else False
        near_bps_for_rearm = eps_bps
        if self._fast_watch_v2_enabled:
            if self._fast_watch_near_bps_default is not None:
                near_bps_for_rearm = float(self._fast_watch_near_bps_default)
            if isinstance(fw_override, dict) and fw_override.get("near_bps") is not None:
                try:
                    near_bps_for_rearm = float(fw_override.get("near_bps"))
                except Exception:
                    pass
            if isinstance(fw_override, dict) and fw_override.get("allow_touch_entry") is not None:
                try:
                    allow_touch_entry = bool(fw_override.get("allow_touch_entry"))
                except Exception:
                    pass

        touch_entry_allowed = False
        if is_recheck and fast_watch_touch_confirmed and allow_touch_entry:
            px_for_touch = fast_watch_px_used if fast_watch_px_used is not None else fast_watch_price
            eps_bps_for_touch = fast_watch_touch_eps_bps
            if px_for_touch is not None and trigger_price is not None and eps_bps_for_touch is not None:
                eps_px = abs(float(trigger_price)) * (float(eps_bps_for_touch) / 10000.0)
                side_norm = str(side_hint or "").strip().lower()
                if side_norm in ("long", "buy"):
                    touch_entry_allowed = bool(float(px_for_touch) <= float(trigger_price) + eps_px)
                elif side_norm in ("short", "sell"):
                    touch_entry_allowed = bool(float(px_for_touch) >= float(trigger_price) - eps_px)

        z_val = None
        if vwap_std_val is not None and math.isfinite(vwap_std_val) and vwap_std_val > 0:
            try:
                z_val = (price - vwap_target) / vwap_std_val
            except Exception:
                z_val = None
        volume_analysis = self._resolve_volume_analysis(
            kwargs=kwargs,
            check_detail=check_detail if isinstance(check_detail, dict) else None,
            df_sig=clean_sig if isinstance(clean_sig, pd.DataFrame) else df_sig,
        )
        volume_strength = self._coerce_finite_float(volume_analysis.get("volume_strength"))
        volume_bucket = self._normalize_volume_bucket(volume_analysis.get("volume_bucket"))
        ema_stack = self._get_ema_stack(
            symbol=symbol,
            df_sig=clean_sig if isinstance(clean_sig, pd.DataFrame) else df_sig,
            market_data=market_data if isinstance(market_data, dict) else None,
            kwargs=kwargs,
        )

        # -----------------------------------------------------------
        # OPPORTUNITY PROMOTION (Recheck override)
        # -----------------------------------------------------------
        promotion_override_candidate = False
        promotion_override = False
        promote_mode_enforced, promote_scope_reason = self._is_promote_override_enforced_for_symbol(symbol)
        if is_recheck and fast_watch_touch_confirmed and (near_str in ["lower", "upper"]):
            _target_band_val = lower if near_str == "lower" else upper
            _dist_bps = None
            if fast_watch_dist_to_band_bps is not None and math.isfinite(float(fast_watch_dist_to_band_bps)):
                _dist_bps = float(fast_watch_dist_to_band_bps)
            elif price and _target_band_val and price > 0:
                try:
                    _dist_bps = abs(price - _target_band_val) / price * 10000.0
                except Exception:
                    _dist_bps = None
            promote_override_meta.update(
                {
                    "near": near_str,
                    "touch_confirmed": bool(fast_watch_touch_confirmed),
                    "dist_bps": self._coerce_finite_float(_dist_bps),
                    "z": self._coerce_finite_float(z_val),
                    "adx": self._coerce_finite_float(adx_val),
                    "volume_strength": self._coerce_finite_float(volume_strength),
                    "volume_bucket": volume_bucket,
                    "shock_state": str(shock_state).upper().strip() if shock_state is not None else None,
                }
            )

            promotion_override_candidate = self.check_promotion_override(
                touch_confirmed=fast_watch_touch_confirmed,
                near=near_str,
                dist_bps=_dist_bps,
                z=z_val,
                adx=adx_val,
                shock_state=shock_state,
                regime_data=regime_data,
                volume_strength=volume_strength,
                ema_stack=ema_stack,
            )
            promotion_override = bool(promotion_override_candidate and promote_mode_enforced)
            if promotion_override_candidate and not promotion_override:
                logger.info(
                    "[MeanReversion] PROMOTE observe-only %s: mode=%s scope=%s near=%s",
                    symbol,
                    str(getattr(self, "_promote_override_mode", "observe")),
                    promote_scope_reason,
                    near_str,
                )
        promote_override_meta.update(
            {
                "scope_reason": str(promote_scope_reason),
                "mode_enforced": bool(promote_mode_enforced),
                "candidate": bool(promotion_override_candidate),
                "applied": bool(promotion_override),
            }
        )

        # -----------------------------------------------------------
        # DYNAMIC Z-THRESHOLD (ADX-sensitive)
        # -----------------------------------------------------------
        required_z = 1.25
        guard_state_pre = self._rsi_guard_state_by_symbol.get(symbol, "IDLE")
        if adx_val is not None and math.isfinite(adx_val):
            if 22.0 < float(adx_val) < 25.0:
                required_z = max(required_z, 1.60)
            elif float(adx_val) >= 25.0:
                required_z = max(required_z, float(self._high_adx_z_threshold))

        dynamic_z_skip_touch_recheck = bool(is_recheck and fast_watch_touch_confirmed)
        if z_val is not None and math.isfinite(z_val):
            if dynamic_z_skip_touch_recheck and guard_state_pre != "ARMED" and abs(float(z_val)) < required_z:
                logger.info(
                    "[MeanReversion] Dynamic Z bypass %s: z=%.2f required=%.2f (touch_confirmed recheck)",
                    symbol,
                    float(abs(z_val)),
                    float(required_z),
                )
            elif guard_state_pre != "ARMED" and abs(float(z_val)) < required_z:
                logger.info(
                    "[MeanReversion] Dynamic Z veto %s: z=%.2f required=%.2f adx=%.2f",
                    symbol,
                    float(abs(z_val)),
                    float(required_z),
                    float(adx_val) if adx_val is not None and math.isfinite(adx_val) else float("nan"),
                )
                _emit_recheck_eval(
                    action="HOLD",
                    gate_reasons=["dynamic_z_veto"],
                    px=price,
                    px_source=px_source,
                    lower=lower,
                    upper=upper,
                    vwap=vwap_target,
                    vwap_std=vwap_std_val,
                    z=z_val,
                    side_value=None,
                )
                return None

        in_band = lower <= price <= upper
        vol_state = None
        try:
            vol_state = str(getattr(controller_decision, "vol_state", "") or "").strip().lower() or None
        except Exception:
            vol_state = None

        eff_adx_threshold = float(self.adx_threshold)
        adx_slope = None
        adx_decision_reason = "strict"

        if not math.isfinite(adx_val):
            adx_ok = adx_val < float(self.adx_threshold)
            adx_decision_reason = "adx_nan"
        else:
            adx_ok = float(adx_val) < float(self.adx_threshold)
            adx_decision_reason = "strict_ok" if adx_ok else "strict_veto"

            squeeze_threshold = None
            try:
                squeeze_threshold = float(getattr(self, "_adx_squeeze_threshold", None))
            except Exception:
                squeeze_threshold = None
            if squeeze_threshold is not None and (not math.isfinite(squeeze_threshold) or squeeze_threshold <= 0):
                squeeze_threshold = None

            if (
                vol_state == "squeeze"
                and squeeze_threshold is not None
                and float(squeeze_threshold) > float(self.adx_threshold)
                and float(adx_val) <= float(squeeze_threshold)
            ):
                if float(adx_val) <= float(self.adx_threshold):
                    # Already inside strict threshold.
                    adx_ok = True
                    eff_adx_threshold = float(self.adx_threshold)
                    adx_decision_reason = "squeeze_inside_strict"
                else:
                    # Extended zone: allow only if ADX is flat/falling (slope <= eps).
                    lookback = int(getattr(self, "_adx_slope_lookback", 5) or 5)
                    eps = float(getattr(self, "_adx_slope_eps", 0.0) or 0.0)
                    if lookback < 1:
                        lookback = 1

                    adx_series = None
                    try:
                        if isinstance(clean_sig, pd.DataFrame) and "adx" in clean_sig.columns:
                            adx_series = pd.to_numeric(clean_sig["adx"], errors="coerce")
                    except Exception:
                        adx_series = None

                    if adx_series is not None and len(adx_series) > lookback:
                        try:
                            cur = float(adx_series.iloc[-1])
                            prev = float(adx_series.iloc[-1 - lookback])
                            if math.isfinite(cur) and math.isfinite(prev):
                                adx_slope = float(cur - prev)
                        except Exception:
                            adx_slope = None

                    if adx_slope is not None and math.isfinite(float(adx_slope)) and float(adx_slope) <= float(eps):
                        adx_ok = True
                        eff_adx_threshold = float(squeeze_threshold)
                        adx_decision_reason = "squeeze_extended_flat_ok"
                    else:
                        adx_ok = False
                        eff_adx_threshold = float(self.adx_threshold)
                        adx_decision_reason = "squeeze_extended_rising_veto"
            elif vol_state == "squeeze" and squeeze_threshold is not None and float(adx_val) > float(squeeze_threshold):
                # Even in squeeze, do not relax above the squeeze threshold.
                adx_ok = False
                eff_adx_threshold = float(self.adx_threshold)
                adx_decision_reason = "squeeze_above_threshold_veto"

        logger.debug(
            "[MeanReversion] ADX veto eval %s vol_state=%s adx=%.4f adx_slope=%s eff_th=%.4f decision=%s",
            symbol,
            vol_state or "none",
            float(adx_val) if math.isfinite(adx_val) else float("nan"),
            f"{float(adx_slope):.4f}" if adx_slope is not None and math.isfinite(float(adx_slope)) else "nan",
            float(eff_adx_threshold) if math.isfinite(eff_adx_threshold) else float("nan"),
            adx_decision_reason,
        )
        reentry_guard_long_active = False
        reentry_guard_short_active = False
        if self._reentry_guard_enabled and self._reentry_guard_require_vwap_reclaim:
            try:
                guard_key = str(symbol)
            except Exception:
                guard_key = ""
            if guard_key and self._reentry_guard_long_by_symbol.get(guard_key):
                if price > vwap_target:
                    self._reentry_guard_long_by_symbol.pop(guard_key, None)
                    logger.info(
                        "[MeanReversion] Reentry LONG guard cleared %s: price %.4f > vwap %.4f",
                        symbol,
                        float(price),
                        float(vwap_target),
                    )
                else:
                    reentry_guard_long_active = True
        if self._reentry_guard_enabled and self._reentry_guard_short_enabled:
            try:
                guard_key = str(symbol)
            except Exception:
                guard_key = ""
            if guard_key and self._reentry_guard_short_by_symbol.get(guard_key):
                clear_reasons = []
                if (
                    self._reentry_guard_short_clear_on_band_breach
                    and upper is not None
                    and math.isfinite(float(upper))
                    and price > float(upper)
                ):
                    clear_reasons.append("band_breach")
                z_thr = float(self._reentry_guard_short_clear_on_z_threshold)
                if z_thr > 0 and z_val is not None and math.isfinite(float(z_val)) and abs(float(z_val)) >= z_thr:
                    clear_reasons.append("z_threshold")
                if clear_reasons:
                    self._reentry_guard_short_by_symbol.pop(guard_key, None)
                    logger.info(
                        "[MeanReversion] Reentry SHORT guard cleared %s: reasons=%s price=%.4f upper=%.4f z=%s",
                        symbol,
                        "|".join(clear_reasons),
                        float(price),
                        float(upper) if upper is not None and math.isfinite(float(upper)) else float("nan"),
                        f"{float(z_val):.3f}" if z_val is not None and math.isfinite(float(z_val)) else "nan",
                    )
                else:
                    reentry_guard_short_active = True

        entry_long = price < lower and adx_ok
        entry_short = price > upper and adx_ok
        if promotion_override and adx_ok:
            if near_str == "lower":
                entry_long = True
            elif near_str == "upper":
                entry_short = True
        elif promotion_override and not adx_ok:
            logger.info(
                "[MeanReversion] PROMOTE suppressed by ADX veto %s: adx=%.2f threshold=%.2f",
                symbol,
                float(adx_val) if adx_val is not None and math.isfinite(adx_val) else float("nan"),
                float(eff_adx_threshold),
            )

        guard_status = None
        guard_rsi_val = None
        guard_block_long = False
        guard_cfg = getattr(self, "_rsi_guard_cfg", {}) or {}
        guard_enabled = bool(guard_cfg.get("enabled", False))
        guard_state = self._rsi_guard_state_by_symbol.get(symbol, "IDLE")
        if guard_enabled and (entry_long or guard_state == "ARMED"):
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            guard_tf = str(guard_cfg.get("tf") or self.vwap_tf or "1m")
            use_closed_only = bool(guard_cfg.get("use_closed_only", True))
            df_guard = None
            if guard_tf == str(self.vwap_tf):
                df_guard = df_vwap
            elif guard_tf == str(self.signal_tf):
                df_guard = df_sig
            elif isinstance(market_data, dict):
                df_guard = market_data.get(guard_tf)
                if df_guard is None and isinstance(market_data.get(symbol), dict):
                    df_guard = market_data.get(symbol, {}).get(guard_tf)
            if df_guard is None and self.market_data_pipeline:
                guard_limit = max(50, int(self.min_signal_rows or 0) or 50)
                try:
                    df_guard = await self.market_data_pipeline.get_latest_ohlcv(
                        symbol,
                        guard_tf,
                        limit=guard_limit,
                        include_forming=not use_closed_only,
                    )
                except Exception:
                    df_guard = None
            if isinstance(df_guard, pd.DataFrame) and not df_guard.empty:
                if not df_guard.index.is_monotonic_increasing:
                    df_guard = df_guard.sort_index()
                eval_df = df_guard
                if use_closed_only and bool(getattr(df_guard, "attrs", {}).get("includes_forming", False)):
                    if len(df_guard) >= 2:
                        eval_df = df_guard.iloc[:-1]
                if len(eval_df) > 0:
                    last_guard = eval_df.iloc[-1]
                    if "rsi" in last_guard.index:
                        try:
                            guard_rsi_val = float(last_guard["rsi"])
                        except Exception:
                            guard_rsi_val = None
                if guard_rsi_val is None and "close" in eval_df.columns:
                    try:
                        rsi_series = calc_rsi(eval_df["close"], period=14)
                        if not rsi_series.empty:
                            guard_rsi_val = float(rsi_series.iloc[-1])
                    except Exception:
                        guard_rsi_val = None
            if guard_rsi_val is not None:
                if not math.isfinite(guard_rsi_val) or guard_rsi_val < 0 or guard_rsi_val > 100:
                    guard_rsi_val = None
            guard_z_abs = None
            if z_val is not None and math.isfinite(z_val):
                guard_z_abs = abs(float(z_val))

            if guard_rsi_val is None or guard_z_abs is None:
                if guard_state == "ARMED":
                    self._rsi_guard_state_by_symbol[symbol] = "IDLE"
                    self._rsi_guard_armed_ts_ms_by_symbol[symbol] = 0
                if entry_long:
                    guard_status = "bypassed_low_z"
            else:
                activation_rsi = float(guard_cfg.get("activation_rsi", 25.0))
                activation_z = float(guard_cfg.get("activation_z_score", 2.2))
                rebound_rsi = float(guard_cfg.get("rebound_rsi", 27.0))
                max_wait_s = int(guard_cfg.get("max_wait_s", 120) or 0)
                max_wait_ms = max_wait_s * 1000
                if guard_state == "IDLE":
                    if entry_long and guard_rsi_val <= activation_rsi and guard_z_abs >= activation_z:
                        self._rsi_guard_state_by_symbol[symbol] = "ARMED"
                        self._rsi_guard_armed_ts_ms_by_symbol[symbol] = now_ms
                        guard_status = "armed"
                        guard_block_long = True
                    elif entry_long:
                        guard_status = "bypassed_low_z"
                else:
                    armed_ts_ms = int(self._rsi_guard_armed_ts_ms_by_symbol.get(symbol, 0) or 0)
                    if max_wait_ms and armed_ts_ms and now_ms - armed_ts_ms > max_wait_ms:
                        self._rsi_guard_state_by_symbol[symbol] = "IDLE"
                        self._rsi_guard_armed_ts_ms_by_symbol[symbol] = 0
                        guard_status = "expired"
                    elif guard_rsi_val >= rebound_rsi:
                        self._rsi_guard_state_by_symbol[symbol] = "IDLE"
                        self._rsi_guard_armed_ts_ms_by_symbol[symbol] = 0
                        guard_status = "triggered_and_valid" if entry_long else "expired"
                    else:
                        guard_status = "waiting_for_rebound"
                        if entry_long:
                            guard_block_long = True

        # ------------------------------------------------------------------
        # Rejection Confirmation (SHORT) - closed-only evaluation
        # ------------------------------------------------------------------
        rejection_meta = None
        rej_cfg = self.strategy_config.get("rejection_confirmation") if isinstance(self.strategy_config, dict) else {}
        rej_enabled = True
        wick_ratio_min = 0.8
        recheck_mode = "observe"
        try:
            rej_enabled = bool(rej_cfg.get("enabled", True)) if isinstance(rej_cfg, dict) else True
        except Exception:
            rej_enabled = True
        try:
            wick_ratio_min = float(rej_cfg.get("upper_wick_ratio_min", 0.8) or 0.8) if isinstance(rej_cfg, dict) else 0.8
        except Exception:
            wick_ratio_min = 0.8
        if isinstance(rej_cfg, dict):
            try:
                recheck_mode = str(rej_cfg.get("recheck_mode", "observe") or "observe").strip().lower()
            except Exception:
                recheck_mode = "observe"
        if recheck_mode not in {"observe", "enforce", "off", "disabled"}:
            recheck_mode = "observe"
        recheck_enforce = bool(is_recheck and recheck_mode == "enforce")

        if rej_enabled and entry_short is not None:
            candle_row = last_sig
            includes_forming = False
            used_prev_closed = False
            try:
                includes_forming = bool(getattr(clean_sig, "attrs", {}).get("includes_forming", False))
            except Exception:
                includes_forming = False
            if includes_forming and isinstance(clean_sig, pd.DataFrame) and len(clean_sig) >= 2:
                candle_row = clean_sig.iloc[-2]
                used_prev_closed = True
            try:
                candle_open = float(candle_row.get("open"))
                candle_close = float(candle_row.get("close"))
                candle_high = float(candle_row.get("high"))
                candle_low = float(candle_row.get("low"))
            except Exception:
                candle_open = candle_close = candle_high = candle_low = None

            if candle_open is None or candle_close is None or candle_high is None:
                rejection_meta = {
                    "enabled": rej_enabled,
                    "recheck_mode": recheck_mode if is_recheck else "n/a",
                    "enforced": bool((not is_recheck) or recheck_enforce),
                    "evaluation": "skipped_missing_ohlc",
                    "includes_forming": includes_forming,
                    "used_prev_closed": used_prev_closed,
                }
                if is_recheck:
                    logger.info(
                        "[MeanReversion] Rejection confirmation skipped on recheck %s: missing OHLC (mode=%s)",
                        symbol,
                        recheck_mode,
                    )
            else:
                has_red = candle_close < candle_open
                body_size = abs(candle_close - candle_open)
                upper_wick = candle_high - max(candle_open, candle_close)
                if upper_wick < 0:
                    upper_wick = 0.0
                if body_size > 0:
                    upper_wick_ratio = upper_wick / body_size
                else:
                    upper_wick_ratio = float("inf") if upper_wick > 0 else 0.0
                close_back_inside = bool(upper) and candle_close < float(upper)
                touched_upper = bool(upper) and candle_high >= float(upper)
                rejected_from_band = close_back_inside or (upper_wick_ratio >= wick_ratio_min)

                rejection_meta = {
                    "enabled": rej_enabled,
                    "has_red": has_red,
                    "close_back_inside_band": close_back_inside,
                    "upper_wick_ratio": upper_wick_ratio,
                    "touched_upper": touched_upper,
                    "threshold_wick_ratio": wick_ratio_min,
                    "includes_forming": includes_forming,
                    "used_prev_closed": used_prev_closed,
                    "recheck_mode": recheck_mode if is_recheck else "n/a",
                    "enforced": bool((not is_recheck) or recheck_enforce),
                }

                # Keep legacy behavior in recheck observe mode: do not mutate entry decisions yet.
                if ((not is_recheck) or recheck_enforce) and (not entry_short) and touched_upper and has_red and rejected_from_band and adx_ok:
                    entry_short = True
                    rejection_meta["forced_entry"] = True

                rejection_failed = bool(entry_short and not (has_red and rejected_from_band))
                if rejection_failed:
                    logger.info(
                        "[MeanReversion] Rejection confirmation failed for %s: has_red=%s close_in_band=%s "
                        "upper_wick_ratio=%.2f thr=%.2f mode=%s",
                        symbol,
                        has_red,
                        close_back_inside,
                        upper_wick_ratio,
                        wick_ratio_min,
                        recheck_mode if is_recheck else "live",
                    )
                    if is_recheck and not recheck_enforce:
                        rejection_meta["observed_fail"] = True
                        rejection_meta["enforced"] = False
                    else:
                        return None

        rejection_shadow = self._update_vsa_rejection_history(
            symbol=symbol,
            rejection_meta=rejection_meta if isinstance(rejection_meta, dict) else None,
        )

        if in_band and not (entry_long or entry_short) and not touch_entry_allowed and not promotion_override:
            if parent_pending_id:
                logger.info(
                    "[MeanReversion] Recheck context; skipping soft deferral near-miss for %s (parent_pending_id=%s)",
                    symbol,
                    parent_pending_id,
                )
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                eps_bps_val = near_bps_for_rearm
                dist_to_trigger_bps = _bps_delta(price, trigger_price)
                fw_meta = check_detail.get("fast_watch_meta") if isinstance(check_detail, dict) else None
                rearm_count = 0
                max_rearms = 1
                expires_at_ms = None
                if isinstance(fw_meta, dict):
                    try:
                        rearm_count = int(fw_meta.get("rearm_count", 0) or 0)
                    except Exception:
                        rearm_count = 0
                    try:
                        max_rearms = int(fw_meta.get("max_rearms", 1) or 1)
                    except Exception:
                        max_rearms = 1
                    expires_at_ms = fw_meta.get("expires_at_ms")
                if expires_at_ms is None:
                    expires_at_ms = condition_data.get("expires_at_ms")
                remaining_ttl_ms = None
                if expires_at_ms is not None:
                    try:
                        remaining_ttl_ms = int(expires_at_ms) - int(now_ms)
                    except Exception:
                        remaining_ttl_ms = None

                rearm_recommended = False
                rearm_reason = "not_near"
                min_remaining_ms = 3000
                if eps_bps_val is None or not math.isfinite(float(eps_bps_val)) or float(eps_bps_val) <= 0:
                    rearm_recommended = False
                    rearm_reason = "eps_missing"
                elif dist_to_trigger_bps is None:
                    rearm_recommended = False
                    rearm_reason = "dist_missing"
                elif abs(float(dist_to_trigger_bps)) > float(eps_bps_val) * 1.5:
                    rearm_recommended = False
                    rearm_reason = "not_near"
                elif remaining_ttl_ms is not None and remaining_ttl_ms < min_remaining_ms:
                    rearm_recommended = False
                    rearm_reason = "ttl_low"
                elif rearm_count >= max_rearms:
                    rearm_recommended = False
                    rearm_reason = "rearm_limit"
                else:
                    rearm_recommended = True
                    rearm_reason = "still_near"

                eval_out = _emit_recheck_eval(
                    action="HOLD",
                    gate_reasons=["in_band"],
                    px=price,
                    px_source=px_source,
                    lower=lower,
                    upper=upper,
                    vwap=vwap_target,
                    vwap_std=vwap_std_val,
                    z=z_val,
                    side_value=None,
                    rearm_recommended=rearm_recommended,
                    rearm_reason=rearm_reason,
                )
                if is_recheck:
                    decision_meta = {
                        "action": "HOLD",
                        "rearm_fast_watch": bool(rearm_recommended),
                        "rearm_reason": rearm_reason,
                        "dist_to_trigger_bps": dist_to_trigger_bps,
                        "eps_bps": eps_bps_val,
                        "remaining_ttl_ms": remaining_ttl_ms,
                        "rearm_count": rearm_count,
                        "max_rearms": max_rearms,
                        "trigger_price": trigger_price,
                        "near": near_str,
                        "touch_confirmed": bool(fast_watch_touch_confirmed),
                        "allow_touch_entry": bool(allow_touch_entry),
                        "promotion_override": dict(promote_override_meta),
                    }
                    if isinstance(eval_out, dict):
                        decision_meta.setdefault("mr_recheck_eval", eval_out)
                    return {
                        "event_type": "strategy_recheck_decision",
                        "decision_meta": decision_meta,
                    }
                return None
            threshold = float(self.soft_deferral_threshold)
            if threshold > 0 and adx_ok and math.isfinite(threshold):
                near_lower = False
                near_upper = False
                dist_lower = None
                dist_upper = None
                if lower > 0:
                    dist_lower = max(0.0, (price - lower) / lower)
                    near_lower = dist_lower <= threshold
                if upper > 0:
                    dist_upper = max(0.0, (upper - price) / upper)
                    near_upper = dist_upper <= threshold

                if near_lower or near_upper:
                    choose_lower = bool(near_lower and not near_upper)
                    choose_upper = bool(near_upper and not near_lower)
                    if near_lower and near_upper:
                        try:
                            choose_lower = (dist_lower or 0.0) <= (dist_upper or 0.0)
                            choose_upper = not choose_lower
                        except Exception:
                            choose_lower = True
                            choose_upper = False

                    side = "long" if choose_lower else "short"

                    ts_ms = None
                    try:
                        ts_val = clean_sig.index[-1]
                        if isinstance(ts_val, pd.Timestamp):
                            ts = ts_val.to_pydatetime()
                        elif isinstance(ts_val, datetime):
                            ts = ts_val
                        else:
                            ts = None
                        if ts is not None and ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        ts_ms = int(ts.timestamp() * 1000) if ts is not None else None
                    except Exception:
                        ts_ms = None

                    if ts_ms is None:
                        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                    tf_ms = self._parse_timeframe_ms(self.signal_tf)
                    setup_anchor_ts_ms = int(ts_ms - (ts_ms % int(tf_ms))) if tf_ms > 0 else int(ts_ms)

                    rate_key = f"{symbol}:{side}:{str(self.signal_tf).strip().lower()}"
                    last_anchor = self._last_soft_deferral_anchor_by_key.get(rate_key)
                    if last_anchor == setup_anchor_ts_ms:
                        logger.debug(
                            "[MeanReversion] Soft deferral rate-limited for %s key=%s anchor=%s",
                            symbol,
                            rate_key,
                            setup_anchor_ts_ms,
                        )
                        return None
                    self._last_soft_deferral_anchor_by_key[rate_key] = setup_anchor_ts_ms

                    soft_router_cfg = (
                        rsi_zone_router_cfg.get("soft_deferral", {})
                        if isinstance(rsi_zone_router_cfg.get("soft_deferral"), dict)
                        else {}
                    )
                    cancel_on_zone_mismatch = bool(soft_router_cfg.get("cancel_on_zone_mismatch", True))
                    if rsi_router_enabled and cancel_on_zone_mismatch:
                        allowed_for_deferral, deferral_reason = is_strategy_allowed(
                            self.strategy_name,
                            side,
                            rsi_zone_snapshot,
                            rsi_zone_router_cfg,
                        )
                        if not allowed_for_deferral:
                            router_log_ctx = rsi_snapshot_log_context(rsi_zone_snapshot)
                            logger.info(
                                "[MeanReversion] Soft deferral cancelled by RSI router %s: reason=%s zone=%s rsi_level=%s rsi_slow=%s rsi_fast=%s ob_threshold=%s str_threshold=%s consensus_status=%s",
                                symbol,
                                str(deferral_reason or "rsi_router.deferral_cancelled"),
                                (rsi_zone_snapshot or {}).get("zone")
                                if isinstance(rsi_zone_snapshot, dict)
                                else None,
                                router_log_ctx.get("rsi_level"),
                                router_log_ctx.get("rsi_slow"),
                                router_log_ctx.get("rsi_fast"),
                                router_log_ctx.get("ob_threshold"),
                                router_log_ctx.get("str_threshold"),
                                router_log_ctx.get("consensus_status"),
                            )
                            return None

                    reason_code = "strategy.mean_reversion.near_miss"
                    trigger_price = lower if choose_lower else upper
                    expires_at_ms = None
                    if tf_ms > 0:
                        expires_at_ms = setup_anchor_ts_ms + int(tf_ms)
                    if self.fast_watch_ttl_ms:
                        ttl_exp = ts_ms + int(self.fast_watch_ttl_ms)
                        expires_at_ms = ttl_exp if expires_at_ms is None else min(expires_at_ms, ttl_exp)
                    return {
                        "event_type": "soft_deferral_event",
                        "strategy": self.strategy_name,
                        "symbol": symbol,
                        "side": side,
                        "timeframe": self.signal_tf,
                        "setup_anchor_ts_ms": setup_anchor_ts_ms,
                        "reason_code": reason_code,
                        "refresh_policy": "FAST_PRICE_WATCH",
                        "condition_data": {
                            "price": price,
                            "lower": lower,
                            "upper": upper,
                            "vwap": vwap_target,
                            "adx": adx_val,
                            "threshold": threshold,
                            "near": "lower" if choose_lower else "upper",
                            "trigger_price": trigger_price,
                            **(
                                {"trigger_sigma": float(vwap_std_val)}
                                if vwap_std_val is not None and math.isfinite(float(vwap_std_val)) and float(vwap_std_val) > 0
                                else {}
                            ),
                            "trigger_kind": "band_touch",
                            "eps_bps": self.fast_watch_eps_bps,
                            "watch_interval_ms": self.fast_watch_interval_ms,
                            "max_checks": self.fast_watch_max_checks,
                            "ttl_ms": self.fast_watch_ttl_ms,
                            "expires_at_ms": expires_at_ms,
                            "band_snapshot_ts_ms": setup_anchor_ts_ms,
                            **(
                                {
                                    "fast_watch": {
                                        "near_bps": self._fast_watch_near_bps_default,
                                        "touch_eps_bps": self._fast_watch_touch_eps_bps_default,
                                        "touch_price_source": self._fast_watch_touch_price_source_default,
                                        "recheck_freshness_ms": self._fast_watch_recheck_freshness_ms_default,
                                        "allow_touch_entry": self._fast_watch_allow_touch_entry_default,
                                    }
                                }
                                if self._fast_watch_v2_enabled
                                else {}
                            ),
                        },
                    }
            logger.info(
                f"[MeanReversion] Price within bands for {symbol}. "
                f"px={price:.4f}, lower={lower:.4f}, upper={upper:.4f}, "
                f"adx={adx_val:.1f}, adx_th={self.adx_threshold:.1f}"
            )
            return None

        if touch_entry_allowed and in_band and adx_ok:
            side_norm = str(side_hint or "").strip().lower()
            if side_norm in ("long", "buy"):
                entry_long = True
            elif side_norm in ("short", "sell"):
                entry_short = True

        if not adx_ok:
            if price > upper:
                breach = "above_upper"
            elif price < lower:
                breach = "below_lower"
            else:
                breach = "outside"
            logger.info(
                f"[MeanReversion] Price outside bands but ADX veto for {symbol}. "
                f"breach={breach}, px={price:.4f}, lower={lower:.4f}, upper={upper:.4f}, "
                f"adx={adx_val:.1f}, adx_th={eff_adx_threshold:.1f}, adx_reason={adx_decision_reason}"
            )
            _emit_recheck_eval(
                action="HOLD",
                gate_reasons=["adx_veto"],
                px=price,
                px_source=px_source,
                lower=lower,
                upper=upper,
                vwap=vwap_target,
                vwap_std=vwap_std_val,
                z=z_val,
                side_value=None,
            )
            return None

        if entry_long and reentry_guard_long_active:
            logger.info(
                "[MeanReversion] Reentry LONG guard veto %s: price %.4f <= vwap %.4f",
                symbol,
                float(price),
                float(vwap_target),
            )
            _emit_recheck_eval(
                action="HOLD",
                gate_reasons=["reentry_guard_vwap"],
                px=price,
                px_source=px_source,
                lower=lower,
                upper=upper,
                vwap=vwap_target,
                vwap_std=vwap_std_val,
                z=z_val,
                side_value="long",
            )
            return None

        if entry_short and reentry_guard_short_active:
            logger.info(
                "[MeanReversion] Reentry SHORT guard veto %s: waiting clear conditions (price=%.4f upper=%.4f z=%s)",
                symbol,
                float(price),
                float(upper) if upper is not None and math.isfinite(float(upper)) else float("nan"),
                f"{float(z_val):.3f}" if z_val is not None and math.isfinite(float(z_val)) else "nan",
            )
            _emit_recheck_eval(
                action="HOLD",
                gate_reasons=["reentry_guard_short"],
                px=price,
                px_source=px_source,
                lower=lower,
                upper=upper,
                vwap=vwap_target,
                vwap_std=vwap_std_val,
                z=z_val,
                side_value="short",
            )
            return None

        if entry_long and guard_block_long:
            _emit_recheck_eval(
                action="HOLD",
                gate_reasons=["rsi_guard_blocked"],
                px=price,
                px_source=px_source,
                lower=lower,
                upper=upper,
                vwap=vwap_target,
                vwap_std=vwap_std_val,
                z=z_val,
                side_value="long",
            )
            return None

        # -----------------------------------------------------------
        # Regime policy (TREND_DOWN / SHOCK) - long-side guardrail
        # -----------------------------------------------------------
        regime_policy_meta = None
        regime_policy_size_mult = None
        if entry_long:
            policy_cfg = self.strategy_config.get("regime_policy") if isinstance(self.strategy_config, dict) else {}
            if policy_cfg is not None and not isinstance(policy_cfg, dict):
                policy_cfg = {}
            policy_cfg = dict(policy_cfg) if isinstance(policy_cfg, dict) else {}
            policy_enabled = bool(policy_cfg.get("enabled", True))

            if policy_enabled:
                trend_cfg = policy_cfg.get("trend_down", {}) if isinstance(policy_cfg.get("trend_down"), dict) else {}
                shock_cfg = policy_cfg.get("shock", {}) if isinstance(policy_cfg.get("shock"), dict) else {}

                long_mode = str(trend_cfg.get("long_mode", "disabled") or "disabled").strip().lower()
                min_breach_bps = float(trend_cfg.get("min_breach_bps", -20) or -20)
                require_reclaim = bool(trend_cfg.get("require_reclaim", True))
                size_mult = float(trend_cfg.get("size_mult", 0.5) or 0.5)
                rising_adx_floor = float(trend_cfg.get("rising_adx_floor", 20) or 20)
                rising_adx_size_mult = float(trend_cfg.get("rising_adx_size_mult", 0.5) or 0.5)
                rising_adx_lookback = int(trend_cfg.get("rising_adx_lookback", 3) or 3)
                adx_floor = float(trend_cfg.get("adx_floor", 25) or 25)

                shock_long_mode = str(shock_cfg.get("long_mode", "disabled") or "disabled").strip().lower()
                shock_min_score = shock_cfg.get("min_score", None)
                shock_state_required = str(shock_cfg.get("state", "ARMED") or "ARMED").strip().upper()
                try:
                    shock_min_score = float(shock_min_score) if shock_min_score is not None else None
                except Exception:
                    shock_min_score = None

                trend_label = None
                trend_strength = None
                if isinstance(regime_data, dict):
                    try:
                        trend_label = str(regime_data.get("trend") or "").strip().lower()
                    except Exception:
                        trend_label = None
                    trend_strength = regime_data.get("trend_strength")
                    if trend_strength is None:
                        trend_strength = regime_data.get("adx")
                    try:
                        trend_strength = float(trend_strength) if trend_strength is not None else None
                    except Exception:
                        trend_strength = None

                shock_state_norm = None
                try:
                    shock_state_norm = str(shock_state or "").strip().upper()
                except Exception:
                    shock_state_norm = None

                is_shock = False
                if shock_state_norm:
                    is_shock = shock_state_norm == shock_state_required
                if (not is_shock) and shock_min_score is not None and shock_score is not None:
                    try:
                        is_shock = float(shock_score) >= float(shock_min_score)
                    except Exception:
                        is_shock = False

                is_trend_down = False
                if trend_label == "bearish":
                    if trend_strength is None:
                        is_trend_down = False
                    else:
                        try:
                            is_trend_down = float(trend_strength) >= float(adx_floor)
                        except Exception:
                            is_trend_down = False

                block_reason = None
                if is_shock and shock_long_mode == "disabled":
                    block_reason = "shock_disabled"
                elif is_trend_down:
                    if long_mode == "disabled":
                        block_reason = "trend_down_disabled"
                    elif long_mode == "confirmed_only":
                        breach_bps = None
                        if lower > 0:
                            try:
                                breach_ref = price
                                if require_reclaim and condition_price is not None:
                                    try:
                                        if math.isfinite(float(condition_price)):
                                            breach_ref = float(condition_price)
                                    except Exception:
                                        breach_ref = price
                                breach_bps = (float(breach_ref) - float(lower)) / float(lower) * 10000.0
                            except Exception:
                                breach_bps = None
                        reclaim_confirmed = bool(in_band)
                        if not reclaim_confirmed:
                            try:
                                last_close = float(last_sig.get("close"))
                            except Exception:
                                last_close = None
                            if last_close is not None and lower <= last_close <= upper:
                                reclaim_confirmed = True

                        breach_ok = breach_bps is not None and float(breach_bps) <= float(min_breach_bps)
                        confirm_ok = (not require_reclaim) or reclaim_confirmed
                        if not (breach_ok and confirm_ok):
                            block_reason = "trend_down_unconfirmed"
                        else:
                            regime_policy_size_mult = float(size_mult)
                            # If ADX rising in downtrend, further reduce size.
                            adx_rising = False
                            if (
                                adx_val is not None
                                and math.isfinite(adx_val)
                                and float(adx_val) >= float(rising_adx_floor)
                            ):
                                try:
                                    adx_series = pd.to_numeric(clean_sig["adx"], errors="coerce")
                                except Exception:
                                    adx_series = None
                                if adx_series is not None and len(adx_series) > max(1, int(rising_adx_lookback)):
                                    try:
                                        cur = float(adx_series.iloc[-1])
                                        prev = float(adx_series.iloc[-1 - int(rising_adx_lookback)])
                                        adx_rising = math.isfinite(cur) and math.isfinite(prev) and (cur - prev) > 0
                                    except Exception:
                                        adx_rising = False
                            if adx_rising:
                                regime_policy_size_mult *= float(rising_adx_size_mult)

                if block_reason:
                    logger.info(
                        "[MeanReversion] Regime policy veto %s: reason=%s trend=%s adx=%.2f shock_state=%s shock_score=%s",
                        symbol,
                        block_reason,
                        trend_label or "unknown",
                        float(trend_strength) if trend_strength is not None else float("nan"),
                        shock_state_norm or "NA",
                        f"{shock_score:.2f}" if shock_score is not None else "NA",
                    )
                    _emit_recheck_eval(
                        action="HOLD",
                        gate_reasons=["regime_policy_veto", block_reason],
                        px=price,
                        px_source=px_source,
                        lower=lower,
                        upper=upper,
                        vwap=vwap_target,
                        vwap_std=vwap_std_val,
                        z=z_val,
                        side_value="long",
                    )
                    return None

                regime_policy_meta = {
                    "trend": trend_label,
                    "trend_strength": float(trend_strength) if trend_strength is not None else None,
                    "shock_state": shock_state_norm,
                    "shock_score": float(shock_score) if shock_score is not None else None,
                    "trend_long_mode": long_mode,
                    "shock_long_mode": shock_long_mode,
                    "size_mult": float(regime_policy_size_mult) if regime_policy_size_mult is not None else None,
                }

        # -----------------------------------------------------------
        # Regime policy (TREND_UP / SHOCK) - short-side guardrail (opt-in)
        # -----------------------------------------------------------
        if entry_short:
            policy_cfg = self.strategy_config.get("regime_policy") if isinstance(self.strategy_config, dict) else {}
            if policy_cfg is not None and not isinstance(policy_cfg, dict):
                policy_cfg = {}
            policy_cfg = dict(policy_cfg) if isinstance(policy_cfg, dict) else {}
            policy_enabled = bool(policy_cfg.get("enabled", True))

            if policy_enabled:
                trend_up_cfg = policy_cfg.get("trend_up", {}) if isinstance(policy_cfg.get("trend_up"), dict) else {}
                shock_cfg = policy_cfg.get("shock", {}) if isinstance(policy_cfg.get("shock"), dict) else {}

                short_mode = str(trend_up_cfg.get("short_mode", "off") or "off").strip().lower()
                short_size_mult = float(trend_up_cfg.get("size_mult", 0.5) or 0.5)
                trend_up_adx_floor = float(trend_up_cfg.get("adx_floor", 25) or 25)
                extreme_z_min = float(trend_up_cfg.get("extreme_z_min", 2.5) or 2.5)

                shock_short_mode = str(shock_cfg.get("short_mode", "off") or "off").strip().lower()
                shock_min_score = shock_cfg.get("min_score", None)
                shock_state_required = str(shock_cfg.get("state", "ARMED") or "ARMED").strip().upper()
                try:
                    shock_min_score = float(shock_min_score) if shock_min_score is not None else None
                except Exception:
                    shock_min_score = None

                trend_label = None
                trend_strength = None
                if isinstance(regime_data, dict):
                    try:
                        trend_label = str(regime_data.get("trend") or "").strip().lower()
                    except Exception:
                        trend_label = None
                    trend_strength = regime_data.get("trend_strength")
                    if trend_strength is None:
                        trend_strength = regime_data.get("adx")
                    try:
                        trend_strength = float(trend_strength) if trend_strength is not None else None
                    except Exception:
                        trend_strength = None

                shock_state_norm = None
                try:
                    shock_state_norm = str(shock_state or "").strip().upper()
                except Exception:
                    shock_state_norm = None

                is_shock = False
                if shock_state_norm:
                    is_shock = shock_state_norm == shock_state_required
                if (not is_shock) and shock_min_score is not None and shock_score is not None:
                    try:
                        is_shock = float(shock_score) >= float(shock_min_score)
                    except Exception:
                        is_shock = False

                is_trend_up = False
                if trend_label == "bullish":
                    if trend_strength is None:
                        is_trend_up = False
                    else:
                        try:
                            is_trend_up = float(trend_strength) >= float(trend_up_adx_floor)
                        except Exception:
                            is_trend_up = False

                z_extreme_ok = bool(z_val is not None and math.isfinite(float(z_val)) and abs(float(z_val)) >= float(extreme_z_min))

                block_reason = None
                if is_shock and shock_short_mode == "disabled":
                    block_reason = "shock_short_disabled"
                elif is_trend_up:
                    if short_mode == "disabled":
                        block_reason = "trend_up_disabled"
                    elif short_mode in {"extreme_only", "confirmed_only"} and not z_extreme_ok:
                        block_reason = "trend_up_extreme_required"
                    elif short_mode == "size_mult":
                        regime_policy_size_mult = float(short_size_mult)

                if block_reason:
                    logger.info(
                        "[MeanReversion] Regime policy veto %s: reason=%s trend=%s adx=%.2f shock_state=%s shock_score=%s",
                        symbol,
                        block_reason,
                        trend_label or "unknown",
                        float(trend_strength) if trend_strength is not None else float("nan"),
                        shock_state_norm or "NA",
                        f"{shock_score:.2f}" if shock_score is not None else "NA",
                    )
                    _emit_recheck_eval(
                        action="HOLD",
                        gate_reasons=["regime_policy_veto", block_reason],
                        px=price,
                        px_source=px_source,
                        lower=lower,
                        upper=upper,
                        vwap=vwap_target,
                        vwap_std=vwap_std_val,
                        z=z_val,
                        side_value="short",
                    )
                    return None

                if regime_policy_meta is None:
                    regime_policy_meta = {
                        "trend": trend_label,
                        "trend_strength": float(trend_strength) if trend_strength is not None else None,
                        "shock_state": shock_state_norm,
                        "shock_score": float(shock_score) if shock_score is not None else None,
                        "size_mult": float(regime_policy_size_mult) if regime_policy_size_mult is not None else None,
                    }
                regime_policy_meta["trend_short_mode"] = short_mode
                regime_policy_meta["shock_short_mode"] = shock_short_mode
                regime_policy_meta["z_extreme_ok"] = z_extreme_ok

        if entry_long:
            side = "buy"
            reason = (
                f"VWAP MR long (px {price:.4f} < lower {lower:.4f}, "
                f"ADX {adx_val:.1f} < {eff_adx_threshold:.1f})"
            )
        elif entry_short:
            side = "sell"
            reason = (
                f"VWAP MR short (px {price:.4f} > upper {upper:.4f}, "
                f"ADX {adx_val:.1f} < {eff_adx_threshold:.1f})"
            )
        else:
            logger.info(
                f"[MeanReversion] Price within bands for {symbol}. "
                f"px={price:.4f}, lower={lower:.4f}, upper={upper:.4f}, "
                f"adx={adx_val:.1f}, adx_th={self.adx_threshold:.1f}"
            )
            _emit_recheck_eval(
                action="HOLD",
                gate_reasons=["in_band"],
                px=price,
                px_source=px_source,
                lower=lower,
                upper=upper,
                vwap=vwap_target,
                vwap_std=vwap_std_val,
                z=z_val,
                side_value=None,
            )
            return None

        # ------------------------------------------------------------------
        # Dynamic exit levels (Z-score consistent with VWAP bands)
        # Goal: as volatility tightens/expands (vwap_std), stop adapts proportionally.
        # ------------------------------------------------------------------
        stop_loss_price = None
        take_profit_price = vwap_target

        effective_vwap_std = None
        effective_band_mult = float(controller_decision.band_multiplier) if controller_decision else float(self.band_mult)
        try:
            if controller_decision is not None:
                effective_vwap_std = float(controller_decision.vwap_std)
        except Exception:
            effective_vwap_std = None
        if effective_vwap_std is None:
            try:
                if "vwap_std" in last_vwap.index:
                    effective_vwap_std = float(last_vwap["vwap_std"])
            except Exception:
                effective_vwap_std = None
        if effective_vwap_std is None:
            try:
                if (
                    math.isfinite(upper)
                    and math.isfinite(lower)
                    and math.isfinite(effective_band_mult)
                    and effective_band_mult > 0
                    and upper > lower
                ):
                    effective_vwap_std = (upper - lower) / (2.0 * effective_band_mult)
            except Exception:
                effective_vwap_std = None

        if (
            effective_vwap_std is not None
            and math.isfinite(effective_vwap_std)
            and effective_vwap_std > 0
            and self.stop_loss_std_delta is not None
            and math.isfinite(self.stop_loss_std_delta)
            and self.stop_loss_std_delta > 0
        ):
            delta = float(self.stop_loss_std_delta)
            if side == "buy":
                stop_candidate = lower - delta * float(effective_vwap_std)
                # Safety: ensure stop is below entry.
                if stop_candidate >= price:
                    stop_candidate = price - delta * float(effective_vwap_std)
                stop_loss_price = stop_candidate
            else:
                stop_candidate = upper + delta * float(effective_vwap_std)
                # Safety: ensure stop is above entry.
                if stop_candidate <= price:
                    stop_candidate = price + delta * float(effective_vwap_std)
                stop_loss_price = stop_candidate

        # Fallback: ATR-based stop if dynamic std-based stop not available.
        if (stop_loss_price is None or not math.isfinite(float(stop_loss_price))) and atr_val and not math.isnan(atr_val):
            if side == "buy":
                stop_loss_price = price - float(atr_val) * 1.5
            else:
                stop_loss_price = price + float(atr_val) * 1.5

        stop = stop_loss_price
        target = take_profit_price

        # Optional: Reject signals that are net-unprofitable vs assumed costs.
        # This addresses the "target too close" (e.g., LOW regime) issue where reward is below costs.
        reward_bps = None
        risk_bps = None
        net_reward_bps = None
        rr_net = None
        try:
            if price and target and math.isfinite(float(price)) and math.isfinite(float(target)) and float(price) > 0:
                reward_bps = abs(float(target) - float(price)) / float(price) * 10000.0
            if price and stop and math.isfinite(float(price)) and math.isfinite(float(stop)) and float(price) > 0:
                risk_bps = abs(float(price) - float(stop)) / float(price) * 10000.0
            if reward_bps is not None and math.isfinite(float(reward_bps)):
                net_reward_bps = float(reward_bps) - float(self.net_profit_cost_bps)
            if net_reward_bps is not None and risk_bps and math.isfinite(float(risk_bps)) and float(risk_bps) > 0:
                rr_net = float(net_reward_bps) / float(risk_bps)
        except Exception:
            pass

        if self.net_profit_filter_enabled and net_reward_bps is not None and math.isfinite(float(net_reward_bps)):
            if float(net_reward_bps) <= float(self.min_net_reward_bps):
                logger.info(
                    "[MeanReversion] Net-profit veto for %s: side=%s reward_bps=%.2f net_reward_bps=%.2f (cost_bps=%.2f) risk_bps=%s",
                    symbol,
                    side,
                    float(reward_bps) if reward_bps is not None else float('nan'),
                    float(net_reward_bps),
                    float(self.net_profit_cost_bps),
                    f"{float(risk_bps):.2f}" if risk_bps is not None else "nan",
                )
                _emit_recheck_eval(
                    action="HOLD",
                    gate_reasons=["net_profit_veto"],
                    px=price,
                    px_source=px_source,
                    lower=lower,
                    upper=upper,
                    vwap=vwap_target,
                    vwap_std=vwap_std_val,
                    z=z_val,
                    side_value="long" if side == "buy" else "short",
                )
                return None

        # ------------------------------------------------------------------
        # Impulse / Shock telemetry (used by IntegrityGuard)
        # ------------------------------------------------------------------
        impulse_meta = None
        try:
            imp_cfg = self.strategy_config.get("impulse_veto") if isinstance(self.strategy_config, dict) else {}
            imp_enabled = bool(imp_cfg.get("enabled", True)) if isinstance(imp_cfg, dict) else True
            body_thr = float(imp_cfg.get("body_atr_mult", 1.5) or 1.5) if isinstance(imp_cfg, dict) else 1.5
            sum2_thr = float(imp_cfg.get("sum2_range_atr_mult", 2.5) or 2.5) if isinstance(imp_cfg, dict) else 2.5

            candle_open = float(last_sig.get("open")) if "open" in last_sig else None
            candle_close = float(last_sig.get("close")) if "close" in last_sig else None
            candle_high = float(last_sig.get("high")) if "high" in last_sig else None
            candle_low = float(last_sig.get("low")) if "low" in last_sig else None

            body_size = None
            range_size = None
            if candle_open is not None and candle_close is not None:
                body_size = abs(candle_close - candle_open)
            if candle_high is not None and candle_low is not None:
                range_size = max(0.0, candle_high - candle_low)

            body_atr_mult = None
            range_atr_mult = None
            sum2_range_atr_mult = None

            if atr_val and atr_val > 0:
                if body_size is not None:
                    body_atr_mult = body_size / float(atr_val)
                if range_size is not None:
                    range_atr_mult = range_size / float(atr_val)
                try:
                    if isinstance(clean_sig, pd.DataFrame) and len(clean_sig) >= 2 and {"high", "low"}.issubset(set(clean_sig.columns)):
                        last_two = clean_sig.tail(2)
                        ranges = (last_two["high"] - last_two["low"]).astype(float)
                        sum2_range_atr_mult = float(ranges.sum()) / float(atr_val) if float(atr_val) > 0 else None
                except Exception:
                    sum2_range_atr_mult = None

            candle_dir = "up" if (candle_close is not None and candle_open is not None and candle_close > candle_open) else \
                ("down" if (candle_close is not None and candle_open is not None and candle_close < candle_open) else "flat")
            trade_dir = "up" if side == "buy" else "down"

            is_shock_move = False
            if body_atr_mult is not None and body_atr_mult >= body_thr:
                is_shock_move = True
            if sum2_range_atr_mult is not None and sum2_range_atr_mult >= sum2_thr:
                is_shock_move = True

            impulse_meta = {
                "enabled": imp_enabled,
                "is_shock_move": bool(is_shock_move),
                "body_atr_mult": body_atr_mult,
                "range_atr_mult": range_atr_mult,
                "sum2_range_atr_mult": sum2_range_atr_mult,
                "thresholds": {
                    "body_atr_mult": body_thr,
                    "sum2_range_atr_mult": sum2_thr,
                },
                "candle_dir": candle_dir,
                "trade_dir": trade_dir,
                "require_opposite": True,
            }
        except Exception:
            impulse_meta = None

        signal = {
            "strategy_name": self.strategy_name,
            "symbol": symbol,
            "side": side,
            "timeframe": self.signal_tf,
            "entry": price,
            "stop": stop,
            "target": target,
            "reward_bps": reward_bps,
            "risk_bps": risk_bps,
            "net_reward_bps": net_reward_bps,
            "rr_net": rr_net,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "reason": reason,
            "signal_type": "MEAN_REVERSION",
            "tp_mode": "DYNAMIC",
            "min_rr_ratio": self.min_rr_ratio,
            "vwap": vwap_target,
            "vwap_lower": lower,
            "vwap_upper": upper,
            "vwap_std": effective_vwap_std,
            "band_multiplier_effective": effective_band_mult,
            "stop_loss_std_delta": self.stop_loss_std_delta,
            "adx": adx_val,
            "volume_strength": volume_strength,
        }
        if volume_bucket is not None:
            signal["volume_bucket"] = volume_bucket
        if (
            regime_policy_size_mult is not None
            and math.isfinite(float(regime_policy_size_mult))
            and float(regime_policy_size_mult) > 0
        ):
            existing_mult = signal.get("position_multiplier")
            try:
                existing_mult = float(existing_mult) if existing_mult is not None else None
            except Exception:
                existing_mult = None
            final_mult = float(regime_policy_size_mult)
            if existing_mult is not None and math.isfinite(existing_mult):
                final_mult *= float(existing_mult)
            signal["position_multiplier"] = float(final_mult)
        if guard_enabled and side == "buy":
            signal["rsi_guard_status"] = guard_status or "bypassed_low_z"
            signal["rsi_val"] = guard_rsi_val
            signal["z_score_val"] = z_val

        vsa_shadow_meta = self._compute_vsa_shadow_meta(
            symbol=symbol,
            side=side,
            clean_vwap=clean_vwap if isinstance(clean_vwap, pd.DataFrame) else None,
            regime_data=regime_data if isinstance(regime_data, dict) else None,
            adx_val=adx_val,
            atr_val=atr_val,
            z_val=z_val,
            volume_analysis=volume_analysis if isinstance(volume_analysis, dict) else None,
            reward_bps=reward_bps,
            risk_bps=risk_bps,
            rejection_shadow=rejection_shadow if isinstance(rejection_shadow, dict) else None,
            timeframe_hint=self.vwap_tf,
        )

        meta_data = signal.get("meta", {})
        if not isinstance(meta_data, dict):
            meta_data = {}
        if impulse_meta:
            meta_data["impulse_guard"] = impulse_meta
        if rejection_meta:
            meta_data["rejection_confirmation"] = rejection_meta
        if regime_policy_meta:
            meta_data["regime_policy"] = regime_policy_meta
        if shock_state is not None:
            meta_data.setdefault("shock_state", shock_state)
        if shock_score is not None:
            meta_data.setdefault("shock_score", shock_score)
        if isinstance(volume_analysis, dict):
            meta_data["volume_analysis"] = volume_analysis
        if isinstance(vsa_shadow_meta, dict):
            meta_data["vsa_shadow"] = vsa_shadow_meta
        if is_recheck or bool(promote_override_meta.get("candidate")):
            meta_data["promotion_override"] = dict(promote_override_meta)
        meta_data.setdefault(
            "price_meta",
            {
                "sig_close": sig_close_price,
                "market_price": market_price,
                "market_price_source": market_price_source,
                "market_price_fallback_chain": market_price_fallback_chain,
                "price_used": price,
                "price_used_source": px_source,
            },
        )
        try:
            last_row = last_vwap if isinstance(last_vwap, pd.Series) else None
            if last_row is None and isinstance(last_sig, pd.Series):
                last_row = last_sig

            if last_row is not None:
                def _safe_float(value: Any) -> Optional[float]:
                    try:
                        if value is not None and pd.notna(value):
                            return float(value)
                    except Exception:
                        return None
                    return None

                meta_data["vol_telemetry"] = {
                    "rs_bps": _safe_float(last_row.get("vol_rs_bps")),
                    "yz_bps": _safe_float(last_row.get("vol_yz_bps")),
                    "gk_bps": _safe_float(last_row.get("vol_gk_bps")),
                    "atr_bps": _safe_float(last_row.get("vol_atr_bps")),
                    "std_bps": _safe_float(last_row.get("vol_std_bps")),
                }
        except Exception:
            pass
        signal["meta"] = meta_data
        if parent_pending_id:
            meta = signal.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                signal["meta"] = meta
            meta.setdefault("parent_pending_id", parent_pending_id)
            try:
                meta.setdefault(
                    "recheck_debug",
                    {
                        "rows_by_tf": {
                            str(self.vwap_tf): int(len(df_vwap)),
                            str(self.signal_tf): int(len(df_sig)),
                        },
                        "rows_used_after_clean": {
                            str(self.vwap_tf): int(len(clean_vwap)),
                            str(self.signal_tf): int(len(clean_sig)),
                        },
                        "vwap_clean_rows": int(len(clean_vwap)),
                        "sig_clean_rows": int(len(clean_sig)),
                        "used_fallback_fetch": bool(used_fallback_fetch),
                        "repair_reason": str(repair_reason) if repair_reason else None,
                    },
                )
            except Exception:
                pass
        if controller_decision is not None:
            signal["mr_controller"] = {
                "band_multiplier": controller_decision.band_multiplier,
                "lookback": controller_decision.lookback,
                "z": controller_decision.z,
                "abs_z": controller_decision.abs_z,
                "target_outside_pct": controller_decision.target_outside_pct,
                "current_outside_pct": controller_decision.current_outside_pct,
                "reason": controller_decision.reason,
                "updated": controller_decision.updated,
            }

        return signal

    def _maybe_warn_pipeline_indicator_mismatch(self) -> None:
        if self._pipeline_cfg_warned:
            return
        pipeline = getattr(self, "market_data_pipeline", None)
        pipeline_cfg = getattr(pipeline, "config", None)
        if not isinstance(pipeline_cfg, dict):
            return
        indicators_cfg = pipeline_cfg.get("indicators", {})
        if not isinstance(indicators_cfg, dict):
            indicators_cfg = {}

        try:
            pipeline_lookback = int(indicators_cfg.get("vwap_lookback", 1440))
        except Exception:
            pipeline_lookback = 1440
        try:
            pipeline_mult = float(indicators_cfg.get("vwap_band_multiplier", 2.0))
        except Exception:
            pipeline_mult = 2.0

        if pipeline_lookback != int(self.vwap_lookback) or not math.isclose(pipeline_mult, float(self.band_mult), rel_tol=0, abs_tol=1e-12):
            logger.warning(
                "[MeanReversion] Strategy config differs from pipeline indicators; "
                f"strategy(vwap_lookback={self.vwap_lookback}, band_multiplier={self.band_mult}) "
                f"pipeline(indicators.vwap_lookback={pipeline_lookback}, indicators.vwap_band_multiplier={pipeline_mult}). "
                "Static bands (vwap_lower/vwap_upper) come from pipeline; enable dynamic_controller or align indicators.*."
            )
        self._pipeline_cfg_warned = True

    def _maybe_apply_dynamic_controller(
        self,
        *,
        symbol: str,
        df_vwap: pd.DataFrame,
        df_sig: pd.DataFrame,
        price: float,
        vwap: float,
        vwap_std: Optional[float],
        adx: float,
        atr: Optional[float],
    ) -> Optional[MRControllerDecision]:
        if not getattr(self._mr_controller, "enabled", False):
            return None

        if df_vwap is not None:
            try:
                if not self._mr_controller.is_symbol_warmed_up(symbol):
                    self._mr_controller.hydrate_symbol_history(symbol=symbol, df_vwap=df_vwap)
            except Exception:
                pass

        ts = None
        try:
            ts_val = df_sig.index[-1]
            if isinstance(ts_val, pd.Timestamp):
                ts = ts_val.to_pydatetime()
            elif isinstance(ts_val, datetime):
                ts = ts_val
        except Exception:
            ts = None

        if ts is None:
            ts = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        is_forming = False
        try:
            is_forming = bool(getattr(df_sig, "attrs", {}).get("includes_forming")) or bool(
                getattr(df_vwap, "attrs", {}).get("includes_forming")
            )
        except Exception:
            is_forming = False

        current_15s_volume = None
        if getattr(self, "_adaptive_lite_enabled", False):
            try:
                sym = str(symbol or "").strip()
            except Exception:
                sym = ""
            variants = [sym] if sym else []
            if sym and ":" in sym:
                variants.append(sym.split(":", 1)[0])
            if sym and ":" not in sym and sym.endswith("/USDT"):
                variants.append(f"{sym}:USDT")
            agg = None
            for key in variants:
                if not key:
                    continue
                cand = self._trade_aggregators.get(key)
                if cand is not None:
                    agg = cand
                    break
            if agg is not None:
                current_15s_volume = agg.get_current_volume()

        decision = self._mr_controller.evaluate(
            symbol=symbol,
            ts=ts,
            price=price,
            vwap=vwap,
            vwap_std=vwap_std,
            adx=adx,
            atr=atr,
            current_15s_volume=current_15s_volume,
            df_vwap=df_vwap,
            is_forming_candle=is_forming,
        )
        if not math.isfinite(float(decision.lower)) or not math.isfinite(float(decision.upper)):
            if not self._controller_fallback_warned:
                logger.warning(
                    f"[MeanReversion] Dynamic controller enabled but bands unavailable; "
                    f"falling back to pipeline bands (reason={decision.reason})."
                )
                self._controller_fallback_warned = True
            return None
        return decision
