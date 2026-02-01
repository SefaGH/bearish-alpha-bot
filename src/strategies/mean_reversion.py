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
        self._fast_watch_v2_cfg = mr_fast_watch_cfg
        v2_keys = {"near_bps", "touch_eps_bps", "touch_price_source", "recheck_freshness_ms", "allow_touch_entry"}
        self._fast_watch_v2_enabled = any(k in mr_fast_watch_cfg for k in v2_keys)

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
        if isinstance(check_detail, dict):
            fast_watch = check_detail.get("fast_watch")
            if isinstance(fast_watch, dict):
                fast_watch_price = _coerce_float(fast_watch.get("price"))
                fast_watch_touch_confirmed = bool(fast_watch.get("touch_confirmed", False))
                fast_watch_touch_eps_bps = _coerce_float(fast_watch.get("touch_eps_bps"))
                fast_watch_px_used = _coerce_float(fast_watch.get("px_used"))
            micro_gate_watch = check_detail.get("micro_gate_watch")
            if isinstance(micro_gate_watch, dict):
                micro_gate_watch_price = _coerce_float(micro_gate_watch.get("price"))

        market_price = None
        recheck_eval_emitted = False

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
            }
            try:
                logger.info("mr_recheck_eval %s", json.dumps(out, ensure_ascii=True, sort_keys=True))
            except Exception:
                logger.info("mr_recheck_eval %s", out)
            return out

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

        # -----------------------------------------------------------
        # OPPORTUNITY PROMOTION (Recheck override)
        # -----------------------------------------------------------
        promotion_override = False
        if is_recheck and fast_watch_touch_confirmed and (near_str in ["lower", "upper"]):
            _target_band_val = lower if near_str == "lower" else upper
            _dist_bps = None
            if price and _target_band_val and price > 0:
                try:
                    _dist_bps = abs(price - _target_band_val) / price * 10000.0
                except Exception:
                    _dist_bps = None

            if (
                _dist_bps is not None
                and _dist_bps <= 6.0
                and z_val is not None
                and math.isfinite(z_val)
                and abs(float(z_val)) >= 1.23
                and adx_val is not None
                and math.isfinite(adx_val)
                and float(adx_val) <= 25.0
            ):
                promotion_override = True
                logger.info(
                    "[MeanReversion] PROMOTE override: near=%s z=%.2f adx=%.2f dist_bps=%.2f",
                    near_str,
                    float(z_val),
                    float(adx_val),
                    float(_dist_bps),
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

        if z_val is not None and math.isfinite(z_val):
            if guard_state_pre != "ARMED" and abs(float(z_val)) < required_z:
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
        if promotion_override:
            adx_ok = True
            adx_decision_reason = "promotion_override"

        entry_long = price < lower and adx_ok
        entry_short = price > upper and adx_ok
        if promotion_override:
            if near_str == "lower":
                entry_long = True
            elif near_str == "upper":
                entry_short = True

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
        if not is_recheck:
            rej_cfg = self.strategy_config.get("rejection_confirmation") if isinstance(self.strategy_config, dict) else {}
            rej_enabled = True
            wick_ratio_min = 0.8
            try:
                rej_enabled = bool(rej_cfg.get("enabled", True)) if isinstance(rej_cfg, dict) else True
            except Exception:
                rej_enabled = True
            try:
                wick_ratio_min = float(rej_cfg.get("upper_wick_ratio_min", 0.8) or 0.8) if isinstance(rej_cfg, dict) else 0.8
            except Exception:
                wick_ratio_min = 0.8

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

                if candle_open is not None and candle_close is not None and candle_high is not None:
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
                    }

                    # Allow rejection-entry even if price is back inside band (wick touch).
                    if not entry_short and touched_upper and has_red and rejected_from_band and adx_ok:
                        entry_short = True
                        rejection_meta["forced_entry"] = True

                    # If a short entry is still active, enforce rejection confirmation.
                    if entry_short and not (has_red and rejected_from_band):
                        logger.info(
                            "[MeanReversion] Rejection confirmation failed for %s: has_red=%s close_in_band=%s "
                            "upper_wick_ratio=%.2f thr=%.2f",
                            symbol,
                            has_red,
                            close_back_inside,
                            upper_wick_ratio,
                            wick_ratio_min,
                        )
                        return None

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
        }
        if guard_enabled and side == "buy":
            signal["rsi_guard_status"] = guard_status or "bypassed_low_z"
            signal["rsi_val"] = guard_rsi_val
            signal["z_score_val"] = z_val

        meta_data = signal.get("meta", {})
        if not isinstance(meta_data, dict):
            meta_data = {}
        if impulse_meta:
            meta_data["impulse_guard"] = impulse_meta
        if rejection_meta:
            meta_data["rejection_confirmation"] = rejection_meta
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
