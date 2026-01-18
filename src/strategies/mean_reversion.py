import json
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


class VWAPMeanReversion(BaseStrategy):
    """
    VWAP-band mean reversion strategy (1m VWAP, 5m signal default).
    Uses a weak-trend filter via ADX to avoid fighting strong trends.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(strategy_name="mean_reversion", config=cfg)
        self.vwap_tf = cfg.get("timeframe", "1m")
        self.signal_tf = cfg.get("signal_timeframe", "5m")
        self.band_mult = float(cfg.get("band_multiplier", 2.0))
        self.stop_loss_std_delta = float(cfg.get("stop_loss_std_delta", 0.5))
        if not math.isfinite(self.stop_loss_std_delta) or self.stop_loss_std_delta < 0:
            self.stop_loss_std_delta = 0.5
        self.vwap_lookback = int(cfg.get("vwap_lookback", 1440))
        self.adx_threshold = float(cfg.get("adx_threshold", 30))
        self.min_rr_ratio = float(cfg.get("min_rr_ratio", 1.0))
        self.soft_deferral_threshold = float(cfg.get("soft_deferral_threshold", 0.005))
        if not math.isfinite(self.soft_deferral_threshold) or self.soft_deferral_threshold < 0:
            self.soft_deferral_threshold = 0.005
        try:
            self.fast_watch_eps_bps = int(cfg.get("fast_watch_eps_bps", 10) or 10)
        except Exception:
            self.fast_watch_eps_bps = 10
        self.fast_watch_eps_bps = max(0, self.fast_watch_eps_bps)
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
        if isinstance(check_detail, dict):
            fast_watch = check_detail.get("fast_watch")
            if isinstance(fast_watch, dict):
                fast_watch_price = _coerce_float(fast_watch.get("price"))
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

        price = float(last_sig["close"])
        market_price = price
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
        px_source = "market_price"
        if is_recheck and micro_gate_watch_price is not None and math.isfinite(micro_gate_watch_price):
            price = float(micro_gate_watch_price)
            px_source = "micro_gate_watch"
        elif is_recheck and fast_watch_price is not None and math.isfinite(fast_watch_price):
            price = float(fast_watch_price)
            px_source = "fast_watch"

        z_val = None
        if vwap_std_val is not None and math.isfinite(vwap_std_val) and vwap_std_val > 0:
            try:
                z_val = (price - vwap_target) / vwap_std_val
            except Exception:
                z_val = None

        in_band = lower <= price <= upper
        adx_ok = adx_val < self.adx_threshold
        entry_long = price < lower and adx_ok
        entry_short = price > upper and adx_ok

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

        if in_band:
            if parent_pending_id:
                logger.info(
                    "[MeanReversion] Recheck context; skipping soft deferral near-miss for %s (parent_pending_id=%s)",
                    symbol,
                    parent_pending_id,
                )
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                eps_bps_val = _coerce_float(condition_data.get("eps_bps"))
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
                            "trigger_kind": "band_touch",
                            "eps_bps": self.fast_watch_eps_bps,
                            "watch_interval_ms": self.fast_watch_interval_ms,
                            "max_checks": self.fast_watch_max_checks,
                            "ttl_ms": self.fast_watch_ttl_ms,
                            "expires_at_ms": expires_at_ms,
                            "band_snapshot_ts_ms": setup_anchor_ts_ms,
                        },
                    }
            logger.info(
                f"[MeanReversion] Price within bands for {symbol}. "
                f"px={price:.4f}, lower={lower:.4f}, upper={upper:.4f}, "
                f"adx={adx_val:.1f}, adx_th={self.adx_threshold:.1f}"
            )
            return None

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
                f"adx={adx_val:.1f}, adx_th={self.adx_threshold:.1f}"
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
                f"ADX {adx_val:.1f} < {self.adx_threshold})"
            )
        elif entry_short:
            side = "sell"
            reason = (
                f"VWAP MR short (px {price:.4f} > upper {upper:.4f}, "
                f"ADX {adx_val:.1f} < {self.adx_threshold})"
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

        signal = {
            "strategy_name": self.strategy_name,
            "symbol": symbol,
            "side": side,
            "timeframe": self.signal_tf,
            "entry": price,
            "stop": stop,
            "target": target,
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

        decision = self._mr_controller.evaluate(
            symbol=symbol,
            ts=ts,
            price=price,
            vwap=vwap,
            vwap_std=vwap_std,
            adx=adx,
            atr=atr,
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
