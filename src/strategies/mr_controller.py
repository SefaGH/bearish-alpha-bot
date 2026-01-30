from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MRControllerDecision:
    enabled: bool
    updated: bool
    band_multiplier: float
    lookback: int
    vwap: float
    vwap_std: float
    lower: float
    upper: float
    z: Optional[float]
    abs_z: Optional[float]
    target_outside_pct: float
    current_outside_pct: Optional[float]
    adx: Optional[float]
    atr: Optional[float]
    atr_pct: Optional[float]
    reason: str
    vol_state: Optional[str] = None


@dataclass
class _SymbolState:
    abs_z_hist: Deque[float]
    volume_hist: Deque[float]
    vwap_hist: Deque[float]
    last_update_ts: Optional[datetime]
    last_band_multiplier: float
    last_lookback: int
    last_vol_state: Optional[str]
    last_vwap_calc_key: Optional[tuple[int, int, Any]]
    last_vwap_calc_vwap: Optional[float]
    last_vwap_calc_std: Optional[float]


class DynamicMRController:
    """
    Per-strategy, per-symbol controller that derives an effective VWAP band width
    (and optionally lookback) without mutating shared pipeline configuration.
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]],
        *,
        static_band_multiplier: float,
        static_lookback: int,
    ) -> None:
        self._cfg = dict(cfg) if isinstance(cfg, dict) else {}
        self._static_band_multiplier = float(static_band_multiplier)
        self._static_lookback = int(static_lookback)

        self._enabled = bool(self._cfg.get("enabled", False))
        self._target_outside_pct = float(self._cfg.get("target_outside_pct", 0.10))
        self._abs_z_window = int(self._cfg.get("abs_z_window", 500))
        self._warmup_samples = int(self._cfg.get("warmup_samples", 50))
        self._update_interval_sec = float(self._cfg.get("update_interval_sec", 300))
        self._min_m_change = float(self._cfg.get("min_m_change", 0.05))
        self._m_min = float(self._cfg.get("m_min", 1.0))
        self._m_max = float(self._cfg.get("m_max", 2.5))
        self._log_every_update = bool(self._cfg.get("log_every_update", True))

        self._freeze_on_trend = bool(self._cfg.get("freeze_on_trend", True))
        self._adx_freeze_threshold = float(self._cfg.get("adx_freeze_threshold", 25.0))

        lookback_cfg = self._cfg.get("dynamic_lookback", {}) if isinstance(self._cfg.get("dynamic_lookback"), dict) else {}
        self._dyn_lookback_enabled = bool(lookback_cfg.get("enabled", False))
        self._lookback_static = int(lookback_cfg.get("lookback_static", self._static_lookback))
        self._lookback_min = int(lookback_cfg.get("lookback_min", 120))
        self._lookback_max = int(lookback_cfg.get("lookback_max", self._static_lookback))
        self._atr_squeeze_pct = float(lookback_cfg.get("atr_squeeze_pct", 0.0015))
        self._atr_expand_pct = float(lookback_cfg.get("atr_expand_pct", 0.0040))
        self._atr_hysteresis_pct = float(lookback_cfg.get("atr_hysteresis_pct", 0.0002))

        adaptive_cfg = self._cfg.get("adaptive_settings", {}) if isinstance(self._cfg.get("adaptive_settings"), dict) else {}
        self._adaptive_enabled = bool(adaptive_cfg.get("enabled", False))
        self._adaptive_volume_enabled = bool(adaptive_cfg.get("enable_volume_adapt", True))
        self._adaptive_slope_enabled = bool(adaptive_cfg.get("enable_slope_shift", True))

        try:
            volume_lookback = int(adaptive_cfg.get("volume_lookback", 20) or 20)
        except Exception:
            volume_lookback = 20
        self._adaptive_volume_lookback = max(1, int(volume_lookback))

        try:
            slope_lookback = int(adaptive_cfg.get("slope_lookback", 50) or 50)
        except Exception:
            slope_lookback = 50
        self._adaptive_slope_lookback = max(1, int(slope_lookback))

        try:
            volume_weight = float(adaptive_cfg.get("volume_weight", 0.5))
        except Exception:
            volume_weight = 0.5
        if not math.isfinite(volume_weight):
            volume_weight = 0.5
        self._adaptive_volume_weight = float(min(max(volume_weight, 0.0), 1.0))

        try:
            vol_mult_min = float(adaptive_cfg.get("volume_mult_min", 0.5))
        except Exception:
            vol_mult_min = 0.5
        try:
            vol_mult_max = float(adaptive_cfg.get("volume_mult_max", 2.0))
        except Exception:
            vol_mult_max = 2.0
        if not math.isfinite(vol_mult_min) or vol_mult_min <= 0:
            vol_mult_min = 0.5
        if not math.isfinite(vol_mult_max) or vol_mult_max <= 0:
            vol_mult_max = 2.0
        if vol_mult_max < vol_mult_min:
            vol_mult_max = vol_mult_min
        self._adaptive_volume_mult_min = float(vol_mult_min)
        self._adaptive_volume_mult_max = float(vol_mult_max)

        try:
            slope_shift_mult = float(adaptive_cfg.get("slope_shift_mult", 0.25))
        except Exception:
            slope_shift_mult = 0.25
        if not math.isfinite(slope_shift_mult):
            slope_shift_mult = 0.25
        self._adaptive_slope_shift_mult = float(slope_shift_mult)

        try:
            slope_shift_std_cap = float(adaptive_cfg.get("slope_shift_std_cap", 0.5))
        except Exception:
            slope_shift_std_cap = 0.5
        if not math.isfinite(slope_shift_std_cap) or slope_shift_std_cap < 0:
            slope_shift_std_cap = 0.5
        self._adaptive_slope_shift_std_cap = float(slope_shift_std_cap)

        self._state_by_symbol: Dict[str, _SymbolState] = {}

        max_lookback = max(self._lookback_static, self._lookback_min, self._lookback_max, self._static_lookback)
        if max_lookback > 1000:
            logger.warning(
                f"[MRController] High Lookback detected (L={max_lookback}). Note that L is in BARS, not minutes."
            )

    @property
    def enabled(self) -> bool:
        return self._enabled or self._adaptive_enabled

    def _get_or_create_state(self, symbol: str) -> _SymbolState:
        state = self._state_by_symbol.get(symbol)
        if state is not None:
            return state

        state = _SymbolState(
            abs_z_hist=deque(maxlen=max(self._abs_z_window, 1)),
            volume_hist=deque(maxlen=max(self._adaptive_volume_lookback, 1)),
            vwap_hist=deque(maxlen=max(self._adaptive_slope_lookback, 1)),
            last_update_ts=None,
            last_band_multiplier=float(self._static_band_multiplier),
            last_lookback=int(self._lookback_static),
            last_vol_state=None,
            last_vwap_calc_key=None,
            last_vwap_calc_vwap=None,
            last_vwap_calc_std=None,
        )
        self._state_by_symbol[symbol] = state
        return state

    def ingest_15s_bar(self, *, symbol: str, start_ts_ms: int, close: float, volume: float) -> None:
        """
        Ingest a closed 15s bar (from a lightweight trade aggregator).

        Notes:
        - This is an optional, best-effort signal booster; it does not affect core controller state unless enabled.
        - `start_ts_ms` and `close` are accepted for future extensions; currently only `volume` is used.
        """
        if not self._adaptive_enabled or not self._adaptive_volume_enabled:
            return

        try:
            vol = float(volume)
        except Exception:
            return
        if not math.isfinite(vol) or vol < 0:
            return

        state = self._get_or_create_state(str(symbol))
        state.volume_hist.append(vol)

    def _apply_adaptive_overlay(
        self,
        *,
        symbol: str,
        price: float,
        base_vwap: float,
        base_std: float,
        decision: MRControllerDecision,
        current_15s_volume: Optional[float],
    ) -> MRControllerDecision:
        if not self._adaptive_enabled:
            return decision

        state = self._get_or_create_state(str(symbol))

        if math.isfinite(base_vwap):
            state.vwap_hist.append(float(base_vwap))

        vol_multiplier = 1.0
        if self._adaptive_volume_enabled:
            cur_vol = None
            if current_15s_volume is not None:
                try:
                    cur_vol = float(current_15s_volume)
                except Exception:
                    cur_vol = None
            if cur_vol is not None and math.isfinite(cur_vol) and cur_vol >= 0 and len(state.volume_hist) > 0:
                try:
                    avg_vol = float(sum(state.volume_hist) / len(state.volume_hist))
                except Exception:
                    avg_vol = float("nan")
                if math.isfinite(avg_vol) and avg_vol > 0:
                    vol_ratio = cur_vol / avg_vol
                    if math.isfinite(vol_ratio):
                        vol_multiplier = 1.0 + (self._adaptive_volume_weight * (vol_ratio - 1.0))
                        vol_multiplier = min(max(vol_multiplier, self._adaptive_volume_mult_min), self._adaptive_volume_mult_max)

        vwap_shift = 0.0
        if self._adaptive_slope_enabled and len(state.vwap_hist) >= 2:
            try:
                delta = float(state.vwap_hist[-1]) - float(state.vwap_hist[0])
            except Exception:
                delta = float("nan")
            if math.isfinite(delta):
                vwap_shift = delta * float(self._adaptive_slope_shift_mult)
                if math.isfinite(base_std) and base_std > 0 and math.isfinite(vwap_shift):
                    cap = abs(float(base_std)) * float(self._adaptive_slope_shift_std_cap)
                    if math.isfinite(cap) and cap > 0:
                        vwap_shift = min(max(vwap_shift, -cap), cap)

        m_base = float(decision.band_multiplier)
        if not math.isfinite(m_base) or m_base <= 0:
            m_base = float(self._static_band_multiplier)

        m_final = m_base * float(vol_multiplier)
        if math.isfinite(m_final):
            m_final = min(max(m_final, float(self._m_min)), float(self._m_max))
        else:
            m_final = m_base

        vwap_final = float(base_vwap) + float(vwap_shift) if math.isfinite(base_vwap) else float("nan")
        if math.isfinite(vwap_final) and math.isfinite(base_std) and base_std > 0:
            decision.enabled = True
            decision.band_multiplier = float(m_final)
            decision.vwap = float(vwap_final)
            decision.vwap_std = float(base_std)
            decision.lower = float(vwap_final - (m_final * base_std))
            decision.upper = float(vwap_final + (m_final * base_std))
            try:
                z_out = (float(price) - float(vwap_final)) / float(base_std)
                if math.isfinite(z_out):
                    decision.z = float(z_out)
                    decision.abs_z = float(abs(z_out))
            except Exception:
                pass

        if (
            (math.isfinite(vol_multiplier) and not math.isclose(vol_multiplier, 1.0, rel_tol=0.0, abs_tol=1e-12))
            or (math.isfinite(vwap_shift) and abs(vwap_shift) > 0)
        ):
            if "adaptive" not in str(decision.reason):
                decision.reason = f"{decision.reason}+adaptive"

        return decision

    def evaluate(
        self,
        *,
        symbol: str,
        ts: datetime,
        price: float,
        vwap: float,
        vwap_std: Optional[float],
        adx: Optional[float],
        atr: Optional[float],
        current_15s_volume: Optional[float] = None,
        df_vwap: Optional[pd.DataFrame] = None,
        is_forming_candle: Optional[bool] = None,
    ) -> MRControllerDecision:
        if is_forming_candle is None and df_vwap is not None:
            try:
                attrs = getattr(df_vwap, "attrs", None)
                if isinstance(attrs, dict):
                    is_forming_candle = bool(attrs.get("includes_forming", False))
                else:
                    is_forming_candle = False
            except Exception:
                is_forming_candle = False
        if is_forming_candle is None:
            is_forming_candle = False

        if not self._enabled:
            decision = self._static_decision(
                ts=ts,
                price=price,
                vwap=vwap,
                vwap_std=vwap_std,
                adx=adx,
                atr=atr,
                reason="disabled",
            )
            std_val = float(vwap_std) if vwap_std is not None else float("nan")
            return self._apply_adaptive_overlay(
                symbol=symbol,
                price=price,
                base_vwap=float(vwap),
                base_std=std_val,
                decision=decision,
                current_15s_volume=current_15s_volume,
            )

        state = self._get_or_create_state(symbol)

        m_prev = float(state.last_band_multiplier or self._static_band_multiplier)
        lookback_prev = int(state.last_lookback or self._lookback_static)
        vol_state_prev = state.last_vol_state

        atr_pct = None
        if atr is not None and price and math.isfinite(atr) and math.isfinite(price) and price > 0:
            atr_pct = float(atr / price)

        reason = "updated"
        should_update = True
        if self._freeze_on_trend and adx is not None and math.isfinite(adx) and adx >= self._adx_freeze_threshold:
            should_update = False
            reason = "freeze_on_trend"
        elif (
            state.last_update_ts is not None
            and self._update_interval_sec > 0
            and (ts - state.last_update_ts).total_seconds() < self._update_interval_sec
        ):
            should_update = False
            reason = "update_interval"

        # Determine which lookback to use for effective vwap/std (and z-score).
        if should_update:
            lookback_eff, vol_state, atr_pct_candidate = self._compute_lookback(
                prev_state=state.last_vol_state,
                price=price,
                atr=atr,
            )
            if atr_pct_candidate is not None:
                atr_pct = atr_pct_candidate
        else:
            lookback_eff = lookback_prev
            vol_state = vol_state_prev

        vwap_eff = vwap
        std_eff = vwap_std if vwap_std is not None else float("nan")
        if self._dyn_lookback_enabled and df_vwap is not None:
            vwap_candidate, std_candidate = self._compute_vwap_and_std_cached(state, df_vwap, lookback_eff)
            if vwap_candidate is not None and std_candidate is not None:
                vwap_eff = vwap_candidate
                std_eff = std_candidate

        if not math.isfinite(std_eff) or std_eff <= 0 or not math.isfinite(vwap_eff):
            # Preserve prior state bands when effective std can't be computed.
            decision = self._decision_from_state(
                state=state,
                ts=ts,
                price=price,
                vwap=vwap,
                vwap_std=vwap_std,
                adx=adx,
                atr=atr,
                z=None,
                abs_z=None,
                updated=False,
                reason="std_unavailable",
                df_vwap=df_vwap,
            )
            std_val = float(vwap_std) if vwap_std is not None else float("nan")
            return self._apply_adaptive_overlay(
                symbol=symbol,
                price=price,
                base_vwap=float(vwap),
                base_std=std_val,
                decision=decision,
                current_15s_volume=current_15s_volume,
            )

        # Compute z/abs_z using the SAME effective vwap/std used for band computation.
        z = None
        abs_z = None
        if all(map(math.isfinite, (price, vwap_eff, std_eff))) and std_eff > 0:
            z = (price - float(vwap_eff)) / float(std_eff)
            abs_z = abs(z)
            if math.isfinite(abs_z):
                state.abs_z_hist.append(float(abs_z))

        if not should_update:
            lower = float(vwap_eff) - (m_prev * float(std_eff))
            upper = float(vwap_eff) + (m_prev * float(std_eff))
            outside_pct = self._current_outside_pct(state.abs_z_hist, m_prev)
            decision = MRControllerDecision(
                enabled=True,
                updated=False,
                band_multiplier=float(m_prev),
                lookback=int(lookback_eff),
                vwap=float(vwap_eff),
                vwap_std=float(std_eff),
                lower=float(lower),
                upper=float(upper),
                z=z,
                abs_z=abs_z,
                target_outside_pct=float(self._target_outside_pct),
                current_outside_pct=outside_pct,
                adx=adx,
                atr=atr,
                atr_pct=atr_pct,
                reason=reason,
                vol_state=str(vol_state) if vol_state is not None else None,
            )
            return self._apply_adaptive_overlay(
                symbol=symbol,
                price=price,
                base_vwap=float(vwap_eff),
                base_std=float(std_eff),
                decision=decision,
                current_15s_volume=current_15s_volume,
            )

        m_eff = self._compute_band_multiplier(state)

        lower = vwap_eff - (m_eff * std_eff)
        upper = vwap_eff + (m_eff * std_eff)
        outside_pct = self._current_outside_pct(state.abs_z_hist, m_eff)

        state.last_update_ts = ts
        state.last_band_multiplier = m_eff
        state.last_lookback = lookback_eff
        state.last_vol_state = vol_state

        decision = MRControllerDecision(
            enabled=True,
            updated=True,
            band_multiplier=m_eff,
            lookback=lookback_eff,
            vwap=float(vwap_eff),
            vwap_std=float(std_eff),
            lower=float(lower),
            upper=float(upper),
            z=z,
            abs_z=abs_z,
            target_outside_pct=float(self._target_outside_pct),
            current_outside_pct=outside_pct,
            adx=adx,
            atr=atr,
            atr_pct=atr_pct,
            reason="updated",
            vol_state=str(vol_state) if vol_state is not None else None,
        )

        if self._log_every_update:
            payload = {
                "event": "mr_controller_decision",
                "symbol": symbol,
                "ts_utc": self._to_utc_iso(ts),
                "params": {
                    "band_multiplier_prev": m_prev,
                    "band_multiplier_new": decision.band_multiplier,
                    "lookback_prev": lookback_prev,
                    "lookback_new": decision.lookback,
                    "vol_state_prev": vol_state_prev,
                    "vol_state_new": vol_state,
                    "update_interval_sec": self._update_interval_sec,
                },
                "inputs": {
                    "px": price,
                    "vwap": decision.vwap,
                    "vwap_std": decision.vwap_std,
                    "adx": adx,
                    "atr": atr,
                    "atr_pct": atr_pct,
                    "is_forming_candle": bool(is_forming_candle),
                },
                "derived": {
                    "z": decision.z,
                    "abs_z": decision.abs_z,
                    "target_outside_pct": decision.target_outside_pct,
                    "current_outside_pct": decision.current_outside_pct,
                    "achieved_outside_rate": decision.current_outside_pct,
                    "outside_rate_window_size": len(state.abs_z_hist),
                    "abs_z_hist_len": len(state.abs_z_hist),
                    "lower": decision.lower,
                    "upper": decision.upper,
                },
                "reason": decision.reason,
            }
            logger.info(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

        return self._apply_adaptive_overlay(
            symbol=symbol,
            price=price,
            base_vwap=float(vwap_eff),
            base_std=float(std_eff),
            decision=decision,
            current_15s_volume=current_15s_volume,
        )

    def _static_decision(
        self,
        *,
        ts: datetime,
        price: float,
        vwap: float,
        vwap_std: Optional[float],
        adx: Optional[float],
        atr: Optional[float],
        reason: str,
    ) -> MRControllerDecision:
        std = float(vwap_std) if vwap_std is not None else float("nan")
        if not math.isfinite(std) or std <= 0:
            std = float("nan")
        lower = float("nan")
        upper = float("nan")
        z = None
        abs_z = None
        if math.isfinite(std) and std > 0 and all(map(math.isfinite, (price, vwap))):
            lower = vwap - (self._static_band_multiplier * std)
            upper = vwap + (self._static_band_multiplier * std)
            z = (price - vwap) / std
            abs_z = abs(z)
        return MRControllerDecision(
            enabled=False,
            updated=False,
            band_multiplier=float(self._static_band_multiplier),
            lookback=int(self._static_lookback),
            vwap=float(vwap),
            vwap_std=float(std),
            lower=float(lower),
            upper=float(upper),
            z=z,
            abs_z=abs_z,
            target_outside_pct=float(self._target_outside_pct),
            current_outside_pct=None,
            adx=adx,
            atr=atr,
            atr_pct=(atr / price if atr is not None and price else None),
            reason=reason,
            vol_state=None,
        )

    def _decision_from_state(
        self,
        *,
        state: _SymbolState,
        ts: datetime,
        price: float,
        vwap: float,
        vwap_std: Optional[float],
        adx: Optional[float],
        atr: Optional[float],
        z: Optional[float],
        abs_z: Optional[float],
        updated: bool,
        reason: str,
        df_vwap: Optional[pd.DataFrame],
    ) -> MRControllerDecision:
        lookback_eff = int(state.last_lookback or self._lookback_static)
        m_eff = float(state.last_band_multiplier or self._static_band_multiplier)

        atr_pct = None
        if atr is not None and price and math.isfinite(atr) and math.isfinite(price) and price > 0:
            atr_pct = float(atr / price)

        vwap_eff = float(vwap)
        std_eff = float(vwap_std) if vwap_std is not None else float("nan")
        if self._dyn_lookback_enabled and df_vwap is not None:
            vwap_candidate, std_candidate = self._compute_vwap_and_std_cached(state, df_vwap, lookback_eff)
            if vwap_candidate is not None and std_candidate is not None:
                vwap_eff = float(vwap_candidate)
                std_eff = float(std_candidate)

        lower = float("nan")
        upper = float("nan")
        if math.isfinite(vwap_eff) and math.isfinite(std_eff) and std_eff > 0:
            lower = vwap_eff - (m_eff * std_eff)
            upper = vwap_eff + (m_eff * std_eff)

        outside_pct = self._current_outside_pct(state.abs_z_hist, m_eff)
        return MRControllerDecision(
            enabled=True,
            updated=updated,
            band_multiplier=m_eff,
            lookback=lookback_eff,
            vwap=vwap_eff,
            vwap_std=std_eff,
            lower=lower,
            upper=upper,
            z=z,
            abs_z=abs_z,
            target_outside_pct=float(self._target_outside_pct),
            current_outside_pct=outside_pct,
            adx=adx,
            atr=atr,
            atr_pct=atr_pct,
            reason=reason,
            vol_state=str(state.last_vol_state) if state.last_vol_state is not None else None,
        )

    def _compute_band_multiplier(self, state: _SymbolState) -> float:
        m_prev = float(state.last_band_multiplier or self._static_band_multiplier)

        if len(state.abs_z_hist) < max(self._warmup_samples, 1):
            return m_prev

        target = float(self._target_outside_pct)
        if not math.isfinite(target) or target <= 0 or target >= 0.5:
            target = 0.10

        q = 1.0 - target
        hist = np.asarray(list(state.abs_z_hist), dtype=float)
        if hist.size == 0:
            return m_prev

        m_raw = float(np.quantile(hist, q))
        if not math.isfinite(m_raw):
            return m_prev

        m_clamped = min(max(m_raw, self._m_min), self._m_max)
        if abs(m_clamped - m_prev) < self._min_m_change:
            return m_prev
        return float(m_clamped)

    def _compute_lookback(self, *, prev_state: Optional[str], price: float, atr: Optional[float]) -> tuple[int, Optional[str], Optional[float]]:
        if not self._dyn_lookback_enabled:
            return (int(self._lookback_static), prev_state, None)

        if atr is None or not price or not math.isfinite(atr) or not math.isfinite(price) or price <= 0:
            return (int(self._lookback_static), prev_state, None)

        atr_pct = float(atr / price)
        low = float(self._atr_squeeze_pct)
        high = float(self._atr_expand_pct)
        h = float(self._atr_hysteresis_pct)

        state = prev_state or "normal"
        if state == "squeeze":
            if atr_pct > low + h:
                state = "normal"
        elif state == "high":
            if atr_pct < high - h:
                state = "normal"
        else:
            if atr_pct < low - h:
                state = "squeeze"
            elif atr_pct > high + h:
                state = "high"

        if state == "squeeze":
            lookback = self._lookback_min
        elif state == "high":
            lookback = self._lookback_max
        else:
            lookback = self._lookback_static

        lookback = int(max(min(lookback, self._lookback_max), self._lookback_min))
        return (lookback, state, atr_pct)

    @staticmethod
    def _compute_vwap_and_std(df: pd.DataFrame, lookback: int) -> tuple[Optional[float], Optional[float]]:
        # Design Note: Using rolling_std(close) to match upstream pipeline behavior,
        # rather than deviation-from-VWAP (classic VWAP bands).
        lookback = int(max(lookback, 1))
        required = ("high", "low", "close", "volume")
        if any(col not in df.columns for col in required):
            return (None, None)

        min_periods = max(0, lookback // 2)
        tail = df.tail(lookback).loc[:, list(required)]

        try:
            high = pd.to_numeric(tail["high"], errors="coerce").to_numpy(dtype=float, copy=False)
            low = pd.to_numeric(tail["low"], errors="coerce").to_numpy(dtype=float, copy=False)
            close = pd.to_numeric(tail["close"], errors="coerce").to_numpy(dtype=float, copy=False)
            volume = pd.to_numeric(tail["volume"], errors="coerce").to_numpy(dtype=float, copy=False)
        except Exception:
            return (None, None)

        # Mirror pandas rolling behavior: skip NaNs but enforce min_periods separately per series.
        volume_isfinite = np.isfinite(volume)
        close_isfinite = np.isfinite(close)

        typical = (high + low + close) / 3.0
        vp = typical * volume
        vp_isfinite = np.isfinite(vp)

        if min_periods > 0:
            if int(np.count_nonzero(volume_isfinite)) < min_periods:
                return (None, None)
            if int(np.count_nonzero(vp_isfinite)) < min_periods:
                return (None, None)

        vol_sum = float(np.nansum(volume))
        if not math.isfinite(vol_sum) or vol_sum <= 0:
            return (None, None)

        vp_sum = float(np.nansum(vp))
        vwap = vp_sum / vol_sum

        # pandas Series.rolling(...).std() uses ddof=1; require at least max(min_periods, 2) samples.
        min_std_samples = max(min_periods, 2)
        if int(np.count_nonzero(close_isfinite)) < min_std_samples:
            return (None, None)

        std = float(np.nanstd(close, ddof=1))
        if not math.isfinite(vwap) or not math.isfinite(std):
            return (None, None)

        return (float(vwap), float(std))

    def _compute_vwap_and_std_cached(
        self,
        state: _SymbolState,
        df: pd.DataFrame,
        lookback: int,
    ) -> tuple[Optional[float], Optional[float]]:
        try:
            last = df.iloc[-1]
            close_last = pd.to_numeric(last.get("close"), errors="coerce")
            vol_last = pd.to_numeric(last.get("volume"), errors="coerce")
            close_key = float(close_last) if close_last is not None and math.isfinite(float(close_last)) else None
            vol_key = float(vol_last) if vol_last is not None and math.isfinite(float(vol_last)) else None
            # Reduce cache flapping due to microscopic floating-point noise.
            if close_key is not None:
                close_key = round(close_key, 5)
            if vol_key is not None:
                vol_key = round(vol_key, 2)
            # Cache key includes last row values so forming-candle updates invalidate the cache.
            key = (int(lookback), int(len(df)), df.index[-1], close_key, vol_key)
        except Exception:
            key = None

        if (
            key is not None
            and state.last_vwap_calc_key == key
            and state.last_vwap_calc_vwap is not None
            and state.last_vwap_calc_std is not None
        ):
            return (state.last_vwap_calc_vwap, state.last_vwap_calc_std)

        vwap_candidate, std_candidate = self._compute_vwap_and_std(df, lookback)
        if key is not None and vwap_candidate is not None and std_candidate is not None:
            state.last_vwap_calc_key = key
            state.last_vwap_calc_vwap = float(vwap_candidate)
            state.last_vwap_calc_std = float(std_candidate)
        return (vwap_candidate, std_candidate)

    @staticmethod
    def _current_outside_pct(hist: Deque[float], m_eff: float) -> Optional[float]:
        if not hist:
            return None
        try:
            m = float(m_eff)
        except Exception:
            return None
        count = sum(1 for x in hist if float(x) > m)
        return float(count / len(hist))

    @staticmethod
    def _to_utc_iso(ts: datetime) -> str:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat()
