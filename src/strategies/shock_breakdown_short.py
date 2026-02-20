from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import pandas as pd

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ShockBreakdownShortStrategy(BaseStrategy):
    """Shock-aware continuation short strategy.

    Intended as a separate path from MR/OB/STR:
    - Trigger only during shock states (default: ARMED)
    - Requires structural breakdown confirmation
    - Requires minimum downside momentum and relative volume
    - Short-only and cooldown-limited by symbol
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        strategy_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        super().__init__(strategy_name="shock_breakdown_short", config=strategy_cfg)
        self.cfg = strategy_cfg
        self._last_signal_ts_ms_by_symbol: Dict[str, int] = {}
        self._last_observe_log_ts_by_symbol: Dict[str, float] = {}

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except Exception:
            return None
        if not math.isfinite(parsed):
            return None
        return float(parsed)

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            parsed = int(float(value))
        except Exception:
            return None
        return int(parsed)

    @staticmethod
    def _norm_token(value: Any) -> str:
        try:
            return str(value or "").strip().lower()
        except Exception:
            return ""

    @classmethod
    def _extract_tokens(cls, raw: Any) -> Set[str]:
        if raw is None:
            return set()
        parts = []
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.split(",")]
        elif isinstance(raw, (list, tuple, set)):
            parts = [str(item).strip() for item in raw]
        out = {cls._norm_token(part) for part in parts if cls._norm_token(part)}
        return out

    @staticmethod
    def _pick_decision_df(
        *,
        timeframe: str,
        market_data: Optional[Dict[str, Any]],
        df_30m: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        tf = str(timeframe or "").strip().lower()
        if isinstance(market_data, dict):
            candidate = market_data.get(tf)
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                return candidate
        if tf == "30m" and isinstance(df_30m, pd.DataFrame) and not df_30m.empty:
            return df_30m
        if isinstance(market_data, dict):
            for fallback_tf in ("5m", "15m", "30m", "1h"):
                candidate = market_data.get(fallback_tf)
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    return candidate
        if isinstance(df_30m, pd.DataFrame) and not df_30m.empty:
            return df_30m
        return None

    @staticmethod
    def _extract_bar_ts_ms(df: pd.DataFrame) -> int:
        try:
            ts_val = df.index[-1]
            if isinstance(ts_val, pd.Timestamp):
                dt = ts_val.to_pydatetime()
            elif isinstance(ts_val, datetime):
                dt = ts_val
            else:
                raise ValueError("unsupported index ts type")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return int(time.time() * 1000)

    def _resolve_rollout_mode(self, symbol: str) -> str:
        rollout_cfg = self.cfg.get("rollout", {}) if isinstance(self.cfg.get("rollout"), dict) else {}
        mode_raw = self._norm_token(rollout_cfg.get("mode", "observe"))
        if mode_raw in {"disabled", "off"}:
            return "off"
        mode = mode_raw if mode_raw in {"observe", "enforce"} else "observe"

        canary_symbols = self._extract_tokens(rollout_cfg.get("canary_symbols"))
        if not canary_symbols or "*" in canary_symbols:
            return mode
        symbol_norm = self._norm_token(symbol)
        if symbol_norm and symbol_norm in canary_symbols:
            return mode
        return "off"

    def _extract_shock_inputs(
        self,
        *,
        market_data: Optional[Dict[str, Any]],
        shock_state: Optional[str],
        shock_score: Optional[float],
    ) -> Dict[str, Optional[Any]]:
        snapshot = market_data.get("shock") if isinstance(market_data, dict) else None
        shock_payload = snapshot if isinstance(snapshot, dict) else {}

        state = shock_state if shock_state is not None else shock_payload.get("state")
        score = self._coerce_float(shock_score)
        if score is None:
            score = self._coerce_float(shock_payload.get("shock_score"))

        return {
            "state": state,
            "score": score,
            "payload": shock_payload,
        }

    def _meets_shock_gate(
        self,
        *,
        state: Optional[str],
        score: Optional[float],
    ) -> bool:
        required_states = self._extract_tokens(self.cfg.get("shock_states"))
        if not required_states:
            required_states = self._extract_tokens(self.cfg.get("shock_state", "ARMED"))
        if not required_states:
            required_states = {"armed"}
        if "*" not in required_states and "any" not in required_states:
            state_norm = self._norm_token(state)
            if not state_norm or state_norm not in required_states:
                return False

        min_score = self._coerce_float(self.cfg.get("min_shock_score", 0.60))
        min_score = 0.60 if min_score is None else max(0.0, min(1.0, float(min_score)))
        if score is None or float(score) < float(min_score):
            return False
        return True

    def _meets_breakdown_momentum_volume(
        self,
        *,
        df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        lookback = self._coerce_int(self.cfg.get("breakdown_lookback_bars", 20))
        lookback = 20 if lookback is None else max(2, int(lookback))

        momentum_lb = self._coerce_int(self.cfg.get("momentum_lookback_bars", 3))
        momentum_lb = 3 if momentum_lb is None else max(1, int(momentum_lb))

        vol_window = self._coerce_int(self.cfg.get("volume_ma_window", 20))
        vol_window = 20 if vol_window is None else max(2, int(vol_window))

        min_rows = max(lookback + 1, momentum_lb + 1, vol_window + 1, 25)
        if not isinstance(df, pd.DataFrame) or len(df) < min_rows:
            return None

        required_cols = {"close", "low", "volume"}
        if not required_cols.issubset(set(df.columns)):
            return None

        close = pd.to_numeric(df["close"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")
        if close.isna().iloc[-1] or low.isna().iloc[-1] or volume.isna().iloc[-1]:
            return None

        last_close = self._coerce_float(close.iloc[-1])
        if last_close is None or last_close <= 0:
            return None

        support_series = low.iloc[-(lookback + 1):-1]
        if support_series.empty:
            return None
        support_level = self._coerce_float(support_series.min())
        if support_level is None or support_level <= 0:
            return None

        confirm_bps = self._coerce_float(self.cfg.get("breakdown_confirm_bps", 5.0))
        confirm_bps = 5.0 if confirm_bps is None else max(0.0, float(confirm_bps))
        breakdown_margin = float(confirm_bps) / 10_000.0
        breakdown_ok = float(last_close) < float(support_level) * (1.0 - breakdown_margin)
        if not breakdown_ok:
            return None

        momentum_base = self._coerce_float(close.iloc[-1 - momentum_lb])
        if momentum_base is None or momentum_base <= 0:
            return None
        downside_momentum_pct = (float(momentum_base) - float(last_close)) / float(momentum_base)
        min_momentum_pct = self._coerce_float(self.cfg.get("min_momentum_pct", 0.003))
        min_momentum_pct = 0.003 if min_momentum_pct is None else max(0.0, float(min_momentum_pct))
        if float(downside_momentum_pct) < float(min_momentum_pct):
            return None

        baseline_volume = self._coerce_float(volume.iloc[-1 - vol_window:-1].mean())
        last_volume = self._coerce_float(volume.iloc[-1])
        if baseline_volume is None or baseline_volume <= 0 or last_volume is None:
            return None
        volume_mult = float(last_volume) / float(baseline_volume)
        min_volume_mult = self._coerce_float(self.cfg.get("min_volume_mult", 1.20))
        min_volume_mult = 1.20 if min_volume_mult is None else max(0.0, float(min_volume_mult))
        if float(volume_mult) < float(min_volume_mult):
            return None

        confidence = max(
            0.05,
            min(
                0.99,
                (downside_momentum_pct / max(min_momentum_pct, 1e-6)) * 0.35
                + (volume_mult / max(min_volume_mult, 1e-6)) * 0.25
                + 0.40,
            ),
        )

        return {
            "last_close": float(last_close),
            "support_level": float(support_level),
            "downside_momentum_pct": float(downside_momentum_pct),
            "volume_mult": float(volume_mult),
            "confidence": float(confidence),
        }

    def signal(
        self,
        df_30m: Optional[pd.DataFrame],
        df_1h: Optional[pd.DataFrame] = None,
        regime_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
        ml_context: Optional[Dict[str, Any]] = None,
        shock_state: Optional[str] = None,
        shock_score: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        del df_1h, ml_context, kwargs  # unused in this strategy revision
        if not bool(self.cfg.get("enabled", False)):
            return None

        symbol_name = str(symbol or "").strip()
        if not symbol_name:
            return None

        rollout_mode = self._resolve_rollout_mode(symbol_name)
        if rollout_mode == "off":
            return None

        decision_tf = str(self.cfg.get("timeframe", "5m") or "5m").strip().lower()
        decision_df = self._pick_decision_df(
            timeframe=decision_tf,
            market_data=market_data if isinstance(market_data, dict) else None,
            df_30m=df_30m if isinstance(df_30m, pd.DataFrame) else None,
        )
        if decision_df is None:
            return None

        now_ts_ms = self._extract_bar_ts_ms(decision_df)
        cooldown_seconds = self._coerce_int(self.cfg.get("cooldown_seconds", 600))
        cooldown_seconds = 600 if cooldown_seconds is None else max(0, int(cooldown_seconds))
        if cooldown_seconds > 0:
            previous_ts = self._last_signal_ts_ms_by_symbol.get(symbol_name)
            if previous_ts is not None:
                if now_ts_ms - int(previous_ts) < int(cooldown_seconds * 1000):
                    return None

        shock_inputs = self._extract_shock_inputs(
            market_data=market_data if isinstance(market_data, dict) else None,
            shock_state=shock_state,
            shock_score=shock_score,
        )
        if not self._meets_shock_gate(state=shock_inputs.get("state"), score=shock_inputs.get("score")):
            return None

        decision = self._meets_breakdown_momentum_volume(df=decision_df)
        if not isinstance(decision, dict):
            return None

        stop_loss_pct = self._coerce_float(self.cfg.get("stop_loss_pct", 0.006))
        take_profit_pct = self._coerce_float(self.cfg.get("take_profit_pct", 0.010))
        stop_loss_pct = 0.006 if stop_loss_pct is None else max(0.0005, float(stop_loss_pct))
        take_profit_pct = 0.010 if take_profit_pct is None else max(0.0005, float(take_profit_pct))
        max_hold_seconds = self._coerce_int(
            (
                self.cfg.get("exit_settings", {}).get("max_hold_seconds")
                if isinstance(self.cfg.get("exit_settings"), dict)
                else self.cfg.get("max_hold_seconds", 900)
            )
        )
        max_hold_seconds = 900 if max_hold_seconds is None else max(60, int(max_hold_seconds))

        observe_payload = {
            "event": "shock_breakdown_short_observe",
            "symbol": symbol_name,
            "timeframe": decision_tf,
            "shock_state": str(shock_inputs.get("state") or "").upper() or None,
            "shock_score": shock_inputs.get("score"),
            "support_level": decision.get("support_level"),
            "last_close": decision.get("last_close"),
            "downside_momentum_pct": decision.get("downside_momentum_pct"),
            "volume_mult": decision.get("volume_mult"),
            "rollout_mode": rollout_mode,
        }

        if rollout_mode == "observe":
            now_s = time.time()
            last_logged = self._last_observe_log_ts_by_symbol.get(symbol_name, 0.0)
            if now_s - last_logged >= 60.0:
                logger.info("%s", observe_payload)
                self._last_observe_log_ts_by_symbol[symbol_name] = now_s
            return None

        self._last_signal_ts_ms_by_symbol[symbol_name] = int(now_ts_ms)
        shock_state_out = str(shock_inputs.get("state") or "").upper() or None
        reason = (
            f"Shock breakdown short ({decision_tf}) | state={shock_state_out or 'NA'} "
            f"score={float(shock_inputs.get('score') or 0.0):.2f} "
            f"mom={float(decision.get('downside_momentum_pct') or 0.0):.4f} "
            f"vol_mult={float(decision.get('volume_mult') or 0.0):.2f}"
        )

        regime_trend = None
        if isinstance(regime_data, dict):
            try:
                regime_trend = str(regime_data.get("trend") or "").strip().lower() or None
            except Exception:
                regime_trend = None

        return {
            "strategy_name": self.strategy_name,
            "strategy": self.strategy_name,
            "symbol": symbol_name,
            "side": "sell",
            "timeframe": decision_tf,
            "reason": reason,
            "reason_code": "strategy.shock_breakdown_short.entry",
            "tp_pct": float(take_profit_pct),
            "sl_pct": float(stop_loss_pct),
            "confidence": float(decision.get("confidence") or 0.5),
            "strategy_type": "shock_breakdown",
            "exit_settings": {"max_hold_seconds": int(max_hold_seconds)},
            "meta": {
                "shock_state": shock_state_out,
                "shock_score": shock_inputs.get("score"),
                "regime_trend": regime_trend,
                "breakdown_support": decision.get("support_level"),
                "breakdown_close": decision.get("last_close"),
                "downside_momentum_pct": decision.get("downside_momentum_pct"),
                "volume_mult": decision.get("volume_mult"),
                "rollout_mode": rollout_mode,
            },
        }

    async def generate_signal(self, symbol: str, ml_context: Optional[Dict] = None) -> Optional[Dict]:
        """BaseStrategy compatibility wrapper.

        Production flow calls `signal(...)` directly via ProductionCoordinator.
        This method exists to satisfy the abstract strategy interface.
        """
        if not self.market_data_pipeline:
            return None

        decision_tf = str(self.cfg.get("timeframe", "5m") or "5m").strip().lower()
        decision_df = None
        try:
            getter = getattr(self.market_data_pipeline, "get_latest_ohlcv", None)
            if callable(getter):
                decision_df = await getter(symbol, decision_tf, limit=300, include_forming=False)
            else:
                legacy_getter = getattr(self.market_data_pipeline, "get_ohlcv", None)
                if callable(legacy_getter):
                    decision_df = await legacy_getter(symbol, decision_tf)
        except Exception:
            decision_df = None

        market_data = {decision_tf: decision_df} if isinstance(decision_df, pd.DataFrame) else {}
        df_30m = decision_df if decision_tf == "30m" else None
        return self.signal(
            df_30m=df_30m,
            symbol=symbol,
            market_data=market_data,
            ml_context=ml_context,
        )
