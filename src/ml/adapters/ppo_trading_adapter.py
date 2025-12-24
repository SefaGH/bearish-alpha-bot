"""PPO adapter used by StrategyCoordinator to soft-filter BTC/USDT longs.

The adapter loads a Stable-Baselines3 PPO policy, rebuilds the RLTradingEnvGym
state vector by leveraging the shared MarketDataPipeline + FeatureEngineering
pipeline, and exposes a single async method that returns a score in [0, 1].

Design goals:
- Never crash the live loop; fall back to a neutral score when the PPO bundle or
  SB3 dependency is missing.
- Keep all heavy imports scoped to this module so PPO stays optional.
- Provide lightweight telemetry metadata for upstream logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import hashlib
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List
import gymnasium as gym

import numpy as np
import pandas as pd  # extra feature hesapları için

from src.ml.ppo.observation_spec import (
    DEFAULT_EXTRA_FEATURE_NAMES,
    ObservationSpec,
    build_observation,
    compute_price_extras,
    load_spec as load_obs_spec,
    spec_from_feature_columns,
)
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

logger = logging.getLogger(__name__)

try:  # Stable-Baselines3 is optional in live mode.
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore


@dataclass
class PPOAdapterConfig:
    """Strongly-typed view for the PPO adapter configuration."""

    enabled: bool = False
    symbols: Tuple[str, ...] = ("BTC/USDT:USDT",)
    timeframe: str = "1h"
    model_path: Path = Path("artifacts/ppo/ppo_trading_agent.zip")
    fallback_score: float = 0.5
    rr_up_mult: float = 1.3
    rr_down_mult: float = 0.9
    position_base: float = 0.5
    position_bonus: float = 0.5
    lookback_bars: int = 240
    lookback_windows: Tuple[int, ...] = (12, 24, 48, 96)
    conf_threshold: float = 0.60
    min_margin: float = 0.0
    health_min_std: float = 1e-3
    health_window: int = 30
    health_clip_frac_limit: float = 1.0
    require_vecnorm: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PPOAdapterConfig":
        rl_cfg = cfg or {}
        symbols_cfg = rl_cfg.get("ppo_symbols", ["BTC/USDT:USDT"])
        if isinstance(symbols_cfg, str):
            s = symbols_cfg.strip()
            if s.startswith("[") and s.endswith("]"):
                import json

                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        symbols_cfg = parsed
                    else:
                        symbols_cfg = [str(parsed)]
                except Exception:
                    inner = s[1:-1]
                    parts = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
                    symbols_cfg = parts or ["BTC/USDT:USDT"]
            else:
                symbols_cfg = [s]

        symbols = tuple(str(s).strip() for s in symbols_cfg if s)

        def _parse_window_sequence(raw_value: Any, fallback: Iterable[int]) -> Tuple[int, ...]:
            if raw_value is None:
                return tuple(int(v) for v in fallback)
            sequence: Iterable[Any]
            if isinstance(raw_value, (list, tuple)):
                sequence = raw_value
            else:
                raw_str = str(raw_value).strip()
                if raw_str.startswith("[") and raw_str.endswith("]"):
                    try:
                        parsed = json.loads(raw_str)
                        if isinstance(parsed, list):
                            sequence = parsed
                        else:
                            sequence = [parsed]
                    except Exception:
                        inner = raw_str[1:-1]
                        sequence = [part.strip() for part in inner.split(",") if part.strip()]
                else:
                    sequence = [part.strip() for part in raw_str.split(",") if part.strip()]

            parsed_values = []
            for value in sequence:
                try:
                    num = int(float(value))
                    if num > 0:
                        parsed_values.append(num)
                except (TypeError, ValueError):
                    continue
            return tuple(parsed_values) or tuple(int(v) for v in fallback)

        lookback_windows = _parse_window_sequence(
            rl_cfg.get("ppo_lookback_windows"),
            fallback=(12, 24, 48, 96),
        )
        lookback_bars = int(rl_cfg.get("ppo_lookback_bars", 240) or 240)
        if lookback_bars <= 0:
            lookback_bars = 240
        health_window = int(rl_cfg.get("ppo_health_window", 30) or 30)
        if health_window <= 1:
            health_window = 30

        return cls(
            enabled=bool(rl_cfg.get("ppo_enabled", False)),
            symbols=symbols or ("BTC/USDT:USDT",),
            timeframe=str(rl_cfg.get("ppo_timeframe", "1h")),
            model_path=Path(rl_cfg.get("ppo_model_path", "artifacts/ppo/ppo_trading_agent.zip")),
            fallback_score=float(rl_cfg.get("ppo_fallback_score", 0.5)),
            rr_up_mult=float(rl_cfg.get("ppo_rr_up_mult", 1.3)),
            rr_down_mult=float(rl_cfg.get("ppo_rr_down_mult", 0.9)),
            position_base=float(rl_cfg.get("ppo_position_base", 0.5)),
            position_bonus=float(rl_cfg.get("ppo_position_bonus", 0.5)),
            lookback_bars=lookback_bars,
            lookback_windows=lookback_windows,
            conf_threshold=float(rl_cfg.get("ppo_conf_threshold", 0.60)),
            min_margin=float(rl_cfg.get("ppo_min_margin", 0.0)),
            health_min_std=float(rl_cfg.get("ppo_health_min_std", 1e-3)),
            health_window=health_window,
            health_clip_frac_limit=float(rl_cfg.get("ppo_health_clip_frac_limit", 0.3)),
            require_vecnorm=bool(rl_cfg.get("ppo_require_vecnorm", True)),
        )


class PPOTradingAdapter:
    """Thin wrapper around a SB3 PPO model for long-signal modulation."""

    STATE_TIMEFRAME = "1h"
    EXTRA_FEATURE_NAMES = list(DEFAULT_EXTRA_FEATURE_NAMES)
    TAIL_DIM = 2

    def __init__(
        self,
        rl_config: Optional[Dict[str, Any]] = None,
        *,
        market_data_pipeline: Any,
        feature_pipeline: Any,
    ) -> None:
        self.cfg = PPOAdapterConfig.from_dict(rl_config or {})
        self.market_data_pipeline = market_data_pipeline
        self.feature_pipeline = feature_pipeline
        self._model: Optional[Any] = None
        self._model_lock = asyncio.Lock()
        self._load_error: Optional[str] = None
        self._tail_defaults = np.array([0.0, 1.0], dtype=np.float32)
        self._symbol_alias_map: Dict[str, str] = {}
        self._normalized_symbols = set()
        self._expected_obs_dim: Optional[int] = None  # PPO modelinin beklediği observation dim
        self._last_state_metadata: Dict[str, Any] = {}
        self._spec: Optional[ObservationSpec] = None
        self._vecnorm: Optional[VecNormalize] = None
        self._p_long_history: List[float] = []
        self._clip_history: List[float] = []

        for raw_symbol in self.cfg.symbols:
            normalized = self._normalize_symbol(raw_symbol)
            if not normalized:
                continue
            canonical = str(raw_symbol).strip().upper()
            self._symbol_alias_map.setdefault(normalized, canonical)
        self._normalized_symbols = set(self._symbol_alias_map.keys())

        logger.info(
            "✅ [PPO-INIT] enabled=%s | cfg.symbols=%s | normalized=%s",
            self.cfg.enabled,
            list(self.cfg.symbols),
            list(self._normalized_symbols),
        )

        if self.cfg.timeframe.lower() != self.STATE_TIMEFRAME:
            logger.warning(
                "PPO adapter forcing timeframe to %s (config requested %s)",
                self.STATE_TIMEFRAME,
                self.cfg.timeframe,
            )
            self.cfg.timeframe = self.STATE_TIMEFRAME

    @property
    def is_ready(self) -> bool:
        return bool(self._model and PPO)

    @property
    def multipliers(self) -> Dict[str, float]:
        return {
            "rr_up_mult": self.cfg.rr_up_mult,
            "rr_down_mult": self.cfg.rr_down_mult,
            "position_base": self.cfg.position_base,
            "position_bonus": self.cfg.position_bonus,
        }

    @staticmethod
    def _normalize_symbol(symbol: Optional[str]) -> str:
        if not symbol:
            return ""
        normalized = str(symbol).strip().upper().replace('-', '/')
        if ':' in normalized:
            normalized = normalized.split(':', 1)[0]
        return normalized

    def _is_symbol_supported(self, symbol: str) -> bool:
        sym = self._normalize_symbol(symbol)
        return sym in self._normalized_symbols

    async def get_long_score(
        self,
        symbol: str,
        *,
        position_fraction: Optional[float] = None,
        normalized_pv: Optional[float] = None,
        override_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Return PPO score for long/flat decision (+ metadata).
        Sniper mode: require high-confidence BUYs before surfacing LONG.
        """
        symbol_norm = self._normalize_symbol(symbol)
        if not self.cfg.enabled:
            return self.cfg.fallback_score, {
                "reason": "disabled",
                "symbol": symbol,
                "normalized_symbol": symbol_norm,
            }
        if not self._is_symbol_supported(symbol_norm):
            return self.cfg.fallback_score, {
                "reason": "unsupported_symbol",
                "symbol": symbol,
                "normalized_symbol": symbol_norm,
                "supported_normalized": sorted(self._normalized_symbols),
                "raw_cfg_symbols": list(self.cfg.symbols),
            }

        await self._ensure_model_loaded()
        if not self._model:
            return self.cfg.fallback_score, {"reason": self._load_error or "model_unavailable"}

        state, state_meta = await self._build_state(
            symbol_norm,
            position_fraction=position_fraction,
            normalized_pv=normalized_pv,
            override_meta=override_meta,
        )
        if state is None:
            return self.cfg.fallback_score, {"reason": "missing_state", "state_meta": state_meta}

        try:
            try:
                import torch as th  # type: ignore
            except Exception:
                th = None

            confidence = 1.0
            p_flat = None
            p_long = None
            entropy_raw = None
            p_margin = None
            if hasattr(self._model, "policy") and th is not None:
                with th.no_grad():
                    obs_tensor = th.as_tensor(state[np.newaxis, :], device=self._model.device)
                    distribution = self._model.policy.get_distribution(obs_tensor)
                    probs = distribution.distribution.probs.cpu().numpy()[0]
                    p_flat = float(probs[0]) if probs.size > 0 else None
                    p_long = float(probs[1]) if probs.size > 1 else None
                    action_int = int(np.argmax(probs))
                    confidence = float(probs[action_int])
                    entropy_raw = float(distribution.distribution.entropy().mean().item())
                    if p_flat is not None and p_long is not None:
                        p_margin = p_long - p_flat
            else:
                action, _ = self._model.predict(state[np.newaxis, :], deterministic=True)
                action_int = int(np.asarray(action).item())

            CONFIDENCE_THRESHOLD = self.cfg.conf_threshold
            meets_conf = confidence >= CONFIDENCE_THRESHOLD
            meets_margin = True if p_margin is None else (p_margin >= self.cfg.min_margin)

            # Use raw long probability as the score for all actions.
            final_score = float(p_long) if p_long is not None else self.cfg.fallback_score
            decision = "FLAT"
            if action_int == 1:
                decision = "LONG" if meets_conf and meets_margin else "WEAK_LONG_IGNORED"
            else:
                decision = "FLAT"

            metadata = {
                "symbol": symbol_norm,
                "action": decision,
                "raw_action": action_int,
                "confidence": confidence,
                "p_flat": p_flat,
                "p_long": p_long,
                "p_margin": p_margin,
                "entropy_raw": entropy_raw,
                "state_tail": {
                    "position_fraction": float(state[-2]),
                    "normalized_pv": float(state[-1]),
                },
                "normalized_symbol": symbol_norm,
                "requested_symbol": symbol,
                "thresholds": {
                    "conf": self.cfg.conf_threshold,
                    "min_margin": self.cfg.min_margin,
                },
                "met_conf_threshold": bool(meets_conf),
                "met_margin_threshold": bool(meets_margin),
            }
            lookback_meta = getattr(self, '_last_state_metadata', None)
            if lookback_meta:
                metadata['lookback'] = lookback_meta
            debug_meta: Dict[str, Any] = {}
            state_summary = (state_meta or {}).get("state_summary", {}) if 'state_meta' in locals() else {}
            ohlcv_meta = (state_meta or {}).get("ohlcv", {}) if 'state_meta' in locals() else {}
            tail_meta = (state_meta or {}).get("tail_meta", {}) if 'state_meta' in locals() else {}
            debug_meta.update(
                {
                    "timeframe": ohlcv_meta.get("timeframe", self.STATE_TIMEFRAME),
                    "source": ohlcv_meta.get("source", "unknown"),
                    "last_ts": ohlcv_meta.get("last_ts"),
                    "age_sec": ohlcv_meta.get("age_sec"),
                    "rows": ohlcv_meta.get("rows"),
                    "state_hash": state_summary.get("state_hash"),
                    "state_mean": state_summary.get("state_mean"),
                    "state_std": state_summary.get("state_std"),
                    "state_min": state_summary.get("state_min"),
                    "state_max": state_summary.get("state_max"),
                    "feat_std": state_summary.get("feat_std"),
                    "feat_min": state_summary.get("feat_min"),
                    "feat_max": state_summary.get("feat_max"),
                    "extra_std": state_summary.get("extra_std"),
                    "tail_pf": state_summary.get("tail_pf"),
                    "tail_pv": state_summary.get("tail_pv"),
                    "tail_default": state_summary.get("tail_default"),
                    "nan_count": state_summary.get("nan_count"),
                    "inf_count": state_summary.get("inf_count"),
                    "state_head3": state_summary.get("state_head3"),
                    "state_tail3": state_summary.get("state_tail3"),
                    "action_int": action_int,
                    "p_flat": p_flat,
                    "p_long": p_long,
                    "p_margin": p_margin,
                    "conf_raw": confidence,
                    "entropy_raw": metadata.get("entropy_raw"),
                }
            )
            override_block = (state_meta or {}).get("tail_meta", {}).get("override_meta", {}) if 'state_meta' in locals() else {}
            debug_meta["override_meta"] = override_block
            metadata["debug"] = debug_meta
            if tail_meta:
                metadata["tail_meta"] = tail_meta
            obs_clip_frac_val = (state_meta or {}).get("obs_clip_frac")
            self._record_health_metrics(p_long, obs_clip_frac_val)
            health_ok, health_reasons, health_stats = self._evaluate_health(obs_clip_frac_val)
            metadata["health_ok"] = health_ok
            metadata["health_reasons"] = health_reasons
            if health_stats:
                metadata["health_stats"] = health_stats
            if not health_ok:
                logger.warning("🚨 HEALTH GUARD TRIGGERED! Reasons: %s | Stats: %s", health_reasons, health_stats)
                metadata.setdefault("reason", "health_guard")
                metadata["guarded_score"] = final_score
                decision = "GUARD_FALLBACK"
                metadata["action"] = decision
            self._maybe_dump_obs(symbol, state, metadata)

            if action_int == 1 and decision == "WEAK_LONG_IGNORED":
                logger.info(
                    "🎯 [SNIPER] Ignored weak PPO signal for %s. Conf: %.2f < %.2f",
                    symbol,
                    confidence,
                    CONFIDENCE_THRESHOLD,
                )

            return final_score, metadata
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("PPO prediction failed for %s: %s", symbol_norm, exc)
            return self.cfg.fallback_score, {"reason": "prediction_error", "error": str(exc)}

    async def _ensure_model_loaded(self) -> None:
        if self._model or self._load_error:
            return
        if PPO is None:
            self._load_error = "stable_baselines3_not_installed"
            logger.warning("SB3 PPO dependency missing; adapter will stay neutral.")
            return

        async with self._model_lock:
            if self._model or self._load_error:
                return
            try:
                model_path = self.cfg.model_path.expanduser().resolve()
                if not model_path.exists():
                    raise FileNotFoundError(model_path)
                self._model = PPO.load(str(model_path))
                spec_path = model_path.with_suffix(".obs_spec.json")
                if spec_path.exists():
                    try:
                        self._spec = load_obs_spec(spec_path)
                        logger.info("✅ Loaded PPO observation spec from %s (obs_dim=%d)", spec_path, self._spec.obs_dim)
                    except Exception as exc:
                        self._load_error = f"spec_load_failed: {exc}"
                        logger.error("❌ Failed to load PPO observation spec: %s", exc)
                        return
                vecnorm_path = model_path.with_suffix(".vecnormalize.pkl")
                if vecnorm_path.exists():
                    try:
                        obs_dim_for_norm = self._spec.obs_dim if self._spec else self._expected_obs_dim
                        if obs_dim_for_norm:
                            class _SimpleEnv(gym.Env):
                                def __init__(self, obs_dim: int):
                                    super().__init__()
                                    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                                    self.action_space = gym.spaces.Discrete(2)
                                def reset(self):
                                    return np.zeros(self.observation_space.shape, dtype=np.float32)
                                def step(self, action):
                                    return self.reset(), 0.0, True, False, {}

                            dummy_env = DummyVecEnv([lambda: _SimpleEnv(int(obs_dim_for_norm))])
                            self._vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
                        else:
                            self._vecnorm = VecNormalize.load(str(vecnorm_path))
                        self._vecnorm.training = False
                        self._vecnorm.norm_reward = False
                        logger.info(" Loaded VecNormalize stats from %s (clip_obs=%s)", vecnorm_path, getattr(self._vecnorm, "clip_obs", None))
                    except Exception as exc:
                        logger.warning(" Failed to load VecNormalize stats: %s", exc)
                # Modelin observation_space boyutunu çıkar
                try:
                    obs_space = getattr(self._model, "observation_space", None)
                    action_space = getattr(self._model, "action_space", None)
                    action_space_n = getattr(action_space, "n", None)
                    if obs_space is not None and getattr(obs_space, "shape", None):
                        self._expected_obs_dim = int(obs_space.shape[0])
                    else:
                        logger.warning(
                            "PPO model loaded but observation_space.shape missing; "
                            "state alignment will be disabled."
                        )
                        self._expected_obs_dim = None
                    manifest_summary = self._get_manifest_summary()
                    spec_obs_dim = self._spec.obs_dim if self._spec else None
                    logger.info(
                        "✅ [PPO-INIT] model_path=%s model_obs_dim=%s spec_obs_dim=%s manifest=%s manifest_feature_count=%s action_space_n=%s deterministic=%s",
                        model_path,
                        self._expected_obs_dim,
                        spec_obs_dim,
                        manifest_summary.get("version"),
                        manifest_summary.get("feature_count"),
                        action_space_n,
                        True,
                    )
                    if self._spec and self._expected_obs_dim and self._expected_obs_dim != self._spec.obs_dim:
                        self._load_error = "obs_dim_mismatch"
                        logger.error(
                            "⚠️ PPO obs_dim mismatch: model=%s spec=%s",
                            self._expected_obs_dim,
                            self._spec.obs_dim,
                        )
                        return
                    if not self._spec:
                        logger.warning("⚠️ PPO observation spec missing alongside model; will derive at runtime.")
                except Exception as exc:  # pragma: no cover - inspection safety
                    logger.warning("Failed to inspect PPO observation_space: %s", exc)
                    self._expected_obs_dim = None

            except Exception as exc:  # pragma: no cover - IO errors
                self._load_error = str(exc)
                logger.error("❌ Failed to load PPO model: %s", exc, exc_info=True)

    async def _build_state(
        self,
        symbol: str,
        *,
        position_fraction: Optional[float] = None,
        normalized_pv: Optional[float] = None,
        override_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Rebuild the PPO feature vector for the requested symbol."""
        self._last_state_metadata = {}
        if not self.market_data_pipeline or not self.feature_pipeline:
            logger.warning("PPO adapter missing data/feature pipeline; returning fallback")
            return None, {}

        try:
            query_symbol = self._symbol_alias_map.get(symbol, symbol)
            df = await self.market_data_pipeline.get_latest_ohlcv(
                query_symbol, self.STATE_TIMEFRAME, limit=2000
            )
            if (df is None or df.empty) and query_symbol != symbol:
                canonical_symbol = self._symbol_alias_map.get(symbol, symbol)
                df = await self.market_data_pipeline.get_latest_ohlcv(
                    canonical_symbol, self.STATE_TIMEFRAME, limit=2000
                )
                if (df is None or df.empty) and canonical_symbol != symbol:
                    df = await self.market_data_pipeline.get_latest_ohlcv(
                        symbol, self.STATE_TIMEFRAME, limit=2000
                    )
            if df is None or df.empty:
                logger.debug("PPO adapter received empty dataframe for %s", symbol)
                return None, {}

            ohlcv_meta: Dict[str, Any] = {
                "source": df.attrs.get("ohlcv_source", "unknown"),
                "rows": int(len(df)),
                "timeframe": self.STATE_TIMEFRAME,
                "retrieved_at": df.attrs.get("retrieved_at"),
            }
            try:
                if hasattr(df, "index") and len(df.index):
                    last_ts = df.index[-1]
                    if hasattr(last_ts, "to_pydatetime"):
                        ts_dt = last_ts.to_pydatetime()
                    else:
                        ts_dt = pd.Timestamp(last_ts).to_pydatetime()
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                    ohlcv_meta["last_ts"] = ts_dt.isoformat()
                    now_ts = datetime.now(timezone.utc)
                    ohlcv_meta["age_sec"] = float((now_ts - ts_dt).total_seconds())
            except Exception:
                ohlcv_meta["last_ts"] = None
                ohlcv_meta["age_sec"] = None

            if not self._spec:
                return None, {"reason": "missing_spec", "ohlcv": ohlcv_meta}

            features_df = self.feature_pipeline.extract_features(df, mode="price")
            if features_df is None or features_df.empty:
                return None, {"ohlcv": ohlcv_meta}

            if not self._spec and self._expected_obs_dim:
                base_len = len(features_df.columns)
                tail_len = len(spec_from_feature_columns([]).tail_names)
                remaining = int(self._expected_obs_dim) - (base_len + tail_len)
                if remaining == len(self.EXTRA_FEATURE_NAMES):
                    self._spec = spec_from_feature_columns(
                        features_df.columns,
                        extra_feature_names=self.EXTRA_FEATURE_NAMES,
                        version="derived",
                    )
                    logger.info("✅ Derived PPO observation spec at runtime (features=%d + extras=%d)", base_len, len(self.EXTRA_FEATURE_NAMES))
                elif remaining == 0:
                    self._spec = spec_from_feature_columns(
                        features_df.columns,
                        extra_feature_names=[],
                        version="derived",
                    )
                    logger.info("✅ Derived PPO observation spec at runtime (features=%d)", base_len)
                else:
                    logger.error("❌ Unable to derive PPO spec: expected_dim=%s base_features=%d tail=%d", self._expected_obs_dim, base_len, tail_len)
                    return None, {"reason": "spec_derivation_failed", "ohlcv": ohlcv_meta}

            latest_row = features_df.iloc[-1]
            if latest_row.isna().any():
                logger.debug("PPO adapter found NaN in feature vector for %s", symbol)
                return None, {"ohlcv": ohlcv_meta}

            extra_values: Dict[str, float] = {}
            if self._spec.extra_feature_names:
                extra_arr = self._compute_extra_features_from_price(df)
                if len(extra_arr) < len(self._spec.extra_feature_names):
                    logger.error("PPO extra feature count mismatch")
                    return None, {"reason": "extra_mismatch", "ohlcv": ohlcv_meta}
                extra_values = {name: float(extra_arr[i]) for i, name in enumerate(self._spec.extra_feature_names)}

            tail_array, tail_meta, tail_values = self._compose_tail_state(position_fraction, normalized_pv)
            tail_meta["override_meta"] = override_meta or {}

            try:
                state = build_observation(
                    self._spec,
                    latest_row,
                    extra_values=extra_values,
                    tail_values=tail_values,
                )
            except Exception as exc:
                logger.error("PPO observation build failed: %s", exc)
                return None, {"reason": "obs_build_failed", "error": str(exc), "ohlcv": ohlcv_meta}

            pre_norm_summary = self._summarize_state(
                state,
                len(self._spec.feature_names),
                len(self._spec.extra_feature_names),
                tail_meta,
            )

            post_norm_summary = {}
            obs_clip_frac = None
            z_abs_mean = None
            z_abs_p99 = None

            if self._vecnorm:
                try:
                    obs_tensor = np.array([state], dtype=np.float32)
                    normed = self._vecnorm.normalize_obs(obs_tensor.copy())
                    if hasattr(self._vecnorm, "clip_obs") and self._vecnorm.clip_obs:
                        clip_val = float(self._vecnorm.clip_obs)
                        clipped = np.clip(obs_tensor, -clip_val, clip_val)
                        obs_clip_frac = float(np.mean(obs_tensor != clipped))
                    state = normed[0]
                    abs_vals = np.abs(state)
                    z_abs_mean = float(np.mean(abs_vals))
                    z_abs_p99 = float(np.quantile(abs_vals, 0.99))
                    post_norm_summary = self._summarize_state(
                        state,
                        len(self._spec.feature_names),
                        len(self._spec.extra_feature_names),
                        tail_meta,
                    )
                except Exception as exc:
                    logger.warning("VecNormalize normalization failed: %s", exc)

            state_summary = post_norm_summary or pre_norm_summary

            self._last_state_metadata = self._generate_lookback_metadata(df, symbol)
            self._last_state_metadata.update(
                {
                    "ohlcv": ohlcv_meta,
                    "state_summary": state_summary,
                    "pre_norm_summary": pre_norm_summary,
                    "post_norm_summary": post_norm_summary,
                    "obs_norm_present": bool(self._vecnorm),
                    "obs_clip_frac": obs_clip_frac,
                    "z_abs_mean": z_abs_mean,
                    "z_abs_p99": z_abs_p99,
                    "feature_len": int(len(self._spec.feature_names)),
                    "extra_len": int(len(self._spec.extra_feature_names)),
                    "tail_meta": tail_meta,
                }
            )
            return state, self._last_state_metadata
        except Exception as exc:
            logger.error("PPO adapter failed to build state for %s: %s", symbol, exc)
            self._last_state_metadata = {}
            return None, {}

    def supported_symbols(self) -> Iterable[str]:
        return self.cfg.symbols

    def _align_state_dim(self, state: np.ndarray) -> np.ndarray:
        """
        Align the state vector to the PPO model's expected observation dimension.

        - Eğer modelden observation dim alınamadıysa, state aynen döner.
        - Eğer state daha kısa ise: sonuna 0.0 ile pad edilir.
        - Eğer state daha uzun ise: sonundan truncate edilir (safety).
        """
        if self._expected_obs_dim is None:
            return state.astype(np.float32)

        current_dim = int(state.shape[0])
        expected_dim = int(self._expected_obs_dim)

        if current_dim == expected_dim:
            return state.astype(np.float32)

        if current_dim > expected_dim:
            logger.warning(
                "PPO state dim (%d) > expected_obs_dim (%d). Truncating extra features.",
                current_dim,
                expected_dim,
            )
            return state[:expected_dim].astype(np.float32)

        # current_dim < expected_dim → pad gerekiyor (normalde olmamalı)
        missing = expected_dim - current_dim
        logger.warning(
            "PPO state dim (%d) < expected_obs_dim (%d). Padding %d dummy features.",
            current_dim,
            expected_dim,
            missing,
        )
        pad_values = np.zeros(missing, dtype=np.float32)
        padded = np.concatenate([state, pad_values])
        return padded.astype(np.float32)

    def _summarize_state(
        self,
        state: np.ndarray,
        feature_len: int,
        extra_len: int,
        tail_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        try:
            summary["state_mean"] = float(np.mean(state))
            summary["state_std"] = float(np.std(state))
            summary["state_min"] = float(np.min(state))
            summary["state_max"] = float(np.max(state))
            summary["state_hash"] = hashlib.sha1(state.tobytes()).hexdigest()[:12]
            feat_slice = state[:feature_len] if feature_len <= len(state) else state
            extra_slice = state[feature_len : feature_len + extra_len] if feature_len + extra_len <= len(state) else np.array([], dtype=np.float32)
            tail_slice = state[-self.TAIL_DIM :] if len(state) >= self.TAIL_DIM else np.array([0.0, 0.0], dtype=np.float32)
            summary["feat_std"] = float(np.std(feat_slice)) if feat_slice.size else 0.0
            summary["extra_std"] = float(np.std(extra_slice)) if extra_slice.size else 0.0
            summary["feat_min"] = float(np.min(feat_slice)) if feat_slice.size else 0.0
            summary["feat_max"] = float(np.max(feat_slice)) if feat_slice.size else 0.0
            summary["tail_pf"] = float(tail_slice[0]) if tail_slice.size else 0.0
            summary["tail_pv"] = float(tail_slice[1]) if tail_slice.size > 1 else 0.0
            summary["tail_default"] = bool(tail_meta.get("default_used"))
            summary["nan_count"] = int(np.isnan(state).sum())
            summary["inf_count"] = int(np.isinf(state).sum())
            summary["state_head3"] = [float(x) for x in state[:3]] if state.size else []
            summary["state_tail3"] = [float(x) for x in state[-3:]] if state.size >= 3 else [float(x) for x in state]
        except Exception:
            summary["state_mean"] = summary.get("state_mean", 0.0)
            summary["state_std"] = summary.get("state_std", 0.0)
        return summary

    def _record_health_metrics(self, p_long: Optional[float], obs_clip_frac: Optional[float]) -> None:
        window = max(2, self.cfg.health_window)
        if p_long is not None:
            self._p_long_history.append(float(p_long))
            if len(self._p_long_history) > window:
                self._p_long_history = self._p_long_history[-window:]
        if obs_clip_frac is not None:
            self._clip_history.append(float(obs_clip_frac))
            if len(self._clip_history) > window:
                self._clip_history = self._clip_history[-window:]

    def _evaluate_health(self, obs_clip_frac: Optional[float]) -> Tuple[bool, List[str], Dict[str, Any]]:
        reasons: List[str] = []
        stats: Dict[str, Any] = {}
        window = max(2, self.cfg.health_window)
        if self.cfg.require_vecnorm and not self._vecnorm:
            reasons.append("vecnorm_missing")
        if self._p_long_history:
            hist = self._p_long_history[-window:]
            std = float(np.std(hist))
            stats["p_long_std"] = std
            if len(hist) >= max(2, window // 2) and std < self.cfg.health_min_std:
                reasons.append("p_long_low_variance")
        if obs_clip_frac is not None:
            clip_hist = self._clip_history[-window:] if self._clip_history else []
            if clip_hist:
                clip_mean = float(np.mean(clip_hist))
                stats["clip_mean"] = clip_mean
                if clip_mean > self.cfg.health_clip_frac_limit:
                    reasons.append("obs_clip_high")
        return not reasons, reasons, stats

    def _compute_extra_features_from_price(self, df: pd.DataFrame) -> np.ndarray:
        """Delegate to shared helper for price-derived extras."""
        return compute_price_extras(df)

    def _compose_tail_state(
        self,
        position_fraction: Optional[float],
        normalized_pv: Optional[float],
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, float]]:
        tail = self._tail_defaults.copy()
        meta: Dict[str, Any] = {
            "default_used": True,
            "source_override": False,
            "reason": None,
        }

        if position_fraction is not None:
            try:
                tail[0] = float(max(0.0, min(1.0, position_fraction)))
                meta["default_used"] = False
                meta["source_override"] = True
            except (TypeError, ValueError):
                meta["reason"] = "invalid_position_fraction"
                logger.debug("PPO tail position_fraction invalid: %s", position_fraction)
        else:
            meta["reason"] = meta.get("reason") or "position_fraction_missing"

        if normalized_pv is not None:
            try:
                tail[1] = float(max(0.1, min(5.0, normalized_pv)))
                meta["default_used"] = False
                meta["source_override"] = True
            except (TypeError, ValueError):
                meta["reason"] = meta.get("reason") or "invalid_normalized_pv"
                logger.debug("PPO tail normalized_pv invalid: %s", normalized_pv)
        else:
            if meta.get("reason") is None:
                meta["reason"] = "normalized_pv_missing"

        tail_values = {
            "position_fraction": float(tail[0]),
            "normalized_pv": float(tail[1]),
        }
        return tail.astype(np.float32), meta, tail_values

    def get_last_lookback_metadata(self) -> Dict[str, Any]:
        """Return a shallow copy of the most recent lookback metadata."""
        return dict(self._last_state_metadata or {})

    def _generate_lookback_metadata(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df is None or df.empty:
            return {}

        lookback_df = df.tail(self.cfg.lookback_bars) if self.cfg.lookback_bars else df
        if lookback_df is None or lookback_df.empty:
            return {}

        metadata: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": self.cfg.timeframe,
            "bars_requested": int(self.cfg.lookback_bars),
            "bars_available": int(len(lookback_df)),
            "start": self._format_timestamp(lookback_df.index[0])
            if hasattr(lookback_df, "index") and len(lookback_df.index)
            else None,
            "end": self._format_timestamp(lookback_df.index[-1])
            if hasattr(lookback_df, "index") and len(lookback_df.index)
            else None,
            "window_stats": {},
        }

        metadata["overall"] = self._summarize_window(lookback_df)

        for window in self.cfg.lookback_windows:
            if window <= 1:
                continue
            window_df = lookback_df.tail(window)
            if window_df is None or len(window_df) < 2:
                continue
            metadata["window_stats"][str(window)] = self._summarize_window(window_df)

        return metadata

    def _summarize_window(self, window_df: pd.DataFrame) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"bars": int(len(window_df))}
        if len(window_df) < 2:
            summary.update(
                {
                    "price_change_pct": 0.0,
                    "volatility_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "avg_volume": 0.0,
                }
            )
            return summary

        close = window_df["close"].astype(float)
        first_close = float(close.iloc[0])
        last_close = float(close.iloc[-1])
        price_change = (last_close / first_close) - 1.0 if first_close else 0.0
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        volatility = float(returns.std(skipna=True) or 0.0)
        volume_series = window_df.get("volume")
        avg_volume = (
            float(volume_series.astype(float).mean())
            if volume_series is not None and not volume_series.empty
            else 0.0
        )
        max_drawdown = self._compute_max_drawdown(close)
        atr_series = window_df.get("atr")
        if atr_series is not None and not atr_series.empty:
            atr_value = float(atr_series.astype(float).iloc[-min(len(atr_series), 14):].mean())
        else:
            atr_value = None

        summary.update(
            {
                "price_change_pct": float(price_change),
                "volatility_pct": float(volatility),
                "max_drawdown_pct": float(max_drawdown),
                "avg_volume": float(avg_volume),
            }
        )
        if atr_value is not None and not math.isnan(atr_value):
            summary["atr"] = float(atr_value)
        return summary

    @staticmethod
    def _compute_max_drawdown(close: pd.Series) -> float:
        if close is None or close.empty:
            return 0.0
        running_max = close.cummax()
        drawdowns = close / running_max - 1.0
        if drawdowns.empty:
            return 0.0
        return float(abs(drawdowns.min()))

    @staticmethod
    def _format_timestamp(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if isinstance(value, pd.Timestamp):
                return value.to_pydatetime().isoformat()
            if isinstance(value, np.datetime64):
                return pd.Timestamp(value).to_pydatetime().isoformat()
            if hasattr(value, "isoformat"):
                return value.isoformat()
        except Exception:  # pragma: no cover - formatting safety
            return str(value)
        return str(value)

    def _get_manifest_summary(self) -> Dict[str, Any]:
        manifest = getattr(self.feature_pipeline, "manifest", {}) if hasattr(self, "feature_pipeline") else {}
        feature_count = None
        version = None
        if isinstance(manifest, dict):
            feature_count = manifest.get("feature_count")
            version = manifest.get("version")
        return {"feature_count": feature_count, "version": version}

    def _maybe_dump_obs(self, symbol: str, state: np.ndarray, metadata: Dict[str, Any]) -> None:
        dump_path = os.getenv("PPO_DUMP_OBS")
        if not dump_path:
            return
        try:
            limit = int(os.getenv("PPO_DUMP_LIMIT", "50") or 0)
        except ValueError:
            limit = 0
        counter_key = "_ppo_dump_count"
        current = getattr(self, counter_key, 0)
        if limit and current >= limit:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "state": state.tolist(),
            "metadata": metadata,
        }
        try:
            Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
            with open(dump_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            setattr(self, counter_key, current + 1)
        except Exception:
            logger.debug("Failed to dump PPO obs to %s", dump_path)
