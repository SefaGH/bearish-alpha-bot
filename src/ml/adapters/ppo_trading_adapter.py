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
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

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

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PPOAdapterConfig":
        rl_cfg = cfg or {}
        symbols_cfg = rl_cfg.get("ppo_symbols", ["BTC/USDT:USDT"])
        if isinstance(symbols_cfg, str):
            symbols = tuple(s.strip() for s in symbols_cfg.split(",") if s.strip())
        else:
            symbols = tuple(str(s).strip() for s in symbols_cfg if s)

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
        )


class PPOTradingAdapter:
    """Thin wrapper around a SB3 PPO model for long-signal modulation."""

    STATE_TIMEFRAME = "1h"

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
    ) -> Tuple[float, Dict[str, Any]]:
        """Return PPO score for long/flat decision (+ metadata)."""
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

        state = await self._build_state(
            symbol_norm,
            position_fraction=position_fraction,
            normalized_pv=normalized_pv,
        )
        if state is None:
            return self.cfg.fallback_score, {"reason": "missing_state"}

        try:
            action, _ = self._model.predict(state[np.newaxis, :], deterministic=True)
            action_int = int(action)
            score = 1.0 if action_int == 1 else 0.0
            metadata = {
                "symbol": symbol_norm,
                "action": "LONG" if action_int == 1 else "FLAT",
                "raw_action": action_int,
            }
            return score, metadata
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
                logger.info("✅ PPO adapter loaded model from %s", model_path)
            except Exception as exc:  # pragma: no cover - IO errors
                self._load_error = str(exc)
                logger.error("❌ Failed to load PPO model: %s", exc, exc_info=True)

    async def _build_state(
        self,
        symbol: str,
        *,
        position_fraction: Optional[float] = None,
        normalized_pv: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Rebuild the PPO feature vector for the requested symbol."""
        if not self.market_data_pipeline or not self.feature_pipeline:
            logger.warning("PPO adapter missing data/feature pipeline; returning fallback")
            return None

        try:
            query_symbol = self._symbol_alias_map.get(symbol, symbol)
            df = await self.market_data_pipeline.get_latest_ohlcv(
                query_symbol, self.STATE_TIMEFRAME
            )
            if (df is None or df.empty) and query_symbol != symbol:
                canonical_symbol = self._symbol_alias_map.get(symbol, symbol)
                df = await self.market_data_pipeline.get_latest_ohlcv(
                    canonical_symbol, self.STATE_TIMEFRAME
                )
                if (df is None or df.empty) and canonical_symbol != symbol:
                    df = await self.market_data_pipeline.get_latest_ohlcv(
                        symbol, self.STATE_TIMEFRAME
                    )
            if df is None or df.empty:
                logger.debug("PPO adapter received empty dataframe for %s", symbol)
                return None

            features_df = self.feature_pipeline.extract_features(df, mode="price")
            if features_df is None or features_df.empty:
                return None

            latest = features_df.iloc[-1].to_numpy(dtype=np.float32)
            if np.isnan(latest).any():
                logger.debug("PPO adapter found NaN in feature vector for %s", symbol)
                return None
            tail = self._compose_tail_state(position_fraction, normalized_pv)
            return np.concatenate([latest, tail])
        except Exception as exc:
            logger.error("PPO adapter failed to build state for %s: %s", symbol, exc)
            return None

    def supported_symbols(self) -> Iterable[str]:
        return self.cfg.symbols

    def _compose_tail_state(
        self,
        position_fraction: Optional[float],
        normalized_pv: Optional[float],
    ) -> np.ndarray:
        tail = self._tail_defaults.copy()

        if position_fraction is not None:
            try:
                tail[0] = float(max(0.0, min(1.0, position_fraction)))
            except (TypeError, ValueError):
                logger.debug("PPO tail position_fraction invalid: %s", position_fraction)
        if normalized_pv is not None:
            try:
                tail[1] = float(max(0.1, min(5.0, normalized_pv)))
            except (TypeError, ValueError):
                logger.debug("PPO tail normalized_pv invalid: %s", normalized_pv)
        return tail.astype(np.float32)
