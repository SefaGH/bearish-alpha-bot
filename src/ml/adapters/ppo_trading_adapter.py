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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd  # extra feature hesapları için

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
        self._normalized_symbols = set()
        self._expected_obs_dim: Optional[int] = None  # PPO modelinin beklediği observation dim
        self._last_state_metadata: Dict[str, Any] = {}

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

        state = await self._build_state(
            symbol_norm,
            position_fraction=position_fraction,
            normalized_pv=normalized_pv,
        )
        if state is None:
            return self.cfg.fallback_score, {"reason": "missing_state"}

        try:
            action, _ = self._model.predict(state[np.newaxis, :], deterministic=True)
            action_int = int(np.asarray(action).item())

            confidence = 1.0
            if hasattr(self._model, "policy"):
                try:  # Torch is an optional runtime dependency outside PPO
                    import torch as th
                except Exception:  # pragma: no cover - safety net for stripped envs
                    th = None

                if th is not None:
                    with th.no_grad():
                        obs_tensor = th.as_tensor(state[np.newaxis, :], device=self._model.device)
                        distribution = self._model.policy.get_distribution(obs_tensor)
                        probs = distribution.distribution.probs.cpu().numpy()[0]
                        confidence = float(probs[action_int])

            CONFIDENCE_THRESHOLD = 0.75
            final_score = 0.0
            decision = "FLAT"

            if action_int == 1:
                if confidence >= CONFIDENCE_THRESHOLD:
                    final_score = 1.0
                    decision = "LONG"
                else:
                    final_score = 0.5
                    decision = "WEAK_LONG_IGNORED"
            else:
                final_score = 0.0
                decision = "FLAT"

            metadata = {
                "symbol": symbol_norm,
                "action": decision,
                "raw_action": action_int,
                "confidence": confidence,
                "state_tail": {
                    "position_fraction": float(state[-2]),
                    "normalized_pv": float(state[-1]),
                },
                "normalized_symbol": symbol_norm,
                "requested_symbol": symbol,
            }
            lookback_meta = getattr(self, '_last_state_metadata', None)
            if lookback_meta:
                metadata['lookback'] = lookback_meta

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

                # Modelin observation_space boyutunu çıkar
                try:
                    obs_space = getattr(self._model, "observation_space", None)
                    if obs_space is not None and getattr(obs_space, "shape", None):
                        self._expected_obs_dim = int(obs_space.shape[0])
                        logger.info(
                            "✅ PPO adapter loaded model from %s (expected_obs_dim=%d)",
                            model_path,
                            self._expected_obs_dim,
                        )
                    else:
                        logger.warning(
                            "PPO model loaded but observation_space.shape missing; "
                            "state alignment will be disabled."
                        )
                        self._expected_obs_dim = None
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
    ) -> Optional[np.ndarray]:
        """Rebuild the PPO feature vector for the requested symbol."""
        self._last_state_metadata = {}
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

            # 1) GEMMA/manifest'e göre price feature'ları (82)
            features_df = self.feature_pipeline.extract_features(df, mode="price")
            if features_df is None or features_df.empty:
                return None

            latest = features_df.iloc[-1].to_numpy(dtype=np.float32)
            if np.isnan(latest).any():
                logger.debug("PPO adapter found NaN in feature vector for %s", symbol)
                return None

            # 2) Ek 5 fiyat türevi feature (sabit, GEMMA'dan bağımsız)
            extra = self._compute_extra_features_from_price(df)

            # 3) Tail (position_fraction, normalized_pv)
            tail = self._compose_tail_state(position_fraction, normalized_pv)

            # Beklenen doğal yapı: 82 (GEMMA) + 5 (extra) + 2 (tail) = 89
            raw_state = np.concatenate([latest, extra, tail]).astype(np.float32)

            # Güvenlik için hala hizalama yapıyoruz ama normalde no-op olmalı
            state = self._align_state_dim(raw_state)
            self._last_state_metadata = self._generate_lookback_metadata(df, symbol)
            return state
        except Exception as exc:
            logger.error("PPO adapter failed to build state for %s: %s", symbol, exc)
            self._last_state_metadata = {}
            return None

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

    def _compute_extra_features_from_price(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute 5 additional price-derived features to bridge PPO's 87-dim legacy
        space with the current GEMMA 82-dim feature space.

        Features (heuristic, ama stable ve yalnızca OHLCV'den türetilmiş):

        1) extra_ret_1           : last log-return
        2) extra_ret_3           : sum of last 3 log-returns
        3) extra_range_norm      : (high - low) / close (last bar)
        4) extra_vol_10          : std of pct_change over last 10 bars
        5) extra_trend_ema_ratio : (ema_10 - ema_50) / ema_50 (last bar)
        """
        # Varsayılan sıfırlar (her durumda 5-dim dönelim)
        extra = np.zeros(5, dtype=np.float32)

        try:
            if df is None or df.empty:
                return extra

            # En az 2 bar yoksa, her şey 0 kalsın
            if len(df) < 2:
                return extra

            close = df["close"].astype(float)
            high = df.get("high", close).astype(float)
            low = df.get("low", close).astype(float)

            # ---------- 1) & 2) Log-return'ler ----------
            # log-return serisi: ln(C_t / C_{t-1})
            log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)

            last_log_ret = float(log_ret.iloc[-1]) if not np.isnan(log_ret.iloc[-1]) else 0.0
            extra[0] = np.float32(last_log_ret)  # extra_ret_1

            if len(log_ret) >= 3:
                last3 = log_ret.iloc[-3:].fillna(0.0).sum()
                extra[1] = np.float32(last3)  # extra_ret_3
            else:
                extra[1] = 0.0

            # ---------- 3) Normalized range ----------
            last_high = float(high.iloc[-1])
            last_low = float(low.iloc[-1])
            last_close = float(close.iloc[-1])
            denom = last_close if last_close != 0 else 1.0
            extra[2] = np.float32((last_high - last_low) / denom)  # extra_range_norm

            # ---------- 4) Kısa vadeli volatilite (10 bar) ----------
            pct = close.pct_change().replace([np.inf, -np.inf], np.nan)
            if len(pct) >= 10:
                extra[3] = np.float32(pct.iloc[-10:].std(skipna=True) or 0.0)  # extra_vol_10
            else:
                extra[3] = np.float32(pct.std(skipna=True) or 0.0)

            # ---------- 5) EMA spread oranı ----------
            # Basit EMA hesapları (price predictor'dan bağımsız)
            ema10 = close.ewm(span=10, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            last_ema10 = float(ema10.iloc[-1])
            last_ema50 = float(ema50.iloc[-1])
            denom_ema = last_ema50 if last_ema50 != 0 else 1.0
            extra[4] = np.float32((last_ema10 - last_ema50) / denom_ema)  # extra_trend_ema_ratio

        except Exception as exc:
            logger.warning("PPO extra feature computation failed: %s", exc)

        return extra

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
