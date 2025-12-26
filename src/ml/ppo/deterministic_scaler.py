from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import numpy as np

# Spec yükleyiciyi kendi proje yoluna göre import et
# from src.ml.ppo.observation_spec import ObservationSpec, load_spec 
# (Yukarıdaki importun çalıştığından emin ol veya mockla)

logger = logging.getLogger(__name__)

class DeterministicScaler:
    """
    Deterministic PPO scaler with explicit name-to-transform mapping.
    v1.3 Fixes: roc/trix handling, strict close validation, nan logging.
    """

    def __init__(
        self,
        obs_spec_path: Optional[Path] = None,
        *,
        spec: Any = None, # Type hint güncellendi
        log_every: int = 100,
    ) -> None:
        # Load spec logic (Basitleştirildi)
        if spec is None:
            if obs_spec_path is None:
                raise ValueError("obs_spec_path or spec is required.")
            from src.ml.ppo.observation_spec import load_spec # Lazy import
            spec = load_spec(Path(obs_spec_path))
            
        self._spec = spec
        # Feature listesini birleştir
        self.feature_names: List[str] = (
            list(spec.feature_names)
            + list(spec.extra_feature_names)
            + list(spec.tail_names)
        )

        # --- 1. SET DEFINITIONS (Explicit Groups) ---
        # v1.3: bb_middle eklendi, trix çıkarıldı
        self._price_level_set = {
            "sma", "ema", "vwap", "bb_upper", "bb_lower", "bb_middle",
            "keltner", "pivot", "fib", "donchian", "r1", "s1", "r2", "s2"
        }
        # v1.3: trix/roc buraya uygun değil, price_diff fiyat birimlidir
        self._price_diff_set = {
            "width", "atr", "range", "macd", "momentum", 
            "dpo", "volatility", "std"
        }
        # v1.3: trix buradan çıkarıldı
        self._oscillator_set = {
            "rsi", "stoch", "mfi", "adx", "plus_di", "minus_di"
        }
        # v1.3: Yeni grup (Returns/Percentage like) -> roc, trix
        self._returns_set = {
            "roc", "trix", "pct_change"
        }

        self._transform_map = self._build_map(self.feature_names)
        
        # --- FAIL-FAST VALIDATION ---
        missing = [name for name in self.feature_names if name not in self._transform_map]
        if missing:
            raise ValueError(f"Missing feature scaling definition for: {missing}")
        
        self._log_every = max(0, int(log_every))
        self._step = 0

    def _build_map(self, features: Iterable[str]) -> Dict[str, Callable[[float, float], float]]:
        mapping = {}
        for name in features:
            # 1. SKIP / PASS-THROUGH (Extra ve Ret)
            # extra_range_norm'un price_diff'e takılmaması için en başta
            if name.startswith("extra_") or name.startswith("ret"):
                mapping[name] = self._scale_identity
                continue

            # bb_position/vortex/trend_strength gibi normalize metrikler için kimlik geçişi
            if "bb_position" in name or "vortex_pos" in name or "trend_strength" in name:
                mapping[name] = self._scale_identity
                continue
            
            # 2. PRICE LEVEL -> (x / close) - 1
            if any(p in name for p in self._price_level_set):
                mapping[name] = self._scale_price_level
            
            # 3. PRICE DIFF -> x / close
            elif any(p in name for p in self._price_diff_set):
                mapping[name] = self._scale_price_diff

            # 3b. Destek/direnç mesafeleri fiyat farkı ölçeğiyle normalize edilir
            elif name in {"support_distance", "resistance_distance"}:
                mapping[name] = self._scale_price_diff

            # 4. RETURNS / PERCENTAGE -> x / 100 (v1.3 Fix: roc_10, trix_15)
            elif any(p in name for p in self._returns_set):
                mapping[name] = self._scale_returns

            # 5. OSCILLATORS -> (x - 50) / 50
            elif any(p in name for p in self._oscillator_set):
                mapping[name] = self._scale_oscillator

            # 6. WILLIAMS -> x / 100
            elif "williams" in name:
                mapping[name] = self._scale_williams

            # 7. VOLUME / OBV -> sign * log1p(|x|) / 10
            elif ("volume" in name or "obv" in name) and "ratio" not in name:
                mapping[name] = self._scale_volume

            # 8. RATIOS -> log(max(x, 1e-6))
            elif "ratio" in name:
                mapping[name] = self._scale_ratio_log

            # 9. CCI -> tanh(x / 100)
            elif "cci" in name:
                mapping[name] = self._scale_cci

            # 10. MARKET PHASE -> (x - 1.5) / 1.5
            elif "market_phase" in name:
                mapping[name] = self._scale_market_phase
            
            # 11. TAILS (Pass through)
            elif name in ["position_fraction", "normalized_pv"]:
                mapping[name] = self._scale_identity

        return mapping

    # --- TRANSFORM METHODS ---

    @staticmethod
    def _safe_close(close: float) -> float:
        # v1.3 Fix: Strict validation. NaN/Inf/Zero is fatal for price scaling.
        if not math.isfinite(close) or close <= 0:
            raise ValueError(f"Invalid reference close price for scaling: {close}")
        return close

    @staticmethod
    def _scale_price_level(x: float, close: float) -> float:
        return (x / close) - 1.0

    @staticmethod
    def _scale_price_diff(x: float, close: float) -> float:
        return x / close

    @staticmethod
    def _scale_oscillator(x: float, close: float) -> float: # noqa: ARG001
        return (x - 50.0) / 50.0

    @staticmethod
    def _scale_williams(x: float, close: float) -> float: # noqa: ARG001
        return x / 100.0
    
    @staticmethod
    def _scale_returns(x: float, close: float) -> float: # noqa: ARG001
        # v1.3: ROC ve TRIX için. %1.5 -> 0.015
        return x / 100.0

    @staticmethod
    def _scale_volume(x: float, close: float) -> float: # noqa: ARG001
        return math.copysign(1.0, x) * math.log1p(abs(x)) / 10.0

    @staticmethod
    def _scale_ratio_log(x: float, close: float) -> float: # noqa: ARG001
        # Clamp to avoid -inf. 
        return math.log(max(x, 1e-6))

    @staticmethod
    def _scale_cci(x: float, close: float) -> float: # noqa: ARG001
        return math.tanh(x / 100.0)

    @staticmethod
    def _scale_market_phase(x: float, close: float) -> float: # noqa: ARG001
        return (x - 1.5) / 1.5

    @staticmethod
    def _scale_identity(x: float, close: float) -> float: # noqa: ARG001
        return x

    def transform(self, row_dict: Dict[str, Any], close_price: float) -> np.ndarray:
        # v1.3: Fail-fast for close price
        try:
            close = self._safe_close(float(close_price))
        except ValueError as exc:
            logger.error("DeterministicScaler invalid close_price=%s: %s", close_price, exc)
            raise

        scaled_values: List[float] = []
        
        # Performance optimization: Local lookup
        t_map = self._transform_map
        
        for name in self.feature_names:
            if name not in row_dict:
                raise KeyError(f"Missing feature value for '{name}'")
            
            raw = float(row_dict[name])
            # Apply transform
            val = t_map[name](raw, close)
            scaled_values.append(val)

        vec = np.asarray(scaled_values, dtype=np.float32)
        
        # v1.3: Observability for silent NaN masking
        if not np.isfinite(vec).all():
            nan_count = int(np.isnan(vec).sum())
            inf_count = int(np.isinf(vec).sum())
            bad_idx = np.where(~np.isfinite(vec))[0].tolist()
            bad_feats = [self.feature_names[i] for i in bad_idx[:5]]
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "DeterministicScaler produced NaN/Inf (nan=%d inf=%d) example_feats=%s; cleaning to 0.0",
                    nan_count,
                    inf_count,
                    bad_feats,
                )

        # Final cleanup safety
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        self._step += 1
        if self._log_every > 0 and logger.isEnabledFor(logging.DEBUG) and self._step % self._log_every == 0:
            # Simple check on the vector
            raw_gt_10 = 0 # Not tracking raw array here for speed, focused on output
            scaled_gt_5 = np.sum(np.abs(vec) > 5.0)
            logger.debug(f"DeterministicScaler stats (step {self._step}): Scaled values > 5.0: {scaled_gt_5}")

        return vec