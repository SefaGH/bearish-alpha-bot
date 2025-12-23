from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


DEFAULT_TAIL_NAMES = ["position_fraction", "normalized_pv"]
DEFAULT_EXTRA_FEATURE_NAMES = ["extra_ret_1", "extra_ret_3", "extra_range_norm", "extra_vol_10", "extra_trend_ema_ratio"]


@dataclass
class ObservationSpec:
    version: str
    feature_names: List[str]
    extra_feature_names: List[str]
    tail_names: List[str]
    dtype: str = "float32"

    @property
    def obs_dim(self) -> int:
        return len(self.feature_names) + len(self.extra_feature_names) + len(self.tail_names)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: Union[str, Dict[str, Any]]) -> "ObservationSpec":
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
        return cls(
            version=data.get("version", "unknown"),
            feature_names=list(data.get("feature_names", [])),
            extra_feature_names=list(data.get("extra_feature_names", [])),
            tail_names=list(data.get("tail_names", DEFAULT_TAIL_NAMES)),
            dtype=str(data.get("dtype", "float32")),
        )


def save_spec(spec: ObservationSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_json(), encoding="utf-8")


def load_spec(path: Path) -> ObservationSpec:
    return ObservationSpec.from_json(path.read_text(encoding="utf-8"))


def spec_from_feature_columns(
    feature_columns: Sequence[str],
    *,
    extra_feature_names: Optional[Iterable[str]] = None,
    tail_names: Optional[Iterable[str]] = None,
    version: str = "1.0",
) -> ObservationSpec:
    return ObservationSpec(
        version=version,
        feature_names=list(feature_columns),
        extra_feature_names=list(extra_feature_names or []),
        tail_names=list(tail_names or DEFAULT_TAIL_NAMES),
        dtype="float32",
    )


def build_observation(
    spec: ObservationSpec,
    feature_row: Union[pd.Series, np.ndarray, Dict[str, Any]],
    *,
    extra_values: Optional[Dict[str, Any]] = None,
    tail_values: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    def _val_from(source: Union[pd.Series, np.ndarray, Dict[str, Any]], name: str, idx: int) -> float:
        if isinstance(source, pd.Series):
            if name in source:
                return float(source[name])
            if idx < len(source):
                return float(source.iloc[idx])
            raise KeyError(name)
        if isinstance(source, dict):
            if name in source:
                return float(source[name])
            raise KeyError(name)
        if isinstance(source, np.ndarray):
            if idx < source.shape[0]:
                return float(source[idx])
            raise KeyError(name)
        raise KeyError(name)

    values: List[float] = []
    for i, name in enumerate(spec.feature_names):
        values.append(_val_from(feature_row, name, i))

    extra_values = extra_values or {}
    for name in spec.extra_feature_names:
        if name not in extra_values:
            raise KeyError(f"Missing extra feature: {name}")
        values.append(float(extra_values[name]))

    tail_values = tail_values or {}
    for name in spec.tail_names:
        val = tail_values.get(name)
        if val is None:
            raise KeyError(f"Missing tail value: {name}")
        values.append(float(val))

    vec = np.asarray(values, dtype=spec.dtype)
    if vec.shape[0] != spec.obs_dim:
        raise ValueError(f"Observation dim mismatch: got {vec.shape[0]}, expected {spec.obs_dim}")
    return vec.astype(np.float32)


def compute_price_extras(df: pd.DataFrame) -> np.ndarray:
    """
    Compute deterministic price-derived extras used to bridge smaller manifest feature sets.
    Returns a 5-dim vector matching DEFAULT_EXTRA_FEATURE_NAMES.
    """
    extra = np.zeros(len(DEFAULT_EXTRA_FEATURE_NAMES), dtype=np.float32)
    try:
        if df is None or df.empty or len(df) < 2:
            return extra

        close = df["close"].astype(float)
        high = df.get("high", close).astype(float)
        low = df.get("low", close).astype(float)

        log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        extra[0] = np.float32(log_ret.iloc[-1] if not np.isnan(log_ret.iloc[-1]) else 0.0)
        if len(log_ret) >= 3:
            extra[1] = np.float32(log_ret.iloc[-3:].fillna(0.0).sum())

        last_close = float(close.iloc[-1])
        denom = last_close if last_close != 0 else 1.0
        extra[2] = np.float32((float(high.iloc[-1]) - float(low.iloc[-1])) / denom)

        pct = close.pct_change().replace([np.inf, -np.inf], np.nan)
        window = pct.iloc[-10:] if len(pct) >= 10 else pct
        extra[3] = np.float32(window.std(skipna=True) or 0.0)

        ema10 = close.ewm(span=10, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        denom_ema = float(ema50.iloc[-1]) if float(ema50.iloc[-1]) != 0 else 1.0
        extra[4] = np.float32((float(ema10.iloc[-1]) - float(ema50.iloc[-1])) / denom_ema)
    except Exception:
        return extra
    return np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
