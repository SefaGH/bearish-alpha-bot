from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityOutput:
    vol_rs_bps: pd.Series
    vol_gk_bps: pd.Series
    vol_yz_bps: pd.Series
    vol_atr_bps: pd.Series
    vol_std_bps: pd.Series


class VolatilityEstimators:
    @staticmethod
    def _sanitize_prices(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for col in cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
            else:
                s = pd.Series(index=df.index, dtype="float64")
            out[col] = s.mask(s <= 0)
        return out

    @staticmethod
    def _safe_bps(numer: pd.Series, denom: pd.Series) -> pd.Series:
        denom_s = denom.mask(denom <= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (numer / denom_s) * 10000.0
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def compute_all(df: pd.DataFrame, *, window: int, ddof: int = 1) -> VolatilityOutput:
        try:
            window = int(window)
        except Exception:
            window = 0
        try:
            ddof = int(ddof)
        except Exception:
            ddof = 1

        empty = VolatilityOutput(
            vol_rs_bps=pd.Series(index=df.index, dtype="float64"),
            vol_gk_bps=pd.Series(index=df.index, dtype="float64"),
            vol_yz_bps=pd.Series(index=df.index, dtype="float64"),
            vol_atr_bps=pd.Series(index=df.index, dtype="float64"),
            vol_std_bps=pd.Series(index=df.index, dtype="float64"),
        )

        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            return empty
        if window < 2:
            return empty
        if ddof < 0 or ddof >= window:
            return empty

        px = VolatilityEstimators._sanitize_prices(df, cols=("open", "high", "low", "close"))
        o, h, l, c = px["open"], px["high"], px["low"], px["close"]

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            log_ho = np.log(h / o)
            log_lo = np.log(l / o)
            log_hl = np.log(h / l)
            log_co = np.log(c / o)
            log_hc = np.log(h / c)
            log_lc = np.log(l / c)
            log_oc1 = np.log(o / c.shift(1))

        log_ho = log_ho.replace([np.inf, -np.inf], np.nan)
        log_lo = log_lo.replace([np.inf, -np.inf], np.nan)
        log_hl = log_hl.replace([np.inf, -np.inf], np.nan)
        log_co = log_co.replace([np.inf, -np.inf], np.nan)
        log_hc = log_hc.replace([np.inf, -np.inf], np.nan)
        log_lc = log_lc.replace([np.inf, -np.inf], np.nan)
        log_oc1 = log_oc1.replace([np.inf, -np.inf], np.nan)

        rs_var_raw = (log_ho * log_hc) + (log_lo * log_lc)
        rs_var = rs_var_raw.rolling(window=window, min_periods=window).mean()

        gk_var_raw = (0.5 * (log_hl * log_hl)) - ((2.0 * math.log(2.0) - 1.0) * (log_co * log_co))
        gk_var = gk_var_raw.rolling(window=window, min_periods=window).mean()

        o_var = log_oc1.rolling(window=window, min_periods=window).var(ddof=ddof)
        c_var = log_co.rolling(window=window, min_periods=window).var(ddof=ddof)

        k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
        yz_var = o_var + k * c_var + (1.0 - k) * rs_var

        rs_var = rs_var.clip(lower=0.0)
        gk_var = gk_var.clip(lower=0.0)
        yz_var = yz_var.clip(lower=0.0)

        vol_rs_bps = (np.sqrt(rs_var) * 10000.0).replace([np.inf, -np.inf], np.nan)
        vol_gk_bps = (np.sqrt(gk_var) * 10000.0).replace([np.inf, -np.inf], np.nan)
        vol_yz_bps = (np.sqrt(yz_var) * 10000.0).replace([np.inf, -np.inf], np.nan)

        atr = pd.to_numeric(df["atr"], errors="coerce") if "atr" in df.columns else pd.Series(index=df.index, dtype="float64")
        vwap_std = (
            pd.to_numeric(df["vwap_std"], errors="coerce") if "vwap_std" in df.columns else pd.Series(index=df.index, dtype="float64")
        )
        vol_atr_bps = VolatilityEstimators._safe_bps(atr, c)
        vol_std_bps = VolatilityEstimators._safe_bps(vwap_std, c)

        return VolatilityOutput(
            vol_rs_bps=vol_rs_bps,
            vol_gk_bps=vol_gk_bps,
            vol_yz_bps=vol_yz_bps,
            vol_atr_bps=vol_atr_bps,
            vol_std_bps=vol_std_bps,
        )
