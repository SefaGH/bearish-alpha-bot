import math
import logging
from typing import Optional, Dict, Any

import pandas as pd

from .base_strategy import BaseStrategy

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
        self.adx_threshold = float(cfg.get("adx_threshold", 30))
        self.min_rr_ratio = float(cfg.get("min_rr_ratio", 1.0))
        # Increase baseline so VWAP window (1440) + signal buffer can coexist
        self.min_rows = int(cfg.get("min_rows", 2100))
        self.min_signal_rows = int(cfg.get("min_signal_rows", 50))
        # Interface guard
        if not hasattr(self, "signal"):
            self.signal = self._default_signal_wrapper  # type: ignore
        assert callable(getattr(self, "signal", None)), "MeanReversion: signal method not callable"
        print("MeanReversion: signal method bound successfully")

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
            return None

        if df_vwap is None or df_vwap.empty or df_sig is None or df_sig.empty:
            logger.warning(f"[MeanReversion] Data missing: vwap_empty={df_vwap is None or df_vwap.empty} "
                           f"sig_empty={df_sig is None or df_sig.empty} for {symbol}")
            return None

        # Ensure time order and required columns for VWAP math
        if not df_vwap.index.is_monotonic_increasing:
            df_vwap = df_vwap.sort_index()
        if not df_sig.index.is_monotonic_increasing:
            df_sig = df_sig.sort_index()
        if "volume" not in df_vwap.columns:
            logger.warning(f"[MeanReversion] Missing volume column in VWAP dataframe for {symbol}")
            return None

        logger.info(f"[MeanReversion] Data rows: vwap={len(df_vwap)}, signal_tf={len(df_sig)}, "
                    f"min_vwap={self.min_rows}, min_signal={self.min_signal_rows}")
        if self.market_data_pipeline and (len(df_vwap) < self.min_rows or len(df_sig) < self.min_signal_rows):
            try:
                vwap_limit = max(self.min_rows, 2100)
                refreshed_vwap = await self.market_data_pipeline.get_latest_ohlcv(
                    symbol=symbol, timeframe=self.vwap_tf, limit=vwap_limit
                )
                if refreshed_vwap is not None and not refreshed_vwap.empty:
                    df_vwap = refreshed_vwap
                sig_limit = max(self.min_signal_rows, 1000)
                refreshed_sig = await self.market_data_pipeline.get_latest_ohlcv(
                    symbol=symbol, timeframe=self.signal_tf, limit=sig_limit
                )
                if refreshed_sig is not None and not refreshed_sig.empty:
                    df_sig = refreshed_sig
                logger.info(f"[MeanReversion] Refreshed from pipeline: vwap={len(df_vwap)}, sig={len(df_sig)}")
            except Exception as e:
                logger.warning(f"[MeanReversion] Pipeline refresh failed: {e}")
        if len(df_vwap) < self.min_rows:
            logger.warning(f"[MeanReversion] VWAP data insufficient. Have vwap={len(df_vwap)}, "
                           f"Need>={self.min_rows}. Aborting.")
            return None
        if len(df_sig) < self.min_signal_rows:
            logger.warning(f"[MeanReversion] Signal data insufficient. Have sig={len(df_sig)}, "
                           f"Need>={self.min_signal_rows}. Aborting.")
            return None

        clean_vwap = df_vwap.dropna()
        clean_sig = df_sig.dropna()
        if clean_vwap.empty:
            logger.warning(f"[MeanReversion] Indicator calculation resulted in empty VWAP dataframe after dropna "
                           f"(input rows={len(df_vwap)}).")
            logger.debug(f"[MeanReversion] VWAP columns: {list(df_vwap.columns)}")
            return None
        if clean_sig.empty:
            logger.warning(f"[MeanReversion] Indicator calculation resulted in empty signal dataframe after dropna "
                           f"(input rows={len(df_sig)}).")
            logger.debug(f"[MeanReversion] Signal columns: {list(df_sig.columns)}")
            return None

        last_vwap = clean_vwap.iloc[-1]
        last_sig = clean_sig.iloc[-1]

        required_cols_vwap = {"vwap", "vwap_lower", "vwap_upper"}
        if not required_cols_vwap.issubset(set(last_vwap.index)):
            logger.warning(f"[MeanReversion] Missing required VWAP columns for {symbol}: "
                           f"{required_cols_vwap - set(last_vwap.index)}")
            return None
        if "adx" not in last_sig.index:
            logger.warning(f"[MeanReversion] Missing ADX column for {symbol}")
            return None

        price = float(last_sig["close"])
        vwap_main = float(last_vwap["vwap"])
        vwap_lower = float(last_vwap["vwap_lower"])
        vwap_upper = float(last_vwap["vwap_upper"])
        adx_val = float(last_sig["adx"])

        if math.isnan(vwap_main) or math.isnan(vwap_lower) or math.isnan(vwap_upper) or math.isnan(adx_val):
            logger.warning(f"[MeanReversion] NaN detected in indicators for {symbol}: "
                           f"vwap={vwap_main}, lower={vwap_lower}, upper={vwap_upper}, adx={adx_val}")
            return None

        atr_val = float(last_sig["atr"]) if "atr" in last_sig.index else None

        in_band = vwap_lower <= price <= vwap_upper
        adx_ok = adx_val < self.adx_threshold

        if in_band:
            logger.info(
                f"[MeanReversion] Price within bands for {symbol}. "
                f"px={price:.4f}, lower={vwap_lower:.4f}, upper={vwap_upper:.4f}, "
                f"adx={adx_val:.1f}, adx_th={self.adx_threshold:.1f}"
            )
            return None

        if not adx_ok:
            if price > vwap_upper:
                breach = "above_upper"
            elif price < vwap_lower:
                breach = "below_lower"
            else:
                breach = "outside"
            logger.info(
                f"[MeanReversion] Price outside bands but ADX veto for {symbol}. "
                f"breach={breach}, px={price:.4f}, lower={vwap_lower:.4f}, upper={vwap_upper:.4f}, "
                f"adx={adx_val:.1f}, adx_th={self.adx_threshold:.1f}"
            )
            return None

        if price < vwap_lower:
            side = "buy"
            reason = (
                f"VWAP MR long (px {price:.4f} < lower {vwap_lower:.4f}, "
                f"ADX {adx_val:.1f} < {self.adx_threshold})"
            )
        elif price > vwap_upper:
            side = "sell"
            reason = (
                f"VWAP MR short (px {price:.4f} > upper {vwap_upper:.4f}, "
                f"ADX {adx_val:.1f} < {self.adx_threshold})"
            )
        else:
            logger.info(
                f"[MeanReversion] Price within bands for {symbol}. "
                f"px={price:.4f}, lower={vwap_lower:.4f}, upper={vwap_upper:.4f}, "
                f"adx={adx_val:.1f}, adx_th={self.adx_threshold:.1f}"
            )
            return None

        stop = None
        if atr_val and not math.isnan(atr_val):
            if side == "buy":
                stop = price - atr_val * 1.5
            else:
                stop = price + atr_val * 1.5

        target = vwap_main

        signal = {
            "strategy_name": self.strategy_name,
            "symbol": symbol,
            "side": side,
            "timeframe": self.signal_tf,
            "entry": price,
            "stop": stop,
            "target": target,
            "reason": reason,
            "signal_type": "MEAN_REVERSION",
            "tp_mode": "DYNAMIC",
            "min_rr_ratio": self.min_rr_ratio,
            "vwap": vwap_main,
            "vwap_lower": vwap_lower,
            "vwap_upper": vwap_upper,
            "adx": adx_val,
        }

        return signal
