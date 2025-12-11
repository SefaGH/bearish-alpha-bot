"""
Core volume analysis utilities for Bearish Alpha Bot.

This module provides a ``VolumeAnalyzer`` class which produces a
``VolumeContext`` containing a dynamic volume index for use across
strategies, risk rules and other components. It is intended as a
centralised, reusable service for computing relative volume strength
based on recent and historical trading activity.

The implementation is deliberately lightweight and self‑contained. The
``VolumeAnalyzer`` relies on ``MarketDataPipeline`` to obtain OHLCV
series for different timeframes and performs simple statistics on the
volume column. The goal is to offer an extendable starting point; more
advanced features like outlier filtering, pattern detection or
exchange‑specific adjustments can be added later.

Lifecycle integration: ``StrategyCoordinator`` calls
``compute_context`` once per signal (when enabled) and enriches the
signal with ``volume_strength`` and ``volume_bucket``. These fields are
then consumed by central bucket gating and ``VolumeAwarePositionSizingRule``
to adjust acceptance, quality boosts, and risk multipliers.

Key concepts:

* Short baseline – median of the last ``short_lookback`` bars on the
  ``baseline_short_tf`` timeframe (e.g. 1h). This represents
  short‑term typical volume.
* Medium baseline – median of the last ``medium_lookback`` bars on
  ``baseline_medium_tf`` timeframe (e.g. 4h). This represents
  medium‑term typical volume.
* Current window volume – sum of ``window_bars`` bars of the trade
  timeframe (e.g. last 3 bars on the 5m chart). This smooths out
  noise in the latest volume read.
* Ratios – current volume divided by each baseline, clipped to a
  reasonable range. The ratios are combined with configurable
  weights.
* Volume strength – a sigmoid transformation of the combined ratio
  into the range [0, 1]. Values below 0.5 represent low relative
  volume, while values above 0.5 represent higher than normal
  activity.
* Bucket – a categorical label derived from the volume strength for
  coarse‑grained decisions (LOW, NORMAL, HIGH, EXTREME).

This module does not handle caching by default; callers can wrap
``compute_context`` in their own caching mechanism if desired.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence
import logging

from core.logger import get_current_run_id


@dataclass
class VolumeContext:
    """Structured results produced by :class:`VolumeAnalyzer`.

    Attributes
    ----------
    symbol : str
        Trading symbol for which the context was computed.
    trade_timeframe : str
        The timeframe (e.g. ``"5m"``) for which the current window
        volume was calculated. This corresponds to the timeframe of the
        signal or strategy using the volume context.
    current_window_volume : float
        Sum of volumes over the most recent ``window_bars`` bars on the
        trade timeframe.
    short_baseline_volume : float
        The baseline volume over the short baseline timeframe,
        rescaled to the trade timeframe.
    medium_baseline_volume : float
        The baseline volume over the medium baseline timeframe,
        rescaled to the trade timeframe.
    ratio_short : float
        Ratio of current volume to the short baseline. Values > 1
        indicate above‑normal recent activity.
    ratio_medium : float
        Ratio of current volume to the medium baseline. Values > 1
        indicate above‑normal longer‑term activity.
    ratio_combined : float
        Weighted combination of ``ratio_short`` and ``ratio_medium``.
    volume_strength : float
        Sigmoid transformation of ``ratio_combined`` into [0, 1].
    bucket : str
        Discrete label representing the volume regime; one of
        ``"LOW"``, ``"NORMAL"``, ``"HIGH"``, or ``"EXTREME"``.
    last_updated_ts : float
        Timestamp (UTC seconds) when the context was computed.
    """

    symbol: str
    trade_timeframe: str
    current_window_volume: float
    short_baseline_volume: float
    medium_baseline_volume: float
    ratio_short: float
    ratio_medium: float
    ratio_combined: float
    volume_strength: float
    bucket: str
    last_updated_ts: float


class VolumeAnalyzer:
    """Compute dynamic volume strength indices for trading symbols.

    Parameters
    ----------
    market_data_pipeline : object
        An object providing access to OHLCV data buffers. It must
        expose a method ``get_latest_ohlcv(symbol, timeframe) ->
        pandas.DataFrame`` or similar which returns OHLCV rows with a
        ``volume`` column. The implementation is agnostic to the
        specific type as long as the returned value behaves like a
        sequence of dicts or objects with attribute/keys for ``volume``.
    config : dict
        Configuration options controlling lookback lengths, weights and
        thresholds. See the default options in ``DEFAULT_CONFIG`` for
        expected keys.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        # Timeframes used for baseline computation. These should exist
        # as WebSocket buffers in MarketDataPipeline.
        "baseline_short_tf": "1h",
        "baseline_medium_tf": "4h",
        # Lookback window sizes (in number of bars) for baseline.
        "short_lookback": 168,  # 7 days on 1h timeframe
        "medium_lookback": 180,  # 30 days on 4h timeframe
        # Number of bars of the trade timeframe to sum for current volume.
        "window_bars": 3,
        # Weights for combining ratios: 0.6 for short, 0.4 for medium.
        "weight_short": 0.6,
        "weight_medium": 0.4,
        # Sigmoid slope controlling sensitivity. Higher values make
        # volume_strength rise faster with ratio_combined.
        "sigmoid_alpha": 1.2,
        # Bucketing thresholds for volume_strength (inclusive lower
        # bounds). The buckets must appear in ascending order of
        # thresholds.
        "buckets": [
            (0.0, "LOW"),
            (0.3, "NORMAL"),
            (0.6, "HIGH"),
            (0.85, "EXTREME"),
        ],
        # Minimum and maximum ratio values to clip to, avoiding
        # overflows and division by small numbers.
        "min_ratio": 0.1,
        "max_ratio": 10.0,
    }

    def __init__(self, market_data_pipeline: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self._mdp = market_data_pipeline
        # Merge user‑provided config with defaults; nested keys are not
        # deep merged intentionally.
        self.config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._readiness_emitted: Dict[str, int] = {}

    def _log_readiness(self, symbol: str, trade_tf: str, short_tf: str, medium_tf: str,
                       trade_len: int, short_len: int, medium_len: int,
                       window_bars: int, short_lb: int, med_lb: int,
                       will_return: bool) -> None:
        """Emit a limited readiness log to show bar availability."""
        key = f"{symbol}:{trade_tf}:{short_tf}:{medium_tf}"
        count = self._readiness_emitted.get(key, 0)
        if count >= 5:
            return  # avoid log spam
        self._readiness_emitted[key] = count + 1

        payload = {
            "event": "volume_analyzer_readiness",
            "run_id": get_current_run_id(),
            "symbol": symbol,
            "trade_timeframe": trade_tf,
            "baseline_short_tf": short_tf,
            "baseline_medium_tf": medium_tf,
            "trade_bars_available": trade_len,
            "required_trade_bars": window_bars,
            "short_bars_available": short_len,
            "required_short_bars": short_lb,
            "medium_bars_available": medium_len,
            "required_medium_bars": med_lb,
            "ready": trade_len >= window_bars and short_len >= short_lb and medium_len >= med_lb,
            "will_return_context": will_return,
        }
        logging.getLogger(__name__).info(payload)

    async def _get_volume_series(self, symbol: str, timeframe: str) -> Sequence[float]:
        """Extract a series of volumes for the given symbol/timeframe.

        This helper awaits the market data pipeline to obtain the
        latest OHLCV data and extracts the ``volume`` column. It
        supports both pandas DataFrame and list of dict outputs.

        Returns a sequence of floats representing volumes. The
        sequence is ordered from oldest to newest.
        """
        df = await self._mdp.get_latest_ohlcv(symbol, timeframe)
        if df is None:
            return []
        try:
            # DataFrame path
            return df["volume"].tolist()
        except Exception:
            volumes = []
            for row in df:
                try:
                    vol = getattr(row, "volume", row.get("volume"))
                except Exception:
                    vol = None
                if vol is not None:
                    volumes.append(float(vol))
            return volumes

    async def _compute_baseline(self, symbol: str, baseline_tf: str, lookback: int) -> Optional[float]:
        """Compute median volume baseline for given timeframe.

        If there are fewer than ``lookback`` bars available, uses all
        available bars. Returns None if no data is available.
        """
        series = await self._get_volume_series(symbol, baseline_tf)
        if not series:
            return None
        recent = series[-lookback:] if len(series) >= lookback else series
        if not recent:
            return None
        return statistics.median(recent)

    def _get_tf_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to number of minutes.

        Accepts strings like "1m", "5m", "15m", "30m", "1h", "4h", etc.
        Returns 0 for unknown formats.
        """
        try:
            if timeframe.endswith("m"):
                return int(timeframe[:-1])
            if timeframe.endswith("h"):
                return int(timeframe[:-1]) * 60
            if timeframe.endswith("d"):
                return int(timeframe[:-1]) * 60 * 24
        except Exception:
            pass
        return 0

    async def compute_context(
        self,
        symbol: str,
        trade_timeframe: str,
        as_of_ts: Optional[float] = None,
    ) -> Optional[VolumeContext]:
        """Compute a :class:`VolumeContext` for the given symbol/timeframe.

        Parameters
        ----------
        symbol : str
            Trading symbol, e.g. ``"BTC/USDT"``. Must be supported by
            the underlying market data pipeline.
        trade_timeframe : str
            Timeframe of the signal/strategy that will consume the
            resulting context (e.g. ``"5m"`` or ``"1h"``). This
            determines how current volume is aggregated.
        as_of_ts : float, optional
            Timestamp for logging; not used in calculation in this
            minimal implementation.

        Returns
        -------
        VolumeContext or None
            VolumeContext containing computed metrics, or None if data
            unavailable.
        """
        cfg = self.config
        short_tf: str = cfg["baseline_short_tf"]
        medium_tf: str = cfg["baseline_medium_tf"]
        short_lb: int = int(cfg["short_lookback"])
        med_lb: int = int(cfg["medium_lookback"])

        # Pull raw series first to measure availability for readiness logging
        series_short = await self._get_volume_series(symbol, short_tf)
        series_medium = await self._get_volume_series(symbol, medium_tf)
        series_trade = await self._get_volume_series(symbol, trade_timeframe)

        short_baseline = statistics.median(series_short[-short_lb:]) if series_short else None
        medium_baseline = statistics.median(series_medium[-med_lb:]) if series_medium else None
        if short_baseline is None or medium_baseline is None:
            self._log_readiness(
                symbol,
                trade_timeframe,
                short_tf,
                medium_tf,
                len(series_trade),
                len(series_short),
                len(series_medium),
                int(cfg["window_bars"]),
                short_lb,
                med_lb,
                False,
            )
            return None

        tf_minutes = self._get_tf_minutes(trade_timeframe)
        if tf_minutes <= 0:
            return None
        short_tf_minutes = self._get_tf_minutes(short_tf)
        medium_tf_minutes = self._get_tf_minutes(medium_tf)
        if short_tf_minutes == 0 or medium_tf_minutes == 0:
            return None
        short_baseline_scaled = short_baseline * (tf_minutes / short_tf_minutes)
        medium_baseline_scaled = medium_baseline * (tf_minutes / medium_tf_minutes)

        window_bars: int = int(cfg["window_bars"])
        if not series_trade:
            self._log_readiness(
                symbol,
                trade_timeframe,
                short_tf,
                medium_tf,
                len(series_trade),
                len(series_short),
                len(series_medium),
                window_bars,
                short_lb,
                med_lb,
                False,
            )
            return None
        current_window = series_trade[-window_bars:] if len(series_trade) >= window_bars else series_trade
        if not current_window:
            return None
        current_volume = float(sum(current_window))

        min_r = float(cfg.get("min_ratio", 0.1))
        max_r = float(cfg.get("max_ratio", 10.0))
        ratio_short = current_volume / short_baseline_scaled if short_baseline_scaled > 0 else 1.0
        ratio_medium = current_volume / medium_baseline_scaled if medium_baseline_scaled > 0 else 1.0
        ratio_short = max(min(ratio_short, max_r), min_r)
        ratio_medium = max(min(ratio_medium, max_r), min_r)

        w_short = float(cfg["weight_short"])
        w_med = float(cfg["weight_medium"])
        ratio_combined = (w_short * ratio_short) + (w_med * ratio_medium)

        alpha = float(cfg["sigmoid_alpha"])
        x = alpha * (ratio_combined - 1.0)
        volume_strength = 1.0 / (1.0 + math.exp(-x))

        bucket_name = cfg["buckets"][0][1]
        for threshold, name in cfg["buckets"]:
            if volume_strength >= threshold:
                bucket_name = name
            else:
                break

        ctx = VolumeContext(
            symbol=symbol,
            trade_timeframe=trade_timeframe,
            current_window_volume=current_volume,
            short_baseline_volume=short_baseline_scaled,
            medium_baseline_volume=medium_baseline_scaled,
            ratio_short=ratio_short,
            ratio_medium=ratio_medium,
            ratio_combined=ratio_combined,
            volume_strength=volume_strength,
            bucket=bucket_name,
            last_updated_ts=as_of_ts or 0.0,
        )

        self._log_readiness(
            symbol,
            trade_timeframe,
            short_tf,
            medium_tf,
            len(series_trade),
            len(series_short),
            len(series_medium),
            window_bars,
            short_lb,
            med_lb,
            True,
        )

        return ctx
