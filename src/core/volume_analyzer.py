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

import ast
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence, List
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
        self.logger = logging.getLogger(__name__)
        self._mdp = market_data_pipeline
        # Merge user‑provided config with defaults; nested keys are not
        # deep merged intentionally.
        self.config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._readiness_emitted: Dict[str, int] = {}

        raw_buckets = self.config.get("buckets", [])
        self._buckets = self._normalize_buckets(raw_buckets)
        self.logger.info(
            "VolumeAnalyzer buckets normalized",
            extra={
                "event": "volume_analyzer_buckets_normalized",
                "bucket_count": len(self._buckets),
                "buckets": self._buckets,
            },
        )

    def _normalize_buckets(self, raw_buckets: Any) -> List[Dict[str, Any]]:
        """Normalize raw bucket config into [{'threshold': float, 'name': str}, ...]."""
        if not isinstance(raw_buckets, list) or not raw_buckets:
            self.logger.warning(
                "VolumeAnalyzer: invalid or empty buckets config: %r (type=%s)",
                raw_buckets,
                type(raw_buckets).__name__,
            )
            return []

        normalized: List[Dict[str, Any]] = []

        for entry in raw_buckets:
            try:
                # Handle stringified entries produced by the minimal YAML parser
                if isinstance(entry, str):
                    try:
                        entry = ast.literal_eval(entry.strip())
                    except (ValueError, SyntaxError) as exc:
                        self.logger.warning(
                            "VolumeAnalyzer: failed to parse string bucket entry %r: %s",
                            entry,
                            exc,
                        )
                        continue

                if isinstance(entry, dict):
                    threshold = entry.get("threshold")
                    name = entry.get("name")
                    if threshold is None or name is None:
                        self.logger.warning(
                            "VolumeAnalyzer: skipping incomplete dict bucket: %r",
                            entry,
                        )
                        continue
                    normalized.append({"threshold": float(threshold), "name": str(name)})
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    threshold, name = entry[0], entry[1]
                    normalized.append({"threshold": float(threshold), "name": str(name)})
                else:
                    self.logger.warning(
                        "VolumeAnalyzer: skipping invalid bucket entry: %r (type=%s)",
                        entry,
                        type(entry).__name__,
                    )
            except Exception as exc:  # Defensive guardrail to keep normalization resilient
                self.logger.error(
                    "VolumeAnalyzer: error processing bucket entry %r: %s",
                    entry,
                    exc,
                )

        normalized = sorted(normalized, key=lambda b: b["threshold"])

        if not normalized:
            self.logger.error(
                "VolumeAnalyzer: no valid buckets after normalization; volume context will be disabled.",
            )

        return normalized

    def _log_readiness(
        self,
        symbol: str,
        trade_tf: str,
        short_tf: str,
        medium_tf: str,
        trade_len: int,
        short_len: int,
        medium_len: int,
        window_bars: int,
        short_lb: int,
        med_lb: int,
        will_return: bool,
        trade_source: Optional[str] = None,
        short_source: Optional[str] = None,
        medium_source: Optional[str] = None,
        trade_limit: Optional[int] = None,
        short_limit: Optional[int] = None,
        medium_limit: Optional[int] = None,
    ) -> None:
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
            "trade_source": trade_source,
            "short_source": short_source,
            "medium_source": medium_source,
            "trade_limit": trade_limit,
            "short_limit": short_limit,
            "medium_limit": medium_limit,
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

    @staticmethod
    def _normalize_ts_seconds(ts: Optional[float]) -> Optional[float]:
        """Normalize epoch timestamps to UTC seconds (accepts ms or seconds)."""
        if ts is None:
            return None
        try:
            value = float(ts)
        except Exception:
            return None
        return value / 1000.0 if value > 10_000_000_000 else value

    @staticmethod
    def _extract_volume_series_with_ts(df: Any) -> tuple[List[float], Optional[float]]:
        """Extract volume series and last timestamp (UTC seconds) from an OHLCV buffer."""
        if df is None:
            return [], None

        # StreamDataCollector path: list of lists [[ts_ms, o, h, l, c, v], ...]
        if isinstance(df, list) and df and isinstance(df[0], (list, tuple)) and len(df[0]) >= 6:
            volumes: List[float] = []
            last_ts: Optional[float] = None
            for row in df:
                if not isinstance(row, (list, tuple)) or len(row) < 6:
                    continue
                try:
                    volumes.append(float(row[5]))
                except Exception:
                    continue
                try:
                    last_ts = float(row[0])
                except Exception:
                    pass
            return volumes, VolumeAnalyzer._normalize_ts_seconds(last_ts)

        # Pandas DataFrame path
        try:
            volumes = df["volume"].tolist()
            last_ts: Optional[float] = None
            ts_col = None
            for col in ("timestamp", "ts", "time"):
                if col in getattr(df, "columns", ()):
                    ts_col = col
                    break
            if ts_col:
                last_ts = float(df[ts_col].iloc[-1])
            else:
                try:
                    idx = df.index
                    if len(idx) > 0 and hasattr(idx[-1], "timestamp"):
                        last_ts = float(idx[-1].timestamp())
                except Exception:
                    pass
            return volumes, VolumeAnalyzer._normalize_ts_seconds(last_ts)
        except Exception:
            pass

        # Generic iterable of dict/object rows
        volumes: List[float] = []
        last_ts: Optional[float] = None
        for row in df:
            try:
                vol = getattr(row, "volume", row.get("volume"))
            except Exception:
                vol = None
            try:
                ts_candidate = getattr(row, "timestamp", None)
                if ts_candidate is None and isinstance(row, dict):
                    ts_candidate = row.get("timestamp") or row.get("ts") or row.get("time")
            except Exception:
                ts_candidate = None

            if vol is None:
                continue
            try:
                volumes.append(float(vol))
            except Exception:
                continue
            if ts_candidate is not None:
                try:
                    last_ts = float(ts_candidate)
                except Exception:
                    pass

        return volumes, VolumeAnalyzer._normalize_ts_seconds(last_ts)

    async def _get_latest_ohlcv_buffer(self, symbol: str, timeframe: str, limit: int) -> tuple[Any, str]:
        """
        Prefer reading raw OHLCV from StreamDataCollector to avoid MarketDataPipeline's
        indicator-driven limit (~255) and avoid recomputing indicators.

        Falls back to MarketDataPipeline.get_latest_ohlcv when the collector is unavailable.
        """
        try:
            ws_mgr = getattr(self._mdp, "websocket_manager", None)
            collector = getattr(ws_mgr, "collector", None) if ws_mgr else None
            if collector and hasattr(collector, "get_latest_ohlcv"):
                exchanges = getattr(self._mdp, "exchanges", None)
                exchange = None
                if isinstance(exchanges, dict) and exchanges:
                    exchange = next(iter(exchanges.keys()))
                exchange = exchange or getattr(self._mdp, "DEFAULT_EXCHANGE", None) or "bingx"
                result = collector.get_latest_ohlcv(exchange, symbol, timeframe, limit)
                if result is not None:
                    return result, "collector"
        except Exception:
            pass

        try:
            try:
                return await self._mdp.get_latest_ohlcv(symbol, timeframe, limit=limit), "pipeline"
            except TypeError:
                return await self._mdp.get_latest_ohlcv(symbol, timeframe), "pipeline"
        except Exception:
            return None, "none"

    async def _compute_baseline(self, symbol: str, baseline_tf: str, lookback: int) -> Optional[float]:
        """Compute median volume baseline for given timeframe."""
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
        shock_state: Optional[str] = None,
        include_forming_trade: bool = False,
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
        window_bars: int = int(cfg["window_bars"])

        # Pull raw series (and last timestamps) to measure availability for readiness logging.
        # NOTE: We need at least lookback+1 bars for baselines because we intentionally
        # drop the latest (potentially forming) bar from baseline calculation.
        trade_limit = max(1, window_bars)
        short_limit = max(2, short_lb + 1)
        medium_limit = max(2, med_lb + 1)

        df_short, short_source = await self._get_latest_ohlcv_buffer(symbol, short_tf, short_limit)
        df_medium, medium_source = await self._get_latest_ohlcv_buffer(symbol, medium_tf, medium_limit)
        df_trade, trade_source = await self._get_latest_ohlcv_buffer(symbol, trade_timeframe, trade_limit)

        series_short, ts_short_last = self._extract_volume_series_with_ts(df_short)
        series_medium, ts_medium_last = self._extract_volume_series_with_ts(df_medium)
        series_trade, ts_trade_last = self._extract_volume_series_with_ts(df_trade)

        # Baseline should exclude the latest (potentially forming) bar
        short_closed = series_short[:-1] if len(series_short) > 1 else []
        medium_closed = series_medium[:-1] if len(series_medium) > 1 else []

        short_baseline = statistics.median(short_closed[-short_lb:]) if short_closed else None
        medium_baseline = statistics.median(medium_closed[-med_lb:]) if medium_closed else None
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
                trade_source=trade_source,
                short_source=short_source,
                medium_source=medium_source,
                trade_limit=trade_limit,
                short_limit=short_limit,
                medium_limit=medium_limit,
            )
            return None

        tf_minutes = self._get_tf_minutes(trade_timeframe)
        if tf_minutes <= 0:
            return None
        short_tf_minutes = self._get_tf_minutes(short_tf)
        medium_tf_minutes = self._get_tf_minutes(medium_tf)
        if short_tf_minutes == 0 or medium_tf_minutes == 0:
            return None
        window_minutes = tf_minutes * window_bars
        short_baseline_scaled = short_baseline * (window_minutes / short_tf_minutes)
        medium_baseline_scaled = medium_baseline * (window_minutes / medium_tf_minutes)
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
        current_volume_closed = float(sum(current_window))

        forming_volume_raw: Optional[float] = None
        forming_volume_added = 0.0
        forming_elapsed_ratio: Optional[float] = None
        forming_update_age_ms: Optional[int] = None
        forming_open_ms: Optional[int] = None
        current_volume_mode = "closed_only"

        if include_forming_trade and str(shock_state or "").upper() == "ARMED":
            try:
                ws_mgr = getattr(self._mdp, "websocket_manager", None)
                collector = getattr(ws_mgr, "collector", None) if ws_mgr else None
                if collector and hasattr(collector, "get_forming_ohlcv"):
                    exchanges = getattr(self._mdp, "exchanges", None)
                    exchange = None
                    if isinstance(exchanges, dict) and exchanges:
                        exchange = next(iter(exchanges.keys()))
                    exchange = exchange or getattr(self._mdp, "DEFAULT_EXCHANGE", None) or "bingx"
                    forming = collector.get_forming_ohlcv(exchange, symbol, trade_timeframe)
                    if isinstance(forming, (list, tuple)) and len(forming) >= 6:
                        forming_open_ms = int(forming[0])
                        forming_volume_raw = float(forming[5])

                        now_ms = int(time.time() * 1000)
                        interval_ms = int(tf_minutes * 60 * 1000)
                        if interval_ms > 0:
                            forming_elapsed_ratio = max(0.0, min((now_ms - forming_open_ms) / interval_ms, 1.0))

                        try:
                            if hasattr(collector, "get_state"):
                                state = collector.get_state(exchange, symbol, trade_timeframe) or {}
                                last_update_ts = state.get("forming_last_update_ts")
                                if last_update_ts is not None:
                                    forming_update_age_ms = max(0, int(now_ms - int(last_update_ts)))
                        except Exception:
                            forming_update_age_ms = None

                        forming_stale_ms = cfg.get("forming_update_stale_ms")
                        try:
                            forming_stale_ms = int(forming_stale_ms) if forming_stale_ms is not None else 3000
                        except Exception:
                            forming_stale_ms = 3000

                        cap_ratio_raw = cfg.get("forming_volume_cap_ratio", 0.6)
                        try:
                            cap_ratio = float(cap_ratio_raw)
                        except Exception:
                            cap_ratio = 0.6
                        cap_ratio = max(0.0, min(cap_ratio, 1.0))

                        already_included = False
                        try:
                            if ts_trade_last is not None and forming_open_ms is not None:
                                last_ts_ms = int(float(ts_trade_last) * 1000)
                                if interval_ms > 0 and abs(last_ts_ms - int(forming_open_ms)) <= (interval_ms // 2):
                                    already_included = True
                        except Exception:
                            already_included = False

                        fresh_ok = (
                            forming_update_age_ms is None
                            or forming_stale_ms <= 0
                            or int(forming_update_age_ms) <= int(forming_stale_ms)
                        )
                        if not already_included and fresh_ok and forming_elapsed_ratio is not None:
                            forming_volume_added = float(forming_volume_raw or 0.0) * min(float(forming_elapsed_ratio), cap_ratio)
                            current_volume_mode = "closed_plus_forming_numerator"
            except Exception:
                pass

        current_volume = float(current_volume_closed + forming_volume_added)

        min_r = float(cfg.get("min_ratio", 0.1))
        max_r = float(cfg.get("max_ratio", 10.0))
        raw_ratio_short = current_volume / short_baseline_scaled if short_baseline_scaled > 0 else 1.0
        raw_ratio_medium = current_volume / medium_baseline_scaled if medium_baseline_scaled > 0 else 1.0
        ratio_short = max(min(raw_ratio_short, max_r), min_r)
        ratio_medium = max(min(raw_ratio_medium, max_r), min_r)

        if raw_ratio_short >= max_r or raw_ratio_medium >= max_r:
            self.logger.warning(
                "VolumeAnalyzer ratio cap hit",
                extra={
                    "event": "volume_ratio_cap",
                    "symbol": symbol,
                    "ratio_short_raw": raw_ratio_short,
                    "ratio_medium_raw": raw_ratio_medium,
                    "ratio_short_clipped": ratio_short,
                    "ratio_medium_clipped": ratio_medium,
                    "current_volume": current_volume,
                    "short_baseline_scaled": short_baseline_scaled,
                    "medium_baseline_scaled": medium_baseline_scaled,
                    "window_bars": window_bars,
                    "max_ratio": max_r,
                    "ratio_clip": {
                        "short": raw_ratio_short >= max_r,
                        "medium": raw_ratio_medium >= max_r,
                        "raw_short": raw_ratio_short,
                        "raw_medium": raw_ratio_medium,
                        "cap": max_r,
                    },
                },
            )

        w_short = float(cfg["weight_short"])
        w_med = float(cfg["weight_medium"])
        ratio_combined = (w_short * ratio_short) + (w_med * ratio_medium)

        alpha = float(cfg["sigmoid_alpha"])
        x = alpha * (ratio_combined - 1.0)
        volume_strength = 1.0 / (1.0 + math.exp(-x))

        if not self._buckets:
            self.logger.warning(
                "volume_analyzer.buckets config is invalid or empty; skipping volume context for %s",
                symbol,
            )
            return None

        bucket_name = self._buckets[0]["name"]
        for bucket in self._buckets:
            try:
                threshold = float(bucket["threshold"])
            except (KeyError, TypeError, ValueError):
                self.logger.warning(
                    "VolumeAnalyzer: skipping malformed bucket entry in compute_context: %r",
                    bucket,
                )
                continue

            if volume_strength >= threshold:
                bucket_name = bucket.get("name", bucket_name)

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

        # Baseline metadata for telemetry (not part of dataclass fields but attached dynamically)
        ctx.baseline_short_last_bar_ts = ts_short_last
        ctx.baseline_medium_last_bar_ts = ts_medium_last
        ctx.baseline_calc_mode = "closed_only"  # last (forming) bar excluded from baseline
        ctx.current_window_volume_closed = current_volume_closed
        ctx.current_window_volume_mode = current_volume_mode
        ctx.forming_volume_raw = forming_volume_raw
        ctx.forming_volume_added = forming_volume_added
        ctx.forming_elapsed_ratio = forming_elapsed_ratio
        ctx.forming_update_age_ms = forming_update_age_ms
        ctx.forming_open_ms = forming_open_ms
        ctx.shock_state = shock_state
        ctx.volume_data_sources = {"trade": trade_source, "short": short_source, "medium": medium_source}
        ctx.volume_data_limits = {"trade": trade_limit, "short": short_limit, "medium": medium_limit}

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
            trade_source=trade_source,
            short_source=short_source,
            medium_source=medium_source,
            trade_limit=trade_limit,
            short_limit=short_limit,
            medium_limit=medium_limit,
        )

        if include_forming_trade:
            try:
                dbg = {
                    "event": "volume_analyzer_debug",
                    "run_id": get_current_run_id(),
                    "symbol": symbol,
                    "trade_timeframe": trade_timeframe,
                    "shock_state": shock_state,
                    "current_window_volume_closed": current_volume_closed,
                    "forming_volume_raw": forming_volume_raw,
                    "forming_elapsed_ratio": forming_elapsed_ratio,
                    "forming_volume_added": forming_volume_added,
                    "current_window_volume_with_forming": current_volume,
                    "short_baseline_volume": short_baseline_scaled,
                    "medium_baseline_volume": medium_baseline_scaled,
                    "ratio_short": ratio_short,
                    "ratio_medium": ratio_medium,
                    "ratio_combined": ratio_combined,
                    "volume_strength": volume_strength,
                    "volume_bucket": bucket_name,
                    "window_bars": window_bars,
                    "trade_bars_available": len(series_trade),
                    "trade_source": trade_source,
                    "forming_update_age_ms": forming_update_age_ms,
                    "forming_open_ms": forming_open_ms,
                    "mode": current_volume_mode,
                }
                self.logger.info("volume_analyzer_debug %s", json.dumps(dbg, separators=(",", ":")))
            except Exception:
                pass

        return ctx
