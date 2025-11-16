"""
Advanced Price Prediction Module for Phase 4 Final.

Manifest-driven loader for GEMMA TorchScript price models with fallback heuristics.
"""

import ast
import asyncio
import logging
import os
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401  # Imported to detect availability only

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False

from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
from src.ml.manifest_manager import ManifestManager

logger = logging.getLogger(__name__)
class AdvancedPricePredictionEngine:
    """Manifest-driven price prediction engine backed by GEMMA TorchScript."""

    def __init__(self, market_data_pipeline, feature_pipeline, config: Dict[str, Any]):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AdvancedPricePredictionEngine.")

        self.market_data_pipeline = market_data_pipeline
        self.feature_pipeline = feature_pipeline
        self.config = config or {}
        self.primary_timeframe = self._determine_primary_timeframe()
        self.prediction_scale_pct = float(self.config.get('classification_to_pct_scale', 1.2))
        self.update_interval = self.config.get('update_interval_seconds', 60)
        self.cache_ttl = timedelta(seconds=self.config.get('cache_ttl_seconds', 300))
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
        self.data_buffers: Dict[str, Dict[str, deque]] = {}
        self.is_running = False
        self.project_root = Path(__file__).resolve().parents[2]

        self.manifest_mgr = ManifestManager()
        self.bundle_path = self._resolve_bundle_path()
        self.manifest = self._load_manifest()
        self.adapter = self._initialize_adapter()
        self.is_trained = self.adapter is not None and getattr(self.adapter, 'model', None) is not None

        status_summary = self.get_status_summary()
        logger.info(f"🤖 PricePredictor Status: {status_summary}")

        if not self.is_trained:
            logger.warning("⚠️ PricePredictor running in FALLBACK mode - predictions based on technical analysis only")

        if not self.market_data_pipeline:
            logger.warning("⚠️ MarketDataPipeline not provided. Prediction updates may fail.")

        logger.info("Advanced Price Prediction Engine initialized.")

    def _parse_timeframes_config(self, raw_config: Any) -> List[str]:
        """Returns a flat list of timeframe strings from arbitrary config input."""

        parsed: List[str] = []

        def _collect(value: Any) -> None:
            if value is None:
                return

            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return
                if text.startswith('[') and text.endswith(']'):
                    try:
                        literal = ast.literal_eval(text)
                        _collect(literal)
                        return
                    except (ValueError, SyntaxError):
                        pass
                tokens = text.split(',')
                for token in tokens:
                    cleaned = token.strip(" []'\"")
                    if cleaned:
                        parsed.append(cleaned)
                return

            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _collect(item)
                return

            cleaned = str(value).strip(" []'\"")
            if cleaned:
                parsed.append(cleaned)

        _collect(raw_config)
        return parsed

    def _normalize_timeframe_value(self, value: Any) -> str:
        parsed = self._parse_timeframes_config(value)
        return parsed[0] if parsed else '15m'

    def _determine_primary_timeframe(self) -> str:
        timeframes_cfg = self.config.get('timeframes') or ['15m']
        return self._normalize_timeframe_value(timeframes_cfg)

    def _current_timestamp(self) -> pd.Timestamp:
        """Returns a timezone-aware UTC timestamp for cache bookkeeping."""

        return pd.Timestamp.now(tz='UTC')

    def _ensure_utc_timestamp(self, value: Any) -> pd.Timestamp:
        """Normalizes arbitrary timestamp inputs to timezone-aware UTC."""

        ts = value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize('UTC')
        try:
            return ts.tz_convert('UTC')
        except TypeError:
            return ts

    def _resolve_bundle_path(self) -> Path:
        candidates: List[str] = []
        if isinstance(self.config.get('active_bundle'), str):
            candidates.append(self.config['active_bundle'])

        pipeline_models_cfg = getattr(self.feature_pipeline, 'models_config', None)
        if isinstance(pipeline_models_cfg, dict):
            resolved = pipeline_models_cfg.get('active_bundle')
            if resolved:
                candidates.append(resolved)

        env_bundle = os.environ.get('ACTIVE_MODEL_BUNDLE')
        if env_bundle:
            candidates.append(env_bundle)

        if not candidates:
            candidates.append('artifacts/legacy')

        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                return path

        return Path(candidates[0])

    def _load_manifest(self) -> Dict[str, Any]:
        bundle = str(self.bundle_path)
        try:
            manifest = self.manifest_mgr.load_manifest(bundle)
            logger.info(
                "✅ Manifest loaded for price predictor | version=%s | feature_count=%s",
                manifest.get('version'),
                manifest.get('feature_count'),
            )
            return manifest
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load manifest from %s: %s", bundle, exc)
            return {
                'version': 'fallback-legacy',
                'feature_count': 42,
                'feature_names_ordered': [f'feature_{i}' for i in range(42)],
                'selected_features_price': list(range(42)),
                'selected_features_regime': list(range(42)),
                'rl_state_size': 42,
            }

    def _resolve_manifest_path(self, relative_path: Optional[str], prefer_bundle: bool = True) -> Optional[Path]:
        if not relative_path:
            return None
        path = Path(relative_path)
        if path.is_absolute():
            return path

        candidates: List[Path] = []
        if prefer_bundle and self.bundle_path:
            candidates.append(Path(self.bundle_path) / relative_path)
        candidates.append(self.project_root / relative_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0] if candidates else None

    def _initialize_adapter(self) -> Optional[GemmaTorchScriptAdapter]:
        model_rel = (
            self.manifest.get('price_model_path')
            or self.manifest.get('gemma_price_model_path')
        )
        model_path = self._resolve_manifest_path(model_rel)
        if not model_path or not model_path.exists():
            logger.warning(
                "🧠 [PRICE-ENGINE] Manifest model path missing (%s). GEMMA adapter disabled.",
                model_rel,
            )
            return None

        scaler_rel = self.manifest.get('price_scaler_path') or self.manifest.get('gemma_price_scaler_path')
        scaler_path = self._resolve_manifest_path(scaler_rel)
        if not scaler_path or not scaler_path.exists():
            logger.warning("🧠 [PRICE-ENGINE] Scaler path missing (%s). GEMMA adapter disabled.", scaler_rel)
            return None

        features_rel = self.manifest.get('active_features_path')
        features_path = self._resolve_manifest_path(features_rel, prefer_bundle=False)

        feature_mask_path = self._resolve_manifest_path(
            self.config.get('feature_mask_path') or self.manifest.get('feature_mask_path'),
            prefer_bundle=False,
        )

        adapter_config = {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'features_path': str(features_path) if features_path else None,
            'feature_mask_path': str(feature_mask_path) if feature_mask_path else None,
            'feature_count': self.manifest.get('feature_count'),
            'feature_names': self.manifest.get('feature_names_ordered'),
            'cache_ttl': self.config.get('cache_ttl_seconds', 300),
            'circuit_breaker': self.config.get('circuit_breaker', {}),
            'shadow_mode': self.config.get('shadow_mode', False),
        }

        try:
            return GemmaTorchScriptAdapter(adapter_config)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize GEMMA adapter: %s", exc, exc_info=True)
            return None

    def get_status_summary(self) -> str:
        if self.is_trained:
            bundle = str(self.bundle_path)
            return f"ML Mode - GEMMA adapter active ({self.primary_timeframe}) from {bundle}"
        return f"FALLBACK Mode - no manifest-backed model (timeframe={self.primary_timeframe})"

    def has_model_for(self, symbol: str) -> bool:
        model_exists = self.is_trained
        logger.debug("🧠 [PRICE-ENGINE] Model check for %s: %s", symbol, 'Exists' if model_exists else 'Not Found')
        return model_exists

    async def _fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.market_data_pipeline:
            logger.error("❌ MarketDataPipeline not available. Cannot fetch price data for %s.", symbol)
            return None
        timeframe = self._normalize_timeframe_value(self.primary_timeframe)

        try:
            df = await self.market_data_pipeline.get_latest_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                exchange=None,
            )
            if df is None or getattr(df, 'empty', True):
                logger.warning("⚠️ No OHLCV data for %s (%s timeframe).", symbol, timeframe)
                return None
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch OHLCV for %s: %s", symbol, exc, exc_info=True)
            return None

    def _extract_feature_snapshot(self, price_data: pd.DataFrame) -> Optional[Dict[str, float]]:
        if not self.feature_pipeline:
            logger.error("Feature pipeline unavailable; cannot extract features.")
            return None
        features = self.feature_pipeline.extract_features(price_data, mode='price')
        if features.empty:
            logger.warning("⚠️ Feature extraction produced empty frame for %s timeframe %s", 'price', self.primary_timeframe)
            return None
        try:
            return features.tail(1).iloc[0].to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to convert feature snapshot: %s", exc, exc_info=True)
            return None

    def _build_prediction_payload(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        adapter_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        close_price = float(price_data['close'].iloc[-1])
        probabilities = adapter_result.get('probabilities', [0.33, 0.34, 0.33])
        prob_bearish = float(probabilities[0]) if probabilities else 0.33
        prob_bullish = float(probabilities[-1]) if probabilities else 0.33
        consensus = abs(prob_bullish - prob_bearish)
        forecast_pct = (prob_bullish - prob_bearish) * self.prediction_scale_pct
        uncertainty = max(1e-3, 1.0 - float(adapter_result.get('price_confidence', 0.5)))

        forecast_arr = np.array([forecast_pct], dtype=float)
        uncertainty_arr = np.array([uncertainty], dtype=float)

        timeframe_block = {
            'forecast': forecast_arr,
            'uncertainty': uncertainty_arr,
            'consensus_strength': consensus,
            'current_price': close_price,
            'forecast_prices': close_price * (1 + forecast_arr / 100),
        }

        aggregated = {
            'forecast': forecast_arr,
            'uncertainty': uncertainty_arr,
            'consensus_strength': consensus,
        }

        return {
            'symbol': symbol,
            'by_timeframe': {self.primary_timeframe: timeframe_block},
            'aggregated': aggregated,
            'adapter': adapter_result,
            'timestamp': self._current_timestamp(),
            'mode': 'gemma',
        }

    def _build_fallback_prediction(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        close_series = price_data['close']
        if len(close_series) < 2:
            slope = 0.0
        else:
            window = min(len(close_series) - 1, self.config.get('fallback_window', 20))
            origin = float(close_series.iloc[-window - 1])
            slope = 0.0 if origin == 0 else (float(close_series.iloc[-1]) - origin) / origin
        forecast_pct = slope * 100
        uncertainty = np.array([1.0], dtype=float)
        forecast_arr = np.array([forecast_pct], dtype=float)
        close_price = float(close_series.iloc[-1])

        timeframe_block = {
            'forecast': forecast_arr,
            'uncertainty': uncertainty,
            'consensus_strength': 0.2,
            'current_price': close_price,
            'forecast_prices': close_price * (1 + forecast_arr / 100),
        }

        return {
            'symbol': symbol,
            'by_timeframe': {self.primary_timeframe: timeframe_block},
            'aggregated': {
                'forecast': forecast_arr,
                'uncertainty': uncertainty,
                'consensus_strength': 0.2,
            },
            'timestamp': self._current_timestamp(),
            'mode': 'fallback',
        }

    async def _make_prediction_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        price_data = await self._fetch_price_data(symbol)
        if price_data is None:
            return None
        feature_snapshot = self._extract_feature_snapshot(price_data)
        if feature_snapshot is None:
            return None

        if self.adapter and self.is_trained:
            try:
                adapter_result = self.adapter.predict(feature_snapshot)
                adapter_result['fallback'] = False
                return self._build_prediction_payload(symbol, price_data, adapter_result)
            except Exception as exc:  # noqa: BLE001
                logger.error("GEMMA adapter failed for %s: %s", symbol, exc, exc_info=True)

        return self._build_fallback_prediction(symbol, price_data)

    async def _update_predictions(self, symbols: List[str]) -> None:
        if not symbols:
            return
        for symbol in symbols:
            prediction = await self._make_prediction_for_symbol(symbol)
            if prediction:
                self.prediction_cache[symbol] = prediction
                mode = prediction.get('mode', 'unknown')
                logger.info("🧠 [PRICE-ENGINE] %s prediction refreshed (%s)", symbol, mode)

    async def start_prediction_loop(self, symbols: List[str], timeframes: Optional[List[str]] = None):
        self.is_running = True
        for symbol in symbols:
            self.data_buffers[symbol] = {self.primary_timeframe: deque(maxlen=200)}

        logger.info(
            "🧠 [PRICE-ENGINE] Starting prediction loop for %d symbols | timeframe=%s",
            len(symbols),
            self.primary_timeframe,
        )
        if timeframes:
            logger.info("🧠 [PRICE-ENGINE] Ignoring deprecated timeframe overrides: %s", timeframes)

        while self.is_running:
            try:
                await self._update_predictions(symbols)
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                logger.info("🧠 [PRICE-ENGINE] Prediction loop cancelled")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in prediction loop: %s", exc, exc_info=True)
                await asyncio.sleep(self.update_interval)

        logger.info("🧠 [PRICE-ENGINE] Prediction loop stopped")

    async def stop_prediction_loop(self):
        self.is_running = False
        logger.info("🧠 [PRICE-ENGINE] Stopping prediction loop...")

    def get_price_forecast(self, symbol: str, horizon: int = 12) -> Optional[Dict[str, Any]]:
        if symbol not in self.prediction_cache:
            return None

        cached = self.prediction_cache[symbol]
        timestamp_raw = cached.get('timestamp')
        try:
            cached_ts = self._ensure_utc_timestamp(timestamp_raw)
            cached['timestamp'] = cached_ts
        except Exception as exc:  # noqa: BLE001
            logger.warning("Invalid timestamp for %s cache entry: %s", symbol, exc, exc_info=True)
            del self.prediction_cache[symbol]
            return None

        age = (self._current_timestamp() - cached_ts).total_seconds()
        if age > self.cache_ttl.total_seconds():
            logger.warning(
                "Cache for %s is stale (age %.1fs > ttl %.1fs). Purging entry.",
                symbol,
                age,
                self.cache_ttl.total_seconds(),
            )
            del self.prediction_cache[symbol]
            return None

        return cached

    def generate_trading_signals(self, symbol: str, current_price: float, threshold: float = 0.02) -> Dict[str, Any]:
        forecast = self.get_price_forecast(symbol)

        if not forecast:
            return {
                'signal': 'neutral',
                'strength': 0.0,
                'reason': 'no_forecast',
            }

        agg = forecast['aggregated']
        forecast_pct = float(agg['forecast'][0]) if len(agg['forecast']) else 0.0
        uncertainty = float(agg['uncertainty'][0]) if len(agg['uncertainty']) else 1.0
        consensus = float(agg.get('consensus_strength', 0.0))

        expected_change = forecast_pct / 100

        if expected_change > threshold and consensus > 0.7:
            signal = 'bullish'
            strength = min(abs(expected_change) * consensus, 1.0)
        elif expected_change < -threshold and consensus > 0.7:
            signal = 'bearish'
            strength = min(abs(expected_change) * consensus, 1.0)
        else:
            signal = 'neutral'
            strength = 0.0

        confidence = 1.0 / (1.0 + uncertainty)
        position_size = strength * confidence

        return {
            'symbol': symbol,
            'signal': signal,
            'strength': float(strength),
            'position_size': float(position_size),
            'expected_change': float(expected_change),
            'uncertainty': float(uncertainty),
            'consensus': float(consensus),
            'confidence': float(confidence),
            'forecast_price': current_price * (1 + expected_change),
            'timestamp': forecast['timestamp'],
        }

    def get_engine_status(self) -> Dict[str, Any]:
        return {
            'running': self.is_running,
            'symbols_tracked': list(self.data_buffers.keys()),
            'n_predictions_cached': len(self.prediction_cache),
            'update_interval': self.update_interval,
            'timeframes': [self.primary_timeframe],
            'mode': 'gemma' if self.is_trained else 'fallback',
        }
