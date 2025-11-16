# src/ml/adapters/gemma/gemma_torchscript_adapter.py
"""
GEMMA TorchScript Adapter for Bearish Alpha Bot
Production-ready with circuit breaker and monitoring
"""

import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import json
from collections import deque
from threading import Lock
import time
import joblib

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self._lock = Lock()

    def call(self, func, *args, **kwargs):
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker: OPEN → HALF_OPEN")
                else:
                    raise RuntimeError("Circuit is open. Call rejected.")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                with self._lock:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker: HALF_OPEN → CLOSED")
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                logger.warning(f"Circuit breaker recorded a failure ({self.failure_count}/{self.failure_threshold}). Error: {e}")
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker has been opened due to excessive failures.")
            raise e

class GemmaTorchScriptAdapter:
    """
    GEMMA model adapter for Bearish Alpha Bot.
    Handles .pt models with manifest-driven feature alignment.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get feature configuration from config (passed from manifest)
        self.expected_feature_count = config.get('feature_count')
        self.feature_names = config.get('feature_names', [])
        
        # Handle circuit breaker configuration
        circuit_config = config.get('circuit_breaker', {}).copy()
        self.circuit_breaker_enabled = circuit_config.pop('enabled', True)
        
        if self.circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=circuit_config.get('failure_threshold', 5),
                recovery_timeout=circuit_config.get('recovery_timeout', 60)
            )
        else:
            self.circuit_breaker = None

        self.model: Optional[torch.jit.ScriptModule] = None
        self.scaler = None
        self.features: Optional[List[str]] = None
        self.feature_mask: Optional[np.ndarray] = None

        self.inference_times = deque(maxlen=1000)
        self.prediction_cache = {}
        self.cache_ttl = config.get('cache_ttl', 30)

        self.shadow_mode = config.get('shadow_mode', False)
        self.shadow_predictions = deque(maxlen=5000)

        self._load_model_and_components()

        if self.expected_feature_count is None:
            if self.features:
                self.expected_feature_count = len(self.features)
            elif getattr(self.scaler, 'n_features_in_', None):
                self.expected_feature_count = self.scaler.n_features_in_
            else:
                self.expected_feature_count = 82

        if self.features and len(self.features) != self.expected_feature_count:
            logger.warning(
                "Adjusting GEMMA feature list from %s to %s entries to match expected feature count.",
                len(self.features),
                self.expected_feature_count
            )
            self.features = self.features[:self.expected_feature_count]

        logger.info(
            f"✅ GEMMA Adapter initialized | "
            f"Features: {self.expected_feature_count} | "
            f"Device: {self.device} | "
            f"Shadow Mode: {self.shadow_mode}"
        )

    def _load_model_and_components(self):
        """Load TorchScript model and all auxiliary components."""
        try:
            # 1. Load Model
            model_path = Path(self.config['model_path'])
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()
            logger.info(f"✅ Loaded TorchScript model from: {model_path}")

            # 2. Load Scaler
            # Updated default path to new production location (Plan 2)
            scaler_path = Path(self.config.get('scaler_path', 'data/models/final/gemma_price_scaler.joblib'))
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Loaded scaler from: {scaler_path}")

            # 3. Load Feature List
            features_path = Path(self.config.get('features_path', 'features/gemma/selected/gemma_price_selected_82.json'))
            if not features_path.exists():
                raise FileNotFoundError(f"Features JSON not found at {features_path}")
            with open(features_path) as f:
                self.features = json.load(f)['features']
            logger.info(f"✅ Loaded {len(self.features)} features from: {features_path}")

            scaler_feature_count = getattr(self.scaler, 'n_features_in_', None)
            if scaler_feature_count and len(self.features) != scaler_feature_count:
                logger.warning(
                    "Scaler expects %s features but feature list contains %s entries. "
                    "Truncating feature list to match scaler input.",
                    scaler_feature_count,
                    len(self.features)
                )
                self.features = self.features[:scaler_feature_count]

            # 4. Load Feature Mask (optional, but recommended)
            mask_source = self.config.get('feature_mask_path')
            if not mask_source:
                mask_source = 'data/cache/gemma/feature_selection_mask.npy'
            mask_path = Path(mask_source)
            if mask_path.exists():
                self.feature_mask = np.load(mask_path)
                logger.info(f"✅ Loaded feature selection mask from: {mask_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load GEMMA components: {e}", exc_info=True)
            self.model = None # Ensure adapter is non-functional if setup fails

    @torch.no_grad()
    def predict(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """Main prediction method with caching, circuit breaker, and monitoring."""
        start_time = time.time()

        cache_key = self._get_cache_key(features_dict)
        if cache_key in self.prediction_cache:
            cached_time, cached_result = self.prediction_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result

        try:
            if self.circuit_breaker and self.circuit_breaker_enabled:
                result = self.circuit_breaker.call(self._predict_internal, features_dict)
            else:
                result = self._predict_internal(features_dict)
        except Exception as e:
            logger.error(f"Prediction failed. Circuit state: {getattr(self.circuit_breaker, 'state', 'N/A')}. Error: {e}")
            result = self._get_fallback_prediction()

        inference_time = (time.time() - start_time) * 1000 # in ms
        self.inference_times.append(inference_time)
        result['inference_time_ms'] = inference_time

        self.prediction_cache[cache_key] = (time.time(), result)
        if self.shadow_mode:
            self.shadow_predictions.append({'timestamp': datetime.now().isoformat(), 'features_hash': cache_key, 'prediction': result})

        return result

    def _predict_internal(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """Internal prediction logic, called by the circuit breaker."""
        if not all([self.model, self.scaler, self.features]):
            raise RuntimeError("Adapter is not fully initialized. Cannot predict.")

        feature_vector_aligned = self._align_features(features_dict)
        feature_vector_scaled = self.scaler.transform([feature_vector_aligned])
        tensor = torch.tensor(feature_vector_scaled, dtype=torch.float32, device=self.device)

        output = self.model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction_idx = torch.max(probs, 1)

        prediction_label = ['bearish', 'neutral', 'bullish'][prediction_idx.item()]

        return {
            'price_confidence': confidence.item(),
            'prediction': prediction_idx.item(),
            'prediction_label': prediction_label,
            'probabilities': probs[0].cpu().numpy().tolist(),
            'timestamp': datetime.now().isoformat(),
            'fallback': False
        }

    def _align_features(self, features_dict: Dict[str, float]) -> np.ndarray:
        """Aligns incoming feature dictionary to the model's required input vector."""
        # This assumes 'self.features' is the list of 82 feature names in the correct order.
        full_vector = np.array([features_dict.get(f, 0.0) for f in self.features])
        return full_vector

    def _get_cache_key(self, features: Dict[str, float]) -> str:
        """Generates a deterministic cache key from features."""
        sorted_features = sorted(features.items())
        feature_str = json.dumps(sorted_features, separators=(',', ':'))
        return hashlib.sha256(feature_str.encode()).hexdigest()

    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Returns a safe, neutral prediction during failures."""
        return {
            'price_confidence': 0.5,
            'prediction': 1,  # Neutral
            'prediction_label': 'neutral',
            'probabilities': [0.33, 0.34, 0.33],
            'timestamp': datetime.now().isoformat(),
            'fallback': True
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Returns current performance metrics of the adapter."""
        return {
            'model_loaded': self.model is not None,
            'circuit_breaker_enabled': getattr(self, 'circuit_breaker_enabled', False),
            'circuit_state': self.circuit_breaker.state if self.circuit_breaker else 'DISABLED',
            'cache_size': len(self.prediction_cache),
            'avg_inference_time_ms': np.mean(self.inference_times) if self.inference_times else 0,
            'p95_inference_time_ms': np.percentile(self.inference_times, 95) if len(self.inference_times) > 1 else 0,
            'shadow_log_size': len(self.shadow_predictions)
        }
