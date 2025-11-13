"""
GEMMA Adapter Usage Example
Demonstrates how to use the GemmaTorchScriptAdapter in production.
"""

from src.ml.adapters.gemma import GemmaTorchScriptAdapter
import numpy as np

# Example 1: Basic Configuration
config_basic = {
    'model_path': 'data/models/final/gemma_price.pt',
    'scaler_path': 'data/models/final/gemma_price_scaler.joblib',
    'features_path': 'features/gemma/selected/gemma_price_selected_82.json',
    'cache_ttl': 30,  # Cache predictions for 30 seconds
    'shadow_mode': False  # Disable shadow logging
}

# Example 2: Production Configuration with Circuit Breaker
config_production = {
    'model_path': 'data/models/final/gemma_price.pt',
    'scaler_path': 'data/models/final/gemma_price_scaler.joblib',
    'features_path': 'features/gemma/selected/gemma_price_selected_82.json',
    'cache_ttl': 30,
    'shadow_mode': True,  # Enable shadow logging for monitoring
    'circuit_breaker': {
        'failure_threshold': 5,  # Open circuit after 5 failures
        'recovery_timeout': 60   # Try recovery after 60 seconds
    }
}

# Initialize the adapter
adapter = GemmaTorchScriptAdapter(config_production)

# Example 3: Making a Prediction
# Prepare feature dictionary (87 features from StrategyCoordinator)
features_dict = {
    'sma_5': 45000.5,
    'ema_5': 45100.2,
    'rsi_5': 65.3,
    'stoch_k_5': 70.1,
    'stoch_d_5': 68.5,
    # ... add all 87 features
    # The adapter will automatically select the 82 features needed
}

# Make prediction
result = adapter.predict(features_dict)

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['price_confidence']:.2%}")
print(f"Probabilities: Bearish={result['probabilities'][0]:.2%}, "
      f"Neutral={result['probabilities'][1]:.2%}, "
      f"Bullish={result['probabilities'][2]:.2%}")
print(f"Inference Time: {result['inference_time_ms']:.2f}ms")
print(f"Is Fallback: {result['fallback']}")

# Example 4: Monitoring Adapter Performance
metrics = adapter.get_metrics()
print(f"\n📊 Adapter Metrics:")
print(f"   Model Loaded: {metrics['model_loaded']}")
print(f"   Circuit State: {metrics['circuit_state']}")
print(f"   Cache Size: {metrics['cache_size']}")
print(f"   Avg Inference Time: {metrics['avg_inference_time_ms']:.2f}ms")
print(f"   P95 Inference Time: {metrics['p95_inference_time_ms']:.2f}ms")
print(f"   Shadow Log Size: {metrics['shadow_log_size']}")

# Example 5: Integration with StrategyCoordinator
"""
In your StrategyCoordinator:

from src.ml.adapters.gemma import GemmaTorchScriptAdapter

class StrategyCoordinator:
    def __init__(self, config):
        self.gemma_adapter = GemmaTorchScriptAdapter(config['gemma_adapter'])
    
    async def generate_signal(self, symbol, market_data):
        # Extract features from market data (87 features)
        features = self._extract_features(market_data)
        
        # Get GEMMA prediction
        prediction = self.gemma_adapter.predict(features)
        
        if prediction['fallback']:
            logger.warning("Using fallback prediction - GEMMA model unavailable")
        
        # Use prediction in signal generation
        if prediction['prediction_label'] == 'bullish' and prediction['price_confidence'] > 0.7:
            # Generate buy signal
            pass
        elif prediction['prediction_label'] == 'bearish' and prediction['price_confidence'] > 0.7:
            # Generate sell signal
            pass
"""
