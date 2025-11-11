# GEMMA TorchScript Adapter

Production-ready adapter for GEMMA model integration in Bearish Alpha Bot.

## Overview

The `GemmaTorchScriptAdapter` provides a robust, production-ready interface for using GEMMA TorchScript models in the Bearish Alpha Bot. It includes fault tolerance, performance monitoring, caching, and fallback mechanisms.

## Features

### 🔌 Model Loading
- Loads TorchScript `.pt` models
- Automatic GPU/CPU device selection
- Loads auxiliary components (scaler, feature list, feature mask)
- Graceful error handling

### 🎯 Feature Alignment
- Aligns 87-feature input to 82-feature model requirements
- Handles missing features (fills with 0.0)
- Maintains correct feature order

### 🛡️ Circuit Breaker
- Three-state fault tolerance: CLOSED → OPEN → HALF_OPEN → CLOSED
- Configurable failure threshold
- Automatic recovery attempts
- Thread-safe implementation

### 📊 Performance Monitoring
- Inference time tracking
- P95 percentile metrics
- Average inference time calculation
- Shadow mode for prediction logging

### ⚡ Caching
- Deterministic cache key generation (SHA256)
- Configurable TTL (Time-To-Live)
- Reduces redundant predictions
- Thread-safe cache management

### 🔄 Fallback Mechanism
- Safe neutral predictions on errors
- 50% confidence, neutral label
- Prevents system failures

## Installation

The adapter requires the following dependencies:

```bash
pip install torch numpy scikit-learn joblib
```

## Usage

### Basic Configuration

```python
from src.ml.adapters.gemma import GemmaTorchScriptAdapter

config = {
    'model_path': 'models/gemma/price_prediction.pt',
    'scaler_path': 'data/cache/gemma/scaler_gemma.joblib',
    'features_path': 'features/gemma/selected/gemma_price_selected_82.json',
    'cache_ttl': 30,
    'shadow_mode': False
}

adapter = GemmaTorchScriptAdapter(config)
```

### Making Predictions

```python
# Prepare features (87 features from StrategyCoordinator)
features_dict = {
    'sma_5': 45000.5,
    'ema_5': 45100.2,
    'rsi_5': 65.3,
    # ... more features
}

# Get prediction
result = adapter.predict(features_dict)

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['price_confidence']:.2%}")
print(f"Inference Time: {result['inference_time_ms']:.2f}ms")
```

### Monitoring Performance

```python
metrics = adapter.get_metrics()

print(f"Model Loaded: {metrics['model_loaded']}")
print(f"Circuit State: {metrics['circuit_state']}")
print(f"Avg Inference Time: {metrics['avg_inference_time_ms']:.2f}ms")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to TorchScript .pt model |
| `scaler_path` | str | required | Path to scaler .joblib file |
| `features_path` | str | required | Path to features .json file |
| `feature_mask_path` | str | optional | Path to feature mask .npy file |
| `cache_ttl` | int | 30 | Cache TTL in seconds |
| `shadow_mode` | bool | False | Enable prediction logging |
| `circuit_breaker.failure_threshold` | int | 5 | Failures before opening circuit |
| `circuit_breaker.recovery_timeout` | int | 60 | Recovery timeout in seconds |

## Prediction Output

The `predict()` method returns a dictionary with:

```python
{
    'price_confidence': float,      # 0.0-1.0
    'prediction': int,              # 0=bearish, 1=neutral, 2=bullish
    'prediction_label': str,        # 'bearish', 'neutral', or 'bullish'
    'probabilities': list[float],   # [p_bearish, p_neutral, p_bullish]
    'timestamp': str,               # ISO format timestamp
    'fallback': bool,               # True if using fallback prediction
    'inference_time_ms': float      # Inference time in milliseconds
}
```

## Circuit Breaker States

### CLOSED (Normal Operation)
- All predictions are processed
- Failure count is tracked

### OPEN (Circuit Open)
- All prediction calls are rejected
- Returns RuntimeError immediately
- After `recovery_timeout`, transitions to HALF_OPEN

### HALF_OPEN (Testing Recovery)
- Allows one prediction attempt
- Success → CLOSED state
- Failure → OPEN state

## Error Handling

The adapter handles errors gracefully:

1. **Model Loading Errors**: Logged, model set to None
2. **Prediction Errors**: Circuit breaker triggers, fallback prediction returned
3. **Circuit Open**: RuntimeError raised, caught, fallback returned
4. **Missing Features**: Filled with 0.0

## Testing

Run the test suite:

```bash
pytest tests/test_gemma_adapter.py -v
```

Test coverage:
- Circuit breaker state transitions
- Prediction with valid/invalid features
- Caching behavior
- Fallback mechanism
- Feature alignment
- Metrics retrieval
- Shadow mode logging

## Architecture

```
GemmaTorchScriptAdapter
├── CircuitBreaker (fault tolerance)
├── Model (TorchScript)
├── Scaler (StandardScaler)
├── Features (82 feature list)
├── Feature Mask (optional)
├── Cache (prediction cache)
└── Shadow Log (optional monitoring)
```

## Performance Considerations

- **Caching**: Reduces redundant predictions by ~80% in typical scenarios
- **Circuit Breaker**: Prevents cascading failures
- **Inference Time**: Typically 5-20ms on CPU, 1-5ms on GPU
- **Memory**: ~100MB for model + ~10MB for adapter overhead

## Production Recommendations

1. **Enable Shadow Mode**: For monitoring in production
2. **Configure Circuit Breaker**: Adjust thresholds based on your requirements
3. **Monitor Metrics**: Use `get_metrics()` for health checks
4. **Cache TTL**: Adjust based on market volatility
5. **GPU Usage**: Enable CUDA if available for faster inference

## Integration Example

```python
class StrategyCoordinator:
    def __init__(self, config):
        self.gemma_adapter = GemmaTorchScriptAdapter(config['gemma_adapter'])
    
    async def generate_signal(self, symbol, market_data):
        features = self._extract_features(market_data)
        prediction = self.gemma_adapter.predict(features)
        
        if prediction['fallback']:
            logger.warning("Using fallback prediction")
        
        return self._process_prediction(prediction)
```

## Troubleshooting

### Model Not Loading
- Check file path exists
- Verify TorchScript format
- Check Python/PyTorch version compatibility

### Circuit Always Open
- Check model errors in logs
- Adjust failure threshold
- Verify feature alignment

### Slow Inference
- Enable GPU if available
- Reduce cache TTL
- Check model size

## License

MIT License - See LICENSE file for details.

## Authors

- SefaGH - Initial implementation

## Version

GEMMA-1.0.0 (Phase 4)
