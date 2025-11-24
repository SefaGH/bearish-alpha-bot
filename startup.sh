#!/bin/bash
# Azure App Service Startup Script for Bearish Alpha Bot
# This script sets up the proper environment and starts the trading bot

set -e  # Exit on any error

echo "========================================"
echo "🤖 Bearish Alpha Bot - Azure Startup"
echo "========================================"
echo "Timestamp: $(date)"
echo "Python Version: $(python3 --version)"
echo "Working Directory: $(pwd)"
echo "User: $(whoami)"

# Set up proper Python path for module imports
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src:$(pwd)/scripts"
echo "✅ PYTHONPATH configured: $PYTHONPATH"

# Ensure required directories exist
mkdir -p logs data artifacts/gemma/final artifacts/ppo features/gemma/selected data/models/final data/cache/gemma
echo "✅ Required directories created"

# Create placeholder files if they don't exist
touch data/state.json data/day_stats.json logs/.placeholder
echo "✅ Placeholder files created"

# Verify critical files exist
CRITICAL_FILES=(
    "config/config.example.yaml"
    "scripts/live_trading_launcher.py"
    "src/core/production_coordinator.py"
)

echo "🔍 Checking critical files..."
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING!"
        exit 1
    fi
done

# Set up GEMMA artifacts if they don't exist
if [ ! -f "artifacts/gemma/final/manifest.json" ]; then
    echo "🧠 Setting up proper GEMMA-2.0.0 manifest for Azure..."
    cat > "artifacts/gemma/final/manifest.json" << 'EOF'
{
    "version": "GEMMA-2.0.0",
    "feature_count": 82,
    "model_type": "gemma",
    "description": "GEMMA-2.0.0 manifest for Azure deployment with 82 features",
    "feature_names_ordered": [
        "open", "high", "low", "close", "volume", "rsi", "rsi_oversold", "rsi_overbought", 
        "macd", "macd_signal", "macd_histogram", "macd_cross", "ema_12", "ema_26", "ema_50", 
        "ema_cross", "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position", "atr", 
        "volatility_parkinson", "volatility_realized", "close_to_high", "close_to_low", 
        "price_change_pct", "volume_sma", "volume_ratio", "obv", "stoch_k", "stoch_d", 
        "williams_r", "cci", "momentum", "roc", "adx", "plus_di", "minus_di", "aroon_up", 
        "aroon_down", "tsi", "ultimate_oscillator", "stochrsi_k", "stochrsi_d", "kama", 
        "tema", "wma", "hma", "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", 
        "ichimoku_senkou_b", "vwap", "pivot_point", "support1", "resistance1", "support2", 
        "resistance2", "fibonacci_23_6", "fibonacci_38_2", "fibonacci_50_0", "fibonacci_61_8", 
        "fibonacci_78_6", "doji", "hammer", "shooting_star", "engulfing", "morning_star", 
        "evening_star", "three_white_soldiers", "three_black_crows", "inside_bar", 
        "outside_bar", "price_ma_20", "price_ma_50", "price_ma_200", "volume_ma_10", 
        "high_low_ratio", "open_close_ratio", "body_size", "upper_shadow", "lower_shadow", 
        "total_shadow", "body_shadow_ratio", "gap_up", "gap_down"
    ],
    "regime_features": [
        "rsi", "macd", "ema_50", "bb_position", "atr", "volatility_realized", "adx", 
        "stoch_k", "williams_r", "cci", "momentum", "roc", "aroon_up", "tsi", 
        "ultimate_oscillator", "stochrsi_k", "kama", "vwap", "price_ma_20", "volume_ratio"
    ],
    "regime_feature_count": 20
}
EOF
    echo "✅ GEMMA-2.0.0 manifest created for Azure"
fi

# Set up ML model artifacts
echo "🧠 Setting up ML model links..."
bash scripts/setup_gemma_artifacts.sh || true
bash scripts/setup_ml_model_links.sh || true

# Set up GEMMA environment variables for Azure
echo "🧠 Setting up GEMMA environment variables..."
export GEMMA_ENABLED=true
export ML_ACTIVE_BUNDLE="artifacts/gemma/final"
echo "✅ GEMMA environment: GEMMA_ENABLED=$GEMMA_ENABLED, ML_ACTIVE_BUNDLE=$ML_ACTIVE_BUNDLE"

# Validate Python environment
echo "🔍 Validating Python environment..."
python3 -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python path: {sys.path}')

# Test critical imports
try:
    import aiohttp
    print('✅ aiohttp imported successfully')
except ImportError as e:
    print(f'❌ aiohttp import failed: {e}')

try:
    import talib
    print('✅ TA-Lib imported successfully')
except ImportError as e:
    print('⚠️  TA-Lib not available (will use fallback indicators)')

try:
    import websocket
    print('✅ websocket-client imported successfully')
except ImportError as e:
    print(f'❌ websocket-client import failed: {e}')
"

# Determine startup mode based on environment
MODE_ARGS=()

# Azure App Service sets PORT environment variable
if [ -n "$PORT" ]; then
    echo "🌐 Detected Azure App Service environment (PORT=$PORT)"
    # Run in paper mode for safety in cloud environment unless explicitly set to live
    if [ "$TRADING_MODE" != "live" ]; then
        MODE_ARGS+=(--paper)
        echo "📝 Running in paper trading mode (Azure default)"
    fi
else
    echo "💻 Running in local environment"
    MODE_ARGS+=(--paper)  # Default to paper mode for safety
fi

# Add debug mode if requested
if [ "$DEBUG_MODE" = "true" ]; then
    MODE_ARGS+=(--debug)
    echo "🔍 Debug mode enabled"
fi

# Add duration if specified
if [ -n "$TRADING_DURATION" ]; then
    MODE_ARGS+=(--duration "$TRADING_DURATION")
    echo "⏱️  Trading duration set to $TRADING_DURATION seconds"
fi

# Build the final command
CMD=(python3 scripts/live_trading_launcher.py "${MODE_ARGS[@]}")

echo "========================================"
echo "🚀 Starting Bearish Alpha Bot"
echo "========================================"
echo "Command: ${CMD[*]}"
echo "Environment Variables:"
echo "  TRADING_MODE: ${TRADING_MODE:-paper}"
echo "  DEBUG_MODE: ${DEBUG_MODE:-false}"
echo "  ML_ENABLED: ${ML_ENABLED:-true}"
echo "  EXCHANGES: ${EXCHANGES:-bingx}"
echo "========================================"

# Execute the trading bot
exec "${CMD[@]}"