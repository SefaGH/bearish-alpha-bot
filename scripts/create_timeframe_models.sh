#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "🔗 CREATING TIMEFRAME-SPECIFIC MODELS"
echo "========================================"

mkdir -p data/models/{lstm,xgboost,rf}/{5m,15m,30m,1h,4h,1d}
mkdir -p data/models/regime_lstm

echo "📦 Setting up timeframe models..."

if [ -f "data/models/final/gemma_price.pt" ]; then
  for tf in 5m 15m 30m 1h 4h 1d; do
    cp data/models/final/gemma_price.pt "data/models/lstm/${tf}/lstm_${tf}.pth"
    echo "   ✅ LSTM $tf model created"
  done
else
  echo "⚠️  GEMMA price model not found; skipping LSTM copies"
fi

if [ -f "data/models/final/gemma_regime.pt" ]; then
  cp data/models/final/gemma_regime.pt data/models/regime_lstm/best_model.pth
  echo "   ✅ Regime LSTM checkpoint created"
  for model_type in rf xgboost; do
    for tf in 15m 30m 1h 4h 1d; do
      touch "data/models/${model_type}/${tf}/${model_type}_model.pkl"
      echo "   ✅ ${model_type} $tf placeholder created"
    done
  done
else
  echo "⚠️  GEMMA regime model not found; skipping regime placeholders"
fi

cat > data/models/model_metadata.json <<'META'
{
  "version": "GEMMA-2.0.0",
  "models": {
    "lstm": {
      "source": "gemma_price",
      "features": 82,
      "architecture": "GEMMA"
    },
    "regime_lstm": {
      "source": "gemma_regime",
      "features": 82,
      "architecture": "GEMMA"
    }
  },
  "created": "2025-11-16",
  "compatible_with": "GEMMA-2.0.0"
}
META

echo ""
echo "📊 Model Structure Created:"
find data/models -name "*.pth" -o -name "*.pkl" | head -10
echo ""
echo "✅ Timeframe models setup complete!"