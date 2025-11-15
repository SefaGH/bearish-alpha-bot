#!/bin/bash
# Environment Setup for Paper Trading Tests

echo "🔧 Setting up environment for paper trading test..."

# Set required environment variables
export ML_ENABLED=true
export GEMMA_ENABLED=false  # Start with legacy system
export PAPER_TRADING=true
export DEBUG_MODE=true
export LOG_LEVEL=INFO

echo "📋 Environment Configuration:"
echo "  ML_ENABLED: $ML_ENABLED"
echo "  GEMMA_ENABLED: $GEMMA_ENABLED"
echo "  PAPER_TRADING: $PAPER_TRADING"
echo "  DEBUG_MODE: $DEBUG_MODE"
echo "  LOG_LEVEL: $LOG_LEVEL"

echo ""
echo "✅ Environment configured for paper trading"
echo "   Use: source scripts/test_environment.sh"
