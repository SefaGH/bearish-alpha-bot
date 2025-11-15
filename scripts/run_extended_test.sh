#!/bin/bash
# Run Extended Paper Trading Test (1 hour)
#
# This script runs a 1-hour paper trading test with comprehensive monitoring.
# Usage: ./scripts/run_extended_test.sh

set -e

echo "🚀 Starting Extended Paper Trading Test (1 hour)"
echo "=================================================="

# Setup environment
source scripts/test_environment.sh

# Step 1: Verify system is ready
echo ""
echo "Step 1: Verifying system readiness..."
python scripts/verify_system_ready.py
if [ $? -ne 0 ]; then
    echo "❌ System not ready. Fix errors before continuing."
    exit 1
fi

# Step 2: Launch paper trading bot for 1 hour
echo ""
echo "Step 2: Launching paper trading bot (1 hour)..."
nohup python scripts/live_trading_launcher.py \
    --paper \
    --debug \
    --duration 3600 \
    --symbols "BTC/USDT,ETH/USDT,SOL/USDT" \
    > paper_trading_1hour.log 2>&1 &

BOT_PID=$!
echo "Bot started with PID: $BOT_PID"
echo $BOT_PID > paper_trading_bot.pid

# Step 3: Monitor in background
echo ""
echo "Step 3: Starting monitoring (1 hour)..."
python scripts/monitor_paper_trading.py paper_trading_1hour.log 3600 &
MONITOR_PID=$!

echo ""
echo "Bot is running (PID: $BOT_PID)"
echo "Monitor is running (PID: $MONITOR_PID)"
echo ""
echo "Progress will be reported every 30 seconds."
echo "Test will complete in 1 hour."
echo ""
echo "To check status manually:"
echo "  tail -f paper_trading_1hour.log"
echo ""
echo "To stop the test:"
echo "  kill $BOT_PID $MONITOR_PID"
echo ""

# Wait for completion
wait $BOT_PID
BOT_EXIT=$?

wait $MONITOR_PID
MONITOR_EXIT=$?

# Clean up PID file
rm -f paper_trading_bot.pid

# Step 4: Run health analysis
echo ""
echo "Step 4: Running health analysis..."
python scripts/paper_trading_health.py paper_trading_1hour.log 3600
HEALTH_EXIT=$?

# Step 5: Verify ML components
echo ""
echo "Step 5: Verifying ML components..."
python scripts/verify_ml_live.py paper_trading_1hour.log
ML_EXIT=$?

# Step 6: Overall assessment
echo ""
echo "=================================================="
if [ $BOT_EXIT -eq 0 ] && [ $HEALTH_EXIT -eq 0 ] && [ $ML_EXIT -eq 0 ]; then
    echo "✅ Extended test completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Review all reports: paper_*.json"
    echo "  2. Make deployment decision: python scripts/production_readiness_decision.py"
    exit 0
else
    echo "❌ Extended test completed with issues"
    echo ""
    echo "Please review:"
    echo "  - paper_trading_1hour.log"
    echo "  - paper_health_*.json"
    echo "  - paper_trading_report_*.json"
    exit 1
fi
