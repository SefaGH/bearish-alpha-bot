#!/bin/bash
# Run Short Paper Trading Test (5 minutes)
#
# This script runs a 5-minute paper trading test and monitors it.
# Usage: ./scripts/run_short_test.sh

set -e

echo "🚀 Starting Short Paper Trading Test (5 minutes)"
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

# Step 2: Launch paper trading bot
echo ""
echo "Step 2: Launching paper trading bot (5 minutes)..."
nohup python scripts/live_trading_launcher.py \
    --paper \
    --debug \
    --duration 300 \
    --symbols "BTC/USDT,ETH/USDT" \
    > paper_trading_test_300s.log 2>&1 &

BOT_PID=$!
echo "Bot started with PID: $BOT_PID"
echo $BOT_PID > paper_trading_bot.pid

# Step 3: Monitor the bot
echo ""
echo "Step 3: Monitoring bot (5 minutes)..."
python scripts/monitor_paper_trading.py paper_trading_test_300s.log 300 &
MONITOR_PID=$!

# Wait for bot to complete
wait $BOT_PID
BOT_EXIT=$?

# Wait for monitor to complete
wait $MONITOR_PID
MONITOR_EXIT=$?

# Clean up PID file
rm -f paper_trading_bot.pid

# Step 4: Check for critical errors
echo ""
echo "Step 4: Checking for critical errors..."
grep -E "(ERROR|CRITICAL|Exception|Traceback)" paper_trading_test_300s.log > errors_found.txt || true

if [ -s errors_found.txt ]; then
    echo "❌ Critical errors found:"
    head -n 10 errors_found.txt
    echo ""
    echo "Full errors saved in: errors_found.txt"
    exit 1
else
    echo "✅ No critical errors in 5-minute test"
    rm -f errors_found.txt
fi

echo ""
echo "=================================================="
echo "✅ Short test completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review the report: paper_trading_report_*.json"
echo "  2. If all looks good, run extended test: ./scripts/run_extended_test.sh"

exit 0
