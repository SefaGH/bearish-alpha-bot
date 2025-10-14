#!/bin/bash
# Ultimate Continuous Trading Mode - Usage Examples
# Demonstrates all features of the multi-layer defense system

echo "=================================="
echo "ULTIMATE CONTINUOUS MODE - DEMO"
echo "=================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Available Commands:${NC}"
echo ""

echo -e "${GREEN}1. Standard Live Trading (Original Behavior)${NC}"
echo "   python scripts/live_trading_launcher.py"
echo ""

echo -e "${GREEN}2. Layer 1: TRUE CONTINUOUS MODE${NC}"
echo "   - Never stops trading"
echo "   - Auto-recovers from errors"
echo "   - Bypasses non-critical circuit breakers"
echo "   python scripts/live_trading_launcher.py --infinite"
echo ""

echo -e "${GREEN}3. Layer 2: AUTO-RESTART FAILSAFE${NC}"
echo "   - External monitoring"
echo "   - Auto-restarts on crashes"
echo "   - Exponential backoff"
echo "   python scripts/live_trading_launcher.py --auto-restart"
echo ""

echo -e "${GREEN}4. ULTIMATE MODE: Both Layers (Recommended)${NC}"
echo "   - Maximum resilience"
echo "   - Never stops, always recovers"
echo "   - Enterprise-grade reliability"
echo "   python scripts/live_trading_launcher.py --infinite --auto-restart"
echo ""

echo -e "${GREEN}5. Paper Trading with Ultimate Mode${NC}"
echo "   python scripts/live_trading_launcher.py --paper --infinite --auto-restart"
echo ""

echo -e "${GREEN}6. Custom Restart Parameters${NC}"
echo "   python scripts/live_trading_launcher.py \\"
echo "       --infinite \\"
echo "       --auto-restart \\"
echo "       --max-restarts 500 \\"
echo "       --restart-delay 60"
echo ""

echo -e "${GREEN}7. Time-Limited Cycles with Auto-Restart${NC}"
echo "   # Runs 1-hour cycles, auto-restarts between them"
echo "   python scripts/live_trading_launcher.py \\"
echo "       --duration 3600 \\"
echo "       --auto-restart \\"
echo "       --max-restarts 24"
echo ""

echo -e "${GREEN}8. Dry Run (Pre-flight Checks Only)${NC}"
echo "   python scripts/live_trading_launcher.py --dry-run"
echo ""

echo -e "${YELLOW}=== IMPORTANT SAFETY NOTES ===${NC}"
echo ""
echo "• Ctrl+C always works for manual stop (no restart)"
echo "• Maximum restart limit prevents infinite loops"
echo "• Telegram notifications keep you informed"
echo "• Health monitoring tracks system status"
echo "• State is preserved across restarts"
echo ""

echo -e "${BLUE}=== MONITORING ===${NC}"
echo ""
echo "Set environment variables for Telegram notifications:"
echo "  export TELEGRAM_BOT_TOKEN='your_bot_token'"
echo "  export TELEGRAM_CHAT_ID='your_chat_id'"
echo ""
echo "Check logs:"
echo "  tail -f live_trading_*.log"
echo ""

echo -e "${BLUE}=== QUICK START ===${NC}"
echo ""
echo "For production 24/7 trading:"
echo ""
echo "  1. Set up Telegram notifications"
echo "  2. Test in paper mode first:"
echo "     python scripts/live_trading_launcher.py --paper --infinite --auto-restart"
echo "  3. Deploy to live:"
echo "     python scripts/live_trading_launcher.py --infinite --auto-restart"
echo ""

echo "=================================="
echo "For detailed documentation, see:"
echo "  ULTIMATE_CONTINUOUS_MODE.md"
echo "=================================="
