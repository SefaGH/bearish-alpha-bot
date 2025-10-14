# Debug Mode for Live Trading Analysis

## Overview

Comprehensive debug mode implementation providing detailed visibility into the trading system's internal operations. This feature enables real-time analysis of strategy signals, AI decisions, risk calculations, order execution, and system health monitoring.

## 🎯 Purpose

The debug mode addresses the challenge of understanding why no trades are being executed by providing:
- Real-time strategy signal analysis
- AI decision reasoning transparency
- Market data flow monitoring
- Risk calculation visibility
- Order execution traces
- Circuit breaker monitoring
- System performance metrics

## 🚀 Usage

### GitHub Actions Workflow

Enable debug mode when launching the trading bot via GitHub Actions:

1. Navigate to **Actions** → **Live Trading Launcher (Ultimate Continuous Mode)**
2. Click **Run workflow**
3. Set **debug_mode** to `true`
4. Configure other parameters as needed
5. Click **Run workflow**

### Command Line

Enable debug mode when running the launcher script directly:

```bash
# Paper trading with debug mode
python scripts/live_trading_launcher.py --paper --debug

# Live trading with debug mode (requires confirmation)
python scripts/live_trading_launcher.py --debug

# Dry run with debug mode (pre-flight checks only)
python scripts/live_trading_launcher.py --dry-run --debug

# Combined with continuous mode
python scripts/live_trading_launcher.py --infinite --auto-restart --debug
```

## 📊 Debug Logging Categories

### 1. Strategy Signal Analysis

**Format:** `🎯 [STRATEGY-{name}] {message}`

**Examples:**
```
🎯 [STRATEGY-AdaptiveOB] Market analysis started
📊 [STRATEGY-AdaptiveOB] Price data: close=$50000.00, RSI=25.00
📊 [STRATEGY-AdaptiveOB] Market regime: {'trend': 'bullish', 'volatility': 'normal'}
📊 [STRATEGY-AdaptiveOB] Adaptive RSI threshold: 22.50
✅ [STRATEGY-AdaptiveOB] Signal result: BUY signal generated
📈 [STRATEGY-AdaptiveOB] Signal strength: RSI 21.50 <= 22.50
❌ [STRATEGY-AdaptiveSTR] Signal result: No signal - RSI 55.00 > 70.00
```

**What it shows:**
- When strategy analysis begins
- Current market prices and indicators
- Market regime classification
- Adaptive threshold calculations
- Whether a signal was generated or rejected
- Signal strength and reasoning

### 2. AI Decision Reasoning

**Format:** `🧠 [ML-{component}] {message}`

**Examples:**
```
🧠 [ML-REGIME] Starting regime prediction for BTC/USDT
🧠 [ML-REGIME] Extracted 47 features from price data
🧠 [ML-REGIME] Market regime: bullish (confidence: 85%)
🧠 [ML-REGIME] Probabilities: Bull=85%, Neutral=10%, Bear=5%

🧠 [ML-ADAPTER] Enhancing signal for BTC/USDT at $50000.00
🧠 [ML-ADAPTER] Base signal: buy (strength: 0.65)
🧠 [ML-ADAPTER] AI signal: buy (strength: 0.78)
🧠 [ML-ADAPTER] Signal enhancement: buy → buy (strength: 0.72)
```

**What it shows:**
- ML model predictions and confidence levels
- Feature extraction process
- AI signal generation
- Signal enhancement logic
- Probability distributions

### 3. Risk Management Calculations

**Format:** `🛡️ [RISK-CALC] {message}`

**Examples:**
```
🛡️ [RISK-CALC] Validating position for BTC/USDT
🛡️ [RISK-CALC] Portfolio value: $100.00
🛡️ [RISK-CALC] Position size check: $15.00 vs $20.00 max
🛡️ [RISK-CALC] Risk per trade: 2.50% (limit: 5.00%)
🛡️ [RISK-CALC] Risk/Reward ratio: 2.00
🛡️ [RISK-CALC] Portfolio heat: 3.50% (limit: 10%)
🛡️ [RISK-CALC] APPROVED: All risk checks passed
🛡️ [RISK-CALC] REJECTED: Position size exceeds limit
```

**What it shows:**
- Available capital
- Position size validation
- Risk per trade calculations
- Risk/reward ratio assessment
- Portfolio heat (total exposure)
- Approval/rejection decisions

### 4. Order Execution Analysis

**Format:** `🎪 [ORDER-MGR] {message}`

**Examples:**
```
🎪 [ORDER-MGR] Signal received: {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': 0.001}
🎪 [ORDER-MGR] Pre-execution checks: {'valid': True, 'reason': 'All checks passed'}
🎪 [ORDER-MGR] Order parameters: algo=limit, symbol=BTC/USDT, side=buy, amount=0.001
🎪 [ORDER-MGR] Execution result: SUCCESS
🎪 [ORDER-MGR] Post-execution state: order_id=12345, executed_price=50000.00, execution_time=0.234s
🎪 [ORDER-MGR] Execution result: FAILED - Insufficient balance
```

**What it shows:**
- Received trading signals
- Pre-execution validation results
- Order parameters and algorithm used
- Execution success/failure
- Post-execution state and metrics

### 5. Circuit Breaker Monitoring

**Format:** `🔥 [CIRCUIT] {message}`

**Examples:**
```
🔥 [CIRCUIT] Daily P&L: -3.50% (limit: -5.00%)
🔥 [CIRCUIT] Volatility spike check: BTC z-score=2.5 (threshold: 3.0)
🔥 [CIRCUIT] System health: {'status': 'normal', 'errors': 0}
🔥 [CIRCUIT] TRIGGERED: Daily loss limit breached
```

**What it shows:**
- Daily profit/loss monitoring
- Volatility spike detection
- System health metrics
- Circuit breaker trigger events

## 🔧 Implementation Details

### DebugLogger Class

Located in: `src/core/debug_logger.py`

```python
from core.debug_logger import DebugLogger

# Initialize debug logger
debug_logger = DebugLogger(debug_mode=True)

# Check if debug mode is enabled
if debug_logger.is_debug_enabled():
    logger.debug("🎯 [STRATEGY] Detailed analysis...")
```

### Adding Debug Logging to Components

**Strategy Example:**
```python
logger.debug(f"🎯 [STRATEGY-{strategy_name}] Market analysis started")
logger.debug(f"📊 [STRATEGY-{strategy_name}] Price data: {price_data}")
logger.debug(f"✅ [STRATEGY-{strategy_name}] Signal result: {signal_decision}")
```

**ML Example:**
```python
logger.debug(f"🧠 [ML-REGIME] Market regime: {regime} (confidence: {confidence}%)")
logger.debug(f"🧠 [ML-ADAPTER] Signal enhancement: {original} → {enhanced}")
```

**Risk Example:**
```python
logger.debug(f"🛡️ [RISK-CALC] Portfolio value: ${portfolio_value:.2f}")
logger.debug(f"🛡️ [RISK-CALC] Risk per trade: {risk_pct:.2%} (limit: {max_risk:.2%})")
```

## 📈 Performance Impact

Debug mode has minimal performance impact:
- Logging operations are asynchronous
- Debug messages only appear when debug mode is enabled
- No performance overhead in production mode (debug disabled)
- Log file rotation prevents disk space issues

## 🧪 Testing

Run debug mode tests:

```bash
# Run all debug mode tests
python tests/test_debug_mode.py

# Run with pytest
pytest tests/test_debug_mode.py -v

# Run all related tests
pytest tests/test_live_trading_workflow.py tests/test_debug_mode.py -v
```

All tests validate:
- ✅ Debug logger initialization
- ✅ Debug level activation
- ✅ Log message formatting
- ✅ Emoji-based categorization
- ✅ Component-specific logging

## 📝 Log Analysis

### Analyzing Debug Logs

1. **Download logs from GitHub Actions:**
   - Navigate to workflow run
   - Download "trading-logs" artifact
   - Extract and open log files

2. **Search for specific categories:**
   ```bash
   # Find all strategy signals
   grep "🎯 \[STRATEGY" live_trading_*.log
   
   # Find risk rejections
   grep "🛡️ \[RISK-CALC\] REJECTED" live_trading_*.log
   
   # Find AI predictions
   grep "🧠 \[ML-" live_trading_*.log
   
   # Find circuit breaker triggers
   grep "🔥 \[CIRCUIT\] TRIGGERED" live_trading_*.log
   ```

3. **Common patterns to investigate:**
   - **No trades executed:** Look for "REJECTED" messages in risk or order logs
   - **Low signal count:** Check strategy logs for market conditions
   - **Failed orders:** Review order execution logs
   - **System halts:** Check circuit breaker logs

## 🎯 Troubleshooting

### No Trades Being Executed

1. **Check Strategy Signals:**
   ```bash
   grep "Signal result" live_trading_*.log
   ```
   - Are strategies generating signals?
   - What are the market conditions?

2. **Check Risk Validation:**
   ```bash
   grep "RISK-CALC.*REJECTED" live_trading_*.log
   ```
   - Which risk checks are failing?
   - Are position sizes too large?

3. **Check Order Execution:**
   ```bash
   grep "ORDER-MGR.*FAILED" live_trading_*.log
   ```
   - Are orders failing to execute?
   - What are the error messages?

### Understanding AI Decisions

1. **Check AI Confidence:**
   ```bash
   grep "ML-REGIME.*confidence" live_trading_*.log
   ```
   - Is AI confidence too low?
   - Are predictions agreeing with strategies?

2. **Check Signal Enhancement:**
   ```bash
   grep "ML-ADAPTER.*enhancement" live_trading_*.log
   ```
   - How is AI modifying signals?
   - Is enhancement helping or hurting?

## 🔒 Security Considerations

- Debug logs may contain sensitive information (prices, positions, balances)
- Logs are stored as workflow artifacts with 30-day retention
- Access to logs requires GitHub repository access
- Never commit debug logs to the repository
- Use `.gitignore` to exclude log files

## 📚 Related Documentation

- [Live Trading Launcher](../LIVE_TRADING_LAUNCHER_SUMMARY.md)
- [Ultimate Continuous Mode](../ULTIMATE_CONTINUOUS_MODE.md)
- [Risk Management](../PHASE3_2_RISK_MANAGEMENT_SUMMARY.md)
- [Phase 4 ML Components](../PHASE4_COMPLETE_SUMMARY.md)

## 🆕 Version History

- **v1.0** - Initial debug mode implementation
  - Strategy signal logging
  - AI decision logging
  - Risk calculation logging
  - Order execution logging
  - Circuit breaker logging
  - Workflow integration
  - Comprehensive test suite
