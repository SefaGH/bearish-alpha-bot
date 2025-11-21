# 🎯 Transition to "Sniper Mode": High-Precision Strategy Implementation

**Date:** 2025-11-21
**Author:** @SefaGH
**Status:** Implemented

## 1. Executive Summary

This document details the strategic pivot from a "Test/High-Frequency" trading logic to a **"Sniper/High-Precision"** logic.

The previous configuration was designed to generate a high volume of signals for system stress testing and data collection. However, this led to trading in choppy markets with low Risk/Reward (R/R) ratios. The new "Sniper Mode" enforces strict filtering, requiring deep oversold/overbought conditions, high AI confidence, and favorable market regimes before executing any trade.

## 2. Problem Statement (The "Before" State)

Prior to this implementation, the bot operated under loose constraints:

*   **Low Barriers to Entry:** Strategies generated signals in neutral market zones (e.g., RSI 45-55).
*   **Negative Expected Value:** The system allowed trades with a Risk/Reward ratio of 0.5 (risking $1.00 to make $0.50).
*   **Aggressive AI Bypass:** The "Extreme Condition Bypass" triggered too easily (RSI < 20), forcing trades even when the ML model signaled "Wait".
*   **Binary AI Decisions:** The Reinforcement Learning (PPO) agent was treated as a binary switch (Buy/Hold) without considering the model's confidence/probability level.
*   **Regime Agnostic:** Strategies often ignored the broader market regime (e.g., opening Long positions during Bearish trends).

## 3. Solution Strategy: "Sniper Mode"

The "Sniper Mode" philosophy is defined by: **"Better to miss a trade than to lose money."**

We implemented a 4-layer filter system to achieve this:

1.  **Strategy Layer:** Tightened RSI bands to target only local extrema (reversals).
2.  **Risk Layer:** Increased minimum R/R ratio requirements.
3.  **Intelligence Layer:** Implemented a confidence threshold for the RL Agent.
4.  **Safety Layer:** Restricted the bypass mechanism to true market crash scenarios.

## 4. Technical Implementation Details

### A. Strategy Enhancements

**Files Modified:**
*   `src/strategies/adaptive_ob.py` (Oversold Bounce)
*   `src/strategies/adaptive_str.py` (Short The Rip)

**Changes:**
*   **RSI Baselines:**
    *   *Oversold Bounce:* Entry threshold lowered from **45** to **32**.
    *   *Short The Rip:* Entry threshold raised from **50** to **68**.
*   **Volatility Adjustment:** During high volatility regimes, position sizes are now reduced by **50%** (previously 25%) to preserve capital.
*   **Regime Enforcement:** The `ignore_regime` flag is now `false`. The bot will not open Long positions if the Regime Predictor indicates a "Bearish" trend, unless specific override conditions are met.

### B. Risk Management Overhaul

**Files Modified:**
*   `config/live_trading_config.yaml`
*   `src/core/risk_manager.py`

**Changes:**
*   **Risk/Reward (R/R):**
    *   `min_rr_ratio` increased from 0.5 to **1.5**.
    *   `base_target_rr` increased from 1.5 to **2.0**.
*   **Signal Scoring:** Minimum composite score required to trade increased from 60 to **75**.
*   **Safety Bypass:**
    *   Oversold Bypass tightened: RSI **< 12** (was < 20).
    *   Overbought Bypass tightened: RSI **> 88** (was > 80).

### C. Artificial Intelligence (PPO) Logic

**Files Modified:**
*   `src/ml/adapters/ppo_trading_adapter.py`

**Changes:**
*   **Probability Filtering:** The adapter no longer blindly accepts the model's `action=1` (Buy) output. It now extracts the underlying probability distribution.
*   **Confidence Threshold:** A `CONFIDENCE_THRESHOLD` of **0.75** was introduced.
    *   *Scenario:* Model says "Buy" with 55% confidence → **Ignored** (Log: `WEAK_LONG_IGNORED`).
    *   *Scenario:* Model says "Buy" with 80% confidence → **Executed**.

## 5. Configuration Diff (Before vs. After)

| Parameter | Previous Value (Test Mode) | New Value (Sniper Mode) | Impact |
| :--- | :--- | :--- | :--- |
| `rsi_oversold_threshold` (Bypass) | 20 | **12** | Only triggers on crashes |
| `adaptive_rsi_base` (OB Strategy) | 45 | **32** | Waits for deeper dips |
| `base_target_rr` (Risk) | 1.5 | **2.0** | Higher profit target |
| `lower_bound_rr` (Risk) | 0.8 | **1.2** | Rejects low-quality setups |
| `min_score_to_trade` (ML) | 60 | **75** | Requires higher confluence |
| `rl_confidence_threshold` | N/A (Binary) | **0.75** | Filters weak AI signals |

## 6. Expected Outcomes

1.  **Signal Frequency:** Significant reduction (est. -70%). Expect long periods of silence during choppy markets.
2.  **Win Rate:** Projected increase due to stricter entry conditions.
3.  **Drawdown:** Reduced exposure during "falling knife" scenarios due to lower RSI thresholds and regime filtering.
4.  **Log Clarity:** Logs will explicitly state "WEAK_LONG_IGNORED" or "RSI above threshold", proving the filters are active.

## 7. Future Recommendations

*   **Monitor Bypass Frequency:** If the RSI < 12 threshold is never hit for weeks, consider relaxing slightly to 14 or 15.
*   **Retrain PPO:** The RL agent should be retrained periodically using the new feature set to align its confidence scores with the new strict market regime.

---
*This document is maintained by the Bearish Alpha Bot development team.*
