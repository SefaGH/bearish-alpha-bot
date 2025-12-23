# Stage 6 — PPO Gating Impact Backtest (Live-Parity Dataset)

Command to reproduce:
```bash
python src/tools/ppo_gating_impact_backtest.py \
  --model artifacts/ppo/ppo_trading_agent.zip \
  --dataset data/training/BTC_USDT_USDT_1h_liveparity_test.npz \
  --spec artifacts/ppo/ppo_trading_agent.obs_spec.json \
  --baseline-mode always_long \
  --conf-threshold 0.60 --min-margin 0.0 \
  --output-json tmp/gating_report.json
```

Summary (baseline = always-long on same OHLCV window):
- p_long distribution: min 1.4e-07, max 0.8946, mean 0.0088, std 0.0495 → heavily flat-biased.
- Vetoes: 899 of 900 baseline long points (veto rate ≈ 0.999). Vetoes on losing trades: 458.

Performance:
- Baseline (no gating): return -21.5%, maxDD -25.5%, trades=1 (buy-and-hold), exposure=1.0.
- PPO-gated: return -0.87%, maxDD -0.87%, trades=2, exposure=0.11%.
- Fees: baseline 6.0, gated 12.0 (two tiny trades).

Interpretation:
- PPO essentially forces “stay flat” on this dataset; it vetoes ~100% of longs.
- Gating prevents large drawdown but also eliminates nearly all participation; this is a degenerate behavior, not a balanced filter.

Recommendation:
- **Do not enable gating for production decisions yet.** Keep PPO in monitor/shadow.
- If gating is desired, only use as veto (never promote) and expect near-zero participation until the policy is improved.
- Focus on improving the policy (Part B/C) before reconsidering thresholds; lowering thresholds would still pass almost no trades given the current p_long distribution.
