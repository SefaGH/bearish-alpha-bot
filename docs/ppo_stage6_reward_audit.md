# Stage 6 — Reward & Environment Audit (Live-Parity PPO Env)

Goal: explain the flat bias by inspecting reward structure and reference policies.

Environment reward (RLTradingEnv / Gym wrapper):
- Action space: {0: flat, 1: full long}.
- Reward per step: `log_return(bot_pv) - log_return(benchmark_buyhold_pv)`, then clipped to [-1, 1], scaled by 1.0.
- Fees: 0.0006 per trade; no trade penalty or idle cost; stop-out if PV < 50% of initial.
- Benchmark = buy-and-hold from t0, so rewards favor outperforming HODL, not absolute profit.

Reference policy audit (dataset: `data/training/BTC_USDT_USDT_1h_liveparity_test.npz`, initial_balance=10k):
```
python src/tools/ppo_reward_audit.py --dataset data/training/BTC_USDT_USDT_1h_liveparity_test.npz
```
- Always flat: avg_reward 2.69e-4, std_reward 5.01e-3, return 0.0, DD 0.0, exposure 0.0.
- Always long: avg_reward 2.69e-4, std_reward 5.01e-3, return -21.2%, DD -25.5%, exposure 1.0.
- Random: avg_reward -3.49e-5, std_reward 5.01e-3, return -22.6%, DD -24.6%, exposure ~0.51.

Findings:
- Reward signal is nearly the same mean for flat vs long despite large PnL differences; clipping plus benchmark-relative formulation yields low gradient toward taking risk.
- Because benchmark buy-and-hold often outperforms, any long sequence that lags HODL is penalized; flat incurs only the benchmark gap, which can still produce small positive clipped rewards when volatility is muted.
- No positive incentive for controlled exposure; trade penalties are zero, but reward clipping + benchmark-relative objective can make “do nothing” competitive.

Recommended adjustments (configurable, not yet applied):
1) Scale reward to emphasize absolute improvement over zero, not just benchmark gap (e.g., mix 50% absolute log-return, 50% relative-to-benchmark).
2) Reduce clipping aggressiveness (or raise clip to e.g., 5.0) so strong positive steps aren’t flattened.
3) Consider adding mild holding bonus when in-position to counter the flat bias if benchmark is being used (or remove benchmark term).
4) Keep fees realistic but consider lowering for training to avoid discouraging any action on noisy data.

Conclusion:
- The current reward strongly favors avoiding underperformance vs HODL; with modest clipping, PPO converges to flat. Adjusting reward composition and retraining is required before enabling gating.
