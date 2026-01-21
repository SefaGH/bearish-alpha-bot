# OB RR-Reject Audit (2026-01-20)

## Executive Summary
- Log: `logs\live_trading_20260120_220231_509146.log`
- Input cases: `rr_rejected_ob_cases_20260120.jsonl`
- Cases: **10** (matched to log rejects: **10/10**)
- RR(cur) mean=2.0833 median=2.0833 | RR(required) mean=2.5500 median=2.5500
- Constant RR driver: `tp_atr_mult/sl_atr_mult` = 2.5/1.2 = **2.0833**
- Offline models (pass counts): TP-only **10/10**, SL-only **10/10**, Hybrid **10/10**

## Evidence Anchors (Log Line Numbers)
| ts | reject | ob_rr | trigger_diag | volume_decision | signal_enriched | ppo_decision |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01-20 22:40:14 | 3610 | 3581 | 3576 | 3587 | 3595 | 3592 |
| 2026-01-20 22:42:21 | 3846 | 3817 | 3812 | 3823 | 3831 | 3828 |
| 2026-01-20 22:43:57 | 4027 | 3990 |  | 3996 | 4012 | 4009 |
| 2026-01-20 22:45:02 | 4150 | 4113 |  | 4119 | 4135 | 4132 |
| 2026-01-20 22:47:10 | 4389 | 4360 |  | 4366 | 4374 | 4371 |
| 2026-01-20 22:49:18 | 4606 | 4577 |  | 4583 | 4591 | 4588 |
| 2026-01-20 22:50:54 | 4797 | 4760 | 4755 | 4766 | 4782 | 4779 |
| 2026-01-20 22:53:02 | 5033 | 5004 | 4999 | 5010 | 5018 | 5015 |
| 2026-01-20 22:55:10 | 5249 | 5220 | 5215 | 5226 | 5234 | 5231 |
| 2026-01-20 23:30:58 | 8234 | 8203 |  | 8210 | 8219 | 8216 |

## Constant RR Phenomenon (Code + Config)
- Startup log prints OB config: `tp_atr_mult=2.5`, `sl_atr_mult=1.2` -> intended RR ~ 2.0833.
- Strategy SL/TP derivation uses ATR multipliers and realigns TP to preserve intended RR (`src/strategies/adaptive_ob.py:1245`).
- RiskRewardRatioRule 'Actual RR' uses only `entry/stop/target` (no spread/fees) (`src/core/risk_rules.py:633`).

## Unused / Under-used Parameters Inventory (Log -> Usage)
| Parameter | Seen in log | Used in OB SL/TP? | Used in Actual RR? | Notes / Evidence |
|---|---:|---:|---:|---|
| volume_strength | Yes | No | No | Logged in `[Signal Enriched]` and `volume_decision_check`; used for volume gating/telemetry (`src/core/strategy_coordinator.py:5465`). |
| momentum_strength | Yes | No | No | Logged in `[Signal Enriched]`; not used by OB SL/TP. |
| bid/ask spread | Yes | No | No | Logged in `[TRIGGER-DIAG]` (`src/core/market_data_pipeline.py:1574`); not used in RR computation. |
| volume_ratio_short/medium/combined | Yes | No | No | Logged via `volume_decision_check`; not consumed by RR/SLTP. |

Note: `Vol=... [BUCKET]` in these logs refers to **volume strength**, not volatility.

## Hybrid Model Sensitivity (This Tool)
- strength range: min=0.36 max=0.42 mean=0.40
- stop_scale range: min=1.048 max=1.081 mean=1.059
- spread range: min=0.20 max=0.20 mean=0.20
- Formula: `stop_scale = 1 + 0.6*(0.5 - strength)` and `stop_dist = risk*stop_scale + 2*spread`

## Aggregate Deltas (Absolute)
| metric | tp_only ΔTP | sl_only ΔSL | hybrid ΔTP / ΔSL |
|---|---:|---:|---:|
| mean | 225.74 | 88.53 | 300.19 / 29.20 |
| median | 220.05 | 86.29 | 290.89 / 27.78 |

## Cases (Current vs Proposed)
| ts | entry | SL(cur) | TP(cur) | RR(cur) | RR(req) | vol_str | mom_str | spread | TP-only TP | SL-only SL | Hybrid TP | Hybrid SL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-20 22:40:14 | 88519.40 | 88047.88 | 89501.73 | 2.0833 | 2.5500 | 0.33 | 0.45 | 0.20 | 89721.78 | 88134.17 | 89802.15 | 88016.36 |
| 2026-01-20 22:42:21 | 88546.50 | 88074.98 | 89528.83 | 2.0833 | 2.5500 | 0.33 | 0.45 | 0.20 | 89748.88 | 88161.27 | 89829.25 | 88043.46 |
| 2026-01-20 22:43:57 | 88407.30 | 87935.78 | 89389.63 | 2.0833 | 2.5500 | 0.33 | 0.45 | nan | 89609.68 | 88022.07 | 89689.03 | 87904.66 |
| 2026-01-20 22:45:02 | 88548.70 | 88077.18 | 89531.03 | 2.0833 | 2.5500 | 0.33 | 0.45 | nan | 89751.08 | 88163.47 | 89830.43 | 88046.06 |
| 2026-01-20 22:47:10 | 88357.90 | 87886.38 | 89340.23 | 2.0833 | 2.5500 | 0.39 | 0.45 | nan | 89560.28 | 87972.67 | 89617.99 | 87863.75 |
| 2026-01-20 22:49:18 | 88484.30 | 88012.78 | 89466.63 | 2.0833 | 2.5500 | 0.39 | 0.45 | nan | 89686.68 | 88099.07 | 89744.39 | 87990.15 |
| 2026-01-20 22:50:54 | 88285.50 | 87813.98 | 89267.83 | 2.0833 | 2.5500 | 0.39 | 0.44 | 0.20 | 89487.88 | 87900.27 | 89550.22 | 87789.53 |
| 2026-01-20 22:53:02 | 88336.40 | 87864.88 | 89318.73 | 2.0833 | 2.5500 | 0.39 | 0.44 | 0.20 | 89538.78 | 87951.17 | 89601.12 | 87840.43 |
| 2026-01-20 22:55:10 | 88446.80 | 87975.28 | 89429.13 | 2.0833 | 2.5500 | 0.39 | 0.44 | 0.20 | 89649.18 | 88061.57 | 89711.52 | 87950.83 |
| 2026-01-20 23:30:58 | 88196.10 | 87602.55 | 89432.66 | 2.0833 | 2.5500 | 0.25 | 0.48 | nan | 89709.65 | 87711.17 | 89832.25 | 87554.47 |

## Deep Dives (3 Cases)
### 2026-01-20 22:40:14 BTC/USDT:USDT
- Entry=88519.40 SL(cur)=88047.88 TP(cur)=89501.73 RR(cur)=2.0833 RR(req)=2.5500
- VolStrength=0.33 MomStrength=0.45 Spread=0.19999999999708962 TriggerSrc=mid
- Recommended (hybrid): SL=88016.36 TP=89802.15 RR=2.5500
- Rationale: hybrid: strength=(volume_strength+momentum_strength)/2=0.39; stop_scale=1+0.6*(0.5-strength)=1.066; stop_dist=risk*stop_scale + 2*spread=503.04 (spread=0.20); TP=entry + stop_dist*required_rr; SL=entry - stop_dist

### 2026-01-20 22:49:18 BTC/USDT:USDT
- Entry=88484.30 SL(cur)=88012.78 TP(cur)=89466.63 RR(cur)=2.0833 RR(req)=2.5500
- VolStrength=0.39 MomStrength=0.45 Spread=None TriggerSrc=mid
- Recommended (hybrid): SL=87990.15 TP=89744.39 RR=2.5500
- Rationale: hybrid: strength=(volume_strength+momentum_strength)/2=0.42; stop_scale=1+0.6*(0.5-strength)=1.048; stop_dist=risk*stop_scale + 2*spread=494.15 (spread=0.00); TP=entry + stop_dist*required_rr; SL=entry - stop_dist

### 2026-01-20 23:30:58 BTC/USDT:USDT
- Entry=88196.10 SL(cur)=87602.55 TP(cur)=89432.66 RR(cur)=2.0833 RR(req)=2.5500
- VolStrength=0.25 MomStrength=0.48 Spread=None TriggerSrc=mid
- Recommended (hybrid): SL=87554.47 TP=89832.25 RR=2.5500
- Rationale: hybrid: strength=(volume_strength+momentum_strength)/2=0.36; stop_scale=1+0.6*(0.5-strength)=1.081; stop_dist=risk*stop_scale + 2*spread=641.63 (spread=0.00); TP=entry + stop_dist*required_rr; SL=entry - stop_dist

## Notes / Limitations
- No forward-window backtest included (OHLCV source not derived from this log alone).
- `[TRIGGER-DIAG]` logging is throttled; bid/ask spread is unavailable for some cases in this run.
- Offline levels do not apply exchange tick-size rounding or fees/slippage unless explicitly modeled.
