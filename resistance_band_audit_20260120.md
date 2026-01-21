# Resistance Band Audit

## Inputs
- Exchange: `bingx`
- Symbol: `BTC/USDT:USDT`
- Timeframes: `5m,30m`
- Overlay PNG: `resistance_overlay_20260120.png`

## Lookahead-Safe Design
- Pivot highs are *confirmed* only after `pivot_right` bars; at eval index `t`, we only use pivots with `i <= t - pivot_right`.
- `exclude_last` is a live-safety guard: in `--live` it shifts evaluation back N bars; offline it acts as an extra pivot-age/confirmation delay (scaled per timeframe).

## Parameters
- `pivot_left`: `3`
- `pivot_right`: `3`
- `sr_lookback_bars`: `300`
- `band_mode`: `pct`
- `band_pct`: `0.003`
- `atr_period`: `14`
- `band_atr_mult`: `0.2`
- `smc_cluster_pct`: `0.0015`
- `kmin`: `3`
- `kmax`: `8`
- `exclude_last_input`: `12`
- `exclude_last_effective_bars`: `{'5m': 12, '30m': 2}`
- `eval_horizon_bars`: `0`

## Methods
- `smc`: confirmed pivot-high extraction + 1D clustering of pivot highs (liquidity highs), then nearest band above price.
- `kmeans`: KMeans clustering on confirmed pivot-high prices (walk-forward), then nearest cluster band above price.

## Method Agreement (SMC vs KMeans)
- Rows: **29** | Agreement rate: **100.0%**
- Level diff pct: n=29 median=0.00002 min=0.00000 max=0.00046

## Per-Timeframe Summary
### 5m
- `kmeans` rows=25/50 | dist_to_level_pct: n=25 median=0.01429 min=0.00225 max=0.02089
- `smc` rows=25/50 | dist_to_level_pct: n=25 median=0.01427 min=0.00223 max=0.02113

### 30m
- `kmeans` rows=4/8 | dist_to_level_pct: n=4 median=0.01626 min=0.00383 max=0.01986
- `smc` rows=4/8 | dist_to_level_pct: n=4 median=0.01626 min=0.00383 max=0.01986

## Spot-Check (2026-01-20T22:40:00Z)
| timeframe | method | price | nearest_res_level | band_low | band_high | distance_pct_to_level |
|---|---|---|---|---|---|---|
| 5m | kmeans | 88543.00 | 89808.40 | 89538.97 | 90077.83 | 0.01429 |
| 5m | smc | 88543.00 | 89806.90 | 89537.48 | 90076.32 | 0.01427 |

## Samples
### 5m
| timestamp | method | price | nearest_res_level | band_low | band_high |
|---|---|---|---|---|---|
| 2026-01-20T21:30:00Z | kmeans | 89454.60 | 89806.00 | 89536.58 | 90075.42 |
| 2026-01-20T21:30:00Z | smc | 89454.60 | 89765.00 | 89495.71 | 90034.29 |
| 2026-01-20T21:35:00Z | kmeans | 89503.00 | 89806.00 | 89536.58 | 90075.42 |
| 2026-01-20T21:35:00Z | smc | 89503.00 | 89765.00 | 89495.71 | 90034.29 |
| 2026-01-20T21:40:00Z | kmeans | 89505.50 | 89806.00 | 89536.58 | 90075.42 |
| 2026-01-20T21:40:00Z | smc | 89505.50 | 89765.00 | 89495.71 | 90034.29 |
| 2026-01-20T21:45:00Z | kmeans | 89401.80 | 89809.32 | 89539.89 | 90078.75 |
| 2026-01-20T21:45:00Z | smc | 89401.80 | 89810.00 | 89540.57 | 90079.43 |
| 2026-01-20T21:50:00Z | kmeans | 89446.20 | 89809.32 | 89539.89 | 90078.75 |
| 2026-01-20T21:50:00Z | smc | 89446.20 | 89810.00 | 89540.57 | 90079.43 |

### 30m
| timestamp | method | price | nearest_res_level | band_low | band_high |
|---|---|---|---|---|---|
| 2026-01-20T22:00:00Z | kmeans | 89479.70 | 89822.60 | 89553.13 | 90092.07 |
| 2026-01-20T22:00:00Z | smc | 89479.70 | 89822.60 | 89553.13 | 90092.07 |
| 2026-01-20T22:30:00Z | kmeans | 88462.00 | 89822.60 | 89553.13 | 90092.07 |
| 2026-01-20T22:30:00Z | smc | 88462.00 | 89822.60 | 89553.13 | 90092.07 |
| 2026-01-20T23:00:00Z | kmeans | 88073.70 | 89822.60 | 89553.13 | 90092.07 |
| 2026-01-20T23:00:00Z | smc | 88073.70 | 89822.60 | 89553.13 | 90092.07 |
| 2026-01-20T23:30:00Z | kmeans | 88385.60 | 89822.60 | 89553.13 | 90092.07 |
| 2026-01-20T23:30:00Z | smc | 88385.60 | 89822.60 | 89553.13 | 90092.07 |

## Notes
- KMeans clusters only historical confirmed pivot highs within `sr_lookback_bars` for each eval point (walk-forward).
