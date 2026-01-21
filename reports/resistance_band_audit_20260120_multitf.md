# Resistance Band Audit

## Inputs
- Exchange: `bingx`
- Symbol: `BTC/USDT:USDT`
- Timeframes: `1m,5m,30m`
- Overlay PNG: `reports\resistance_overlay_20260120_multitf.png`

## Lookahead-Safe Design
- Pivot highs are *confirmed* only after `pivot_right` bars; at eval index `t`, we only use pivots with `i <= t - pivot_right`.
- `smc_lib` swing highs are *confirmed* only after `smc_swing_length` bars; at eval index `t`, we only use swings with `i <= t - smc_swing_length`.
- `exclude_last` is a live-safety guard: in `--live` it shifts evaluation back N bars; offline it acts as an extra pivot-age/confirmation delay (scaled per timeframe).

## Parameters
- `pivot_left`: `3`
- `pivot_right`: `3`
- `smc_swing_length`: `50`
- `smc_liquidity_range_pct`: `0.01`
- `sr_lookback_bars`: `300`
- `band_mode`: `pct`
- `band_pct`: `0.003`
- `atr_period`: `14`
- `band_atr_mult`: `0.2`
- `smc_cluster_pct`: `0.0015`
- `kmin`: `3`
- `kmax`: `8`
- `exclude_last_input`: `12`
- `exclude_last_effective_bars`: `{'1m': 12, '5m': 3, '30m': 1}`
- `eval_horizon_bars`: `0`

## Methods
- `smc`: confirmed pivot-high extraction + 1D clustering of pivot highs (liquidity highs), then nearest band above price.
- `kmeans`: KMeans clustering on confirmed pivot-high prices (walk-forward), then nearest cluster band above price.
- `smc_lib`: SmartMoneyConcepts README-style `swing_highs_lows` + `liquidity` approximation (confirmed swing highs clustered by `range_percent`).

## Method Agreement (Pairwise)
- Rows: **151**
- `kmeans_smc`: agreement=100.0% | level_diff_pct: n=151 median=0.00007 min=0.00000 max=0.00270

## Per-Timeframe Summary
### 1m
- `kmeans` rows=121/242 | dist_to_level_pct: n=121 median=0.00245 min=0.00001 max=0.01607
- `smc` rows=121/242 | dist_to_level_pct: n=121 median=0.00200 min=0.00001 max=0.01624

### 5m
- `kmeans` rows=25/50 | dist_to_level_pct: n=25 median=0.01400 min=0.00225 max=0.02089
- `smc` rows=25/50 | dist_to_level_pct: n=25 median=0.01424 min=0.00223 max=0.02113

### 30m
- `kmeans` rows=5/10 | dist_to_level_pct: n=5 median=0.01626 min=0.00383 max=0.02124
- `smc` rows=5/10 | dist_to_level_pct: n=5 median=0.01626 min=0.00383 max=0.02159

## Spot-Check (2026-01-20T22:40:00Z)
| timeframe | method | price | nearest_res_level | band_low | band_high | distance_pct_to_level |
|---|---|---|---|---|---|---|
| 1m | kmeans | 88460.60 | 89556.61 | 89287.94 | 89825.28 | 0.01239 |
| 1m | smc | 88460.60 | 89571.20 | 89302.49 | 89839.91 | 0.01255 |
| 5m | kmeans | 88543.00 | 89782.93 | 89513.58 | 90052.28 | 0.01400 |
| 5m | smc | 88543.00 | 89803.80 | 89534.39 | 90073.21 | 0.01424 |

## Samples
### 1m
| timestamp | method | price | nearest_res_level | band_low | band_high |
|---|---|---|---|---|---|
| 2026-01-20T21:30:00Z | kmeans | 89556.50 | 89567.52 | 89298.82 | 89836.22 |
| 2026-01-20T21:30:00Z | smc | 89556.50 | 89580.60 | 89311.86 | 89849.34 |
| 2026-01-20T21:31:00Z | kmeans | 89573.20 | 89796.01 | 89526.62 | 90065.40 |
| 2026-01-20T21:31:00Z | smc | 89573.20 | 89580.60 | 89311.86 | 89849.34 |
| 2026-01-20T21:32:00Z | kmeans | 89568.00 | 89796.01 | 89526.62 | 90065.40 |
| 2026-01-20T21:32:00Z | smc | 89568.00 | 89580.60 | 89311.86 | 89849.34 |
| 2026-01-20T21:33:00Z | kmeans | 89475.00 | 89567.52 | 89298.82 | 89836.22 |
| 2026-01-20T21:33:00Z | smc | 89475.00 | 89580.60 | 89311.86 | 89849.34 |
| 2026-01-20T21:34:00Z | kmeans | 89454.60 | 89567.52 | 89298.82 | 89836.22 |
| 2026-01-20T21:34:00Z | smc | 89454.60 | 89580.60 | 89311.86 | 89849.34 |

### 5m
| timestamp | method | price | nearest_res_level | band_low | band_high |
|---|---|---|---|---|---|
| 2026-01-20T21:30:00Z | kmeans | 89454.60 | 89808.40 | 89538.97 | 90077.83 |
| 2026-01-20T21:30:00Z | smc | 89454.60 | 89806.90 | 89537.48 | 90076.32 |
| 2026-01-20T21:35:00Z | kmeans | 89503.00 | 89808.40 | 89538.97 | 90077.83 |
| 2026-01-20T21:35:00Z | smc | 89503.00 | 89806.90 | 89537.48 | 90076.32 |
| 2026-01-20T21:40:00Z | kmeans | 89505.50 | 89808.40 | 89538.97 | 90077.83 |
| 2026-01-20T21:40:00Z | smc | 89505.50 | 89806.90 | 89537.48 | 90076.32 |
| 2026-01-20T21:45:00Z | kmeans | 89401.80 | 89808.40 | 89538.97 | 90077.83 |
| 2026-01-20T21:45:00Z | smc | 89401.80 | 89806.90 | 89537.48 | 90076.32 |
| 2026-01-20T21:50:00Z | kmeans | 89446.20 | 89808.40 | 89538.97 | 90077.83 |
| 2026-01-20T21:50:00Z | smc | 89446.20 | 89806.90 | 89537.48 | 90076.32 |

### 30m
| timestamp | method | price | nearest_res_level | band_low | band_high |
|---|---|---|---|---|---|
| 2026-01-20T21:30:00Z | kmeans | 89437.80 | 91337.90 | 91063.89 | 91611.91 |
| 2026-01-20T21:30:00Z | smc | 89437.80 | 91368.45 | 91094.34 | 91642.56 |
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
