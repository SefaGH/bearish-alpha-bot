# RR-Reject OB: TP vs Multi-TF Resistance Bands

- Cases: **10**
- Reachability OHLCV: `5m` (`data_cache\ohlcv\bingx_BTC_USDT_USDT_5m.csv`)
- Horizons (bars): `12,24,36,48`
- Band timeframes: `1m,5m,30m` | Select policy: `closest_level`
- Band method preference: `kmeans,smc`

## TP vs Band (Per TF)
### 1m (current TP)
- **TP_IN_BAND**: 8
- **TP_BELOW_BAND**: 1
- **TP_ABOVE_BAND**: 1
### 5m (current TP)
- **TP_BELOW_BAND**: 8
- **TP_IN_BAND**: 2
### 30m (current TP)
- **TP_BELOW_BAND**: 10

## Reachability Summary (Selected Band)
### Horizon h=12 (5m)
- `current`: stopout=8/10 | tp_touch=0/10 | band_high_touch=0/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0
- `hybrid`: stopout=7/10 | tp_touch=0/10 | band_high_touch=0/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0
- `sl_only`: stopout=9/10 | tp_touch=0/10 | band_high_touch=0/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0

### Horizon h=24 (5m)
- `current`: stopout=8/10 | tp_touch=0/10 | band_high_touch=1/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0
- `hybrid`: stopout=7/10 | tp_touch=0/10 | band_high_touch=1/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0
- `sl_only`: stopout=9/10 | tp_touch=0/10 | band_high_touch=1/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0

### Horizon h=36 (5m)
- `current`: stopout=8/10 | tp_touch=0/10 | band_high_touch=1/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0
- `hybrid`: stopout=7/10 | tp_touch=0/10 | band_high_touch=1/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0
- `sl_only`: stopout=9/10 | tp_touch=0/10 | band_high_touch=1/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0

### Horizon h=48 (5m)
- `current`: stopout=8/10 | tp_touch=1/10 | band_high_touch=1/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0
- `hybrid`: stopout=7/10 | tp_touch=0/10 | band_high_touch=1/10
  - stopout->TP (both): 0/0
  - stopout->band_high (both): 0/0
- `sl_only`: stopout=9/10 | tp_touch=1/10 | band_high_touch=1/10
  - stopout->TP (both): 1/1
  - stopout->band_high (both): 0/0

## Cases (Preview)
| case_ts | reach_bar_open_ts | entry | tp_current | selected_band_tf | selected_band_low | selected_band_high | selected_tp_current_vs_band_code | hybrid_tp | selected_tp_hybrid_vs_band_code | h12_cur_stopout_within_h | h12_cur_touch_tp_within_h | h12_hybrid_stopout_within_h | h12_hybrid_touch_tp_within_h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01-20 22:40:14 | 2026-01-20T22:40:00Z | 88519.4 | 89501.73 | 1m | 89287.93926363636 | 89825.27891818181 | TP_IN_BAND | 89802.15281599996 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:42:21 | 2026-01-20T22:40:00Z | 88546.5 | 89528.83 | 1m | 89287.93926363636 | 89825.27891818181 | TP_IN_BAND | 89829.25281600007 | TP_ABOVE_BAND | True | False | True | False |
| 2026-01-20 22:43:57 | 2026-01-20T22:40:00Z | 88407.3 | 89389.63 | 1m | 89287.93926363636 | 89825.27891818181 | TP_IN_BAND | 89689.03281600002 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:45:02 | 2026-01-20T22:45:00Z | 88548.7 | 89531.03 | 1m | 89287.93926363636 | 89825.27891818181 | TP_IN_BAND | 89830.43281600002 | TP_ABOVE_BAND | True | False | True | False |
| 2026-01-20 22:47:10 | 2026-01-20T22:45:00Z | 88357.9 | 89340.23 | 1m | 89287.93926363636 | 89825.27891818181 | TP_IN_BAND | 89617.99004799996 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:49:18 | 2026-01-20T22:45:00Z | 88484.3 | 89466.63 | 1m | 89287.93926363636 | 89825.27891818181 | TP_IN_BAND | 89744.39004800002 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:50:54 | 2026-01-20T22:50:00Z | 88285.5 | 89267.83 | 1m | 89273.32968888889 | 89810.58142222223 | TP_BELOW_BAND | 89550.21717600006 | TP_IN_BAND | False | False | False | False |
| 2026-01-20 22:53:02 | 2026-01-20T22:50:00Z | 88336.4 | 89318.73 | 1m | 89273.32968888889 | 89810.58142222223 | TP_IN_BAND | 89601.11717599996 | TP_IN_BAND | True | False | False | False |
| 2026-01-20 22:55:10 | 2026-01-20T22:55:00Z | 88446.8 | 89429.13 | 1m | 89273.32968888889 | 89810.58142222223 | TP_IN_BAND | 89711.517176 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 23:30:58 | 2026-01-20T23:30:00Z | 88196.1 | 89432.66 | 1m | 88202.16396666666 | 88732.96936666667 | TP_ABOVE_BAND | 89832.25025250002 | TP_ABOVE_BAND | False | False | False | False |
