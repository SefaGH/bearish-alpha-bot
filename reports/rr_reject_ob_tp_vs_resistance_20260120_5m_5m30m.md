# RR-Reject OB: TP vs Multi-TF Resistance Bands

- Cases: **10**
- Reachability OHLCV: `5m` (`data_cache\ohlcv\bingx_BTC_USDT_USDT_5m.csv`)
- Horizons (bars): `12,24,36,48`
- Band timeframes: `5m,30m` | Select policy: `prefer_tf_order`
- Band method preference: `kmeans,smc`

## TP vs Band (Per TF)
### 5m (current TP)
- **TP_BELOW_BAND**: 10
### 30m (current TP)
- **TP_BELOW_BAND**: 10

## Reachability Summary (Selected Band)
### Horizon h=12 (5m)
- Current `stopout_within_h`: 80.0% (8/10)
- Current `touch_tp_within_h`: 0.0% (0/10)
- Current `touch_band_high_within_h`: 0.0% (0/10)
- Current stopout-before-TP: 0.0% (0/10)
- Current stopout-before-band_high: 0.0% (0/10)
- Hybrid `stopout_within_h`: 70.0% (7/10)
- Hybrid `touch_tp_within_h`: 0.0% (0/10)
- Hybrid stopout-before-TP: 0.0% (0/10)

### Horizon h=24 (5m)
- Current `stopout_within_h`: 80.0% (8/10)
- Current `touch_tp_within_h`: 0.0% (0/10)
- Current `touch_band_high_within_h`: 0.0% (0/10)
- Current stopout-before-TP: 0.0% (0/10)
- Current stopout-before-band_high: 0.0% (0/10)
- Hybrid `stopout_within_h`: 70.0% (7/10)
- Hybrid `touch_tp_within_h`: 0.0% (0/10)
- Hybrid stopout-before-TP: 0.0% (0/10)

### Horizon h=36 (5m)
- Current `stopout_within_h`: 80.0% (8/10)
- Current `touch_tp_within_h`: 0.0% (0/10)
- Current `touch_band_high_within_h`: 0.0% (0/10)
- Current stopout-before-TP: 0.0% (0/10)
- Current stopout-before-band_high: 0.0% (0/10)
- Hybrid `stopout_within_h`: 70.0% (7/10)
- Hybrid `touch_tp_within_h`: 0.0% (0/10)
- Hybrid stopout-before-TP: 0.0% (0/10)

### Horizon h=48 (5m)
- Current `stopout_within_h`: 80.0% (8/10)
- Current `touch_tp_within_h`: 10.0% (1/10)
- Current `touch_band_high_within_h`: 0.0% (0/10)
- Current stopout-before-TP: 0.0% (0/10)
- Current stopout-before-band_high: 0.0% (0/10)
- Hybrid `stopout_within_h`: 70.0% (7/10)
- Hybrid `touch_tp_within_h`: 0.0% (0/10)
- Hybrid stopout-before-TP: 0.0% (0/10)

## Cases (Preview)
| case_ts | reach_bar_open_ts | entry | tp_current | selected_band_tf | selected_band_low | selected_band_high | selected_tp_current_vs_band_code | hybrid_tp | selected_tp_hybrid_vs_band_code | h12_cur_stopout_within_h | h12_cur_touch_tp_within_h | h12_hybrid_stopout_within_h | h12_hybrid_touch_tp_within_h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01-20 22:40:14 | 2026-01-20T22:40:00Z | 88519.4 | 89501.73 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89802.15281599996 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:42:21 | 2026-01-20T22:40:00Z | 88546.5 | 89528.83 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89829.25281600007 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:43:57 | 2026-01-20T22:40:00Z | 88407.3 | 89389.63 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89689.03281600002 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:45:02 | 2026-01-20T22:45:00Z | 88548.7 | 89531.03 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89830.43281600002 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:47:10 | 2026-01-20T22:45:00Z | 88357.9 | 89340.23 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89617.99004799996 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:49:18 | 2026-01-20T22:45:00Z | 88484.3 | 89466.63 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89744.39004800002 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 22:50:54 | 2026-01-20T22:50:00Z | 88285.5 | 89267.83 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89550.21717600006 | TP_IN_BAND | False | False | False | False |
| 2026-01-20 22:53:02 | 2026-01-20T22:50:00Z | 88336.4 | 89318.73 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89601.11717599996 | TP_IN_BAND | True | False | False | False |
| 2026-01-20 22:55:10 | 2026-01-20T22:55:00Z | 88446.8 | 89429.13 | 5m | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89711.517176 | TP_IN_BAND | True | False | True | False |
| 2026-01-20 23:30:58 | 2026-01-20T23:30:00Z | 88196.1 | 89432.66 | 5m | 89513.57978571429 | 90052.27735714287 | TP_BELOW_BAND | 89832.25025250002 | TP_IN_BAND | False | False | False | False |
