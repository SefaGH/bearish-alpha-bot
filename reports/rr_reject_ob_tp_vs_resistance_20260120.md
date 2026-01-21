# RR-Reject OB: TP vs 5m Resistance Band

- Cases: **10**
- Time alignment: case ts -> 5m bar open-time floor (`bar_open_ts`).
- Band selection: `5m` using `kmeans,smc` (first available).
- Horizon: **12 bars** (5m) for reachability (MFE/MAE/touch/stopout).

## TP vs Band (Current)
- **TP_BELOW_BAND**: 10

## Reachability (Within Horizon)
- Current `touch_tp_within_h`: 0.0% (0/10)
- Current `stopout_within_h`: 80.0% (8/10)
- Current `touch_band_high_within_h`: 0.0% (0/10)
- Hybrid `touch_tp_within_h`: 0.0% (0/10)
- Hybrid `stopout_within_h`: 70.0% (7/10)

## Cases (Preview)
| case_ts | bar_open_ts | entry | tp_current | band_low | band_high | tp_current_vs_band_code | hybrid_tp | tp_hybrid_vs_band_code | cur_touch_tp_within_h | cur_stopout_within_h | hybrid_touch_tp_within_h | hybrid_stopout_within_h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01-20 22:40:14 | 2026-01-20T22:40:00Z | 88519.4 | 89501.73 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89802.15281599996 | TP_IN_BAND | False | True | False | True |
| 2026-01-20 22:42:21 | 2026-01-20T22:40:00Z | 88546.5 | 89528.83 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89829.25281600007 | TP_IN_BAND | False | True | False | True |
| 2026-01-20 22:43:57 | 2026-01-20T22:40:00Z | 88407.3 | 89389.63 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89689.03281600002 | TP_IN_BAND | False | True | False | True |
| 2026-01-20 22:45:02 | 2026-01-20T22:45:00Z | 88548.7 | 89531.03 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89830.43281600002 | TP_IN_BAND | False | True | False | True |
| 2026-01-20 22:47:10 | 2026-01-20T22:45:00Z | 88357.9 | 89340.23 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89617.99004799996 | TP_IN_BAND | False | True | False | True |
| 2026-01-20 22:49:18 | 2026-01-20T22:45:00Z | 88484.3 | 89466.63 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89744.39004800002 | TP_IN_BAND | False | True | False | True |
| 2026-01-20 22:50:54 | 2026-01-20T22:50:00Z | 88285.5 | 89267.83 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89550.21717600006 | TP_IN_BAND | False | False | False | False |
| 2026-01-20 22:53:02 | 2026-01-20T22:50:00Z | 88336.4 | 89318.73 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89601.11717599996 | TP_IN_BAND | False | True | False | False |
| 2026-01-20 22:55:10 | 2026-01-20T22:55:00Z | 88446.8 | 89429.13 | 89538.9748 | 90077.8252 | TP_BELOW_BAND | 89711.517176 | TP_IN_BAND | False | True | False | True |
| 2026-01-20 23:30:58 | 2026-01-20T23:30:00Z | 88196.1 | 89432.66 | 89513.57978571429 | 90052.27735714287 | TP_BELOW_BAND | 89832.25025250002 | TP_IN_BAND | False | False | False | False |
