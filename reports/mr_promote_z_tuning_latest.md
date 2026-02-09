# MR PROMOTE min_z_score Tuning Report

- Generated at (UTC): `2026-02-09T17:01:55.605839+00:00`
- Files: `60`
- Eval cases: `303`
- Base gate pool: `35/303` (`11.6%`)

## Sweep

| min_z_score | pass_count | pass_rate_vs_base | pass_rate_vs_all_eval |
| --- | --- | --- | --- |
| 1.80 | 13 | 37.1% | 4.3% |
| 2.00 | 7 | 20.0% | 2.3% |
| 2.20 | 1 | 2.9% | 0.3% |

## Trade-Labeled Sweep (TRADE_CLOSED)

- Trade coverage: scope=`54`, with_promotion_meta=`0`, base=`0`

## Recommendation

- keep 2.00 (no safer threshold with acceptable opportunity retention)

## Telemetry Gaps

- trend_veto / ema_stack data not reconstructed in this sweep
- volume_strength gate not reconstructed in this sweep
- shock_state gate not reconstructed in this sweep
- no direct PnL label in mr_recheck_eval; result is opportunity analysis, not win-rate optimization
- TRADE_CLOSED.entry_metadata.promotion_override missing on part of historical trades
