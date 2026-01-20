# MR Stop Scale Audit

## Executive Summary
- Kok neden: vwap_std (1m close std) bps olcegi 4-8 bps bandinda oldugu icin stop = delta*std (20260118 logu: delta=1.0) direkt mikro stop uretiyor; veri: 20260118 run'inda 68/68 sinyal <15 bps ve std_bps_theory p50 ~5.91-6.03.
- Geometri etkisi ikincil: LOW bucket'ta k_implied_obs p50=0.95, k_implied_obs<1 orani %45.6; dist_outside arttikca stop bps azaliyor (corr -0.18).
- ATR fallback pratikte yok: std-based stop her sinyalde mevcut (20260118 vwap_std/atr eksigi 0); 20251227 run'inda vwap_std/atr telemetry eksik (belirsizlik).
- Reward-consistent outlier yok: target_bps>25 & stop_bps<5 kosulunda 0 vaka.

## Formul Haritasi (SSOT: kod referanslari)
- vwap_std ve bantlar: rolling std(close, lookback) ve vwap +/- band_mult * vwap_std (price birimi). `src/core/indicators.py:116` `src/core/indicators.py:131`.
- ATR (price birimi): true range EWM (period=14). `src/core/indicators.py:37` `src/core/indicators.py:41`.
- MR stop/target:
  - effective_vwap_std controller/pipeline'dan gelir; yoksa (upper-lower)/(2*band_mult) ile turetilir. `src/strategies/mean_reversion.py:946` `src/strategies/mean_reversion.py:968`.
  - Stop band-anchored: long `stop = lower - delta*std`, short `stop = upper + delta*std`, safety clamp ile entry tarafi garanti edilir. `src/strategies/mean_reversion.py:980` `src/strategies/mean_reversion.py:992`.
  - ATR fallback: stop yoksa `price +/- 1.5*ATR`. `src/strategies/mean_reversion.py:994` `src/strategies/mean_reversion.py:999`.
  - Target = vwap. `src/strategies/mean_reversion.py:944` `src/strategies/mean_reversion.py:1002`.
- stop_loss_std_delta config default 0.5. `src/strategies/mean_reversion.py:27` `src/strategies/mean_reversion.py:30`.
- Dynamic controller std/lookup: dinamik lookback ile vwap/std hesaplayabilir. `src/strategies/mr_controller.py:369` `src/strategies/mr_controller.py:383`.
- Gate stop_pct hesaplari (entry-stop)/entry: `src/core/strategy_coordinator.py:3747` `src/core/strategy_coordinator.py:3762`.

## Olcek sanity-check (std/atr -> stop uretimi)
Teorik std_bps = vwap_std / entry * 1e4, teorik stop_bps = delta * std_bps (entry-anchored).

20260118 (delta=1.0, log kaniti: `logs/live_trading_20260118_202844_781510.log:106`)

| bucket | std_bps_theory p10/p50/p90 | observed_stop_bps p10/p50/p90 | k_implied_obs p10/p50/p90 | n |
| --- | --- | --- | --- | --- |
| (missing) | 4.44 / 5.91 / 7.06 | 4.44 / 5.91 / 7.06 | 1.00 / 1.00 / 1.00 | 20 |
| LOW | 5.28 / 6.03 / 7.97 | 3.84 / 5.59 / 6.56 | 0.65 / 0.95 / 1.00 | 48 |

- Mikro stop: 20260118'de 68/68 <15 bps. 20251227'de 64/120 <15 bps (37 satirda stop bilgisi yok).
- vwap_std/atr metrikleri 20260118'de tam, 20251227 universe CSV'de eksik (std_bps ve atr_bps yok). `reports/mr_stop_scale_examples.csv` bu farki net gosteriyor.

## Geometri iptali testi (korelasyon + ornekler)
- Korelasyon (20260118):
  - observed_stop_bps vs dist_outside_bps: -0.18
  - observed_stop_bps vs band_width_bps: +0.81
  - k_implied_obs vs dist_outside_bps: -0.23

Yorum: stop mesafesi esasen band_width/vwap_std ile olcekleniyor; dist_outside arttikca band-anchored stop mesafesi azaliyor (geometri iptali). Ancak etki ikincil.

Ornekler (band anchor vs entry anchor, 10 satir):

| ts_ms | side | entry_price | lower | upper | vwap_std | stop_band | stop_entry | stop_effective | obs_stop_bps | stop_bps_band | stop_bps_entry | dist_outside_bps | band_width_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1768773332000 | short | 95371.10 | 95229.41 | 95385.03 | 43.23 | 95428.26 | 95414.33 | 95428.26 | 5.99 | 5.99 | 4.53 | -1.46 | 16.33 |
| 1768777883000 | long | 94993.90 | 94997.28 | 95375.41 | 105.04 | 94892.24 | 94888.86 | 94892.24 | 10.70 | 10.70 | 11.06 | 0.36 | 39.73 |
| 1768773941000 | short | 95394.40 | 95219.57 | 95389.73 | 47.27 | 95436.99 | 95441.67 | 95436.99 | 4.46 | 4.46 | 4.95 | 0.49 | 17.85 |
| 1768770097055 | long | 95260.30 | 95267.44 | 95480.37 | 59.15 | 95208.29 | 95201.15 | 95208.29 | 5.70 | 5.46 | 6.21 | 0.75 | 22.33 |
| 1768770193000 | long | 95260.30 | 95267.44 | 95480.37 | 59.15 | 95208.29 | 95201.15 | 95208.29 | 6.10 | 5.46 | 6.21 | 0.75 | 22.33 |
| 1768774294000 | short | 95420.90 | 95221.84 | 95403.21 | 50.38 | 95453.60 | 95471.28 | 95453.60 | 3.98 | 3.43 | 5.28 | 1.85 | 19.03 |
| 1768774550000 | short | 95470.40 | 95252.63 | 95449.30 | 54.63 | 95503.93 | 95525.03 | 95503.93 | 3.51 | 3.51 | 5.72 | 2.21 | 20.63 |
| 1768774647000 | short | 95470.40 | 95252.63 | 95449.30 | 54.63 | 95503.93 | 95525.03 | 95503.93 | 4.58 | 3.51 | 5.72 | 2.21 | 20.63 |
| 1768774583097 | short | 95470.40 | 95252.63 | 95449.30 | 54.63 | 95503.93 | 95525.03 | 95503.93 | 4.14 | 3.51 | 5.72 | 2.21 | 20.63 |
| 1768777659000 | long | 94979.30 | 95172.90 | 95440.00 | 74.19 | 95098.71 | 94905.11 | 94905.11 | 2.04 | -12.57 | 7.81 | 20.38 | 28.02 |

Not: Son satirda dist_outside buyuk oldugu icin stop_band entry'nin ustune cikiyor (stop_bps_band negatif); safety clamp stop_effective = stop_entry yapiyor.

## ATR fallback neden yok (kosul analizi)
- Kod: ATR sadece stop_loss_price None veya non-finite ise devreye giriyor. `src/strategies/mean_reversion.py:994` `src/strategies/mean_reversion.py:999`.
- 20260118 universe'da vwap_std ve atr eksigi 0; dolayisiyla std-based stop her sinyalde mevcut.
- 20251227 universe CSV'de vwap_std/atr yok; bu telemetry gap olabilir (run logunda vwap_std cikmiyor). ATR fallback'in runtime'da kullanilip kullanilmadigi teyit edilemiyor.

## Sonuc: model duzeltmesi icin 2-3 minimal yon (kod degistirmeden oneri)
1) Vol proxy kalibrasyonu: std_bps bu rejimde 5-8 bps; tradeable stop istiyorsaniz delta veya std proxy (lookback/timeframe) kalibrasyonu gerekecek.
2) Geometri etkisi: band-anchored stop dist_outside kadar kuculuyor; entry-anchored stop veya dist_outside'i ayri bir gate metrigi olarak kullanma secenegi dusunulebilir.
3) Telemetry tamamlama: 20251227 gibi run'larda vwap_std/atr/stop_loss_std_delta eksik; bu alanlar olmadan k_implied ve scale analizi yapilamiyor.

## Karar sorulari (Evet/Hayir + gerekce)
1) Vol proxy (std/atr) birim/olcek hatasi var mi?
   - Hayir (20260118): vwap_std price biriminde, std_bps_theory ile observed_stop_bps birebir/uyumlu. 20251227 icin veri yok (belirsiz).
2) Mikro-stop'un ana nedeni proxy kucuklugu mu, geometri mi, clamp mi?
   - Ana neden proxy olcegi kucuk (std_bps ~5-8). Geometri ikincil (k_implied p50 0.95, corr -0.18). Clamp yalniz dist_outside > delta*std durumunda devreye giriyor (ornek: ts_ms=1768777659000).
3) ATR fallback pratikte neden devreye girmiyor? Tasarim olarak girmeli mi?
   - Devreye girmiyor cunku std-based stop her sinyalde mevcut. Tasarim olarak sadece std yoksa girmesi beklenir; burada std var.
4) Reward-consistent istisna mekanizmasi gercekten kucuk bir alt kumeyi mi kurtariyor?
   - Hayir: target_bps>25 ve stop_bps<5 kosulunda 0 vaka (20260118+20251227).
