# MR stratejisinin sinyal üretmemesi — grafik↔log korelasyonu ve bant analizi (kodsuz inceleme)

**Kısıt:** Koda müdahale yok. Sadece mevcut log/CSV/görsel üzerinden analiz + öneri.

**Girdiler:**
- Ana run log: `logs/live_trading_20260107_223016_911605.log`
- UTC pencere: `bot_window_utc.log`
- MR satır extract: `mr_lines_extract.txt`
- Türetilmiş MR metrikleri (291 satır): `mr_metrics.csv`
- Grafik screenshot (BTC 4m): `screenshots/Ekran görüntüsü 2026-01-08 040823.png`

---

## Executive Summary

- Bot **BingX VST** ortamında çalışıyor (kanıt: `[BINGX-ENV] env=vst ... REST_BASE_URL=https://open-api-vst.bingx.com` + `[MODE-BANNER] ... CCXT_SANDBOX=true ...`).
- MR efektif konfig: `timeframe=1m`, `signal_timeframe=5m`, `vwap_lookback=1440`, `band_multiplier=2.0`, `adx_threshold=25` (logdaki `MR Config` satırından).
- MR, 22:30:38–01:05:36 UTC arasında **291 checkpoint** çalıştırmış ve **291/291** kez `Price within bands` + `Action: HOLD` üretmiş (**outside/buy/sell yok**).
- Grafik (4m) üzerinde “range/iniş-çıkış” hissi olmasına rağmen, MR bandı **~%3** genişlikte: `band_width_pct p50=3.209%` (p95=3.348%).
- Gözlenen fiyat aralığı (log px): `px_range=435.4` (≈ **%0.474**) → band genişliğinin yalnızca **~%14.8**’i kadar; bu yüzden fiyat band sınırlarına yaklaşmıyor.
- Nicel tetik kanıtı: `|z|_max=1.2317 < band_multiplier(2.0)` ve `min(dist_to_nearest_band)=586.14` → en “uç” anlarda bile band dışına çıkış yok.
- 23:00 ve 00:45 bölgeleri log ile birebir örtüşüyor: 23:10:42 (dip) ve 00:44:45/00:45:17 (toparlanma) checkpoint’leri hâlâ **within bands + HOLD**.
- ADX (ikincil filtre): checkpoint’lerin **67/291 (~%23.0)** anında `adx > 25`. Band daraltılırsa, bazı sinyaller ADX gate ile ayrıca elenebilir.
- Operasyonel risk: run’da Azure App Configuration çekilememiş (`curl command not found`) → beklenen prod parametreleri bu run’da uygulanmamış olabilir.

---

## 1) Ortam doğrulama (VST mi prod mu?)

`bot_window_utc.log` içinden kanıt satırları:

```
2026-01-07 22:30:19 - [core.ccxt_client] - INFO - [BINGX-ENV] env=vst ccxt_sandbox=True rest_base_url=https://open-api-vst.bingx.com
2026-01-07 22:30:26 - [core.production_coordinator] - WARNING - [MODE-BANNER] TRADING_MODE=live EXECUTION_BACKEND=ccxt BINGX_ENV=vst | CCXT_SANDBOX=true REST_BASE_URL=https://open-api-vst.bingx.com | ...
```

**Yorum:** Grafik prod piyasadan izlendiyse “hareket var ama botta yok” algısı doğabilir. Bu screenshot’taki fiyat etiketi (91,317.8) ise logdaki son px ile çok yakın (01:05:36’da px=91317.6) olduğundan, bu koşuda grafik↔log ürün/fiyat senkronu yüksek görünüyor.

---

## 2) MR konfig doğrulama

`bot_window_utc.log` kanıtı:

```
2026-01-07 22:30:20 - [bearish-alpha-bot] - INFO -   - MR Config: {'execution_profile': 'scalp_tight', 'timeframe': '1m', 'signal_timeframe': '5m', 'vwap_lookback': 1440, 'band_multiplier': 2.0, 'adx_threshold': 25, 'allocation_pct': 0.2}
```

---

## 3) MR checkpoint kapsamı (within/outside ve aksiyon)

`mr_lines_extract.txt` üzerinden MR dağılımı:
- `Price within bands`: **291**
- `Price outside bands`: **0**
- `Cycle complete ... Action: HOLD`: **291**

Örnek kalıp (her checkpoint’te aynı karar):
```
[MeanReversion] Price within bands ... px=..., lower=..., upper=..., adx=..., adx_th=25.0
[MeanReversion] Cycle complete ... Action: HOLD
```

---

## 4) Grafik↔log eşleme (23:00 ve 00:45)

**Zaman notu:** Log UTC. Screenshot x-ekseni 22:00–01:00 aralığını gösteriyor; ayrıca sağdaki fiyat etiketi 91,317.8, log sonundaki px=91317.6 (01:05:36) ile uyumlu → bu eşleme pratikte UTC akışıyla tutarlı.

### Kanıt tablosu (seçilmiş checkpoint’ler)

Tüm metrikler `mr_metrics.csv` türetimi:  
`band_mid=(upper+lower)/2`, `std=(upper-lower)/(2*band_multiplier)`, `z=(px-band_mid)/std`, `dist=min(px-lower, upper-px)`

| Bölge | Timestamp (UTC) | px | lower | upper | band_mid | std | z | dist_to_nearest | Aksiyon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ~23:00 impuls/dönüş (dip) | 2026-01-07 23:10:42 | 90975.0 | 90387.7983 | 93444.9619 | 91916.3801 | 764.2909 | -1.2317 | 587.2017 | HOLD |
| ~23:00 impuls/dönüş (tepe) | 2026-01-07 23:25:40 | 91410.4 | 90390.9999 | 93412.1367 | 91901.5683 | 755.2842 | -0.6503 | 1019.4001 | HOLD |
| ~00:45 toparlanma | 2026-01-08 00:44:45 | 91156.6 | 90438.9346 | 93198.6513 | 91818.7930 | 689.9292 | -0.9598 | 717.6654 | HOLD |
| ~00:45 toparlanma | 2026-01-08 00:45:17 | 91203.3 | 90439.1656 | 93196.7276 | 91817.9466 | 689.3905 | -0.8916 | 764.1344 | HOLD |
| ~00:45–01:00 yükseliş | 2026-01-08 00:55:27 | 91380.3 | 90438.4167 | 93177.5133 | 91807.9650 | 684.7742 | -0.6245 | 941.8833 | HOLD |

**Log kanıtı (aynı satırlar):** `bot_window_utc.log` içinde bu timestamp’lerde `[MeanReversion] Price within bands ...` ve `Action: HOLD` satırları birebir mevcut.

---

## 5) Bant metrik analizi (kodsuz KPI’lar)

`mr_metrics.csv` (291 checkpoint) özet KPI:
- `|z|_max = 1.2317` (tetik eşiği: `|z| > band_multiplier = 2.0`)
- `min(dist_to_nearest_band) = 586.14`
- `band_width_pct p50 = 3.209%` (p95 = 3.348%)
- `px_range = 435.4` (≈ `0.474%` of mid)  
- `px_range / median(band_width) = 0.148`  → fiyat dalgası, band genişliğinin yalnızca ~%15’i

**15dk dilim özeti (range vs band):**

| 15dk bin (UTC) | n | px_range | width_pct_p50 | \|z\|_max | dist_min | outside_if_m=1.0 |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01-07 22:30 | 27 | 150.7 | 3.345 | 1.185 | 626.9 | 17 |
| 2026-01-07 22:45 | 28 | 146.2 | 3.348 | 1.194 | 620.1 | 20 |
| 2026-01-07 23:00 | 29 | 189.9 | 3.331 | 1.232 | 586.1 | 28 |
| 2026-01-07 23:15 | 28 | 220.1 | 3.296 | 0.949 | 799.2 | 0 |
| 2026-01-07 23:30 | 28 | 132.2 | 3.248 | 0.979 | 755.7 | 0 |
| 2026-01-07 23:45 | 28 | 137.2 | 3.192 | 0.937 | 783.5 | 0 |
| 2026-01-08 00:00 | 28 | 64.6 | 3.118 | 0.826 | 833.4 | 0 |
| 2026-01-08 00:15 | 28 | 104.3 | 3.060 | 1.069 | 649.3 | 18 |
| 2026-01-08 00:30 | 28 | 69.7 | 3.020 | 1.069 | 647.7 | 19 |
| 2026-01-08 00:45 | 28 | 177.0 | 2.989 | 0.892 | 762.5 | 0 |
| 2026-01-08 01:00 | 11 | 62.7 | 2.975 | 0.702 | 885.0 | 0 |

**Okuma:** En “hareketli” dilimlerde bile (`23:00` bin’i) `|z|_max≈1.232` → band dışına çıkış yok. Bu nedenle MR sinyal üretmemesi, grafikteki volatilite “gözle” var olsa da **MR band ölçeğinde yetersiz** kalmasından kaynaklanıyor.

---

## 6) Kök neden ağacı (Root Cause Tree)

**(A) Band dışına çıkış olmadı → tetik yok (Birincil neden)**  
- Kanıt: `Price within bands=291`, `outside=0`, `Action: HOLD=291`  
- Kanıt: `|z|_max=1.2317 < 2.0`, `min(dist)=586.14`

**(B) Bandlar geniş (parametre kaynaklı) → fiyat dalgası band eşiğine yaklaşamıyor**  
- Kanıt: `band_width_pct p50=3.209%` vs `px_range_pct≈0.474%`  
- Muhtemel kök: `vwap_lookback=1440` (24h) + `band_multiplier=2.0`

**(C) ADX gate ikincil filtre (band daralırsa önem kazanır)**  
- Kanıt: `adx_th=25`, checkpoint’lerin `67/291` anında `adx>25`  
- Etki: Band tetiklenmeye başlasa bile bazı sinyaller “trend” filtresiyle kesilebilir.

**(D) Operasyonel uyumsuzluklar (yanılsama/konfig sapması)**  
- VST vs prod: bot VST; grafiğin kaynağı farklıysa davranış farkı görülebilir.  
- Konfig yükleme: `curl command not found` → Azure App Config uygulanamamış olabilir (beklenen prod tuning yok).

---

## 7) Öneriler (önceliklendirilmiş, çoğu kodsuz)

1) **`band_multiplier` düşür (2.0 → 1.15/1.10 aralığı ile A/B test)**  
   - Beklenen etki (bu run datası üzerinden): outside sayısı `m=1.15` için **37/291 (~%12.7)**, `m=1.10` için **48/291 (~%16.5)**.  
   - Risk: trade sayısı artar; whipsaw/noise artabilir.

2) **`vwap_lookback` kısalt (1440 → 240 veya 120) ve band_width_pct’yi hedefle**  
   - Beklenen etki: band daralır, `|z|` büyür, tetik olasılığı artar.  
   - Hedef metrik: mevcut `band_width_pct p50=3.209%` → tetik görebilmek için pratikte anlamlı daralma gerekir (bu koşuda fiyat dalgası ~%0.47).  
   - Risk: bant “aşırı reaktif” olup aşırı sinyal üretebilir.

3) **Grafik ve botu aynı ürün/ortam/timeframe’e hizala (özellikle VST/prod + 4m/5m)**  
   - Beklenen etki: “grafikte var ama botta yok” yanlış kıyasları biter; tanı koyma hızlanır.  
   - Risk: yok.

4) **ADX threshold’ü MR hedefiyle uyumla (örn. 25 → 30) / gate etkisini ölç**  
   - Beklenen etki: band daraltınca gelecek sinyallerin ~%23’lük bölümünün ek filtreden dolayı kaybolması azalabilir.  
   - Risk: trend piyasasında MR ters-yön işlemleri artabilir.

5) **Operasyonel rutin: her run sonrası otomatik MR metrik raporu (z/dist/width_pct) üret**  
   - Bu çalışmada kullanılan çıktılar: `mr_metrics.csv` + 15dk özet tablosu.  
   - Beklenen etki: “neden sinyal yok?” sorusu sayısal olarak anında cevaplanır.  
   - Risk: yok (salt analiz).

6) **Opsiyonel (kod/deploy gerektirir): log’ları zenginleştir (band_mid/std/width_pct/z + eşik)**  
   - Beklenen etki: gelecekte CSV/regex ihtiyacı azalır; gözlem/alert daha sağlam olur.  
   - Risk: log hacmi artar; deploy/rebuild gerekir.

---

## Ekler / Referans çıktılar

- Grafik↔log eşleme eki (detay anlatım): `diagnostics/mr_investigation_20260107_223016/report_addendum_grafik_esleme.md`
- UTC pencere: `bot_window_utc.log`
- MR satır extract: `mr_lines_extract.txt`
- MR metrikleri: `mr_metrics.csv`

