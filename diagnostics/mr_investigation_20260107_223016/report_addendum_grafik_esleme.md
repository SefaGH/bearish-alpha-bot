# MR Sinyal Üretmeme İncelemesi — Grafik↔Log Eşleştirme Eki

Bu ek, `screenshots/Ekran görüntüsü 2026-01-08 040823.png` (BTC 4m) üzerindeki iki kritik bölgeyi (`~23:00` ve `~00:45`) `logs/live_trading_20260107_223016_911605.log` içindeki **spesifik Mean Reversion (MR) checkpoint** satırlarıyla eşler ve “neden sinyal yok?” sorusunu bant/z-score perspektifinde **grafik üstü** kanıtla bağlar.

## 1) Zaman/TZ doğrulaması (grafik ↔ log)

- Log açıkça UTC: `[SYSTEM INFORMATION] Timestamp (UTC): 2026-01-07 22:30:34`.
- Screenshot dosya adı yerel saat izlenimi veriyor (`... 040823`), ancak grafikte x-ekseni 22:00–01:00 aralığını gösteriyor ve log MR seansı da **22:30–01:05 UTC** aralığında.
- Sonuç: Bu screenshot’taki saat etiketleri, pratikte log UTC akışıyla **tutarlı** okunabiliyor (en azından bu oturum için).

Not: Grafik **4m**, MR karar TF’i **5m** (MR config: `signal_timeframe=5m`). Bu nedenle birebir “mum sınırı” değil, **zaman damgasına en yakın checkpoint** eşleştirmesi kullanıldı.

## 2) 23:00 bölgesi eşleştirmesi (grafikteki ilk impuls / dönüş)

Grafikte 23:00 etiketinden sonra görülen impuls/dönüş bölgesi, log tarafında 23:00–23:30 içinde kalan checkpoint’lerle örtüşüyor. Bu penceredeki **en düşük** ve **en yüksek** MR px anları:

### A) Dip’e yakın checkpoint (23:10 civarı)

Log kanıtı:

- `2026-01-07 23:10:42 - ... [MeanReversion] Price within bands ... px=90975.0000, lower=90387.7983, upper=93444.9619 ... Action: HOLD`

Türetilen metrikler (`band_multiplier=2.0`):

- `band_mid=(upper+lower)/2 = 91916.3801`
- `band_width=3057.1636` (≈ `3.3260%`)
- `std=band_width/(2*2.0)=764.2909`
- `z=(px-mid)/std = -1.2317`  → **|z| < 2.0 ⇒ bant dışına çıkış yok**
- En yakın banda mesafe: `min(px-lower, upper-px)=587.20`

Grafik üstü yorum:

- Bu anda bant sınırları yaklaşık **90.388k–93.445k**. Screenshot’un gördüğü y-ekseni aralığı (yaklaşık **90.800k–91.400k**) içinde **upper band zaten ekran dışı**, lower band ise ekranın altına yakın/altında kalıyor. Görselde “dalga var” hissi, 24h VWAP bandına göre **çok küçük** kalıyor.

### B) Tepe’ye yakın checkpoint (23:25 civarı)

Log kanıtı:

- `2026-01-07 23:25:40 - ... [MeanReversion] Price within bands ... px=91410.4000, lower=90390.9999, upper=93412.1367 ... Action: HOLD`

Türetilen metrikler:

- `band_mid=91901.5683`, `std=755.2842`
- `z=-0.6503` (bant dışından çok uzak)
- En yakın banda mesafe: `1019.40`

Grafik üstü yorum:

- px 91.4k seviyesine çıksa bile MR upper band **93.4k** civarında → “sert yukarı itki” görsel olarak belirgin olsa da MR için hâlâ bandın **çok içinde**.

## 3) 00:45 bölgesi eşleştirmesi (gece geri çekilme / toparlanma)

Grafikte “00:45 civarı” olarak tarif edilen bölgede (00:45–01:00), log tarafında bu pencerenin minimum ve maksimum MR px checkpoint’leri:

### A) 00:45 başlangıcı (pencere min’i)

Log kanıtı:

- `2026-01-08 00:45:17 - ... [MeanReversion] Price within bands ... px=91203.3000, lower=90439.1656, upper=93196.7276 ... Action: HOLD`

Türetilen metrikler:

- `band_mid=91817.9466`, `std=689.3905`
- `z=-0.8916` (bant dışı değil)
- En yakın banda mesafe: `764.13`

### B) 00:55 civarı (pencere max’ı)

Log kanıtı:

- `2026-01-08 00:55:27 - ... [MeanReversion] Price within bands ... px=91380.3000, lower=90438.4167, upper=93177.5133 ... Action: HOLD`

Türetilen metrikler:

- `band_mid=91807.9650`, `std=684.7742`
- `z=-0.6245`
- En yakın banda mesafe: `941.88`

Grafik üstü yorum:

- Bu bölgedeki yükseliş (≈ +177 USDT) görselde “toparlanma” gibi görünse de MR bandı yine **~2.7–2.8k** genişliğinde (≈ %3) olduğundan, fiyat bant sınırlarına yaklaşmıyor.

## 4) Sonuç (grafik üstü kanıtla kök neden)

- Bu screenshot’taki en “hareketli” görünen bölgelerde bile MR tarafında tüm checkpoint’ler **within bands** ve **Action: HOLD**.
- Sayısal olarak: bu oturumda `|z|_max = 1.2317` (gerekli eşik: `|z| > band_multiplier = 2.0`).
- Bantlar fiyat ölçeğine göre “fazla geniş” kaldığı için grafikteki dalga/impuls hareketleri **band dışı tetik eşiğine hiç gelmiyor**.

## 5) Ek dosyalar

- MR ham satır extract: `diagnostics/mr_investigation_20260107_223016/mr_lines_extract.txt`
- MR checkpoint metrikleri: `diagnostics/mr_investigation_20260107_223016/mr_metrics.csv`
- 15dk özet tablo: `diagnostics/mr_investigation_20260107_223016/mr_summary_15m.csv`

