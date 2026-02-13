# BTC "Mavi Cizgi Davranisi" ML Entegrasyon Plani (Mimari Uyumlu Revizyon)

Bu dokuman, BTC fiyatinin belirli bir referans seviyenin ustunde/altinda rejimsel sekilde davranmasi hipotezini mevcut kod mimarisiyla uyumlu sekilde ML pipeline'a entegre etmek icin hazirlandi.

## 1) Hedef ve Hipotez

- Hedef: "fiyat referans cizginin ustunde/altinda kalma" bilgisini modele acik bir feature set olarak vermek.
- Hipotez: Bu bilgi, ozellikle yatay ve sinir kirilimli piyasalarda false signal oranini azaltir.
- Kapsam: Sadece feature engineering + model bundle entegrasyonu. Trade execution/risk kurallari bu asamada degismeyecek.

## 2) Mevcut Mimari Gercegi (Kod ile Uyum)

- Feature extraction girisi `src/ml/feature_engineering.py` icindeki `FeatureEngineeringPipeline.extract_features(...)`.
- GEMMA aktifken pipeline `extract_gemma_features(...)` uzerinden 87 ham feature uretir.
- Sonrasinda secilen feature'lar manifest'ten alinip filtrelenir:
  - `artifacts/gemma/final/manifest.json` (`feature_count=82`, `feature_names_ordered`).
  - `features/gemma/selected/gemma_price_selected_82.json`.
- Adapter tarafi feature sayisini manifest/scaler uyumuna gore bekler:
  - `src/ml/adapters/gemma/gemma_torchscript_adapter.py`.

## 3) Onceki Taslakta Revizyon Gerektiren Noktalar

- `create_features(...)` fonksiyonu bu projede ana entegrasyon noktasi degil; dogru nokta `extract_gemma_features(...)`.
- `model_trainer.py` icinde sabit `INPUT_DIM` guncellemesi tek basina yeterli degil.
- Bu projede dogru akış: feature uretimi + feature secimi + selected json + manifest + scaler/model uyumu.
- Place-holder kalan `_find_sr_levels -> return [], []` yaklasimi urunde deger uretmez.

## 4) Onerilen Feature Tasarimi (Causal, Look-ahead Guvenli)

Asagidaki feature'lar "gelecegi gormeden" yalnizca t anina kadar olan veriyle hesaplanacak:

1. `sr_anchor_price`
- Rolling pencere icinde (or. 300-500 mum) fiyat yogunlugundan turetilen referans seviye.

2. `sr_side`
- `+1`: fiyat anchor ustunde, `-1`: anchor altinda, `0`: cok yakin/noise bandi.

3. `sr_dist_anchor_atr`
- `(close - anchor) / atr_20`.
- Farkli fiyat rejimlerinde (30k/100k gibi) olcek bagimsiz davranis saglar.

4. `sr_bars_since_cross`
- Son anchor gecisinden beri gecen mum sayisi.

5. `sr_time_above_ratio`
- Son N mumda anchor ustunde kalma orani.

6. `sr_retest_reject_score`
- Anchor'a yakinlasip (mesafe < X*ATR) ters yone donme kuvvetini olcer.

Not:
- Hesaplama "strictly causal" olacak: rolling window sag siniri `t-1` veya `t` ile sinirli olacak.
- `shift(-k)` ya da gelecegi kullanan herhangi bir hesap bu feature'larda kesinlikle olmayacak.

## 5) Entegrasyon Noktalari (Kod Seviyesi)

### 5.1 Feature Uretimi

- Yeni sinif: `src/ml/features/sr_advanced.py` (onerilen) veya dogrudan `src/ml/feature_engineering.py` icine entegre.
- Cikti kolonlari `extract_gemma_features(...)` sonunda `features[...]` DataFrame'ine eklenecek.
- Mevcut `support_distance`/`resistance_distance` korunacak; yeni feature'lar bunlari tamamlayacak.

### 5.2 Feature Listesi ve Manifest

- `scripts/generate_gemma_features.py` tam feature listesini guncellemeli (87 -> yeni toplam).
- `scripts/analyze_features.py` ile maske yeniden uretilecek.
- Yeni secilen liste `features/gemma/selected/gemma_price_selected_<N>.json` olarak yazilacak.
- Bundle manifest'i `artifacts/gemma/final/manifest.json` yeni `feature_count` ve `feature_names_ordered` ile uyumlu olacak.
- Saglik kontrolu:
  - `scripts/gemma_manifest_health_check.py`
  - `scripts/verify_gemma_workflow.py`

### 5.3 Adapter Uyum Kontrolu

- `GemmaTorchScriptAdapter` manifest/scaler uyumunu zaten kontrol ediyor.
- Buna ragmen retrain sonrasi su dortlu mutlaka ayni feature setini gostermeli:
  1. Model input boyutu
  2. Scaler `n_features_in_`
  3. Selected feature json uzunlugu
  4. Manifest `feature_count`

## 6) Egitim ve A/B Test Plani

Model A (Kontrol):
- Mevcut bundle (82 feature).

Model B (Deney):
- Yeni SR feature seti eklenmis bundle.

Karsilastirma metrikleri:
- Classification: macro F1, class-wise recall (bullish/neutral/bearish), confusion matrix.
- Trading etkisi: win rate, avg R, max drawdown, false breakout sonrasi hatali yon sinyal sayisi.
- Slice bazli analiz:
  - Low volatility vs high volatility
  - Range market vs trend market
  - Anchor'a yakin bolge (< 0.5 ATR) vs uzak bolge

Basari kriteri (ornek):
- Macro F1'da dusus olmadan,
- Range diliminde false signal oraninda anlamli azalma,
- Toplam PnL veya risk-adjusted metriklerde gerileme olmamasi.

## 7) Performans ve Operasyon Notlari

- Ilk asamada offline/backtest hesaplama maliyeti kabul edilebilir.
- Live modda her tikte tam recluster yapilmayacak:
  - Anchor seviyesi periyodik (or. 30-60 dk) guncellenecek.
  - Aradaki mumlarda sadece incremental durum feature'lari guncellenecek.
- Hesaplama fallback:
  - Yetersiz lookback varsa feature'lar kontrollu sekilde `NaN/0` ile doldurulacak.
  - Pipeline fallback davranislari korunacak.

## 8) Uygulama Sirasina Gore Net Checklist

1. Yeni causal SR feature hesaplayicisini ekle.
2. `extract_gemma_features(...)` icine yeni kolonlari bagla.
3. Feature listesi/mask/manifest uretim akisini guncelle.
4. GEMMA model + scaler retrain et ve yeni bundle olustur.
5. A/B backtest ve paper sonuclarini raporla.
6. Basari kriterleri saglanirsa rollout planina gec.

## 9) Bu Asama Sonu Beklenen Cikti

- Mimariyle uyumlu, calisan bir "SR state aware" GEMMA feature pipeline.
- Manifest ve artifact zinciri bozulmadan yeni model versiyonu.
- Sayisal olarak dogrulanmis A/B sonuc raporu.
