# VSA Phase-1 Monitoring Checklist

Bu checklist, Faz-1 degisikliklerinin amacina uygun calisip calismadigini izlemek icindir:
- botu susturmak degil
- short tarafinda anlamsiz riski dusurmek
- EV olan yerde islem acmaya devam etmek

## 1) Scope

- Izleme penceresi: minimum 14 gun (tercihen 2-4 hafta)
- Karsilastirma: degisiklik oncesi son 14 gun vs degisiklik sonrasi 14 gun
- Segment:
  - tum semboller
  - likit 5-10 sembol ayri
  - sadece `mean_reversion` ayri

## 2) Faz-1 Config Sanity

Asagidaki degerler aktif olmali:
- `strategies.mean_reversion.rejection_confirmation.recheck_mode = enforce`
- `strategies.mean_reversion.regime_policy.trend_up.short_mode = extreme_only`
- `strategies.mean_reversion.regime_policy.shock.short_mode = disabled`
- `strategies.mean_reversion.vsa_shadow.enabled = true`

## 3) Telemetry Contract

Sinyal metasinda su alanlar gelmeli:
- `meta.vsa_shadow.selected_class` (`BA|GO|FR`)
- `meta.vsa_shadow.probabilities.BA|GO|FR`
- `meta.vsa_shadow.scores.I|T|A|R|z_norm`
- `meta.vsa_shadow.edge.E|Q|M|M_fill|M_rr|RR|RR_min|risk_mult_shadow`
- `meta.rejection_confirmation.*`
- `meta.regime_policy.*`

Veri kalite kontrolu:
- `meta.vsa_shadow` missing rate < %5
- `probabilities` toplami yaklasik 1.0 olmali (0.99-1.01)
- `scores` 0-1 araliginda olmali

## 4) Ana KPI Seti

1. `short_trade_count`
- hedef: sert dusus degil, kontrollu azalis kabul

2. `short_stopout_rate`
- hedef: anlamli dusus

3. `short_net_pnl_per_trade`
- hedef: iyilesme veya en azindan bozulmama

4. `short_win_rate`
- hedef: artis veya stabilite

5. `expectancy_per_trade`
- hedef: yukari yon

6. `no_trade_drift`
- tanim: `(post_trade_count - pre_trade_count) / pre_trade_count`
- alarm: trade sayisinda buyuk dusus var ama kalite artisi yok

## 5) Shadow KPI Seti (Yeni)

1. `class_distribution`
- `BA/GO/FR` dagilimi gunluk/haftalik

2. `ba_short_attempt_rate`
- tanim: `selected_class=BA iken short acilan oran`
- hedef: dusuk olmali

3. `edge_calibration`
- `E` bucket (0.0-0.2, 0.2-0.4, ... 0.8-1.0) bazinda sonuc
- hedef: yuksek `E` bucket'larinda daha iyi outcome

4. `rejection_persistency_health`
- `meta.vsa_shadow.diagnostics.rejection_persistency_ok` orani
- hedef: tek atim pass yerine kalici pass agirlikli olsun

## 6) Alarm Kurallari

1. Kirmizi alarm:
- short_stopout_rate artiyor
- short_net_pnl_per_trade belirgin kotulesiyor
- no_trade_drift cok yuksek ve performans iyilesmiyor

2. Sari alarm:
- `meta.vsa_shadow` missing rate > %5
- BA dagilimi aniden patliyor ve BA short attempt de artiyor
- `probabilities` toplam tutarsizligi tekrarlaniyor

## 7) Gunluk Rutin

1. Config ve servis sagligi kontrolu
2. Son 24 saat KPI snapshot
3. BA/GO/FR dagilimi
4. En kotu 10 short trade incelemesi:
- class
- E
- rejection durumu
- regime_policy veto/allow izi

## 8) Haftalik Go/No-Go (Faz-2 Karari Icin)

Faz-2 canary'e gecis icin asgari kosullar:
- short_stopout_rate iyilesmis olmali
- short_net_pnl_per_trade bozulmamis olmali
- no_trade_drift kabul edilebilir bantta olmali
- `meta.vsa_shadow` veri kalitesi saglanmis olmali

Kosullar saglanmazsa:
- once threshold tuning
- sonra tekrar 1 hafta gozlem

## 9) Faz-2 Oncesi Teknik Hazirlik

- feature-flag plani hazir olmali (`observe -> enforce`)
- canary sembol listesi net olmali
- rollback adimi net olmali (flag kapat, eski davranisa don)

## 10) Otomatik Rapor Komutu

Script:
- `scripts/vsa_phase1_report.py`

Ornek:
```powershell
python scripts/vsa_phase1_report.py `
  --log-glob "logs/*.log" `
  --symbol "BTC/USDT:USDT" `
  --strategy "mean_reversion" `
  --from-utc "2026-02-01T00:00:00Z" `
  --to-utc "2026-02-11T23:59:59Z" `
  --output-json "artifacts/vsa/phase1_report.json" `
  --output-md "artifacts/vsa/phase1_report.md"
```

Baseline ile drift karsilastirmasi:
```powershell
python scripts/vsa_phase1_report.py `
  --log-glob "logs/*.log" `
  --strategy "mean_reversion" `
  --baseline-json "artifacts/vsa/phase1_report_prev.json" `
  --output-json "artifacts/vsa/phase1_report.json" `
  --output-md "artifacts/vsa/phase1_report.md"
```
