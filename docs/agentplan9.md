# Agent Plan 9: VSA + BA/GO/FR Entegrasyonu (2 Faz, Risk-Odakli)

## 1) Amac ve Ilke

Bu planin amaci "islem acmayan bot" yapmak degil, **pozitif EV oldugunda mantikli risk alan** bir davranis uretmektir.

Temel ilke:
- Sert global veto yerine once **siniflandirma + risk ayari**.
- Islemi tamamen kesmek yerine uygun yerde **micro-risk / normal-risk** gecisi.
- Degisiklikleri once config ve telemetry ile gozleyip, sonra davranisa almak.

---

## 2) Mevcut Durumdan Gelen Kritik Noktalar

- MR tarafinda impulse/shock telemetri ve rejection yapisi var.
- Kisa taraf korumalari kodda var ama defaultta zayif:
  - `recheck_mode: "observe"`
  - `trend_up.short_mode: "off"`
  - `shock.short_mode: "off"`
- VSA dokumanindaki kritik kural (**ignition + %50 halfback ustunde short lock**) birebir uygulanmis degil.
- `agentplan8`'deki BA/GO/FR ve softmax tabanli secim mantigi mevcut akista dogrudan yok.

---

## 3) Faz 1 (Dusuk Risk): Config Hardening + Shadow Siniflandirma

### 3.1 Canli Davranis Degisimi (Config-Only)

MR short tarafinda korumayi arttir:
- `strategies.mean_reversion.rejection_confirmation.recheck_mode: "enforce"`
- `strategies.mean_reversion.regime_policy.trend_up.short_mode: "extreme_only"` (gerekirse `disabled`)
- `strategies.mean_reversion.regime_policy.shock.short_mode: "disabled"`

Beklenen etki:
- Zayif rejection short'lari azalir.
- Bullish/impulsif kosullarda erken short riski duser.
- Trade sayisi biraz azalir ama kalite artar.

### 3.2 Shadow Mod (Trade Etkilemeden)

`agentplan8` skorlarini sadece telemetry olarak hesapla, order kararina dokunma.

Hesaplar:

1. ImpulseScore `I` (0-1):
- `TR_1m = max(high-low, abs(high-prev_close), abs(low-prev_close))`
- `I = clip(max(|r_1m|/0.25%, TR_1m/(2.5*ATR_1m), VR_1m/2.0), 0, 1)`

2. TrendScore `T` (0-1):
- `T = clip((ADX_coord-25)/(45-25), 0, 1) * 1[slope(VWAP/EMA)>0]`

3. RejectionScore `R` (0-1):
- `pass_rate_rej(N) = sum(rej_pass)/N`  (rej_pass = rejection test pass ise 1)
- `R = 0.7*pass_rate_rej(N) + 0.3*clip(upperwick/0.8, 0, 1)`
- Persistency cezasi: ardarda 2 onay yoksa `R = 0.6*R`

4. AcceptanceScore `A` (0-1):
- `A = clip(time_above_VWAP/120s, 0, 1) * (1-R)`

5. `z` normalizasyonu:
- `z_norm = clip((|z|-z_entry)/(z_cap-z_entry), 0, 1)`

6. Sinif skor/olasilik:
- `S_BA = 1.2I + 1.0T + 0.8A - 1.0R`
- `S_GO = 1.0R + 0.8z_norm - 1.0I - 0.8T`
- `S_FR = 1.0I + 1.2R - 1.0A`
- `p_k = exp(S_k) / sum_j exp(S_j)`

7. Risk edge (shadow):
- `E = p_selected * Q * M`
- `Q = clip((quality-0.50)/0.20, 0, 1)`
- `M_fill = 1 - clip(fill_delay/60s, 0, 1)`
- `M_rr = 1 - clip(max(0, RR_min - RR)/RR_span, 0, 1)`  (sadece RR_min alti cezali)
- `M = M_fill * M_rr`

Gozlem ciktilari:
- `p_BA/p_GO/p_FR`, `selected_class`, `E_shadow`
- mevcut karar ile sinif uyumu
- "BA ama MR short denendi mi?" gibi mismatch oranlari

### 3.3 Faz 1 Cikis Kriteri (Go/No-Go)

En az 2-4 hafta veya anlamli ornek:
- Short stop-out orani: iyilesme
- Net PnL/trade: bozulmama
- "No-trade drift": belirgin artis olmamasi
- BA mismatch orani: olculebilir hale gelmesi

---

## 4) Faz 2 (Kontrollu Davranis): Feature-Flag ile Kademeli Uygulama

### 4.1 Feature Flag Seti

- `vsa_guard.enabled`
- `vsa_guard.ignition_halfback_lock.enabled`
- `vsa_guard.class_aware_risk.enabled`
- `vsa_guard.mode`: `observe | enforce`
- `vsa_guard.canary_symbols`

### 4.2 Ignition-Halfback Lock (VSA cekirdegi)

Ignition tespitinde state tut:
- `ignition_high`, `ignition_low`, `mid_point`, `expires_at`

Kural:
- Fiyat `mid_point` ustundeyken counter-trend short **block veya micro-risk**.
- `mid_point` alti/timeout olunca normal kurallara don.

Not:
- "islem acmama" yerine once `micro-risk` tercih edilir.
- Sert block sadece ekstrem BA guveninde kullanilir.

### 4.3 Sinif Bazli Risk/Secim

1. BA dominant:
- MR short: risk carpani dusur (`0.1-0.35` araligi)
- MR short RR esigi yukseltilir
- Mumkunse trend-yonlu stratejiye oncelik artirilir (mevcut strateji havuzuna uygun sekilde)

2. GO dominant:
- MR fade normal risk

3. FR dominant:
- MR reversal kontrollu daha yuksek risk/RR

4. Risk carpani:
- `risk_mult = clip(sigmoid(8*(E-0.55)), 0.1, 1.0)`
- Alt sinir korunur (tam sifirlama yok), fakat cok dusuk edge'de trade atlanabilir.

### 4.4 Canary ve Rollout

1. Canary (sembollerin %10-20'si, 1-2 hafta)
2. Kademeli artis (%50 -> %100)
3. Her adimda geri donus plani:
- flag kapat -> onceki davranisa aninda don

---

## 5) Mimari Uyumluluk (Nereye Ekleyecegiz)

- **Faz 1 Shadow hesap**: `StrategyCoordinator` enrichment/quality akisina telemetry olarak.
- **Faz 2 davranis**: MR short karari oncesi guardrail bloklarina entegre.
- **Execution tutarliligi**: fill-sonrasi stop/TP rebase davranisi korunur.

Bu sayede buyuk refactor olmadan, mevcut pipeline uzerinden ilerlenir.

---

## 6) Basari Metrikleri (Iki Fazda da Ortak)

- Trade sayisi (botun "susmasini" engellemek icin)
- Short stop-out orani
- Win rate / net PnL per trade
- Max drawdown ve tail-loss olaylari
- BA/GO/FR sinif dagilimi ve mismatch raporu
- Rejection fail -> gercek zarar korelasyonu

---

## 7) Onerilen Siralama

1. Faz 1 config hardening + shadow telemetry
2. 2-4 hafta gozlem + KPI karsilastirma
3. Faz 2 canary (`observe` -> `enforce`)
4. Kademeli rollout

Bu siralama mevcut sisteme en dusuk operasyonel riskle uyar ve "islem acmayan bot" riskine dusmeden davranisi kaliteye tasir.
