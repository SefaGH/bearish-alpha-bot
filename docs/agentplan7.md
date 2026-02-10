# Agent Plan 7 - Shock Sonrasi V-Reversal Firsatini Korumali Sekilde Yakalama

## Sorun Ozeti

Grafikte isaretlenen segment (5m, yaklasik **14:35-14:55 UTC**) tipik bir **flash dump -> V-reversal** davranisi gosteriyor:

- Fiyat hizli dusup kisa surede reclaim yapiyor.
- Dipte hacim zirveye cikiyor, toparlanma bacaklarinda hacim gorece yuksek kaliyor.
- Bu yapi, uzayan trend-cokusu kadar "stop-hunt / liquidation sweep + buyback" karakteri tasiyor.

Mevcut shock korumasi falling-knife riskini azaltmada faydali; ancak blokaj suresi uzun kaldiginda toparlanma bacagi kacar ve firsat maliyeti uretilir.

## Kod Tabanina Gore Guncel Tespitler

1. Shock kapisi su an `DISARMED -> ARMED -> COOLDOWN` akisinda ilerliyor.
2. Varsayilanlar ARMED penceresini uzun tutuyor (`ttl_s=600`, `max_ttl_s=900`, `cooldown_s=180`).
3. MR tarafinda shock long politikasi varsayilan olarak `disabled`; shock aktifken long veto yeme olasiligi yuksek.
4. Promote override, shock state listesi ile engellenebiliyor (`blocked_shock_states`).
5. `TRIGGERED` state tanimi configte geciyor ancak fast shock state machine'de pratikte gorulen state'ler `DISARMED/ARMED/COOLDOWN`.

## Revize Cozum Stratejisi

Ana ilke: **guvenligi koruyup firsati geri almak**.

- ARMED icinde tek blok yerine iki fazli davranis:
  - `HARD`: ilk sure tam blok.
  - `RECOVERY`: sadece teyitli (confirmed) ve sinirli riskli giris.
- Promote override, global ac-kapa yerine sadece recovery fazinda kontrollu kullanilsin.
- TTL/cooldown kisaltma ancak ana degisiklikten sonra ikincil tuning olarak ele alinsin.

## Uygulama Plani (3 PR)

### PR-1: Gozlemlenebilirlik (Davranis Degismeden)

Amac: Neden kacirdigimizi ve hangi gate'in blokladigini tek bakista gormek.

Degisiklikler:

1. `adaptive_ob` snapshot'ina `shock_event_id`, `armed_since_ms`, `state_elapsed_s` ekle.
2. Bu alanlari production dispatch'te strategy kwargs ve signal meta'ya tası.
3. MR tarafinda reason-code standardi ekle:
   - `shock_hard_block`
   - `shock_recovery_unconfirmed`
   - `promote_blocked_shock_phase`

Kabul kriteri:

1. Recheck/log satirinda shock state + phase + reason code birlikte gorulebilmeli.
2. Varsayilan feature flag kapaliyken trade davranisi degismemeli.

### PR-2: Iki Fazli Shock Mantigi (Feature Flag ile)

Amac: İlk dakikalarda koruma, sonrasinda kontrollu firsat.

Yeni config alanlari (default: legacy davranis):

1. `strategies.mean_reversion.regime_policy.shock.two_phase_enabled: false`
2. `hard_block_s`
3. `recovery_long_mode: confirmed_only`
4. `recovery_max_entries_per_event: 1`
5. `recovery_size_mult`
6. `recovery_allow_promote_override`

Davranis:

1. `ARMED` ve `elapsed < hard_block_s` ise faz = `HARD` -> long block.
2. Aksi durumda faz = `RECOVERY` -> sadece confirmed-only kosullariyla long adayi.
3. Event basina max 1 giris kurali uygulanir.
4. Promote override:
   - `HARD`: kapali
   - `RECOVERY`: confige bagli

Kabul kriteri:

1. Flag `false` iken birebir eski davranis.
2. Flag `true` iken "ilk blok + sonra tek firsat" deseni log ve sinyal akisinda gozlenir.

### PR-3: Test ve Golden Window Regresyonu

Amac: Davranisi kalici olarak dogrulamak ve regressioni yakalamak.

Degisiklikler:

1. Unit testler:
   - HARD fazinda veto
   - RECOVERY fazinda confirmed-only izin
   - promote override faza gore farkli davranis
   - event basina tek-entry limiti
2. `scripts/windows.yaml` dosyasina ilgili flash dump -> reclaim penceresi eklenir.
3. `scripts/windows_expectations.yaml` esikleri yeni pencereyi de kapsayacak sekilde guncellenir.

Kabul kriteri:

1. Unit test paketi pass.
2. Golden regression raporunda yeni pencere icin beklenen desen saglanir.

## Mevcut Durum vs Revize Plan (Fayda/Zarar)

| Baslik | Mevcut Durum | Revize Plan | Fayda | Zarar / Risk |
|---|---|---|---|---|
| Shock guvenligi | Guclu ama uzun blokaj | HARD + RECOVERY | Falling-knife korumasi korunur | Recovery erken acilırsa yanlis long riski |
| Firsat yakalama | Shock sonrasi reclaim firsatlari kacanabilir | Recovery'de confirmed-only + tek atis | V-reversal firsatlarini yakalama sansi artar | Confirm kurali kotu ayarlanirsa gec/yanlis giris |
| Promote override | Shock state'te toplu bloklanabilir | Faz-bazli kontrollu acma | Near-miss recheck isabeti artabilir | Faz/threshold tuning hassasiyeti artar |
| Gozlemlenebilirlik | Kismi telemetri | Standart reason-code + phase + event_id | Tuning kararları veriyle verilir | Log hacmi artar |
| Davranis degisim riski | Dusuk (stabil ama firsat maliyetli) | Orta (yeni faz mantigi) | Feature-flag ile guvenli rollout | Kod karmasikligi artar |
| Operasyonel maliyet | Dusuk | Orta | Daha olculebilir ve yonetilebilir akış | Config/test bakim yukleri artar |
| Test guvencesi | Senaryo-ozel kapsama sinirli | Unit + golden window kapsami | Regresyon erken yakalanir | CI suresi bir miktar uzar |

## Rollout ve Risk Kontrolu

1. Ilk asama sadece PR-1 (telemetri), davranis degisimi yok.
2. PR-2 feature-flag kapali merge edilir; canary sembolde acilir.
3. PR-3 ile golden-window ve unit testler zorunlu gate olur.
4. Beklenmeyen davranista hizli geri donus: `two_phase_enabled=false`.

## Sonuc

Bu plan, mevcut korumaci yaklasimin ana avantajini (risk azaltma) korurken, shock sonrasi toparlanma penceresinde olusan firsat maliyetini kontrollu bicimde azaltmayi hedefler. En kritik basari kosulu, faz mantigini once gozlemlenebilirlik ile olculebilir hale getirip sonra adim adim acmaktir.
