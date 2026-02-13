# RSI Router Uygulanabilir Teknik Spesifikasyon (Pragmatik Minimum v1)

## 0) Karar Ozeti

Bu dokumanin v1 kapsami su minimum planla sinirlidir:

1. Merkezi Router gate (isleme almadan once).
2. Recheck (`dispatch_strategy`) path'inde ayni gate.
3. MR icinde hafif guard (ozellikle `soft_deferral_event` oncesi).
4. `side split` ve `min band penetration` zorunlu degil; kalite/tuning katmani olarak sonra acilacak.

## 1) Problem ve Hedef

### Problem
- MR (`mean_reversion`) tarafinda RSI zone bazli strategy-agnostic gate yok.
- Bu nedenle baglam disi MR denemeleri (ornegin yuksek RSI'da long) sistemde gorulebiliyor.

### Hedef (v1)
- Tek kaynakli, dinamik RSI zone router kurmak.
- Zone yetkisini iki kritik yerde zorunlu uygulamak:
  - ana islem akisi (`_process_trading_loop`)
  - recheck akisi (`dispatch_strategy`)
- MR tarafinda hafif ve emniyetli bir ic guard eklemek:
  - zone uyumsuzsa `soft_deferral_event` uretmemek.

## 2) Dinamik Zone Tasarimi

## 2.1 Girdi Kaynaklari
- Slow RSI: `30m` (kapali mum), ana karar kaynagi.
- Fast RSI: `5m` (kapali mum), opsiyonel consensus kaynagi.
- OB efektif esik:
  - `adaptive_ob.get_symbol_specific_threshold(symbol)` varsa onu kullan.
  - Yoksa `adaptive_ob.get_adaptive_rsi_threshold(market_regime)`.
- STR efektif esik:
  - `adaptive_str.get_symbol_specific_threshold(symbol)` varsa onu kullan.
  - Yoksa `adaptive_str.get_adaptive_rsi_threshold(market_regime)`.

## 2.2 Esik Normalizasyonu
- `ob_thr` clamp: `[ob_floor, ob_cap]` (default: `[10, 45]`).
- `str_thr` clamp: `[str_floor, str_cap]` (default: `[55, 90]`).
- Min gap kurali:
  - `str_thr - ob_thr < min_gap` ise otomatik normalize edilir.

## 2.3 Zone Tanimi
- `transition_width = w` (default `5.0` RSI puani).
- Zone'lar:
  - `OVERSOLD`: `rsi <= ob_thr`
  - `TRANSITION_LOW`: `ob_thr < rsi < ob_thr + w`
  - `MR`: `ob_thr + w <= rsi <= str_thr - w`
  - `TRANSITION_HIGH`: `str_thr - w < rsi < str_thr`
  - `OVERBOUGHT`: `rsi >= str_thr`

## 2.4 Consensus Modu
- `mode = slow_only | consensus`.
- `consensus` modunda:
  - Slow/fast ayni ust zone grubunda ise (`OVERSOLD/MR/OVERBOUGHT`) onu kullan.
  - Ayrisma varsa `TRANSITION` kabul et (`new entry` yok).
  - Fast veri yoksa slow-only fallback.

## 3) Yetki Matrisi (v1)

- `adaptive_ob`: sadece `OVERSOLD` zone'da yeni entry.
- `adaptive_str`: sadece `OVERBOUGHT` zone'da yeni entry.
- `mean_reversion`: sadece `MR` zone'da yeni entry.
- `TRANSITION_*`: yeni entry yok (`no_trade_new_entry=true` iken).

Not: v1'de MR side split ve penetrasyon filtresi yoktur (opsiyonel faza alinmistir).

## 4) Uygulama Noktalari (Dosya/Fonksiyon Bazli)

## 4.1 Yeni dosya: `src/core/rsi_zone_router.py`

### Eklenecek yapilar
- `Enum RsiZone`: `OVERSOLD`, `TRANSITION_LOW`, `MR`, `TRANSITION_HIGH`, `OVERBOUGHT`.
- `@dataclass RsiZoneSnapshot`:
  - `symbol`, `ts_ms`, `rsi_slow`, `rsi_fast`, `mode`
  - `ob_threshold`, `str_threshold`, `zone`, `transition_width`, `version`, `meta`

### Eklenecek fonksiyonlar
- `compute_effective_thresholds(...)`
- `resolve_zone(...) -> RsiZoneSnapshot`
- `is_strategy_allowed(strategy_name, side, snapshot, cfg) -> (bool, reason_code)`

## 4.2 Ana dongu gate: `src/core/production_coordinator.py`

### Fonksiyon: `_process_trading_loop()`
- Symbol bazinda, strategy cagrilarindan once `rsi_zone_snapshot` uret.
- `signal_kwargs` icine `rsi_zone_snapshot` ekle.
- Strategy cagrisindan once merkezi gate uygula:
  - yetki yoksa strategy call skip et.
  - telemetry event: `rsi_router_skip`.

## 4.3 Recheck gate: `src/core/production_coordinator.py`

### Fonksiyon: `dispatch_strategy()`
- Recheck path'te de ayni `rsi_zone_snapshot` uretilir.
- Strategy cagrisindan once merkezi gate tekrar uygulanir.
- Zone mismatch durumunda:
  - `dispatched=false`
  - `rearm_fast_watch=false`
  - `final_reason="rsi_router.zone_mismatch"`
- Cross-strategy pass yok (MR -> STR/OB paslanmaz).

## 4.4 MR hafif guard: `src/strategies/mean_reversion.py`

### Fonksiyon: `signal(...)`
- `kwargs` veya `signal_kwargs` icinden `rsi_zone_snapshot` oku.
- Zone `MR` degilse yeni signal uretme.
- Ozellikle `soft_deferral_event` olusturmadan hemen once zone tekrar kontrol et:
  - zone uyumsuzsa deferral event donme.
- Recheck context'te (`parent_pending_id` varken) uyumsuzluk:
  - `strategy_recheck_decision` don
  - `rearm_fast_watch=false`
  - reason code: `rsi_router.deferral_cancelled`

## 4.5 Waiting-room telemetry: `src/core/strategy_coordinator.py`

### Fonksiyonlar
- `handle_soft_deferral(...)`
- `incubator_tick(...)`

### Telemetry
- Drop/outcome reason code'lari normalize edilir:
  - `rsi_router.zone_mismatch`
  - `rsi_router.transition_no_trade`
  - `rsi_router.deferral_cancelled`

## 5) Konfigurasyon (v1)

`config/config.example.yaml` icin onerilen blok:

```yaml
strategies:
  rsi_zone_router:
    enabled: true
    source:
      mode: "consensus"              # slow_only | consensus
      slow_tf: "30m"
      fast_tf: "5m"
    thresholds:
      ob_floor: 10.0
      ob_cap: 45.0
      str_floor: 55.0
      str_cap: 90.0
      min_gap: 8.0
    transition:
      width: 5.0
      no_trade_new_entry: true
    soft_deferral:
      cancel_on_zone_mismatch: true
    telemetry:
      enabled: true
      log_state_changes_only: true
```

## 6) Config Validasyon

Dosya: `src/config/schema.py`

- Yeni helper: `_validate_rsi_zone_router(config_data)`.
- `validate_config_safety()` icinde cagrilir.
- Minimum kontroller:
  - numeric alanlar finite.
  - `min_gap > 0`.
  - `transition.width >= 0`.
  - `mode` enum degeri gecerli.

## 7) Acceptance Kriterleri (v1)

- MR baglam disi zone'da yeni entry uretemez.
- Recheck path'inde de ayni zone gate zorunludur.
- Zone mismatch durumunda pending setup rearm olmaz.
- `soft_deferral_event` zone uyumsuzken uretilmez.
- Router kapaliyken mevcut davranis korunur (backward compatibility).

## 8) Test Plani (v1)

### Yeni testler
- `tests/unit/test_rsi_zone_router.py`
  - esik normalizasyonu
  - zone siniflandirma
  - consensus ayrisma -> transition
- `tests/unit/test_mr_rsi_router_guard.py`
  - MR zone mismatch -> hold
  - MR deferral oncesi zone guard

### Guncellenecek testler
- `tests/test_production_coordinator_dispatch.py`
  - recheck gate deny -> `rearm_fast_watch=false`
- `tests/integration/test_soft_deferral_flow.py`
  - zone mismatch'te iptal + no rearm

## 9) Opsiyonel Faz (v1.1 - Kalite/Tuning)

V1 canli stabil olduktan sonra:

1. MR side split (`long`/`short` RSI alt bolgesi).
2. MR min band penetration (`min_band_penetration_bps`).

Bu iki madde zorunlu guvenlik degil, sinyal kalitesi/temizligi icindir.

## 10) Beklenen Sonuc

- MR'in baglam disi islemleri azalir.
- Recheck akisinda stale setup birikimi azalir.
- Degisiklik kapsami minimum tutuldugu icin production riski kontrollu kalir.
