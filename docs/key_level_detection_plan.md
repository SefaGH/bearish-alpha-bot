# Key Level Detection & Directional Bias — Mimari Uyumlu Revizyon (v2)

> **Tarih:** 2026-02-14  
> **Durum:** Uygulandı (BTC canary, observe)  
> **Amaç:** ML dışı yöntemlerle destek/direnç seviyelerini hesaplamak, fakat karar uygulamasını merkezi router mimarisiyle yapmak.

---

## 0) Karar Özeti

Bu revizyonda iki yaklaşım birleştirildi ve üretim koduna taşındı:

1. **Planın güçlü tarafı korundu:**  
   `KeyLevelDetector` seviye, proximity, range konumu ve touch bilgisi üretiyor.
2. **Entegrasyon merkezi hale getirildi:**  
   Karar/gate uygulaması `ProductionCoordinator` içinde, `RSI Router` pattern'iyle çalışıyor.
3. **Belirsizlik anı risk yönetimi olarak ele alındı:**  
   `AT_LEVEL` durumunda uygun akışta `FAST_PRICE_WATCH` / `STRATEGY_RECHECK` ile deferral uygulanıyor.
4. **DirectionalBias karar verici değil, kalite artırıcı:**  
   `buy/sell` üretmiyor, sadece bounded quality etkisi veriyor.

---

## 1) Problem Tanımı

BTC fiyatı belirli yatay seviyelerin üstünde/altında salınım gösteriyor.  
Fiyat kritik seviyeye yaklaştığında iki olasılık var:

- Kırılım (breakout)
- Red/retest (geri dönüş)

Bu anlar yüksek belirsizlik içeriyor. Belirsizlikte zorunlu yön seçmek yerine sistemin kontrollü şekilde beklemesi daha güvenli.

---

## 2) Mevcut Mimari ile Uyum Prensibi

Bu projede merkezi gate pattern'i zaten mevcut:

- Ana döngü gate: `src/core/production_coordinator.py`
- Recheck gate (`dispatch_strategy`): `src/core/production_coordinator.py`
- Router pattern referansı: `src/core/rsi_zone_router.py`

Dolayısıyla key-level bilgisi de aynı şekilde ele alınmalı:

- **Sembol bazlı snapshot üret**
- **Strateji çağrısından önce merkezi gate uygula**
- **Recheck path'inde aynı gate'i tekrar uygula**
- Snapshot yoksa kontrollü `fail-open` + telemetry

---

## 3) Revize Mimari (v2)

```text
MarketDataPipeline (OHLCV)
        |
        v
KeyLevelDetector (hesaplama katmanı, yeni)
        |
        v
LevelZoneRouter Snapshot (yeni, merkezi state)
        |
        +--> ProductionCoordinator main loop gate (yeni)
        |
        +--> ProductionCoordinator dispatch_strategy gate (yeni)
        |
        +--> Strategy signal kwargs + market_data'ya snapshot ekleme
                 |
                 +--> StrategyCoordinator (opsiyonel: size/quality ayarı, incubator reason handling)
```

---

## 4) Bileşenler

Bu bölümdeki P0/P1/P1.5/P2 kapsamı kodda uygulanmıştır.

## 4.1 KeyLevelDetector (P0)

**Konum:** `src/core/key_level_detector.py`

**Amaç:** `src/core/resistance_band.py` altyapısını sarmalayıp tek çağrıda:

- nearest resistance
- nearest support
- proximity (bps)
- range position
- range width (ATR cinsinden)
- touch counts

üretmek.

### Önemli revizyonlar

1. **Birim standardı:** yüzde yerine `bps` kullan.
2. **Veri yokken güvenli çıktı:** `state="unknown"` ve neutral değerler; otomatik size reduction yok.
3. **Lookahead-safe:** sadece confirmed pivot + closed candle.

### Önerilen çıktı modeli

```python
@dataclass(frozen=True)
class KeyLevels:
    nearest_resistance: Optional[Band]
    nearest_support: Optional[Band]
    distance_to_resistance_bps: Optional[float]
    distance_to_support_bps: Optional[float]
    position_in_range: Optional[float]       # 0..1, None if range unavailable
    range_width_atr: Optional[float]
    touch_count_resistance: int
    touch_count_support: int
    state: str                               # ok | unknown
    reason: str
```

---

## 4.2 LevelZoneRouter (P1, merkezi gate)

**Konum:** `src/core/level_zone_router.py` (yeni)

`rsi_zone_router` ile benzer pattern:

- `build_level_zone_snapshot(...)`
- `is_strategy_allowed(strategy_name, side, snapshot, cfg)`
- `snapshot_to_dict(...)`

### State modeli (minimum)

- `AT_LEVEL` (belirsiz, karar noktası)
- `IN_RANGE` (destek/direnç arası salınım)
- `BREAKOUT_UP_CONFIRMED`
- `BREAKOUT_DOWN_CONFIRMED`
- `UNKNOWN` (veri yetersiz)

### Basit yetki matrisi (v1)

- `adaptive_ob`: `AT_SUPPORT` benzeri koşul veya `BREAKOUT_UP_CONFIRMED` sonrası retest context
- `adaptive_str`: `AT_RESISTANCE` benzeri koşul veya `BREAKOUT_DOWN_CONFIRMED` sonrası retest context
- `mean_reversion`: ağırlıklı `IN_RANGE`
- `AT_LEVEL`: yeni entry yok (`no_trade_new_entry=true`)

Not: v1'de detaylı side-split opsiyonel; önce güvenli gate davranışı.

---

## 4.3 ProductionCoordinator entegrasyonu (P1)

Merkezi kullanım noktaları:

1. Ana döngüde symbol snapshot üret
2. Strateji çağrısından önce `is_strategy_allowed(...)` uygula
3. Snapshot'ı `signal_kwargs` ve `market_data` içine ekle
4. `dispatch_strategy` recheck yolunda aynı gate'i uygula

Bu tasarım, stale setup ve recheck sapmasını azaltır.

---

## 4.4 Belirsizlikte incubator davranışı (P1.5)

`AT_LEVEL` durumunda hard trade yerine:

- `FAST_PRICE_WATCH` (kısa süreli temas/kırılım izlemesi)
- `STRATEGY_RECHECK` (teyit sonrası yeniden değerlendirme)

Uygulanan davranış (özet):

- Main loop tarafında `level_router.at_level` ve `soft_deferral.enabled=true` ise sentetik `soft_deferral_event` üretilip incubator'a gönderilir.
- Recheck (`dispatch_strategy`) tarafında `level_router.at_level` durumunda:
  `final_reason=level_router.breakout_unconfirmed`
- Recheck tarafında AT_LEVEL dışı level gate deny durumunda:
  `final_reason=level_router.recheck_cancelled`
- `refresh_policy=FAST_PRICE_WATCH` ise recheck sonucu uygun senaryoda `rearm_fast_watch=true` döner.

Reason code seti:

- `level_router.at_level`
- `level_router.breakout_unconfirmed`
- `level_router.recheck_cancelled`

---

## 4.5 DirectionalBiasScorer (P2, karar verici değil)

**Konum:** `src/core/directional_bias.py`

Revize rol:

- `buy/sell/wait` üretmez
- bounded `bias_score` üretir (`-1..+1`)
- sadece `quality_score` ve `priority` üstünde sınırlı etki

Örnek:

- `quality_delta = clamp(bias_score * weight, -0.08, +0.08)`
- tek başına entry açtırmaz, yalnızca mevcut sinyal kalitesini ayarlar

Bu sayede strateji karar mekanizması parçalanmaz.

---

## 5) Konfigürasyon (revize)

Bu özellikler `strategies:` altına konumlanır (router ile tutarlı).

```yaml
strategies:
  level_zone_router:
    enabled: true
    source:
      timeframes: ["15m", "1h"]
      mode: "consensus"            # single_tf | consensus
    levels:
      method: "smc"                # smc | kmeans
      pivot_left: 5
      pivot_right: 3
      lookback_bars: 200
      band_pct: 0.005
      smc_cluster_pct: 0.01
      min_cluster_n: 2
      kmin: 2
      kmax: 8
      touch_proximity_bps: 30      # 30 bps = 0.30%
    zones:
      near_level_bps: 50           # 0.50%
      decision_zone_low: 0.40
      decision_zone_high: 0.60
      no_trade_new_entry: true
    breakout:
      min_close_bars: 2
      min_volume_mult: 1.5
      use_trend_guard_confirmation: true
    soft_deferral:
      enabled: true
      mode: "fast_watch_then_recheck"
    telemetry:
      enabled: true
      log_state_changes_only: true

signals:
  directional_bias:
    enabled: true
    mode: "quality_adjust_only"    # decision mode yok
    max_quality_delta: 0.08
    rollout:
      mode: "observe"
      canary_symbols: ["BTC/USDT:USDT"]
```

```yaml
strategies:
  level_zone_router:
    rollout:
      mode: "observe"
      canary_symbols: ["BTC/USDT:USDT"]
```

---

## 6) Safety Zinciri Uyum Notu

Mevcut sıra korunur; yeni yapı bunu bozmaz:

1. IntegrityGuard
2. RegimeFilter
3. TrendGuard
4. SafetyOverride
5. (Opsiyonel) LevelProximity size adjust

Ana veto noktası `ProductionCoordinator` router gate'idir.  
`StrategyCoordinator` tarafında mümkünse önce **gözlem/shadow**, sonra apply.

---

## 7) Test Planı (revize)

### Unit

- `tests/unit/test_key_level_detector.py`
- `tests/unit/test_level_zone_router.py`
- `tests/unit/test_directional_bias_quality_adjust.py`
- `tests/unit/test_production_level_router_soft_deferral.py`

### Integration

- `tests/test_production_coordinator_dispatch.py`  
  Recheck gate deny -> `rearm_fast_watch=false`
- `tests/integration/test_soft_deferral_flow.py`  
  `AT_LEVEL` -> deferral/recheck akışı
- `tests/integration/test_strategy_coordinator_with_levels.py`  
  Snapshot taşınması + reason code telemetry (**uygulandı**)

---

## 8) Uygulama Sıralaması

| Faz | İş | Durum | Not |
|---|---|---|---|
| P0 | `KeyLevelDetector` + unit test | Tamamlandı | Uygulandı |
| P1 | `level_zone_router.py` + snapshot modeli | Tamamlandı | Uygulandı |
| P1 | `ProductionCoordinator` ana loop + recheck gate entegrasyonu | Tamamlandı | Uygulandı |
| P1.5 | `AT_LEVEL` deferral/recheck reason handling | Tamamlandı | Rearm + reason normalization aktif |
| P2 | `DirectionalBias` quality adjust mode | Tamamlandı | Quality-only, bounded delta |
| P2 | Integration testler + telemetry doğrulama | Tamamlandı | StrategyCoordinator+levels integration testi eklendi |

---

## 9) Kabul Kriterleri (v2)

1. Key level snapshot hem main loop hem recheck path'te kullanılıyor.  
   Durum: Sağlandı
2. `AT_LEVEL` durumunda yeni entry açılmıyor; deferral/recheck akışı uygulanıyor.  
   Durum: Sağlandı
3. Snapshot yoksa sistem fail-open kalıyor, reason üretiyor.  
   Durum: Sağlandı
4. DirectionalBias tek başına trade kararı vermiyor.  
   Durum: Sağlandı
5. Mevcut strategy guard sırası korunuyor.  
   Durum: Sağlandı

Not: Bu plan kapsamındaki integration test kalemleri tamamlandı.

---

## 10) Post-MVP

- Multi-timeframe weighted fusion (`15m/1h/4h`)
- Seviye aging ve reliability score
- Breakout+retest state genişletmesi
- SafetyOverride tarafında support-distance simetrisi (long için)
- Order book / depth tabanlı seviye güçlendirme

---

## 11) Operasyonel Sonraki Adım (Observe -> Enforce)

Bu revizyonda `observe` rollout ölçümü için structured telemetry eklendi:

- `level_router_decision`
- `strategy_recheck_request`
- `soft_deferral_recheck_outcome`
- `waiting_room_drop`

Bu event'lerden otomatik karar raporu üretmek için:

```bash
python scripts/level_router_go_no_go_report.py \
  --symbol BTC/USDT:USDT \
  --log-glob "logs/live_trading_*.log" \
  --output-json artifacts/level_router_go_no_go_btc.json
```

Önerilen minimum geçiş kriterleri (varsayılan threshold):

- `total_level_router_decisions >= 500`
- `would_block_count >= 20`
- `unknown_rate <= 0.40`
- `at_level_rate <= 0.65`
- `out_of_scope_rate <= 0.00` (BTC canary scope'ta)
- level recheck error varsa `<= 0.02`
- level recheck coverage varsa `>= 0.95`

Not:
- `level_recheck_*` metrikleri sadece level-router kaynaklı pending/recheck akışı oluştuysa zorlayıcıdır.
- Observe modunda yeterli örnek toplandıktan sonra `strategies.level_zone_router.rollout.mode` değeri `enforce` yapılır.
