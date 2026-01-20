# Runtime vol_* Telemetri Audit (Kök Neden + Minimal Patch)

Amaç: Faz‑0 offline kıyasın üretimde doğrulanabilir “runtime karar”a dönüşebilmesi için run loglarında `vol_*_bps` sayısal telemetriyi görünür kılmak.

Kapsam: **Stop/target modeli değişmez**. Sadece observability/telemetri.

---

## 1) Gözlem: 20260119 oturumunda vol_* yok
[reports/session_20260119_231242_313193.md](reports/session_20260119_231242_313193.md) içeriğinde `entry_metadata.entry_indicators` boş (`{}`) görünüyor ve logda `vol_rs_bps/vol_gk_bps/vol_yz_bps/vol_atr_bps/vol_std_bps` stringleri yok.

---

## 2) Beklenen akış: “trade kapanışı” telemetrisi nereden geliyor?

### 2.1 TRADE_CLOSED payload
`TRADE_CLOSED` payload’u [src/core/position_manager.py](src/core/position_manager.py) içinde hazırlanıyor.

- Payload’da `entry_metadata.entry_indicators` alanı, pozisyon açılışında yakalanan `position['entry_metadata']` üzerinden geliyor.
- Kapanış anında yeniden hesap yapılmıyor; **açılış anında doğru şekilde capture edilmesi şart**.

### 2.2 Entry metadata extraction
Pozisyon açılışında signal’dan metadata çıkarımı: `AdvancedPositionManager._extract_entry_metadata(signal)`.

- Bu fonksiyon sadece `signal['entry_indicators']` alanını alıyor.
- Eğer signal bu alanı set etmezse, TRADE_CLOSED tarafında `entry_indicators` doğal olarak `{}` kalıyor.

---

## 3) Kök neden: vol telemetri “meta” içinde ama log/entry_indicators’a taşınmıyor

### 3.1 Mean Reversion signal formatı
Mean Reversion stratejisi `vol_*` snapshot’ını **`signal['meta']['vol_telemetry']`** içine koyuyor:
- Kaynak: [src/strategies/mean_reversion.py](src/strategies/mean_reversion.py)
- İçerik anahtarları: `rs_bps/yz_bps/gk_bps/atr_bps/std_bps`

Bu iki nedenle runtime rapora yansımıyordu:

1) **PositionManager sadece `signal['entry_indicators']` okuyor** → `meta.vol_telemetry` yok sayılıyor.
2) **SIGNAL_BREAKDOWN logger, signal['meta'] içeriğini loglamıyor** → run log içinde vol telemetri görünmüyor.

Ek not: Eski/ayrı bir orchestrator akışında (production_coordinator) `entry_indicators` sadece RSI ile set ediliyor; MR’nin vol metrikleri bu yola hiç girmediği için zaten taşınmıyor.

---

## 4) Minimal patch (tek amaç: telemetriyi görünür kılmak)

### 4.1 TRADE_CLOSED için: entry_metadata.entry_indicators doldurma
Patch noktası: [src/core/position_manager.py](src/core/position_manager.py)

- `_extract_entry_metadata` içinde `signal['meta']['vol_telemetry']` → `entry_indicators` alanına map ediliyor:
  - `rs_bps` → `vol_rs_bps`
  - `gk_bps` → `vol_gk_bps`
  - `yz_bps` → `vol_yz_bps`
  - `atr_bps` → `vol_atr_bps`
  - `std_bps` → `vol_std_bps`

Böylece TRADE_CLOSED payload’unda `entry_metadata.entry_indicators` içinde **sayısal** vol metrikleri garantileniyor (signal meta varsa).

### 4.2 SIGNAL_BREAKDOWN için: log’a vol bloğu ekleme
Patch noktası: [src/core/strategy_coordinator.py](src/core/strategy_coordinator.py)

- `emit_signal_breakdown` artık `meta.vol_telemetry` varsa log’a `volatility` bloğunu ekliyor.

### 4.3 Ek alanlar (karar için gerekli minimum bağlam)
TRADE_CLOSED payload’unda `entry_metadata.volatility` içine best‑effort bağlam ekleniyor:
- `timeframe`
- `window` / `ddof` (config’ten)
- `selected_estimator` (şimdilik default `std`)

---

## 5) Risk değerlendirmesi
- Trading kararlarına etkisi yok.
- Payload boyutu artar (küçük, sabit boyutlu dict).
- Fail‑closed: meta.vol_telemetry yoksa hiçbir şey eklenmez.

---

## 6) DoD doğrulama (beklenen)
Telemetri patch’ten sonra alınacak yeni bir canlı oturum logunda:
- Her `SIGNAL_BREAKDOWN` event’inde `volatility.vol_*_bps` görünecek (MR sinyali meta taşıyorsa).
- Her `TRADE_CLOSED` payload’unda `entry_metadata.entry_indicators.vol_*_bps` görünecek.
