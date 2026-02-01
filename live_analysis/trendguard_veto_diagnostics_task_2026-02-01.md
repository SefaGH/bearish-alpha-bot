# Task: TrendGuard veto kararını kanıta bağlayan teşhis (diagnostics) çıktısı

## Özet
Canlı loglarda MeanReversion sinyalleri sık sık `trend_guard_veto_long_breakout_down` ile reddediliyor; ancak veto’nun hangi metriklerle tetiklendiği log’a yazılmadığı için “falling knife tespiti doğru muydu?” sorusu kanıta dayalı cevaplanamıyor.

Bu task, TrendGuard veto’su gerçekleştiğinde **hesaplanan metrikleri** ve **kullanılan candle zaman damgalarını** kontrollü bir şekilde loglayarak, kararın geriye dönük olarak doğrulanabilir olmasını sağlar.

## Amaç
- Veto gerçekleştiğinde: *sıkışma / breakout / slope* koşullarını sayısal değerleriyle görmek.
- Zaman uyumsuzluğu tartışmalarını bitirmek: TrendGuard’ın kullandığı OHLCV’nin `last_closed_ts`/`retrieved_at` gibi kaynak-zaman metadata’sını loglamak.
- Log gürültüsünü kontrol altında tutmak: throttle/sampling ile spam’ı engellemek.

## Mevcut Durum (Bulgu)
### 1) Veto etiketinin kaynağı ve gerçek tanımı
TrendGuard veto etiketini şu koşullarla üretir:
- `squeeze_recent == True` (son `squeeze_lookback` içinde BBW, `bbw_squeeze_thr` altına inmiş)
- `close < lower` (Bollinger alt bandı altına kırılım)
- `bbw_ratio >= bbw_expand_thr` (volatilite genişlemesi; BBW / son `bbw_expand_lookback` medyanı)
- `slope <= -slope_dn_thr` (EMA eğimi; ATR ile normalize)

Kod referansı:
- `src/safety/trend_guard.py` → `check_veto()`

### 2) Coordinator meta’yı sinyale yazıyor ama loglamıyor
`src/core/strategy_coordinator.py` TrendGuard sonucunu:
- `enriched_signal['meta']['trend_guard'] = guard_result.meta_data` olarak ekliyor
- veto olursa sadece reason string’i logluyor: `REJECTED (TrendGuard): <reason>`

Bu yüzden canlı logdan yalnızca reason etiketi görülüyor; metrikler görünmüyor.

### 3) TrendGuard’ın kullandığı OHLCV kaynağı metadata üretiyor
`src/core/market_data_pipeline.py` (WS path) dönüş dataframe’ine şu attrs alanlarını set edebiliyor:
- `ohlcv_source` (örn. `ws`)
- `retrieved_at` (UTC ISO)
- `last_closed_ts`, `forming_ts`, `forming_last_update_ts`, `gap_count` (collector state)

Bu alanlar, “hangi bar’a göre veto oldu?” sorusunu çok hızlı netleştirir.

## Kapsam / Değişiklikler
### A) Veto anında teşhis log’u (structured) ekle
**Yer:** `src/core/strategy_coordinator.py` TrendGuard veto branch’i.

**Davranış:** `guard_result.is_vetoed == True` olduğunda, mevcut warning loguna ek olarak **tek satır** daha yaz:
- log level: default `WARNING` (konfigürasyonla değişebilir)
- mesaj, hem reason hem de özet metrikleri içersin

Önerilen log alanları:
- identity: `symbol`, `strategy_name`, `side`, `guard_tf`
- TrendGuard meta: `squeeze_recent`, `breakout_dir`, `close`, `upper`, `lower`, `bbw`, `bbw_ratio`, `bbw_squeeze_thr`, `bbw_expand_thr`, `slope`, `slope_up_thr`, `slope_dn_thr`, `body_ratio`
- OHLCV attrs (varsa): `ohlcv_source`, `retrieved_at`, `last_closed_ts`, `forming_ts`, `gap_count`

Örnek format (tek satır):
- `[TREND-GUARD][VETO] sym=... strat=... side=... tf=... reason=... squeeze_recent=... breakout_dir=... close=... lower=... bbw_ratio=... bbw_expand_thr=... slope=... slope_dn_thr=... src=... last_closed_ts=... retrieved_at=...`

Notlar:
- Sayısal değerleri `round()`/format ile kısalt (örn. slope 6-8 decimal).
- `guard_result.meta_data` zaten dict; güvenli şekilde `get()` ile çek.

### B) Log spam kontrolü (throttle / sampling)
Yeni konfig anahtarları önerisi (trend_guard altında):
- `veto_diag_enabled: true|false` (default: false)
- `veto_diag_log_level: "WARNING"|"INFO"|"DEBUG"` (default: "WARNING")
- `veto_diag_throttle_seconds: 60` (default: 60)
- `veto_diag_key_fields: ["symbol", "timeframe", "side", "reason"]` (opsiyonel, default bu)

Uygulama önerisi:
- Coordinator içinde basit bir in-memory throttle map:
  - key = `(symbol, guard_tf, side, guard_result.reason)`
  - value = `last_logged_monotonic` (time.monotonic)
- throttle süresi dolmadan tekrar veto olursa *diagnostic satırı* basma; mevcut short reason warning’ini basmaya devam edebilirsin (ya da onu da throttle’layabilirsin, ama scope’u büyütmemek için sadece diag satırını throttle’lamak yeterli).

### C) Meta formatını sabitle (stabil schema)
TrendGuard meta_data zaten zengin; ancak log alanları sabit bir “minimum set” içermeli.

Öneri:
- Coordinator, loglayacağı alanların isimlerini sabit tutar.
- Eksik alanlar için `n/a` / `None` basar.

### D) Dokümantasyon / opsiyonel config örneği
**Yer:** `config/config.example.yaml`
- `trend_guard:` altına yukarıdaki `veto_diag_*` anahtarlarını ekle.
- 2 örnek preset:
  - `veto_diag_enabled: false` (default)
  - Canlı analiz modu: `true` + throttle 60s

### E) Unit test (minimum)
**Yer:** test altyapısına göre (repo’da `pytest.ini` var).

Minimum test seti:
1) **breakout_down veto**
- Yapay bir OHLCV DataFrame oluştur.
- Sıkışma + breakout şartlarını tetikleyecek şekilde:
  - önce dar BBW (düşük volatilite) bar’ları
  - sonra alt band altı close ve yüksek BBW ratio
  - slope’u negatif yapacak trend
- `TrendGuard.check_veto(... side='long')` → `is_vetoed == True` ve `reason == 'trend_guard_veto_long_breakout_down'`.

2) **pass case**
- Aynı setup’ta `side='short'` → pass veya farklı reason.

Not: Bu testler için “dinamik threshold” hesapları quantile bazlı olduğu için, test DF’yi yeterince uzun tutmak (min_history_bars’ı test config ile düşürmek) daha deterministik olur.

## Kabul Kriterleri
- Veto olduğunda (ve diag enabled iken) loglarda en az şu alanlar görünür:
  - `reason`, `squeeze_recent`, `breakout_dir`, `close`, `lower`, `bbw_ratio`, `bbw_expand_thr`, `slope`, `slope_dn_thr`
  - mümkünse `ohlcv_source`, `last_closed_ts`, `retrieved_at`
- Throttle çalışır: aynı key için `veto_diag_throttle_seconds` dolmadan ikinci diag satırı basılmaz.
- Diag kapalıyken davranış değişmez (mevcut reason warning logu aynı kalır).
- Unit test en az 1 veto senaryosunu doğrular.

## Operasyonel Doğrulama (Canlı)
- Canlı botta `trend_guard.veto_diag_enabled=true` aç.
- 1-2 dakika içinde TrendGuard veto olduğunda logda `[TREND-GUARD][VETO]` satırı görülsün.
- Satırdaki `last_closed_ts` ve `retrieved_at` değerleri UTC olarak mantıklı olsun.

## Notlar / Riskler
- Coordinator throttle map process-memory’dir; restart’ta sıfırlanır (kabul edilebilir).
- Log satırı PII içermez; yalnızca piyasa metrikleri içerir.
- Bu task, trade logic’i değiştirmez; sadece gözlemlenebilirliği artırır.
