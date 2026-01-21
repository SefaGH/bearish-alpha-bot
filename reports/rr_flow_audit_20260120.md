# Durum Tespiti (Read-only): Mevcut R/R Hesaplama Akışı (Strategy → Risk → Execution)

## 1) Executive Summary (≤25 satır)

- **Strateji tarafında (adaptive_ob)** R/R, `entry_price=trigger_price` (default `mid`) baz alınarak **ATR çarpanları** ile türetilen `stop/target` üzerinden hesaplanıp loglanıyor; log’daki **2.08** değeri doğrudan `tp_atr_mult/sl_atr_mult` oranından geliyor (varsayılan `2.5/1.2=2.08`). `src/strategies/adaptive_ob.py:1245` + `src/strategies/adaptive_ob.py:1266`.
- **RiskManager tarafında “Actual R/R”** yeniden hesaplanıyor ve **signal içindeki `entry/stop/target` alanlarından** okunuyor; spread/fee/slippage veya expected-fill bid/ask düzeltmesi yok. `src/core/risk_rules.py:633`.
- **Dynamic Required R/R** `RiskRewardRatioRule._calculate_dynamic_target()` içinde üretiliyor: `base - relaxation + tightening`, regime soft-weight ve **PPO çarpanı (`ppo_rr_multiplier`) ile çarpım** var. `src/core/risk_rules.py:700` + `src/core/risk_rules.py:786`.
- **PPO Action=HOLD** kuralı: PPO score `<0.5` ⇒ `Action: HOLD` loglanır, aynı skor `ppo_rr_multiplier` üretiminde “up” çarpanını seçer (default `1.3`) ve **required R/R’yi yükseltir**. `src/core/strategy_coordinator.py:6454` + `src/core/strategy_coordinator.py:7594`.
- **Post-fill**: pozisyon açıldıktan sonra `entry_price=avg_fill_price` ile `PositionManager._derive_exit_levels()` çalışır ve `signal['stop'] / signal['target']` alanlarını **fill entry** ile uyumlu olacak şekilde gerekirse düzeltir; burada ayrı bir “post-fill R/R sanity check” logu yok. `src/core/position_manager.py:1201`.

---

## 2) Uçtan Uca Akış Diyagramı (metin tabanlı)

`WebSocket Ticker/MarketDataPipeline → Strategy(adaptive_ob.signal) → Signal(entry/stop/target/rr_ratio/meta.trigger_price_source) → StrategyCoordinator(process_strategy_signal → _apply_ppo_long_filter → _enrich_signal_for_dynamic_rr) → RiskManager(validate_new_position → RiskRewardRatioRule.validate/_calculate_dynamic_target) → LiveTradingEngine(execute_signal) → PositionManager(open_position → _derive_exit_levels post-fill)`

---

## 3) “Fiyat Kaynağı ve Alanlar” Tablosu

| Aşama | Entry price source | Entry field | Stop/Target field | Clamp/Round/Normalize | R/R burada hesaplanıyor mu? |
|---|---|---|---|---|---|
| MarketDataPipeline | WS ticker’dan `mid` (bid/ask), fallback `mark/last/forming_close` | (return value) | N/A | source resolution + diag log | Hayır |
| Strategy `adaptive_ob` | `trigger_price` (default config: `mid`) veya `closed_close` | `signal['entry']` | `signal['stop']`, `signal['target']` | SL cap `max_sl_pct`, TP min `min_tp_pct`, yön LONG varsayımı | **Evet** (`rr_ratio`) |
| StrategyCoordinator enrichment | price source değiştirme yok | aynı | aynı | `side` normalize, PPO/ML/RL/regime/vol/mom eklenir, `ppo_rr_multiplier` set edilir | Hayır (opsiyonel `rr_ratio` yoksa hesaplar) |
| RiskRewardRatioRule | price source değiştirme yok | `signal['entry']` (fallback `signal['price']`) | `signal['stop']`, `signal['target']` (fallback stop_loss/take_profit) | clamp yok; sadece matematik ve eşik kontrol | **Evet** (Actual & Required) |
| PositionManager (post-fill) | `avg_fill_price` | position `entry_price` | position `stop_loss`, `take_profit` | yön düzeltmesi + eksik alanlar için fallback türetim | R/R log yok (ama P&L metriklerinde rr hesaplanır) |

Notlar:
- `adaptive_ob_trigger_price_source` default: `mid` (`config/config.example.yaml:629`), ayrıca global trigger default’u adaptive_ob’a propagate edilir (`src/config/live_trading_config.py:1419`).
- `PPO` R/R çarpanları config: `ml.reinforcement_learning.ppo_rr_up_mult` / `ppo_rr_down_mult` (`config/config.example.yaml:890`).

---

## 4) Log → Kod Eşlemesi (dosya / fonksiyon / satır)

### 🔎 `[OB R/R] Entry=..., Stop=..., Target=..., R/R=...`
- Kaynak: `AdaptiveOversoldBounce.signal()` içinde basılıyor. `src/strategies/adaptive_ob.py:1295`.
- `Entry` kaynağı: `entry_price = float(trigger_price)` (`src/strategies/adaptive_ob.py:1242`) ve `trigger_price` seçimi `MarketDataPipeline.get_live_trigger_price()` üzerinden (forming kullanılıyorsa) veya `closed_close` (`src/strategies/adaptive_ob.py:686`).

### 📊 `[Signal Enriched] ... PPO_RR=...`
- Kaynak: `StrategyCoordinator._enrich_signal_for_dynamic_rr()` sonunda basılıyor. `src/core/strategy_coordinator.py:7411` + `src/core/strategy_coordinator.py:7603`.
- `PPO_RR` üretimi: `signal['ppo_rr_multiplier']` set edilir; long ise `ppo_long_score < 0.5` ⇒ `rr_up_mult`, aksi ⇒ `rr_down_mult`. `src/core/strategy_coordinator.py:7594`.

### 🤖 `[PPO-DECISION] ... Action: HOLD ...`
- Kaynak: `StrategyCoordinator._apply_ppo_long_filter()`. `src/core/strategy_coordinator.py:6396` + `src/core/strategy_coordinator.py:6459`.
- Kural: `action_label = 'BUY' if score >= 0.5 else 'HOLD'`. `src/core/strategy_coordinator.py:6454`.

### 📊 `[Dynamic R/R Calc] Base=... → Final=...`
- Kaynak: `RiskRewardRatioRule._calculate_dynamic_target()` detailed log. `src/core/risk_rules.py:700` + `src/core/risk_rules.py:790`.
- PPO etkisi: `final_target *= max(0.1, ppo_rr_multiplier)`. `src/core/risk_rules.py:786`.

### 📊 `[R/R Analysis] Prices: Entry=..., Stop=..., Target=...`
- Kaynak: `RiskRewardRatioRule.validate()` içinde basılıyor. `src/core/risk_rules.py:615` + `src/core/risk_rules.py:665`.
- Alan okuma: `entry = signal['entry'] (fallback 'price')`, `stop=signal['stop'] (fallback 'stop_loss')`, `target=signal['target'] (fallback 'take_profit')`. `src/core/risk_rules.py:633`.

### 🚫 `[RiskRewardRatioRule] REJECTED ... below dynamic target ...`
- Kaynak: `RiskRewardRatioRule.validate()` karar kısmı. `src/core/risk_rules.py:686` + `src/core/risk_rules.py:690`.

### `TRIGGER-DIAG ... resolved_source=mid bid=... ask=...`
- Kaynak: `MarketDataPipeline.get_live_trigger_price()` diag log. `src/core/market_data_pipeline.py:1383` + `src/core/market_data_pipeline.py:1573`.

### (Varsa) post-fill exit normalize eden loglar / fonksiyonlar
- Post-fill türetim/düzeltme: `PositionManager._derive_exit_levels()` ve çağıranı. `src/core/position_manager.py:1013` + `src/core/position_manager.py:1201`.
- Post-fill stop/tp log’u (R/R değil): `Position opened ... Stop-loss ... Take-profit ...`. `src/core/position_manager.py:1343`.

---

## 5) Bulgular: Tutarsızlık / Çakışma Noktaları

1) **Strategy R/R vs Risk “Actual R/R”** aynı giriş/stop/target alanlarını kullanıyorsa aynı çıkar; fakat **execution fill** (avg_price) ile post-fill entry farklılaştığında pozisyonun efektif R/R’si strateji/risk log’larından sapabilir (post-fill stop/target yeniden “RR koruyacak” şekilde realign edilmiyor). `src/core/position_manager.py:1201`.

2) **Spread/fee/slippage** RiskRewardRatioRule “Actual R/R” hesabına dahil değil (sadece geometrik mesafe). `src/core/risk_rules.py:647`.

3) **PPO HOLD etkisi iki katmanlı**:
   - Risk eşiğini yükseltir (`ppo_rr_multiplier` ile required R/R çarpılır). `src/core/risk_rules.py:786`.
   - Position sizing’i düşürür (`ppo_position_multiplier = base + bonus*score`, score düşükse küçülür). `src/core/strategy_coordinator.py:6702`.

4) `risk.rr_dynamic.weights` içinde `volume_strength/momentum_strength` tanımlı olsa da **mevcut `RiskRewardRatioRule._calculate_dynamic_target()` bunları formüle katmıyor** (yalnızca log’da gösteriyor). `config/config.example.yaml:88` + `src/core/risk_rules.py:751`.

5) Test gözlemi (read-only): `pytest tests/test_dynamic_rr.py` koşumunda tüm testler `RiskConfiguration` “starting capital” şartı nedeniyle fail oluyor; ayrıca hata mesajı `CAPITAL_USDT` env override iddia etse de constructor sadece `equity_usd`/`initial_capital` okuyor. `src/config/risk_config.py:149`.

---

## 6) Sonuç: Durum Tespiti Özeti + İzlenmesi gereken “kanıt metrikleri”

Takip edilmesi faydalı kanıt metrikleri (kod değişikliği önermeden):
- `planned_entry (signal.entry)` vs `avg_fill_entry (position.entry_price)` farkı ve bunun `effective_rr_postfill` etkisi (post-fill stop/tp ile).
- `trigger_price_source` (mid/mark/last/forming_close) dağılımı ve `TRIGGER-DIAG` fallback oranları. `src/strategies/adaptive_ob.py:683`.
- PPO `score/action` ve `ppo_rr_multiplier` korelasyonu (HOLD → required RR yükseliyor mu). `src/core/strategy_coordinator.py:6454`.
- RiskRule seviyesinde `dynamic_target_rr` ile `calculated_rr_ratio` zaman serisi (rule zaten `signal['dynamic_rr_target']` ve `signal['calculated_rr_ratio']` set ediyor). `src/core/risk_rules.py:659`.

