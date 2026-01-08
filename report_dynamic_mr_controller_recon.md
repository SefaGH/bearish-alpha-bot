# Codebase Reconnaissance — Dynamic MR Controller (read-only)

Goal: identify *exact* integration points and constraints to implement a **Dynamic MR Controller** that adapts **`band_multiplier`** and **`vwap_lookback`** using live metrics (`z`, `band_width_pct`, `ADX`, volatility proxies), with strict safety (bounds, hysteresis, fallback) and **backward compatibility** (“disabled ⇒ identical behavior”).

This report is **recon only**: no behavior changes, no code edits.

---

## 1) File map (where things live)

### MR strategy (signal logic + MR logs)
- `src/strategies/mean_reversion.py`
  - `class VWAPMeanReversion(BaseStrategy)`
  - `async def signal(self, symbol: str, market_data: Optional[Dict[str, Any]] = None, ml_context=None, **kwargs)`
  - `async def generate_signal(self, symbol: str, market_data: Optional[Dict[str, Any]] = None, ml_context=None, **kwargs)`
  - **Important:** `__init__` reads `band_multiplier` into `self.band_mult` and logs `adx_threshold`, but current `generate_signal()` compares price only against **precomputed** `vwap_lower/vwap_upper` columns coming from the pipeline (not recomputed inside MR).

### Bands / ADX / volatility computation
- `src/core/indicators.py`
  - `def add_indicators(df, cfg=None) -> pd.DataFrame`
    - Computes and appends: `rsi`, `atr`, `ema*`, **`vwap`**, **`vwap_std`**, **`vwap_upper`**, **`vwap_lower`**, **`adx`**
    - VWAP uses `cfg["vwap_lookback"]` (default `1440`) and bands use `cfg["vwap_band_multiplier"]` (default `2.0`)
  - `def adx(df, period=14) -> pd.Series`
  - `def atr(df, period=14) -> pd.Series`

### Market data access (where indicators are attached)
- `src/core/market_data_pipeline.py`
  - `async def get_latest_ohlcv(self, symbol, timeframe, ..., limit=None, include_forming=False) -> Optional[pd.DataFrame]`
  - Always calls `add_indicators(closed_df, self.config.get('indicators'))` before returning.
  - This means **MR receives DataFrames that already include** `vwap/vwap_std/vwap_upper/vwap_lower/adx/atr`.

### Config loading (YAML + ENV + Azure App Config) and the `curl` constraint
- `src/config/live_trading_config.py`
  - `LiveTradingConfiguration.load(...)`
  - `def _load_and_merge_configs(self) -> Dict[str, Any]`
  - `def _load_from_app_config(self) -> Dict[str, Any]` (Azure App Configuration via **REST + curl + IMDS token**)
    - If `curl` is missing: logs error and returns `{}` (cloud overrides not applied).

### Strategy orchestration / where MR gets called
- `scripts/live_trading_launcher.py`
  - Instantiates MR from `config['strategies']['mean_reversion']`:
    - `mr_strategy = VWAPMeanReversion(mr_config)` (around the “MR Config” log).
- `src/core/production_coordinator.py`
  - `async def _process_trading_loop(self)` performs the per-iteration orchestration.
  - In the “STRATEGY EXECUTION AND SIGNAL FORWARDING STAGE” it calls each strategy’s `signal(**signal_kwargs)` and **awaits coroutines**:
    - Builds `signal_kwargs` containing `symbol`, `ml_context`, and conditionally `market_data` if supported by signature.
    - If a signal is produced: `await self.strategy_coordinator.process_strategy_signal(strategy_name, signal)`

### Risk/cooldowns/limits (safety envelopes already in system)
- `src/core/risk_manager.py`
  - `_create_default_rules()` adds `DailyTradeLimitRule` via `_get_daily_max_trades_from_config()`
- `src/core/strategy_coordinator.py`
  - `validate_duplicate()` implements cooldown + min price movement checks (duplicate prevention)
  - `PrioritySignalQueue` enforces TTL, queue depth, per-symbol pending caps, etc.

---

## 2) Integration points (where to inject a Dynamic MR Controller)

### Candidate A (recommended): inside MR at evaluation time
**Where:** `src/strategies/mean_reversion.py` → `VWAPMeanReversion.generate_signal()`, right after `last_vwap/last_sig` extraction and before:
- `in_band = vwap_lower <= price <= vwap_upper`

**How:** compute “effective” `lookback` and `band_multiplier` (with bounds + hysteresis) and derive **effective** `vwap_lower_eff/vwap_upper_eff` used for `in_band` decision.

**Why recommended:**
- Per-strategy isolation: does not affect other strategies that share `MarketDataPipeline`.
- Easy backward compatibility: `controller.enabled=False` ⇒ use existing `vwap_lower/vwap_upper` columns exactly.
- Data required for controller already present in `df_vwap/df_sig` (OHLCV + indicators).

**Constraints / gotchas:**
- Dynamic `vwap_lookback` requires recomputing VWAP + std on the fly using `df_vwap` raw columns (`high/low/close/volume`) to match `add_indicators()` math, otherwise you only adapt multiplier (not lookback).
- Must avoid thrashing: strategy loop runs frequently; last closed `5m` candle changes every 5 minutes; implement update interval / “only update on new candle” logic.

### Candidate B: extend `MarketDataPipeline.get_latest_ohlcv()` with per-call indicator overrides
**Where:** `src/core/market_data_pipeline.py` → `get_latest_ohlcv(...)` signature.

**How:** add an optional `indicator_overrides` argument merged into `self.config['indicators']` *for that call* before calling `add_indicators(...)`.

**Pros:**
- Keeps band computation centralized and consistent.
- Controller can request a different lookback/mult without re-implementing VWAP math in strategy.

**Cons / constraints:**
- Cross-strategy coupling risk if implemented by mutating shared `self.config`.
- Requires threading/async safety (pipeline is shared; avoid global mutation).
- API surface change ripples to all callers (but can be optional arg with default `None`).

### Candidate C: controller in orchestrator, pass overrides to MR via kwargs/market_data
**Where:** `src/core/production_coordinator.py` → `_process_trading_loop()` where `signal_kwargs` are assembled.

**How:** compute controller outputs in coordinator and pass:
- a dedicated kwarg (e.g., `mr_controller={...}`) because MR `signal()` accepts `**kwargs`, or
- embed in `market_data` under a reserved key.

**Pros:**
- Controller can be shared across symbols/strategies, with centralized telemetry.

**Cons:**
- Tighter coupling to call-site; multiple call sites exist historically.
- More moving parts to ensure “disabled ⇒ identical behavior”.

**Recommended integration point:** **Candidate A (inside `VWAPMeanReversion.generate_signal`)** for least blast radius + easiest backward compatibility.

---

## 3) Data availability per MR cycle (confirmed from code)

From `MarketDataPipeline.get_latest_ohlcv()` → `add_indicators()`:

Available every time `df_*` is returned:
- Raw: `open, high, low, close` (+ `volume` when present)
- Indicators: `rsi, atr, ema21/50/200, vwap, vwap_std, vwap_upper, vwap_lower, adx`

From `VWAPMeanReversion.generate_signal()`:
- **px** = `last_sig["close"]` (signal timeframe close; default `5m`)
- **bands** = `last_vwap["vwap_lower"] / ["vwap_upper"]` (vwap timeframe; default `1m`)
- **vwap mid** = `last_vwap["vwap"]`
- **ADX** = `last_sig["adx"]`
- **ATR** (vol proxy) = optional `last_sig["atr"]`
- Candle history lengths are available as `len(df_vwap)`, `len(df_sig)`; MR also logs them.

Volatility proxies you can use without new dependencies:
- `atr` (already computed)
- `vwap_std` (already computed with the same lookback as vwap)
- realized volatility from `df_sig["close"].pct_change().rolling(...)` (cheap to compute in-strategy)

---

## 4) Config wiring (how MR config reaches runtime, and what it currently controls)

### Strategy config path (MR)
- Source: `config/config.example.yaml` → `strategies.mean_reversion`
- Loaded by: `LiveTradingConfiguration.load()` (deep-merged with ENV/AppConfig)
- Instantiated by: `scripts/live_trading_launcher.py`:
  - `mr_config = config['strategies']['mean_reversion']`
  - `VWAPMeanReversion(mr_config)`

### Indicator config path (VWAP/ADX/ATR math)
- Source: `config/config.example.yaml` → `indicators` (global)
- Applied by: `MarketDataPipeline.get_latest_ohlcv()`:
  - `add_indicators(df, self.config.get('indicators'))`
- **Key constraint:** `add_indicators()` expects:
  - `vwap_lookback`
  - `vwap_band_multiplier`
  - `adx_period`
  - but `strategies.mean_reversion` uses `vwap_lookback` + `band_multiplier` naming.

### Practical implication (important constraint)
- MR’s `band_multiplier` and `vwap_lookback` in `strategies.mean_reversion` are **not** currently what `add_indicators()` consumes (it uses `indicators.vwap_band_multiplier` and `indicators.vwap_lookback`).
- In your logs the values match the defaults (1440 / 2.0), so behavior “looks consistent”, but if an operator changes MR config only, bands may **not** change unless the indicator config path is also wired.

### Safest approach to add optional controller config (future)
- Put controller config under MR strategy block to keep it opt-in:
  - `strategies.mean_reversion.dynamic_controller.enabled: false`
  - Include explicit bounds, update cadence, hysteresis settings.
- Ensure **disabled path** is byte-for-byte equivalent in behavior:
  - `if not enabled: use existing vwap_lower/vwap_upper as today`

---

## 5) Logging / telemetry (current state + recommended JSON schema)

### Current MR logs (from `VWAPMeanReversion.generate_signal()`)
- `Processing signal...`
- `Data rows: vwap=..., signal_tf=..., min_vwap=..., min_signal=...`
- `Price within bands ... px=..., lower=..., upper=..., adx=..., adx_th=...`
- `Price outside bands but ADX veto ...`
- `Cycle complete ... Action: HOLD|SIGNAL` (note: doesn’t log BUY/SELL explicitly)

### Recommended controller decision event (JSON-friendly)
Emit *one* structured event when controller evaluates (and only when it changes params or on a fixed interval):

```json
{
  "event": "mr_controller_decision",
  "symbol": "BTC/USDT:USDT",
  "ts_utc": "2026-01-07T23:10:42Z",
  "vwap_tf": "1m",
  "signal_tf": "5m",
  "inputs": {
    "px": 90975.0,
    "vwap": 91916.38,
    "vwap_std": 764.29,
    "adx": 16.5,
    "atr": 123.4
  },
  "derived": {
    "z": -1.2317,
    "band_width_pct": 3.3260
  },
  "params": {
    "band_multiplier_prev": 2.0,
    "band_multiplier_new": 1.15,
    "vwap_lookback_prev": 1440,
    "vwap_lookback_new": 240
  },
  "safety": {
    "clamped": false,
    "hysteresis_hold": true,
    "min_update_interval_s": 300
  },
  "reason": "z_abs_below_target; narrowing bands to increase trigger sensitivity"
}
```

Implementation detail (future): use `logger.info("...", extra={...})` or a dedicated JSON logger to keep parsing stable.

---

## 6) Risk & safety constraints already present (and how controller should respect them)

### Existing envelopes
- Daily max trades:
  - `src/core/risk_manager.py:_create_default_rules()` adds `DailyTradeLimitRule` if configured.
- Duplicate prevention & cooldown:
  - `src/core/strategy_coordinator.py:validate_duplicate()` enforces cooldown and min price change thresholds.
- Signal queue throttling:
  - `src/core/strategy_coordinator.py:PrioritySignalQueue` caps queue depth and per-symbol pending signals.

### Controller-specific safety constraints (design requirements)
- Hard bounds:
  - `band_multiplier ∈ [min,max]`
  - `vwap_lookback ∈ [min,max]` and ≤ available rows (or fetch more data)
- Hysteresis:
  - deadband around target z (avoid flipping every cycle)
  - minimum update interval (e.g., 5–15 minutes) or only update on new closed candle.
- Fallback:
  - if data insufficient, NaNs, or volatility spikes → revert to static defaults
- Backward compatibility:
  - `controller.enabled=false` must follow the exact existing path (use pipeline-provided bands as-is).

---

## 7) Test strategy (unit tests + VST smoke)

### Unit tests (pytest)
Existing patterns:
- `tests/test_live_trading_config.py` (pytest style)
- `tests/validate_vwap_strategy.py` (async harness for MR + `add_indicators`)

Recommended new unit tests (future):
- `tests/test_dynamic_mr_controller.py`
  - “disabled ⇒ identical behavior”: same inputs yield same `in_band` decision and same bands used.
  - “bounds & clamp”: extreme z/volatility clamps params to min/max.
  - “hysteresis”: small metric noise does not change params.
  - “data insufficiency”: missing volume/NaN returns fallback defaults and logs reason.

How to run:
- `pytest -q`

### VST smoke run (ops-level)
VM helper exists:
- `scripts/vm_run_session.py` builds a consistent `docker run` for the VM (`--env-file`, volumes, container name).
Suggested smoke procedure (future):
1) Run in `BINGX_ENV=vst` with a single symbol.
2) Enable controller (config flag) but keep tight bounds + long update interval for first run.
3) Collect a short window log and verify:
   - controller emits decision logs
   - MR produces occasional outside events without blowing through cooldown/daily limits

---

## Curl / AppConfig constraint (deploy-time)

Observed in logs: App Configuration fetch fails when `curl` is missing.

Code path:
- `src/config/live_trading_config.py:_load_from_app_config()` calls `subprocess.check_output(['curl', ...])` for IMDS token + AppConfig query.
- If `curl` is missing → returns `{}` and cloud overrides are skipped.

Remediation options (not implemented here):
1) Ensure the deployed image actually includes `curl` (verify the Docker build context and the Dockerfile used in CI/CD).
2) Replace the curl+IMDS path with Azure SDK (since the module already has optional Azure imports) to remove runtime dependency on `curl`.

