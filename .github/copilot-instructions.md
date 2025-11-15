# GitHub Copilot Instructions for Bearish Alpha Bot

## Runtime Guardrails
- Python 3.11.x is the only supported runtime; keep `.python-version`, `pyproject.toml` (`>=3.11,<3.12`), `runtime.txt`, Dockerfiles, and docs aligned.
- GitHub Actions must call `actions/setup-python@v5` with `python-version: "3.11"` (or `python-version-file: ".python-version"`); never introduce `3.x` shorthands or version matrices.
- Core deps (`aiohttp==3.8.6`, `ccxt.pro`, `torch`) crash on 3.12; verify upgrades against `PYTHON_311_TEST_RESULTS.md` before touching them.
- Local shells can source `setup_python311_env.sh`; Windows runners rely on `pytest.cmd` to prepend `src/` to `PYTHONPATH`.

## Architecture Map
- `scripts/live_trading_launcher.py` is the production entrypoint: it shells into `core/production_coordinator.py`, which wires phases (multi-exchange → market intelligence → risk/portfolio → execution → ML).
- `core/production_coordinator.py` manages lifecycle objects (`MarketDataPipeline`, `WebSocketManager`, `RiskManager`, `PortfolioManager`, `StrategyCoordinator`, `LiveTradingEngine`) and enforces phased health checks.
- `src/main.py` offers a lighter “scan + optional execute” loop for Actions workflows; it still leans on `core.multi_exchange`, `core.exec_engine`, and adaptive strategies.
- Real-time data flows through `core/market_data_pipeline.py` and `core/websocket_manager.py`; strategy gating and duplicate prevention live in `core/strategy_coordinator.py` (config-driven via `signals.duplicate_prevention`).
- Execution & accounting are isolated in `core/live_trading_engine.py`, `core/order_manager.py`, `core/position_manager.py`, and `core/portfolio_manager.py`; ML augmentation sits under `src/ml/` with GEMMA manifests in `artifacts/`.

## Config & Environment
- `config/live_trading_config.py` parses `config/config.example.yaml`, honoring `# Override with:` annotations to map env vars → nested keys; keep comments intact when editing YAML.
- Required env surface: `EXCHANGES` (comma list), exchange credentials (`{EXCHANGE}_KEY/SECRET[/PASSWORD]`), `EXECUTION_EXCHANGE`, optional `MODE=live|paper`, Telegram IDs, and `LOG_LEVEL`.
- By default the universe is fixed (`config.universe.fixed_symbols`); enabling auto-select loads markets via `src/universe.py` and applies USDT-only filters.
- Adaptive strategy toggles, duplicate prevention thresholds, and TP/SL multipliers live under `config.signals.*`; ML switches reside under `config.ml` and must match trained artifacts.
- Runtime artifacts land in `data/` (signal CSVs, state, quarantine) and `logs/` (queue-based logging from `core/logger.py`, with `live_trading_latest.log` symlinked).

## Developer Workflow
- On Windows, run tests with `.\pytest.cmd` (sets `PYTHONPATH`): e.g. `.\pytest.cmd tests/test_live_trading_launcher.py -v` or `.\pytest.cmd tests/test_phase3_low_priority.py -k WebSocket`.
- Full suites: `.\pytest.cmd tests -m "not slow"` (markers documented in `tests/README.md`); integration smoke lives in `tests/test_integration_smoke.py` and `tests/test_live_trading_workflow.py`.
- Production dry run: `python scripts/live_trading_launcher.py --dry-run` performs pre-flight checks; paper tests: `python scripts/live_trading_launcher.py --paper --duration 900`.
- Minimal scanner flow for Actions: set `EXCHANGES=bingx`, credentials, then `MODE=paper python src/main.py` (writes `data/RUN_SUMMARY.txt` and signal CSVs).
- For ML pipelines ensure `artifacts/` manifests are synced, then run `python scripts/train_all_models.py --config config/config.example.yaml` followed by targeted verifiers (`.\pytest.cmd tests/test_gemma_integration.py`).

## Patterns & Gotchas
- Logging uses a queue listener (`core/logger.py`); call `setup_logger` once and respect the queue to avoid duplicate logs—direct `logging.basicConfig` calls will be ignored.
- Duplicate signal handling depends on `StrategyCoordinator.validate_duplicate`; keep `signals.duplicate_prevention.min_price_change_pct` expressed as decimals (0.0005 = 0.05%).
- WebSocket collectors normalize symbols to `BTC/USDT:USDT`; when adding pairs ensure both REST and WS symbols align via `_normalize_symbol_for_ws`.
- Config loader caches results; tests that mutate env should clear `config.live_trading_config._config_instance` before reloading to avoid shared state bleed.
- GEMMA adapters expect manifest-driven paths (`artifacts/<bundle>/manifest.json`); adding models without updating manifests breaks `StrategyCoordinator` AI-gate initialization.
