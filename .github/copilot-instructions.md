# GitHub Copilot Instructions for Bearish Alpha Bot

## 🐍 Runtime & Environment
- **Python 3.11.x ONLY**: Strictly required due to `aiohttp==3.8.6` and `ccxt.pro` compatibility.
  - ❌ Do NOT use Python 3.12+ or 3.10-.
  - ✅ Use `actions/setup-python@v5` with `python-version: "3.11"`.
- **Windows Dev**: Use `.\pytest.cmd` to run tests (sets `PYTHONPATH` correctly).
- **Imports**: The project supports dual import styles, but prefer **package-style** (`from src.core...`) for production code.
  - `scripts/` add `src/` to `sys.path` dynamically.

## 🏗️ Architecture & Core Components
- **Entry Point**: `scripts/live_trading_launcher.py` initializes `ProductionCoordinator`.
- **Orchestrator**: `core/production_coordinator.py` wires up:
  - `MarketDataPipeline` (Data aggregation & validation)
  - `WebSocketManager` (Real-time data, optimized with `OptimizedWebSocketManager`)
  - `RiskManager` & `PortfolioManager` (Execution safety)
  - `StrategyCoordinator` (Signal generation & gating)
- **ML Integration**: `src/ml/` hosts the ML stack.
  - **Gemma Adapter**: `src/ml/adapters/gemma/` handles TorchScript models (`.pt`).
  - **Circuit Breaker**: Implemented in Gemma adapter for fault tolerance.
- **Execution**: `core/live_trading_engine.py` handles order lifecycle.

## ⚙️ Configuration
- **Central Config**: `config/live_trading_config.py` loads `config/config.example.yaml`.
- **Risk Config**: `config/risk_config.py` normalizes risk parameters (e.g., `max_position_size`).
- **Env Vars**: `EXCHANGES`, `BINGX_KEY`, `BINGX_SECRET`, `TELEGRAM_BOT_TOKEN`.
- **Pattern**: Config loader caches results; clear `_config_instance` in tests if mutating env.

## 🧪 Testing Strategy
- **Framework**: `pytest` with `pytest-asyncio`.
- **Structure**:
  - `tests/unit/`: Fast, isolated tests.
  - `tests/integration/`: Slower, full-system tests.
- **Markers**: `@pytest.mark.integration`, `@pytest.mark.unit`, `@pytest.mark.slow`.
- **Commands**:
  - Run all: `.\pytest.cmd tests/ -v`
  - Unit only: `.\pytest.cmd tests/ -m "not integration"`
  - Integration: `.\pytest.cmd tests/integration/ -v`

## 🚀 Deployment (Azure VM)
- **Docker**: Multi-container setup on Azure VM (`BearishAlphaBot-VM-01`).
  - `bearish-bot`: Main trading container.
  - `log-parser`: Sidecar for structured log parsing (ndjson).
  - `fluent-bit`: Log forwarder to Azure Log Analytics.
- **Image**: `bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-4` (or latest).
- **Volumes**: Host `/mnt/bearish/logs` -> Container `/app/logs`; Host `/mnt/bearish/data` -> Container `/app/data`.
- **Management**:
  - **Helper**: Use `python scripts/vm_run_session.py` inside VM for one-step update/restart.
  - **Remote**: Use `az vm run-command invoke ...` for agentless management.
  - **Analysis**: Run `python scripts/run_last_session_analysis.py` inside container to sync logs and analyze.

## 📊 Reporting & Analytics (Azure)
- **Flow**: Logs -> Log Parser (ndjson) -> Fluent Bit -> Log Analytics -> ADX -> Power BI.
- **Automation**: Logic App triggers Azure Function on container stop to generate PDF reports & email.
- **Key Files**:
  - Raw Logs: `logs/live_trading_*.log` (Source of Truth).
  - Parsed Events: `/data/parsed/<run_id>.ndjson`.
  - ADX Table: `bearish_events`.

## 💡 Key Patterns & Gotchas
- **Logging**: Use `core.logger.setup_logger`. Do NOT use `logging.basicConfig`.
- **Async**: Heavy use of `asyncio`. Ensure `await` on async calls.
- **Duplicate Prevention**: `StrategyCoordinator` uses `symbol:strategy` keys with price delta bypass.
- **Symbol Normalization**: `BTC/USDT` -> `BTC/USDT:USDT` for futures. Use `_normalize_symbol_for_ws`.
- **ML Fallback**: Gemma adapter returns neutral prediction on failure (Circuit Breaker OPEN).
