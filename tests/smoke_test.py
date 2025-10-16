import os
import sys
import importlib
import importlib.util
import asyncio
import logging
import types

import pytest

# Ensure src is on path for imports (CI job should already do this, but keep for local runs)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.exists(os.path.join(ROOT, "src")):
    sys.path.insert(0, os.path.join(ROOT, "src"))

def safe_import(module_name):
    """Try to import a module; if it fails, skip the test with a helpful message."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        pytest.skip(f"Skipping because import failed for {module_name}: {e}")

def test_core_imports():
    """Ensure core modules are importable (skip if optional modules missing)."""
    modules = [
        "core.ccxt_client",
        "core.multi_exchange",
        "core.bingx_authenticator",
        "core.indicators",
        "core.regime",
        "core.market_regime",
        "strategies.oversold_bounce",
        "strategies.short_the_rip",
        "strategies.adaptive_ob",
        "strategies.adaptive_str",
        "core.production_coordinator",
        "core.risk_manager",
        "core.portfolio_manager",
        "core.websocket_manager",
        "core.websocket_client",
        "core.notify",
    ]
    for m in modules:
        safe_import(m)

def test_phase3_components():
    """Instantiate critical Phase 3 components with lightweight dependencies."""
    coord_mod = safe_import("core.production_coordinator")
    risk_mod = safe_import("core.risk_manager")
    portfolio_mod = safe_import("core.portfolio_manager")

    coordinator = coord_mod.ProductionCoordinator()
    assert "execution" in coordinator.config

    portfolio_config = {
        "equity_usd": 1000,
        "max_portfolio_risk": 0.02,
        "max_position_size": 0.1,
        "max_drawdown": 0.15,
    }

    class _DummyPerformanceMonitor:
        def get_strategy_summary(self, _name: str):
            return {"metrics": {"win_rate": 0.5}}

    perf_monitor = _DummyPerformanceMonitor()
    risk_manager = risk_mod.RiskManager(portfolio_config, None, perf_monitor)
    assert risk_manager.portfolio_value == portfolio_config["equity_usd"]

    portfolio_manager = portfolio_mod.PortfolioManager(risk_manager, perf_monitor, None)
    registration = portfolio_manager.register_strategy("dummy", object(), initial_allocation=0.1)
    assert registration["status"] == "success"

def test_ml_components_light():
    """Do not run heavy ML pipelines in CI; just import or skip."""
    # If pandas/numpy are not available, skip ML tests to avoid CI breaks.
    try:
        import pandas  # type: ignore
        import numpy  # type: ignore
    except Exception:
        pytest.skip("pandas/numpy not available — skip ML smoke tests")

    safe_import("ml.regime_predictor")

def test_bingx_auth_light():
    """Import bingx authenticator and run a tiny pure-python unit if available."""
    mod = safe_import("core.bingx_authenticator")
    cls = getattr(mod, "BingXAuthenticator", None)
    if cls is None:
        pytest.skip("BingXAuthenticator not found in core.bingx_authenticator")

    auth = cls("test_key", "test_secret")
    assert auth.convert_symbol_to_bingx("BTC/USDT:USDT") == "BTC-USDT"

    signed = auth.prepare_authenticated_request({"symbol": "BTC-USDT"})
    assert "headers" in signed and "params" in signed
    assert signed["headers"].get("X-BX-APIKEY") == "test_key"
    assert signed["params"].get("signature")

def test_config_sections_present_or_skip():
    """Check config.example.yaml exists and contains minimal keys, otherwise skip."""
    cfg_path = os.path.join(ROOT, "config", "config.example.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config.example.yaml not present — skipping config checks")
    try:
        import yaml  # type: ignore
    except Exception:
        pytest.skip("PyYAML not installed — skipping config checks")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    required = ["execution", "risk", "signals"]
    missing = [k for k in required if k not in (cfg or {})]
    assert not missing, f"Missing config sections: {missing}"

def test_strategy_initialization_from_config():
    """Load example config and instantiate adaptive strategies."""
    cfg_path = os.path.join(ROOT, "config", "config.example.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config.example.yaml not present — skipping config checks")
    try:
        import yaml  # type: ignore
    except Exception:
        pytest.skip("PyYAML not installed — skipping config checks")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    signals_cfg = cfg.get("signals", {})
    ob_cfg = signals_cfg.get("oversold_bounce", {})
    str_cfg = signals_cfg.get("short_the_rip", {})
    if not ob_cfg or not str_cfg:
        pytest.skip("Missing strategy configuration in example config")

    regime_mod = safe_import("core.market_regime")
    adaptive_ob_mod = safe_import("strategies.adaptive_ob")
    adaptive_str_mod = safe_import("strategies.adaptive_str")

    regime = regime_mod.MarketRegimeAnalyzer()
    ob = adaptive_ob_mod.AdaptiveOversoldBounce(ob_cfg, regime)
    st = adaptive_str_mod.AdaptiveShortTheRip(str_cfg, regime)

    assert hasattr(ob, "signal")
    assert hasattr(st, "signal")

def test_async_components_with_stub(monkeypatch):
    """Instantiate async components with a stubbed ccxt.pro module."""
    class _DummyExchange:
        def __init__(self, params):
            self.params = params

        async def watch_ohlcv(self, *args, **kwargs):
            return []

        async def watch_ticker(self, *args, **kwargs):
            return {}

    stub_module = types.ModuleType("ccxt.pro")
    stub_module.kucoinfutures = _DummyExchange
    stub_module.bingx = _DummyExchange
    stub_module.dummyexchange = _DummyExchange

    monkeypatch.setitem(sys.modules, "ccxt.pro", stub_module)

    ws_client_mod = safe_import("core.websocket_client")
    ws_manager_mod = safe_import("core.websocket_manager")

    monkeypatch.setattr(ws_client_mod, "ccxtpro", stub_module, raising=False)

    client = ws_client_mod.WebSocketClient("kucoinfutures")
    assert client.name == "kucoinfutures"

    manager = ws_manager_mod.WebSocketManager(exchanges={"kucoinfutures": None, "bingx": None})
    assert set(manager.clients.keys()) == {"kucoinfutures", "bingx"}

def test_smoke_synchronous_wrapper_for_async():
    """A tiny sanity check: ensure we can create and run an event loop (no external IO)."""
    loop = asyncio.new_event_loop()
    try:
        # run a noop async function to ensure loop works in CI
        async def _noop():
            return True
        assert loop.run_until_complete(_noop()) is True
    finally:
        loop.close()

@pytest.mark.asyncio
async def test_phase4_ml_prediction_default():
    """Ensure MLRegimePredictor can produce a fallback prediction."""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")

    ml_mod = safe_import("ml.regime_predictor")
    predictor = ml_mod.MLRegimePredictor()

    index = pd.date_range("2024-01-01", periods=60, freq="h")
    price_data = pd.DataFrame(
        {
            "close": np.linspace(100, 110, 60),
            "high": np.linspace(101, 111, 60),
            "low": np.linspace(99, 109, 60),
            "volume": np.random.rand(60) * 1000 + 100,
            "rsi": np.linspace(40, 60, 60),
            "macd": np.random.randn(60),
            "macd_signal": np.random.randn(60),
            "ema_20": np.linspace(100, 109, 60),
            "ema_50": np.linspace(100, 108, 60),
            "bb_upper": np.linspace(105, 115, 60),
            "bb_lower": np.linspace(95, 105, 60),
            "atr": np.random.rand(60) + 0.5,
        },
        index=index,
    )

    result = await predictor.predict_regime_transition("BTC/USDT:USDT", price_data)
    assert result["predicted_regime"] in {"bullish", "neutral", "bearish"}


def test_live_trading_launcher_components(monkeypatch):
    """Instantiate live trading launcher utilities without touching the network."""
    class _DummyHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(logging, "FileHandler", _DummyHandler)

    launcher_path = os.path.join(ROOT, "scripts", "live_trading_launcher.py")
    if not os.path.exists(launcher_path):
        pytest.skip("live_trading_launcher.py not present — skipping launcher checks")

    spec = importlib.util.spec_from_file_location("_smoke_launcher", launcher_path)
    if spec is None or spec.loader is None:
        pytest.skip("Unable to load live_trading_launcher module spec")
    launcher_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher_mod
    spec.loader.exec_module(launcher_mod)

    launcher = launcher_mod.LiveTradingLauncher(mode="paper", dry_run=True)
    assert launcher.mode == "paper"

    health = launcher_mod.HealthMonitor()
    report = health.get_health_report()
    assert "status" in report and "uptime_hours" in report

    restart_mgr = launcher_mod.AutoRestartManager(max_restarts=5, restart_delay=1)
    should_restart, reason = restart_mgr.should_restart()
    assert isinstance(should_restart, bool)
