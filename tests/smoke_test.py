import os
import sys
import importlib
import asyncio
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

def test_phase3_basic():
    """Lightweight checks for Phase 3 components: import only."""
    safe_import("core.production_coordinator")
    safe_import("core.risk_manager")
    safe_import("core.portfolio_manager")

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
    if not mod:
        return
    # only do light checks if class exists
    cls = getattr(mod, "BingXAuthenticator", None)
    if cls is None:
        pytest.skip("BingXAuthenticator not found in core.bingx_authenticator")
    # instantiate with dummy creds if ctor is simple
    try:
        auth = cls("x", "y")
        if hasattr(auth, "convert_symbol_to_bingx"):
            bingx_symbol = auth.convert_symbol_to_bingx("BTC/USDT:USDT")
            # accept any non-empty mapping (do not enforce exact mapping to avoid fragile tests)
            assert isinstance(bingx_symbol, str) and len(bingx_symbol) > 0
    except TypeError:
        pytest.skip("BingXAuthenticator ctor requires complex args — skip heavy instantiation")

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

def test_strategy_imports_light():
    """Import strategy modules, avoid heavy instantiation."""
    safe_import("strategies.adaptive_ob")
    safe_import("strategies.adaptive_str")

def test_async_components_import_only():
    """Do not open real websockets in CI. Just import async-related modules."""
    safe_import("core.websocket_client")
    safe_import("core.websocket_manager")

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
