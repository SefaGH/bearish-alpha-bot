import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Ensure we can print "✅" on Windows terminals with non-UTF8 code pages.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure both repo root (for `import src.*`) and src/ (for `import core.*`, `import config.*`) are importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from config.live_trading_config import LiveTradingConfiguration  # noqa: E402
from config.risk_config import RiskConfiguration  # noqa: E402
from core.risk_manager import RiskManager  # noqa: E402
from core.portfolio_manager import PortfolioManager  # noqa: E402
from core.position_sizing import AdvancedPositionSizing  # noqa: E402
from core.risk_rules import CapitalLimitRule  # noqa: E402


def _pass(name: str) -> None:
    print(f"✅ PASS: {name}")


def _assert_close(actual: float, expected: float, tol: float, name: str) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected ~{expected} (+/- {tol}), got {actual}")


def _reset_network_env() -> None:
    # Guardrail: ensure no Azure AppConfig calls are attempted.
    for key in (
        "AZURE_APPCONFIG_ENDPOINT",
        "AZURE_APPCONFIG_CONNECTION_STRING",
        "APPCONFIG_ENDPOINT",
        "APPCONFIG_CONNECTION_STRING",
    ):
        os.environ.pop(key, None)


def _set_test_env() -> None:
    # Force the scenario config regardless of local developer env.
    os.environ["CAPITAL_USDT"] = "1000"
    os.environ["TRADING_LEVERAGE"] = "10"

    # Global default risk: 0.5% (adaptive uses fallback).
    os.environ["PER_TRADE_RISK_PCT"] = "0.005"


def _build_components() -> tuple[Dict[str, Any], RiskConfiguration, RiskManager, PortfolioManager, AdvancedPositionSizing]:
    _reset_network_env()
    _set_test_env()

    cfg = LiveTradingConfiguration.load(force_reload=True, log_summary=False)
    risk_cfg = RiskConfiguration(custom_limits=cfg.get("risk", {}))
    rm = RiskManager(portfolio_value=risk_cfg.initial_capital, risk_config=risk_cfg)
    pm = PortfolioManager(risk_manager=rm, performance_monitor=None)
    sizer = AdvancedPositionSizing(rm)
    return cfg, risk_cfg, rm, pm, sizer


def _resolve_dca_enabled(cfg: Dict[str, Any], strategy_name: str) -> bool:
    strategies = cfg.get("strategies") or {}
    strategy_cfg = strategies.get(strategy_name) or {}
    profile_name = strategy_cfg.get("execution_profile")
    profiles = cfg.get("execution_profiles") or {}
    profile_cfg = profiles.get(profile_name) if profile_name else None
    dca_cfg = (profile_cfg or {}).get("dca") or {}
    return bool(dca_cfg.get("enabled"))


def _make_signal(strategy_name: str) -> Dict[str, Any]:
    # Use a 0.5% stop to keep min_stop_pct from triggering and to make math deterministic:
    # - MR: risk $20 => notional $20/0.005 = $4000 (uses leverage margin affordability)
    # - Adaptive: risk $5 => notional $5/0.005 = $1000
    return {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "strategy_name": strategy_name,
        "entry": 100.0,
        "stop": 99.5,
        "leverage": 10,
    }


async def _run() -> None:
    cfg, risk_cfg, rm, pm, sizer = _build_components()

    # Scenario 1: Capital & Config Loading
    equity = float((cfg.get("risk") or {}).get("equity_usd"))
    leverage = int((cfg.get("trading") or {}).get("leverage"))
    assert equity == 1000.0, f"risk.equity_usd expected 1000.0, got {equity}"
    assert leverage == 10, f"trading.leverage expected 10, got {leverage}"
    _pass("Scenario 1: Capital & Config Loading")

    # Scenario 2: Polymorphic Risk Sizing (Sniper MR)
    mr_signal = _make_signal("mean_reversion")
    mr_sized = await sizer.calculate_optimal_size(mr_signal, return_signal=True, portfolio_manager=pm)
    mr_meta = (mr_sized or {}).get("sizing_meta") or {}
    _assert_close(float(mr_meta.get("base_risk_usd", 0.0)), 20.0, 0.25, "MR base_risk_usd")

    mr_notional = float(mr_sized.get("notional") or 0.0)
    assert mr_notional > equity, f"Expected MR notional > equity (1000), got {mr_notional}"

    # Affordability proof: CapitalLimitRule uses margin math (affordable = available * leverage * 0.95)
    ok, reason = CapitalLimitRule().validate(mr_sized, pm)
    assert ok, f"Expected leveraged affordability to pass, got ok={ok}, reason={reason}"
    _pass("Scenario 2: Polymorphic Risk Sizing (Sniper MR)")

    # Scenario 3: Global Risk Fallback (Adaptive)
    adaptive_signal = _make_signal("adaptive_str")
    adaptive_sized = await sizer.calculate_optimal_size(adaptive_signal, return_signal=True, portfolio_manager=pm)
    adaptive_meta = (adaptive_sized or {}).get("sizing_meta") or {}
    _assert_close(float(adaptive_meta.get("base_risk_usd", 0.0)), 5.0, 0.25, "Adaptive base_risk_usd")
    _pass("Scenario 3: Global Risk Fallback (Adaptive)")

    # Scenario 4: Execution Profile Routing
    assert _resolve_dca_enabled(cfg, "mean_reversion") is False, "mean_reversion dca.enabled expected False"
    assert _resolve_dca_enabled(cfg, "adaptive_str") is True, "adaptive_str dca.enabled expected True"
    _pass("Scenario 4: Execution Profile Routing")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
