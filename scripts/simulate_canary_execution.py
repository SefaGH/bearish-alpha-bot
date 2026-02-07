#!/usr/bin/env python3
"""Generate deterministic canary telemetry with a stubbed execution harness.

Purpose:
- Exercise LiveTradingEngine smart-entry decision flow without real exchange I/O.
- Emit parseable telemetry markers used by canary_go_no_go_report.py:
  - order_decision_trace
  - order_decision_outcome
  - order_manager_decision
  - TRADE_CLOSED

This is an offline simulation helper for CI and local validation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path for `src.*` imports when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.config.risk_config import RiskConfiguration
from src.core.live_trading_engine import LiveTradingEngine, TradingMode
from src.core.risk_manager import RiskManager


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _setup_logging(log_file: Path, level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


class StubPortfolioManager:
    def __init__(self, equity: float):
        self._equity = float(equity)

    def get_total_equity(self) -> float:
        return self._equity

    def get_total_exposure(self) -> float:
        return 0.0

    def get_current_drawdown(self) -> float:
        return 0.0

    def get_open_positions(self) -> Dict[str, Any]:
        return {}


@dataclass(frozen=True)
class Scenario:
    signal_id: str
    symbol: str
    side: str
    entry: float
    atr: float | None
    bucket: str
    strategy_name: str
    should_abort_timeout: bool
    slippage_bps: float
    notional: float
    time_to_fill_ms: float


class StubOrderManager:
    def __init__(self, slippage_by_signal: Dict[str, float], abort_signals: set[str], time_to_fill_by_signal: Dict[str, float]):
        self._slippage_by_signal = dict(slippage_by_signal)
        self._abort_signals = set(abort_signals)
        self._time_to_fill_by_signal = dict(time_to_fill_by_signal)
        self._counter = 0
        self._logger = logging.getLogger("core.order_manager")

    async def place_order(self, order_request: Dict[str, Any], execution_algo: str) -> Dict[str, Any]:
        signal = (order_request or {}).get("signal") or {}
        signal_id = str(signal.get("signal_id") or "")
        symbol = str(order_request.get("symbol") or signal.get("symbol") or "UNKNOWN")
        side = str(order_request.get("side") or signal.get("side") or "buy")

        is_abort = signal_id in self._abort_signals
        fallback_reason = "limit_timeout_market_fallback_disabled:extreme_bucket" if is_abort else None
        reason = "ABORT:NO_FILL_TIMEOUT" if is_abort else None
        effective = str(execution_algo or "market").lower()

        self._logger.info(
            "order_manager_decision %s",
            {
                "event": "order_manager_decision",
                "timestamp": _utc_now_iso(),
                "symbol": symbol,
                "side": side,
                "requested_execution_algo": effective,
                "effective_execution_algo": effective,
                "env_forced_order_type": None,
                "fallback_reason": fallback_reason,
                "success": not is_abort,
                "reason": reason,
            },
        )

        if is_abort:
            return {
                "success": False,
                "order_id": None,
                "reason": reason,
                "fallback_reason": fallback_reason,
                "effective_execution_algo": effective,
                "time_to_fill_ms": float(self._time_to_fill_by_signal.get(signal_id, 400.0)),
            }

        self._counter += 1
        avg_price = float(signal.get("entry") or 0.0)
        amount = float(order_request.get("amount") or signal.get("position_size") or 0.0)
        slippage_bps = float(self._slippage_by_signal.get(signal_id, 2.0))
        return {
            "success": True,
            "order_id": f"sim-order-{self._counter}",
            "avg_price": avg_price,
            "filled_amount": amount,
            "effective_execution_algo": effective,
            "slippage": slippage_bps / 10000.0,
            "fallback_reason": None,
            "time_to_fill_ms": float(self._time_to_fill_by_signal.get(signal_id, 150.0)),
        }


class StubPositionManager:
    def __init__(self):
        self._counter = 0

    async def open_position(self, signal: Dict[str, Any], execution_result: Dict[str, Any]) -> Dict[str, Any]:
        self._counter += 1
        position_id = f"sim-pos-{self._counter}"
        return {
            "success": True,
            "position_id": position_id,
            "position": {
                "symbol": signal.get("symbol"),
                "side": signal.get("side"),
            },
        }


def _trade_closed_payload(
    *,
    scenario: Scenario,
    entry_slippage_bps: float,
    stop_overshoot_bps: float,
) -> Dict[str, Any]:
    ts = _utc_now_iso()
    side = "BUY" if str(scenario.side).lower() in {"buy", "long"} else "SELL"
    return {
        "event": "TRADE_CLOSED",
        "timestamp": ts,
        "run_id": "sim_canary",
        "trade_id": f"trade_{scenario.signal_id}",
        "position_id": f"pos_{scenario.signal_id}",
        "symbol": scenario.symbol,
        "timeframe": "5m",
        "side": side,
        "strategy": scenario.strategy_name,
        "strategy_name": scenario.strategy_name,
        "entry_price": scenario.entry,
        "entry_time": ts,
        "exit_price": scenario.entry,
        "exit_order_id": None,
        "exit_time": ts,
        "exit_reason": "time_exit",
        "entry_order_id": f"entry_{scenario.signal_id}",
        "position_size": scenario.notional / scenario.entry,
        "pnl_usd": 0.0,
        "realized_pnl_usd": 0.0,
        "realized_pnl_usdt": 0.0,
        "pnl_pct": 0.0,
        "rr": 1.0,
        "rr_achieved": 1.0,
        "rr_after_fill": 1.2,
        "planned_vs_realized_rr_drift": -0.2,
        "duration_min": 1.0,
        "entry_slippage_bps": float(entry_slippage_bps),
        "entry_notional_usd": float(scenario.notional),
        "stop_ref_price": scenario.entry * 0.995,
        "stop_overshoot_bps": float(stop_overshoot_bps),
        "volume_bucket_at_entry": scenario.bucket,
        "volume_strength_at_entry": 0.95 if scenario.bucket == "EXTREME" else 0.6,
    }


def _build_engine(order_manager: StubOrderManager) -> LiveTradingEngine:
    cfg = RiskConfiguration(
        custom_limits={
            "max_position_size": 0.10,
            "position_size_policy": "clip",
            "min_notional_threshold": 5.0,
            "size_planner_enabled": True,
            "max_portfolio_risk": 1.0,
        },
        initial_capital=1000,
    )
    pm = StubPortfolioManager(equity=1000.0)
    rm = RiskManager(portfolio_value=1000.0, risk_config=cfg)

    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=pm,
        risk_manager=rm,
        order_manager=order_manager,
        position_manager=StubPositionManager(),
        exchange_clients={"paper": object()},
    )
    engine.config = {
        "trading": {"order_type": "market"},
        "smart_entry_policy": {
            "enabled": True,
            "force_override": False,
            "volatility_threshold_bps": 5.0,
            "force_market_on_missing_atr": False,
            "force_market_on_low_vol": False,
            "extreme_market_ban": True,
            "fallback_timeout_seconds": 30,
            "params": {
                "LONG": {"atr_multiplier": 0.90, "timeout_seconds": 60, "gate_bps": 6.0},
                "SHORT": {"atr_multiplier": 0.85, "timeout_seconds": 60, "gate_bps": 8.0},
            },
        },
        "order_manager": {
            "market_fallback_on_timeout_enabled": True,
            "disable_market_fallback_on_extreme_bucket": True,
            "disable_market_fallback_on_fast_move": True,
        },
    }
    return engine


def _build_signal(s: Scenario) -> Dict[str, Any]:
    qty = s.notional / s.entry
    stop = s.entry * 0.99
    target = s.entry * 1.02
    signal: Dict[str, Any] = {
        "signal_id": s.signal_id,
        "symbol": s.symbol,
        "side": s.side,
        "entry": s.entry,
        "stop": stop,
        "target": target,
        "strategy_name": s.strategy_name,
        "strategy": s.strategy_name,
        "volume_bucket": s.bucket,
        "volume_strength": 0.95 if s.bucket == "EXTREME" else 0.60,
        "position_size": qty,
        "notional": s.notional,
        "exchange": "paper",
    }
    if s.atr is not None:
        signal["atr"] = s.atr
    return signal


async def _run(output_log: Path) -> Dict[str, Any]:
    scenarios: List[Scenario] = [
        Scenario(
            signal_id="sig_extreme_limit_1",
            symbol="BTC/USDT:USDT",
            side="buy",
            entry=10000.0,
            atr=12.0,
            bucket="EXTREME",
            strategy_name="adaptive_ob",
            should_abort_timeout=False,
            slippage_bps=3.5,
            notional=100.0,
            time_to_fill_ms=140.0,
        ),
        Scenario(
            signal_id="sig_normal_limit_1",
            symbol="BTC/USDT:USDT",
            side="buy",
            entry=10100.0,
            atr=14.0,
            bucket="NORMAL",
            strategy_name="adaptive_ob",
            should_abort_timeout=False,
            slippage_bps=2.1,
            notional=110.0,
            time_to_fill_ms=160.0,
        ),
        Scenario(
            signal_id="sig_missing_atr_extreme",
            symbol="BTC/USDT:USDT",
            side="buy",
            entry=10200.0,
            atr=None,
            bucket="EXTREME",
            strategy_name="adaptive_ob",
            should_abort_timeout=False,
            slippage_bps=4.8,
            notional=90.0,
            time_to_fill_ms=190.0,
        ),
        Scenario(
            signal_id="sig_timeout_abort_extreme",
            symbol="BTC/USDT:USDT",
            side="buy",
            entry=10300.0,
            atr=13.0,
            bucket="EXTREME",
            strategy_name="adaptive_ob",
            should_abort_timeout=True,
            slippage_bps=5.0,
            notional=95.0,
            time_to_fill_ms=620.0,
        ),
    ]

    order_manager = StubOrderManager(
        slippage_by_signal={s.signal_id: s.slippage_bps for s in scenarios},
        abort_signals={s.signal_id for s in scenarios if s.should_abort_timeout},
        time_to_fill_by_signal={s.signal_id: s.time_to_fill_ms for s in scenarios},
    )
    engine = _build_engine(order_manager)
    trade_logger = logging.getLogger("core.position_manager")

    successful = 0
    failed = 0

    for idx, scenario in enumerate(scenarios):
        signal = _build_signal(scenario)
        result = await engine.execute_signal(signal)
        if result.get("success"):
            successful += 1
            payload = _trade_closed_payload(
                scenario=scenario,
                entry_slippage_bps=scenario.slippage_bps,
                stop_overshoot_bps=2.5 + (idx * 1.8),
            )
            trade_logger.info("TRADE_CLOSED %s", json.dumps(payload))
        else:
            failed += 1

    logging.getLogger("src.core.live_trading_engine").warning(
        "[RECON-WATCHDOG] stale_removed=0 orphans_detected=0 orphans_adopted=0 active_positions=0"
    )

    summary = {
        "log_file": str(output_log),
        "scenarios_total": len(scenarios),
        "successful": successful,
        "failed": failed,
    }
    logging.getLogger("simulate_canary_execution").info("simulation_summary %s", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic canary telemetry with a stub harness.")
    parser.add_argument(
        "--log-file",
        default="artifacts/canary/sim_harness.log",
        help="Output log file for simulated telemetry.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    _setup_logging(log_path, args.log_level)

    summary = asyncio.run(_run(log_path))
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
