import pytest
from types import SimpleNamespace

from core.live_trading_engine import LiveTradingEngine, TradingMode


class DummyPortfolioManager:
    def __init__(self, equity: float):
        self._equity = equity

    def get_total_equity(self):
        return self._equity

    def get_total_exposure(self):
        return 0.0


class DummyRiskManager:
    def __init__(self):
        self.calculate_calls = 0
        self.validated_signals = []

    async def validate_new_position(self, signal, portfolio_manager):
        self.validated_signals.append(signal.copy())
        notional = (signal.get("position_size", 0) or 0) * (signal.get("entry", 0) or 0)
        equity = portfolio_manager.get_total_equity() if portfolio_manager else 0.0
        risk_metrics = {
            "new_position_value": notional,
            "position_size_pct": (notional / equity) if equity else 0.0,
        }
        return True, "ok", risk_metrics

    async def calculate_position_size(self, signal):
        self.calculate_calls += 1
        return 42.0


class DummyOrderManager:
    def __init__(self):
        self.last_request = None

    async def place_order(self, order_request, execution_algo):
        self.last_request = order_request
        return {"success": True, "order_id": "order-1"}


class DummyPositionManager:
    async def open_position(self, signal, execution_result):
        return {"success": True, "position_id": "pos-1", "position": {}}


@pytest.mark.asyncio
async def test_execute_signal_uses_planner_size_and_skips_multipliers():
    portfolio_manager = DummyPortfolioManager(equity=100.0)
    risk_manager = DummyRiskManager()
    order_manager = DummyOrderManager()
    position_manager = DummyPositionManager()

    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        exchange_clients={"binance": object()},
    )

    engine.config = {"trading": {"order_type": "limit"}}
    engine.execution_analytics = SimpleNamespace(get_best_execution_algorithm=lambda notional, urgency: "limit")

    signal = {
        "signal_id": "sig-1",
        "symbol": "BTC/USDT",
        "side": "buy",
        "entry": 10.0,
        "planner_active": True,
        "planner_planned_notional": 10.0,
        "planner_raw_notional": 15.0,
        "planner_cap_flags": {},
        "position_size": 1.0,
        "notional": 10.0,
        "sizing_meta": {"ppo_position_multiplier": 1.5},
        "position_multiplier": 1.2,
        "risk_assessment": {"metrics": {"final_position_size": 1.0, "final_notional": 10.0}},
    }

    result = await engine.execute_signal(signal)

    assert result["success"] is True
    expected_qty = float(signal["notional"]) / float(signal["entry"])
    assert order_manager.last_request["amount"] == pytest.approx(expected_qty, rel=1e-6)
    assert risk_manager.calculate_calls == 0
    assert risk_manager.validated_signals, "validate_new_position should be called"
    validated = risk_manager.validated_signals[-1]
    assert validated.get("notional") == pytest.approx(10.0)
    assert validated.get("position_size") == pytest.approx(expected_qty, rel=1e-6)
    assert engine.trade_history[-1]["risk_metrics"]["new_position_value"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_execute_signal_postfill_rr_early_exit_closes_position(monkeypatch):
    portfolio_manager = DummyPortfolioManager(equity=100.0)
    risk_manager = DummyRiskManager()
    order_manager = DummyOrderManager()

    class DummyPositionManager:
        async def open_position(self, signal, execution_result):
            return {
                "success": True,
                "position_id": "pos-1",
                "position": {
                    "position_id": "pos-1",
                    "symbol": signal.get("symbol"),
                    "side": signal.get("side"),
                    "exchange": "binance",
                    "amount": 1.0,
                    "entry_price": signal.get("entry"),
                    "current_price": signal.get("entry"),
                    "rr_required": 1.58,
                    "rr_after_fill": 1.2,
                    "postfill_action": "early_exit",
                    "postfill_reason_code": "rr_below_required",
                },
            }

    position_manager = DummyPositionManager()

    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        exchange_clients={"binance": object()},
    )

    engine.config = {"trading": {"order_type": "limit"}}
    engine.execution_analytics = SimpleNamespace(get_best_execution_algorithm=lambda notional, urgency: "limit")

    called = {"exit": 0, "exit_reason": None}

    async def fake_exit(position_id, exit_signal):
        called["exit"] += 1
        called["exit_reason"] = (exit_signal or {}).get("exit_reason")
        engine.active_positions.pop(position_id, None)
        return {"success": True, "exit_reason": called["exit_reason"]}

    monkeypatch.setattr(engine, "_execute_position_exit", fake_exit)

    signal = {
        "signal_id": "sig-rr-1",
        "symbol": "BTC/USDT",
        "side": "buy",
        "entry": 10.0,
        "planner_active": True,
        "planner_planned_notional": 10.0,
        "planner_raw_notional": 15.0,
        "planner_cap_flags": {},
        "position_size": 1.0,
        "notional": 10.0,
        "risk_assessment": {"metrics": {"final_position_size": 1.0, "final_notional": 10.0}},
    }

    result = await engine.execute_signal(signal)

    assert result["stage"] == "postfill_rr_early_exit"
    assert called["exit"] == 1
    assert called["exit_reason"] == "postfill_rr_below_required"
    assert result["close_result"]["success"] is True
    assert "pos-1" not in engine.active_positions


def test_canary_metrics_snapshot_aggregates_postfill_same_signal_band_and_slippage():
    portfolio_manager = DummyPortfolioManager(equity=100.0)
    risk_manager = DummyRiskManager()
    order_manager = DummyOrderManager()
    position_manager = DummyPositionManager()

    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        exchange_clients={"binance": object()},
    )

    engine.config = {
        "monitoring": {
            "canary_metrics": {
                "enabled": True,
                "interval_sec": 300,
                "canary_symbols": ["BTC/USDT:USDT"],
            }
        }
    }

    engine._record_execution_quality_sample(
        signal={"symbol": "BTC/USDT:USDT"},
        execution_result={"slippage": 0.0004},
        postfill_early_exit=True,
    )
    engine._record_execution_quality_sample(
        signal={"symbol": "BTC/USDT:USDT"},
        execution_result={"slippage": 0.0001},
        postfill_early_exit=False,
    )
    engine._record_execution_quality_sample(
        signal={"symbol": "ETH/USDT:USDT"},
        execution_result={"slippage": 0.0009},
        postfill_early_exit=False,
    )

    class StubCoordinator:
        def get_duplicate_prevention_stats(self):
            return {
                "total_signals_processed": 20,
                "rejected_by_same_signal": 2,
                "same_signal_repeat_rate": 10.0,
            }

        def get_processing_stats(self):
            return {
                "stats": {
                    "band_snapshot_mismatch_count": 3,
                    "band_snapshot_checks": 50,
                }
            }

    engine.strategy_coordinator = StubCoordinator()

    snapshot = engine._build_canary_metrics_snapshot()

    assert isinstance(snapshot, dict)
    assert snapshot["scope"]["kind"] == "canary_symbols"
    assert snapshot["entries_total"] == 2
    assert snapshot["postfill_early_exit_count"] == 1
    assert snapshot["postfill_exit_rate"] == pytest.approx(0.5, rel=1e-6)
    assert snapshot["same_signal_repeat_rate"] == pytest.approx(10.0, rel=1e-6)
    assert snapshot["band_snapshot_mismatch_count"] == 3
    assert snapshot["band_snapshot_checks"] == 50
    assert snapshot["slippage_sample_count"] == 2
    assert snapshot["slippage_p95_bps"] == pytest.approx(3.85, rel=1e-3)


def test_canary_metrics_snapshot_alert_thresholds_trigger_critical():
    portfolio_manager = DummyPortfolioManager(equity=100.0)
    risk_manager = DummyRiskManager()
    order_manager = DummyOrderManager()
    position_manager = DummyPositionManager()

    engine = LiveTradingEngine(
        mode=TradingMode.PAPER.value,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        exchange_clients={"binance": object()},
    )

    engine.config = {
        "monitoring": {
            "canary_metrics": {
                "enabled": True,
                "interval_sec": 300,
                "canary_symbols": ["BTC/USDT:USDT"],
                "alerts": {
                    "enabled": True,
                    "min_entries": 1,
                    "min_same_signal_total": 1,
                    "min_band_checks": 1,
                    "min_slippage_samples": 1,
                    "thresholds": {
                        "postfill_exit_rate": {"warning": 0.20, "critical": 0.35},
                        "same_signal_repeat_rate": {"warning": 5.0, "critical": 10.0},
                        "band_snapshot_mismatch_rate": {"warning": 0.02, "critical": 0.05},
                        "slippage_p95_bps": {"warning": 2.0, "critical": 4.0},
                    },
                },
            }
        }
    }

    engine._record_execution_quality_sample(
        signal={"symbol": "BTC/USDT:USDT"},
        execution_result={"slippage": 0.0005},  # 5 bps
        postfill_early_exit=True,
    )

    class StubCoordinator:
        def get_duplicate_prevention_stats(self):
            return {
                "total_signals_processed": 10,
                "rejected_by_same_signal": 2,
                "same_signal_repeat_rate": 20.0,
            }

        def get_processing_stats(self):
            return {
                "stats": {
                    "band_snapshot_mismatch_count": 1,
                    "band_snapshot_checks": 5,
                }
            }

    engine.strategy_coordinator = StubCoordinator()

    snapshot = engine._build_canary_metrics_snapshot()

    assert isinstance(snapshot, dict)
    assert snapshot["alert_status"] == "critical"
    assert snapshot["alert_count"] == 4
    alert_meta = snapshot.get("alerts", {})
    assert alert_meta.get("evaluated_metrics") == 4

    reason_codes = {item.get("reason_code") for item in alert_meta.get("alerts", [])}
    assert "monitoring.canary.postfill_exit_rate.critical" in reason_codes
    assert "monitoring.canary.same_signal_repeat_rate.critical" in reason_codes
    assert "monitoring.canary.band_snapshot_mismatch_rate.critical" in reason_codes
    assert "monitoring.canary.slippage_p95_bps.critical" in reason_codes
