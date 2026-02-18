import ast
import asyncio
import logging

import pytest

from core.risk_rules import VolumeAwarePositionSizingRule
from core.live_trading_engine import LiveTradingEngine


class DummyRiskManager:
    async def validate_new_position(self, signal, portfolio_manager):
        return True, "", {}

    async def calculate_position_size(self, signal):
        return signal.get("position_size", 0)


class DummyOrderManager:
    async def place_order(self, request, execution_algo=None, exchange_clients=None):
        return {"success": True, "avg_price": request.get("signal", {}).get("entry")}


class DummyPositionManager:
    async def open_position(self, signal, execution_result):
        return {"success": True, "position_id": "test"}


class DummyStrategyProfileRiskConfig:
    def get_strategy_profile(self, strategy_name):
        key = str(strategy_name or "").strip().lower()
        if key != "mean_reversion":
            return {}
        return {
            "volume_bucket_risk_matrix": {
                "LOW": {
                    "stop_loss_multiplier": 1.05,
                    "take_profit_multiplier": 0.97,
                }
            }
        }


@pytest.mark.asyncio
async def test_volume_bucket_risk_logs_pre_post_caps(caplog):
    rule = VolumeAwarePositionSizingRule(
        {
            "EXTREME": {
                "position_size_multiplier": 1.5,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
            }
        }
    )

    signal = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "5m",
        "volume_bucket": "EXTREME",
        "volume_ctx_source": "analyzer",
        "position_size": 1.0,
        "entry": 100.0,
        "notional": 100.0,
        "planner_caps_snapshot": {
            "max_notional_cap": 90.0,
            "max_size_pct_notional": 120.0,
            "heat_cap_notional": 80.0,
        },
    }

    with caplog.at_level(logging.INFO):
        allowed, reason = rule.validate(signal, portfolio_manager=None)

    assert allowed is True
    assert "Applied volume bucket EXTREME" in reason

    record = next(rec for rec in caplog.records if "volume_bucket_risk" in rec.getMessage())
    payload_str = record.getMessage().split(" ", 1)[1]
    payload = ast.literal_eval(payload_str)

    assert pytest.approx(payload["base_position_size"], rel=1e-3) == 1.0
    assert pytest.approx(payload["scaled_position_size"], rel=1e-3) == 1.5
    assert pytest.approx(payload["scaled_notional"], rel=1e-3) == 150.0
    assert payload["caps_snapshot"]["heat_cap_notional"] == 80.0
    assert payload["would_breach_caps_after_volume"] is True


def test_volume_bucket_risk_uses_strategy_profile_override(caplog):
    rule = VolumeAwarePositionSizingRule(
        {
            "LOW": {
                "position_size_multiplier": 0.5,
                "stop_loss_multiplier": 1.3,
                "take_profit_multiplier": 0.9,
            },
            "NORMAL": {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
            },
        },
        risk_config=DummyStrategyProfileRiskConfig(),
    )

    signal = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "5m",
        "strategy_name": "mean_reversion",
        "volume_bucket": "LOW",
        "volume_ctx_source": "analyzer",
        "position_size": 1.0,
        "stop_loss_dist": 100.0,
        "take_profit_dist": 200.0,
    }

    with caplog.at_level(logging.INFO):
        allowed, reason = rule.validate(signal, portfolio_manager=None)

    assert allowed is True
    assert "Applied volume bucket LOW" in reason
    # Partial strategy override should inherit global position multiplier.
    assert signal["position_size"] == pytest.approx(0.5)
    assert signal["stop_loss_dist"] == pytest.approx(105.0)
    assert signal["take_profit_dist"] == pytest.approx(194.0)

    record = next(rec for rec in caplog.records if "volume_bucket_risk" in rec.getMessage())
    payload_str = record.getMessage().split(" ", 1)[1]
    payload = ast.literal_eval(payload_str)

    assert payload["strategy_name"] == "mean_reversion"
    assert payload["volume_matrix_source"] == "strategy_profile:mean_reversion"


@pytest.mark.asyncio
async def test_trade_execution_size_debug_log(caplog):
    engine = LiveTradingEngine(
        mode="paper",
        portfolio_manager=None,
        risk_manager=DummyRiskManager(),
        order_manager=DummyOrderManager(),
        position_manager=DummyPositionManager(),
        exchange_clients={"paper": object()},
    )

    signal = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "5m",
        "strategy_name": "unit_test",
        "side": "buy",
        "entry": 100.0,
        "volume_bucket": "EXTREME",
        "position_size_multiplier": 1.5,
        "planner_active": True,
        "position_size": 0.5,
        "notional": 50.0,
        "risk_assessment": {
            "metrics": {
                "final_position_size": 0.5,
                "final_notional": 50.0,
            }
        },
        "exchange": "paper",
    }

    with caplog.at_level(logging.INFO):
        await engine.execute_signal(signal)

    record = next(rec for rec in caplog.records if "trade_execution_size_debug" in rec.getMessage())
    payload_str = record.getMessage().split(" ", 1)[1]
    payload = ast.literal_eval(payload_str)

    assert pytest.approx(payload["final_position_size"], rel=1e-6) == 0.5
    assert pytest.approx(payload["final_notional"], rel=1e-6) == 50.0
    assert payload["volume_bucket_at_entry"] == "EXTREME"
    assert payload["planner_active"] is True
