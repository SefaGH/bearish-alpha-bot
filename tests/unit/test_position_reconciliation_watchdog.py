import pytest

from src.core.position_manager import AdvancedPositionManager


class DummyRiskManager:
    def __init__(self):
        self.active_positions = {}

    def register_position(self, position_id, position):
        self.active_positions[position_id] = dict(position)


class DummyPortfolioManager:
    def __init__(self):
        self.active_positions = {}

    def register_position(self, position_id, position):
        self.active_positions[position_id] = dict(position)


class FakeExchangeClient:
    def __init__(self, positions):
        self._positions = list(positions)

    def fetch_positions(self, symbols=None, params=None):
        return list(self._positions)


@pytest.mark.asyncio
async def test_runtime_reconcile_detects_orphan_and_removes_stale_local(clean_env):
    risk = DummyRiskManager()
    portfolio = DummyPortfolioManager()
    pm = AdvancedPositionManager(risk_manager=risk, order_manager=object(), portfolio_manager=portfolio)

    pm.positions["pos_local"] = {
        "position_id": "pos_local",
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "amount": 0.01,
        "entry_price": 100.0,
        "current_price": 100.0,
        "exchange": "bingx",
    }
    risk.active_positions["pos_local"] = dict(pm.positions["pos_local"])
    portfolio.active_positions["pos_local"] = dict(pm.positions["pos_local"])

    client = FakeExchangeClient(
        positions=[
            {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 0.02, "entryPrice": 2000.0},
        ]
    )
    result = await pm.reconcile_runtime_positions(exchange_clients={"bingx": client}, adopt_orphans=False)

    assert result["stale_removed"] == 1
    assert result["orphans_detected"] == 1
    assert result["orphans_adopted"] == 0
    assert "pos_local" not in pm.positions
    assert "pos_local" not in risk.active_positions
    assert "pos_local" not in portfolio.active_positions


@pytest.mark.asyncio
async def test_runtime_reconcile_adopts_orphan_when_enabled(clean_env):
    risk = DummyRiskManager()
    portfolio = DummyPortfolioManager()
    pm = AdvancedPositionManager(risk_manager=risk, order_manager=object(), portfolio_manager=portfolio)

    client = FakeExchangeClient(
        positions=[
            {"symbol": "BTC/USDT:USDT", "side": "short", "contracts": 0.0538, "entryPrice": 98500.0},
        ]
    )
    result = await pm.reconcile_runtime_positions(exchange_clients={"bingx": client}, adopt_orphans=True)

    assert result["stale_removed"] == 0
    assert result["orphans_detected"] == 1
    assert result["orphans_adopted"] == 1
    assert len(pm.positions) == 1

    position_id, position = next(iter(pm.positions.items()))
    assert position_id.startswith("orphan_bingx_")
    assert position.get("strategy_name") == "exchange_orphan_watchdog"
    assert position.get("orphan_adopted") is True
    assert position.get("amount") == pytest.approx(0.0538, rel=1e-12)
    assert position_id in risk.active_positions
    assert position_id in portfolio.active_positions
