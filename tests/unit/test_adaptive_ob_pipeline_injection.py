from src.core.production_coordinator import ProductionCoordinator
from src.strategies.adaptive_ob import AdaptiveOversoldBounce


class DummyPortfolioManager:
    def register_strategy(self, strategy_name, strategy_instance, initial_allocation):
        return {"status": "success"}


def test_adaptive_ob_pipeline_injection():
    coordinator = ProductionCoordinator(config={"debug": {}})
    coordinator.is_initialized = True
    coordinator.portfolio_manager = DummyPortfolioManager()
    dummy_pipeline = object()
    coordinator.market_data_pipeline = dummy_pipeline

    strategy = AdaptiveOversoldBounce(cfg={"debug": {"strategy_logging": False}})
    assert strategy.market_data_pipeline is None

    result = coordinator.register_strategy("adaptive_ob", strategy, initial_allocation=0.25)

    assert strategy.market_data_pipeline is dummy_pipeline
    assert result.get("status") == "success"
