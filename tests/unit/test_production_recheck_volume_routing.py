import pandas as pd
import pytest

from src.core.production_coordinator import ProductionCoordinator


class _CaptureMeanReversion:
    strategy_name = "mean_reversion"
    vwap_tf = "1m"
    signal_tf = "5m"
    min_rows = 2
    min_signal_rows = 2

    def __init__(self):
        self.last_kwargs = None

    async def signal(self, **kwargs):
        self.last_kwargs = kwargs
        return {
            "event_type": "strategy_recheck_decision",
            "decision_meta": {"rearm_fast_watch": False},
        }


class _PortfolioStub:
    def __init__(self, strategy):
        self.strategies = {"mean_reversion": strategy}
        self.strategy_metadata = {"mean_reversion": {"active": True}}


class _PipelineStub:
    def __init__(self, df_vwap: pd.DataFrame, df_sig: pd.DataFrame):
        self._df_vwap = df_vwap
        self._df_sig = df_sig

    async def get_latest_ohlcv(self, _symbol: str, timeframe: str, limit=None, include_forming=True):
        if timeframe == "1m":
            return self._df_vwap
        if timeframe == "5m":
            return self._df_sig
        return self._df_sig


def _build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "close": [100.0, 100.0],
            "volume": [1.0, 1.0],
            "vwap": [100.0, 100.0],
            "vwap_lower": [99.0, 99.0],
            "vwap_upper": [101.0, 101.0],
            "vwap_std": [1.0, 1.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame({"close": [100.0, 100.0], "adx": [10.0, 10.0], "volume": [1.0, 1.0]}, index=idx)
    return df_vwap, df_sig


@pytest.mark.asyncio
async def test_dispatch_strategy_passes_recheck_upstream_volume_context_to_mean_reversion():
    strategy = _CaptureMeanReversion()
    df_vwap, df_sig = _build_frames()

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.portfolio_manager = _PortfolioStub(strategy)
    prod.market_data_pipeline = _PipelineStub(df_vwap, df_sig)
    prod.strategy_coordinator = object()
    prod.ml_integration = None
    prod.market_regime_analyzer = None
    prod.strategies = {}
    prod.config = {}

    detail = await prod.dispatch_strategy(
        "BTC/USDT:USDT",
        "mean_reversion",
        parent_pending_id="pending-1",
        pending_id="pending-1",
        condition_data={"near": "upper"},
        check_detail={
            "fast_watch": {"touch_confirmed": True, "dist_to_band_bps": 1.0},
            "volume": {"volume_strength": 0.83, "volume_bucket": "HIGH", "source": "analyzer"},
        },
        return_detail=True,
    )

    assert isinstance(detail, dict)
    assert detail.get("dispatched") is False
    assert strategy.last_kwargs is not None
    assert strategy.last_kwargs.get("volume_strength") == pytest.approx(0.83)
    assert strategy.last_kwargs.get("volume_bucket") == "HIGH"
    assert strategy.last_kwargs.get("volume_source") == "analyzer"
    assert strategy.last_kwargs.get("volume_analysis", {}).get("source") == "analyzer"
