import numpy as np
import pandas as pd
import pytest

from ml.adapters.ppo_trading_adapter import PPOTradingAdapter


class DummyMarketDataPipeline:
    def __init__(self):
        self._df = pd.DataFrame(
            {
                "timestamp": [0, 1, 2],
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [10, 12, 11],
            }
        )

    async def get_latest_ohlcv(self, symbol: str, timeframe: str):
        return self._df


class DummyFeaturePipeline:
    FEATURE_DIM = 4

    def __init__(self):
        self._features = pd.DataFrame(
            [[0.1, 0.2, 0.3, 0.4]],
            columns=[f"f{i}" for i in range(self.FEATURE_DIM)],
        )

    def extract_features(self, df, mode: str = "price"):
        return self._features


@pytest.mark.asyncio
async def test_build_state_includes_tail_overrides():
    adapter = PPOTradingAdapter(
        rl_config={"ppo_enabled": True},
        market_data_pipeline=DummyMarketDataPipeline(),
        feature_pipeline=DummyFeaturePipeline(),
    )
    state = await adapter._build_state(
        "BTC/USDT",
        position_fraction=0.25,
        normalized_pv=1.4,
    )
    # 5 extra handcrafted price features + 2 tail entries appended to base features
    assert state.shape[0] == DummyFeaturePipeline.FEATURE_DIM + 7
    assert state[-2] == pytest.approx(0.25)
    assert state[-1] == pytest.approx(1.4)


@pytest.mark.asyncio
async def test_build_state_defaults_when_tail_missing():
    adapter = PPOTradingAdapter(
        rl_config={"ppo_enabled": True},
        market_data_pipeline=DummyMarketDataPipeline(),
        feature_pipeline=DummyFeaturePipeline(),
    )
    state = await adapter._build_state("BTC/USDT")
    assert state[-2] == pytest.approx(0.0)
    assert state[-1] == pytest.approx(1.0)


class _StubModel:
    def predict(self, *_args, **_kwargs):
        return np.array([1]), None


@pytest.mark.asyncio
async def test_get_long_score_includes_lookback_metadata():
    adapter = PPOTradingAdapter(
        rl_config={
            "ppo_enabled": True,
            "ppo_lookback_bars": 2,
            "ppo_lookback_windows": [2],
        },
        market_data_pipeline=DummyMarketDataPipeline(),
        feature_pipeline=DummyFeaturePipeline(),
    )
    adapter._model = _StubModel()
    score, metadata = await adapter.get_long_score("BTC/USDT")

    assert score == pytest.approx(1.0)
    assert "lookback" in metadata
    lookback = metadata["lookback"]
    assert lookback["bars_available"] >= 2
    assert "overall" in lookback
    assert lookback["window_stats"]["2"]["bars"] >= 2
