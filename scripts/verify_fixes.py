import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from strategies.mean_reversion import VWAPMeanReversion  # noqa: E402
from strategies.adaptive_ob import AdaptiveOversoldBounce  # noqa: E402
from strategies.adaptive_str import AdaptiveShortTheRip  # noqa: E402
from core.stream_data_collector import StreamDataCollector  # noqa: E402
from ml.adapters.ppo_trading_adapter import PPOTradingAdapter  # noqa: E402
from ml.ppo.observation_spec import spec_from_feature_columns  # noqa: E402
from ml.strategy_integration import MLStrategyIntegrationManager  # noqa: E402
from core.indicator_validator import IndicatorValidator  # noqa: E402
from ml.price_predictor import AdvancedPricePredictionEngine  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def make_mock_df(rows: int = 2200) -> pd.DataFrame:
    now = pd.Timestamp.utcnow().floor("1min")
    times = [now - timedelta(minutes=i) for i in range(rows)][::-1]
    base = 100.0
    prices = base + np.cumsum(np.random.randn(rows) * 0.1)
    highs = prices + np.random.rand(rows) * 0.2
    lows = prices - np.random.rand(rows) * 0.2
    opens = prices + np.random.randn(rows) * 0.05
    closes = prices + np.random.randn(rows) * 0.05
    volume = np.abs(np.random.randn(rows) * 1000) + 10

    df = pd.DataFrame(
        {
            "timestamp": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
    )
    df.set_index("timestamp", inplace=True)

    # Add mock indicators that strategies expect
    df["vwap"] = df["close"].rolling(20, min_periods=1).mean()
    df["vwap_lower"] = df["vwap"] - df["close"].rolling(20, min_periods=1).std().fillna(0)
    df["vwap_upper"] = df["vwap"] + df["close"].rolling(20, min_periods=1).std().fillna(0)
    df["adx"] = 25  # constant value below threshold to allow signals
    df["atr"] = df["close"].rolling(14, min_periods=1).std().fillna(0.1)
    df["rsi"] = 50
    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_mid"] = df["close"].ewm(span=26, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df


async def test_mean_reversion():
    cfg = {"timeframe": "1m", "signal_timeframe": "1m"}
    strat = VWAPMeanReversion(cfg=cfg)
    assert hasattr(strat, "signal"), "MeanReversion missing signal attribute"
    df = make_mock_df()
    symbol_key = "BTC/USDT:USDT"
    market_data = {symbol_key: {cfg["timeframe"]: df, cfg["signal_timeframe"]: df}}
    # inject pipeline by mocking get_latest_ohlcv
    class DummyPipeline:
        async def get_latest_ohlcv(self, symbol, timeframe, exchange=None, limit=None):
            return df

    strat.set_market_data_pipeline(DummyPipeline())
    res = await strat.signal(symbol=symbol_key, market_data=market_data)
    assert res is None or isinstance(res, dict), "MeanReversion signal returned invalid type"
    print("[PASS] MeanReversion Interface")


async def test_mean_reversion_backfill():
    cfg = {"timeframe": "1m", "signal_timeframe": "1m"}
    strat = VWAPMeanReversion(cfg=cfg)
    assert hasattr(strat, "signal")
    # Short buffer (255 rows) plus pipeline history (1500 rows)
    short_df = make_mock_df(rows=255)
    full_df = make_mock_df(rows=2200)
    symbol_key = "BTC/USDT:USDT"
    market_data = {symbol_key: {cfg["timeframe"]: short_df, cfg["signal_timeframe"]: short_df}}

    class DummyPipeline:
        async def get_latest_ohlcv(self, symbol, timeframe, exchange=None, limit=None):
            # Return full buffer to simulate in-memory pipeline cache
            return full_df

    strat.set_market_data_pipeline(DummyPipeline())
    res = await strat.signal(symbol=symbol_key, market_data=market_data)
    assert res is None or isinstance(res, dict)
    print("[PASS] MeanReversion Backfill Merge")


async def test_mean_reversion_asymmetric_lengths():
    cfg = {"timeframe": "1m", "signal_timeframe": "5m"}
    strat = VWAPMeanReversion(cfg=cfg)
    assert hasattr(strat, "signal")
    vwap_df = make_mock_df(rows=2200)
    sig_df = make_mock_df(rows=255)
    sig_df["adx"] = 10
    # Force price well below VWAP lower band to guarantee a long signal path
    sig_df.loc[sig_df.index[-1], "close"] = vwap_df["vwap_lower"].iloc[-1] - 1

    symbol_key = "BTC/USDT:USDT"
    market_data = {symbol_key: {cfg["timeframe"]: vwap_df, cfg["signal_timeframe"]: sig_df}}
    res = await strat.signal(symbol=symbol_key, market_data=market_data)
    assert res is None or isinstance(res, dict)
    print("[PASS] MeanReversion Asymmetric Lengths")


def test_adaptive_ob():
    cfg = {"debug": {"strategy_logging": True}}
    strat = AdaptiveOversoldBounce(cfg=cfg)
    assert hasattr(strat, "signal"), "Adaptive OB missing signal attribute"
    df = make_mock_df()
    # Expect no "Insufficient data" because df has plenty of rows
    res = strat.signal(df_30m=df, df_1h=df, regime_data={}, symbol="TEST/USDT")
    assert res is None or isinstance(res, dict), "Adaptive OB signal returned invalid type"
    print("[PASS] Adaptive Data Check")


def test_adaptive_str():
    cfg = {"debug": {"strategy_logging": True}}
    strat = AdaptiveShortTheRip(cfg=cfg)
    assert hasattr(strat, "signal"), "Adaptive STR missing signal attribute"
    df = make_mock_df()
    res = strat.signal(df_30m=df, df_1h=df, regime_data={}, symbol="TEST/USDT")
    assert res is None or isinstance(res, dict), "Adaptive STR signal returned invalid type"
    print("[PASS] Adaptive STR Data Check")


async def test_mean_reversion_nan_volume():
    cfg = {"timeframe": "1m", "signal_timeframe": "1m", "min_rows": 100}
    strat = VWAPMeanReversion(cfg=cfg)
    assert hasattr(strat, "signal")
    df = make_mock_df(rows=200)
    # Introduce NaN volume to force dropna to potentially empty; ensure graceful handling
    df["volume"] = float("nan")
    symbol_key = "BTC/USDT:USDT"
    market_data = {symbol_key: {cfg["timeframe"]: df, cfg["signal_timeframe"]: df}}
    res = await strat.signal(symbol=symbol_key, market_data=market_data)
    assert res is None, "Expected None when volume is NaN and indicators drop out"
    print("[PASS] MeanReversion NaN Volume Handling")


def test_collector_buffer_capacity():
    collector = StreamDataCollector()
    df = make_mock_df(rows=3000)
    collector.prime_buffer_with_dataframe("bingx", "BTC/USDT:USDT", "1m", df)
    data = collector.get_latest_ohlcv("bingx", "BTC/USDT:USDT", "1m", limit=4000)
    assert data is not None, "Collector returned no data"
    assert len(data) == 3000, f"Expected 3000 candles retained, got {len(data)}"
    print("[PASS] StreamDataCollector Buffer Capacity")


async def test_ppo_deep_fetch():
    class DummyPipeline:
        def __init__(self):
            self.last_limit = None

        async def get_latest_ohlcv(self, symbol, timeframe, limit=None):
            self.last_limit = limit
            rows = limit or 0
            return make_mock_df(rows=rows)

    class DummyFeaturePipeline:
        def extract_features(self, df, mode="price"):
            return pd.DataFrame({"f1": df["close"]}, index=df.index)

    pipeline = DummyPipeline()
    feature_pipeline = DummyFeaturePipeline()
    adapter = PPOTradingAdapter(
        rl_config={"ppo_enabled": True, "ppo_symbols": ["BTC/USDT:USDT"]},
        market_data_pipeline=pipeline,
        feature_pipeline=feature_pipeline,
    )
    adapter._spec = spec_from_feature_columns(["f1"], extra_feature_names=[], version="test")
    state, meta = await adapter._build_state("BTC/USDT:USDT")
    assert pipeline.last_limit == 2000, f"PPO fetch limit expected 2000, got {pipeline.last_limit}"
    assert state is not None, "PPO state should be built with deep history"
    print("[PASS] PPO Deep Fetch Limit")


async def test_price_predictor_deep_fetch():
    class DummyPipeline:
        def __init__(self):
            self.last_limit = None
            self.websocket_manager = None

        async def get_latest_ohlcv(self, symbol, timeframe, exchange=None, limit=None):
            self.last_limit = limit
            rows = limit or 0
            return make_mock_df(rows=rows)

    dummy_pipeline = DummyPipeline()
    # Bypass heavy __init__ by constructing object manually
    engine = AdvancedPricePredictionEngine.__new__(AdvancedPricePredictionEngine)
    engine.market_data_pipeline = dummy_pipeline
    engine.primary_timeframe = "1h"
    engine.config = {}
    engine._normalize_timeframe_value = lambda tf: tf
    res = await engine._fetch_price_data("BTC/USDT:USDT")
    assert dummy_pipeline.last_limit == 2000, f"Price predictor fetch limit expected 2000, got {dummy_pipeline.last_limit}"
    assert res is not None and len(res) == 2000, "Price predictor should return deep history"
    print("[PASS] Price Predictor Deep Fetch Limit")


async def test_ml_context_deep_fetch():
    class DummyPipeline:
        def __init__(self):
            self.last_limit = None
            self.websocket_manager = None

        async def get_latest_ohlcv(self, symbol, timeframe, exchange=None, limit=None):
            self.last_limit = limit
            rows = limit or 0
            return make_mock_df(rows=rows)

    pipeline = DummyPipeline()
    mgr = MLStrategyIntegrationManager(price_engine=None, regime_predictor=None, config={}, market_data_pipeline=pipeline)
    ctx = await mgr.get_ml_context("BTC/USDT:USDT", horizon="1h")
    assert pipeline.last_limit == 2000, f"ML context fetch limit expected 2000, got {pipeline.last_limit}"
    assert ctx is not None, "ML context should be returned"
    print("[PASS] ML Context Deep Fetch Limit")


def test_indicator_validator_deep_limit():
    collector = StreamDataCollector()
    df = make_mock_df(rows=2200)
    collector.prime_buffer_with_dataframe("bingx", "BTC/USDT:USDT", "1m", df)
    validator = IndicatorValidator(collector)
    # Directly call the internal availability check logic by invoking validate_symbol synchronously via loop
    res = asyncio.run(validator.validate_symbol("BTC/USDT:USDT", ["1m"]))
    assert "reason" in res
    assert res["status"] == "OK", f"Validator should pass with deep history, got {res}"
    print("[PASS] IndicatorValidator Deep Limit")


async def main():
    await test_mean_reversion()
    await test_mean_reversion_backfill()
    await test_mean_reversion_asymmetric_lengths()
    await test_mean_reversion_nan_volume()
    await test_ppo_deep_fetch()
    await test_price_predictor_deep_fetch()
    await test_ml_context_deep_fetch()
    test_adaptive_ob()
    test_adaptive_str()
    test_collector_buffer_capacity()


if __name__ == "__main__":
    asyncio.run(main())
