import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure UTF-8 output for symbols on Windows shells
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure repository root and src on sys.path (adaptive strategies import "core.*")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.core.indicators import add_indicators
from src.strategies.mean_reversion import VWAPMeanReversion


class MockPipeline:
    """Minimal MarketDataPipeline stub."""

    def __init__(self, df):
        self.df = df

    async def get_latest_ohlcv(self, symbol: str, timeframe: str, exchange: str | None = None):
        return self.df


def make_mock_df(rows: int = 1440) -> pd.DataFrame:
    """Create a simple increasing price series with random noise and volume."""
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq="1min")
    base = np.linspace(100, 105, rows)  # gentle drift
    noise = np.random.normal(0, 0.2, rows)
    close = base + noise
    high = close + np.random.uniform(0.01, 0.1, rows)
    low = close - np.random.uniform(0.01, 0.1, rows)
    open_ = close + np.random.uniform(-0.05, 0.05, rows)
    volume = np.random.uniform(50, 150, rows)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df


def validate_indicators(df: pd.DataFrame):
    required = ["vwap", "vwap_upper", "vwap_lower", "adx"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing indicator columns: {missing}"
    last = df.iloc[-1]
    for col in required:
        assert not pd.isna(last[col]), f"{col} is NaN on last row"


async def run_scenario():
    # 1) Mock data and indicators
    raw_df = make_mock_df(1440)
    ind_df = add_indicators(raw_df, {"vwap_lookback": 1440, "vwap_band_multiplier": 2.0, "adx_period": 14})
    validate_indicators(ind_df)

    # 2) Force scenario: price below lower band, ADX low
    forced_df = ind_df.copy()
    forced_df.iloc[-1, forced_df.columns.get_loc("close")] = forced_df["vwap_lower"].iloc[-1] * 0.99
    forced_df.iloc[-1, forced_df.columns.get_loc("high")] = forced_df["close"].iloc[-1] * 1.001
    forced_df.iloc[-1, forced_df.columns.get_loc("low")] = forced_df["close"].iloc[-1] * 0.999
    forced_df.iloc[-1, forced_df.columns.get_loc("adx")] = 20.0

    pipeline = MockPipeline(forced_df)
    strat_cfg = {
        "timeframe": "1m",
        "signal_timeframe": "5m",
        "band_multiplier": 2.0,
        "adx_threshold": 30,
        "min_rr_ratio": 1.0,
    }
    strat = VWAPMeanReversion(strat_cfg)
    strat.set_market_data_pipeline(pipeline)

    signal = await strat.generate_signal("TEST/USDT")
    if signal and signal.get("side") == "buy":
        print("✅ SIGNAL GENERATED: LONG")
    else:
        print("❌ NO SIGNAL")


if __name__ == "__main__":
    asyncio.run(run_scenario())
