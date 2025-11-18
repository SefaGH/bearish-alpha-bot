"""Shared helpers for RL dataset loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_npz_dataset(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Load engineered features and prices from an npz archive."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        features = data["features"]
        prices = data["prices"]
        feature_columns = data["feature_columns"].tolist()
        price_columns = data["price_columns"].tolist()
        timestamps = data["timestamps"].astype("datetime64[ns]")
        symbol = data["symbol"].item() if "symbol" in data.files else "unknown"
        timeframe = data["timeframe"].item() if "timeframe" in data.files else "unknown"

    price_df = pd.DataFrame(prices, columns=price_columns)
    price_df["timestamp"] = pd.to_datetime(timestamps)
    price_df = price_df.set_index("timestamp")

    missing_price_cols = [col for col in PRICE_COLUMNS if col not in price_df.columns]
    if missing_price_cols:
        raise ValueError(f"Dataset is missing required price columns: {missing_price_cols}")

    features_df = pd.DataFrame(features, columns=feature_columns)
    features_df.index = price_df.index

    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
    price_df = price_df.loc[features_df.index]

    if len(features_df) < 500:
        raise RuntimeError("Dataset must contain at least 500 aligned rows for meaningful RL runs")

    return (
        features_df.reset_index(drop=True),
        price_df.reset_index(drop=True),
        {"symbol": symbol, "timeframe": timeframe},
    )
