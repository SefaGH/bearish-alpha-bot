#!/usr/bin/env python3
"""Utility for building RL-ready datasets from raw OHLCV candles.

Steps performed:
1. Fetch OHLCV candles (or load a supplied CSV/Parquet file).
2. Run the FeatureEngineeringPipeline to generate the 82-feature matrix expected by RL agents.
3. Clean/align price + feature frames, then emit train/val/test splits under ``data/training``.
4. Produce a metadata JSON file describing the export.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.live_trading_config import LiveTradingConfiguration  # noqa: E402
from src.core.ccxt_client import CcxtClient  # noqa: E402
from src.ml.feature_engineering import FeatureEngineeringPipeline  # noqa: E402

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DatasetSplits:
    train: slice
    val: slice
    test: slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RL training dataset with engineered features.")
    parser.add_argument("--exchange", default="bingx", help="Exchange id for ccxt (default: bingx)")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading symbol (CCXT format)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe, e.g. 1h, 4h")
    parser.add_argument("--candles", type=int, default=6000, help="Number of candles to fetch (unlimited batches)")
    parser.add_argument("--output-dir", default="data/training", help="Directory for generated npz files")
    parser.add_argument("--config", default="config/config.example.yaml", help="Config file used for manifests & hyperparams")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of samples for training split")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of samples for validation split")
    parser.add_argument("--min-rows", type=int, default=4000, help="Minimum rows required after cleaning")
    parser.add_argument("--min-split", type=int, default=512, help="Minimum rows required per split")
    parser.add_argument("--input-file", help="Optional CSV/Parquet with OHLCV columns (timestamp,open,high,low,close,volume)")
    parser.add_argument("--start-date", type=str, help="Optional start date (e.g. 2020-01-01) to filter OHLCV by timestamp")
    parser.add_argument("--end-date", type=str, help="Optional end date (e.g. 2024-01-01) to filter OHLCV by timestamp")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING,...)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").replace("-", "-")


def _apply_date_filter(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Filter DataFrame by start/end date if provided."""
    if start_date:
        start_ts = pd.to_datetime(start_date)
        df = df.loc[df.index >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        df = df.loc[df.index <= end_ts]
    return df


def load_price_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_file:
        path = Path(args.input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        logging.info("Loading OHLCV data from %s", path)
        if path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported input file type. Use CSV or Parquet.")
    else:
        df = fetch_from_exchange(args)

    if "timestamp" not in df.columns:
        raise ValueError("Input data must include a 'timestamp' column (ms or ISO format).")

    timestamp = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
    df = df.assign(timestamp=timestamp)
    df = df.sort_values("timestamp").set_index("timestamp")

    # Yeni: tarih filtresi uygula
    original_len = len(df)
    df = _apply_date_filter(df, args.start_date, args.end_date)
    if len(df) != original_len:
        logging.info(
            "Applied date filter: start_date=%s, end_date=%s → rows %d → %d",
            args.start_date,
            args.end_date,
            original_len,
            len(df),
        )

    missing = [col for col in PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    return df[PRICE_COLUMNS]


def fetch_from_exchange(args: argparse.Namespace) -> pd.DataFrame:
    logging.info("Fetching %s candles for %s on %s (%s)", args.candles, args.symbol, args.exchange, args.timeframe)
    client = CcxtClient(args.exchange)
    candles = client.fetch_ohlcv_bulk(args.symbol, args.timeframe, target_limit=args.candles)
    if not candles:
        raise RuntimeError("Exchange did not return OHLCV data")

    df = pd.DataFrame(candles, columns=["timestamp", *PRICE_COLUMNS])
    return df


def build_feature_frame(price_df: pd.DataFrame, config_path: str) -> Tuple[pd.DataFrame, Dict[str, any]]:
    config = LiveTradingConfiguration.load(config_path=config_path, log_summary=False)
    feature_engine = FeatureEngineeringPipeline(config=config)
    features = feature_engine.extract_features(price_df.copy(), mode="price")
    if features.empty:
        raise RuntimeError("FeatureEngineeringPipeline returned an empty DataFrame. Check indicators and manifests.")

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(method="ffill").fillna(method="bfill")
    features = features.dropna()

    aligned_index = features.index.intersection(price_df.index)
    if aligned_index.empty:
        raise RuntimeError("Feature frame and price frame have no overlapping timestamps")

    features = features.loc[aligned_index]
    prices = price_df.loc[aligned_index]

    if len(features) < 2:
        raise RuntimeError("Not enough aligned rows after cleaning")

    logging.info("Aligned dataset rows: %d", len(features))
    return features.reset_index(drop=True), {
        "timestamps": aligned_index.astype("datetime64[ns]").tolist(),
        "price_frame": prices.reset_index(drop=True),
        "config": config,
    }


def ensure_ratios(train_ratio: float, val_ratio: float) -> None:
    total = train_ratio + val_ratio
    if total >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError("train_ratio must be > 0 and val_ratio must be >= 0")


def compute_splits(n_rows: int, train_ratio: float, val_ratio: float, min_split: int) -> DatasetSplits:
    ensure_ratios(train_ratio, val_ratio)
    train_end = max(min_split, int(n_rows * train_ratio))
    val_end = train_end + max(int(n_rows * val_ratio), min_split if val_ratio > 0 else 0)
    val_end = min(val_end, n_rows)
    if train_end >= n_rows:
        raise ValueError("Training split exhausted entire dataset; lower train_ratio or fetch more data")
    if val_end > n_rows:
        val_end = n_rows
    if n_rows - val_end < min_split:
        raise ValueError("Test split would be too small; fetch more data or adjust ratios")
    return DatasetSplits(slice(0, train_end), slice(train_end, val_end), slice(val_end, n_rows))


def save_npz(path: Path, features: np.ndarray, prices: np.ndarray, feature_columns, price_columns, timestamps, symbol, timeframe, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        prices=prices.astype(np.float32),
        feature_columns=np.array(feature_columns),
        price_columns=np.array(price_columns),
        timestamps=np.array(timestamps, dtype="datetime64[ns]"),
        symbol=np.array(symbol),
        timeframe=np.array(timeframe),
    )
    logging.info("Saved dataset → %s (%s rows)", path, len(features))


def write_metadata(path: Path, meta: Dict[str, any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logging.info("Wrote metadata → %s", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_df = load_price_data(args)
    if len(price_df) < args.min_rows:
        raise RuntimeError(f"Need at least {args.min_rows} rows before feature engineering (got {len(price_df)})")

    feature_df, extras = build_feature_frame(price_df, args.config)
    timestamps = extras["timestamps"]
    price_frame = extras["price_frame"]

    n_rows = len(feature_df)
    if n_rows < args.min_rows:
        raise RuntimeError(f"Only {n_rows} usable rows after cleaning; fetch more data or loosen min_rows")

    splits = compute_splits(n_rows, args.train_ratio, args.val_ratio, args.min_split)
    base = f"{sanitize_symbol(args.symbol)}_{args.timeframe}"

    features_np = feature_df.to_numpy(dtype=np.float32)
    price_np = price_frame[PRICE_COLUMNS].to_numpy(dtype=np.float32)

    split_map = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }

    summary = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "exchange": args.exchange,
        "rows": n_rows,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "splits": {},
        "start_date": args.start_date,
        "end_date": args.end_date,
    }

    for name, slc in split_map.items():
        subset_features = features_np[slc]
        subset_prices = price_np[slc]
        subset_timestamps = timestamps[slc]
        target_path = output_dir / f"{base}_{name}.npz"
        save_npz(
            target_path,
            subset_features,
            subset_prices,
            feature_df.columns.tolist(),
            PRICE_COLUMNS,
            subset_timestamps,
            args.symbol,
            args.timeframe,
            args.overwrite,
        )
        summary["splits"][name] = {
            "rows": int(len(subset_features)),
            "path": str(target_path),
            "start": str(subset_timestamps[0]) if len(subset_timestamps) else None,
            "end": str(subset_timestamps[-1]) if len(subset_timestamps) else None,
        }

    metadata_path = output_dir / f"{base}_metadata.json"
    write_metadata(metadata_path, summary)
    logging.info("Dataset preparation complete. Train/Val/Test rows: %s", {k: v["rows"] for k, v in summary["splits"].items()})


if __name__ == "__main__":
    main()
