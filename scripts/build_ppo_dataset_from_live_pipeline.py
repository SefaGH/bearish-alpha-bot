#!/usr/bin/env python3
"""
Build a PPO-ready dataset using the same live feature pipeline and ObservationSpec.

Goals:
- Recompute features from raw OHLCV (or an existing NPZ price snapshot) using FeatureEngineeringPipeline.
- Enforce the ObservationSpec (feature names + order) and fail loudly on missing columns.
- Emit train/val/test NPZ splits plus spec/metadata sidecars for training & evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.live_trading_config import LiveTradingConfiguration  # noqa: E402
from src.ml.feature_engineering import FeatureEngineeringPipeline  # noqa: E402
from src.ml.ppo.observation_spec import (  # noqa: E402
    DEFAULT_EXTRA_FEATURE_NAMES,
    ObservationSpec,
    compute_price_extras,
    load_spec,
    save_spec,
    spec_from_feature_columns,
)
from src.ml.ppo.deterministic_scaler import DeterministicScaler  # noqa: E402

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DatasetSplits:
    train: slice
    val: slice
    test: slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PPO dataset with live feature pipeline + spec validation.")
    src = parser.add_argument_group("data source")
    src.add_argument("--source-npz", type=Path, help="Existing NPZ with prices (will recompute features).")
    src.add_argument("--input-file", type=Path, help="CSV/Parquet with columns timestamp,open,high,low,close,volume.")
    src.add_argument("--exchange", default="bingx", help="CCXT exchange id (fallback if no file/npz).")
    src.add_argument("--symbol", default="BTC/USDT:USDT", help="Symbol in CCXT format.")
    src.add_argument("--timeframe", default="1h", help="Candle timeframe.")
    src.add_argument("--candles", type=int, default=6000, help="Number of candles to fetch when using CCXT.")
    src.add_argument("--config", default="config/config.example.yaml", help="Config path for FeatureEngineeringPipeline.")

    out = parser.add_argument_group("output/validation")
    out.add_argument("--output-dir", type=Path, default=Path("data/training"), help="Directory for generated NPZ files.")
    out.add_argument(
        "--base-name",
        type=str,
        help="Base filename (default: <symbol>_<timeframe>_liveparity, sanitized).",
    )
    out.add_argument("--spec", type=Path, help="Optional ObservationSpec to enforce. If omitted, build from features.")
    out.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    out.add_argument("--min-rows", type=int, default=4000, help="Minimum usable rows after alignment.")
    out.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    out.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    out.add_argument("--min-split", type=int, default=512, help="Minimum rows per split.")
    out.add_argument("--validate-dataset", type=Path, help="Validate an existing dataset against --spec and exit.")
    out.add_argument("--emit-scaled", action="store_true", help="Also emit *_scaled.npz splits with DeterministicScaler applied (debug-only).")
    out.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").replace("-", "_")


def compute_splits(n_rows: int, train_ratio: float, val_ratio: float, min_split: int) -> DatasetSplits:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio must be >0, val_ratio >=0, and train_ratio+val_ratio < 1.0")
    train_end = max(min_split, int(n_rows * train_ratio))
    val_end = train_end + max(int(n_rows * val_ratio), min_split if val_ratio > 0 else 0)
    val_end = min(val_end, n_rows)
    if train_end >= n_rows:
        raise ValueError("Training split exhausted dataset; reduce ratios or fetch more data.")
    if n_rows - val_end < min_split:
        raise ValueError("Test split too small; fetch more data or adjust ratios.")
    return DatasetSplits(slice(0, train_end), slice(train_end, val_end), slice(val_end, n_rows))


def _load_price_from_npz(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    with np.load(path, allow_pickle=True) as data:
        prices = data["prices"]
        price_columns = data["price_columns"].tolist()
        timestamps = pd.to_datetime(data["timestamps"])
        symbol = data["symbol"].item() if "symbol" in data.files else "unknown"
        timeframe = data["timeframe"].item() if "timeframe" in data.files else "unknown"
    price_df = pd.DataFrame(prices, columns=price_columns)
    price_df["timestamp"] = timestamps
    price_df = price_df.set_index("timestamp")
    missing = [col for col in PRICE_COLUMNS if col not in price_df.columns]
    if missing:
        raise ValueError(f"NPZ missing OHLCV columns: {missing}")
    return price_df[PRICE_COLUMNS], {"symbol": symbol, "timeframe": timeframe}


def _load_price_from_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file type for --input-file (use CSV or Parquet)")
    if "timestamp" not in df.columns:
        raise ValueError("Input data must include a 'timestamp' column.")
    ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.assign(timestamp=ts).sort_values("timestamp").set_index("timestamp")
    missing = [col for col in PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input file missing OHLCV columns: {missing}")
    return df[PRICE_COLUMNS]


def _fetch_from_exchange(args: argparse.Namespace) -> pd.DataFrame:
    from src.core.ccxt_client import CcxtClient  # lazy import

    client = CcxtClient(args.exchange)
    candles = client.fetch_ohlcv_bulk(args.symbol, args.timeframe, target_limit=args.candles)
    if not candles:
        raise RuntimeError("Exchange returned no OHLCV data")
    df = pd.DataFrame(candles, columns=["timestamp", *PRICE_COLUMNS])
    ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    return df.assign(timestamp=ts).sort_values("timestamp").set_index("timestamp")[PRICE_COLUMNS]


def load_price_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if args.source_npz:
        price_df, meta = _load_price_from_npz(args.source_npz)
        return price_df, meta
    if args.input_file:
        price_df = _load_price_from_file(args.input_file)
        return price_df, {"symbol": args.symbol, "timeframe": args.timeframe}
    price_df = _fetch_from_exchange(args)
    return price_df, {"symbol": args.symbol, "timeframe": args.timeframe}


def build_features(price_df: pd.DataFrame, config_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = LiveTradingConfiguration.load(config_path=config_path, log_summary=False)
    pipe = FeatureEngineeringPipeline(config=cfg)
    feats = pipe.extract_features(price_df.copy(), mode="price")
    if feats is None or feats.empty:
        raise RuntimeError("FeatureEngineeringPipeline returned empty features")
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    aligned_index = feats.index.intersection(price_df.index)
    if aligned_index.empty:
        raise RuntimeError("No overlapping timestamps between features and price data")
    feats = feats.loc[aligned_index]
    prices_aligned = price_df.loc[aligned_index]
    return feats, prices_aligned


def align_to_spec(features_df: pd.DataFrame, spec: ObservationSpec) -> Tuple[pd.DataFrame, Dict[str, Iterable[str]]]:
    missing = [c for c in spec.feature_names if c not in features_df.columns]
    extra = [c for c in features_df.columns if c not in spec.feature_names]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    aligned = features_df.reindex(columns=spec.feature_names)
    return aligned, {"missing": missing, "extra": extra}


def save_npz(
    path: Path,
    features: np.ndarray,
    prices: np.ndarray,
    feature_columns: Iterable[str],
    price_columns: Iterable[str],
    timestamps: Iterable[pd.Timestamp],
    symbol: str,
    timeframe: str,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        prices=prices.astype(np.float32),
        feature_columns=np.array(list(feature_columns)),
        price_columns=np.array(list(price_columns)),
        timestamps=np.array(list(timestamps), dtype="datetime64[ns]"),
        symbol=np.array(symbol),
        timeframe=np.array(timeframe),
    )
    logging.info("Saved dataset %s (%s rows)", path, len(features))


def validate_dataset(dataset_path: Path, spec_path: Path) -> Dict[str, any]:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)
    spec = load_spec(spec_path)
    with np.load(dataset_path, allow_pickle=True) as data:
        features = pd.DataFrame(data["features"], columns=data["feature_columns"].tolist())
    missing = [c for c in spec.feature_names if c not in features.columns]
    extra = [c for c in features.columns if c not in spec.feature_names]
    ok = not missing
    if not ok:
        logging.error("Dataset %s missing required features: %s", dataset_path, missing)
    else:
        logging.info("Dataset %s matches spec feature set (%d)", dataset_path, len(spec.feature_names))
    if extra:
        logging.warning("Dataset has %d extra columns not in spec: %s", len(extra), extra[:5])
    return {"ok": ok, "missing": missing, "extra": extra, "spec_dim": len(spec.feature_names)}


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.validate_dataset:
        if not args.spec:
            raise SystemExit("--spec is required with --validate-dataset")
        validate_dataset(args.validate_dataset, args.spec)
        return

    price_df, meta = load_price_data(args)
    if len(price_df) < args.min_rows:
        raise RuntimeError(f"Need at least {args.min_rows} price rows (got {len(price_df)})")

    price_df_full = price_df.copy()
    features_df, price_df = build_features(price_df, args.config)
    if len(features_df) < args.min_rows:
        raise RuntimeError(f"Only {len(features_df)} usable rows after feature engineering; fetch more data")

    if args.spec and args.spec.exists():
        spec = load_spec(args.spec)
    else:
        spec = spec_from_feature_columns(features_df.columns, extra_feature_names=DEFAULT_EXTRA_FEATURE_NAMES)

    aligned_features, diff = align_to_spec(features_df, spec)
    if diff["extra"]:
        logging.warning("Dropping %d extra feature columns not in spec: %s", len(diff["extra"]), diff["extra"][:5])
        aligned_features = aligned_features  # explicit for clarity

    splits = compute_splits(len(aligned_features), args.train_ratio, args.val_ratio, args.min_split)
    base_name = args.base_name or f"{sanitize_symbol(meta.get('symbol', args.symbol))}_{meta.get('timeframe', args.timeframe)}_liveparity"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_path = output_dir / f"{base_name}.obs_spec.json"
    save_spec(spec, spec_path)
    logging.info("Saved ObservationSpec to %s (obs_dim=%d)", spec_path, spec.obs_dim)

    scaler = None
    try:
        scaler = DeterministicScaler(spec_path)
        sample_row = aligned_features.iloc[-1].to_dict()
        extra_arr = compute_price_extras(price_df)
        extra_values = {
            name: float(extra_arr[i]) for i, name in enumerate(spec.extra_feature_names or [])
        }
        sample_row.update(extra_values)
        sample_row.update({"position_fraction": 0.0, "normalized_pv": 1.0})
        close_price = float(price_df["close"].iloc[-1])
        scaled_vec = scaler.transform(sample_row, close_price)
        logging.info("DeterministicScaler sanity check ok (dim=%d)", scaled_vec.shape[0])
    except Exception as exc:
        logging.error("DeterministicScaler sanity check failed: %s", exc)

    timestamps = price_df.index
    features_np = aligned_features.to_numpy(dtype=np.float32)
    prices_np = price_df[PRICE_COLUMNS].to_numpy(dtype=np.float32)

    for split_name, split in {"train": splits.train, "val": splits.val, "test": splits.test}.items():
        target = output_dir / f"{base_name}_{split_name}.npz"
        save_npz(
            target,
            features_np[split],
            prices_np[split],
            aligned_features.columns,
            PRICE_COLUMNS,
            timestamps[split],
            meta.get("symbol", args.symbol),
            meta.get("timeframe", args.timeframe),
            args.overwrite,
        )

        if args.emit_scaled:
            if scaler is None:
                raise RuntimeError("emit-scaled requested but scaler init failed.")
            scaled_rows: List[np.ndarray] = []
            ts_index = timestamps[split]
            for ts in ts_index:
                window = price_df_full.loc[:ts].tail(256)
                extra_arr = compute_price_extras(window)
                try:
                    if getattr(extra_arr, "ndim", 1) > 1:
                        extra_arr = extra_arr[-1]
                except Exception:
                    pass
                extra_values = {name: float(extra_arr[i]) for i, name in enumerate(spec.extra_feature_names or [])}
                row = aligned_features.loc[ts].to_dict()
                row.update(extra_values)
                row.update({"position_fraction": 0.0, "normalized_pv": 1.0})
                close_price = float(window["close"].iloc[-1])
                scaled_rows.append(scaler.transform(row, close_price).astype(np.float32))
            target_scaled = output_dir / f"{base_name}_{split_name}_scaled.npz"
            save_npz(
                target_scaled,
                np.vstack(scaled_rows),
                prices_np[split],
                list(scaler.feature_names),
                PRICE_COLUMNS,
                ts_index,
                meta.get("symbol", args.symbol),
                meta.get("timeframe", args.timeframe),
                args.overwrite,
            )

    metadata = {
        "symbol": meta.get("symbol", args.symbol),
        "timeframe": meta.get("timeframe", args.timeframe),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(len(aligned_features)),
        "spec_path": str(spec_path),
        "spec_dim": spec.obs_dim,
        "feature_columns": list(aligned_features.columns),
        "extra_columns_dropped": list(diff["extra"]),
        "splits": {
            "train": {"rows": len(aligned_features[splits.train]), "path": str(output_dir / f"{base_name}_train.npz")},
            "val": {"rows": len(aligned_features[splits.val]), "path": str(output_dir / f"{base_name}_val.npz")},
            "test": {"rows": len(aligned_features[splits.test]), "path": str(output_dir / f"{base_name}_test.npz")},
        },
    }
    meta_path = output_dir / f"{base_name}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logging.info("Wrote metadata to %s", meta_path)
    logging.info("Dataset build complete.")


if __name__ == "__main__":
    main()
