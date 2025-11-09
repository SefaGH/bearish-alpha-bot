"""
Prepare and cache real training data for hyperparameter tuning.
Uses existing CcxtClient and FeatureEngineeringPipeline.

Usage:
    python scripts/prepare_training_data.py --symbol BTC/USDT

Author: SefaGH
Date: 2025-11-08
"""

import asyncio
import argparse
import sys
import os
import numpy as np
from pathlib import Path
import logging

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ CORRECT IMPORTS (matching train_all_models.py)
from src.core.ccxt_client import CcxtClient  # ✅ CcxtClient, not CCXTClient
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.label_generator import generate_regime_labels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration (matching train_all_models.py)
CANDLE_LIMIT = 1440
REGIME_TRAINING_TIMEFRAMES = ['15m', '30m', '1h', '4h', '1d']


async def fetch_and_process_data(symbol='BTC/USDT', 
                                 timeframes=None,
                                 use_feature_selection=True,
                                 use_all_features=False):
    """
    Fetch real market data and prepare for training.
    
    Args:
        symbol: Trading symbol
        timeframes: List of timeframes
        use_feature_selection: Apply feature selection mask
        use_all_features: Use all 87 features instead of legacy 42
    """
    if timeframes is None:
        timeframes = REGIME_TRAINING_TIMEFRAMES
    
    logger.info("="*70)
    logger.info(f"📊 FETCHING REAL MARKET DATA: {symbol}")
    logger.info("="*70)
    
    # 1. Initialize exchange client (same as train_all_models.py)
    logger.info("Initializing BingX exchange client...")
    exchange_client = CcxtClient('bingx')
    
    # 2. Initialize feature pipeline
    logger.info("Initializing feature engineering pipeline...")
    feature_engine = FeatureEngineeringPipeline()
    
    # 3. Fetch data for each timeframe
    all_features = []
    all_labels = []
    
    for tf in timeframes:
        logger.info(f"\n--- Processing {tf} data ---")
        
        try:
            # Fetch OHLCV data (async)
            logger.info(f"  Fetching {CANDLE_LIMIT} candles...")
            ohlcv_df = await exchange_client.ohlcv(
                symbol,
                timeframe=tf,
                limit=CANDLE_LIMIT,
                add_indicators=False  # Raw data only
            )
            
            if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 200:
                logger.warning(f"  ⚠️ Insufficient data ({len(ohlcv_df)} candles), skipping")
                continue
            
            logger.info(f"  ✅ Fetched {len(ohlcv_df)} candles")
            
            # Extract features
            logger.info(f"  Extracting features...")
            features_df = feature_engine.extract_features(ohlcv_df)
            logger.info(f"  ✅ Extracted {features_df.shape[1]} features")
            
            # Generate labels
            logger.info("  Generating regime labels...")
            regime_labels = generate_regime_labels(
                candles,
                window=20,                    # Lookback period
                threshold=0.015,              # 1.5% threshold (was 0.01)
                prediction_horizon=5,         # Predict 5 periods ahead
                volume_confirm=True,          # Require volume confirmation
                multi_timeframe=True          # Use multi-timeframe consensus
            )
            logger.info(f"  ✅ Generated {len(regime_labels)} labels")
            
            # Prepare for training with flag
            X_prepared, y_prepared = feature_engine.prepare_for_training(
                features_df,
                regime_labels,
                use_all_features=use_all_features  # ← Pass the flag
            )
            
            if X_prepared.shape[0] > 0:
                all_features.append(X_prepared)
                all_labels.append(y_prepared)
                logger.info(f"  ✅ Added {X_prepared.shape[0]} samples")
            else:
                logger.warning(f"  ⚠️ No samples after preparation")
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {tf}: {e}", exc_info=True)
            continue
    
    # 4. Combine all timeframes
    if not all_features:
        raise RuntimeError("No data fetched successfully from any timeframe")
    
    logger.info("\n" + "="*70)
    logger.info("📊 COMBINING DATA FROM ALL TIMEFRAMES")
    logger.info("="*70)
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    logger.info(f"✅ Total samples: {len(X)}")
    logger.info(f"✅ Original features: {X.shape[1]}")
    
    # Apply feature selection if enabled
    if use_feature_selection:
        feature_mask_path = Path('data/cache/feature_selection_mask.npy')
        
        if feature_mask_path.exists():
            try:
                feature_mask = np.load(feature_mask_path)
                
                # Validate mask shape
                if len(feature_mask) != X.shape[1]:
                    logger.warning(
                        f"⚠️ Feature mask size mismatch! "
                        f"Mask: {len(feature_mask)}, Features: {X.shape[1]}. "
                        f"Skipping feature selection."
                    )
                else:
                    # Apply mask
                    original_count = X.shape[1]
                    X = X[:, feature_mask]
                    removed_count = (~feature_mask).sum()
                    
                    logger.info(
                        f"✅ Selected {X.shape[1]} features (removed {removed_count})"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to load feature mask: {e}. Continuing without selection.")
        else:
            logger.warning(
                f"⚠️ Feature selection mask not found at {feature_mask_path}. "
                f"Continuing with all features."
            )
    else:
        logger.info("⚠️ Feature selection skipped (disabled via --no-feature-selection)")
    
    logger.info(f"✅ Final features: {X.shape[1]}")
    logger.info(f"✅ Label distribution:")
    
    # Show label distribution
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        label_name = ['Bullish', 'Bearish', 'Neutral', 'Volatile'][label]
        logger.info(f"     {label_name}: {count} ({percentage:.1f}%)")
    
    # 5. Save to cache
    cache_dir = Path('data/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Use same naming convention
    cache_file = cache_dir / f'{symbol.replace("/", "-")}_training_data.npz'
    
    logger.info(f"\n💾 Saving to cache: {cache_file}")
    np.savez_compressed(cache_file, X=X, y=y)
    
    logger.info("="*70)
    logger.info("✅ TRAINING DATA READY")
    logger.info("="*70)
    
    return X, y


async def async_main():
    """Async entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare real training data for hyperparameter tuning'
    )
    parser.add_argument(
        '--symbol',
        default='BTC/USDT',
        help='Trading symbol (default: BTC/USDT)'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=REGIME_TRAINING_TIMEFRAMES,
        help='Timeframes to fetch (default: 15m 30m 1h 4h 1d)'
    )
    # CLI argument
    parser.add_argument(
        '--use-all-features',
        action='store_true',
        help='Use all 87 extracted features instead of legacy 42 (TEST MODE)'
    )
    parser.add_argument(
        '--no-feature-selection',
        action='store_true',
        help='Disable automatic feature selection (default: enabled)'
    )
    
    # Parse and use
    args = parser.parse_args()
    X, y = await fetch_and_process_data(
        args.symbol,
        args.timeframes,
        use_feature_selection=not args.no_feature_selection,
        use_all_features=args.use_all_features  # ← Use the flag
    )
    
    logger.info(f"\n✅ COMPLETE: {len(X)} samples ready for tuning")


def main():
    """Synchronous entry point."""
    asyncio.run(async_main())


if __name__ == '__main__':
    main()
