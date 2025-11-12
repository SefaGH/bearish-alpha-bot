#!/usr/bin/env python3
"""
Modified training script for Phase 2.1 validation.
This version uses local CSV data instead of live exchange data for validation purposes.
"""

import asyncio
import pandas as pd
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

async def main():
    """Main training function using local data."""
    from scripts.train_all_models import (
        logger, train_gemma_model, ModelPerformanceTracker
    )
    from src.config.live_trading_config import LiveTradingConfiguration
    from src.ml.feature_engineering import FeatureEngineeringPipeline
    
    # Load configuration
    config = LiveTradingConfiguration.load()
    
    # Initialize feature engine and tracker
    feature_engine = FeatureEngineeringPipeline()
    tracker = ModelPerformanceTracker()
    
    logger.info("="*60)
    logger.info("Phase 2.1 Validation: Training with local CSV data")
    logger.info("="*60)
    
    # Load the synthetic training data we created
    csv_path = "data/temp_training_data.csv"
    logger.info(f"Loading training data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    logger.info(f"✅ Loaded {len(df)} rows of training data")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data shape: {df.shape}")
    
    # Prepare training_data dictionary structure as expected by train_gemma_model
    training_data = {
        'BTC/USDT': {
            '1d': df  # Use our 2000-row synthetic data as 1d timeframe data
        }
    }
    
    logger.info("\n" + "="*60)
    logger.info("💎 STARTING GEMMA MODEL TRAINING 💎")
    logger.info("="*60)
    
    # Train GEMMA model
    gemma_results = train_gemma_model(
        training_data=training_data,
        feature_engine=feature_engine,
        config=config.get('ml', {}),  # Pass ML config section, not entire config
        tracker=tracker
    )
    
    logger.info("\n" + "="*60)
    logger.info("GEMMA TRAINING RESULTS")
    logger.info("="*60)
    logger.info(f"Status: {gemma_results.get('status', 'unknown')}")
    
    if gemma_results.get('status') == 'completed':
        logger.info("✅ GEMMA training completed successfully!")
        logger.info(f"Training samples: {gemma_results.get('train_samples', 'N/A')}")
        logger.info(f"Validation samples: {gemma_results.get('val_samples', 'N/A')}")
        logger.info(f"Final validation accuracy: {gemma_results.get('best_accuracy', 'N/A')}")
        logger.info(f"Training time: {gemma_results.get('training_time', 'N/A'):.2f}s")
        
        # Verify artifacts were created
        import os.path
        model_path = 'data/models/gemma/final/gemma_price.pt'
        scaler_path = 'data/cache/gemma/scaler_gemma.joblib'
        
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            logger.info(f"✅ Model artifact created: {model_path} ({size} bytes)")
        else:
            logger.error(f"❌ Model artifact NOT found: {model_path}")
            
        if os.path.exists(scaler_path):
            size = os.path.getsize(scaler_path)
            logger.info(f"✅ Scaler artifact created: {scaler_path} ({size} bytes)")
        else:
            logger.error(f"❌ Scaler artifact NOT found: {scaler_path}")
    else:
        logger.error(f"❌ GEMMA training failed: {gemma_results.get('error', 'unknown error')}")
        logger.error(f"Reason: {gemma_results.get('reason', 'N/A')}")
    
    logger.info("="*60)
    logger.info("PHASE 2.1 VALIDATION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())
