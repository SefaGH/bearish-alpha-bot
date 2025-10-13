#!/usr/bin/env python3
"""
Phase 4.1: ML Market Regime Prediction - Usage Examples

Demonstrates ML-powered predictive market regime detection including:
- Feature engineering from market data
- Model training and validation
- Real-time regime prediction
- Integration with Phase 2 regime detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime

from ml.regime_predictor import MLRegimePredictor
from ml.model_trainer import RegimeModelTrainer
from ml.prediction_engine import RealTimePredictionEngine
from ml.feature_engineering import FeatureEngineeringPipeline
from config.ml_config import MLConfiguration
from core.market_regime import MarketRegimeAnalyzer


def create_sample_data(n_bars=500, trend='mixed'):
    """Create sample OHLCV data with indicators for demonstration."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    
    # Create price series with trend
    if trend == 'bullish':
        base_trend = np.linspace(0, 10, n_bars)
    elif trend == 'bearish':
        base_trend = np.linspace(10, 0, n_bars)
    else:  # mixed
        base_trend = np.sin(np.linspace(0, 4*np.pi, n_bars)) * 5
    
    close = 100 + base_trend + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_price = close + np.random.randn(n_bars) * 0.2
    volume = np.abs(np.random.randn(n_bars) * 1000 + 5000)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'rsi': 50 + np.random.randn(n_bars) * 15,
        'macd': np.random.randn(n_bars) * 0.5,
        'macd_signal': np.random.randn(n_bars) * 0.5,
    }, index=dates)
    
    # Add EMA and Bollinger Bands
    df['ema_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ema_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['bb_upper'] = df['close'] + 2
    df['bb_lower'] = df['close'] - 2
    df['atr'] = np.abs(np.random.randn(n_bars) * 0.5 + 1)
    
    return df


def create_historical_regime_labels(price_data):
    """Create historical regime labels based on price trends."""
    returns = price_data['close'].pct_change(20)
    
    labels = []
    for ret in returns:
        if pd.isna(ret):
            labels.append(1)  # neutral
        elif ret > 0.03:
            labels.append(0)  # bullish
        elif ret < -0.03:
            labels.append(2)  # bearish
        else:
            labels.append(1)  # neutral
    
    return pd.Series(labels, index=price_data.index)


def example_1_feature_engineering():
    """Example 1: Feature Engineering Pipeline."""
    print("=" * 70)
    print("Example 1: Feature Engineering for ML Models")
    print("=" * 70)
    
    # Create sample data
    price_data = create_sample_data(n_bars=200)
    
    # Initialize feature engineering pipeline
    feature_engine = FeatureEngineeringPipeline()
    
    # Extract features
    print("\nExtracting features from price data...")
    features = feature_engine.extract_features(price_data)
    
    print(f"\n📊 Feature Extraction Results:")
    print(f"   Total Features Extracted:   {len(features.columns)}")
    print(f"   Number of Samples:          {len(features)}")
    print(f"   Feature Completeness:       {(1 - features.isna().sum().sum() / (features.shape[0] * features.shape[1])) * 100:.1f}%")
    
    print("\n✓ Feature engineering completed successfully")
    print()


def example_2_model_training():
    """Example 2: Training ML Regime Prediction Models."""
    print("=" * 70)
    print("Example 2: Training ML Regime Prediction Models")
    print("=" * 70)
    
    # Create training data
    price_data = create_sample_data(n_bars=500)
    regime_labels = create_historical_regime_labels(price_data)
    
    # Initialize ML predictor
    predictor = MLRegimePredictor()
    
    # Train models
    print("\nTraining ensemble of ML models...")
    print("   - Random Forest Classifier")
    print("   - LSTM Network")
    print("   - Transformer Network")
    
    result = predictor.train_regime_models(price_data, regime_labels)
    
    if result['success']:
        print(f"\n✅ Training Successful!")
        print(f"   Training Samples:           {result['n_samples']}")
        print(f"   Number of Features:         {result['n_features']}")
        print(f"   Training Accuracy:          {result['train_accuracy']:.2%}")
        print(f"   Models Trained:             {', '.join(result['models_trained'])}")
    else:
        print(f"\n❌ Training Failed: {result.get('error', 'Unknown error')}")
    
    print("\n✓ Model training completed")
    print()
    
    return predictor


async def example_3_regime_prediction(predictor):
    """Example 3: Making Regime Predictions."""
    print("=" * 70)
    print("Example 3: ML-Based Regime Prediction")
    print("=" * 70)
    
    # Create test data
    price_data = create_sample_data(n_bars=200, trend='bullish')
    
    # Make prediction
    print("\nMaking regime prediction for BTC/USDT...")
    prediction = await predictor.predict_regime_transition('BTC/USDT', price_data, horizon='1h')
    
    print(f"\n🔮 Regime Prediction:")
    print(f"   Symbol:                     {prediction.get('symbol', 'N/A')}")
    print(f"   Predicted Regime:           {prediction['predicted_regime'].upper()}")
    print(f"   Prediction Horizon:         {prediction.get('horizon', 'N/A')}")
    print(f"   Confidence Score:           {prediction['confidence']:.2%}")
    print(f"   Quality Score:              {prediction['quality_score']:.2%}")
    
    print(f"\n📊 Regime Probabilities:")
    probs = prediction['probabilities']
    print(f"   Bullish:                    {probs['bullish']:.2%}")
    print(f"   Neutral:                    {probs['neutral']:.2%}")
    print(f"   Bearish:                    {probs['bearish']:.2%}")
    
    print("\n✓ Regime prediction completed")
    print()


async def example_4_real_time_prediction_engine(predictor):
    """Example 4: Real-Time Prediction Engine."""
    print("=" * 70)
    print("Example 4: Real-Time ML Prediction Engine")
    print("=" * 70)
    
    # Initialize prediction engine
    engine = RealTimePredictionEngine(trained_models=predictor.models)
    
    print("\nStarting real-time prediction engine...")
    await engine.start_prediction_engine(symbols=['BTC/USDT', 'ETH/USDT'])
    
    # Simulate market data updates
    print("\nSimulating market data updates...")
    for i in range(5):
        data = {
            'close': 100 + i * 0.5,
            'volume': 1000.0,
            'high': 101 + i * 0.5,
            'low': 99 + i * 0.5,
            'timestamp': pd.Timestamp.now()
        }
        await engine.on_market_data_update('BTC/USDT', data)
        print(f"   Update {i+1}/5: Price=${data['close']:.2f}")
    
    # Get engine status
    status = engine.get_engine_status()
    print(f"\n⚙️ Engine Status:")
    print(f"   Running:                    {status['running']}")
    print(f"   Symbols Tracked:            {', '.join(status['symbols_tracked'])}")
    print(f"   Predictions Cached:         {status['n_predictions_cached']}")
    print(f"   Update Interval:            {status['update_interval']}s")
    
    # Stop engine
    print("\nStopping prediction engine...")
    await engine.stop_prediction_engine()
    
    print("\n✓ Real-time prediction engine demonstration completed")
    print()


async def example_5_integrated_workflow():
    """Example 5: Complete Integrated ML Workflow."""
    print("=" * 70)
    print("Example 5: Integrated ML Market Regime Prediction Workflow")
    print("=" * 70)
    
    print("\n📋 Workflow Steps:")
    print("   1. Create historical market data")
    print("   2. Train ML regime prediction models")
    print("   3. Make predictions on new data")
    print("   4. Generate trading signals")
    print("   5. Compare with Phase 2 regime detection")
    
    # Step 1: Create data
    print("\n" + "=" * 70)
    print("Step 1: Creating Historical Market Data")
    print("=" * 70)
    historical_data = create_sample_data(n_bars=1000, trend='mixed')
    historical_labels = create_historical_regime_labels(historical_data)
    print(f"✓ Created {len(historical_data)} bars of historical data")
    
    # Step 2: Train models
    print("\n" + "=" * 70)
    print("Step 2: Training ML Models")
    print("=" * 70)
    ml_predictor = MLRegimePredictor()
    train_result = ml_predictor.train_regime_models(historical_data, historical_labels)
    print(f"✓ Training accuracy: {train_result['train_accuracy']:.2%}")
    
    # Step 3: Make predictions
    print("\n" + "=" * 70)
    print("Step 3: Making Regime Predictions")
    print("=" * 70)
    test_data = create_sample_data(n_bars=100, trend='bullish')
    ml_prediction = await ml_predictor.predict_regime_transition('BTC/USDT', test_data)
    print(f"✓ ML Prediction: {ml_prediction['predicted_regime']} (confidence: {ml_prediction['confidence']:.2%})")
    
    # Step 4: Generate signals
    print("\n" + "=" * 70)
    print("Step 4: Generating Trading Signals")
    print("=" * 70)
    engine = RealTimePredictionEngine(trained_models=ml_predictor.models)
    await engine.start_prediction_engine(symbols=['BTC/USDT'])
    
    # Add some data
    for i in range(60):
        await engine.on_market_data_update('BTC/USDT', {
            'close': 100 + i * 0.1,
            'volume': 1000.0,
            'high': 101,
            'low': 99,
            'timestamp': pd.Timestamp.now()
        })
    
    signal = engine.get_regime_transition_signals('BTC/USDT', threshold=0.6)
    print(f"✓ Trading Signal: {signal['signal']} (strength: {signal['strength']:.2f})")
    
    await engine.stop_prediction_engine()
    
    # Step 5: Compare with Phase 2
    print("\n" + "=" * 70)
    print("Step 5: Comparison with Phase 2 Regime Detection")
    print("=" * 70)
    
    regime_analyzer = MarketRegimeAnalyzer()
    # Create multi-timeframe data for Phase 2
    df_30m = test_data
    df_1h = test_data.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h = test_data.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    phase2_regime = regime_analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
    
    print(f"\n📊 Comparison Results:")
    print(f"   ML Prediction:              {ml_prediction['predicted_regime'].upper()}")
    print(f"   Phase 2 Detection:          {phase2_regime['trend'].upper()}")
    print(f"   ML Confidence:              {ml_prediction['confidence']:.2%}")
    print(f"   Phase 2 Risk Multiplier:    {phase2_regime['risk_multiplier']:.2f}x")
    
    print("\n✅ Complete integrated workflow finished successfully!")
    print()


async def main():
    """Run all examples."""
    print("\n" + "🚀" * 35)
    print("Phase 4.1: ML Market Regime Prediction - Examples")
    print("🚀" * 35 + "\n")
    
    # Example 1: Feature Engineering
    example_1_feature_engineering()
    
    # Example 2: Model Training
    predictor = example_2_model_training()
    
    # Example 3: Regime Prediction
    await example_3_regime_prediction(predictor)
    
    # Example 4: Real-Time Engine
    await example_4_real_time_prediction_engine(predictor)
    
    # Example 5: Integrated Workflow
    await example_5_integrated_workflow()
    
    print("=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    asyncio.run(main())
