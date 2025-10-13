#!/usr/bin/env python3
"""
Phase 4 Final: Advanced Price Prediction - Usage Examples

Demonstrates advanced LSTM and Transformer models for real-time price movement
prediction with multi-timeframe forecasting, ensemble predictions, confidence
intervals, and integration with existing trading strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime

from ml.price_predictor import (
    LSTMPricePredictor,
    TransformerPricePredictor,
    EnsemblePricePredictor,
    MultiTimeframePricePredictor,
    AdvancedPricePredictionEngine
)
from ml.strategy_integration import (
    AIEnhancedStrategyAdapter,
    StrategyPerformanceTracker,
    MLStrategyIntegrationManager
)
from ml.regime_predictor import MLRegimePredictor
from config.ml_config import MLConfiguration


def create_sample_data(n_bars=500, timeframe='5m', trend='upward'):
    """Create sample OHLCV data for demonstration."""
    np.random.seed(42)
    
    if timeframe == '5m':
        freq = '5min'
    elif timeframe == '15m':
        freq = '15min'
    elif timeframe == '1h':
        freq='1H'
    else:
        freq = '5min'
    
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq=freq)
    
    # Create realistic price data with trend
    if trend == 'upward':
        base_trend = np.linspace(40000, 45000, n_bars)
    elif trend == 'downward':
        base_trend = np.linspace(45000, 40000, n_bars)
    else:
        base_trend = np.ones(n_bars) * 42000
    
    # Add volatility
    volatility = np.random.randn(n_bars) * 200
    close = base_trend + volatility
    
    high = close + np.abs(np.random.randn(n_bars) * 100)
    low = close - np.abs(np.random.randn(n_bars) * 100)
    open_price = close + np.random.randn(n_bars) * 50
    volume = np.abs(np.random.randn(n_bars) * 100 + 1000)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


def example_1_lstm_transformer_models():
    """Example 1: LSTM and Transformer Price Prediction Models."""
    print("=" * 70)
    print("Example 1: LSTM and Transformer Price Prediction")
    print("=" * 70)
    
    # Initialize LSTM model
    print("\n📊 Initializing LSTM Price Predictor...")
    lstm_model = LSTMPricePredictor(
        input_size=50,
        hidden_size=128,
        num_layers=3,
        forecast_horizon=12
    )
    print(f"   LSTM Model initialized")
    print(f"   Forecast Horizon: {lstm_model.forecast_horizon} steps")
    
    # Initialize Transformer model
    print("\n📊 Initializing Transformer Price Predictor...")
    transformer_model = TransformerPricePredictor(
        d_model=256,
        nhead=8,
        num_layers=6,
        forecast_horizon=12
    )
    print(f"   Transformer Model initialized")
    print(f"   Forecast Horizon: {transformer_model.forecast_horizon} steps")
    
    # Make sample predictions
    print("\n🔮 Making Sample Predictions...")
    sample_input = np.random.randn(1, 100, 50)  # 1 sample, 100 timesteps, 50 features
    
    lstm_forecast, lstm_uncertainty = lstm_model.predict(sample_input)
    print(f"   LSTM Forecast: {lstm_forecast[0][:3]}... (first 3 steps)")
    print(f"   LSTM Uncertainty: {lstm_uncertainty[0][:3]}...")
    
    print("\n✅ LSTM and Transformer models demonstrated successfully!")
    print()


def example_2_ensemble_prediction():
    """Example 2: Ensemble Prediction Combining Multiple Models."""
    print("=" * 70)
    print("Example 2: Ensemble Price Prediction")
    print("=" * 70)
    
    # Create individual models
    print("\n🏗️ Building Ensemble...")
    models = {
        'lstm': LSTMPricePredictor(forecast_horizon=12),
        'transformer': TransformerPricePredictor(forecast_horizon=12)
    }
    
    # Create ensemble with custom weights
    weights = {
        'lstm': 0.5,
        'transformer': 0.5
    }
    
    ensemble = EnsemblePricePredictor(models, weights)
    print(f"   Ensemble: {len(ensemble.models)} models")
    print(f"   Weights: LSTM={weights['lstm']}, Transformer={weights['transformer']}")
    
    # Make ensemble prediction
    print("\n🔮 Making Ensemble Prediction...")
    sample_input = np.random.randn(1, 100, 50)
    
    forecast, uncertainty = ensemble.predict(sample_input)
    print(f"   Ensemble Forecast: {forecast[0][:5]}...")
    print(f"   Ensemble Uncertainty: {uncertainty[0][:5]}...")
    print(f"   Mean Forecast: {np.mean(forecast[0]):.2f}%")
    print(f"   Mean Uncertainty: {np.mean(uncertainty[0]):.2f}%")
    
    print("\n✅ Ensemble prediction completed successfully!")
    print()


def example_3_multi_timeframe_forecasting():
    """Example 3: Multi-Timeframe Forecasting."""
    print("=" * 70)
    print("Example 3: Multi-Timeframe Price Forecasting")
    print("=" * 70)
    
    # Create models for different timeframes
    print("\n🏗️ Creating Multi-Timeframe Models...")
    models = {}
    timeframes = ['5m', '15m', '1h']
    
    for tf in timeframes:
        tf_models = {
            'lstm': LSTMPricePredictor(forecast_horizon=12),
            'transformer': TransformerPricePredictor(forecast_horizon=12)
        }
        models[tf] = EnsemblePricePredictor(tf_models)
        print(f"   ✓ Created ensemble for {tf} timeframe")
    
    mt_predictor = MultiTimeframePricePredictor(models)
    
    # Create sample data for each timeframe
    print("\n📈 Generating Sample Data...")
    data_by_tf = {}
    for tf in timeframes:
        n_bars = 500 if tf == '5m' else (200 if tf == '15m' else 100)
        data_by_tf[tf] = create_sample_data(n_bars=n_bars, timeframe=tf, trend='upward')
        print(f"   ✓ {tf}: {len(data_by_tf[tf])} bars from {data_by_tf[tf].index[0]} to {data_by_tf[tf].index[-1]}")
    
    # Make multi-timeframe prediction
    print("\n🔮 Making Multi-Timeframe Predictions...")
    result = mt_predictor.predict_multi_timeframe(data_by_tf)
    
    # Display results by timeframe
    print("\n📊 Results by Timeframe:")
    for tf, pred in result['by_timeframe'].items():
        current_price = pred['current_price']
        forecast = pred['forecast']
        ci = pred['confidence_interval']
        
        print(f"\n   {tf} Timeframe:")
        print(f"      Current Price:     ${current_price:,.2f}")
        print(f"      Forecast (Step 1): ${current_price * (1 + forecast[0]/100):,.2f} ({forecast[0]:+.2f}%)")
        print(f"      Confidence Interval:")
        print(f"         Lower:  ${ci['lower'][0]:,.2f}")
        print(f"         Upper:  ${ci['upper'][0]:,.2f}")
        print(f"      Uncertainty:       {pred['uncertainty'][0]:.2f}%")
    
    # Display aggregated forecast
    print("\n📊 Aggregated Multi-Timeframe Forecast:")
    agg = result['aggregated']
    base_price = data_by_tf['5m']['close'].iloc[-1]
    print(f"   Base Price:          ${base_price:,.2f}")
    print(f"   Forecast Change:     {agg['forecast'][0]:+.2f}%")
    print(f"   Forecast Price:      ${base_price * (1 + agg['forecast'][0]/100):,.2f}")
    print(f"   Uncertainty:         {agg['uncertainty'][0]:.2f}%")
    print(f"   Consensus Strength:  {agg['consensus_strength']:.2%}")
    
    print("\n✅ Multi-timeframe forecasting completed successfully!")
    print()


async def example_4_real_time_prediction_engine():
    """Example 4: Real-Time Price Prediction Engine."""
    print("=" * 70)
    print("Example 4: Real-Time Price Prediction Engine")
    print("=" * 70)
    
    # Create multi-timeframe predictor
    print("\n🏗️ Setting up Real-Time Engine...")
    models = {}
    for tf in ['5m', '15m', '1h']:
        tf_models = {
            'lstm': LSTMPricePredictor(forecast_horizon=12),
            'transformer': TransformerPricePredictor(forecast_horizon=12)
        }
        models[tf] = EnsemblePricePredictor(tf_models)
    
    mt_predictor = MultiTimeframePricePredictor(models)
    engine = AdvancedPricePredictionEngine(mt_predictor)
    
    # Start engine
    print("   Starting prediction engine...")
    symbols = ['BTC/USDT', 'ETH/USDT']
    await engine.start_prediction_loop(symbols, ['5m', '15m', '1h'])
    
    print(f"   ✓ Engine started for {len(symbols)} symbols")
    
    # Simulate predictions
    print("\n🔮 Simulating Price Forecasts...")
    for symbol in symbols:
        # Mock a forecast
        engine.prediction_cache[symbol] = {
            'aggregated': {
                'forecast': np.array([1.5, 1.2, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]),
                'uncertainty': np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]),
                'consensus_strength': 0.82
            },
            'timestamp': pd.Timestamp.now()
        }
        
        # Get forecast
        forecast = engine.get_price_forecast(symbol)
        
        print(f"\n   {symbol}:")
        print(f"      Forecast: {forecast['aggregated']['forecast'][:3]}... (first 3 steps)")
        print(f"      Consensus: {forecast['aggregated']['consensus_strength']:.2%}")
        
        # Generate trading signals
        current_price = 42000 if symbol == 'BTC/USDT' else 2200
        signals = engine.generate_trading_signals(symbol, current_price)
        
        print(f"      Signal: {signals['signal'].upper()}")
        print(f"      Strength: {signals['strength']:.2%}")
        print(f"      Position Size: {signals['position_size']:.2f}")
        print(f"      Expected Change: {signals['expected_change']:+.2%}")
    
    # Get engine status
    print("\n⚙️ Engine Status:")
    status = engine.get_engine_status()
    print(f"   Running: {status['running']}")
    print(f"   Symbols: {', '.join(status['symbols_tracked'])}")
    print(f"   Cached Predictions: {status['n_predictions_cached']}")
    
    # Stop engine
    await engine.stop_prediction_loop()
    print("\n   ✓ Engine stopped")
    
    print("\n✅ Real-time prediction engine demonstration completed!")
    print()


async def example_5_strategy_integration():
    """Example 5: AI-Enhanced Strategy Integration."""
    print("=" * 70)
    print("Example 5: AI-Enhanced Trading Strategy Integration")
    print("=" * 70)
    
    # Setup components
    print("\n🏗️ Setting up Integration Components...")
    models = {}
    for tf in ['5m', '15m']:
        tf_models = {
            'lstm': LSTMPricePredictor(forecast_horizon=12),
            'transformer': TransformerPricePredictor(forecast_horizon=12)
        }
        models[tf] = EnsemblePricePredictor(tf_models)
    
    mt_predictor = MultiTimeframePricePredictor(models)
    price_engine = AdvancedPricePredictionEngine(mt_predictor)
    regime_predictor = MLRegimePredictor()
    
    # Create integration manager
    manager = MLStrategyIntegrationManager(price_engine, regime_predictor)
    print("   ✓ Integration manager created")
    
    # Simulate base strategy signal
    print("\n📊 Base Strategy Signal:")
    base_signal = {
        'signal': 'bullish',
        'strength': 0.7,
        'reason': 'RSI oversold + MACD crossover'
    }
    print(f"   Signal: {base_signal['signal'].upper()}")
    print(f"   Strength: {base_signal['strength']:.2%}")
    print(f"   Reason: {base_signal['reason']}")
    
    # Mock AI forecast
    price_engine.prediction_cache['BTC/USDT'] = {
        'aggregated': {
            'forecast': np.array([2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
            'uncertainty': np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
            'consensus_strength': 0.85
        },
        'timestamp': pd.Timestamp.now()
    }
    
    # Process signal with AI enhancement
    print("\n🤖 Processing with AI Enhancement...")
    result = await manager.process_strategy_signal(
        'BTC/USDT', base_signal, 42000.0, 1.0
    )
    
    # Display enhanced signal
    enhancement = result['enhancement']
    print("\n📊 Enhanced Signal:")
    print(f"   Original Signal:       {enhancement['original_signal'].upper()}")
    print(f"   AI Signal:             {enhancement['ai_signal'].upper()}")
    print(f"   Final Signal:          {enhancement['final_signal'].upper()}")
    print(f"   Original Strength:     {enhancement['original_strength']:.2%}")
    print(f"   AI Strength:           {enhancement['ai_strength']:.2%}")
    print(f"   Final Strength:        {enhancement['final_strength']:.2%}")
    
    if 'recommendations' in enhancement:
        print("\n💡 Recommendations:")
        for rec in enhancement['recommendations']:
            print(f"      • {rec}")
    
    # Display position sizing
    sizing = result['position_sizing']
    print("\n💰 Position Sizing:")
    print(f"   Base Position:         {sizing['base_position']:.2f}")
    print(f"   Adjusted Position:     {sizing['adjusted_position']:.2f}")
    print(f"   Confidence Multiplier: {sizing['confidence_multiplier']:.2f}x")
    print(f"   Risk Multiplier:       {sizing['risk_multiplier']:.2f}x")
    print(f"   Final Multiplier:      {sizing['final_multiplier']:.2f}x")
    
    # Display risk metrics
    risk = result['risk_metrics']
    print("\n⚠️ Risk Metrics:")
    print(f"   Risk Level:            {risk['risk_level'].upper()}")
    print(f"   Uncertainty:           {risk['uncertainty']:.2%}")
    print(f"   Consensus:             {risk['consensus']:.2%}")
    print(f"   Confidence:            {risk['confidence']:.2%}")
    
    print("\n✅ Strategy integration demonstration completed!")
    print()


async def example_6_performance_tracking():
    """Example 6: Strategy Performance Tracking."""
    print("=" * 70)
    print("Example 6: Strategy Performance Tracking")
    print("=" * 70)
    
    # Create tracker
    print("\n📊 Initializing Performance Tracker...")
    tracker = StrategyPerformanceTracker()
    
    # Simulate trades
    print("\n💼 Simulating Trades...")
    
    # Base strategy trades
    print("\n   Base Strategy Trades:")
    for i in range(5):
        pnl = np.random.randn() * 100
        trade = {
            'strategy_type': 'base',
            'symbol': 'BTC/USDT',
            'pnl': pnl,
            'entry_price': 42000,
            'exit_price': 42000 + pnl
        }
        tracker.record_trade(trade)
        print(f"      Trade {i+1}: PnL = ${pnl:+.2f}")
    
    # AI-enhanced strategy trades
    print("\n   AI-Enhanced Strategy Trades:")
    for i in range(5):
        pnl = np.random.randn() * 120 + 20  # Slightly better
        trade = {
            'strategy_type': 'ai_enhanced',
            'symbol': 'BTC/USDT',
            'pnl': pnl,
            'entry_price': 42000,
            'exit_price': 42000 + pnl
        }
        tracker.record_trade(trade)
        print(f"      Trade {i+1}: PnL = ${pnl:+.2f}")
    
    # Get performance summary
    print("\n📊 Performance Summary:")
    summary = tracker.get_performance_summary()
    print(f"   Total Trades:          {summary['total_trades']}")
    print(f"   Base Strategy Wins:    {summary['base_strategy_wins']}")
    print(f"   AI Enhanced Wins:      {summary['ai_enhanced_wins']}")
    if summary['improvement_rate'] != 0:
        print(f"   Improvement Rate:      {summary['improvement_rate']:+.2f}%")
    
    # Get recent trades
    recent = tracker.get_recent_performance(n_trades=10)
    print(f"\n📈 Recent Trade History: {len(recent)} trades")
    
    print("\n✅ Performance tracking demonstration completed!")
    print()


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Phase 4 Final: Advanced Price Prediction Examples")
    print("=" * 70)
    print("\nDemonstrating LSTM/Transformer models for real-time price prediction")
    print("with multi-timeframe forecasting, ensemble predictions, confidence")
    print("intervals, and trading strategy integration.")
    print("=" * 70)
    
    # Run examples
    example_1_lstm_transformer_models()
    example_2_ensemble_prediction()
    example_3_multi_timeframe_forecasting()
    await example_4_real_time_prediction_engine()
    await example_5_strategy_integration()
    await example_6_performance_tracking()
    
    print("=" * 70)
    print("🎉 All examples completed successfully!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ LSTM and Transformer price prediction models")
    print("  ✓ Ensemble predictions combining multiple models")
    print("  ✓ Multi-timeframe forecasting (5m, 15m, 1h)")
    print("  ✓ Confidence intervals for predictions")
    print("  ✓ Real-time prediction engine")
    print("  ✓ Trading signal generation from forecasts")
    print("  ✓ AI-enhanced strategy integration")
    print("  ✓ Position sizing with confidence adjustments")
    print("  ✓ Risk metrics and performance tracking")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
