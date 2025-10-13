#!/usr/bin/env python3
"""
Phase 4 Complete Integration Example

Demonstrates the full Phase 4 AI enhancement system including:
- Phase 4.1: ML Regime Prediction
- Phase 4.2: Adaptive Learning (conceptual)
- Phase 4.3: Strategy Optimization (conceptual)
- Phase 4 Final: Advanced Price Prediction

Shows how all components work together for complete AI-enhanced trading.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime

# Phase 4.1: Regime Prediction
from ml.regime_predictor import MLRegimePredictor
from ml.feature_engineering import FeatureEngineeringPipeline

# Phase 4 Final: Price Prediction
from ml.price_predictor import (
    LSTMPricePredictor,
    TransformerPricePredictor,
    EnsemblePricePredictor,
    MultiTimeframePricePredictor,
    AdvancedPricePredictionEngine
)
from ml.strategy_integration import MLStrategyIntegrationManager


def create_sample_data(n_bars=500, timeframe='5m'):
    """Create realistic sample data."""
    np.random.seed(42)
    
    freq_map = {'5m': '5min', '15m': '15min', '1h': '1H'}
    freq = freq_map.get(timeframe, '5min')
    
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq=freq)
    
    # Create uptrending market with volatility
    trend = np.linspace(40000, 45000, n_bars)
    volatility = np.random.randn(n_bars) * 200
    close = trend + volatility
    
    high = close + np.abs(np.random.randn(n_bars) * 100)
    low = close - np.abs(np.random.randn(n_bars) * 100)
    open_price = close + np.random.randn(n_bars) * 50
    volume = np.abs(np.random.randn(n_bars) * 100 + 1000)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


def create_regime_labels(data):
    """Create regime labels from price data."""
    # Simple regime labeling based on trend
    returns = data['close'].pct_change(20)
    
    labels = []
    for ret in returns:
        if pd.isna(ret):
            labels.append(1)  # neutral
        elif ret > 0.02:
            labels.append(0)  # bullish
        elif ret < -0.02:
            labels.append(2)  # bearish
        else:
            labels.append(1)  # neutral
    
    return pd.Series(labels, index=data.index)


async def demonstrate_complete_system():
    """Demonstrate the complete Phase 4 AI system."""
    
    print("\n" + "=" * 70)
    print("Phase 4 Complete AI Enhancement System")
    print("=" * 70)
    print("\nIntegrating all Phase 4 components for comprehensive AI trading\n")
    
    # ========================================================================
    # Step 1: Setup Data
    # ========================================================================
    print("=" * 70)
    print("Step 1: Data Preparation")
    print("=" * 70)
    
    print("\n📊 Creating multi-timeframe market data...")
    data_by_tf = {
        '5m': create_sample_data(n_bars=500, timeframe='5m'),
        '15m': create_sample_data(n_bars=200, timeframe='15m'),
        '1h': create_sample_data(n_bars=100, timeframe='1h')
    }
    
    for tf, data in data_by_tf.items():
        print(f"   {tf:>3}: {len(data):>4} bars, ${data['close'].iloc[-1]:>8,.2f} current")
    
    # ========================================================================
    # Step 2: Phase 4.1 - Regime Prediction
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Phase 4.1 - ML Regime Prediction")
    print("=" * 70)
    
    print("\n🧠 Training regime prediction model...")
    regime_predictor = MLRegimePredictor()
    
    # Use 5m data for regime training
    training_data = data_by_tf['5m']
    regime_labels = create_regime_labels(training_data)
    
    train_result = regime_predictor.train_regime_models(training_data, regime_labels)
    print(f"   ✓ Trained on {train_result['n_samples']} samples")
    print(f"   ✓ Accuracy: {train_result['train_accuracy']:.2%}")
    
    # Make regime prediction
    regime_pred = await regime_predictor.predict_regime_transition(
        'BTC/USDT', training_data
    )
    
    print(f"\n📊 Current Regime Prediction:")
    print(f"   Regime:     {regime_pred['predicted_regime'].upper()}")
    print(f"   Confidence: {regime_pred['confidence']:.2%}")
    print(f"   Probabilities:")
    for regime, prob in regime_pred['probabilities'].items():
        print(f"      {regime:>8}: {prob:.2%}")
    
    # ========================================================================
    # Step 3: Phase 4 Final - Price Prediction
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Phase 4 Final - Advanced Price Prediction")
    print("=" * 70)
    
    print("\n🔮 Building multi-timeframe price prediction system...")
    
    # Create models for each timeframe
    price_models = {}
    for tf in ['5m', '15m', '1h']:
        tf_models = {
            'lstm': LSTMPricePredictor(forecast_horizon=12),
            'transformer': TransformerPricePredictor(forecast_horizon=12)
        }
        price_models[tf] = EnsemblePricePredictor(tf_models)
        print(f"   ✓ Created ensemble for {tf}")
    
    mt_predictor = MultiTimeframePricePredictor(price_models)
    
    # Make multi-timeframe prediction
    print("\n🔮 Making multi-timeframe price forecast...")
    price_forecast = mt_predictor.predict_multi_timeframe(data_by_tf)
    
    print("\n📊 Price Forecast by Timeframe:")
    for tf, pred in price_forecast['by_timeframe'].items():
        current = pred['current_price']
        forecast_pct = pred['forecast'][0]
        forecast_price = current * (1 + forecast_pct / 100)
        ci = pred['confidence_interval']
        
        print(f"\n   {tf} Timeframe:")
        print(f"      Current:  ${current:,.2f}")
        print(f"      Forecast: ${forecast_price:,.2f} ({forecast_pct:+.2f}%)")
        print(f"      Range:    ${ci['lower'][0]:,.2f} - ${ci['upper'][0]:,.2f}")
        print(f"      Uncertainty: {pred['uncertainty'][0]:.2f}%")
    
    # Aggregated forecast
    agg = price_forecast['aggregated']
    base_price = data_by_tf['5m']['close'].iloc[-1]
    forecast_price = base_price * (1 + agg['forecast'][0] / 100)
    
    print(f"\n📊 Aggregated Multi-Timeframe Forecast:")
    print(f"   Current Price:     ${base_price:,.2f}")
    print(f"   Forecast Price:    ${forecast_price:,.2f} ({agg['forecast'][0]:+.2f}%)")
    print(f"   Uncertainty:       {agg['uncertainty'][0]:.2f}%")
    print(f"   Consensus:         {agg['consensus_strength']:.2%}")
    
    # ========================================================================
    # Step 4: Strategy Integration
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: AI-Enhanced Strategy Integration")
    print("=" * 70)
    
    print("\n🤝 Integrating AI predictions with trading strategy...")
    
    # Create integration manager
    price_engine = AdvancedPricePredictionEngine(mt_predictor)
    integration_manager = MLStrategyIntegrationManager(price_engine, regime_predictor)
    
    # Mock price forecast in engine
    price_engine.prediction_cache['BTC/USDT'] = price_forecast
    
    # Simulate base strategy signal
    print("\n📈 Base Strategy Signal:")
    base_signal = {
        'signal': 'bullish',
        'strength': 0.75,
        'reason': 'Strong momentum + volume confirmation'
    }
    print(f"   Signal:   {base_signal['signal'].upper()}")
    print(f"   Strength: {base_signal['strength']:.2%}")
    print(f"   Reason:   {base_signal['reason']}")
    
    # Enhance with AI
    print("\n🤖 Enhancing with AI predictions...")
    enhanced = await integration_manager.process_strategy_signal(
        'BTC/USDT', base_signal, base_price, 1.0
    )
    
    # Display results
    enh = enhanced['enhancement']
    print(f"\n📊 Enhanced Signal:")
    print(f"   Original Signal:  {enh['original_signal'].upper()}")
    print(f"   AI Signal:        {enh['ai_signal'].upper()}")
    print(f"   Final Signal:     {enh['final_signal'].upper()}")
    print(f"   Original Strength: {enh['original_strength']:.2%}")
    print(f"   Final Strength:    {enh['final_strength']:.2%}")
    
    if 'recommendations' in enh:
        print(f"\n💡 AI Recommendations:")
        for rec in enh['recommendations']:
            print(f"      • {rec}")
    
    # Position sizing
    sizing = enhanced['position_sizing']
    print(f"\n💰 Position Sizing:")
    print(f"   Base Position:      {sizing['base_position']:.2f}")
    print(f"   Adjusted Position:  {sizing['adjusted_position']:.2f}")
    print(f"   Multiplier:         {sizing['final_multiplier']:.2f}x")
    print(f"   Confidence Adj:     {sizing['confidence_multiplier']:.2f}x")
    print(f"   Risk Adj:           {sizing['risk_multiplier']:.2f}x")
    
    # Risk metrics
    risk = enhanced['risk_metrics']
    print(f"\n⚠️  Risk Assessment:")
    print(f"   Risk Level:         {risk['risk_level'].upper()}")
    print(f"   Uncertainty:        {risk['uncertainty']:.2%}")
    print(f"   Consensus:          {risk['consensus']:.2%}")
    print(f"   Confidence:         {risk['confidence']:.2%}")
    
    # ========================================================================
    # Step 5: Complete Decision Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Complete AI-Enhanced Trading Decision")
    print("=" * 70)
    
    print("\n🎯 FINAL TRADING DECISION:")
    print(f"\n   Symbol:              BTC/USDT")
    print(f"   Current Price:       ${base_price:,.2f}")
    print(f"   Predicted Price:     ${forecast_price:,.2f}")
    print(f"   Expected Move:       {agg['forecast'][0]:+.2f}%")
    print(f"\n   Market Regime:       {regime_pred['predicted_regime'].upper()}")
    print(f"   Regime Confidence:   {regime_pred['confidence']:.2%}")
    print(f"\n   Trading Signal:      {enh['final_signal'].upper()}")
    print(f"   Signal Strength:     {enh['final_strength']:.2%}")
    print(f"   Position Size:       {sizing['adjusted_position']:.2f}")
    print(f"\n   Risk Level:          {risk['risk_level'].upper()}")
    print(f"   Overall Confidence:  {risk['confidence']:.2%}")
    print(f"   Timeframe Consensus: {agg['consensus_strength']:.2%}")
    
    # Trading recommendation
    print(f"\n📋 RECOMMENDATION:")
    
    if enh['final_signal'] == 'bullish' and enh['final_strength'] > 0.6:
        action = "ENTER LONG"
        entry = base_price
        target = forecast_price
        stop = base_price * 0.98
        
        print(f"   Action:    {action}")
        print(f"   Entry:     ${entry:,.2f}")
        print(f"   Target:    ${target:,.2f} ({((target/entry)-1)*100:+.2f}%)")
        print(f"   Stop Loss: ${stop:,.2f} ({((stop/entry)-1)*100:+.2f}%)")
        print(f"   Position:  {sizing['adjusted_position']:.2f} units")
        
    elif enh['final_signal'] == 'bearish' and enh['final_strength'] > 0.6:
        action = "ENTER SHORT"
        entry = base_price
        target = forecast_price
        stop = base_price * 1.02
        
        print(f"   Action:    {action}")
        print(f"   Entry:     ${entry:,.2f}")
        print(f"   Target:    ${target:,.2f} ({((target/entry)-1)*100:+.2f}%)")
        print(f"   Stop Loss: ${stop:,.2f} ({((stop/entry)-1)*100:+.2f}%)")
        print(f"   Position:  {sizing['adjusted_position']:.2f} units")
        
    else:
        print(f"   Action:    STAY NEUTRAL / WAIT")
        print(f"   Reason:    Insufficient signal strength or conflicting indicators")
        print(f"   Suggest:   Monitor for clearer setup")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Phase 4 Complete AI System Summary")
    print("=" * 70)
    
    print("\n✅ Components Demonstrated:")
    print("   ✓ Phase 4.1: ML Regime Prediction")
    print("   ✓ Phase 4 Final: Advanced Price Prediction")
    print("   ✓ Multi-Timeframe Forecasting")
    print("   ✓ Ensemble Predictions")
    print("   ✓ Confidence Intervals")
    print("   ✓ Strategy Integration")
    print("   ✓ Position Sizing")
    print("   ✓ Risk Management")
    
    print("\n🎯 Key Insights:")
    print(f"   • {len(price_forecast['by_timeframe'])} timeframes analyzed")
    print(f"   • Consensus strength: {agg['consensus_strength']:.0%}")
    print(f"   • AI enhanced signal from {enh['original_strength']:.0%} to {enh['final_strength']:.0%}")
    print(f"   • Position adjusted by {sizing['final_multiplier']:.2f}x based on confidence")
    print(f"   • Risk level: {risk['risk_level']}")
    
    print("\n" + "=" * 70)
    print("🎉 Complete Phase 4 AI System Demonstration Finished!")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(demonstrate_complete_system())
