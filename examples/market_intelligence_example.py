#!/usr/bin/env python3
"""
Phase 2: Market Intelligence Engine - Usage Examples
Demonstrates real-time regime detection, adaptive strategies, VST intelligence, and performance monitoring.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

from core.market_regime import MarketRegimeAnalyzer
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip
from core.vst_intelligence import VSTMarketAnalyzer
from core.performance_monitor import RealTimePerformanceMonitor
from core.ccxt_client import CcxtClient
from core.indicators import add_indicators


def create_sample_data(n_bars=100, trend='neutral'):
    """Create sample OHLCV data for demonstration."""
    np.random.seed(42)
    
    if trend == 'bullish':
        base = np.linspace(100, 120, n_bars)
    elif trend == 'bearish':
        base = np.linspace(120, 100, n_bars)
    else:
        base = np.ones(n_bars) * 110
    
    noise = np.random.randn(n_bars) * 0.5
    close = base + noise
    high = close + np.abs(np.random.randn(n_bars) * 0.5)
    low = close - np.abs(np.random.randn(n_bars) * 0.5)
    open_price = close + np.random.randn(n_bars) * 0.2
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_bars, freq='30min'),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars)
    })
    
    # Add indicators
    df = add_indicators(df)
    
    # Don't dropna - keep all rows with indicators
    return df


def example_1_market_regime_detection():
    """Example 1: Multi-Timeframe Market Regime Detection."""
    print("=" * 70)
    print("Example 1: Market Regime Detection")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = MarketRegimeAnalyzer()
    
    # Create sample data for different timeframes
    df_30m = create_sample_data(n_bars=100, trend='bearish')
    df_1h = create_sample_data(n_bars=50, trend='bearish')
    df_4h = create_sample_data(n_bars=50, trend='bearish')
    
    # Analyze market regime
    print("\nAnalyzing market regime across multiple timeframes...")
    regime = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
    
    print(f"\n📊 Market Regime Analysis:")
    print(f"   Primary Trend (4H):         {regime['trend'].upper()}")
    print(f"   Momentum (1H):              {regime['momentum'].upper()}")
    print(f"   Volatility Regime:          {regime['volatility'].upper()}")
    print(f"   Risk Multiplier:            {regime['risk_multiplier']:.2f}x")
    print(f"   Micro-Trend Strength:       {regime['micro_trend_strength']:.2f}")
    print(f"   Entry Score:                {regime['entry_score']:.2f}")
    
    print("\n✓ Market regime detected successfully")
    print()


def example_2_adaptive_oversold_bounce():
    """Example 2: Adaptive OversoldBounce Strategy."""
    print("=" * 70)
    print("Example 2: Adaptive OversoldBounce Strategy")
    print("=" * 70)
    
    # Initialize components
    analyzer = MarketRegimeAnalyzer()
    
    # Strategy configuration
    cfg = {
        'rsi_max': 25,
        'tp_pct': 0.015,
        'sl_atr_mult': 1.0
    }
    
    strategy = AdaptiveOversoldBounce(cfg, analyzer)
    
    # Create data with different market regimes
    regimes = ['bullish', 'bearish', 'neutral']
    
    print("\nTesting adaptive RSI thresholds in different market regimes:")
    print()
    
    for regime_type in regimes:
        df_30m = create_sample_data(n_bars=100, trend=regime_type)
        df_1h = create_sample_data(n_bars=50, trend=regime_type)
        df_4h = create_sample_data(n_bars=50, trend=regime_type)
        
        # Analyze regime
        regime_data = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
        
        # Get adaptive threshold
        threshold = strategy.get_adaptive_rsi_threshold(regime_data)
        
        # Get position size multiplier
        pos_mult = strategy.calculate_dynamic_position_size(regime_data['volatility'])
        
        print(f"📈 {regime_type.upper()} REGIME:")
        print(f"   Adaptive RSI Threshold:     {threshold:.1f} (base: {cfg['rsi_max']})")
        print(f"   Position Size Multiplier:   {pos_mult:.2f}x")
        print(f"   Volatility:                 {regime_data['volatility']}")
        
        # Force oversold condition for signal generation
        df_30m_test = df_30m.copy()
        # Make sure all indicator columns have values in last row
        last_idx = df_30m_test.index[-1]
        if pd.isna(df_30m_test.loc[last_idx, 'rsi']):
            # Fill NaN values from previous row
            for col in ['rsi', 'atr', 'ema21', 'ema50', 'ema200', 'ema_fast', 'ema_mid', 'ema_slow']:
                if col in df_30m_test.columns and pd.isna(df_30m_test.loc[last_idx, col]):
                    df_30m_test.loc[last_idx, col] = df_30m_test.loc[df_30m_test.index[-2], col]
        df_30m_test.loc[last_idx, 'rsi'] = 20
        
        signal = strategy.signal(df_30m_test, regime_data)
        if signal:
            print(f"   Signal Generated:           ✓ {signal['reason']}")
        else:
            print(f"   Signal Generated:           ✗ No signal")
        print()
    
    print("✓ Adaptive OversoldBounce demonstration complete")
    print()


def example_3_adaptive_short_the_rip():
    """Example 3: Adaptive ShortTheRip Strategy."""
    print("=" * 70)
    print("Example 3: Adaptive ShortTheRip Strategy")
    print("=" * 70)
    
    # Initialize components
    analyzer = MarketRegimeAnalyzer()
    
    # Strategy configuration
    cfg = {
        'rsi_min': 65,
        'tp_pct': 0.012,
        'sl_atr_mult': 1.2
    }
    
    strategy = AdaptiveShortTheRip(cfg, analyzer)
    
    # Create bearish market data
    df_30m = create_sample_data(n_bars=100, trend='bearish')
    df_1h = create_sample_data(n_bars=50, trend='bearish')
    df_4h = create_sample_data(n_bars=50, trend='bearish')
    
    # Analyze regime
    regime_data = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
    
    print(f"\n📊 Market Regime: {regime_data['trend'].upper()}")
    
    # Get adaptive parameters
    threshold = strategy.get_adaptive_rsi_threshold(regime_data)
    pos_mult = strategy.calculate_dynamic_position_size(regime_data['volatility'])
    
    print(f"\n🎯 Adaptive Parameters:")
    print(f"   Base RSI Threshold:         {cfg['rsi_min']}")
    print(f"   Adaptive RSI Threshold:     {threshold:.1f}")
    print(f"   Position Size Multiplier:   {pos_mult:.2f}x")
    print(f"   Volatility Regime:          {regime_data['volatility']}")
    
    # Force overbought condition
    df_30m_test = df_30m.copy()
    last_idx = df_30m_test.index[-1]
    # Fill NaN values
    if pd.isna(df_30m_test.loc[last_idx, 'rsi']):
        for col in ['rsi', 'atr', 'ema21', 'ema50', 'ema200', 'ema_fast', 'ema_mid', 'ema_slow']:
            if col in df_30m_test.columns and pd.isna(df_30m_test.loc[last_idx, col]):
                df_30m_test.loc[last_idx, col] = df_30m_test.loc[df_30m_test.index[-2], col]
    df_30m_test.loc[last_idx, 'rsi'] = 75
    
    signal = strategy.signal(df_30m_test, df_1h, regime_data)
    
    if signal:
        print(f"\n✓ Signal Generated:")
        print(f"   {signal['reason']}")
        print(f"   Position Multiplier: {signal['position_multiplier']:.2f}x")
    else:
        print(f"\n✗ No signal generated")
    
    print("\n✓ Adaptive ShortTheRip demonstration complete")
    print()


def example_4_vst_intelligence():
    """Example 4: VST Market Intelligence."""
    print("=" * 70)
    print("Example 4: VST Market Intelligence for BingX")
    print("=" * 70)
    
    # Mock BingX client (in production, use actual CcxtClient)
    class MockBingXClient:
        def __init__(self):
            self.exchange_id = 'bingx'
    
    client = MockBingXClient()
    vst_analyzer = VSTMarketAnalyzer(client)
    
    # Create VST data with high volatility
    vst_data = create_sample_data(n_bars=100, trend='neutral')
    vst_data['close'] = vst_data['close'] + np.random.randn(100) * 2  # Add volatility
    
    print("\nAnalyzing VST price patterns...")
    patterns = vst_analyzer.analyze_vst_price_patterns(vst_data)
    
    print(f"\n📊 VST Pattern Analysis:")
    print(f"   Volatility Profile:         {patterns['volatility_profile'].upper()}")
    print(f"   Average Move:               {patterns['average_move']:.4f}")
    print(f"   Average Volatility:         {patterns.get('average_volatility', 0):.4f}")
    print(f"   Volume Trend:               {patterns.get('volume_trend', 'unknown').upper()}")
    
    # Optimize test trading parameters
    print("\n🎯 Optimizing VST Test Trading Parameters...")
    
    market_regime = {
        'trend': 'bearish',
        'momentum': 'weak',
        'volatility': 'high'
    }
    
    params = vst_analyzer.optimize_test_trading_parameters(market_regime)
    
    print(f"\n   Position Size Multiplier:   {params['position_size_mult']:.1%}")
    print(f"   Max Positions:              {params['max_positions']}")
    print(f"   Risk per Trade:             {params['risk_per_trade']:.1%}")
    print(f"   OB RSI Max:                 {params.get('ob_rsi_max', 'N/A')}")
    print(f"   STR RSI Min:                {params.get('str_rsi_min', 'N/A')}")
    print(f"   Exchange:                   {params['exchange']}")
    print(f"   Symbol:                     {params['symbol']}")
    
    # Simulate performance monitoring
    print("\n📈 Simulating VST Performance Monitoring...")
    
    trade_results = [
        {'pnl': 15.5},
        {'pnl': -8.2},
        {'pnl': 22.1},
        {'pnl': -5.3},
        {'pnl': 18.7}
    ]
    
    for i, result in enumerate(trade_results, 1):
        monitoring = vst_analyzer.monitor_vst_performance(result)
        print(f"   Trade {i}: PnL ${result['pnl']:.2f}")
    
    # Get final status
    status = vst_analyzer.get_vst_status()
    print(f"\n✓ VST Intelligence active:")
    print(f"   Tracked Trades:             {status['performance_trades']}")
    print(f"   Symbol:                     {status['symbol']}")
    
    print("\n✓ VST Intelligence demonstration complete")
    print()


def example_5_performance_monitoring():
    """Example 5: Real-Time Performance Monitoring."""
    print("=" * 70)
    print("Example 5: Real-Time Performance Monitoring")
    print("=" * 70)
    
    monitor = RealTimePerformanceMonitor()
    
    # Simulate trading performance for two strategies
    strategies = {
        'oversold_bounce': [10, -5, 15, -3, 12, -7, 20, -4, 18, -6, 25, -8, 14, -5, 22],
        'short_the_rip': [8, -6, 12, -9, 15, -7, 10, -5, 18, -8, 20, -6, 16, -4, 11]
    }
    
    print("\nSimulating strategy performance tracking...")
    print()
    
    for strategy_name, pnls in strategies.items():
        print(f"📊 Tracking {strategy_name}:")
        for pnl in pnls:
            result = {'pnl': pnl}
            monitor.track_strategy_performance(strategy_name, result)
        
        # Get metrics
        metrics = monitor.performance_history[strategy_name]['metrics']
        print(f"   Trades:                     {metrics['trade_count']}")
        print(f"   Win Rate:                   {metrics['win_rate']:.1%}")
        print(f"   Average Win:                ${metrics['avg_win']:.2f}")
        print(f"   Average Loss:               ${metrics['avg_loss']:.2f}")
        print(f"   Risk/Reward:                {metrics['risk_reward']:.2f}")
        print(f"   Total PnL:                  ${metrics['total_pnl']:.2f}")
        print(f"   Sharpe Ratio:               {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown:               ${metrics['max_drawdown']:.2f}")
        print()
    
    # Get optimization feedback
    print("🎯 Optimization Feedback:")
    print()
    
    for strategy_name in strategies.keys():
        feedback = monitor.provide_optimization_feedback(strategy_name)
        print(f"   {strategy_name}:")
        print(f"      Status: {feedback['status']}")
        if feedback['recommendations']:
            for rec in feedback['recommendations']:
                print(f"      - {rec}")
        print()
    
    # Check parameter drift
    print("🔍 Parameter Drift Detection:")
    print()
    
    for strategy_name in strategies.keys():
        params = {'rsi_max': 25, 'tp_pct': 0.015}
        needs_adjustment, reasons = monitor.detect_parameter_drift(strategy_name, params)
        
        print(f"   {strategy_name}:")
        print(f"      Needs Adjustment: {'Yes' if needs_adjustment else 'No'}")
        if reasons:
            for reason in reasons:
                print(f"      - {reason}")
        print()
    
    # Get summary for all strategies
    print("📋 Strategy Summary:")
    summary = monitor.get_all_strategies_summary()
    for strategy, info in summary.items():
        print(f"\n   {strategy}:")
        print(f"      Status: {info['status']}")
        print(f"      Trades: {info['trade_count']}")
        if 'win_rate' in info.get('metrics', {}):
            print(f"      Win Rate: {info['metrics']['win_rate']:.1%}")
    
    print("\n✓ Performance monitoring demonstration complete")
    print()


def example_6_integrated_workflow():
    """Example 6: Complete Integrated Workflow."""
    print("=" * 70)
    print("Example 6: Complete Integrated Workflow")
    print("=" * 70)
    
    print("\n🚀 Demonstrating complete market intelligence workflow...")
    
    # 1. Initialize all components
    regime_analyzer = MarketRegimeAnalyzer()
    performance_monitor = RealTimePerformanceMonitor()
    
    # 2. Create adaptive strategies
    ob_cfg = {'rsi_max': 25, 'tp_pct': 0.015, 'sl_atr_mult': 1.0}
    str_cfg = {'rsi_min': 65, 'tp_pct': 0.012, 'sl_atr_mult': 1.2}
    
    adaptive_ob = AdaptiveOversoldBounce(ob_cfg, regime_analyzer)
    adaptive_str = AdaptiveShortTheRip(str_cfg, regime_analyzer)
    
    # 3. Simulate market analysis and trading
    print("\n📊 Step 1: Analyze Market Regime")
    df_30m = create_sample_data(n_bars=100, trend='bearish')
    df_1h = create_sample_data(n_bars=50, trend='bearish')
    df_4h = create_sample_data(n_bars=50, trend='bearish')
    
    regime = regime_analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
    print(f"   Trend: {regime['trend']}, Volatility: {regime['volatility']}, Momentum: {regime['momentum']}")
    
    # 4. Generate adaptive signals
    print("\n📈 Step 2: Generate Adaptive Signals")
    
    # Force conditions for signals
    df_30m_ob = df_30m.copy()
    last_idx = df_30m_ob.index[-1]
    if pd.isna(df_30m_ob.loc[last_idx, 'rsi']):
        for col in ['rsi', 'atr', 'ema21', 'ema50', 'ema200', 'ema_fast', 'ema_mid', 'ema_slow']:
            if col in df_30m_ob.columns and pd.isna(df_30m_ob.loc[last_idx, col]):
                df_30m_ob.loc[last_idx, col] = df_30m_ob.loc[df_30m_ob.index[-2], col]
    df_30m_ob.loc[last_idx, 'rsi'] = 20
    
    ob_signal = adaptive_ob.signal(df_30m_ob, regime)
    if ob_signal:
        print(f"   ✓ OversoldBounce: {ob_signal['reason']}")
        print(f"     Position Mult: {ob_signal['position_multiplier']:.2f}x")
    
    df_30m_str = df_30m.copy()
    last_idx = df_30m_str.index[-1]
    if pd.isna(df_30m_str.loc[last_idx, 'rsi']):
        for col in ['rsi', 'atr', 'ema21', 'ema50', 'ema200', 'ema_fast', 'ema_mid', 'ema_slow']:
            if col in df_30m_str.columns and pd.isna(df_30m_str.loc[last_idx, col]):
                df_30m_str.loc[last_idx, col] = df_30m_str.loc[df_30m_str.index[-2], col]
    df_30m_str.loc[last_idx, 'rsi'] = 75
    
    str_signal = adaptive_str.signal(df_30m_str, df_1h, regime)
    if str_signal:
        print(f"   ✓ ShortTheRip: {str_signal['reason']}")
        print(f"     Position Mult: {str_signal['position_multiplier']:.2f}x")
    
    # 5. Track performance
    print("\n📊 Step 3: Track Performance")
    
    # Simulate some trades
    ob_trades = [12, -5, 18, -3, 15, -7, 20]
    str_trades = [10, -6, 14, -4, 16, -8, 12]
    
    for pnl in ob_trades:
        performance_monitor.track_strategy_performance('adaptive_ob', {'pnl': pnl})
    
    for pnl in str_trades:
        performance_monitor.track_strategy_performance('adaptive_str', {'pnl': pnl})
    
    # 6. Get optimization feedback
    print("\n🎯 Step 4: Get Optimization Feedback")
    
    ob_feedback = performance_monitor.provide_optimization_feedback('adaptive_ob')
    print(f"   Adaptive OB:")
    if 'metrics' in ob_feedback and ob_feedback['metrics']:
        print(f"     Win Rate: {ob_feedback['metrics']['win_rate']:.1%}")
        print(f"     Total PnL: ${ob_feedback['metrics']['total_pnl']:.2f}")
    if ob_feedback.get('recommendations'):
        print(f"     Recommendations: {len(ob_feedback['recommendations'])} suggestions")
    
    str_feedback = performance_monitor.provide_optimization_feedback('adaptive_str')
    print(f"   Adaptive STR:")
    if 'metrics' in str_feedback and str_feedback['metrics']:
        print(f"     Win Rate: {str_feedback['metrics']['win_rate']:.1%}")
        print(f"     Total PnL: ${str_feedback['metrics']['total_pnl']:.2f}")
    if str_feedback.get('recommendations'):
        print(f"     Recommendations: {len(str_feedback['recommendations'])} suggestions")
    
    print("\n✅ Complete workflow demonstration finished!")
    print("   All components working together seamlessly.")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  PHASE 2: MARKET INTELLIGENCE ENGINE - EXAMPLES")
    print("=" * 70)
    print()
    
    examples = [
        example_1_market_regime_detection,
        example_2_adaptive_oversold_bounce,
        example_3_adaptive_short_the_rip,
        example_4_vst_intelligence,
        example_5_performance_monitoring,
        example_6_integrated_workflow
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"✗ Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 70)
    print("  ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
