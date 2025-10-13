"""
Comprehensive tests for Phase 2: Market Intelligence Engine.
Tests market regime detection, adaptive strategies, VST intelligence, and performance monitoring.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from core.market_regime import MarketRegimeAnalyzer
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip
from core.vst_intelligence import VSTMarketAnalyzer
from core.performance_monitor import RealTimePerformanceMonitor


def create_test_ohlcv(n_bars=100, trend='neutral', volatility='normal'):
    """Create test OHLCV data with indicators."""
    np.random.seed(42)
    
    # Base price
    if trend == 'bullish':
        base = np.linspace(100, 120, n_bars)
    elif trend == 'bearish':
        base = np.linspace(120, 100, n_bars)
    else:
        base = np.ones(n_bars) * 110
    
    # Add volatility
    if volatility == 'high':
        noise = np.random.randn(n_bars) * 2
    elif volatility == 'low':
        noise = np.random.randn(n_bars) * 0.2
    else:
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
    df['rsi'] = 50 + np.random.randn(n_bars) * 15
    df['rsi'] = df['rsi'].clip(0, 100)
    
    df['atr'] = np.abs(np.random.randn(n_bars)) * 2 + 1
    
    # EMAs based on trend
    if trend == 'bullish':
        df['ema21'] = df['close'] * 0.98
        df['ema50'] = df['close'] * 0.96
        df['ema200'] = df['close'] * 0.94
    elif trend == 'bearish':
        df['ema21'] = df['close'] * 1.02
        df['ema50'] = df['close'] * 1.04
        df['ema200'] = df['close'] * 1.06
    else:
        df['ema21'] = df['close'] * 1.01
        df['ema50'] = df['close'] * 0.99
        df['ema200'] = df['close'] * 1.00
    
    return df


class TestMarketRegimeAnalyzer:
    """Test MarketRegimeAnalyzer functionality."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = MarketRegimeAnalyzer()
        assert analyzer is not None
        assert 'trend' in analyzer.regime_states
        assert 'volatility' in analyzer.regime_states
        assert 'momentum' in analyzer.regime_states
    
    def test_detect_bullish_trend(self):
        """Test bullish trend detection."""
        analyzer = MarketRegimeAnalyzer()
        df = create_test_ohlcv(n_bars=100, trend='bullish')
        
        trend = analyzer.detect_primary_trend_4h(df)
        assert trend in ['bullish', 'bearish', 'neutral']
        # With properly aligned EMAs, should detect bullish
        assert trend == 'bullish'
    
    def test_detect_bearish_trend(self):
        """Test bearish trend detection."""
        analyzer = MarketRegimeAnalyzer()
        df = create_test_ohlcv(n_bars=100, trend='bearish')
        
        trend = analyzer.detect_primary_trend_4h(df)
        assert trend in ['bullish', 'bearish', 'neutral']
        assert trend == 'bearish'
    
    def test_momentum_confirmation(self):
        """Test momentum confirmation."""
        analyzer = MarketRegimeAnalyzer()
        df = create_test_ohlcv(n_bars=50, trend='bullish')
        
        momentum = analyzer.confirm_momentum_1h(df)
        assert momentum in ['strong', 'weak', 'sideways']
    
    def test_micro_trend_analysis(self):
        """Test micro-trend analysis."""
        analyzer = MarketRegimeAnalyzer()
        df = create_test_ohlcv(n_bars=50)
        
        metrics = analyzer.analyze_micro_trends_30m(df)
        assert 'trend_strength' in metrics
        assert 'entry_score' in metrics
        assert 0 <= metrics['trend_strength'] <= 1
        assert 0 <= metrics['entry_score'] <= 1
    
    def test_volatility_classification(self):
        """Test volatility regime classification."""
        analyzer = MarketRegimeAnalyzer()
        
        # High volatility
        df_high = create_test_ohlcv(n_bars=100, volatility='high')
        vol_class, risk_mult = analyzer.classify_volatility_regime(df_high)
        assert vol_class in ['high', 'normal', 'low']
        assert 0.1 <= risk_mult <= 2.0
        
        # Low volatility
        df_low = create_test_ohlcv(n_bars=100, volatility='low')
        vol_class_low, risk_mult_low = analyzer.classify_volatility_regime(df_low)
        assert vol_class_low in ['high', 'normal', 'low']
    
    def test_comprehensive_regime_analysis(self):
        """Test comprehensive multi-timeframe analysis."""
        analyzer = MarketRegimeAnalyzer()
        
        df_30m = create_test_ohlcv(n_bars=100, trend='bullish')
        df_1h = create_test_ohlcv(n_bars=50, trend='bullish')
        df_4h = create_test_ohlcv(n_bars=50, trend='bullish')
        
        regime = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
        
        assert 'trend' in regime
        assert 'momentum' in regime
        assert 'volatility' in regime
        assert 'risk_multiplier' in regime
        assert 'micro_trend_strength' in regime
        assert 'entry_score' in regime


class TestAdaptiveStrategies:
    """Test adaptive strategy implementations."""
    
    def test_adaptive_ob_initialization(self):
        """Test AdaptiveOversoldBounce initialization."""
        cfg = {'rsi_max': 25, 'tp_pct': 0.015}
        strategy = AdaptiveOversoldBounce(cfg)
        assert strategy is not None
        assert strategy.base_cfg == cfg
    
    def test_adaptive_ob_rsi_threshold(self):
        """Test adaptive RSI threshold calculation."""
        cfg = {'rsi_max': 25}
        strategy = AdaptiveOversoldBounce(cfg)
        
        # Bullish regime - more selective
        regime_bullish = {'trend': 'bullish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold_bull = strategy.get_adaptive_rsi_threshold(regime_bullish)
        assert threshold_bull < 25
        
        # Bearish regime - more aggressive
        regime_bearish = {'trend': 'bearish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold_bear = strategy.get_adaptive_rsi_threshold(regime_bearish)
        assert threshold_bear >= 25
    
    def test_adaptive_ob_position_sizing(self):
        """Test adaptive position sizing."""
        cfg = {'rsi_max': 25}
        strategy = AdaptiveOversoldBounce(cfg)
        
        # High volatility - reduce size
        mult_high = strategy.calculate_dynamic_position_size('high', 1.0)
        assert mult_high == 0.5
        
        # Low volatility - increase size
        mult_low = strategy.calculate_dynamic_position_size('low', 1.0)
        assert mult_low == 1.5
        
        # Normal volatility - no change
        mult_normal = strategy.calculate_dynamic_position_size('normal', 1.0)
        assert mult_normal == 1.0
    
    def test_adaptive_ob_signal_generation(self):
        """Test adaptive signal generation."""
        cfg = {'rsi_max': 25, 'tp_pct': 0.015, 'sl_atr_mult': 1.0}
        strategy = AdaptiveOversoldBounce(cfg)
        
        df = create_test_ohlcv(n_bars=100)
        df.loc[df.index[-1], 'rsi'] = 20  # Force oversold
        
        regime_data = {
            'trend': 'bearish',
            'momentum': 'weak',
            'volatility': 'normal',
            'micro_trend_strength': 0.5
        }
        
        signal = strategy.signal(df, regime_data)
        assert signal is not None
        assert signal['side'] == 'buy'
        assert 'position_multiplier' in signal
        assert 'market_regime' in signal
    
    def test_adaptive_str_initialization(self):
        """Test AdaptiveShortTheRip initialization."""
        cfg = {'rsi_min': 65, 'tp_pct': 0.012}
        strategy = AdaptiveShortTheRip(cfg)
        assert strategy is not None
        assert strategy.base_cfg == cfg
    
    def test_adaptive_str_rsi_threshold(self):
        """Test adaptive STR RSI threshold calculation."""
        cfg = {'rsi_min': 65}
        strategy = AdaptiveShortTheRip(cfg)
        
        # Bearish regime - more aggressive
        regime_bearish = {'trend': 'bearish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold_bear = strategy.get_adaptive_rsi_threshold(regime_bearish)
        assert threshold_bear <= 65
        
        # Bullish regime - more selective
        regime_bullish = {'trend': 'bullish', 'momentum': 'strong', 'volatility': 'normal'}
        threshold_bull = strategy.get_adaptive_rsi_threshold(regime_bullish)
        assert threshold_bull >= 65
    
    def test_adaptive_str_signal_generation(self):
        """Test adaptive STR signal generation."""
        cfg = {'rsi_min': 65, 'tp_pct': 0.012, 'sl_atr_mult': 1.2}
        strategy = AdaptiveShortTheRip(cfg)
        
        df_30m = create_test_ohlcv(n_bars=100, trend='bearish')
        df_1h = create_test_ohlcv(n_bars=50, trend='bearish')
        df_30m.loc[df_30m.index[-1], 'rsi'] = 75  # Force overbought
        
        regime_data = {
            'trend': 'bearish',
            'momentum': 'strong',
            'volatility': 'normal',
            'micro_trend_strength': 0.5
        }
        
        signal = strategy.signal(df_30m, df_1h, regime_data)
        assert signal is not None
        assert signal['side'] == 'sell'
        assert 'position_multiplier' in signal


class TestVSTIntelligence:
    """Test VST Market Intelligence system."""
    
    def test_vst_analyzer_initialization(self):
        """Test VST analyzer initialization."""
        # Mock client
        class MockClient:
            pass
        
        client = MockClient()
        analyzer = VSTMarketAnalyzer(client)
        assert analyzer is not None
        assert analyzer.vst_symbol == 'VST/USDT:USDT'
    
    def test_vst_price_pattern_analysis(self):
        """Test VST price pattern analysis."""
        class MockClient:
            pass
        
        analyzer = VSTMarketAnalyzer(MockClient())
        df = create_test_ohlcv(n_bars=100, volatility='high')
        
        patterns = analyzer.analyze_vst_price_patterns(df)
        assert 'volatility_profile' in patterns
        assert 'average_move' in patterns
        assert patterns['volatility_profile'] in ['high', 'moderate', 'low', 'unknown']
    
    def test_vst_parameter_optimization(self):
        """Test VST parameter optimization."""
        class MockClient:
            pass
        
        analyzer = VSTMarketAnalyzer(MockClient())
        
        # Optimize without regime
        params = analyzer.optimize_test_trading_parameters()
        assert 'position_size_mult' in params
        assert params['position_size_mult'] == 0.1  # 10% for testing
        assert params['exchange'] == 'bingx'
        
        # Optimize with regime
        regime = {'trend': 'bearish', 'momentum': 'weak', 'volatility': 'normal'}
        params_regime = analyzer.optimize_test_trading_parameters(regime)
        assert 'ob_rsi_max' in params_regime
    
    def test_vst_performance_monitoring(self):
        """Test VST performance monitoring."""
        class MockClient:
            pass
        
        analyzer = VSTMarketAnalyzer(MockClient())
        
        # Track some trades
        for i in range(5):
            result = {'pnl': 10 if i % 2 == 0 else -5}
            monitoring = analyzer.monitor_vst_performance(result)
        
        assert monitoring is not None
        assert 'metrics' in monitoring
        assert 'recommendations' in monitoring


class TestPerformanceMonitor:
    """Test Real-Time Performance Monitor."""
    
    def test_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = RealTimePerformanceMonitor()
        assert monitor is not None
        assert isinstance(monitor.performance_history, dict)
    
    def test_track_strategy_performance(self):
        """Test strategy performance tracking."""
        monitor = RealTimePerformanceMonitor()
        
        # Track some trades
        for i in range(10):
            result = {'pnl': 10 if i % 2 == 0 else -5}
            metrics = monitor.track_strategy_performance('test_strategy', result)
        
        assert 'test_strategy' in monitor.performance_history
        assert 'metrics' in monitor.performance_history['test_strategy']
        assert metrics.get('trade_count') == 10
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        monitor = RealTimePerformanceMonitor()
        
        # Track trades with known outcomes
        for i in range(20):
            result = {'pnl': 10 if i < 12 else -5}  # 60% win rate
            monitor.track_strategy_performance('test_strategy', result)
        
        metrics = monitor.performance_history['test_strategy']['metrics']
        assert metrics['win_rate'] == 0.6
        assert metrics['trade_count'] == 20
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_parameter_drift_detection(self):
        """Test parameter drift detection."""
        monitor = RealTimePerformanceMonitor()
        
        # Track poor performance
        for i in range(15):
            result = {'pnl': 10 if i < 5 else -10}  # Poor win rate
            monitor.track_strategy_performance('test_strategy', result)
        
        params = {'rsi_max': 25}
        needs_adjustment, reasons = monitor.detect_parameter_drift('test_strategy', params, 0.4)
        
        # Should detect drift due to low win rate
        assert isinstance(needs_adjustment, bool)
        assert isinstance(reasons, list)
    
    def test_optimization_feedback(self):
        """Test optimization feedback generation."""
        monitor = RealTimePerformanceMonitor()
        
        # Track trades
        for i in range(20):
            result = {'pnl': 10 if i < 12 else -5}
            monitor.track_strategy_performance('test_strategy', result)
        
        feedback = monitor.provide_optimization_feedback('test_strategy')
        assert feedback is not None
        assert 'status' in feedback
        assert 'recommendations' in feedback
        assert isinstance(feedback['recommendations'], list)
    
    def test_strategy_summary(self):
        """Test strategy summary retrieval."""
        monitor = RealTimePerformanceMonitor()
        
        # Track some trades
        for i in range(5):
            monitor.track_strategy_performance('test_strategy', {'pnl': 10})
        
        summary = monitor.get_strategy_summary('test_strategy')
        assert summary is not None
        assert summary['strategy'] == 'test_strategy'
        assert 'metrics' in summary
        assert summary['trade_count'] == 5


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_regime_to_adaptive_strategy_flow(self):
        """Test full flow from regime detection to adaptive strategy."""
        # Create regime analyzer
        analyzer = MarketRegimeAnalyzer()
        
        # Create test data
        df_30m = create_test_ohlcv(n_bars=100, trend='bearish')
        df_1h = create_test_ohlcv(n_bars=50, trend='bearish')
        df_4h = create_test_ohlcv(n_bars=50, trend='bearish')
        
        # Analyze regime
        regime = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
        assert regime is not None
        
        # Create adaptive strategy
        cfg = {'rsi_max': 25, 'tp_pct': 0.015, 'sl_atr_mult': 1.0}
        strategy = AdaptiveOversoldBounce(cfg, analyzer)
        
        # Force oversold condition
        df_30m.loc[df_30m.index[-1], 'rsi'] = 20
        
        # Generate signal
        signal = strategy.signal(df_30m, regime)
        
        # Verify adaptive parameters are present
        if signal:
            assert 'position_multiplier' in signal
            assert 'market_regime' in signal
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring with adaptive strategies."""
        monitor = RealTimePerformanceMonitor()
        
        # Simulate trades
        strategy_name = 'adaptive_ob'
        for i in range(15):
            pnl = 15 if i < 10 else -5  # Good performance
            monitor.track_strategy_performance(strategy_name, {'pnl': pnl})
        
        # Get feedback
        feedback = monitor.provide_optimization_feedback(strategy_name)
        assert feedback['status'] == 'analyzed'
        assert len(feedback['recommendations']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
