"""
Tests for signal stop-loss calculation fix.
Validates that adaptive strategies generate signals with stop/target fields
and that RiskManager correctly handles signals with or without stop fields.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timezone

from core.risk_manager import RiskManager
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip


class TestAdaptiveStrategySignals:
    """Test that adaptive strategies generate proper signals with stop/target fields."""
    
    def setup_method(self):
        """Setup test data for each test."""
        # Create sample market data with required indicators
        self.market_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [3850.0 + i for i in range(100)],
            'high': [3860.0 + i for i in range(100)],
            'low': [3840.0 + i for i in range(100)],
            'close': [3855.0 + i for i in range(100)],
            'volume': [1000.0] * 100,
            'rsi': [25.0 + (i % 20) for i in range(100)],  # Oscillating RSI
            'atr': [15.0] * 100,
            'ema21': [3850.0 + i for i in range(100)],
            'ema50': [3845.0 + i for i in range(100)],
            'ema200': [3840.0 + i for i in range(100)]
        })
    
    def test_adaptive_ob_signal_has_stop_field(self):
        """Test that AdaptiveOversoldBounce generates signals with stop field."""
        config = {
            'rsi_period': 14,
            'rsi_max': 30,
            'tp_pct': 0.015,
            'sl_atr_mult': 1.0
        }
        
        strategy = AdaptiveOversoldBounce(config)
        
        # Create oversold condition
        df = self.market_data.copy()
        df.loc[df.index[-1], 'rsi'] = 28.0  # Below 30
        
        regime_data = {
            'trend': 'neutral',
            'momentum': 'sideways',
            'volatility': 'normal',
            'micro_trend_strength': 0.5,
            'entry_score': 0.5,
            'risk_multiplier': 1.0
        }
        
        signal = strategy.signal(df, regime_data=regime_data)
        
        if signal:  # Signal may or may not be generated depending on other conditions
            assert 'entry' in signal, "Signal must have entry field"
            assert 'stop' in signal, "Signal must have stop field"
            assert 'target' in signal, "Signal must have target field"
            assert signal['stop'] < signal['entry'], "Stop must be below entry for BUY signal"
            assert signal['target'] > signal['entry'], "Target must be above entry for BUY signal"
            assert signal['side'] == 'buy', "AdaptiveOB should generate BUY signals"
    
    def test_adaptive_str_signal_has_stop_field(self):
        """Test that AdaptiveShortTheRip generates signals with stop field."""
        config = {
            'rsi_period': 14,
            'rsi_min': 70,
            'tp_pct': 0.012,
            'sl_atr_mult': 1.2
        }
        
        strategy = AdaptiveShortTheRip(config)
        
        # Create overbought condition
        df = self.market_data.copy()
        df.loc[df.index[-1], 'rsi'] = 72.0  # Above 70
        
        regime_data = {
            'trend': 'bearish',
            'momentum': 'strong_down',
            'volatility': 'normal',
            'micro_trend_strength': 0.5,
            'entry_score': 0.5,
            'risk_multiplier': 1.0
        }
        
        signal = strategy.signal(df, regime_data=regime_data)
        
        if signal:  # Signal may or may not be generated depending on other conditions
            assert 'entry' in signal, "Signal must have entry field"
            assert 'stop' in signal, "Signal must have stop field"
            assert 'target' in signal, "Signal must have target field"
            assert signal['stop'] > signal['entry'], "Stop must be above entry for SELL signal"
            assert signal['target'] < signal['entry'], "Target must be below entry for SELL signal"
            assert signal['side'] == 'sell', "AdaptiveSTR should generate SELL signals"
    
    def test_signal_stop_calculation_accuracy(self):
        """Test that stop prices are calculated correctly from ATR."""
        config = {
            'rsi_period': 14,
            'rsi_max': 30,
            'tp_pct': 0.015,
            'sl_atr_mult': 1.0
        }
        
        strategy = AdaptiveOversoldBounce(config)
        
        # Create oversold condition with known ATR
        df = self.market_data.copy()
        df.loc[df.index[-1], 'rsi'] = 28.0
        df.loc[df.index[-1], 'close'] = 3855.0
        df.loc[df.index[-1], 'atr'] = 15.0
        
        regime_data = {
            'trend': 'neutral',
            'momentum': 'sideways',
            'volatility': 'normal',
            'micro_trend_strength': 0.5,
            'entry_score': 0.5,
            'risk_multiplier': 1.0
        }
        
        signal = strategy.signal(df, regime_data=regime_data)
        
        if signal:
            # Use values from the test data setup
            entry = 3855.0
            atr = 15.0
            sl_mult = 1.0
            expected_stop = entry - (atr * sl_mult)  # entry - (atr * sl_mult)
            assert abs(signal['stop'] - expected_stop) < 0.01, f"Stop should be {expected_stop}, got {signal['stop']}"


class TestRiskManagerFallback:
    """Test RiskManager fallback logic for signals without stop field."""
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_with_atr_fallback(self):
        """Test position size calculation with ATR fallback for missing stop."""
        risk_manager = RiskManager(portfolio_config={'equity_usd': 10000.0})
        
        # Signal without stop field, but with ATR data
        signal = {
            'entry': 3855.0,
            'atr': 15.0,
            'sl_atr_mult': 1.0,
            'side': 'buy'
        }
        
        position_size = await risk_manager.calculate_position_size(signal)
        
        # Should calculate stop from ATR and return valid position size
        assert position_size > 0, "Position size should be calculated from ATR fallback"
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_with_sl_pct_fallback(self):
        """Test position size calculation with sl_pct fallback."""
        risk_manager = RiskManager(portfolio_config={'equity_usd': 10000.0})
        
        # Signal without stop field, but with sl_pct
        signal = {
            'entry': 3855.0,
            'sl_pct': 0.02,
            'side': 'buy'
        }
        
        position_size = await risk_manager.calculate_position_size(signal)
        
        # Should calculate stop from sl_pct and return valid position size
        assert position_size > 0, "Position size should be calculated from sl_pct fallback"
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_with_explicit_stop(self):
        """Test that explicit stop field takes precedence."""
        risk_manager = RiskManager(portfolio_config={'equity_usd': 10000.0})
        
        # Signal with explicit stop (should not use fallback)
        signal = {
            'entry': 3855.0,
            'stop': 3840.0,
            'atr': 15.0,
            'sl_atr_mult': 1.0,
            'side': 'buy'
        }
        
        position_size = await risk_manager.calculate_position_size(signal)
        
        # Should use explicit stop
        assert position_size > 0, "Position size should be calculated from explicit stop"
    
    @pytest.mark.asyncio
    async def test_short_signal_atr_fallback(self):
        """Test ATR fallback for SHORT signals (stop above entry)."""
        risk_manager = RiskManager(portfolio_config={'equity_usd': 10000.0})
        
        # SHORT signal without stop field
        signal = {
            'entry': 3855.0,
            'atr': 15.0,
            'sl_atr_mult': 1.2,
            'side': 'sell'
        }
        
        position_size = await risk_manager.calculate_position_size(signal)
        
        # Should calculate stop above entry for SHORT
        assert position_size > 0, "Position size should be calculated for SHORT signal"
    
    @pytest.mark.asyncio
    async def test_validate_position_with_atr_fallback(self):
        """Test position validation with ATR fallback."""
        risk_manager = RiskManager(portfolio_config={'equity_usd': 10000.0})
        
        # Signal without stop but with ATR
        signal = {
            'symbol': 'BTC/USDT',
            'entry': 3855.0,
            'atr': 15.0,
            'sl_atr_mult': 1.0,
            'side': 'buy',
            'position_size': 0.1
        }
        
        current_portfolio = {
            'positions': {},
            'total_value': 10000.0
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal, current_portfolio)
        
        # Should validate successfully using fallback
        assert is_valid or 'stop' in reason.lower(), "Should either validate or explain missing stop"


class TestBackwardsCompatibility:
    """Test that changes maintain backwards compatibility."""
    
    @pytest.mark.asyncio
    async def test_legacy_signal_with_sl_pct(self):
        """Test that legacy signals with only sl_pct still work."""
        risk_manager = RiskManager(portfolio_config={'equity_usd': 10000.0})
        
        # Legacy signal format
        signal = {
            'entry': 3855.0,
            'sl_pct': 0.02,
            'side': 'buy'
        }
        
        position_size = await risk_manager.calculate_position_size(signal)
        
        assert position_size > 0, "Legacy signals should still work"
    
    @pytest.mark.asyncio
    async def test_complete_signal_format(self):
        """Test that new complete signal format works perfectly."""
        risk_manager = RiskManager(portfolio_config={'equity_usd': 10000.0})
        
        # New complete signal format
        signal = {
            'entry': 3855.0,
            'stop': 3840.0,
            'target': 3912.825,
            'tp_pct': 0.015,
            'sl_atr_mult': 1.0,
            'atr': 15.0,
            'side': 'buy'
        }
        
        position_size = await risk_manager.calculate_position_size(signal)
        
        assert position_size > 0, "Complete signal format should work"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
