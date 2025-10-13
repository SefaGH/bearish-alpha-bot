"""
Advanced Position Sizing Algorithms.
Implements Kelly Criterion, volatility-adjusted, and regime-based position sizing.
"""

import logging
import numpy as np
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)


class AdvancedPositionSizing:
    """Advanced position sizing algorithms for optimal capital allocation."""
    
    def __init__(self, risk_manager):
        """
        Initialize position sizing engine.
        
        Args:
            risk_manager: Risk manager instance for portfolio state
        """
        self.risk_manager = risk_manager
        
        # Available sizing methods
        self.sizing_methods = {
            'kelly': self._kelly_criterion,
            'fixed_risk': self._fixed_risk_sizing,
            'volatility_adjusted': self._volatility_adjusted_sizing,
            'regime_based': self._regime_based_sizing
        }
        
        logger.info("AdvancedPositionSizing initialized")
    
    def _kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float,
                        portfolio_value: float, **kwargs) -> float:
        """
        Kelly Criterion position sizing.
        
        Optimal fraction calculation for maximizing long-term growth.
        Uses fractional Kelly (50%) for safety.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            portfolio_value: Current portfolio value
            **kwargs: Additional parameters
            
        Returns:
            Position size as fraction of portfolio (0-1)
        """
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                logger.warning("Invalid Kelly inputs, using conservative sizing")
                return 0.01  # 1% default
            
            # Kelly formula: f = (p * b - q) / b
            # where p = win_rate, q = 1-win_rate, b = avg_win/avg_loss
            b = avg_win / avg_loss  # Win/loss ratio
            q = 1 - win_rate
            
            kelly_fraction = (win_rate * b - q) / b
            
            # Apply fractional Kelly (50% for safety)
            fractional_kelly = kwargs.get('kelly_fraction', 0.5)
            kelly_fraction *= fractional_kelly
            
            # Clamp to reasonable bounds (0.5% - 10%)
            kelly_fraction = max(0.005, min(kelly_fraction, 0.10))
            
            logger.debug(f"Kelly criterion: {kelly_fraction:.4f} "
                        f"(win_rate={win_rate:.2f}, b={b:.2f})")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error in Kelly calculation: {e}")
            return 0.01
    
    def _fixed_risk_sizing(self, risk_per_trade: float, entry_price: float,
                          stop_loss: float, **kwargs) -> float:
        """
        Fixed risk position sizing.
        
        Position size based on fixed dollar risk amount.
        
        Args:
            risk_per_trade: Dollar amount to risk
            entry_price: Entry price
            stop_loss: Stop loss price
            **kwargs: Additional parameters
            
        Returns:
            Position size in base currency units
        """
        try:
            risk_distance = abs(entry_price - stop_loss)
            
            if risk_distance <= 0:
                logger.warning("Invalid risk distance for fixed sizing")
                return 0.0
            
            position_size = risk_per_trade / risk_distance
            
            logger.debug(f"Fixed risk sizing: {position_size:.4f} "
                        f"(risk=${risk_per_trade:.2f}, distance={risk_distance:.4f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error in fixed risk sizing: {e}")
            return 0.0
    
    def _volatility_adjusted_sizing(self, signal: Dict, market_volatility: float,
                                    target_risk: float, **kwargs) -> float:
        """
        Volatility-adjusted position sizing.
        
        Scales position size inversely with volatility using ATR.
        
        Args:
            signal: Trading signal with entry, stop, ATR
            market_volatility: Current market volatility (ATR)
            target_risk: Target risk amount
            **kwargs: Additional parameters
            
        Returns:
            Position size in base currency units
        """
        try:
            entry_price = signal.get('entry', 0)
            atr = signal.get('atr', market_volatility)
            
            if atr <= 0 or entry_price <= 0:
                logger.warning("Invalid inputs for volatility sizing")
                return 0.0
            
            # Base position size from target risk
            base_size = target_risk / atr
            
            # Volatility adjustment factor
            # Reduce size in high volatility, increase in low volatility
            avg_volatility = kwargs.get('avg_volatility', atr)
            vol_ratio = avg_volatility / atr if atr > 0 else 1.0
            
            # Clamp adjustment (0.5x - 2x)
            vol_ratio = max(0.5, min(vol_ratio, 2.0))
            
            adjusted_size = base_size * vol_ratio
            
            logger.debug(f"Volatility adjusted sizing: {adjusted_size:.4f} "
                        f"(ATR={atr:.4f}, vol_ratio={vol_ratio:.2f})")
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error in volatility sizing: {e}")
            return 0.0
    
    def _regime_based_sizing(self, signal: Dict, market_regime: Dict,
                            performance_history: Dict = None, **kwargs) -> float:
        """
        Market regime-aware position sizing.
        
        Adjusts position size based on market regime from Phase 2.
        
        Args:
            signal: Trading signal
            market_regime: Market regime analysis from Phase 2
            performance_history: Historical performance data
            **kwargs: Additional parameters
            
        Returns:
            Position size in base currency units
        """
        try:
            # Base sizing from fixed risk
            entry_price = signal.get('entry', 0)
            stop_loss = signal.get('stop', 0)
            base_risk = kwargs.get('base_risk', 100)  # Default $100 risk
            
            if entry_price <= 0 or stop_loss <= 0:
                return 0.0
            
            base_size = self._fixed_risk_sizing(base_risk, entry_price, stop_loss)
            
            # Regime multiplier
            regime_multiplier = market_regime.get('risk_multiplier', 1.0)
            
            # Trend alignment
            trend = market_regime.get('trend', 'neutral')
            signal_side = signal.get('side', 'long')
            
            trend_bonus = 1.0
            if (trend == 'bullish' and signal_side == 'long') or \
               (trend == 'bearish' and signal_side == 'short'):
                trend_bonus = 1.2  # 20% bonus for trend alignment
            
            # Volatility adjustment
            volatility = market_regime.get('volatility', 'normal')
            vol_adjustment = {
                'low': 1.2,
                'normal': 1.0,
                'high': 0.7
            }.get(volatility, 1.0)
            
            # Performance-based adjustment
            perf_multiplier = 1.0
            if performance_history:
                win_rate = performance_history.get('win_rate', 0.5)
                if win_rate > 0.6:
                    perf_multiplier = 1.1
                elif win_rate < 0.4:
                    perf_multiplier = 0.8
            
            # Combine all factors
            adjusted_size = base_size * regime_multiplier * trend_bonus * vol_adjustment * perf_multiplier
            
            logger.debug(f"Regime-based sizing: {adjusted_size:.4f} "
                        f"(regime={regime_multiplier:.2f}, trend={trend_bonus:.2f}, "
                        f"vol={vol_adjustment:.2f}, perf={perf_multiplier:.2f})")
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error in regime-based sizing: {e}")
            return 0.0
    
    async def calculate_optimal_size(self, signal: Dict, method: str = 'kelly',
                                    **kwargs) -> float:
        """
        Calculate optimal position size using specified method.
        
        Args:
            signal: Trading signal with entry, stop, target, etc.
            method: Sizing method ('kelly', 'fixed_risk', 'volatility_adjusted', 'regime_based')
            **kwargs: Method-specific parameters
            
        Returns:
            Optimal position size
        """
        try:
            if method not in self.sizing_methods:
                logger.warning(f"Unknown sizing method '{method}', using fixed_risk")
                method = 'fixed_risk'
            
            sizing_func = self.sizing_methods[method]
            
            # Prepare parameters based on method
            if method == 'kelly':
                # Need performance history for Kelly
                performance_history = kwargs.get('performance_history', {})
                win_rate = performance_history.get('win_rate', 0.5)
                avg_win = performance_history.get('avg_win', 50)
                avg_loss = performance_history.get('avg_loss', 25)
                portfolio_value = self.risk_manager.portfolio_value
                
                kelly_fraction = sizing_func(win_rate, avg_win, avg_loss, portfolio_value, **kwargs)
                position_size = (kelly_fraction * portfolio_value) / signal.get('entry', 1)
                
            elif method == 'fixed_risk':
                risk_per_trade = kwargs.get('risk_per_trade',
                                           self.risk_manager.portfolio_value * 0.02)
                position_size = sizing_func(risk_per_trade, signal.get('entry', 0),
                                          signal.get('stop', 0), **kwargs)
                
            elif method == 'volatility_adjusted':
                target_risk = kwargs.get('target_risk',
                                        self.risk_manager.portfolio_value * 0.02)
                market_volatility = kwargs.get('market_volatility', signal.get('atr', 1))
                position_size = sizing_func(signal, market_volatility, target_risk, **kwargs)
                
            elif method == 'regime_based':
                market_regime = kwargs.get('market_regime', {})
                performance_history = kwargs.get('performance_history', {})
                position_size = sizing_func(signal, market_regime, performance_history, **kwargs)
            
            else:
                position_size = 0.0
            
            # Validate against risk limits
            is_valid, reason, risk_metrics = await self.risk_manager.validate_new_position(
                {**signal, 'position_size': position_size},
                kwargs.get('current_portfolio', {})
            )
            
            if not is_valid:
                logger.warning(f"Position size validation failed: {reason}")
                # Try with reduced size
                position_size *= 0.5
            
            logger.info(f"Optimal position size ({method}): {position_size:.4f}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal size: {e}")
            return 0.0
