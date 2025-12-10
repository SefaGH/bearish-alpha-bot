"""
Advanced Position Sizing Algorithms.
Implements Kelly Criterion, volatility-adjusted, and regime-based position sizing.
"""

import logging
import numpy as np
from typing import Dict, Union

from src.core.interfaces import PositionSizingProtocol

logger = logging.getLogger(__name__)


class AdvancedPositionSizing(PositionSizingProtocol):
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
            
            # Get regime_weight for soft-weighting (default to 1.0 if not available)
            regime_weight = float(signal.get('regime_weight', 1.0))
            
            # Regime multiplier with soft-weighting
            regime_multiplier = market_regime.get('risk_multiplier', 1.0)
            # Apply soft-weighted regime adjustment: interpolate between 1.0 and regime_multiplier
            regime_adjustment = 1.0 + (regime_multiplier - 1.0) * regime_weight
            
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
            
            # Combine all factors with soft-weighted regime
            adjusted_size = base_size * regime_adjustment * trend_bonus * vol_adjustment * perf_multiplier
            
            logger.debug(f"Regime-based sizing: {adjusted_size:.4f} "
                        f"(regime_mult={regime_multiplier:.2f}, regime_weight={regime_weight:.2f}, "
                        f"regime_adj={regime_adjustment:.2f}, trend={trend_bonus:.2f}, "
                        f"vol={vol_adjustment:.2f}, perf={perf_multiplier:.2f})")
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error in regime-based sizing: {e}")
            return 0.0
    
    async def calculate_optimal_size(self, signal: Dict, method: str = 'fixed_risk_capped', return_signal: bool = False, **kwargs) -> Union[Dict, float]:
        """
        Calculate the uncapped, risk-based proposal for a position size.

        The RiskManager now owns all cap enforcement, so this method only computes
        the proposed notional based on risk inputs and annotates the signal.
        """
        try:
            # Allow explicit risk overrides for backward compatibility
            risk_per_trade_override = kwargs.pop('risk_per_trade', None)

            # Get required parameters
            symbol = signal.get('symbol', 'UNKNOWN')
            entry_price = signal.get('entry', 0) or signal.get('entry_price', 0)
            stop_loss = signal.get('stop', 0) or signal.get('stop_loss', 0)
            
            # Calculate stop loss percentage
            if entry_price > 0 and stop_loss > 0:
                stop_pct = abs(entry_price - stop_loss) / entry_price
            else:
                logger.warning(f"[SIZING] Missing price data for {symbol}, cannot calculate position size")
                # Return signal with zero position to indicate sizing failure
                signal['amount'] = 0
                signal['notional'] = 0
                signal['position_size'] = 0
                return signal
            
            # Get portfolio state from RiskManager
            portfolio = self.risk_manager.get_portfolio_summary()
            capital = float(portfolio.get('portfolio_value', 0))
            
            if capital <= 0:
                logger.error(f"[SIZING] Invalid capital: {capital}, cannot calculate position size")
                # Return signal with zero position to indicate sizing failure
                signal['amount'] = 0
                signal['notional'] = 0
                signal['position_size'] = 0
                return signal
            
            # Get configuration (use config from risk_manager)
            config = getattr(self.risk_manager, 'config', {})

            # Use configuration with GitHub Variables priority
            risk_pct = float(config.get('per_trade_risk_pct', 0.01))
            risk_cap = config.get('risk_usd_cap')
            leverage = float(signal.get('leverage', config.get('leverage_default', 5)))

            # Risk-based calculation only (no caps here)
            base_risk_usd = risk_per_trade_override if risk_per_trade_override is not None else capital * risk_pct
            if risk_cap:
                base_risk_usd = min(base_risk_usd, float(risk_cap))

            risk_based_notional = base_risk_usd / stop_pct if stop_pct > 0 else 0
            proposed_notional = risk_based_notional
            proposed_amount = proposed_notional / entry_price if entry_price > 0 else 0

            sizing_meta = {
                'method': method,
                'capital': capital,
                'risk_pct': risk_pct,
                'stop_pct': stop_pct,
                'base_risk_usd': base_risk_usd,
                'risk_based_notional': risk_based_notional,
                'proposed_notional': proposed_notional,
                'capped': False,
                'cap_applied_by': None,
            }

            logger.info(f"📊 [SIZING-PROPOSED] {symbol}")
            logger.info(f"   Capital: ${capital:.2f}")
            logger.info(f"   Risk-based notional: ${risk_based_notional:.2f}")
            logger.info("   ⚠️ Limits will be applied by RiskManager.apply_position_limits()")

            signal['amount'] = proposed_amount
            signal['notional'] = proposed_notional
            signal['position_size'] = proposed_amount
            signal['leverage'] = leverage
            signal['sizing_meta'] = sizing_meta

            return signal if return_signal else proposed_amount
            
        except Exception as e:
            logger.error(f"[SIZING] Error calculating size for {signal.get('symbol')}: {e}", exc_info=True)
            return signal if return_signal else 0.0
