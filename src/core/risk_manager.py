"""
Comprehensive Risk Management Engine.
Provides portfolio-level risk management, position validation, and capital allocation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np

# Dual import strategy: supports both script and package execution
try:
    # Absolute import for script execution
    from src.utils.pnl_calculator import calculate_unrealized_pnl
except ImportError:
    # Relative import for package context
    from ..utils.pnl_calculator import calculate_unrealized_pnl

logger = logging.getLogger(__name__)


class RiskManager:
    """Comprehensive risk management engine for multi-strategy portfolio."""
    
    def __init__(self, portfolio_config: Dict, websocket_manager=None, performance_monitor=None):
        """Initialize risk manager."""
        self.portfolio_config = portfolio_config
        self.ws_manager = websocket_manager
        self.performance_monitor = performance_monitor
        
        # Risk limits
        self.risk_limits = {
            'max_portfolio_risk': portfolio_config.get('max_portfolio_risk', 0.02),
            'max_position_size': portfolio_config.get('max_position_size', 0.10),
            'max_drawdown': portfolio_config.get('max_drawdown', 0.15),
            'max_correlation': portfolio_config.get('max_correlation', 0.70),
        }
        
        # Active positions tracking
        self.active_positions = {}
        
        # Portfolio state - DÜZELTME
        self.portfolio_value = float(portfolio_config.get('equity_usd', 100))  # 100 default!
        self.current_drawdown = 0.0
        self.peak_portfolio_value = self.portfolio_value
        
        # DEBUG LOG EKLE
        logger.info(f"RiskManager initialized with portfolio value: ${self.portfolio_value:.2f}")
        logger.info(f"Config received: {portfolio_config}")  # Debug için
        logger.info(f"Risk limits: {self.risk_limits}")
    
    def set_risk_limits(self, max_portfolio_risk: float = 0.02, max_position_size: float = 0.10,
                       max_drawdown: float = 0.15, max_correlation: float = 0.7):
        """
        Configure portfolio-level risk limits.
        
        Args:
            max_portfolio_risk: Maximum portfolio risk per trade (default 2%)
            max_position_size: Maximum single position size (default 10% of portfolio)
            max_drawdown: Maximum allowed drawdown (default 15%)
            max_correlation: Maximum correlation between positions (default 70%)
        """
        self.risk_limits = {
            'max_portfolio_risk': max_portfolio_risk,
            'max_position_size': max_position_size,
            'max_drawdown': max_drawdown,
            'max_correlation': max_correlation,
        }
        logger.info(f"Risk limits updated: {self.risk_limits}")
    
    async def validate_new_position(self, signal: Dict, current_portfolio: Dict) -> Tuple[bool, str, Dict]:
        """
        Validate if new position meets risk criteria.
        
        Args:
            signal: Trading signal with entry, stop, size, etc.
            current_portfolio: Current portfolio state with positions and metrics
            
        Returns:
            Tuple of (is_valid, reason, risk_metrics)
        """
        try:
            # Extract signal details
            symbol = signal.get('symbol', 'UNKNOWN')
            position_size = signal.get('position_size', 0)
            entry_price = signal.get('entry', 0)
            stop_loss = signal.get('stop', 0)
            
            # Fallback: Calculate stop if missing (same logic as calculate_position_size)
            if not stop_loss and signal.get('sl_atr_mult') and signal.get('atr'):
                atr = signal['atr']
                sl_mult = signal['sl_atr_mult']
                side = signal.get('side', 'buy')
                
                if side in ['buy', 'long']:
                    stop_loss = entry_price - (atr * sl_mult)
                else:
                    stop_loss = entry_price + (atr * sl_mult)
                
                logger.info(f"📊 [RISK-VALIDATE] Calculated ATR-based stop: {stop_loss:.2f}")
            
            if not stop_loss and signal.get('sl_pct'):
                sl_pct = signal['sl_pct']
                side = signal.get('side', 'buy')
                
                if side in ['buy', 'long']:
                    stop_loss = entry_price * (1 - sl_pct)
                else:
                    stop_loss = entry_price * (1 + sl_pct)
                
                logger.info(f"📊 [RISK-VALIDATE] Calculated percentage-based stop: {stop_loss:.2f}")
            
            logger.debug(f"🛡️ [RISK-CALC] Validating position for {symbol}")
            logger.debug(f"🛡️ [RISK-CALC] Portfolio value: ${self.portfolio_value:.2f}")
            
            risk_metrics = {}
            
            # 1. Position size validation
            position_value = position_size * entry_price
            max_position_value = self.portfolio_value * self.risk_limits['max_position_size']
            
            logger.debug(f"🛡️ [RISK-CALC] Position size check: ${position_value:.2f} vs ${max_position_value:.2f} max")
            
            if position_value > max_position_value:
                logger.debug(f"🛡️ [RISK-CALC] REJECTED: Position size exceeds limit")
                return (False, f"Position size ${position_value:.2f} exceeds max ${max_position_value:.2f}", risk_metrics)
            
            risk_metrics['position_value'] = position_value
            risk_metrics['max_position_value'] = max_position_value
            risk_metrics['position_size_pct'] = position_value / self.portfolio_value
            
            # 2. Portfolio risk check
            risk_amount = abs(entry_price - stop_loss) * position_size
            max_risk = self.portfolio_value * self.risk_limits['max_portfolio_risk']
            
            risk_pct = risk_amount / self.portfolio_value
            logger.debug(f"🛡️ [RISK-CALC] Risk per trade: {risk_pct:.2%} (limit: {self.risk_limits['max_portfolio_risk']:.2%})")
            
            if risk_amount > max_risk:
                logger.debug(f"🛡️ [RISK-CALC] REJECTED: Risk amount exceeds limit")
                return (False, f"Risk amount ${risk_amount:.2f} exceeds max ${max_risk:.2f}", risk_metrics)
            
            risk_metrics['risk_amount'] = risk_amount
            risk_metrics['max_risk_amount'] = max_risk
            risk_metrics['risk_pct'] = risk_pct
            
            # 3. Risk/reward ratio assessment
            target_price = signal.get('target', entry_price * 1.02)  # Default 2% target
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(target_price - entry_price)
            
            if risk_distance > 0:
                risk_reward_ratio = reward_distance / risk_distance
                risk_metrics['risk_reward_ratio'] = risk_reward_ratio
                logger.debug(f"🛡️ [RISK-CALC] Risk/Reward ratio: {risk_reward_ratio:.2f}")
                
                if risk_reward_ratio < 1.5:
                    logger.debug(f"🛡️ [RISK-CALC] REJECTED: R/R ratio too low")
                    return (False, f"Risk/reward ratio {risk_reward_ratio:.2f} below minimum 1.5", risk_metrics)
            
            # 4. Drawdown check
            if self.current_drawdown > self.risk_limits['max_drawdown']:
                logger.debug(f"🛡️ [RISK-CALC] REJECTED: Drawdown limit exceeded")
                return (False, f"Current drawdown {self.current_drawdown:.2%} exceeds max {self.risk_limits['max_drawdown']:.2%}", risk_metrics)
            
            risk_metrics['current_drawdown'] = self.current_drawdown
            risk_metrics['max_drawdown'] = self.risk_limits['max_drawdown']
            
            # 5. Portfolio heat (total risk exposure)
            total_risk = sum(pos.get('risk_amount', 0) for pos in self.active_positions.values())
            total_risk += risk_amount
            portfolio_heat = total_risk / self.portfolio_value
            
            logger.debug(f"🛡️ [RISK-CALC] Portfolio heat: {portfolio_heat:.2%} (limit: 10%)")
            
            if portfolio_heat > 0.10:  # Max 10% total portfolio heat
                logger.debug(f"🛡️ [RISK-CALC] REJECTED: Portfolio heat too high")
                return (False, f"Portfolio heat {portfolio_heat:.2%} would exceed 10%", risk_metrics)
            
            risk_metrics['portfolio_heat'] = portfolio_heat
            
            # 6. Market condition alignment (if performance monitor available)
            if self.performance_monitor:
                strategy_name = signal.get('strategy', 'unknown')
                summary = self.performance_monitor.get_strategy_summary(strategy_name)
                metrics = summary.get('metrics', {})
                
                win_rate = metrics.get('win_rate', 0.5)
                if win_rate < 0.35:  # Very low win rate
                    logger.debug(f"🛡️ [RISK-CALC] REJECTED: Strategy win rate too low")
                    return (False, f"Strategy win rate {win_rate:.2%} too low", risk_metrics)
                
                risk_metrics['strategy_win_rate'] = win_rate
            
            logger.debug(f"🛡️ [RISK-CALC] APPROVED: All risk checks passed")
            logger.info(f"Position validation PASSED for {symbol}: {risk_metrics}")
            return (True, "Position validated successfully", risk_metrics)
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return (False, f"Validation error: {str(e)}", {})
    
    async def monitor_position_risk(self, position_id: str) -> Dict[str, Any]:
        """
        Real-time position risk monitoring.
        
        Args:
            position_id: Unique position identifier
            
        Returns:
            Dictionary with position risk metrics and alerts
        """
        try:
            if position_id not in self.active_positions:
                return {'status': 'not_found', 'alerts': []}
            
            position = self.active_positions[position_id]
            alerts = []
            
            # Current position state
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', entry_price)
            stop_loss = position.get('stop_loss', 0)
            position_size = position.get('size', 0)
            
            # Calculate unrealized P&L
            unrealized_pnl = calculate_unrealized_pnl(
                position.get('side', 'long'), entry_price, current_price, position_size
            )
            
            position['unrealized_pnl'] = unrealized_pnl
            
            # Stop-loss breach check
            if position.get('side') == 'long' and current_price <= stop_loss:
                alerts.append({
                    'type': 'stop_loss_breach',
                    'severity': 'high',
                    'message': f"Stop loss breached: {current_price} <= {stop_loss}"
                })
            elif position.get('side') == 'short' and current_price >= stop_loss:
                alerts.append({
                    'type': 'stop_loss_breach',
                    'severity': 'high',
                    'message': f"Stop loss breached: {current_price} >= {stop_loss}"
                })
            
            # Large unrealized loss check
            loss_threshold = self.portfolio_value * self.risk_limits['max_portfolio_risk']
            if unrealized_pnl < -loss_threshold:
                alerts.append({
                    'type': 'large_loss',
                    'severity': 'high',
                    'message': f"Unrealized loss ${unrealized_pnl:.2f} exceeds threshold ${loss_threshold:.2f}"
                })
            
            # Time-based exit (if position held too long)
            entry_time = position.get('entry_time')
            if entry_time:
                hold_duration = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                max_hold_hours = position.get('max_hold_hours', 72)
                
                if hold_duration > max_hold_hours:
                    alerts.append({
                        'type': 'time_limit',
                        'severity': 'medium',
                        'message': f"Position held for {hold_duration:.1f}h, exceeds {max_hold_hours}h"
                    })
            
            return {
                'status': 'active',
                'position_id': position_id,
                'unrealized_pnl': unrealized_pnl,
                'current_price': current_price,
                'alerts': alerts
            }
            
        except Exception as e:
            logger.error(f"Error monitoring position {position_id}: {e}")
            return {'status': 'error', 'message': str(e), 'alerts': []}
    
    async def calculate_position_size(self, signal: Dict, market_regime: Dict = None,
                                     portfolio_state: Dict = None) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            signal: Trading signal with entry, stop, target
            market_regime: Market regime data from Phase 2 (optional)
            portfolio_state: Current portfolio state (optional)
            
        Returns:
            Optimal position size
        """
        try:
            entry_price = signal.get('entry', 0)
            stop_loss = signal.get('stop', 0)
            
            # Fallback 1: Calculate from ATR if stop missing
            if not stop_loss and signal.get('sl_atr_mult') and signal.get('atr'):
                atr = signal['atr']
                sl_mult = signal['sl_atr_mult']
                side = signal.get('side', 'buy')
                
                if side in ['buy', 'long']:
                    stop_loss = entry_price - (atr * sl_mult)
                else:  # sell/short
                    stop_loss = entry_price + (atr * sl_mult)
                
                logger.info(f"📊 [RISK] Calculated ATR-based stop: {stop_loss:.2f} (ATR={atr:.2f}, mult={sl_mult})")
            
            # Fallback 2: Calculate from sl_pct if still missing
            if not stop_loss and signal.get('sl_pct'):
                sl_pct = signal['sl_pct']
                side = signal.get('side', 'buy')
                
                if side in ['buy', 'long']:
                    stop_loss = entry_price * (1 - sl_pct)
                else:
                    stop_loss = entry_price * (1 + sl_pct)
                
                logger.info(f"📊 [RISK] Calculated percentage-based stop: {stop_loss:.2f} (pct={sl_pct:.2%})")
            
            if entry_price <= 0 or stop_loss <= 0:
                logger.warning("Invalid entry or stop price after all fallbacks")
                return 0.0
            
            # Base risk amount
            risk_per_trade = self.portfolio_value * self.risk_limits['max_portfolio_risk']
            
            # Adjust for market regime
            if market_regime:
                risk_multiplier = market_regime.get('risk_multiplier', 1.0)
                risk_per_trade *= risk_multiplier
                logger.debug(f"Risk adjusted by regime multiplier: {risk_multiplier:.2f}")
            
            # Adjust for strategy performance
            if self.performance_monitor and signal.get('strategy'):
                strategy_name = signal['strategy']
                summary = self.performance_monitor.get_strategy_summary(strategy_name)
                metrics = summary.get('metrics', {})
                
                win_rate = metrics.get('win_rate', 0.5)
                sharpe = metrics.get('sharpe_ratio', 0)
                
                # Reduce size for poor performing strategies
                if win_rate < 0.4:
                    risk_per_trade *= 0.5
                    logger.debug("Risk reduced due to low win rate")
                elif win_rate > 0.6 and sharpe > 1.0:
                    risk_per_trade *= 1.2
                    logger.debug("Risk increased due to good performance")
            
            # Calculate position size
            risk_distance = abs(entry_price - stop_loss)
            position_size = risk_per_trade / risk_distance
            
            # Apply maximum position size limit
            max_position_value = self.portfolio_value * self.risk_limits['max_position_size']
            max_size_by_limit = max_position_value / entry_price
            position_size = min(position_size, max_size_by_limit)
            
            logger.info(f"Calculated position size: {position_size:.4f} (risk: ${risk_per_trade:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def register_position(self, position_id: str, position_data: Dict):
        """
        Register a new active position.
        
        Args:
            position_id: Unique position identifier
            position_data: Position data including entry, stop, size, etc.
        """
        self.active_positions[position_id] = {
            **position_data,
            'entry_time': datetime.now(timezone.utc),
            'current_price': position_data.get('entry_price', 0)
        }
        logger.info(f"Position registered: {position_id} - {position_data.get('symbol', 'UNKNOWN')}")
    
    def close_position(self, position_id: str, exit_price: float, realized_pnl: float):
        """
        Close and remove a position.
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            realized_pnl: Realized profit/loss
        """
        if position_id in self.active_positions:
            position = self.active_positions.pop(position_id)
            
            # Update portfolio value
            self.portfolio_value += realized_pnl
            
            # Update drawdown metrics
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
            logger.info(f"Position closed: {position_id} - PnL: ${realized_pnl:.2f}, Portfolio: ${self.portfolio_value:.2f}")
        else:
            logger.warning(f"Attempted to close non-existent position: {position_id}")
    
    def update_position_price(self, position_id: str, current_price: float):
        """
        Update current price for a position.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
        """
        if position_id in self.active_positions:
            self.active_positions[position_id]['current_price'] = current_price
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        total_unrealized_pnl = sum(
            pos.get('unrealized_pnl', 0) 
            for pos in self.active_positions.values()
        )
        
        total_risk = sum(
            pos.get('risk_amount', 0) 
            for pos in self.active_positions.values()
        )
        
        return {
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_portfolio_value,
            'current_drawdown': self.current_drawdown,
            'active_positions': len(self.active_positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_risk': total_risk,
            'portfolio_heat': total_risk / self.portfolio_value if self.portfolio_value > 0 else 0,
            'risk_limits': self.risk_limits
        }
